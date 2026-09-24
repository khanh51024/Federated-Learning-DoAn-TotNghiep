"""
Group-aware non-IID partitioning for Federated Learning (GĐ2).

Scenarios
---------
- `iid`                   : uniform split; reference point for the α sweep.
- `label_skew`            : per-class proportions p_k ~ Dirichlet(alpha * 1_C).
                            This is the canonical NIID-Bench label skew
                            (Li et al., 2021, arXiv:2102.02079) and the one
                            named in the thesis outline ("lệch nhãn").
- `quantity_skew`         : client volumes q ~ Dirichlet(quantity_alpha).
                            label distribution inside each client stays
                            ~global ("lệch số lượng").
- `label_quantity_skew`   : both at once ("lệch nhãn + lệch số lượng").

Feature skew ("lệch đặc trưng") is the *third* axis required by GĐ2 but it is
not a partition of the sample set -- it is a per-client domain transform
applied at load time (see `transforms.py`). Keeping it out of the partition
means the same manifest can be replayed with feature skew on/off, which is
what an ablation needs.

Design guarantees
-----------------
1. **Group atomicity** -- the unit of assignment is a whole leaf group, so
   near-duplicate images of one physical leaf never straddle two clients or
   the train/test boundary.
2. **Exact conservation** -- sum of client sizes == size of the input pool.
3. **No duplication / no loss** -- every item lands in exactly one client.
4. **Determinism** -- `numpy.random.default_rng(seed)`; identical inputs and
   seed produce identical partitions on any machine.
5. **Minimum-size repair** -- instead of throwing the whole draw away and
   retrying (the previous behaviour, which silently changes the partition
   seed), clients below `min_samples_per_client` are topped up by moving
   whole groups from the largest clients. Retries remain as a last resort.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# A "group" is {"group_id": str, "items": [sample, ...]}
Group = Dict[str, Any]
GroupsByClass = Dict[int, List[Group]]


# --------------------------------------------------------------------------
# integer allocation
# --------------------------------------------------------------------------
def _allocate_counts(total_count: int, proportions: Sequence[float]) -> np.ndarray:
    """
    Largest-remainder (Hamilton) allocation of `total_count` integer slots over
    `proportions`. Guarantees `counts.sum() == total_count` exactly.
    """
    p = np.asarray(proportions, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        # Degenerate proportions -> spread evenly rather than crash.
        p = np.ones_like(p)
        s = p.sum()
    p = p / s

    exact = total_count * p
    counts = np.floor(exact).astype(np.int64)
    remainder = int(total_count - counts.sum())
    if remainder > 0:
        frac = exact - counts
        # Stable sort: ties broken by client index -> deterministic.
        order = np.lexsort((np.arange(len(frac)), -frac))
        counts[order[:remainder]] += 1
    return counts


def _dirichlet(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    return rng.dirichlet(np.full(n, alpha, dtype=np.float64))


# --------------------------------------------------------------------------
# group assignment
# --------------------------------------------------------------------------
def _assign_groups(
    groups: List[Group],
    targets: np.ndarray,
    client_order: np.ndarray,
) -> Dict[int, List[Any]]:
    """
    Distribute whole groups over clients so each client ends up as close as
    possible to its `targets` count.

    Greedy largest-deficit placement: for each group (in the given shuffled
    order) pick the eligible client with the largest remaining deficit
    (ties -> lowest client id). Exact conservation is automatic because every
    group is placed exactly once.
    """
    deficit = targets.astype(np.float64).copy()
    placed: Dict[int, List[Any]] = {int(c): [] for c in client_order}
    eligible = set(int(c) for c in client_order)

    for g in groups:
        if not eligible:
            # Every client already met its target; round-robin over the rest.
            eligible = set(int(c) for c in client_order)
        cand = np.array(sorted(eligible), dtype=np.int64)
        pick = int(cand[np.argmax(deficit[cand])])
        placed[pick].extend(g["items"])
        deficit[pick] -= len(g["items"])
        if deficit[pick] <= 0:
            eligible.discard(pick)

    return placed


def _repair_min_size(
    client_samples: Dict[int, List[Any]],
    min_samples_per_client: int,
) -> Tuple[Dict[int, List[Any]], int]:
    """
    Top up small clients by moving complete leaf groups from large clients.
    Returns the repaired mapping and the number of moved groups.
    """
    clients = sorted(client_samples)
    moved_groups = 0
    for _ in range(len(clients) * 4 + 8):
        sizes = {c: len(client_samples[c]) for c in clients}
        poor = [c for c in clients if sizes[c] < min_samples_per_client]
        if not poor:
            return client_samples, moved_groups
        poor.sort(key=lambda c: (sizes[c], c))
        needy = poor[0]
        gap = min_samples_per_client - sizes[needy]
        donors = sorted((c for c in clients if c != needy), key=lambda c: (-sizes[c], c))
        moved = False
        for d in donors:
            grouped: Dict[str, List[Any]] = {}
            for item in client_samples[d]:
                gid = str(item.get("split_id", item.get("group_id", item.get("relative_path"))))
                grouped.setdefault(gid, []).append(item)
            candidates = [
                rows for _, rows in sorted(grouped.items())
                if sizes[d] - len(rows) >= min_samples_per_client
            ]
            if not candidates:
                continue
            # Prefer the smallest group that fills the gap; otherwise use the
            # largest eligible group. This limits overshoot and preserves groups.
            filling = [rows for rows in candidates if len(rows) >= gap]
            rows = min(filling, key=lambda x: len(x)) if filling else max(candidates, key=lambda x: len(x))
            ids = {id(item) for item in rows}
            client_samples[d] = [item for item in client_samples[d] if id(item) not in ids]
            client_samples[needy].extend(rows)
            moved_groups += 1
            moved = True
            break
        if not moved:
            break  # nothing left to give; report the shortfall to the caller
    return client_samples, moved_groups


def _sizes_ok(client_samples: Dict[int, List[Any]], min_samples_per_client: int) -> bool:
    return all(len(v) >= min_samples_per_client for v in client_samples.values())


def _finalize(
    client_samples: Dict[int, List[Any]],
    seed: int,
) -> Dict[int, List[Any]]:
    """Deterministic in-client shuffle (so epoch 0 order is not class-sorted)."""
    for c in sorted(client_samples):
        rng = np.random.default_rng(seed * 1_000_003 + c)
        items = client_samples[c]
        order = rng.permutation(len(items))
        client_samples[c] = [items[i] for i in order]
    return client_samples


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
def partition_iid(
    groups_by_class: GroupsByClass,
    num_clients: int,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 20,
    return_diagnostics: bool = False,
) -> Dict[int, List[Any]]:
    """Uniform reference split: every client gets ~1/num_clients of each class."""
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    result = _run_with_retries(
        groups_by_class,
        num_clients,
        seed,
        min_samples_per_client,
        max_retries,
        proportion_fn=lambda rng, class_id, n_groups: np.full(num_clients, 1.0 / num_clients),
        scenario="iid",
    )
    return result if return_diagnostics else result[0]


def partition_label_skew(
    groups_by_class: GroupsByClass,
    num_clients: int,
    alpha: float = 0.1,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 20,
    return_diagnostics: bool = False,
) -> Dict[int, List[Any]]:
    """
    Dirichlet label skew. For each class k an independent
    p_k ~ Dirichlet(alpha * 1_C) decides how that class is shared out.

    alpha -> inf  : p_k ~ uniform, i.e. IID.
    alpha -> 0    : p_k concentrates on few clients, i.e. strong non-IID.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    result = _run_with_retries(
        groups_by_class,
        num_clients,
        seed,
        min_samples_per_client,
        max_retries,
        proportion_fn=lambda rng, class_id, n_groups: _dirichlet(rng, num_clients, alpha),
        scenario="label_skew",
        alpha=alpha,
    )
    return result if return_diagnostics else result[0]


def partition_quantity_skew(
    groups_by_class: GroupsByClass,
    num_clients: int,
    quantity_alpha: float = 0.1,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 20,
    return_diagnostics: bool = False,
    **_: Any,
) -> Dict[int, List[Any]]:
    """
    Quantity skew: q ~ Dirichlet(quantity_alpha). One size vector is
    drawn per attempt and reused for every class, so client *volumes* differ
    while each client's internal label mix stays close to the global one.

    This follows NIID-Bench section IV-D. `quantity_alpha` is deliberately
    separate from label-skew `alpha` so the two axes remain independently tunable.
    """
    if quantity_alpha <= 0:
        raise ValueError(f"quantity_alpha must be positive, got {quantity_alpha}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")

    def proportion_fn(rng, class_id, n_groups):
        if class_id == _FIRST_CLASS_SENTINEL:
            proportion_fn.q = _dirichlet(rng, num_clients, quantity_alpha)
        return proportion_fn.q

    result = _run_with_retries(
        groups_by_class,
        num_clients,
        seed,
        min_samples_per_client,
        max_retries,
        proportion_fn=proportion_fn,
        scenario="quantity_skew",
        quantity_alpha=quantity_alpha,
    )
    return result if return_diagnostics else result[0]


# Sentinel used to draw the size vector exactly once per attempt.
_FIRST_CLASS_SENTINEL = object()


def partition_label_quantity_skew(
    groups_by_class: GroupsByClass,
    num_clients: int,
    alpha: float = 0.1,
    quantity_alpha: float = 0.1,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 20,
    return_diagnostics: bool = False,
) -> Dict[int, List[Any]]:
    """Joint skew: label Dirichlet(alpha) modulated by quantity Dirichlet(beta)."""
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if quantity_alpha <= 0:
        raise ValueError(f"quantity_alpha must be positive, got {quantity_alpha}")

    def proportion_fn(rng, class_id, n_groups):
        if class_id == _FIRST_CLASS_SENTINEL:
            proportion_fn.q = _dirichlet(rng, num_clients, quantity_alpha)
        return _dirichlet(rng, num_clients, alpha) * proportion_fn.q

    result = _run_with_retries(
        groups_by_class,
        num_clients,
        seed,
        min_samples_per_client,
        max_retries,
        proportion_fn=proportion_fn,
        scenario="label_quantity_skew",
        alpha=alpha,
        quantity_alpha=quantity_alpha,
    )
    return result if return_diagnostics else result[0]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _run_with_retries(
    groups_by_class: GroupsByClass,
    num_clients: int,
    seed: int,
    min_samples_per_client: int,
    max_retries: int,
    proportion_fn,
    scenario: str,
    **meta: Any,
) -> tuple[Dict[int, List[Any]], Dict[str, Any]]:
    """
    Shared driver. Attempt 0 uses `seed`; later attempts use a deterministic
    derived seed. Repair moves complete groups. If no attempt reaches the
    configured minimum, fail before any manifests can be written.
    """
    total_items = sum(len(g["items"]) for groups in groups_by_class.values() for g in groups)
    if total_items < num_clients * min_samples_per_client:
        raise RuntimeError(
            f"Cannot satisfy min_samples_per_client={min_samples_per_client}: "
            f"{total_items} samples < {num_clients} clients x minimum. No partition was written."
        )
    client_ids = np.arange(num_clients, dtype=np.int64)
    for attempt in range(max(1, max_retries)):
        attempt_seed = seed if attempt == 0 else seed + attempt * 10007
        rng = np.random.default_rng(attempt_seed)

        client_samples: Dict[int, List[Any]] = {c: [] for c in range(num_clients)}
        first = True

        for class_id in sorted(groups_by_class):
            groups = groups_by_class[class_id]
            if not groups:
                continue

            n_class = sum(len(g["items"]) for g in groups)
            if n_class == 0:
                continue

            # Deterministic group shuffle, then proportional targets.
            order = rng.permutation(len(groups))
            shuffled = [groups[i] for i in order]

            key = _FIRST_CLASS_SENTINEL if first else class_id
            first = False
            props = proportion_fn(rng, key, len(groups))
            targets = _allocate_counts(n_class, props)

            placed = _assign_groups(shuffled, targets, client_ids)
            for c, items in placed.items():
                client_samples[c].extend(items)

        client_samples, repaired_groups = _repair_min_size(client_samples, min_samples_per_client)

        cur_min = min(len(v) for v in client_samples.values())
        if _sizes_ok(client_samples, min_samples_per_client):
            return _finalize(client_samples, attempt_seed), {
                "effective_seed": attempt_seed,
                "attempts_used": attempt + 1,
                "repaired_groups": repaired_groups,
            }
    raise RuntimeError(
        f"Cannot satisfy min_samples_per_client={min_samples_per_client} for "
        f"scenario={scenario}, num_clients={num_clients}, seed={seed} after "
        f"{max(1, max_retries)} attempts. No partition was written. meta={meta}"
    )


SCENARIOS = {
    "iid": partition_iid,
    "label_skew": partition_label_skew,
    "quantity_skew": partition_quantity_skew,
    "label_quantity_skew": partition_label_quantity_skew,
}


def partition(
    groups_by_class: GroupsByClass,
    scenario: str,
    num_clients: int,
    alpha: float = 0.1,
    quantity_alpha: float = 0.1,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 20,
) -> tuple[Dict[int, List[Any]], Dict[str, Any]]:
    """
    Dispatch to the requested scenario.

    Returns `(client_samples, diagnostics)` where diagnostics reports the
    effective parameters plus whether the minimum-size threshold was actually
    met (`min_size_satisfied`). A `False` there means the partition is usable
    but one or more clients are smaller than requested -- always worth stating
    in the report rather than hiding.
    """
    key = scenario.lower()
    if key not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Supported: {sorted(SCENARIOS)}"
        )
    fn = SCENARIOS[key]
    if key == "iid":
        client_samples, run_diag = fn(
            groups_by_class, num_clients, seed, min_samples_per_client, max_retries, True
        )
        effective = {"scenario": key}
    elif key == "quantity_skew":
        client_samples, run_diag = fn(
            groups_by_class, num_clients, quantity_alpha, seed,
            min_samples_per_client, max_retries, True,
        )
        effective = {"scenario": key, "quantity_alpha": quantity_alpha}
    elif key == "label_skew":
        client_samples, run_diag = fn(
            groups_by_class, num_clients, alpha, seed,
            min_samples_per_client, max_retries, True,
        )
        effective = {"scenario": key, "alpha": alpha}
    else:
        client_samples, run_diag = fn(
            groups_by_class, num_clients, alpha, quantity_alpha, seed,
            min_samples_per_client, max_retries, True,
        )
        effective = {"scenario": key, "alpha": alpha, "quantity_alpha": quantity_alpha}

    sizes = [len(client_samples[c]) for c in range(num_clients)]
    total_in = sum(len(g["items"]) for groups in groups_by_class.values() for g in groups)
    diagnostics = {
        **effective,
        **run_diag,
        "num_clients": num_clients,
        "seed": seed,
        "min_samples_per_client": min_samples_per_client,
        "client_size_min": int(min(sizes)) if sizes else 0,
        "min_size_satisfied": bool(min(sizes) >= min_samples_per_client) if sizes else False,
        "conserved": int(sum(sizes)) == int(total_in),
        "input_items": int(total_in),
        "output_items": int(sum(sizes)),
    }
    return client_samples, diagnostics

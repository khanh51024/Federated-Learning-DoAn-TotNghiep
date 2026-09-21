"""
Quantitative non-IID and integrity metrics for a partition (GĐ2).

Label-distribution heterogeneity
--------------------------------
For every client c with local label distribution p_c and the pooled global
distribution p_g:

  * shannon_entropy      H(p_c) in bits; max = log2(K)
  * entropy_norm         H(p_c) / log2(K)              in [0, 1]
  * tvd_from_global      0.5 * sum_k |p_c,k - p_g,k|   in [0, 1]
  * kl_from_global       KL(p_c || p_g), smoothed       >= 0
  * js_from_global       Jensen-Shannon divergence      in [0, 1] (log2)
  * cosine_to_global     cos(p_c, p_g)                  in [0, 1]
  * classes_present      #classes with >= 1 sample
  * classes_missing      K - classes_present
  * rare_class_recall    share of globally-rare classes the client still holds

Volume heterogeneity (a separate GĐ2 axis: "lệch số lượng")
-----------------------------------------------------------
  * client_size_min / max / mean / std / cv
  * client_size_gini    Gini coefficient of client volumes in [0, 1]

Fairness ("tính công bằng giữa các cơ sở")
------------------------------------------
  * weighted_entropy    global entropy of the pooled client-mix (proxy for how
                        much a sample-count-weighted server sees each class)
  * fairness_spread     max - min per-client entropy (clients left behind)

Group integrity (leaf-level leakage audit)
-----------------------------------------
  * groups_split_across_clients  must be 0
  * groups_in_both_train_and_test must be 0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-12


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector; 0 = perfectly equal."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _cosine(p: np.ndarray, q: np.ndarray) -> float:
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    return float(np.dot(p, q) / denom) if denom > 0 else 0.0


def _count_matrix(
    client_samples: Mapping[int, Sequence[Any]],
    num_classes: int,
) -> np.ndarray:
    """Vectorised client x class count matrix."""
    client_ids = sorted(client_samples)
    mat = np.zeros((len(client_ids), num_classes), dtype=np.int64)
    for i, c in enumerate(client_ids):
        items = client_samples[c]
        if len(items) == 0:
            continue
        labels = np.fromiter((int(s["label"]) for s in items), dtype=np.int64, count=len(items))
        mat[i] = np.bincount(labels, minlength=num_classes)[:num_classes]
    return mat


# --------------------------------------------------------------------------
# client x class matrix artifact
# --------------------------------------------------------------------------
def compute_client_class_matrix(
    client_samples: Mapping[int, Sequence[Any]],
    num_classes: int = 38,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Counts of samples per (client, class), as a DataFrame for CSV/plots."""
    mat = _count_matrix(client_samples, num_classes)
    columns = list(class_names) if class_names else [f"Class_{i}" for i in range(num_classes)]
    index = [f"Client_{c:02d}" for c in sorted(client_samples)]
    return pd.DataFrame(mat, index=index, columns=columns)


# --------------------------------------------------------------------------
# main metric bundle
# --------------------------------------------------------------------------
def calculate_partition_metrics(
    client_samples: Mapping[int, Sequence[Any]],
    total_classes: int = 38,
    class_names: Optional[List[str]] = None,
    rare_class_threshold: int = 500,
) -> Dict[str, Any]:
    """
    Compute the full metric bundle. Returns a dict with a `client_details`
    list (one row per client, ALWAYS present even for empty clients) and flat
    summary statistics.
    """
    client_ids = sorted(client_samples)
    num_clients = len(client_ids)
    mat = _count_matrix(client_samples, total_classes)          # [C, K]
    sizes = mat.sum(axis=1)                                     # [C]
    total_samples = int(sizes.sum())

    global_counts = mat.sum(axis=0)                             # [K]
    global_probs = (global_counts / total_samples) if total_samples else np.zeros(total_classes)
    max_entropy = math.log2(total_classes) if total_classes > 1 else 1.0

    # Globally rare classes: few samples in the whole training pool. These are
    # the ones a strong non-IID split is most likely to starve.
    rare_mask = (global_counts > 0) & (global_counts < rare_class_threshold)
    rare_idx = np.flatnonzero(rare_mask)

    entropies = np.zeros(num_clients)
    tvds = np.zeros(num_clients)
    kls = np.zeros(num_clients)
    jss = np.zeros(num_clients)
    cosines = np.zeros(num_clients)
    present = np.zeros(num_clients, dtype=np.int64)
    rare_recall = np.zeros(num_clients)

    client_details: List[Dict[str, Any]] = []

    for i, c in enumerate(client_ids):
        size = int(sizes[i])
        counts = mat[i]

        if size == 0:
            # Empty client: maximally diverged, zero coverage. Recorded (the
            # previous implementation skipped the row entirely, which silently
            # misaligned statistics.csv against client ids).
            entropies[i] = 0.0
            tvds[i] = 1.0
            kls[i] = float("inf")
            jss[i] = 1.0
            cosines[i] = 0.0
            present[i] = 0
            rare_recall[i] = 0.0
        else:
            probs = counts / size
            entropies[i] = _entropy_bits(probs)
            tvds[i] = 0.5 * float(np.sum(np.abs(probs - global_probs)))
            kls[i] = _kl(probs, global_probs)
            jss[i] = _js(probs, global_probs)
            cosines[i] = _cosine(counts.astype(np.float64), global_counts.astype(np.float64))
            present[i] = int(np.count_nonzero(counts))
            if rare_idx.size:
                rare_recall[i] = float(np.count_nonzero(counts[rare_idx]) / rare_idx.size)

        finite_kl = kls[i] if np.isfinite(kls[i]) else None
        client_details.append({
            "client_id": int(c),
            "client_name": f"client_{c:02d}",
            "sample_count": size,
            "sample_ratio": round(size / total_samples, 6) if total_samples else 0.0,
            "num_classes": int(present[i]),
            "num_classes_missing": int(total_classes - present[i]),
            "shannon_entropy": round(float(entropies[i]), 4),
            "entropy_normalized": round(float(entropies[i] / max_entropy), 4),
            "tvd_from_global": round(float(tvds[i]), 4),
            "kl_from_global": round(float(finite_kl), 4) if finite_kl is not None else "",
            "js_from_global": round(float(jss[i]), 4),
            "cosine_to_global": round(float(cosines[i]), 4),
            "rare_class_recall": round(float(rare_recall[i]), 4),
        })

    # ---- summary ----
    mean_size = float(sizes.mean()) if num_clients else 0.0
    std_size = float(sizes.std()) if num_clients else 0.0
    finite_kls = kls[np.isfinite(kls)]

    summary: Dict[str, Any] = {
        "total_samples": total_samples,
        "total_classes": total_classes,
        "num_clients": num_clients,
        # volume heterogeneity
        "client_size_min": int(sizes.min()) if num_clients else 0,
        "client_size_max": int(sizes.max()) if num_clients else 0,
        "client_size_mean": round(mean_size, 2),
        "client_size_std": round(std_size, 2),
        "client_size_cv": round(std_size / mean_size, 4) if mean_size > 0 else 0.0,
        "client_size_gini": round(_gini(sizes), 4),
        # label heterogeneity
        "classes_per_client_mean": round(float(present.mean()), 2) if num_clients else 0.0,
        "classes_per_client_min": int(present.min()) if num_clients else 0,
        "classes_per_client_max": int(present.max()) if num_clients else 0,
        "clients_with_all_classes": int(np.sum(present == total_classes)),
        "shannon_entropy_mean": round(float(entropies.mean()), 4) if num_clients else 0.0,
        "shannon_entropy_min": round(float(entropies.min()), 4) if num_clients else 0.0,
        "shannon_entropy_max": round(float(entropies.max()), 4) if num_clients else 0.0,
        "entropy_norm_mean": round(float(entropies.mean() / max_entropy), 4) if num_clients else 0.0,
        "tvd_mean": round(float(tvds.mean()), 4) if num_clients else 0.0,
        "tvd_min": round(float(tvds.min()), 4) if num_clients else 0.0,
        "tvd_max": round(float(tvds.max()), 4) if num_clients else 0.0,
        "kl_mean": round(float(finite_kls.mean()), 4) if finite_kls.size else None,
        "js_mean": round(float(jss.mean()), 4) if num_clients else 0.0,
        "cosine_to_global_mean": round(float(cosines.mean()), 4) if num_clients else 0.0,
        # rare-class / fairness
        "rare_class_threshold": int(rare_class_threshold),
        "num_rare_classes": int(rare_idx.size),
        "rare_class_recall_mean": round(float(rare_recall.mean()), 4) if num_clients else 0.0,
        "fairness_entropy_spread": round(float(entropies.max() - entropies.min()), 4) if num_clients else 0.0,
        "empty_clients": int(np.sum(sizes == 0)),
        "max_entropy_bits": round(max_entropy, 4),
        "client_details": client_details,
    }
    return summary


# --------------------------------------------------------------------------
# group (leaf) integrity audit
# --------------------------------------------------------------------------
def audit_group_integrity(
    client_samples: Mapping[int, Sequence[Any]],
    test_samples: Sequence[Dict[str, Any]] = (),
    val_samples: Sequence[Dict[str, Any]] = (),
    client_val: Optional[Mapping[int, Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """
    Verify that no leaf group straddles two clients or the train/eval boundary.
    This is the check that proves the split is not leaking near-duplicates.

    `client_val` (per-client local validation sets) is audited too: a leaf used
    for a client's early stopping must not also be in that client's training
    data, or the local val score is optimistic.
    """
    def _groups(items: Sequence[Dict[str, Any]]) -> set:
        return set(s["group_id"] for s in items if "group_id" in s)

    client_val = client_val or {}
    # Train-side = local train + local val: both belong to the same facility,
    # so neither may share a leaf with a *different* facility.
    client_groups: Dict[int, set] = {
        c: _groups(client_samples.get(c, [])) | _groups(client_val.get(c, []))
        for c in sorted(set(client_samples) | set(client_val))
    }
    local_train_groups: Dict[int, set] = {c: _groups(v) for c, v in client_samples.items()}
    local_val_groups: Dict[int, set] = {c: _groups(v) for c, v in client_val.items()}

    test_groups = _groups(test_samples)
    val_groups = _groups(val_samples)

    shared_between_clients = 0
    seen: Dict[str, int] = {}
    for c in sorted(client_groups):
        for g in client_groups[c]:
            if g in seen:
                shared_between_clients += 1
            else:
                seen[g] = c

    # A leaf must not be in the same client's train AND val.
    train_val_within_client = sum(
        len(local_train_groups.get(c, set()) & local_val_groups.get(c, set()))
        for c in set(local_train_groups) | set(local_val_groups)
    )

    train_groups = set(seen)
    train_test_shared = train_groups & test_groups
    train_val_shared = train_groups & val_groups
    test_val_shared = test_groups & val_groups

    n_train_images = sum(len(v) for v in client_samples.values())
    n_groups = len(train_groups)

    return {
        "train_images": n_train_images,
        "train_leaf_groups": n_groups,
        "images_per_group_mean": round(n_train_images / n_groups, 3) if n_groups else 0.0,
        "groups_split_across_clients": shared_between_clients,
        "groups_in_both_train_and_test": len(train_test_shared),
        "groups_in_both_train_and_val": len(train_val_shared),
        "groups_in_both_test_and_val": len(test_val_shared),
        "groups_in_both_client_train_and_client_val": train_val_within_client,
        "test_leaf_groups": len(test_groups),
        "val_leaf_groups": len(val_groups),
        "client_val_leaf_groups": sum(len(v) for v in local_val_groups.values()),
        "leakage_free": (
            shared_between_clients == 0
            and len(train_test_shared) == 0
            and len(train_val_shared) == 0
            and len(test_val_shared) == 0
            and train_val_within_client == 0
        ),
    }


def audit_sample_integrity(
    client_samples: Mapping[int, Sequence[Any]],
    train_pool: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]] = (),
    val_samples: Sequence[Dict[str, Any]] = (),
    client_val: Optional[Mapping[int, Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """
    Exact conservation / no-duplication / no-overlap audit at image level.

    Per-client validation rows count as *assigned* -- they are carved out of the
    training pool, so omitting them would report a false "missing samples".
    """
    counts: Dict[str, int] = {}
    for source in (client_samples, client_val or {}):
        for c, items in source.items():
            for s in items:
                counts[s["relative_path"]] = counts.get(s["relative_path"], 0) + 1

    duplicated = [p for p, n in counts.items() if n > 1]
    pool_paths = set(s["relative_path"] for s in train_pool)
    assigned = set(counts)

    test_paths = set(s["relative_path"] for s in test_samples)
    val_paths = set(s["relative_path"] for s in val_samples)

    return {
        "pool_images": len(pool_paths),
        "assigned_images": len(assigned),
        "assigned_rows": int(sum(counts.values())),
        "missing_from_clients": len(pool_paths - assigned),
        "extra_not_in_pool": len(assigned - pool_paths),
        "duplicated_images": len(duplicated),
        "train_test_overlap": len(assigned & test_paths),
        "train_val_overlap": len(assigned & val_paths),
        "test_val_overlap": len(test_paths & val_paths),
        "conserved": (
            not duplicated
            and assigned == pool_paths
            and not (assigned & test_paths)
            and not (assigned & val_paths)
            and not (test_paths & val_paths)
        ),
    }

"""Mô phỏng học phân tán cộng đồng bằng đúng chính sách của máy chủ (numpy + `src.federated.community`).

Phần học cục bộ là bản port numpy của `trainSoftmax` / `splitHoldout` / `makeRng` trong `web/static/js/core/community-math.js`
(cùng bộ sinh số mulberry32, cùng thứ tự xáo trộn), nên một người tham gia mô phỏng học giống hệt người thật trong trình duyệt.
Phần máy chủ (kiểm cập nhật, cắt chuẩn, gộp FedAvg có trần, cổng chất lượng) gọi thẳng hàm của `src.federated.community`.
Không có mạng, không có ảnh: chỉ nhận ma trận đặc trưng 1024 chiều đã trích sẵn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.federated import community as core
from src.federated.community import CommunityError, EvalSet, Head, Policy

M32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Port của community-math.js
# ---------------------------------------------------------------------------
def make_rng(seed: int) -> Callable[[], float]:
    """mulberry32, đúng từng bit với `makeRng` của JS."""
    state = [seed & M32]

    def imul(a: int, b: int) -> int:
        return (a * b) & M32

    def next_value() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & M32
        t = state[0]
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    return next_value


def shuffle_in_place(items: list, rng: Callable[[], float]) -> list:
    for i in range(len(items) - 1, 0, -1):
        j = int(rng() * (i + 1))
        items[i], items[j] = items[j], items[i]
    return items


def split_holdout(labels: Sequence[int], fraction: float, seed: int = 7, min_total: int = 15) -> Tuple[List[int], List[int]]:
    n = len(labels)
    everything = list(range(n))
    if n < min_total or not fraction > 0:
        return everything, []
    rng = make_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for i in everything:
        by_class.setdefault(int(labels[i]), []).append(i)
    holdout: List[int] = []
    for members in by_class.values():
        if len(members) < 2:
            continue
        shuffle_in_place(members, rng)
        holdout.extend(members[: max(1, int(np.floor(len(members) * fraction + 0.5)))])  # Math.round
    held = set(holdout)
    train = [i for i in everything if i not in held]
    if not train or not holdout:
        return everything, []
    return train, holdout


def train_softmax(features: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, epochs: int, lr: float,
                  batch: int, seed: int = 1) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Mini-batch SGD softmax regression; không sửa đầu vào, trả (weight, bias, loss mỗi epoch)."""
    w = np.array(weight, dtype=np.float32)
    b = np.array(bias, dtype=np.float32)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    rng = make_rng(seed)
    order = list(range(n))
    history: List[float] = []
    classes = w.shape[0]
    for _epoch in range(epochs):
        shuffle_in_place(order, rng)
        loss_sum = 0.0
        for start in range(0, n, batch):
            index = np.array(order[start:start + batch], dtype=np.int64)
            xb = x[index].astype(np.float64)
            logits = xb @ w.astype(np.float64).T + b.astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            expd = np.exp(logits)
            prob = expd / expd.sum(axis=1, keepdims=True)
            target = y[index]
            loss_sum += float(-np.log(np.maximum(prob[np.arange(len(index)), target], 1e-12)).sum())
            prob[np.arange(len(index)), target] -= 1.0
            prob /= len(index)
            grad_w = prob.T @ xb
            grad_b = prob.sum(axis=0)
            w = (w.astype(np.float64) - lr * grad_w).astype(np.float32)
            b = (b.astype(np.float64) - lr * grad_b).astype(np.float32)
        history.append(loss_sum / max(n, 1))
    del classes
    return w, b, history


# ---------------------------------------------------------------------------
# Mô phỏng
# ---------------------------------------------------------------------------
@dataclass
class Pool:
    """Kho ảnh (đã là đặc trưng) mà người tham gia mô phỏng được lấy mẫu."""
    features: np.ndarray
    labels: np.ndarray
    sources: np.ndarray  # 0 = PlantVillage, 1 = PlantDoc

    def __post_init__(self) -> None:
        self.features = np.ascontiguousarray(self.features, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.sources = np.asarray(self.sources, dtype=np.int8)
        self.classes = sorted(set(self.labels.tolist()))
        self.by_class = {c: np.flatnonzero(self.labels == c) for c in self.classes}
        self.plantdoc = np.flatnonzero(self.sources == core.SOURCE_PLANTDOC)


SCENARIOS = ("iid", "three_groups", "one_group", "plantdoc_only")


def draw_user_indices(pool: Pool, scenario: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Chỉ số ảnh của một người: iid = ảnh bất kỳ; N nhóm = chỉ ảnh của N nhóm bệnh ngẫu nhiên; plantdoc_only = ảnh chụp ngoài đồng."""
    if scenario == "iid":
        universe = np.arange(len(pool.labels))
    elif scenario in ("three_groups", "one_group"):
        k = 3 if scenario == "three_groups" else 1
        chosen = rng.choice(pool.classes, size=k, replace=False)
        universe = np.concatenate([pool.by_class[int(c)] for c in chosen])
    elif scenario == "plantdoc_only":
        universe = pool.plantdoc
    else:
        raise ValueError(scenario)
    return rng.choice(universe, size=min(n, len(universe)), replace=False)


@dataclass
class LocalConfig:
    epochs: int
    lr: float
    batch: int
    holdout_fraction: float
    n_min_images: int = 30
    n_max_images: int = 100


def local_update(pool: Pool, indices: np.ndarray, parent: Head, config: LocalConfig, seed: int) -> Tuple[Head, int]:
    """Một người tham gia: tách phần giữ lại, học lớp cuối trên phần còn lại. Trả (head, số ảnh học)."""
    labels = pool.labels[indices]
    train, _holdout = split_holdout(labels.tolist(), config.holdout_fraction, seed=seed + 7)
    train_index = np.array(train, dtype=np.int64)
    weight, bias, _ = train_softmax(pool.features[indices][train_index], labels[train_index], parent[core.WEIGHT_KEY],
                                    parent[core.BIAS_KEY], epochs=config.epochs, lr=config.lr, batch=config.batch, seed=seed)
    return core.as_head(weight, bias), len(train_index)


def run_scenario(
    pool: Pool, gate_set: EvalSet, test_set: EvalSet, base: Head, policy: Policy, config: LocalConfig, *,
    scenario: str, k: int, rounds: int, seed: int, gate_enabled: bool = True,
    progress: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Một lần chạy: `rounds` vòng, mỗi vòng K người. Giữ nguyên quy tắc máy chủ (cắt chuẩn, trần n_cap, cổng giám sát).

    Cổng chạy trên `gate_set` (tập giám sát như production); số báo cáo lấy từ `test_set` (tập test giữ riêng, chưa dùng chọn mô hình)."""
    rng = np.random.default_rng(seed)
    head = base
    gate_history = [core.evaluate_head(base, gate_set)]
    curve = [core.evaluate_head(base, test_set)]
    rounds_log: List[Dict[str, Any]] = []
    for round_index in range(rounds):
        parent = head
        parent_sha = core.head_sha(parent)
        updates, users = [], []
        for user in range(k):
            n = int(rng.integers(config.n_min_images, config.n_max_images + 1))
            indices = draw_user_indices(pool, scenario, n, rng)
            child, n_train = local_update(pool, indices, parent, config, seed=int(rng.integers(1, 2 ** 31)))
            try:
                check = core.check_update(parent, child, parent_sha, policy)
            except CommunityError:
                continue
            updates.append({"head": check.head, "num_examples": n_train})
            users.append({"n": int(len(indices)), "n_train": n_train, "norm": check.norm_raw, "clipped": bool(check.clipped)})
        entry: Dict[str, Any] = {"round": round_index + 1, "users": users, "accepted": False, "reasons": []}
        if len(updates) >= max(policy.k_min, 1):
            candidate = core.aggregate_heads(updates, parent, policy)
            cand_gate = core.evaluate_head(candidate, gate_set)
            par_gate = core.evaluate_head(parent, gate_set)
            accepted, reasons = core.gate_decision(cand_gate, par_gate, policy, core.best_of(gate_history))
            if not gate_enabled:
                accepted, reasons = True, []
            entry["accepted"], entry["reasons"] = accepted, reasons
            entry["delta_norm"] = float(np.sqrt(sum(float(((candidate[key] - parent[key]).astype(np.float64) ** 2).sum()) for key in parent)))
            if accepted:
                head = candidate
                gate_history.append(cand_gate)
        curve.append(core.evaluate_head(head, test_set))
        entry["test"] = curve[-1]
        rounds_log.append(entry)
        if progress:
            progress(round_index + 1, entry)
    return {"scenario": scenario, "k": k, "seed": seed, "rounds": rounds_log, "curve": curve, "gate_enabled": gate_enabled}

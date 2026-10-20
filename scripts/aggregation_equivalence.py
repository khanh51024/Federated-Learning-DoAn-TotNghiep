"""Phép gộp FedAvg có gây sai số không, và chênh lệch với huấn luyện tập trung đến từ đâu?

Thí nghiệm kiểm chứng trên lớp cuối (softmax 1024 -> 38) với đặc trưng đã trích sẵn của FedAvg v7 (data/community_sim/features.npz;
4.225 ảnh calibration làm dữ liệu học, 11.015 ảnh test giữ riêng để đo). CPU, vài giây. KHÔNG tạo mô hình mới cho đồ án: chỉ đo
sai lệch của phép gộp và của việc học cục bộ.

A. FedSGD (mỗi client một bước gradient đầy đủ trên shard rồi gộp theo n_k) phải TRÙNG từng bước với gradient descent tập trung
   trên toàn bộ dữ liệu, vì sum_k (n_k/N) (W - lr g_k) = W - lr g_toàn. Đo chênh trọng số sau T bước ở float64 và float32.
   Nếu phép gộp có sai số thì sai số này sẽ lớn hơn làm tròn số học.
B. FedAvg với E bước cục bộ mỗi vòng (tổng số bước giữ cố định) trên shard lệch nhãn Dirichlet alpha: đo khoảng cách tới lời giải
   tập trung cùng số bước, độ chính xác của mô hình gộp, độ chính xác của từng mô hình client TRƯỚC khi gộp, và khoảng cách từ từng
   client tới mô hình gộp (client drift). E = 1 phải trùng tập trung ở mọi alpha; E > 1 lệch, lệch nhiều hơn khi alpha nhỏ.

Chạy:
    python scripts/aggregation_equivalence.py     # ghi artifacts/aggregation_equivalence.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

FEATURES = "data/community_sim/features.npz"  # gitignored (data/*)
CLIENTS = 5
SEED = 20261020
TOTAL_STEPS = 100
LR_GRID = (0.05, 0.1, 0.2, 0.5, 1.0)
ALPHAS = (100.0, 1.0, 0.1)
LOCAL_STEPS = (1, 5, 20)
L2 = 1e-4


# --------------------------------------------------------------------------- phép tính thuần numpy (có test)
def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    """Cross-entropy trung bình + l2/2 ||W||^2 và gradient. `w` = [d + 1, C] (hàng cuối là bias)."""
    d = x.shape[1]
    logits = x @ w[:d] + w[d]
    p = softmax_rows(logits)
    n = len(y)
    loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean() + 0.5 * l2 * np.sum(w[:d] ** 2))
    p[np.arange(n), y] -= 1.0
    p /= n
    grad = np.empty_like(w)
    grad[:d] = x.T @ p + l2 * w[:d]
    grad[d] = p.sum(axis=0)
    return loss, grad


def dirichlet_shards(labels: np.ndarray, clients: int, alpha: float, seed: int, min_size: int = 20) -> list[np.ndarray]:
    """Chia chỉ số ảnh theo nhãn Dirichlet(alpha); lấy lại mẫu cho tới khi mọi client có >= min_size ảnh."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(200):
        shards: list[list[int]] = [[] for _ in range(clients)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            share = rng.dirichlet(np.full(clients, alpha))
            cuts = (np.cumsum(share) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                shards[k].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.array(sorted(s)) for s in shards]
    raise RuntimeError("không chia được shard đủ lớn")


def fedavg_round(w: np.ndarray, x: np.ndarray, y: np.ndarray, shards: list[np.ndarray], local_steps: int, lr: float,
                 l2: float, dtype: Any = np.float64) -> tuple[np.ndarray, list[np.ndarray]]:
    """Một vòng: mỗi client chạy `local_steps` bước GD đầy đủ trên shard từ `w`; gộp theo n_k (tích lũy float64, làm tròn về `dtype`)."""
    clients = []
    for idx in shards:
        wk = w.copy()
        xs, ys = x[idx], y[idx]
        for _ in range(local_steps):
            _, g = loss_and_grad(wk, xs, ys, l2)
            wk = (wk - lr * g).astype(dtype)
        clients.append(wk)
    total = float(sum(len(s) for s in shards))
    acc = np.zeros(w.shape, dtype=np.float64)
    for wk, idx in zip(clients, shards):
        acc += wk.astype(np.float64) * len(idx)
    return (acc / total).astype(dtype), clients


def centralized_steps(w: np.ndarray, x: np.ndarray, y: np.ndarray, steps: int, lr: float, l2: float, dtype: Any = np.float64) -> np.ndarray:
    for _ in range(steps):
        _, g = loss_and_grad(w, x, y, l2)
        w = (w - lr * g).astype(dtype)
    return w


def accuracy(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    return float(((x @ w[:d] + w[d]).argmax(axis=1) == y).mean())


def relative_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) / max(np.linalg.norm(b.astype(np.float64)), 1e-30))


def _reading(equivalence: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Kết luận sinh từ số đo, không viết sẵn: chỉ khẳng định điều mà ngưỡng sau đây thỏa."""
    exact = equivalence["float64"]["final_relative_diff"] < 1e-9 and equivalence["float32"]["final_relative_diff"] < 1e-5
    one_step_exact = all(r["relative_distance_to_centralized"] < 1e-9 for r in rows if r["local_steps"] == 1)
    drift = [r for r in rows if r["local_steps"] > 1]
    drifts = bool(drift) and all(r["relative_distance_to_centralized"] > 1e-3 for r in drift)
    if exact and one_step_exact and drifts:
        return ("Phép gộp không gây sai số (A: chênh ở mức làm tròn; B: E = 1 trùng tập trung ở mọi alpha). Chênh lệch chỉ xuất hiện khi "
                "mỗi client chạy nhiều hơn một bước cục bộ (B), tức là do học cục bộ chứ không phải do truyền hoặc gộp trọng số.")
    return "KHÔNG đạt điều kiện đồng nhất đã đặt (xem fedsgd_equals_centralized và drift_grid): không kết luận."


# --------------------------------------------------------------------------- thí nghiệm
def pick_lr(x: np.ndarray, y: np.ndarray, w0: np.ndarray) -> tuple[float, dict[str, float]]:
    """Chọn lr theo loss học (tập calibration) sau TOTAL_STEPS bước tập trung; không đụng tập test."""
    losses = {}
    for lr in LR_GRID:
        w = centralized_steps(w0, x, y, TOTAL_STEPS, lr, L2)
        loss, _ = loss_and_grad(w, x, y, L2)
        losses[str(lr)] = loss if np.isfinite(loss) else float("inf")
    return float(min(losses, key=lambda k: losses[k])), losses


def main() -> int:
    path = reg.abs_path(FEATURES)
    if not path.is_file():
        print(f"[equiv] thiếu {FEATURES}; chạy scripts/build_community_sim_features.py", file=sys.stderr)
        return 1
    z = np.load(path, allow_pickle=False)
    x, y = z["cal_x"].astype(np.float64), z["cal_y"].astype(np.int64)
    tx, ty = z["test_x"].astype(np.float64), z["test_y"].astype(np.int64)
    rng = np.random.default_rng(SEED)
    d, c = x.shape[1], 38
    w0 = np.zeros((d + 1, c))
    w0[:d] = rng.normal(0.0, 0.01, size=(d, c))
    lr_text, lr_losses = pick_lr(x, y, w0)
    lr = float(lr_text)
    started = time.time()
    print(f"[equiv] lr chọn trên loss học: {lr} {lr_losses}", flush=True)

    # ---- A. FedSGD == GD tập trung
    shards = dirichlet_shards(y, CLIENTS, 0.1, SEED)  # lệch nặng: phép đồng nhất phải đúng kể cả lúc này
    equivalence: dict[str, Any] = {"alpha": 0.1, "n_k": [int(len(s)) for s in shards], "lr": lr, "steps": TOTAL_STEPS, "l2": L2}
    for name, dtype in (("float64", np.float64), ("float32", np.float32)):
        wc = w0.astype(dtype)
        wf = w0.astype(dtype)
        worst = 0.0
        for _ in range(TOTAL_STEPS):
            wc = centralized_steps(wc, x, y, 1, lr, L2, dtype)
            wf, _ = fedavg_round(wf, x, y, shards, 1, lr, L2, dtype)
            worst = max(worst, float(np.abs(wc.astype(np.float64) - wf.astype(np.float64)).max()))
        equivalence[name] = {
            "max_abs_weight_diff_over_all_steps": worst,
            "final_relative_diff": relative_distance(wf, wc),
            "float_eps": float(np.finfo(dtype).eps),
            "test_accuracy_centralized": round(100 * accuracy(wc.astype(np.float64), tx, ty), 3),
            "test_accuracy_fedsgd": round(100 * accuracy(wf.astype(np.float64), tx, ty), 3),
        }
        print(f"[equiv] A {name}: max diff {worst:.3e}", flush=True)

    # ---- B. client drift khi tăng bước cục bộ và độ lệch dữ liệu
    wc_ref = centralized_steps(w0, x, y, TOTAL_STEPS, lr, L2)
    rows = []
    for alpha in ALPHAS:
        sh = dirichlet_shards(y, CLIENTS, alpha, SEED)
        for local_steps in LOCAL_STEPS:
            rounds = TOTAL_STEPS // local_steps
            w = w0.copy()
            clients: list[np.ndarray] = []
            for _ in range(rounds):
                w, clients = fedavg_round(w, x, y, sh, local_steps, lr, L2)
            client_acc = [100 * accuracy(wk, tx, ty) for wk in clients]
            rows.append({
                "alpha": alpha, "local_steps": local_steps, "rounds": rounds, "n_k": [int(len(s)) for s in sh],
                "relative_distance_to_centralized": relative_distance(w, wc_ref),
                "test_accuracy_aggregated": round(100 * accuracy(w, tx, ty), 3),
                "test_accuracy_centralized": round(100 * accuracy(wc_ref, tx, ty), 3),
                "test_accuracy_clients_before_aggregation_mean": round(float(np.mean(client_acc)), 3),
                "test_accuracy_clients_before_aggregation_min": round(float(np.min(client_acc)), 3),
                "test_accuracy_clients_before_aggregation_max": round(float(np.max(client_acc)), 3),
                "client_to_aggregate_relative_distance_mean": float(np.mean([relative_distance(wk, w) for wk in clients])),
            })
            r = rows[-1]
            print(f"[equiv] B alpha={alpha} E={local_steps}: dist={r['relative_distance_to_centralized']:.3e} "
                  f"agg={r['test_accuracy_aggregated']} clients={r['test_accuracy_clients_before_aggregation_mean']}", flush=True)

    report = {
        "evidence": "thí nghiệm numpy trên lớp cuối với đặc trưng 1024 chiều đã trích sẵn; KHÔNG phải mô hình của đồ án",
        "features": FEATURES, "train_examples": int(len(y)), "test_examples": int(len(ty)), "clients": CLIENTS, "seed": SEED,
        "lr_selection": {"rule": "lr có loss học thấp nhất sau 100 bước tập trung (không dùng tập test)", "losses": lr_losses, "chosen": lr},
        "fedsgd_equals_centralized": equivalence,
        "drift_grid": rows,
        "reading": _reading(equivalence, rows),
        "seconds": round(time.time() - started, 1),
    }
    out = reg.abs_path(reg.AGGREGATION_EQUIVALENCE_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[equiv] đã ghi {reg.AGGREGATION_EQUIVALENCE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

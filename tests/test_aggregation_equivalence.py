"""Phép gộp FedAvg trùng GD tập trung khi mỗi client chạy một bước (FedSGD): kiểm trên dữ liệu nhỏ ngẫu nhiên."""
from __future__ import annotations

import numpy as np

from scripts.aggregation_equivalence import (
    centralized_steps,
    dirichlet_shards,
    fedavg_round,
    loss_and_grad,
    relative_distance,
)


def _toy(n: int = 400, d: int = 16, c: int = 5, seed: int = 1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, size=n)
    x = rng.normal(size=(n, d)) + np.eye(c, d)[y] * 2.0
    w0 = np.zeros((d + 1, c))
    w0[:d] = rng.normal(0, 0.01, size=(d, c))
    return x, y, w0


def test_gradient_matches_finite_difference():
    x, y, w = _toy(60, 6, 3)
    w = w + 0.1
    _, g = loss_and_grad(w, x, y, 1e-3)
    eps = 1e-6
    for (i, j) in [(0, 0), (3, 2), (6, 1)]:  # hàng 6 là bias
        plus, minus = w.copy(), w.copy()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric = (loss_and_grad(plus, x, y, 1e-3)[0] - loss_and_grad(minus, x, y, 1e-3)[0]) / (2 * eps)
        assert abs(numeric - g[i, j]) < 1e-7


def test_dirichlet_shards_partition_all_examples_once():
    _, y, _ = _toy(1000, 4, 10)
    shards = dirichlet_shards(y, 5, 0.1, seed=3, min_size=10)
    joined = np.sort(np.concatenate(shards))
    assert np.array_equal(joined, np.arange(1000)) and min(len(s) for s in shards) >= 10


def test_one_local_step_fedavg_equals_centralized_gd_even_with_extreme_label_skew():
    x, y, w0 = _toy()
    shards = dirichlet_shards(y, 5, 0.1, seed=7, min_size=10)
    wc, wf = w0.copy(), w0.copy()
    for _ in range(30):
        wc = centralized_steps(wc, x, y, 1, 0.5, 1e-4)
        wf, _ = fedavg_round(wf, x, y, shards, 1, 0.5, 1e-4)
    assert relative_distance(wf, wc) < 1e-12


def test_many_local_steps_on_skewed_shards_drift_from_centralized():
    x, y, w0 = _toy()
    shards = dirichlet_shards(y, 5, 0.1, seed=7, min_size=10)
    wc = centralized_steps(w0, x, y, 40, 0.5, 1e-4)
    wf = w0.copy()
    for _ in range(8):  # 8 vòng x 5 bước = 40 bước tổng
        wf, _ = fedavg_round(wf, x, y, shards, 5, 0.5, 1e-4)
    assert relative_distance(wf, wc) > 1e-3

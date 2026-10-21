"""Phần thuần của scripts/eval_ood_gate.py: chia đôi theo mã băm, AUROC, chọn ngưỡng, khoảng tin cậy, độ tương tự láng giềng, điểm "lá"."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.eval_ood_gate import (
    auroc, evaluate_gate, half_of, knn_similarity, leaf_probability, rate, threshold_for_accept, threshold_for_reject, wilson,
)


def test_half_of_is_stable_and_roughly_balanced():
    keys = [f"key{i}" for i in range(2000)]
    halves = [half_of(k) for k in keys]
    assert halves == [half_of(k) for k in keys]
    assert 0.45 < sum(h == "calibration" for h in halves) / len(halves) < 0.55
    assert set(halves) == {"calibration", "report"}


def test_auroc_handles_ties_and_separation():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert auroc(np.array([0.3, 0.9]), np.array([0.5])) == pytest.approx(0.5)
    assert np.isnan(auroc(np.array([]), np.array([1.0])))


def test_threshold_for_accept_keeps_the_requested_share_of_good_images():
    scores = np.arange(100, dtype=float)                   # 0..99
    t = threshold_for_accept(scores, 0.95)
    assert (scores >= t).mean() >= 0.95 and (scores >= t).mean() < 0.97
    assert threshold_for_accept(scores, 1.0) == 0.0


def test_threshold_for_reject_rejects_the_requested_share_of_bad_images():
    scores = np.arange(100, dtype=float)
    t = threshold_for_reject(scores, 0.90)
    assert (scores < t).mean() >= 0.90 and (scores < t).mean() < 0.92


def test_wilson_interval_contains_the_rate_and_is_sane_at_the_edges():
    lo, hi = wilson(90, 100)
    assert lo < 0.9 < hi
    assert wilson(0, 0)[0] != wilson(0, 0)[0]              # NaN
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1
    assert rate(np.array([True, True, False, False])) == {"n": 4, "k": 2, "rate": 0.5, "ci95": wilson(2, 4)}


def test_knn_similarity_is_one_for_identical_vectors_and_low_for_orthogonal_ones():
    gallery = np.eye(4, dtype=np.float32)
    assert knn_similarity(np.eye(4, dtype=np.float32)[:2], gallery, k=1).tolist() == [1.0, 1.0]
    query = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    assert knn_similarity(query, gallery, k=2)[0] == pytest.approx(2 ** -0.5)
    assert knn_similarity(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), gallery[:3], k=1)[0] == pytest.approx(0.0)


def test_leaf_probability_sums_the_positive_prompts_only():
    text = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)    # hai câu 'lá' đầu, một câu loại
    leaf, other = np.array([[1.0, 0.0]], dtype=np.float32), np.array([[0.0, 1.0]], dtype=np.float32)
    assert leaf_probability(leaf, text, 2)[0] > 0.95 and leaf_probability(other, text, 2)[0] < 0.05
    assert leaf_probability(leaf, text, 3)[0] == pytest.approx(1.0)


def test_evaluate_gate_picks_thresholds_on_calibration_and_reports_on_the_other_half():
    rng = np.random.default_rng(0)
    id_scores = np.concatenate([rng.normal(0.9, 0.03, 400)])
    ood_scores = np.concatenate([rng.normal(0.3, 0.1, 200), rng.normal(0.90, 0.03, 100)])
    id_half = np.array(["calibration", "report"] * 200)
    ood_half = np.array(["calibration", "report"] * 150)
    ood_group = np.array(["general"] * 200 + ["other_leaf"] * 100)
    out = evaluate_gate(id_scores, ood_scores, id_half, ood_half, ood_group)
    assert out["auroc_general"] > 0.99 and out["auroc_other_leaf"] < 0.7           # cổng tách được ảnh không phải lá nhưng không tách được lá loài khác
    a = out["at_id_accept_95"]
    assert a["id_accept"]["rate"] >= 0.9 and a["general_reject"]["rate"] >= 0.9 and a["meets_target"] is True
    assert a["other_leaf_reject"]["rate"] < 0.5
    b = out["at_general_reject_90"]
    assert b["general_reject"]["rate"] >= 0.8 and b["id_accept"]["rate"] > 0.9

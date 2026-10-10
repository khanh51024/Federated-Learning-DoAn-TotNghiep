"""Phần phân tích của tái đánh giá cục bộ và audit leakage (thuần numpy, không cần dataset)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import audit_release_leakage as audit
from scripts import evaluate_published_models as ev

CLASSES = [f"Crop{i // 2}___c{i}" for i in range(6)]


def test_softmax_rows_sum_to_one_and_temperature_flattens():
    logits = np.array([[4.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    p = ev.softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ev.softmax(logits, 5.0)[0].max() < p[0].max()


def test_ece_is_zero_when_perfectly_calibrated_and_large_when_overconfident():
    conf = np.full(1000, 0.9)
    correct = np.zeros(1000)
    correct[:900] = 1.0
    assert ev.ece(conf, correct)[0] == pytest.approx(0.0, abs=1e-9)
    over, bins = ev.ece(np.full(1000, 0.99), np.r_[np.ones(500), np.zeros(500)])
    assert over == pytest.approx(0.49, abs=1e-6)
    assert bins[0]["n"] == 1000 and bins[0]["accuracy"] == 0.5


def test_fit_temperature_recovers_the_generating_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3.0, size=(30000, 8))
    true_t = 2.5
    p = ev.softmax(logits, true_t)
    targets = np.array([rng.choice(8, p=row) for row in p])
    fitted = ev.fit_temperature(logits, targets)
    assert fitted == pytest.approx(true_t, rel=0.08)
    assert ev.nll(logits, targets, fitted) < ev.nll(logits, targets, 1.0)


def test_temperature_scaling_never_changes_predictions():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(500, 6))
    assert (logits.argmax(1) == ev.softmax(logits, 3.7).argmax(1)).all()


def test_group_scores_and_per_class_recall():
    targets = np.array([0, 0, 1, 2, 4, 4])
    preds = np.array([0, 1, 1, 2, 5, 4])  # lớp 4 -> 5 cùng cây (Crop2) nên crop_accuracy đúng
    conf = np.array([0.95, 0.99, 0.6, 0.7, 0.92, 0.8])
    s = ev.group_scores(targets, preds, conf, CLASSES)
    assert s["count"] == 6 and s["correct"] == 4 and s["supported_classes"] == 4
    assert s["high_confidence_wrong_count_0_9"] == 2  # (0->1 @0.99) và (4->5 @0.92)
    assert s["crop_accuracy"] == pytest.approx(1.0)  # cả hai lỗi (0->1, 4->5) đều nhầm bệnh trong CÙNG một cây
    recall = ev.per_class_recall(targets, preds, CLASSES)
    assert recall[CLASSES[0]] == {"support": 2, "recall": 0.5}
    assert CLASSES[3] not in recall


def test_confusion_counts_every_sample_once():
    t = np.array([0, 1, 1, 2])
    p = np.array([0, 2, 1, 2])
    m = ev.confusion(t, p, 3)
    assert m.sum() == 4 and m[1, 2] == 1 and m[1, 1] == 1


def test_compare_with_published_flags_a_single_image_difference():
    own = {"count": 269, "correct": 127, "top1_accuracy": 127 / 269, "macro_f1_supported": 0.4687,
           "crop_accuracy": 0.7546, "high_confidence_wrong_count_0_9": 41}
    published = {"count": 269, "top1_accuracy": 127 / 269, "macro_f1_supported": 0.4686847,
                 "crop_accuracy": 0.7546468, "high_confidence_wrong_count_0_9": 41}
    assert ev.compare_with_published(own, published)["reproduced"] is True
    off = dict(own, correct=126, top1_accuracy=126 / 269)
    result = ev.compare_with_published(off, published)
    assert result["reproduced"] is False and result["correct_diff_images"] == -1
    assert ev.compare_with_published(own, {}) == {"available": False}


def test_analyze_model_reports_calibration_per_source():
    rng = np.random.default_rng(2)
    n = 400
    targets = rng.integers(0, 6, n)
    logits = rng.normal(0, 1, (n, 6)) + 4.0 * np.eye(6)[targets]  # có tín hiệu thật, hơi quá tự tin
    sources = np.array(["plantvillage"] * 300 + ["plantdoc"] * 100)
    cal_t = rng.integers(0, 6, 500)
    cal_l = rng.normal(0, 1, (500, 6)) + 4.0 * np.eye(6)[cal_t]
    out = ev.analyze_model(logits, targets, sources, cal_l, cal_t, CLASSES)
    assert set(out["calibration"]) == {"overall", "plantdoc", "plantvillage"}
    assert out["by_source"]["plantdoc"]["count"] == 100
    assert out["temperature_fitted_on_calibration"] > 0
    assert out["calibration"]["overall"]["ece_temperature_scaled"] <= out["calibration"]["overall"]["ece_raw"] + 0.05


def _rows(prefix: str, groups: list[str]) -> list[dict]:
    return [{"group_id": g, "parent_scene_id": g, "byte_sha256": f"{prefix}{i}", "sample_id": f"{prefix}s{i}",
             "image_id": f"{prefix}i{i}"} for i, g in enumerate(groups)]


def test_audit_overlaps_detects_shared_group_but_not_disjoint_sets():
    rows = {"train": _rows("t", ["g1", "g2"]), "val": _rows("v", ["g3"]), "calibration": _rows("c", ["g2"])}
    out = audit.audit_overlaps(rows)
    assert out["train__val"]["group_id"] == 0
    assert out["train__calibration"]["group_id"] == 1
    assert out["train__calibration"]["byte_sha256"] == 0  # cùng nhóm nhưng khác file


def test_audit_partition_flags_shards_that_are_not_a_clean_split_of_train():
    train = _rows("t", ["g1", "g2", "g3", "g4"])
    ids = {r["sample_id"] for r in train}
    clean = audit.audit_partition({"c0": train[:2], "c1": train[2:]}, ids)
    assert clean["disjoint"] and clean["subset_of_train"] and clean["union_equals_train"]
    assert clean["groups_split_across_clients"] == 0
    dup = audit.audit_partition({"c0": train[:3], "c1": train[2:]}, ids)
    assert dup["disjoint"] is False
    foreign = audit.audit_partition({"c0": train[:2] + _rows("x", ["gx"])}, ids)
    assert foreign["subset_of_train"] is False
    shared_group = audit.audit_partition({"c0": [train[0]], "c1": [dict(train[1], group_id="g1")]}, ids)
    assert shared_group["groups_split_across_clients"] == 1


def test_within_tolerance_is_explicit_and_stricter_than_it_looks():
    base = {"count": 4344, "top1_accuracy": 2397 / 4344, "macro_f1_supported": 0.5, "crop_accuracy": 0.9,
            "high_confidence_wrong_count_0_9": 100}
    published = dict(base)
    one_off = {**base, "correct": 2396, "top1_accuracy": 2396 / 4344}
    r = ev.compare_with_published(one_off, published)
    assert r["reproduced"] is False and r["within_tolerance"] is True and r["correct_diff_images"] == -1
    three_off = {**base, "correct": 2394, "top1_accuracy": 2394 / 4344}
    assert ev.compare_with_published(three_off, published)["within_tolerance"] is False
    hcw_far = {**base, "correct": 2397, "high_confidence_wrong_count_0_9": 104}
    assert ev.compare_with_published(hcw_far, published)["within_tolerance"] is False


def test_exact_match_requires_equal_high_confidence_wrong_count():
    base = {"count": 100, "correct": 80, "top1_accuracy": 0.8, "macro_f1_supported": 0.7, "crop_accuracy": 0.9,
            "high_confidence_wrong_count_0_9": 10}
    published = {k: v for k, v in base.items() if k != "correct"}
    assert ev.compare_with_published(base, published)["reproduced"] is True
    differ = {**base, "high_confidence_wrong_count_0_9": 12}
    r = ev.compare_with_published(differ, published)
    assert r["reproduced"] is False and r["within_tolerance"] is True and r["high_confidence_wrong_diff_images"] == 2

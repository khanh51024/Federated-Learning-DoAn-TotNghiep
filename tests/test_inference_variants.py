"""Phép tính thuần numpy của scripts/eval_inference_variants.py (không cần dataset hay checkpoint)."""
from __future__ import annotations

import numpy as np

from scripts.eval_inference_variants import (
    VARIANTS,
    VIEWS,
    crop_masks,
    paired_cluster_ci,
    restrict_to_crop,
    selective_table,
    softmax,
    topk_correct,
    view_images,
)

CLASSES = ["Apple___scab", "Apple___healthy", "Tomato___blight", "Tomato___healthy"]


def test_variants_only_reference_known_views():
    for name, views in VARIANTS.items():
        assert views and set(views) <= set(VIEWS), name
    assert VARIANTS["base"] == ("squash",)


def test_crop_masks_group_classes_by_crop():
    names, masks = crop_masks(CLASSES)
    assert names == ["Apple", "Tomato"]
    assert masks.tolist() == [[True, True, False, False], [False, False, True, True]]


def test_restrict_to_crop_zeroes_other_crops_and_renormalises():
    probs = np.array([[0.1, 0.1, 0.7, 0.1]])
    free = restrict_to_crop(probs, CLASSES)  # tự đoán: Tomato có 0,8 > Apple 0,2
    assert free.argmax() == 2 and np.isclose(free.sum(), 1.0) and free[0, :2].sum() == 0
    forced = restrict_to_crop(probs, CLASSES, np.array([0]))  # người dùng chọn Apple
    assert forced.argmax() in (0, 1) and np.isclose(forced.sum(), 1.0) and forced[0, 2:].sum() == 0


def test_predicted_crop_uses_probability_mass_not_top_class():
    # lớp Apple___scab cao nhất (0,34) nhưng Tomato có tổng 0,66: cây phải là Tomato
    probs = np.array([[0.34, 0.0, 0.33, 0.33]])
    assert restrict_to_crop(probs, CLASSES).argmax() in (2, 3)


def test_topk_correct():
    probs = np.array([[0.5, 0.3, 0.15, 0.05], [0.05, 0.1, 0.3, 0.55]])
    assert topk_correct(probs, np.array([1, 0]), 2).tolist() == [True, False]


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isclose(p.sum(), 1.0) and p[0, 0] > 0.999


def test_paired_ci_detects_clear_gain_and_zero_when_identical():
    rng = np.random.default_rng(0)
    groups = [f"g{i // 2}" for i in range(200)]
    base = np.zeros(200, dtype=bool)
    better = np.ones(200, dtype=bool)
    gain = paired_cluster_ci(groups, base, better, rng, resamples=300)
    assert gain["delta"] == 100.0 and gain["low"] == 100.0
    same = paired_cluster_ci(groups, base, base, rng, resamples=300)
    assert same["delta"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0


def test_selective_table_accuracy_rises_when_confidence_is_informative():
    conf = np.linspace(1.0, 0.1, 100)
    correct = np.arange(100) < 70  # 70 ảnh tự tin nhất đúng, 30 ảnh kém tự tin nhất sai
    out = selective_table(conf, correct)
    assert out["table"]["coverage_100"]["accuracy"] == 70.0
    assert out["table"]["coverage_60"]["accuracy"] == 100.0
    assert out["auroc_confidence_detects_wrong"] == 1.0


def test_view_images_shapes():
    from PIL import Image

    img = Image.new("RGB", (80, 40), (10, 200, 30))
    views = view_images(img)
    assert set(views) == set(VIEWS)
    assert views["letterbox"].size == (80, 80)  # giữ tỉ lệ: đệm đen thành hình vuông
    assert views["center80"].size == (64, 32)
    assert views["squash_flip"].size == (80, 40)

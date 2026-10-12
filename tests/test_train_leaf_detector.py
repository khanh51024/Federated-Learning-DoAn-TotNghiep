"""Test phần thuần của scripts/train_leaf_detector.py: IoU, ghép khung, chống rò rỉ cảnh."""
from __future__ import annotations

import numpy as np

from scripts import train_leaf_detector as td


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    iou = td.iou_matrix(a, b)[0]
    assert iou[0] == 1.0
    assert abs(iou[1] - (50 / 150)) < 1e-9
    assert iou[2] == 0.0
    assert td.iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_match_boxes_counts_tp_fp_fn_and_respects_threshold():
    gt = np.array([[0, 0, 10, 10], [50, 50, 70, 70]], dtype=float)
    pred = np.array([[0, 0, 10, 10], [200, 200, 220, 220], [51, 51, 70, 70]], dtype=float)
    scores = np.array([0.9, 0.8, 0.3])
    assert td.match_boxes(pred, scores, gt, score_thr=0.5) == {"tp": 1, "fp": 1, "fn": 1}
    assert td.match_boxes(pred, scores, gt, score_thr=0.2) == {"tp": 2, "fp": 1, "fn": 0}


def test_match_boxes_does_not_match_one_ground_truth_twice():
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], dtype=float)
    assert td.match_boxes(pred, np.array([0.9, 0.8]), gt) == {"tp": 1, "fp": 1, "fn": 0}


def test_training_ids_exclude_every_non_train_split():
    images = {
        "a": {"split": "train"}, "b": {"split": "validation"}, "c": {"split": "test"},
        "d": {"split": "calibration"}, "e": {"split": "quarantine"}, "f": {"split": "train"},
    }
    assert td.training_image_ids(images) == ["a", "f"]

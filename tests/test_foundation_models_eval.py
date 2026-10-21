"""Phần thuần của scripts/eval_foundation_models.py: tên tự nhiên của lớp, kNN, zero-shot (không cần torch)."""
from __future__ import annotations

import json

import numpy as np

from scripts.eval_foundation_models import crop_allowed, knn_scores, natural_names, prompts_for, zero_shot_scores
from src import registry as reg


def test_natural_names_cover_special_taxonomy_names():
    assert natural_names("Tomato___Late_blight") == ("tomato", "late blight")
    assert natural_names("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot") == ("corn", "gray leaf spot")
    assert natural_names("Pepper,_bell___Bacterial_spot") == ("bell pepper", "bacterial spot")
    assert natural_names("Orange___Haunglongbing_(Citrus_greening)") == ("orange", "citrus greening")
    assert natural_names("Cherry_(including_sour)___healthy") == ("cherry", "healthy")


def test_prompts_for_every_taxonomy_class_are_non_empty_and_distinct():
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    texts = [tuple(prompts_for(c)) for c in classes]
    assert len(set(texts)) == len(classes) == 38
    assert all("___" not in p and "_" not in p for t in texts for p in t)  # không còn gạch dưới của tên lớp thô


def test_knn_votes_the_nearest_class_and_respects_the_allowed_mask():
    gallery = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    query = np.array([[0.95, 0.05], [0.05, 0.95]])
    free = knn_scores(query, gallery, labels, k=2, classes=3)
    assert free.argmax(1).tolist() == [0, 1]
    allowed = np.array([[False, True, False], [False, True, False]])  # chỉ cho lớp 1
    restricted = knn_scores(query, gallery, labels, k=2, classes=3, allowed=allowed)
    assert restricted.argmax(1).tolist() == [1, 1] and restricted[:, 0].sum() == 0


def test_zero_shot_scores_are_a_distribution_and_pick_the_matching_text():
    text = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    image = np.array([[0.9, 0.1], [0.1, 0.9]])
    p = zero_shot_scores(image, text)
    assert np.allclose(p.sum(1), 1.0) and p.argmax(1).tolist() == [0, 1]


def test_crop_allowed_marks_same_crop_classes():
    classes = ["Apple___scab", "Apple___healthy", "Tomato___blight"]
    allowed = crop_allowed(classes, None, np.array([0, 2]))
    assert allowed.tolist() == [[True, True, False], [False, False, True]]


def test_leaf_filter_probability_sums_positive_prompts():
    import numpy as np

    from scripts.filter_leaf_photos import leaf_probability, prompts_for

    pos, neg = prompts_for("an apple tree leaf")
    assert all("apple tree leaf" in p for p in pos) and len(neg) >= 8
    probs = np.array([[0.3, 0.3, 0.2, 0.1, 0.1], [0.05, 0.05, 0.05, 0.5, 0.35]])
    assert np.allclose(leaf_probability(probs, 3), [0.8, 0.15])

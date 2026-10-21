"""Phần thuần của scripts/check_own_photos.py: tên lớp ngắn, chia theo cây, kiểm dòng."""
from __future__ import annotations

from scripts.check_own_photos import normalize_class, plant_test_assignment, validate_row

CLASSES = ["Tomato___Late_blight", "Corn_(maize)___Common_rust_", "Pepper,_bell___healthy"]


def test_normalize_class_accepts_short_names():
    assert normalize_class("tomato late blight", CLASSES) == "Tomato___Late_blight"
    assert normalize_class("Tomato - Late blight", CLASSES) == "Tomato___Late_blight"
    assert normalize_class("corn (maize) common rust", CLASSES) == "Corn_(maize)___Common_rust_"
    assert normalize_class("pepper bell healthy", CLASSES) == "Pepper,_bell___healthy"
    assert normalize_class("banana leaf", CLASSES) is None


def test_plant_split_keeps_each_plant_on_one_side_and_both_sides_populated():
    plants = {"A": [f"p{i}" for i in range(10)] * 3, "B": ["x", "y"], "C": ["only"]}
    a = plant_test_assignment(plants)
    b = plant_test_assignment({k: list(reversed(v)) for k, v in plants.items()})
    assert a == b  # không phụ thuộc thứ tự dòng
    assert sum(1 for (c, _), s in a.items() if c == "A" and s == "test") == 3  # 30% của 10 cây
    assert {s for (c, _), s in a.items() if c == "B"} == {"train", "test"}  # 2 cây: mỗi phía một cây
    assert a[("C", "only")] == "train"  # 1 cây: không thử được


def test_validate_row_rejects_unknown_missing_plant_and_unsure():
    ok = {"class": "tomato late blight", "plant_id": "t1", "verified": "self"}
    assert validate_row(ok, CLASSES) == ("Tomato___Late_blight", None)
    assert validate_row({**ok, "class": "kiwi"}, CLASSES)[1] == "unknown_class"
    assert validate_row({**ok, "plant_id": " "}, CLASSES)[1] == "missing_plant_id"
    assert validate_row({**ok, "verified": "unsure"}, CLASSES)[1] == "marked_unsure"

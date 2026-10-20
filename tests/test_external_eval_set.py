"""Hàm thuần của scripts/build_external_eval_set.py: dHash, khoảng cách Hamming, bảng nhãn khớp taxonomy."""
from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.build_external_eval_set import (
    DISEASE_MAP,
    HEALTHY_MAP,
    class_names_in_maps,
    dhash64,
    hamming_to_all,
    nearest_distance,
)


def _gradient(width: int, height: int, flip: bool = False) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    arr = np.tile(x, (height, 1))
    if flip:
        arr = arr[:, ::-1]
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1))


def test_dhash_is_stable_under_resize_and_jpeg_like_noise():
    a = dhash64(_gradient(256, 256))
    b = dhash64(_gradient(97, 131))  # cùng nội dung, kích thước khác
    refs = np.array([b], dtype=np.uint64)
    assert nearest_distance(a, refs) <= 6
    assert nearest_distance(a, np.array([dhash64(_gradient(256, 256, flip=True))], dtype=np.uint64)) > 6


def test_hamming_counts_differing_bits():
    refs = np.array([0, 1, 3, 2**63], dtype=np.uint64)
    assert hamming_to_all(0, refs).tolist() == [0, 1, 2, 1]
    assert nearest_distance(7, np.array([], dtype=np.uint64)) == 64


def test_label_maps_use_only_taxonomy_classes_and_no_ambiguous_names():
    from web.demo_evidence import load_evidence

    classes = load_evidence()["classes"]
    assert set(class_names_in_maps()) <= set(classes)
    assert not set(DISEASE_MAP) & set(HEALTHY_MAP)
    assert "grape leaf spot" not in DISEASE_MAP  # mơ hồ với Leaf blight (Isariopsis), cố ý bỏ
    assert all(name.endswith("___healthy") for name in HEALTHY_MAP.values())

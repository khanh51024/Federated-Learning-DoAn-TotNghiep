"""Phần thuần của scripts/import_external_source.py: so khớp nhãn, khóa nhãn từ tên tệp, chuẩn hóa ảnh."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from scripts.import_external_source import label_key, normalize_image_bytes, normalize_key, resolve_class


def test_normalize_key_ignores_case_and_separators():
    assert normalize_key("Early_Blight") == normalize_key("early  blight") == normalize_key("EARLY-BLIGHT")


def test_resolve_class_prefers_exact_then_alias_then_unmapped():
    exact = {"Early Blight": "Tomato___Early_blight"}
    alias = {"Curl": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"}
    assert resolve_class("early_blight", exact, alias) == ("Tomato___Early_blight", "exact_match")
    assert resolve_class("Curl", exact, alias) == ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "alias")
    assert resolve_class("Leaf Miner Flies", exact, alias) == (None, "unmapped")


def test_label_key_from_folder_and_filename_prefix():
    raw = Path("raw")
    assert label_key(Path("raw/a/Healthy/x.jpg"), raw, "folder") == "Healthy"
    assert label_key(Path("raw/Early Blight_12.jpg"), raw, "filename_prefix") == "Early Blight"
    assert label_key(Path("raw/Bacteiral Spot_100.jpg"), raw, "filename_prefix") == "Bacteiral Spot"
    assert label_key(Path("raw/Late Blight (3).jpg"), raw, "filename_prefix") == "Late Blight"


def test_normalize_image_bytes_resizes_long_side_and_returns_rgb_jpeg():
    big = Image.new("RGB", (2000, 1000), (10, 120, 30))
    out = Image.open(io.BytesIO(normalize_image_bytes(big)))
    assert out.format == "JPEG" and out.mode == "RGB" and max(out.size) == 640 and out.size == (640, 320)
    small = Image.new("RGBA", (200, 100), (255, 0, 0, 0))  # trong suốt hoàn toàn -> nền đen như lúc train
    arr = Image.open(io.BytesIO(normalize_image_bytes(small)))
    assert arr.size == (200, 100) and sum(arr.getpixel((5, 5))) < 40


def test_group_of_keeps_offline_augmentations_together():
    from scripts.import_external_source import group_of

    a = group_of("src", "T", Path("x/128110613-ab_aug_207.jpg"), "1" * 64, {}, "aug_prefix")
    b = group_of("src", "T", Path("x/128110613-ab_aug_208.jpg"), "2" * 64, {}, "aug_prefix")
    c = group_of("src", "T", Path("x/134201~aug_80.jpg"), "3" * 64, {}, "aug_prefix")
    assert a == b and a != c
    assert group_of("src", "T", Path("x/a.jpg"), "f" * 64, {}, "image") == "src:" + "f" * 16
    assert group_of("inat", "T", Path("x/a.jpg"), "f" * 64, {"observation_id": 77}, "image") == "inat:obs77"

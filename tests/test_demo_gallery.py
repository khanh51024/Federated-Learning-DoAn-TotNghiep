"""Thư viện ảnh mẫu của trang Chẩn đoán: catalog khớp ảnh trên đĩa, nhãn thuộc 38 lớp, byte không bị đổi."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import build_demo_gallery as gallery
from src import registry as reg

CATALOG_PATH = reg.abs_path(reg.DEMO_GALLERY_CATALOG_JSON)
GALLERY_DIR = reg.abs_path(reg.DEMO_GALLERY_DIR)


@pytest.fixture(scope="module")
def catalog() -> dict:
    if not CATALOG_PATH.is_file():
        pytest.skip("chưa dựng thư viện ảnh mẫu (scripts/build_demo_gallery.py)")
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_parse_difficulty_splits_level_and_field_photo():
    assert gallery.parse_difficulty("Rất khó, ngoài đồng") == ("Rất khó", "very_hard", True)
    assert gallery.parse_difficulty("Dễ") == ("Dễ", "easy", False)
    with pytest.raises(ValueError):
        gallery.parse_difficulty("Cực kỳ khó")


def test_catalog_items_match_files_and_taxonomy(catalog):
    classes = set(json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"])
    assert len(classes) == 38
    assert catalog["count"] == len(catalog["items"]) == 19
    ids = [item["id"] for item in catalog["items"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for item in catalog["items"]:
        path = GALLERY_DIR / item["file"]
        assert path.is_file(), item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], f"{item['file']} bị đổi byte"
        assert path.stat().st_size == item["bytes"]
        assert (GALLERY_DIR / item["thumb"]).is_file()
        assert item["label"] in classes
        assert item["source"] in {"plantvillage", "plantdoc"}
        assert item["difficulty_key"] in set(gallery.DIFFICULTY_KEYS.values())
        assert 0 <= item["recall_fedavg_pct"] <= 100 and 0 <= item["recall_centralized_pct"] <= 100
        assert item["class_val_count"] > 0


def test_catalog_is_ordered_from_easy_to_hard(catalog):
    order = list(gallery.DIFFICULTY_KEYS.values())
    ranks = [order.index(item["difficulty_key"]) for item in catalog["items"]]
    assert ranks == sorted(ranks), "ảnh phải xếp từ dễ đến khó"
    assert {item["source"] for item in catalog["items"]} == {"plantvillage", "plantdoc"}


def test_catalog_states_how_validation_membership_was_checked(catalog):
    membership = catalog["validation_membership"]
    assert membership["status"] in {"verified", "not_verified"}
    assert membership["reason"]
    assert "VALIDATION" in catalog["evidence"]

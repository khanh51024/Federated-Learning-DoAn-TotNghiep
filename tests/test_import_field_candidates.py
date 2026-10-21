"""scripts/import_field_candidates.py: nạp ảnh ứng viên đã sàng lọc từng ảnh vào thư mục nguồn."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import import_field_candidates as imp
from src import registry as reg

CLASS = "Strawberry___Leaf_scorch"


def candidate(photo_id: int, target_class: str = CLASS, obs: int = 1) -> dict:
    return {"photo_id": photo_id, "observation_id": obs, "target_class": target_class, "sample_id": f"c{photo_id}", "byte_sha256": "a" * 64,
            "license": "cc-by", "author": "x", "observation_url": f"https://www.inaturalist.org/observations/{obs}", "taxon": "Diplocarpon fragariae",
            "relative_path": f"src/images/{photo_id}.jpg"}


def test_select_accepted_keeps_only_listed_photos_and_sorts_them():
    rows = [candidate(3), candidate(1), candidate(2)]
    assert [r["photo_id"] for r in imp.select_accepted(rows, [3, 1, 3], CLASS)] == [1, 3]


def test_select_accepted_refuses_a_photo_missing_from_the_candidate_set():
    with pytest.raises(SystemExit, match="thiếu 1 ảnh"):
        imp.select_accepted([candidate(1)], [1, 9], CLASS)


def test_select_accepted_refuses_a_photo_of_another_class():
    with pytest.raises(SystemExit, match="không thuộc lớp"):
        imp.select_accepted([candidate(1, "Tomato___healthy")], [1], CLASS)


def test_manifest_row_matches_the_format_of_the_other_inaturalist_sources():
    row = imp.manifest_row("inat_scorch", 26, candidate(7, obs=55), "b" * 64, 300, 400, 5, 6)
    assert row["sample_id"] == "ext_inat_scorch_" + "b" * 16
    assert row["relative_path"] == f"inat_scorch/images/{CLASS}/{'b' * 16}.jpg"
    assert row["group_id"] == "inat_scorch:obs55"                       # cùng quan sát = cùng nhóm, không tách qua split
    assert (row["dhash64"], row["dhash64_flip"]) == ("0000000000000005", "0000000000000006")
    assert row["label_trust"] == "community" and row["label_review_status"] == "unreviewed"
    assert row["source_domain"] == "plantdoc" and row["class_id"] == 26


def test_check_existing_refuses_an_unknown_sample(tmp_path, monkeypatch):
    base = tmp_path / "sources" / "s1"
    base.mkdir(parents=True)
    (base / "manifest.jsonl").write_text(json.dumps({"sample_id": "ext_s1_1"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(imp, "EXT_SOURCES", str(tmp_path / "sources"))
    monkeypatch.setattr(reg, "abs_path", lambda p: Path(p))
    assert imp.check_existing(["ext_s1_1"]) == ["ext_s1_1"]
    with pytest.raises(SystemExit, match="không có trong manifest"):
        imp.check_existing(["ext_s1_1", "ext_nope_9"])


def test_import_source_normalizes_image_bytes_and_records_their_sha(tmp_path, monkeypatch):
    root, sources = tmp_path / "cand", tmp_path / "sources"
    (root / "src" / "images").mkdir(parents=True)
    Image.new("RGB", (40, 30), (10, 120, 30)).save(root / "src" / "images" / "1.jpg")
    (root / "src" / "manifest.jsonl").write_text(json.dumps(candidate(1)) + "\n", encoding="utf-8")
    monkeypatch.setattr(imp, "EXT_SOURCES", str(sources))
    monkeypatch.setattr(reg, "abs_path", lambda p: Path(p))
    classes = ["x"] * 26 + [CLASS]
    rows = imp.import_source("src", {"accepted_photo_ids": [1], "target_class": CLASS, "taxon": "T", "candidates_total": 1}, root, classes)
    (row,) = rows
    stored = (sources / row["relative_path"]).read_bytes()
    import hashlib

    assert hashlib.sha256(stored).hexdigest() == row["byte_sha256"]      # byte lưu = byte đã băm, hợp đồng của bộ chạy train
    assert Image.open(io.BytesIO(stored)).size == (row["width"], row["height"]) == (40, 30)
    assert json.loads((sources / "src" / "import_report.json").read_text(encoding="utf-8"))["images_imported"] == 1


def test_import_source_keeps_one_row_for_two_photos_with_identical_bytes(tmp_path, monkeypatch):
    root, sources = tmp_path / "cand", tmp_path / "sources"
    (root / "src" / "images").mkdir(parents=True)
    for name in ("1.jpg", "2.jpg"):
        Image.new("RGB", (40, 30), (10, 120, 30)).save(root / "src" / "images" / name)
    rows = [candidate(1), candidate(2, obs=2)]
    (root / "src" / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(imp, "EXT_SOURCES", str(sources))
    monkeypatch.setattr(reg, "abs_path", lambda p: Path(p))
    out = imp.import_source("src", {"accepted_photo_ids": [1, 2], "target_class": CLASS, "taxon": "T", "candidates_total": 2}, root, ["x"] * 26 + [CLASS])
    assert len(out) == 1                       # sample_id suy từ sha: hai dòng cùng sha sẽ đè nhau ở manifest

"""Test phần thuần của scripts/eval_fullframe_plantdoc.py: hình học cắt, nhãn theo cảnh, từ chối ảnh sai SHA-256."""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

from scripts import eval_fullframe_plantdoc as ff


def test_crop_box_adds_margin_per_side_and_clamps_to_image():
    # khung 100x200 ở giữa ảnh 1000x1000: lề 8% mỗi phía = 8 và 16 px
    assert ff.crop_box(1000, 1000, [300, 400, 400, 600]) == (292, 384, 408, 616)
    # sát mép: kẹp trong ảnh, không âm / không vượt
    assert ff.crop_box(500, 400, [0, 0, 500, 400]) == (0, 0, 500, 400)
    left, top, right, bottom = ff.crop_box(500, 400, [490, 390, 500, 400])
    assert 0 <= left < right <= 500 and 0 <= top < bottom <= 400


def test_crop_box_never_returns_empty_box():
    left, top, right, bottom = ff.crop_box(100, 100, [50, 50, 50, 50])
    assert right > left and bottom > top


def test_center_square_is_centered_and_square():
    assert ff.center_square(1600, 1200) == (200, 0, 1400, 1200)
    assert ff.center_square(600, 900) == (0, 150, 600, 750)


def test_single_class_scenes_only_keeps_scenes_with_one_class():
    annotations = {
        "a1": {"image_id": "s1"}, "a2": {"image_id": "s1"},  # cùng lớp 5
        "a3": {"image_id": "s2"}, "a4": {"image_id": "s2"},  # hai lớp khác nhau
    }
    rows = [
        {"annotation_id": "a1", "class_id": 5}, {"annotation_id": "a2", "class_id": 5},
        {"annotation_id": "a3", "class_id": 1}, {"annotation_id": "a4", "class_id": 2},
    ]
    assert ff.single_class_scenes(rows, annotations) == {"s1": 5}


def test_download_rejects_file_with_wrong_sha256(tmp_path: Path, monkeypatch):
    payload = b"not the published image"
    images = {"img1": {"relative_path": "PlantDoc-Object-Detection-Dataset/TEST/a.jpg", "byte_sha256": hashlib.sha256(b"other").hexdigest()}}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=30: _Response(payload))
    stats = ff.download_missing(tmp_path, images, {"img1"})
    assert stats["rejected_sha"] == 1 and stats["downloaded"] == 0
    assert not (tmp_path / "TEST" / "a.jpg").exists(), "ảnh sai SHA-256 không được giữ lại"


def test_download_keeps_file_with_correct_sha256(tmp_path: Path, monkeypatch):
    payload = b"published bytes"
    images = {"img1": {"relative_path": "PlantDoc-Object-Detection-Dataset/TEST/a.jpg", "byte_sha256": hashlib.sha256(payload).hexdigest()}}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=30: _Response(payload))
    stats = ff.download_missing(tmp_path, images, {"img1"})
    assert stats["downloaded"] == 1
    assert (tmp_path / "TEST" / "a.jpg").read_bytes() == payload
    # lần sau: đã có và khớp → không tải lại
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được tải lại")))
    assert ff.download_missing(tmp_path, images, {"img1"})["present"] == 1

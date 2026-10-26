"""Đếm ảnh của scripts/preflight_plantvillage.py: mỗi tệp đếm đúng một lần, kể cả trên hệ tệp không phân biệt hoa thường (Windows)."""
from __future__ import annotations

from scripts import preflight_plantvillage as pf


def test_each_image_is_counted_once_whatever_the_suffix_case(tmp_path):
    cls = tmp_path / "Apple___Apple_scab"
    cls.mkdir()
    for name in ("a.JPG", "b.jpg", "c.jpeg", "d.PNG"):
        (cls / name).write_bytes(b"x")
    (cls / "notes.txt").write_text("không phải ảnh", encoding="utf-8")
    (cls / "sub.jpg").mkdir()                       # thư mục mang đuôi ảnh không được đếm
    (tmp_path / "README.md").write_text("tệp ở gốc", encoding="utf-8")
    (tmp_path / "empty_class").mkdir()              # lớp rỗng không xuất hiện
    assert pf._class_counts(tmp_path) == {"Apple___Apple_scab": 4}

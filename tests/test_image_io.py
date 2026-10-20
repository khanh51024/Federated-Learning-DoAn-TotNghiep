"""Giải mã ảnh người dùng: xoay EXIF và nền trong suốt phải khớp cách nhóm train đọc ảnh."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.common.image_io import to_rgb


def _tagged_jpeg(orientation: int) -> Image.Image:
    # Ảnh gốc 40x20: nửa trái đỏ, nửa phải xanh; gắn thẻ xoay EXIF rồi nạp lại từ byte (đúng đường tải lên).
    base = Image.new("RGB", (40, 20), (255, 0, 0))
    base.paste((0, 0, 255), (20, 0, 40, 20))
    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    base.save(buffer, format="JPEG", quality=100, exif=exif)
    buffer.seek(0)
    return Image.open(buffer)


def test_exif_orientation_6_and_8_are_rotated_upright():
    # Thẻ 6 = cần xoay 90 độ theo chiều kim đồng hồ; thẻ 8 = ngược chiều. Ảnh 40x20 thành 20x40.
    for orientation in (6, 8):
        out = to_rgb(_tagged_jpeg(orientation))
        assert out.size == (20, 40), orientation
        assert out.mode == "RGB"


def test_exif_rotation_matches_pre_rotated_image():
    tagged = to_rgb(_tagged_jpeg(6))
    manual = _tagged_jpeg(1).convert("RGB").transpose(Image.Transpose.ROTATE_270)
    diff = np.abs(np.asarray(tagged, dtype=int) - np.asarray(manual, dtype=int)).max()
    assert diff <= 4  # JPEG hai lần mã hóa cùng nội dung, chỉ chênh nhiễu nén


def test_to_rgb_is_idempotent_and_never_rotates_twice():
    once = to_rgb(_tagged_jpeg(6))
    twice = to_rgb(once)
    assert once.size == twice.size == (20, 40)
    assert np.array_equal(np.asarray(once), np.asarray(twice))


def test_transparent_png_is_pasted_on_black():
    rgba = Image.new("RGBA", (8, 8), (255, 255, 255, 0))  # trắng nhưng trong suốt hoàn toàn
    rgba.putpixel((0, 0), (0, 255, 0, 255))
    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG")
    buffer.seek(0)
    out = np.asarray(to_rgb(Image.open(buffer)))
    assert tuple(out[0, 0]) == (0, 255, 0)
    assert tuple(out[4, 4]) == (0, 0, 0)  # đúng như lúc train, không phải nền trắng ẩn sau alpha


def test_palette_with_transparency_is_pasted_on_black():
    pal = Image.new("P", (4, 4), 0)
    pal.putpalette([255, 255, 255] + [0, 0, 0] * 255)
    pal.info["transparency"] = 0
    out = np.asarray(to_rgb(pal))
    assert tuple(out[2, 2]) == (0, 0, 0)


def test_plain_rgb_without_exif_is_unchanged():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (16, 24, 3), dtype=np.uint8))
    assert np.array_equal(np.asarray(to_rgb(image)), np.asarray(image))

"""Giải mã ảnh về RGB đúng như lúc train và lúc chấm điểm.

Một cửa duy nhất cho web: xoay theo EXIF, ảnh trong suốt dán lên nền đen, rồi đổi sang RGB. Cùng quy tắc với
`CanonicalClassificationDataset` của nhóm train và `scripts/evaluate_published_models.py`, nên ảnh người dùng chụp dọc
bằng điện thoại không còn bị đưa vào mô hình ở tư thế xoay 90 độ.
"""
from __future__ import annotations

from typing import Any


def to_rgb(image: Any) -> Any:
    """Trả về ảnh RGB đã xoay thẳng; gọi lại nhiều lần trên cùng ảnh cho cùng kết quả (không xoay hai lần)."""
    from PIL import Image, ImageOps

    try:
        if image.mode == "RGB" and image.getexif().get(0x0112, 1) in (0, 1, None):
            image.load()  # đã thẳng và đã là RGB: không cần sao chép (nhanh gấp đôi trên ảnh 12-40 MP)
            return image
    except Exception:  # noqa: BLE001 — EXIF hỏng: đi tiếp đường đầy đủ bên dưới
        pass
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # noqa: BLE001 — EXIF hỏng thì dùng ảnh gốc, giống normalize_exif_orientation của nhóm train
        pass
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (0, 0, 0))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")

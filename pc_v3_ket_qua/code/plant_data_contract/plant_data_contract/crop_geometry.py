"""Bounding box margin expansion, clamping, EXIF normalization, and cropping."""

import math
from typing import List, Tuple
from PIL import Image, ImageOps


def normalize_exif_orientation(image: Image.Image) -> Image.Image:
    """Transpose image according to EXIF orientation tag and strip orientation."""
    return ImageOps.exif_transpose(image)


def validate_bbox_xyxy(image_width, image_height, bbox_xyxy, overflow_tolerance_px=1.0):
    """Validate post-EXIF float XYXY; permit only <=1px boundary rounding.

    Return clipped display XYXY. The caller must retain the raw box separately.
    Invalid geometry raises ValueError and is never repaired into a tiny crop.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image frame")
    if not math.isfinite(overflow_tolerance_px) or overflow_tolerance_px < 0:
        raise ValueError("Invalid overflow tolerance")
    if len(bbox_xyxy) != 4:
        raise ValueError("Expected four XYXY coordinates")
    xmin, ymin, xmax, ymax = map(float, bbox_xyxy)
    if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        raise ValueError("Nonfinite box")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Empty or reversed box")
    if max(-xmin, -ymin, xmax-image_width, ymax-image_height) > overflow_tolerance_px:
        raise ValueError("Box exceeds boundary tolerance")
    clipped = (max(0.0, xmin), max(0.0, ymin), min(float(image_width), xmax), min(float(image_height), ymax))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Box has no overlap with frame")
    return clipped


def crop_bounding_box(
    image_width: int,
    image_height: int,
    bbox_xyxy: List[float],
    margin_ratio: float = 0.08,
) -> Tuple[int, int, int, int]:
    """Calculate integer pixel coordinates for cropping a bounding box with safety margin.

    Args:
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        bbox_xyxy: [xmin, ymin, xmax, ymax] in pixel coordinates.
        margin_ratio: Fractional margin expansion per side (default: 8%).

    Returns:
        (left, top, right, bottom) integer coordinates clamped to image boundaries.
    """
    validate_bbox_xyxy(image_width, image_height, bbox_xyxy)
    # Expand the raw float box, then round and clip. Display clipping is separate.
    xmin, ymin, xmax, ymax = map(float, bbox_xyxy)
    if not math.isfinite(margin_ratio) or margin_ratio < 0:
        raise ValueError("Invalid margin ratio")
    box_w = xmax - xmin
    box_h = ymax - ymin

    dx = box_w * margin_ratio
    dy = box_h * margin_ratio

    left = max(0, int(math.floor(xmin - dx)))
    top = max(0, int(math.floor(ymin - dy)))
    right = min(image_width, int(math.ceil(xmax + dx)))
    bottom = min(image_height, int(math.ceil(ymax + dy)))

    if right <= left or bottom <= top:
        raise ValueError("Empty crop")

    return left, top, right, bottom


def extract_leaf_crop(
    image: Image.Image,
    bbox_xyxy: List[float],
    margin_ratio: float = 0.08,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Extract a cropped leaf patch from an image with orientation normalization."""
    normalized_img = normalize_exif_orientation(image)
    w, h = normalized_img.size

    left, top, right, bottom = crop_bounding_box(w, h, bbox_xyxy, margin_ratio=margin_ratio)
    cropped_patch = normalized_img.crop((left, top, right, bottom))
    return cropped_patch, (left, top, right, bottom)

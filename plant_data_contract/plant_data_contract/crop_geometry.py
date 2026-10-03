"""Bounding box margin expansion, clamping, EXIF normalization, and cropping."""

import math
from typing import List, Tuple
from PIL import Image, ImageOps


def normalize_exif_orientation(image: Image.Image) -> Image.Image:
    """Transpose image according to EXIF orientation tag and strip orientation."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


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
    xmin, ymin, xmax, ymax = bbox_xyxy
    box_w = max(0.0, xmax - xmin)
    box_h = max(0.0, ymax - ymin)

    dx = box_w * margin_ratio
    dy = box_h * margin_ratio

    left = max(0, int(math.floor(xmin - dx)))
    top = max(0, int(math.floor(ymin - dy)))
    right = min(image_width, int(math.ceil(xmax + dx)))
    bottom = min(image_height, int(math.ceil(ymax + dy)))

    # Ensure non-empty crop
    if right <= left:
        right = min(image_width, left + 1)
        if right <= left and left > 0:
            left = right - 1
    if bottom <= top:
        bottom = min(image_height, top + 1)
        if bottom <= top and top > 0:
            top = bottom - 1

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

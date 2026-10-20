"""Tăng cường ảnh mạnh cho lần train sau: kích thước, chuẩn hóa, tái lập, và suy luận không đổi."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.data.robust_transforms import RandomJpeg, RandomLowResolution, get_eval_transform, get_robust_train_transform, seed_everything


def _image(w: int = 300, h: int = 200) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_train_transform_output_shape_and_dtype():
    out = get_robust_train_transform()(_image())
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32 and torch.isfinite(out).all()


def test_train_transform_is_reproducible_with_seed_and_varies_without_it():
    t = get_robust_train_transform()
    seed_everything(5)
    a = t(_image())
    seed_everything(5)
    b = t(_image())
    assert torch.equal(a, b)
    c = t(_image())
    assert not torch.equal(a, c)


def test_eval_transform_matches_canonical_v1_geometry():
    out = get_eval_transform()(_image(97, 311))
    assert out.shape == (3, 224, 224)
    flat = get_eval_transform()(Image.new("RGB", (50, 80), (124, 116, 104)))
    assert float(flat.abs().max()) < 0.1  # màu trung bình ImageNet -> gần 0 sau chuẩn hóa


def test_jpeg_and_low_resolution_keep_image_size_and_mode():
    torch.manual_seed(1)
    img = _image()
    assert RandomJpeg(p=1.0)(img).size == img.size and RandomJpeg(p=1.0)(img).mode == "RGB"
    assert RandomLowResolution(p=1.0)(img).size == img.size
    assert RandomJpeg(p=0.0)(img) is img and RandomLowResolution(p=0.0)(img) is img

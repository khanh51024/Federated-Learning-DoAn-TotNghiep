"""src/training/gpu_augment.py và phần CPU của công thức gpu_v2 (chạy trên CPU; trên máy GPU cùng mã chạy với tensor CUDA)."""
from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from src.data.robust_transforms import IMAGENET_MEAN, IMAGENET_STD, get_cpu_train_transform
from src.training.gpu_augment import GpuAugment

OFF = dict(hflip=0, vflip=0, rotate_p=0, brightness=0, contrast=0, saturation=0, hue=0, blur_p=0, erase_p=0)


def batch(n: int = 8, size: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, size, size), generator=g, dtype=torch.uint8)


def gen(seed: int = 1) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_output_is_a_normalized_float_batch_of_the_same_shape():
    out = GpuAugment(gen())(batch())
    assert out.shape == (8, 3, 32, 32) and out.dtype == torch.float32
    low = min((0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD))
    high = max((1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD))
    assert float(out.min()) >= low - 1e-4 and float(out.max()) <= high + 1e-4


def test_with_every_probability_zero_only_normalization_remains():
    x = batch()
    out = GpuAugment(gen(), **OFF)(x)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(out, (x.float() / 255 - mean) / std, atol=1e-6)


def test_same_generator_seed_reproduces_the_batch_and_other_seeds_differ():
    x = batch()
    a, b, c = GpuAugment(gen(5))(x), GpuAugment(gen(5))(x), GpuAugment(gen(6))(x)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_each_image_gets_its_own_random_parameters():
    x = batch(1).repeat(16, 1, 1, 1)                          # 16 bản của cùng một ảnh
    out = GpuAugment(gen(), hflip=0, vflip=0, rotate_p=0, blur_p=0, erase_p=0)(x)
    distinct = {tuple(out[i].flatten()[:50].tolist()) for i in range(16)}
    assert len(distinct) > 8                                    # màu ngẫu nhiên theo từng ảnh, không dùng chung một tham số cho cả batch


def test_horizontal_flip_with_probability_one_mirrors_every_image():
    x = batch()
    out = GpuAugment(gen(), **{**OFF, "hflip": 1.0})(x)
    expected = GpuAugment(gen(), **OFF)(x.flip(3))
    assert torch.allclose(out, expected, atol=1e-6)


def test_brightness_factor_stays_within_the_configured_range():
    x = torch.full((64, 3, 8, 8), 100, dtype=torch.uint8)
    out = GpuAugment(gen(), **{**OFF, "brightness": 0.35})(x)
    raw = out * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    factor = raw[:, 0, 0, 0] / (100 / 255)
    assert float(factor.min()) >= 0.65 - 1e-4 and float(factor.max()) <= 1.35 + 1e-4
    assert float(factor.max() - factor.min()) > 0.3


def test_hue_rotation_keeps_gray_pixels_gray():
    x = torch.full((4, 3, 8, 8), 128, dtype=torch.uint8)       # xám: không có sắc độ để xoay
    out = GpuAugment(gen(), **{**OFF, "hue": 0.04})(x)
    raw = out * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    assert torch.allclose(raw, torch.full_like(raw, 128 / 255), atol=1e-4)


def test_blur_with_probability_one_keeps_mean_and_smooths():
    x = batch(4, 32)
    out = GpuAugment(gen(), **{**OFF, "blur_p": 1.0, "blur_sigma": (1.5, 1.5)})(x)
    plain = GpuAugment(gen(), **OFF)(x)
    assert torch.allclose(out.mean((2, 3)), plain.mean((2, 3)), atol=0.05)
    assert float(out.diff(dim=3).abs().mean()) < float(plain.diff(dim=3).abs().mean())


def test_rotation_by_probability_one_moves_pixels_and_blackens_corners():
    x = torch.full((2, 3, 32, 32), 200, dtype=torch.uint8)
    aug = GpuAugment(gen(), **{**OFF, "rotate_p": 1.0, "degrees": 25.0})
    out = aug(x)
    raw = out * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    corners = raw[:, :, [0, 0, -1, -1], [0, -1, 0, -1]]
    assert float(corners.min()) < 0.5                            # góc ngoài ảnh xoay bị tô đen như RandomRotation(fill=0)


def test_erasing_area_stays_within_two_to_fifteen_percent():
    x = torch.full((200, 3, 40, 40), 255, dtype=torch.uint8)
    out = GpuAugment(gen(), **{**OFF, "erase_p": 1.0})(x)
    erased = (out[:, 0] == 0).float().mean((1, 2))
    assert float(erased.min()) > 0.0
    assert float(erased.max()) <= 0.16
    assert float(erased.mean()) == pytest.approx(0.085, abs=0.03)


def test_cpu_part_returns_uint8_crops_of_the_training_size():
    image = Image.new("RGB", (300, 200), (30, 140, 40))
    torch.manual_seed(0)
    for trivial in (False, True):
        out = get_cpu_train_transform(192, trivial_augment=trivial)(image)
        assert out.dtype == torch.uint8 and tuple(out.shape) == (3, 192, 192)


def test_rgb_to_yiq_matrices_are_inverse():
    from src.training.gpu_augment import _RGB2YIQ, _YIQ2RGB

    assert torch.allclose(_YIQ2RGB @ _RGB2YIQ, torch.eye(3), atol=1e-5)
    assert math.isclose(float(_RGB2YIQ[0].sum()), 1.0, abs_tol=1e-6)

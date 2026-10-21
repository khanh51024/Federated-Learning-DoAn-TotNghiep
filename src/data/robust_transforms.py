"""Tăng cường ảnh mạnh khi train (`robust_v1`: toàn bộ trên CPU; `gpu_v2`: phần CPU ở `get_cpu_train_transform`, phần còn lại ở src/training/gpu_augment.py).

Lý do: các checkpoint đã công bố chỉ thấy RandomHorizontalFlip trên ảnh lá đã cắt gọn (`canonical_v1`), nên gặp ảnh ngoài đồng
(nghiêng, nền rối, ánh sáng lạ, ảnh nén) thì sai nhiều. Bộ biến đổi này mô phỏng các khác biệt đó nhưng GIỮ NGUYÊN phần chuẩn hóa
cuối và kích thước 224 của `canonical_v1`, nên suy luận không đổi (chỉ phần học đổi).

Mọi phép ngẫu nhiên lấy từ bộ sinh của torch: đặt `torch.manual_seed` trước khi gọi để tái lập.
"""
from __future__ import annotations

import io
import random
from typing import Any

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


class RandomJpeg:
    """Nén JPEG với chất lượng ngẫu nhiên (ảnh tải từ mạng thường bị nén nặng và nhỏ)."""

    def __init__(self, quality: tuple[int, int] = (35, 95), p: float = 0.5):
        self.quality, self.p = quality, p

    def __call__(self, image: Any) -> Any:
        import torch
        from PIL import Image

        if float(torch.rand(1)) >= self.p:
            return image
        q = int(torch.randint(self.quality[0], self.quality[1] + 1, (1,)))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomLowResolution:
    """Thu nhỏ rồi phóng lại (ảnh thumbnail 200-300 px phóng to)."""

    def __init__(self, min_side: int = 96, p: float = 0.3):
        self.min_side, self.p = min_side, p

    def __call__(self, image: Any) -> Any:
        import torch
        from PIL import Image

        if float(torch.rand(1)) >= self.p:
            return image
        w, h = image.size
        side = int(torch.randint(self.min_side, max(self.min_side + 1, min(w, h)), (1,)))
        small = image.resize((max(1, round(w * side / min(w, h))), max(1, round(h * side / min(w, h)))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def get_robust_train_transform(image_size: int = IMAGE_SIZE, scale: tuple[float, float] = (0.35, 1.0)) -> Any:
    """Biến đổi học: cắt-phóng ngẫu nhiên, xoay, lật, đổi màu/ánh sáng, mờ, nén JPEG, độ phân giải thấp, xóa mảng.

    Đầu ra: tensor float [3, image_size, image_size] đã chuẩn hóa ImageNet, như `canonical_v1`."""
    from torchvision import transforms as T

    return T.Compose([
        RandomLowResolution(),
        RandomJpeg(),
        T.RandomResizedCrop((image_size, image_size), scale=scale, ratio=(3 / 4, 4 / 3), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.RandomApply([T.RandomRotation(25, interpolation=T.InterpolationMode.BILINEAR)], p=0.5),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.04),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.25),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])


def get_cpu_train_transform(image_size: int = IMAGE_SIZE, scale: tuple[float, float] = (0.35, 1.0), trivial_augment: bool = False) -> Any:
    """Phần CPU của công thức `gpu_v2`: cắt-phóng ngẫu nhiên TRƯỚC, rồi thu nhỏ và nén JPEG trên ảnh đã cắt (rẻ hơn 2–3 lần so với làm trên ảnh gốc).

    Đầu ra: tensor uint8 [3, image_size, image_size], CHƯA chuẩn hóa; lật, xoay, màu, mờ, chuẩn hóa, xóa mảng làm ở src/training/gpu_augment.py.
    `trivial_augment=True`: thêm TrivialAugmentWide (một phép ngẫu nhiên mỗi ảnh, không cần dò tham số) thay cho phần màu và xoay trên GPU."""
    from torchvision import transforms as T

    steps = [
        T.RandomResizedCrop((image_size, image_size), scale=scale, ratio=(3 / 4, 4 / 3), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        RandomLowResolution(min_side=max(48, image_size * 3 // 8)),
        RandomJpeg(),
    ]
    if trivial_augment:
        steps.append(T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose([*steps, T.PILToTensor()])


def get_eval_transform(image_size: int = IMAGE_SIZE) -> Any:
    """Giống hệt `canonical_v1` lúc suy luận: resize thẳng về vuông, chuẩn hóa ImageNet."""
    from torchvision import transforms as T

    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

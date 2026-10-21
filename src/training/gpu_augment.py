"""Phần tăng cường ảnh chạy trên GPU theo batch (công thức `gpu_v2`), thay cho phần tốn CPU của `robust_v1`.

Vì sao: đo trên ảnh thật của release, chuỗi `robust_v1` tốn khoảng 10 ms/ảnh trên một luồng CPU (ColorJitter 4,5 ms, GaussianBlur 2–2,6 ms),
nên CPU 8 lõi của máy thuê là nghẽn còn GPU chỉ dùng 14–30%. Ở đây CPU chỉ giải mã, cắt-phóng ngẫu nhiên, nén JPEG và thu nhỏ trên ảnh đã
cắt (`get_cpu_train_transform` trong src/data/robust_transforms.py); các phép còn lại chạy theo batch trên GPU.

Cùng xác suất và biên độ với `robust_v1`: lật ngang 0,5, lật dọc 0,2, xoay ±25° (p 0,5, góc ngoài ảnh tô đen), độ sáng/tương phản ±0,35,
bão hòa ±0,3, sắc độ ±0,04, mờ Gauss nhân 5 (sigma 0,1–1,6, p 0,25), chuẩn hóa ImageNet, xóa mảng (p 0,25, 2–15% diện tích, giá trị 0 sau chuẩn hóa).
Khác biệt nhỏ đã chấp nhận: bốn phép màu chạy theo thứ tự cố định thay vì ngẫu nhiên; sắc độ xoay trong không gian YIQ thay cho HSV.
Mỗi ảnh có tham số riêng, mọi số ngẫu nhiên lấy từ một `torch.Generator` nên tái lập được (lưu trạng thái vào checkpoint để chạy tiếp).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_LUMA = (0.299, 0.587, 0.114)
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


class GpuAugment:
    """Gọi `augment(batch_uint8)` -> tensor float đã chuẩn hóa, cùng thiết bị. `color=False, rotate=False` khi phần màu/xoay đã làm ở CPU (TrivialAugment)."""

    def __init__(self, generator: torch.Generator, *, hflip: float = 0.5, vflip: float = 0.2, rotate_p: float = 0.5, degrees: float = 25.0,
                 brightness: float = 0.35, contrast: float = 0.35, saturation: float = 0.3, hue: float = 0.04,
                 blur_p: float = 0.25, blur_sigma: tuple[float, float] = (0.1, 1.6), erase_p: float = 0.25,
                 erase_scale: tuple[float, float] = (0.02, 0.15), erase_ratio: tuple[float, float] = (0.3, 3.3),
                 color: bool = True, rotate: bool = True):
        self.g = generator
        self.hflip, self.vflip, self.rotate_p, self.degrees = hflip, vflip, rotate_p if rotate else 0.0, degrees
        self.brightness, self.contrast, self.saturation, self.hue = (brightness, contrast, saturation, hue) if color else (0.0, 0.0, 0.0, 0.0)
        self.blur_p, self.blur_sigma = blur_p, blur_sigma
        self.erase_p, self.erase_scale, self.erase_ratio = erase_p, erase_scale, erase_ratio

    def _rand(self, n: int, device: Any) -> torch.Tensor:
        return torch.rand(n, generator=self.g, device=device)

    def _uniform(self, n: int, low: float, high: float, device: Any) -> torch.Tensor:
        return low + (high - low) * self._rand(n, device)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        n, device = images.shape[0], images.device
        x = images.float() / 255.0
        x = torch.where((self._rand(n, device) < self.hflip).view(n, 1, 1, 1), x.flip(3), x)
        x = torch.where((self._rand(n, device) < self.vflip).view(n, 1, 1, 1), x.flip(2), x)
        x = self._rotate(x)
        x = self._color(x)
        x = self._blur(x)
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        return self._erase((x - mean) / std)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        n, device = x.shape[0], x.device
        apply = self._rand(n, device) < self.rotate_p
        angle = self._uniform(n, -self.degrees, self.degrees, device) * math.pi / 180 * apply
        cos, sin, zero = torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)
        theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        rotated = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return torch.where(apply.view(n, 1, 1, 1), rotated, x)

    def _color(self, x: torch.Tensor) -> torch.Tensor:
        n, device = x.shape[0], x.device
        luma = torch.tensor(_LUMA, device=device).view(1, 3, 1, 1)
        if self.brightness:
            x = (x * self._uniform(n, 1 - self.brightness, 1 + self.brightness, device).view(n, 1, 1, 1)).clamp(0, 1)
        if self.contrast:
            factor = self._uniform(n, 1 - self.contrast, 1 + self.contrast, device).view(n, 1, 1, 1)
            gray_mean = (x * luma).sum(1, keepdim=True).mean((2, 3), keepdim=True)
            x = (factor * x + (1 - factor) * gray_mean).clamp(0, 1)
        if self.saturation:
            factor = self._uniform(n, 1 - self.saturation, 1 + self.saturation, device).view(n, 1, 1, 1)
            gray = (x * luma).sum(1, keepdim=True)
            x = (factor * x + (1 - factor) * gray).clamp(0, 1)
        if self.hue:
            angle = self._uniform(n, -self.hue, self.hue, device) * 2 * math.pi
            cos, sin = torch.cos(angle), torch.sin(angle)
            rot = torch.zeros(n, 3, 3, device=device)
            rot[:, 0, 0] = 1
            rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
            m = _YIQ2RGB.to(device) @ rot @ _RGB2YIQ.to(device)          # [n, 3, 3]: RGB -> YIQ, xoay I/Q, -> RGB
            x = torch.einsum("nij,njhw->nihw", m, x).clamp(0, 1)
        return x

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        device = x.device
        apply = self._rand(n, device) < self.blur_p
        sigma = self._uniform(n, *self.blur_sigma, device)
        offsets = torch.arange(-2, 3, device=device, dtype=x.dtype)
        kernel = torch.exp(-(offsets.view(1, 5) ** 2) / (2 * sigma.view(n, 1) ** 2))
        kernel = (kernel / kernel.sum(1, keepdim=True)).repeat_interleave(c, 0)       # [n*c, 5]
        flat = F.pad(x.reshape(1, n * c, h, w), (2, 2, 2, 2), mode="reflect")
        flat = F.conv2d(flat, kernel.view(n * c, 1, 1, 5), groups=n * c)
        flat = F.conv2d(flat, kernel.view(n * c, 1, 5, 1), groups=n * c)
        return torch.where(apply.view(n, 1, 1, 1), flat.view(n, c, h, w), x)

    def _erase(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        device = x.device
        apply = self._rand(n, device) < self.erase_p
        area = self._uniform(n, *self.erase_scale, device) * h * w
        log_ratio = self._uniform(n, math.log(self.erase_ratio[0]), math.log(self.erase_ratio[1]), device)
        eh = torch.sqrt(area * torch.exp(log_ratio)).round().clamp(1, h)
        ew = torch.sqrt(area / torch.exp(log_ratio)).round().clamp(1, w)
        top = ((h - eh + 1) * self._rand(n, device)).floor()
        left = ((w - ew + 1) * self._rand(n, device)).floor()
        rows = torch.arange(h, device=device).view(1, h, 1)
        cols = torch.arange(w, device=device).view(1, 1, w)
        mask = ((rows >= top.view(n, 1, 1)) & (rows < (top + eh).view(n, 1, 1)) & (cols >= left.view(n, 1, 1)) & (cols < (left + ew).view(n, 1, 1)))
        mask = (mask & apply.view(n, 1, 1)).unsqueeze(1)
        return x.masked_fill(mask, 0.0)

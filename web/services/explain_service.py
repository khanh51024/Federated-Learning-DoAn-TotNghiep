"""Grad-CAM cho MobileNetV3-Small: vùng ảnh mà mô hình dựa vào để chọn một lớp.

- Lớp đích: `model.features[-1]` (Conv 1×1 96→576 + BN + Hardswish), activation 576×7×7.
- Trọng số mỗi kênh = trung bình gradient của logit lớp đích theo không gian; CAM = ReLU(tổng có trọng số), chuẩn hóa 0–1.
- Đây là GỢI Ý TRỰC QUAN, không phải bằng chứng nhân quả: độ phân giải gốc chỉ 7×7.
- Không cần thư viện mới (torch, numpy, PIL).
- KHÔNG gắn hook lên mô hình: mô hình được cache và dùng chung giữa các luồng (predict chạy song song). Tính lại forward
  theo đúng thứ tự của torchvision `MobileNetV3.forward` (features → avgpool → flatten → classifier) và lấy gradient bằng
  `torch.autograd.grad`, nên không có trạng thái chung nào bị sửa và gọi đồng thời vẫn an toàn.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

SIZE = 224


def gradcam(model: Any, tensor: Any, class_index: int) -> np.ndarray:
    """Bản đồ Grad-CAM 7×7 (float, 0–1) của `class_index` cho ảnh `tensor` (1×3×224×224)."""
    import torch

    with torch.enable_grad():  # chế độ grad là cục bộ theo luồng: không ảnh hưởng inference_mode của luồng khác
        x = tensor.detach().clone().requires_grad_(True)  # trọng số đã khóa grad: cần input để có đồ thị
        features = model.features(x)  # đầu ra của features[-1]: 1×576×7×7
        logits = model.classifier(torch.flatten(model.avgpool(features), 1))
        (gradient,) = torch.autograd.grad(logits[0, int(class_index)], features)
    activation = features[0].detach()
    gradient = gradient[0].detach()
    weights = gradient.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
    peak = float(cam.max())
    cam = cam / peak if peak > 0 else cam
    return cam.numpy().astype(np.float32)


def _colormap(values: np.ndarray) -> np.ndarray:
    """Thang màu xanh → vàng → đỏ (0–1 -> RGB uint8) không cần matplotlib."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 * v - 0.25, 0, 1)
    g = np.clip(1.6 - np.abs(2.2 * v - 1.25) * 1.6, 0, 1)
    b = np.clip(1.0 - 2.0 * v, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: Any, cam: np.ndarray, alpha: float = 0.45) -> Any:
    """Ảnh đầu vào đã resize 224×224 (đúng thứ mô hình thấy) phủ bản đồ nhiệt."""
    from PIL import Image

    from src.common.image_io import to_rgb

    base = to_rgb(image).resize((SIZE, SIZE), Image.BILINEAR)
    heat = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L").resize((SIZE, SIZE), Image.BILINEAR)
    colored = _colormap(np.asarray(heat, dtype=np.float32) / 255.0)
    weight = (np.asarray(heat, dtype=np.float32) / 255.0)[..., None] * alpha
    mixed = np.asarray(base, dtype=np.float32) * (1 - weight) + colored.astype(np.float32) * weight
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def overlay_data_url(image: Any, cam: np.ndarray, quality: int = 80) -> str:
    buffer = io.BytesIO()
    overlay(image, cam).save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

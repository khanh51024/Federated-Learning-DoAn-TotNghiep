"""
src.model.mobilenetv3: Kiến trúc MobileNetV3-Small chuẩn hóa cho PlantVillage (38 lớp).
Hỗ trợ cả chế độ Pretrained ImageNet (Giai đoạn 1) và Scratch (Giai đoạn 2).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def create_model(num_classes: int = 38, pretrained: bool = False) -> nn.Module:
    """
    Khởi tạo MobileNetV3-Small:
    - pretrained=True: nạp weights ImageNet-1K V1 rồi thay thế classifier cuối bằng 38 lớp.
    - pretrained=False: khởi tạo ngẫu nhiên toàn bộ mạng (Scratch).
    """
    if pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress=False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        model = mobilenet_v3_small(weights=None, num_classes=num_classes)
    return model


def get_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Trích xuất state_dict của mô hình ra CPU (bản sao độc lập)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_parameters(
    model: nn.Module,
    parameters: Dict[str, torch.Tensor | np.ndarray],
    strict: bool = True,
) -> None:
    """
    Nạp parameters/buffers vào model.
    Tự động chuyển đổi numpy sang torch tensor và đảm bảo đúng kiểu dữ liệu (dtype).
    """
    current_state = model.state_dict()
    converted: Dict[str, torch.Tensor] = {}

    for k, v in parameters.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            t = v.clone()
        else:
            raise TypeError(f"Tensor '{k}' có kiểu không hợp lệ: {type(v)}")

        if k in current_state:
            target_dtype = current_state[k].dtype
            if t.dtype != target_dtype:
                t = t.to(dtype=target_dtype)

        converted[k] = t

    model.load_state_dict(converted, strict=strict)


def check_parameters_finite(parameters: Dict[str, Any]) -> None:
    """Kiểm tra không có NaN hoặc Inf trong toàn bộ tensor dấu phẩy động."""
    for k, v in parameters.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"Phát hiện giá trị non-finite (NaN/Inf) trong tham số '{k}'")
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            if not np.isfinite(v).all():
                raise FloatingPointError(f"Phát hiện giá trị non-finite (NaN/Inf) trong numpy parameter '{k}'")


def compute_model_hash(parameters: Dict[str, torch.Tensor | np.ndarray]) -> str:
    """
    Tính mã băm SHA-256 tất định của bộ trọng số mô hình.
    Dùng kiểm chứng toàn vẹn truyền nhận: CLIENT_HASH == SERVER_HASH.
    """
    hasher = hashlib.sha256()
    for k in sorted(parameters.keys()):
        hasher.update(k.encode("utf-8"))
        v = parameters[k]
        if isinstance(v, torch.Tensor):
            hasher.update(v.detach().cpu().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            hasher.update(v.tobytes())
    return hasher.hexdigest()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Trả về (tổng số tham số, tổng dung lượng byte)."""
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_params, total_bytes

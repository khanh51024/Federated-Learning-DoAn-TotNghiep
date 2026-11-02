"""Nén bản cập nhật client gửi lên máy chủ ở MỖI vòng (N4.2 của yêu cầu đề tài). Chỉ nén chiều lên; chiều xuống máy chủ gửi nguyên trọng số.

Chỉ tensor số thực từ 2 chiều trở lên (trọng số conv, depthwise, tuyến tính) bị nén; bias, tham số và thống kê BatchNorm, bộ đếm giữ nguyên, vì chúng chiếm rất ít dung lượng
và lượng tử hóa `running_var` làm hỏng mô hình (docs/GD4_NEN_LUONG_TU_HOA_2026-10-26.md mục 5).

  fp16        gửi trọng số dạng float16
  delta_int8  gửi độ lệch w_k − w_chung dạng int8 đối xứng, một hệ số tỉ lệ cho mỗi kênh đầu ra

`encode_update` tạo đúng các tensor sẽ nằm trong tin nhắn, nên số byte đo được là số byte thật; `decode_update` là việc của máy chủ.
"""
from __future__ import annotations

from typing import Any

MODES = ("fp16", "delta_int8")
_SCALE_SUFFIX = "::scale"


def is_quantizable(tensor: Any) -> bool:
    """Trọng số conv/depthwise/tuyến tính: số thực từ 2 chiều trở lên."""
    return tensor.is_floating_point() and tensor.ndim >= 2


def quantize_int8(tensor: Any, per_channel: bool) -> tuple[Any, Any]:
    """Lượng tử hóa đối xứng int8. Trả (giá trị nguyên dạng phẳng theo kênh, hệ số tỉ lệ từng kênh)."""
    import torch

    flat = tensor.reshape(tensor.shape[0], -1) if per_channel else tensor.reshape(1, -1)
    scale = (flat.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    return torch.round(flat / scale[:, None]).clamp(-127, 127), scale


def fake_quant_int8(tensor: Any, per_channel: bool) -> tuple[Any, int]:
    """Lượng tử hóa rồi giải lượng tử. Trả (tensor sau khi dựng lại, số hệ số tỉ lệ phải gửi kèm)."""
    quantized, scale = quantize_int8(tensor, per_channel)
    return (quantized * scale[:, None]).reshape(tensor.shape), int(scale.numel())


def encode_update(state: dict[str, Any], parent: dict[str, Any], mode: str) -> dict[str, Any]:
    import torch

    if mode not in MODES:
        raise ValueError(f"cách nén không có: {mode}")
    payload: dict[str, Any] = {}
    for key, tensor in state.items():
        if not is_quantizable(tensor):
            payload[key] = tensor
        elif mode == "fp16":
            payload[key] = tensor.half()
        else:
            quantized, scale = quantize_int8(tensor - parent[key].to(tensor.dtype), per_channel=True)
            payload[key] = quantized.to(dtype=torch.int8).reshape(tensor.shape)
            payload[key + _SCALE_SUFFIX] = scale.float()
    return payload


def decode_update(payload: dict[str, Any], parent: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"cách nén không có: {mode}")
    state: dict[str, Any] = {}
    for key, reference in parent.items():
        received = payload[key]
        if not is_quantizable(reference):
            state[key] = received
        elif mode == "fp16":
            state[key] = received.to(reference.dtype)
        else:
            scale = payload[key + _SCALE_SUFFIX].to(reference.dtype)
            flat = received.reshape(received.shape[0], -1).to(reference.dtype) * scale[:, None]
            state[key] = reference + flat.reshape(reference.shape)
    return state

"""
src.federated.faulty_client: Giả lập các hành vi client bất thường / client tấn công (Faulty / Adversarial Clients).
Mục đích: Đo lường định lượng mức độ suy giảm chất lượng của FedAvg trước các loại update dị thường.
"""

from __future__ import annotations

from typing import Dict
import torch


TAMPER_AFTER_HASH = "tamper_after_hash"


def tamper_after_hash(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Đổi một con số SAU KHI client đã băm, giả lập trọng số bị đổi trên đường truyền. Máy chủ băm lại phải thấy lệch."""
    changed = {key: value.clone() for key, value in weights.items()}
    key = next(name for name in sorted(changed) if changed[name].is_floating_point() and changed[name].numel() > 0)
    changed[key].view(-1)[0] += 1e-3
    return changed


def apply_fault_mode(
    mode: str,
    trained_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
    scale_factor: float = 10.0,
    noise_std: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Biến đổi trọng số trả về theo chế độ giả lập:
    - 'normal': giữ nguyên trọng số sau huấn luyện.
    - 'no_train': không huấn luyện, gửi trả nguyên vẹn trọng số toàn cục cũ w_global.
    - 'random_weights': gửi trọng số ngẫu nhiên Gaussian.
    - 'scaled_update': phóng đại độ lệch update: w_malicious = w_global + scale * (w_k - w_global).
    - 'noise_update': thêm nhiễu Gaussian N(0, noise_std^2) vào weights.
    """
    if mode in ("normal", TAMPER_AFTER_HASH):      # TAMPER_AFTER_HASH đổi sau khi băm: xem tamper_after_hash, client_app gọi đúng chỗ
        return trained_weights

    corrupted: Dict[str, torch.Tensor] = {}

    if mode == "no_train":
        for k, v in global_weights.items():
            corrupted[k] = v.clone()
        return corrupted

    if mode == "random_weights":
        for k, v in trained_weights.items():
            if v.is_floating_point():
                corrupted[k] = torch.randn_like(v)
            else:
                corrupted[k] = v.clone()
        return corrupted

    if mode == "scaled_update":
        for k, v in trained_weights.items():
            if v.is_floating_point() and k in global_weights:
                delta = v - global_weights[k]
                corrupted[k] = global_weights[k] + (delta * scale_factor)
            else:
                corrupted[k] = v.clone()
        return corrupted

    if mode == "noise_update":
        for k, v in trained_weights.items():
            if v.is_floating_point():
                noise = torch.randn_like(v) * noise_std
                corrupted[k] = v + noise
            else:
                corrupted[k] = v.clone()
        return corrupted

    raise ValueError(f"Chế độ fault không được hỗ trợ: {mode}")

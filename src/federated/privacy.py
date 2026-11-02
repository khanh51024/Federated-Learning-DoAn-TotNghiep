"""Differential privacy phía máy chủ theo cấp client (DP-FedAvg, McMahan và cộng sự, ICLR 2018) cho N4.3 của yêu cầu đề tài.

Mỗi vòng: cắt chuẩn L2 của độ lệch tham số học được từng client về `clip`, gộp như FedAvg, rồi cộng nhiễu Gaussian độ lệch chuẩn z · clip · max(n_k / n).
Một client thay đổi làm kết quả gộp đổi tối đa clip · n_k / n, nên đây đúng là cơ chế Gaussian với hệ số nhiễu z. Không dùng DP-SGD của Opacus vì nó không hợp với BatchNorm của MobileNetV3.

Giới hạn phải nói khi báo cáo:
  * Bảo vệ ở cấp CLIENT (có hay không cả một cơ sở), không phải cấp từng ảnh.
  * Máy chủ được tin cậy để cắt và thêm nhiễu; máy chủ vẫn thấy từng bản cập nhật.
  * Thống kê BatchNorm (running_mean, running_var) vẫn gộp không nhiễu: thêm nhiễu làm phương sai âm và hỏng mô hình. Vì vậy ε chỉ tính cho tham số học được, chưa phải bảo đảm trọn vẹn cho mọi thứ gửi đi.
  * Chỉ 5 client và tham gia đầy đủ (không có khuếch đại do lấy mẫu), nên ε nhỏ đòi hỏi nhiễu lớn so với tín hiệu.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def rdp_epsilon(noise_multiplier: float, rounds: int, delta: float) -> float:
    """ε của `rounds` lần cơ chế Gaussian với hệ số nhiễu z, tham gia đầy đủ: RDP bậc α là rounds·α/(2z²), đổi sang (ε, δ) bằng ε = RDP(α) + log(1/δ)/(α − 1), lấy α tốt nhất."""
    if noise_multiplier <= 0:
        return math.inf
    if not 0 < delta < 1:
        raise ValueError(f"delta phải trong (0, 1): {delta}")
    orders = np.concatenate([np.linspace(1.01, 20, 2000), np.geomspace(20, 100000, 4000)])
    return float(np.min(rounds * orders / (2 * noise_multiplier ** 2) + math.log(1 / delta) / (orders - 1)))


def clip_update(weights: dict[str, Any], parent: dict[str, Any], trainable: set[str], clip: float) -> tuple[dict[str, Any], float]:
    """Trả (trọng số đã cắt, chuẩn L2 gốc của độ lệch trên tham số học được). Thành phần khác giữ nguyên."""
    square = 0.0
    for name in trainable:
        diff = weights[name].double() - parent[name].double()
        square += float((diff * diff).sum())
    norm = math.sqrt(square)
    if norm <= clip:
        return dict(weights), norm
    scale = clip / norm
    clipped = dict(weights)
    for name in trainable:
        clipped[name] = (parent[name].double() + scale * (weights[name].double() - parent[name].double())).to(parent[name].dtype)
    return clipped, norm


def add_noise(merged: dict[str, Any], trainable: set[str], std: float, seed: int) -> dict[str, Any]:
    """Cộng nhiễu Gaussian vào tham số học được; `seed` theo từng vòng để chạy tiếp sau khi bị ngắt vẫn cùng nhiễu."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    noisy = dict(merged)
    for name in sorted(trainable):
        noisy[name] = merged[name] + torch.normal(0.0, std, size=merged[name].shape, generator=generator, dtype=merged[name].dtype)
    return noisy


def noise_std(noise_multiplier: float, clip: float, num_examples: list[int]) -> float:
    """z · clip · max(n_k / n): độ nhạy của phép gộp có trọng số theo số ảnh."""
    return noise_multiplier * clip * max(num_examples) / sum(num_examples)

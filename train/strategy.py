"""
train.strategy: Triển khai thuật toán Federated Averaging (FedAvg) chuẩn hóa.
Tính toán tổng hợp có trọng số theo số mẫu n_k:
    w_(t+1) = sum(n_k * w_k) / sum(n_k)
Bao gồm kiểm tra chặt chẽ tính toàn vẹn (shapes, dtype, non-finite).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .model import check_parameters_finite


def fedavg_weighted_average(
    weights_list: List[np.ndarray | torch.Tensor | float],
    num_examples_list: List[int],
) -> np.ndarray | float:
    """
    Tính trung bình có trọng số cho một mảng số học đơn giản (phục vụ unit test):
        result = sum(n_k * w_k) / sum(n_k)
    """
    if len(weights_list) != len(num_examples_list):
        raise ValueError("Độ dài weights_list và num_examples_list phải bằng nhau")
    if not weights_list:
        raise ValueError("Danh sách weights không được rỗng")

    total_examples = sum(num_examples_list)
    if total_examples <= 0:
        raise ValueError("Tổng số mẫu n_k phải lớn hơn 0")

    is_scalar = isinstance(weights_list[0], (int, float, np.number))

    if is_scalar:
        weighted_sum = sum(w * n for w, n in zip(weights_list, num_examples_list))
        return float(weighted_sum / total_examples)

    # Tensor hoặc ndarray
    first = np.asarray(weights_list[0], dtype=np.float64)
    weighted_acc = np.zeros_like(first)

    for w, n in zip(weights_list, num_examples_list):
        arr = np.asarray(w, dtype=np.float64)
        if arr.shape != first.shape:
            raise ValueError(f"Shape tensor không đồng nhất: {arr.shape} != {first.shape}")
        if not np.isfinite(arr).all():
            raise FloatingPointError("Phát hiện NaN hoặc Inf trong tensor cần tổng hợp")
        weighted_acc += arr * n

    return weighted_acc / total_examples


def aggregate_fedavg(
    client_updates: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Tổng hợp các bản cập nhật từ K clients bằng thuật toán FedAvg.

    Mỗi update trong client_updates gồm:
      - "weights": Dict[str, torch.Tensor | np.ndarray]
      - "num_examples": int (n_k > 0)
      - "client_id": int

    Quy tắc BatchNorm:
      - Floating-point tensors (weight, bias, running_mean, running_var): tổng hợp trung bình có trọng số.
      - Integer buffers (num_batches_tracked): lấy giá trị lớn nhất (max) qua các client.
    """
    if not client_updates:
        raise ValueError("client_updates rỗng, không thể tổng hợp")

    # Xác thực n_k và kiểm tra cơ bản
    total_samples = 0
    for idx, update in enumerate(client_updates):
        n_k = update.get("num_examples")
        if not isinstance(n_k, (int, np.integer)) or int(n_k) <= 0:
            raise ValueError(f"Client tại vị trí {idx} có num_examples không hợp lệ: {n_k}")
        total_samples += int(n_k)

        weights = update.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Client tại vị trí {idx} thiếu weights hợp lệ")
        check_parameters_finite(weights)

    # Lấy reference keys từ client đầu tiên
    first_weights = client_updates[0]["weights"]
    param_keys = sorted(first_weights.keys())

    # Kiểm tra tính khớp nhau về danh sách keys và shapes
    for update in client_updates[1:]:
        curr_weights = update["weights"]
        if sorted(curr_weights.keys()) != param_keys:
            raise ValueError("Danh sách layers không đồng nhất giữa các client updates")
        for key in param_keys:
            t1 = first_weights[key]
            t2 = curr_weights[key]
            shape1 = tuple(t1.shape)
            shape2 = tuple(t2.shape)
            if shape1 != shape2:
                raise ValueError(f"Lớp '{key}' bị sai lệch shape: {shape1} vs {shape2}")

    aggregated: Dict[str, torch.Tensor] = {}

    for key in param_keys:
        sample_tensor = first_weights[key]
        is_torch = isinstance(sample_tensor, torch.Tensor)
        original_dtype = sample_tensor.dtype if is_torch else torch.float32

        # Trường hợp buffer số nguyên: num_batches_tracked
        if "num_batches_tracked" in key:
            max_val = 0
            for update in client_updates:
                val = update["weights"][key]
                val_int = int(val.item() if hasattr(val, "item") else val)
                if val_int > max_val:
                    max_val = val_int
            aggregated[key] = torch.tensor(max_val, dtype=torch.int64)
            continue

        # Các tensor số thực: tính tổng n_k * w_k bằng float64 để chống tràn/sai số
        first_np = sample_tensor.detach().cpu().numpy() if is_torch else sample_tensor
        acc = np.zeros(first_np.shape, dtype=np.float64)

        for update in client_updates:
            n_k = update["num_examples"]
            t = update["weights"][key]
            t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
            acc += (t_np.astype(np.float64) * n_k)

        acc = acc / total_samples
        aggregated[key] = torch.from_numpy(acc).to(dtype=original_dtype)

    return aggregated

"""
src.federated.weight_inspector: Phân tích chi tiết thống kê trọng số và chuẩn cập nhật (Update Norm).
    Delta_k = w_k - w_global
    UpdateNorm = ||Delta_k||_2
    L2 Norm = ||w_k||_2
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch


def inspect_layer_weights(
    tensor: torch.Tensor | np.ndarray,
    reference_tensor: Optional[torch.Tensor | np.ndarray] = None,
    layer_name: str = "",
) -> Dict[str, Any]:
    """
    Tính toán thống kê chi tiết cho 1 tensor trọng số:
    - shape, parameter count, bytes
    - mean, std, min, max
    - L2 norm
    - Update norm (nếu có reference_tensor w_global)
    - sample_values: 5 phần tử đầu tiên
    """
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    arr_flat = arr.flatten().astype(np.float64)

    param_count = int(arr.size)
    param_bytes = int(arr.nbytes)
    l2_norm = float(np.linalg.norm(arr_flat))
    mean_val = float(np.mean(arr_flat)) if param_count > 0 else 0.0
    std_val = float(np.std(arr_flat)) if param_count > 0 else 0.0
    min_val = float(np.min(arr_flat)) if param_count > 0 else 0.0
    max_val = float(np.max(arr_flat)) if param_count > 0 else 0.0

    update_norm: Optional[float] = None
    if reference_tensor is not None:
        ref_arr = reference_tensor.detach().cpu().numpy() if isinstance(reference_tensor, torch.Tensor) else np.asarray(reference_tensor)
        delta = arr_flat - ref_arr.flatten().astype(np.float64)
        update_norm = round(float(np.linalg.norm(delta)), 4)

    sample_values = [round(float(x), 4) for x in arr_flat[:5]]

    return {
        "layer_name": layer_name,
        "shape": list(arr.shape),
        "param_count": param_count,
        "param_bytes": param_bytes,
        "mean": round(mean_val, 5),
        "std": round(std_val, 5),
        "min": round(min_val, 5),
        "max": round(max_val, 5),
        "l2_norm": round(l2_norm, 4),
        "update_norm": update_norm,
        "sample_values": sample_values,
    }


def inspect_model_state(
    current_weights: Dict[str, torch.Tensor | np.ndarray],
    global_weights: Optional[Dict[str, torch.Tensor | np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """Phân tích thống kê cho toàn bộ các layer trong model."""
    results: List[Dict[str, Any]] = []
    for key in sorted(current_weights.keys()):
        curr_t = current_weights[key]
        ref_t = global_weights.get(key) if global_weights else None
        stats = inspect_layer_weights(curr_t, ref_t, layer_name=key)
        results.append(stats)
    return results


def export_weight_stats_csv(stats: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Xuất danh sách thống kê trọng số ra file CSV phục vụ báo cáo hoặc dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stats:
        return

    fields = [
        "layer_name", "shape", "param_count", "param_bytes",
        "mean", "std", "min", "max", "l2_norm", "update_norm",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in stats:
            row_copy = dict(row)
            row_copy["shape"] = str(row_copy["shape"])
            writer.writerow(row_copy)

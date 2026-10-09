"""
src.common.metrics: Các hàm tính toán chỉ số thống kê & hiệu năng chuẩn hóa.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def compute_macro_f1(confusion_matrix: np.ndarray) -> float:
    """Tính Macro-F1 từ Confusion Matrix."""
    num_classes = confusion_matrix.shape[0]
    f1_scores = []
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))

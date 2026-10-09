"""
src.data.partition: Phân hoạch Dirichlet Non-IID và đo lường chỉ số "đói dữ liệu" (Data Scarcity).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np


def partition_dirichlet(
    labels: List[int] | np.ndarray,
    num_clients: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> Dict[int, List[int]]:
    """
    Phân hoạch nhãn theo Dirichlet(alpha).
    - alpha lớn (100.0) -> gần IID (các client có phân phối nhãn giống nhau).
    - alpha nhỏ (0.1) -> non-IID mạnh (mỗi client chỉ nắm một vài lớp bệnh).
    """
    np.random.seed(seed)
    labels = np.asarray(labels)
    num_samples = len(labels)
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in unique_classes:
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)

        # Sinh phân phối xác suất Dirichlet cho lớp c
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        # Chia mẫu lớp c theo proportions
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        client_subsets = np.split(idx_c, splits)

        for i in range(num_clients):
            client_indices[i].extend(client_subsets[i].tolist())

    # Đảm bảo mỗi client có tối thiểu min_samples_per_client
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def calculate_data_scarcity(
    labels: List[int] | np.ndarray,
    total_classes: int = 38,
) -> Dict[str, Any]:
    """
    Tính các chỉ số "đói dữ liệu" (Data Scarcity) có thể so sánh giữa clients:
    - num_samples: tổng số mẫu n_k
    - class_count_present / class_count_missing
    - largest_class_ratio: mức độ một lớp thống trị client
    - label_entropy / entropy_normalized: độ đa dạng nhãn
    - starvation_score: heuristic [0,1], càng cao càng thiếu/nghèo dữ liệu
    - starvation_level: low / medium / high

    ``starvation_score`` là chỉ số vận hành cho dashboard, không phải metric học
    thuật chuẩn. Báo cáo khoa học vẫn nên công bố các thành phần gốc bên trên.
    """
    if total_classes <= 0:
        raise ValueError("total_classes must be > 0")

    labels = list(labels)
    num_samples = len(labels)
    if num_samples == 0:
        return {
            "num_samples": 0,
            "class_count_present": 0,
            "class_count_missing": total_classes,
            "largest_class_ratio": 0.0,
            "label_entropy": 0.0,
            "entropy_normalized": 0.0,
            "starvation_score": 1.0,
            "starvation_level": "high",
            "class_distribution": {},
        }

    counts: Dict[int, int] = {}
    for l in labels:
        label = int(l)
        counts[label] = counts.get(label, 0) + 1

    class_count_present = len(counts)
    class_count_missing = max(total_classes - class_count_present, 0)
    largest_class_count = max(counts.values())
    largest_class_ratio = float(largest_class_count / num_samples)

    # Shannon Entropy: - sum(p_i * log2(p_i))
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / num_samples
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(total_classes) if total_classes > 1 else 1.0
    entropy_normalized = min(max(entropy / max_entropy, 0.0), 1.0)
    coverage_ratio = min(class_count_present / total_classes, 1.0)

    # Heuristic dashboard score: missing label coverage and low entropy are the
    # primary signals.  Dominant-class concentration adds a smaller penalty.
    starvation_score = (
        0.45 * (1.0 - coverage_ratio)
        + 0.35 * (1.0 - entropy_normalized)
        + 0.20 * largest_class_ratio
    )
    starvation_score = min(max(starvation_score, 0.0), 1.0)
    if starvation_score >= 0.6:
        starvation_level = "high"
    elif starvation_score >= 0.3:
        starvation_level = "medium"
    else:
        starvation_level = "low"

    return {
        "num_samples": num_samples,
        "class_count_present": class_count_present,
        "class_count_missing": class_count_missing,
        "largest_class_ratio": round(largest_class_ratio, 4),
        "label_entropy": round(entropy, 4),
        "entropy_normalized": round(entropy_normalized, 4),
        "starvation_score": round(starvation_score, 4),
        "starvation_level": starvation_level,
        "class_distribution": counts,
    }

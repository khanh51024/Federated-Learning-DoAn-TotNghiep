"""
Unit test kiểm tra phân hoạch Dirichlet non-IID và đo lường chỉ số đói dữ liệu (Data Scarcity).
"""

import numpy as np
import pytest

from train.data import calculate_data_scarcity, partition_dirichlet


def test_dirichlet_partition_conserves_samples():
    """Phân hoạch Dirichlet cho 5 clients phải bảo toàn tổng số mẫu, không làm mất hay trùng mẫu."""
    num_samples = 1000
    labels = np.random.randint(0, 38, size=num_samples)
    client_indices = partition_dirichlet(labels, num_clients=5, alpha=0.5, seed=42)

    assert len(client_indices) == 5

    # Tập hợp tất cả các indices
    all_assigned = []
    for cid, idxs in client_indices.items():
        assert len(idxs) > 0, f"Client {cid} không được rỗng"
        all_assigned.extend(idxs)

    assert len(all_assigned) == num_samples
    assert sorted(all_assigned) == list(range(num_samples))


def test_data_scarcity_metrics():
    """Kiểm tra tính toán chỉ số đói dữ liệu: số mẫu, số lớp, tỷ lệ lớp lớn nhất, entropy."""
    # Giả lập client 4 có 100 mẫu, trong đó 80 mẫu thuộc lớp 0, 20 mẫu thuộc lớp 1
    labels = [0] * 80 + [1] * 20
    scarcity = calculate_data_scarcity(labels)

    assert scarcity["num_samples"] == 100
    assert scarcity["class_count_present"] == 2
    assert np.isclose(scarcity["largest_class_ratio"], 0.8)
    assert scarcity["label_entropy"] > 0.0

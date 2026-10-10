"""
Unit test kiểm định toán học thuật toán Federated Averaging (FedAvg).
Chứng minh FedAvg tổng hợp đúng theo công thức w_(t+1) = sum(n_k * w_k) / sum(n_k).
"""

import numpy as np
import pytest
import torch

from src.federated.strategy import aggregate_fedavg, fedavg_weighted_average


def test_fedavg_scalar():
    """
    Test theo yêu cầu của GVHD:
    Client A: n = 100, weight = 1.0
    Client B: n = 300, weight = 3.0
    Expected: (100*1 + 300*3) / 400 = 2.5
    """
    weights = [1.0, 3.0]
    num_examples = [100, 300]
    expected = 2.5

    actual = fedavg_weighted_average(weights, num_examples)
    assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"


def test_fedavg_vector():
    """Test tổng hợp vector 1 chiều với 3 client."""
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([4.0, 5.0, 6.0])
    w3 = np.array([7.0, 8.0, 9.0])
    n = [10, 20, 30]  # total = 60

    # Expected: (10*w1 + 20*w2 + 30*w3) / 60
    expected = (10 * w1 + 20 * w2 + 30 * w3) / 60.0
    actual = fedavg_weighted_average([w1, w2, w3], n)

    assert np.allclose(actual, expected)


def test_fedavg_multidimensional_tensor_5_clients():
    """Test tổng hợp tensor 4D (conv filter [16, 3, 3, 3]) cho 5 clients có n_k khác nhau."""
    np.random.seed(42)
    shape = (16, 3, 3, 3)
    n_k_list = [7568, 7699, 7817, 8032, 7868]
    total_n = sum(n_k_list)

    client_updates = []
    expected_tensor = np.zeros(shape, dtype=np.float64)

    for cid, n_k in enumerate(n_k_list):
        t = np.random.randn(*shape).astype(np.float32)
        expected_tensor += t.astype(np.float64) * n_k
        client_updates.append({
            "client_id": cid,
            "num_examples": n_k,
            "weights": {
                "features.0.0.weight": torch.from_numpy(t),
                "features.0.1.num_batches_tracked": torch.tensor(10 * (cid + 1), dtype=torch.int64),
            }
        })

    expected_tensor = expected_tensor / total_n
    parent = {"features.0.1.num_batches_tracked": torch.tensor(4, dtype=torch.int64)}
    aggregated = aggregate_fedavg(client_updates, parent_weights=parent)

    # 1. Kiểm tra conv weights
    actual_conv = aggregated["features.0.0.weight"].numpy()
    assert np.allclose(actual_conv, expected_tensor, atol=1e-6)

    # 2. num_batches_tracked là bộ đếm tích lũy: parent + tổng phần tăng của các client.
    #    Client k kết thúc ở 10*(k+1) từ cùng parent=4 -> phần tăng (6+16+26+36+46)=130 -> 134.
    assert aggregated["features.0.1.num_batches_tracked"].item() == 4 + 130


def test_fedavg_integer_buffer_requires_parent_and_rejects_regression():
    """Bộ đếm BatchNorm: thiếu parent hoặc client đi lùi so với parent phải bị từ chối, không đoán."""
    def update(cid, counter):
        return {
            "client_id": cid,
            "num_examples": 10,
            "weights": {
                "features.0.0.weight": torch.ones(2, 2),
                "features.0.1.num_batches_tracked": torch.tensor(counter, dtype=torch.int64),
            },
        }

    with pytest.raises(ValueError, match="Missing parent value"):
        aggregate_fedavg([update(0, 5), update(1, 7)])

    parent = {"features.0.1.num_batches_tracked": torch.tensor(6, dtype=torch.int64)}
    with pytest.raises(ValueError, match="regressed"):
        aggregate_fedavg([update(0, 5), update(1, 7)], parent_weights=parent)


def test_fedavg_rejects_nan_and_inf():
    """FedAvg phải từ chối ngay lập tức nếu client gửi weights chứa NaN hoặc Inf."""
    w_normal = np.array([1.0, 2.0])
    w_nan = np.array([1.0, np.nan])
    w_inf = np.array([np.inf, 2.0])

    with pytest.raises(FloatingPointError):
        fedavg_weighted_average([w_normal, w_nan], [10, 10])

    with pytest.raises(FloatingPointError):
        fedavg_weighted_average([w_normal, w_inf], [10, 10])


def test_fedavg_rejects_shape_mismatch():
    """FedAvg phải từ chối nếu có client gửi shape sai lệch."""
    w1 = np.array([1.0, 2.0])
    w2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        fedavg_weighted_average([w1, w2], [10, 10])

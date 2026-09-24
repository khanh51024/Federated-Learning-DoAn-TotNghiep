"""Integration tests for Flower FedAvg backend: parity with sequential oracle and failure gate."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
flwr = pytest.importorskip("flwr")
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays

from stage1_compat.config import JobConfig
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.runner import get_parameters, set_parameters
from stage2_scratch.flower_adapter import VerifiedFedAvgStrategy, FlowerPlantClient
from stage2_scratch.aggregation import aggregate_fedavg_parameters


class SmallModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.bn(self.conv(x)))).flatten(1))


def test_flower_failure_gate_rejects_failures_when_disabled():
    """Khi accept_failures=False và có ít nhất 1 failure, aggregate_fit phải trả về (None, {}) và không commit round."""
    model = SmallModel(4)
    val_ds = TensorDataset(torch.zeros(4, 3, 16, 16), torch.zeros(4, dtype=torch.long))
    val_loader = DataLoader(val_ds, batch_size=2)

    strategy = VerifiedFedAvgStrategy(
        global_model=model,
        val_loader=val_loader,
        target_device=torch.device("cpu"),
        total_rounds=1,
        accept_failures=False,
    )

    params = get_parameters(model)
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(params),
        num_examples=10,
        metrics={"train_loss": 1.0},
    )

    # 1 client thành công, 1 failure
    agg, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[(None, fit_res)],
        failures=[RuntimeError("Simulated client connection drop")],
    )

    assert agg is None, "aggregate_fit must return None when failures occur and accept_failures=False"
    assert metrics == {}, "metrics must be empty when round is rejected"
    assert len(strategy.history) == 0, "No round should be committed to history on failure"


def test_bn_counter_aggregation_policy():
    """Kiểm tra BatchNorm counter: base + sum(local - base), không nhân base lên K lần."""
    base_counter = np.array(10, dtype=np.int64)
    base_weight = np.array([1.0, 2.0], dtype=np.float32)
    base_params = [base_weight, base_counter]

    # 5 clients, mỗi client xử lý thêm 2 batches (local counter = 12)
    client_params = []
    for _ in range(5):
        c_weight = np.array([1.1, 2.1], dtype=np.float32)
        c_counter = np.array(12, dtype=np.int64)  # delta = +2
        client_params.append([c_weight, c_counter])

    client_samples = [20, 20, 20, 20, 20]
    agg = aggregate_fedavg_parameters(base_params, client_params, client_samples)

    # Expected counter = 10 + 5 * 2 = 20 (thay vì 12 * 5 = 60 nếu sum thô)
    assert agg[1] == 20, f"Expected counter 20, got {agg[1]}"
    assert agg[1].dtype == np.int64


def test_flower_and_sequential_single_round_parity():
    """Kiểm tra parity kết quả tổng hợp giữa Flower strategy và Sequential aggregation trên cùng dữ liệu."""
    torch.manual_seed(42)
    np.random.seed(42)

    model_flower = SmallModel(4)
    model_seq = SmallModel(4)
    model_seq.load_state_dict(model_flower.state_dict())

    # Tạo dữ liệu giả lập cho 5 clients
    client_weights = []
    client_samples = [16, 24, 20, 16, 24]
    fit_results = []

    for cid, samples in enumerate(client_samples):
        # Giả lập tham số sau local training (giữ nguyên dtype cho integer buffer)
        state = []
        for p in model_flower.state_dict().values():
            arr = p.clone().detach().numpy()
            if np.issubdtype(arr.dtype, np.integer):
                state.append(arr + (cid + 1))
            else:
                state.append(arr + np.float32(0.01 * (cid + 1)))
        client_weights.append(state)
        fit_results.append((
            None,
            FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(state),
                num_examples=samples,
                metrics={"train_loss": 0.5 + 0.05 * cid},
            ),
        ))

    val_ds = TensorDataset(torch.randn(20, 3, 16, 16), torch.randint(0, 4, (20,)))
    val_loader = DataLoader(val_ds, batch_size=4)

    # 1. Chạy Flower strategy aggregate_fit
    strategy = VerifiedFedAvgStrategy(
        global_model=model_flower,
        val_loader=val_loader,
        target_device=torch.device("cpu"),
        total_rounds=1,
        accept_failures=False,
    )
    agg_params, _ = strategy.aggregate_fit(1, fit_results, [])
    assert agg_params is not None
    flower_ndarrays = parameters_to_ndarrays(agg_params)

    # 2. Chạy sequential aggregation oracle
    base_params = get_parameters(model_seq)
    oracle_ndarrays = aggregate_fedavg_parameters(base_params, client_weights, client_samples)

    # So sánh bitwise / tolerance giữa 2 kết quả
    assert len(flower_ndarrays) == len(oracle_ndarrays)
    for idx, (f_w, o_w) in enumerate(zip(flower_ndarrays, oracle_ndarrays)):
        np.testing.assert_allclose(
            f_w, o_w, rtol=1e-5, atol=1e-6,
            err_msg=f"Parameter {idx} mismatch between Flower strategy and Oracle"
        )

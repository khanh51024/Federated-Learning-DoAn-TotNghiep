"""Real 2-round training parity test between Sequential and Flower backends on CPU."""

import copy
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from stage1_compat.config import JobConfig
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.runner import get_parameters, set_parameters
from stage1_compat.upstream_loader import get_upstream_evaluate
from stage2_scratch.aggregation import aggregate_fedavg_parameters
from stage2_scratch.flower_adapter import FlowerPlantClient, VerifiedFedAvgStrategy
from stage2_scratch.runner import Stage2PlantClient, set_client_rng


class DummyTinyImageDataset(Dataset):
    def __init__(self, num_samples: int, num_classes: int = 38):
        # 3x32x32 random images for fast CPU training test
        torch.manual_seed(12345)
        self.data = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])


def test_real_two_round_flower_sequential_parity():
    """Train >= 2 real rounds with MobileNetV3-Small on both backends and assert exact tensor parity."""
    num_classes = 38
    num_clients = 5
    samples_per_client = 6  # 30 total train samples
    total_samples = num_clients * samples_per_client
    val_samples = 15

    train_ds = DummyTinyImageDataset(total_samples, num_classes=num_classes)
    val_ds = DummyTinyImageDataset(val_samples, num_classes=num_classes)
    val_loader = DataLoader(val_ds, batch_size=5, shuffle=False)

    partitions = [
        list(range(i * samples_per_client, (i + 1) * samples_per_client))
        for i in range(num_clients)
    ]
    client_samples = [len(p) for p in partitions]

    # Initial random weights from seed 42
    torch.manual_seed(42)
    np.random.seed(42)
    initial_model = create_mobilenetv3_stage1(num_classes=num_classes, pretrained=False)
    initial_params = get_parameters(initial_model)

    lr = 0.01
    weight_decay = 0.0
    momentum = 0.0
    local_epochs = 1
    batch_size = 4
    num_rounds = 2

    # -------------------------------------------------------------
    # Backend 1: Sequential Runner
    # -------------------------------------------------------------
    seq_model = create_mobilenetv3_stage1(num_classes=num_classes, pretrained=False)
    set_parameters(seq_model, initial_params)
    seq_current_params = [p.copy() for p in initial_params]

    seq_clients = [
        Stage2PlantClient(
            client_id=cid,
            train_subset=Subset(train_ds, partitions[cid]),
            class_count=num_classes,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            target_device=torch.device("cpu"),
            optimizer_name="sgd",
            momentum=momentum,
        )
        for cid in range(num_clients)
    ]

    eval_mod = get_upstream_evaluate()
    criterion = nn.CrossEntropyLoss()
    seq_history = []

    for r in range(1, num_rounds + 1):
        fitted_params, sample_counts, losses = [], [], []
        for client in seq_clients:
            params, samples, metrics = client.fit(seq_current_params, r, 42)
            fitted_params.append(params)
            sample_counts.append(samples)
            losses.append(metrics["train_loss"])

        seq_current_params = aggregate_fedavg_parameters(
            base_parameters=seq_current_params,
            client_parameters=fitted_params,
            client_samples=sample_counts,
            client_ids=list(range(num_clients)),
        )
        set_parameters(seq_model, seq_current_params)
        v_metrics = eval_mod.evaluate(seq_model, val_loader, criterion, torch.device("cpu"))
        seq_history.append((v_metrics["loss"], v_metrics["accuracy"]))

    # -------------------------------------------------------------
    # Backend 2: Flower Strategy + Flower Clients
    # -------------------------------------------------------------
    flower_model = create_mobilenetv3_stage1(num_classes=num_classes, pretrained=False)
    set_parameters(flower_model, initial_params)

    flower_clients = [
        FlowerPlantClient(
            cid=cid,
            train_subset=Subset(train_ds, partitions[cid]),
            class_count=num_classes,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            device=torch.device("cpu"),
            optimizer_name="sgd",
            momentum=momentum,
        )
        for cid in range(num_clients)
    ]

    strategy = VerifiedFedAvgStrategy(
        global_model=flower_model,
        val_loader=val_loader,
        target_device=torch.device("cpu"),
        total_rounds=num_rounds,
        accept_failures=False,
    )

    flower_current_params = [p.copy() for p in initial_params]
    flower_history = []

    for r in range(1, num_rounds + 1):
        fit_results = []
        for cid, client in enumerate(flower_clients):
            # Same deterministic RNG policy
            set_client_rng(42, r, cid)
            p, num_ex, metrics = client.fit(flower_current_params, {})
            fit_results.append((
                None,
                FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(p),
                    num_examples=num_ex,
                    metrics=metrics,
                ),
            ))

        agg_params, _ = strategy.aggregate_fit(r, fit_results, [])
        assert agg_params is not None
        flower_current_params = parameters_to_ndarrays(agg_params)
        v_row = strategy.history[-1]
        flower_history.append((v_row["validation_loss"], v_row["validation_accuracy"]))

    # -------------------------------------------------------------
    # Assert Parity Across All Rounds
    # -------------------------------------------------------------
    assert len(seq_current_params) == len(flower_current_params)
    for idx, (s_p, f_p) in enumerate(zip(seq_current_params, flower_current_params)):
        if np.issubdtype(s_p.dtype, np.integer):
            assert int(s_p) == int(f_p), f"Integer buffer mismatch at param {idx}: {s_p} vs {f_p}"
        else:
            np.testing.assert_allclose(
                s_p, f_p, rtol=1e-5, atol=1e-6,
                err_msg=f"Tensor mismatch after 2 rounds at param {idx}"
            )

    # Compare validation histories
    for r_idx, (seq_v, flw_v) in enumerate(zip(seq_history, flower_history)):
        np.testing.assert_allclose(seq_v[0], flw_v[0], rtol=1e-5, err_msg=f"Val loss mismatch round {r_idx+1}")
        np.testing.assert_allclose(seq_v[1], flw_v[1], rtol=1e-5, err_msg=f"Val acc mismatch round {r_idx+1}")

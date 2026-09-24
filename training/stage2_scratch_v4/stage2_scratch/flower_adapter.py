"""Flower (flwr) simulation adapter for Stage 2 FedAvg."""
from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import flwr as fl
    import flwr.simulation
    from flwr.common import (
        FitRes,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
    HAS_FLWR = True
except ImportError:
    fl = None
    HAS_FLWR = False
    FedAvg = object
    ClientProxy = object
    FitRes = Any
    NDArrays = Any
    Parameters = Any
    Scalar = Any

from stage1_compat.checkpoint import compute_state_dict_sha256
from stage1_compat.config import JobConfig
from stage1_compat.integrity import atomic_json
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.runner import get_parameters, set_parameters, weighted_parameters
from stage1_compat.upstream_loader import get_upstream_evaluate, get_upstream_reproducibility

_BaseClient = fl.client.NumPyClient if HAS_FLWR else object


class FlowerPlantClient(_BaseClient):
    """Flower client executing local training using the verified upstream trainer."""

    def __init__(
        self,
        cid: int,
        train_subset: Any,
        class_count: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        local_epochs: int,
        device: torch.device,
        optimizer_name: str = "sgd",
        momentum: float = 0.0,
    ):
        self.cid = cid
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.train_samples = len(train_subset)
        self.class_count = class_count
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.device = device
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        self.model = create_mobilenetv3_stage1(num_classes=class_count, pretrained=False).to(self.device)
        self._upstream_eval = get_upstream_evaluate()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        losses = []
        for _ in range(self.local_epochs):
            loss = self._upstream_eval.train_one_epoch(
                self.model, self.train_loader, optimizer, self.criterion, self.device
            )
            losses.append(loss)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return get_parameters(self.model), self.train_samples, {"train_loss": float(avg_loss)}


from stage2_scratch.aggregation import aggregate_fedavg_parameters


class VerifiedFedAvgStrategy(FedAvg):
    """Flower strategy ensuring parity with sequential runner and single global validation."""

    def __init__(
        self,
        global_model: nn.Module,
        val_loader: DataLoader,
        target_device: torch.device,
        total_rounds: int,
        accept_failures: bool = False,
        **kwargs,
    ):
        super().__init__(accept_failures=accept_failures, **kwargs)
        self.global_model = global_model
        self.val_loader = val_loader
        self.target_device = target_device
        self.total_rounds = total_rounds
        self.criterion = nn.CrossEntropyLoss()
        self.eval_mod = get_upstream_evaluate()
        self.history: List[Dict[str, Any]] = []
        self.best_acc = -1.0
        self.best_round = 0
        self.best_parameters: Optional[NDArrays] = None
        self.current_parameters: Optional[NDArrays] = get_parameters(global_model)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Enforce strict failure gate: if failures occur and accept_failures is False, reject the round
        if failures and not self.accept_failures:
            print(
                f"[Flower R{server_round}] REJECTED: {len(failures)} client failures with accept_failures=False"
            )
            return None, {}

        if not results:
            return None, {}

        # Extract weights and sample counts
        client_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        client_samples = [fit_res.num_examples for _, fit_res in results]

        # Use the verified FedAvg parameter aggregation with correct BN counter policy
        base_params = self.current_parameters if self.current_parameters is not None else get_parameters(self.global_model)
        aggregated_ndarrays = aggregate_fedavg_parameters(base_params, client_weights, client_samples)
        self.current_parameters = [w.copy() for w in aggregated_ndarrays]

        # Compute round training loss
        total_samples = sum(client_samples)
        round_train_loss = sum(
            fit_res.num_examples * float(fit_res.metrics.get("train_loss", 0.0))
            for _, fit_res in results
        ) / max(total_samples, 1)

        # Single server-side global validation
        set_parameters(self.global_model, aggregated_ndarrays)
        val_metrics = self.eval_mod.evaluate(
            self.global_model, self.val_loader, self.criterion, self.target_device
        )
        val_acc = float(val_metrics["accuracy"])
        val_f1 = float(val_metrics["macro_f1"])
        val_loss = float(val_metrics["loss"])

        row = {
            "round": server_round,
            "train_loss": float(round_train_loss),
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
        }
        self.history.append(row)

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_round = server_round
            self.best_parameters = [w.copy() for w in aggregated_ndarrays]

        print(
            f"[Flower R{server_round}/{self.total_rounds}] "
            f"train_loss={round_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | (best_val_acc={self.best_acc:.4f} @ R{self.best_round})"
        )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        return parameters_aggregated, {
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "train_loss": round_train_loss,
        }


def run_flower_fedavg(
    job: JobConfig,
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    class_names: List[str],
    partitions: List[List[int]],
    output_dir: Path | str,
    target_device: torch.device,
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
) -> Dict[str, Any]:
    """Run FedAvg simulation using Flower with verified parity."""
    if not HAS_FLWR:
        raise ImportError(
            "Flower (flwr) is not installed. To run flower simulation/verification, install the 'flwr' extra/dependency: pip install .[flower]"
        )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Re-seed RNG for initial global model
    get_upstream_reproducibility().set_seed(job.seed)
    global_model = create_mobilenetv3_stage1(len(class_names), pretrained=False).to(target_device)
    init_sha256 = compute_state_dict_sha256(global_model.state_dict())
    initial_parameters = get_parameters(global_model)

    val_loader = DataLoader(val_set, batch_size=job.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=job.batch_size, shuffle=False, num_workers=0)

    # Strategy
    strategy = VerifiedFedAvgStrategy(
        global_model=global_model,
        val_loader=val_loader,
        target_device=target_device,
        total_rounds=job.rounds,
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # evaluate done via server strategy
        min_fit_clients=len(partitions),
        min_available_clients=len(partitions),
        initial_parameters=ndarrays_to_parameters(initial_parameters),
    )

    # Client function
    def client_fn(cid_str: str) -> fl.client.Client:
        cid = int(cid_str)
        client = FlowerPlantClient(
            cid=cid,
            train_subset=Subset(train_set, partitions[cid]),
            class_count=len(class_names),
            batch_size=job.batch_size,
            lr=job.lr,
            weight_decay=job.weight_decay,
            local_epochs=job.local_epochs,
            device=target_device,
            optimizer_name=optimizer_name,
            momentum=momentum,
        )
        return client.to_client()

    # Run simulation
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 1.0 if target_device.type == "cuda" else 0.0,
    }
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(partitions),
        config=fl.server.ServerConfig(num_rounds=job.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    # Final test evaluation only if not diagnostic / sentinel
    test_metrics = None
    if test_set is not None and len(test_set) > 0 and type(test_set).__name__ != "SentinelTestDataset":
        best_model = copy.deepcopy(global_model).cpu()
        if strategy.best_parameters is not None:
            set_parameters(best_model, strategy.best_parameters)

        eval_mod = get_upstream_evaluate()
        raw_metrics = eval_mod.evaluate(best_model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        test_metrics = {k: float(v) if k != "samples" else int(v) for k, v in raw_metrics.items()}

    result = {
        "job_id": job.job_id,
        "framework": "flower",
        "flower_version": fl.__version__,
        "initialization_sha256": init_sha256,
        "best_round": strategy.best_round,
        "best_validation_accuracy": strategy.best_acc,
        "test_metrics": test_metrics,
        "history": strategy.history,
    }
    atomic_json(output / f"{job.job_id}_flower_metrics.json", result)
    return result

"""Sequential FedAvg runner for Stage 2 Scratch (FedAvg-only).

Guarantees:
1. Strict FedAvg aggregation using stage2_scratch.aggregation.aggregate_fedavg_parameters:
   - Float parameters & running stats: weighted average by unique train samples n_k.
   - Integer BN buffer num_batches_tracked: base + sum(local_k - base).
2. Exactly ONE global validation per round at server (not repeated per client).
3. Deterministic per-client seed policy:
   seed_k = (job.seed * 1000003 + server_round * 1009 + client_id * 31) & 0xFFFFFFFF.
4. Final test set is NEVER accessed during diagnostic/calibration runs (asserted by Sentinel).
5. Atomic checkpointing with stage2_scratch_fedavg_v4 identity context.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from stage1_compat.artifacts import validate_completed
from stage1_compat.budget import BudgetLedger, DeadlineLoader, check_deadline
from stage1_compat.checkpoint import (
    atomic_torch_save,
    compute_state_dict_sha256,
    load_checkpoint,
    save_atomic_checkpoint,
)
from stage1_compat.config import JobConfig
from stage1_compat.identity import generate_job_identity
from stage1_compat.data import make_stage1_loader
from stage1_compat.integrity import atomic_json, digest, file_hash
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.runner import get_parameters, set_parameters, get_upstream_reproducibility
from stage1_compat.upstream_loader import get_upstream_evaluate
from stage2_scratch.aggregation import aggregate_fedavg_parameters

PROTOCOL = "stage2_scratch_fedavg_v4"


def set_client_rng(seed: int, server_round: int, client_id: int) -> int:
    """Deterministic RNG policy per client/round."""
    client_seed = (seed * 1000003 + server_round * 1009 + client_id * 31) & 0xFFFFFFFF
    random.seed(client_seed)
    np.random.seed(client_seed)
    torch.manual_seed(client_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(client_seed)
    return client_seed


class Stage2PlantClient:
    """Client for local training in Stage 2 FedAvg."""

    def __init__(
        self,
        client_id: int,
        train_subset: Subset,
        class_count: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        local_epochs: int,
        target_device: torch.device,
        optimizer_name: str = "sgd",
        momentum: float = 0.0,
    ):
        self.client_id = client_id
        self.train_samples = len(train_subset)
        self.train_loader = DeadlineLoader(make_stage1_loader(train_subset, batch_size=batch_size, shuffle=True))
        self.class_count = class_count
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.device = target_device
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        self.model = create_mobilenetv3_stage1(num_classes=class_count, pretrained=False).to(self.device)
        self._upstream_eval = get_upstream_evaluate()

    def fit(
        self, parameters: list[np.ndarray], server_round: int, base_seed: int
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        set_client_rng(base_seed, server_round, self.client_id)
        set_parameters(self.model, parameters)

        if self.optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
            )
        else:
            optimizer = optim.AdamW(
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


def run_single_fedavg_job(
    job: JobConfig,
    train_set: Any,
    val_set: Any,
    test_set: Any,
    class_names: list[str],
    partitions: list[list[int]],
    output_dir: Path,
    ledger: BudgetLedger,
    target_device: torch.device,
    resume: bool = True,
    calibration: bool = False,
) -> dict[str, Any]:
    """Execute a single FedAvg job with Stage 2 protocol v4."""
    job_started = time.monotonic()
    job_id = job.job_id
    get_upstream_reproducibility().set_seed(job.seed)

    global_model = create_mobilenetv3_stage1(num_classes=len(class_names), pretrained=False).to(target_device)
    initialization_sha256 = compute_state_dict_sha256(global_model.state_dict())

    expected_initialization = getattr(train_set, "initialization_sha256", None)
    if expected_initialization and initialization_sha256 != expected_initialization:
        raise ValueError("Initialization differs from recorded seed state")

    context = getattr(train_set, "identity_context", {"scope": PROTOCOL})
    context = {
        **context,
        "protocol": PROTOCOL,
        "partition_sha256": digest(partitions),
        "classes": class_names,
        "train_count": len(train_set),
        "validation_count": len(val_set),
        "test_count": len(test_set),
        "train_device": str(target_device),
        "evaluation_device": str(target_device),
        "calibration": calibration,
        "initialization_sha256": initialization_sha256,
        "bn_policy": "base_plus_sum_delta",
    }
    identity = generate_job_identity(job, context=context)
    job_output_dir = Path(output_dir) / job_id
    checkpoint_dir = job_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{job_id}_checkpoint.pth"

    entry = ledger.jobs.get(job_id)
    metrics_path = job_output_dir / "fedavg_metrics.json"
    if metrics_path.exists() or (entry and entry.status == "COMPLETED"):
        metrics = validate_completed(job_output_dir, identity)
        ledger.record_job_start(job_id, job.alpha, job.seed, job.rounds, identity.identity_hash)
        ledger.record_job_completion(job_id, metrics["test_metrics"]["accuracy"], metrics["test_metrics"]["macro_f1"])
        return metrics
    if entry and entry.current_round and not checkpoint_file.exists():
        raise ValueError("Ledger references missing committed checkpoint")
    if not resume and checkpoint_file.exists():
        raise ValueError("Refuse overwrite existing checkpoint; choose a new output directory")
    ledger.record_job_start(job_id, job.alpha, job.seed, job.rounds, identity.identity_hash)

    current_parameters = get_parameters(global_model)
    best_parameters = [p.copy() for p in current_parameters]
    best_model_state = copy.deepcopy(global_model.state_dict())
    best_validation_accuracy = -1.0
    best_round = 0
    start_round = 1
    history_rows = []
    round_timings = {}

    checkpoint_data = None
    if resume and checkpoint_file.exists():
        print(f"[{job_id}] Loading checkpoint from {checkpoint_file}...")
        checkpoint_data = load_checkpoint(checkpoint_file, expected_identity=identity, device="cpu")
        saved_round = checkpoint_data["server_round"]
        for idx, elapsed in checkpoint_data.get("round_timings", {}).items():
            ledger.record_round_progress(job_id, int(idx), elapsed, checkpoint_data["best_round"], checkpoint_data["best_validation_accuracy"])
        if saved_round >= job.rounds:
            print(f"[{job_id}] Checkpoint already completed {saved_round}/{job.rounds} rounds.")
            start_round = job.rounds + 1
        else:
            start_round = saved_round + 1
            print(f"[{job_id}] Resuming from round {start_round}...")

        global_model.load_state_dict(checkpoint_data["global_model_state"])
        current_parameters = get_parameters(global_model)
        history_rows = list(checkpoint_data["history"])
        best_round = checkpoint_data["best_round"]
        best_validation_accuracy = checkpoint_data["best_validation_accuracy"]
        best_model_state = copy.deepcopy(checkpoint_data["best_model_state"])
        round_timings = dict(checkpoint_data.get("round_timings", {}))

    # Initialize clients
    clients = [
        Stage2PlantClient(
            client_id=client_id,
            train_subset=Subset(train_set, indices),
            class_count=len(class_names),
            batch_size=job.batch_size,
            lr=job.lr,
            weight_decay=job.weight_decay,
            local_epochs=job.local_epochs,
            target_device=target_device,
            optimizer_name=getattr(job, "optimizer", "sgd"),
            momentum=getattr(job, "momentum", 0.0),
        )
        for client_id, indices in enumerate(partitions)
    ]

    server_val_loader = DeadlineLoader(make_stage1_loader(val_set, batch_size=job.batch_size, shuffle=False))
    eval_module = get_upstream_evaluate()
    criterion = nn.CrossEntropyLoss()

    round_duration_estimate = getattr(train_set, "round_seconds", 60.0)

    # FedAvg training loop
    for server_round in range(start_round, job.rounds + 1):
        round_start_time = time.time()
        check_deadline()

        can_run, reason = ledger.can_continue(estimated_next_round_seconds=round_duration_estimate)
        if not can_run:
            print(f"[{job_id}] SAFE STOP BEFORE ROUND {server_round}: {reason}")
            return {
                "status": "PAUSED_QUOTA",
                "reason": reason,
                "job_id": job_id,
                "paused_at_round": server_round - 1,
            }

        print(f"[{job_id}] --- FedAvg Round {server_round}/{job.rounds} --- (alpha={job.alpha})", flush=True)

        # 1. Fit clients sequentially
        fitted_parameters, sample_counts, train_losses, client_ids = [], [], [], []
        for client in clients:
            params, samples, metrics = client.fit(current_parameters, server_round, job.seed)
            fitted_parameters.append(params)
            sample_counts.append(samples)
            train_losses.append(metrics["train_loss"])
            client_ids.append(client.client_id)

        # 2. Strict FedAvg parameter aggregation
        current_parameters = aggregate_fedavg_parameters(
            base_parameters=current_parameters,
            client_parameters=fitted_parameters,
            client_samples=sample_counts,
            client_ids=client_ids,
        )

        # 3. Exactly ONE server-side global validation
        set_parameters(global_model, current_parameters)
        val_metrics = eval_module.evaluate(global_model, server_val_loader, criterion, target_device)
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])
        val_f1 = float(val_metrics["macro_f1"])

        total_train_samples = sum(sample_counts)
        round_train_loss = sum(loss * samples for loss, samples in zip(train_losses, sample_counts)) / max(total_train_samples, 1)

        row = {
            "epoch": server_round,
            "train_loss": float(round_train_loss),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "validation_macro_f1": float(val_f1),
        }
        history_rows.append(row)

        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            best_round = server_round
            best_model_state = copy.deepcopy(global_model.state_dict())
            best_parameters = [p.copy() for p in current_parameters]

        elapsed_round = time.time() - round_start_time
        round_timings[str(server_round)] = elapsed_round
        round_duration_estimate = max(round_duration_estimate, elapsed_round * 1.5)

        print(
            f"[{job_id}] R{server_round}: train_loss={round_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | (best_val_acc={best_validation_accuracy:.4f} @ R{best_round})"
        )

        # Atomic checkpoint commit
        save_atomic_checkpoint(
            checkpoint_path=checkpoint_file,
            server_round=server_round,
            global_model_state=global_model.state_dict(),
            best_model_state=best_model_state,
            best_round=best_round,
            best_validation_accuracy=best_validation_accuracy,
            history=history_rows,
            identity=identity,
            round_timings=round_timings,
        )

        ledger.record_round_progress(
            job_id=job_id,
            round_idx=server_round,
            elapsed_seconds=elapsed_round,
            best_round=best_round,
            best_val_acc=best_validation_accuracy,
        )

    if calibration:
        result = {
            "status": "CALIBRATED_NO_TEST",
            "job_id": job_id,
            "protocol": PROTOCOL,
            "identity": identity.to_dict(),
            "rounds": job.rounds,
            "best_round": best_round,
            "best_validation_accuracy": float(best_validation_accuracy),
            "history": history_rows,
            "round_timings": round_timings,
            "elapsed_seconds": time.monotonic() - job_started,
            "round_seconds": [ledger.jobs[job_id].round_events[str(i)] for i in range(1, job.rounds + 1)] if (job_id in ledger.jobs and hasattr(ledger.jobs[job_id], "round_events")) else [],
            "test_evaluated": False,
        }
        atomic_json(job_output_dir / "calibration_metrics.json", result)
        return result

    # Completed training all rounds
    print(f"[{job_id}] Evaluating final test set on best checkpoint (Round {best_round})...")
    best_model = create_mobilenetv3_stage1(num_classes=len(class_names), pretrained=False).to(target_device)
    best_model.load_state_dict(best_model_state)
    test_loader = make_stage1_loader(test_set, batch_size=job.batch_size, shuffle=False)
    final_test_metrics = eval_module.evaluate(best_model, test_loader, criterion, target_device)
    final_test_metrics = {
        k: float(v) if k != "samples" else int(v)
        for k, v in final_test_metrics.items()
    }
    print(f"[{job_id}] Test Accuracy: {final_test_metrics['accuracy']:.4f}")

    best_checkpoint_sha256 = compute_state_dict_sha256(best_model.state_dict())

    # Lưu best model checkpoint riêng biệt cho bàn giao
    best_model_target = job_output_dir / "best_model.pth"
    atomic_torch_save(best_model_target, {
        "model_state": best_model.state_dict(),
        "state_sha256": best_checkpoint_sha256,
        "class_names": class_names,
        "best_round": best_round,
        "best_validation_accuracy": float(best_validation_accuracy),
        "test_metrics": final_test_metrics,
        "identity": identity.to_dict(),
    })

    metrics_payload = {
        "experiment": "fedavg",
        "profile": PROTOCOL,
        "scientific_stage2_complete": False,
        "job_id": job_id,
        "seed": job.seed,
        "alpha": float(job.alpha),
        "rounds": job.rounds,
        "local_epochs": job.local_epochs,
        "clients": job.num_clients,
        "batch_size": job.batch_size,
        "optimizer": getattr(job, "optimizer", "sgd"),
        "lr": float(job.lr),
        "weight_decay": float(job.weight_decay),
        "split": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "selection_metric": "validation_accuracy",
        "best_round": best_round,
        "best_validation_accuracy": float(best_validation_accuracy),
        "test_metrics": final_test_metrics,
        "checkpoint_file": str(best_model_target),
        "checkpoint_sha256": best_checkpoint_sha256,
        "checkpoint_file_sha256": file_hash(best_model_target),
        "elapsed_seconds": max(ledger.jobs[job_id].elapsed_seconds, time.monotonic() - job_started) if (job_id in ledger.jobs and hasattr(ledger.jobs[job_id], "elapsed_seconds")) else (time.monotonic() - job_started),
        "resolved_config": job.to_dict(),
        "payload": {
            "kind": "estimated",
            "scope": "global weights download + client upload; no transport overhead",
            "bytes": 2 * len(partitions) * job.rounds * sum(p.nbytes for p in current_parameters),
        },
        "sample_count": sum(len(ids) for ids in partitions) * job.rounds * job.local_epochs,
        "step_count": sum((len(ids) + job.batch_size - 1) // job.batch_size for ids in partitions) * job.rounds * job.local_epochs,
        "identity": identity.to_dict(),
        "history": history_rows,
        "status": "COMPLETED",
    }
    metrics_file = job_output_dir / "fedavg_metrics.json"
    atomic_json(metrics_file, metrics_payload)
    atomic_json(job_output_dir / "job_result.json", metrics_payload)
    validate_completed(job_output_dir, identity)

    ledger.record_job_completion(
        job_id=job_id,
        test_accuracy=final_test_metrics["accuracy"],
        test_macro_f1=final_test_metrics["macro_f1"],
    )
    return metrics_payload

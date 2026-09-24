"""Sequential FedAvg Runner cho profile stage1_compat.

Chỉ huấn luyện FedAvg (alpha=100, alpha=1, seed=42).
Cam kết ghi atomic checkpoint từng round, khôi phục RNG chính xác khi resume,
và kiểm soát quota/deadline an toàn.
"""

import copy
import json
import time
from collections import OrderedDict
from collections.abc import Sized
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset

from stage1_compat.budget import BudgetLedger, DeadlineLoader, BudgetExhausted, check_deadline
from stage1_compat.integrity import atomic_json, file_hash, digest, read_json
from stage1_compat.artifacts import validate_completed
from stage1_compat.checkpoint import (
    compute_state_dict_sha256,
    atomic_torch_save,
    load_checkpoint,
    save_atomic_checkpoint,
    set_rng_states,
)
from stage1_compat.config import JobConfig, Stage1CompatProfileConfig, get_default_profile_config
from stage1_compat.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROUNDS,
    DEFAULT_WEIGHT_DECAY,
    NUM_CLASSES,
    SEED,
)
from stage1_compat.data import load_stage1_datasets, load_stage1_partition, make_stage1_loader
from stage1_compat.identity import generate_job_identity
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.upstream_loader import (
    get_upstream_evaluate,
    get_upstream_reproducibility,
)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state = OrderedDict((key, torch.tensor(value)) for key, value in zip(model.state_dict().keys(), parameters))
    model.load_state_dict(state, strict=True)


def weighted_metrics(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float]:
    """Tổng hợp accuracy/macro-F1 có trọng số theo số mẫu của client."""
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    return {
        name: sum(num_examples * values.get(name, 0.0) for num_examples, values in metrics) / total
        for name in ("accuracy", "macro_f1")
    }


def weighted_parameters(client_parameters: list[list[np.ndarray]], client_samples: list[int]) -> list[np.ndarray]:
    """Tổng hợp tham số bình quân gia quyền theo số mẫu (chuẩn GĐ1 upstream)."""
    total_samples = sum(client_samples)
    if (not client_parameters or len(client_parameters) != len(client_samples)
            or any(samples <= 0 for samples in client_samples)):
        raise ValueError("Aggregation requires one positive sample count per client")
    return [
        np.asarray(sum(samples * parameters[index] for parameters, samples in zip(client_parameters, client_samples)) / total_samples)
        for index in range(len(client_parameters[0]))
    ]


class Stage1PlantClient:
    """Client huấn luyện cục bộ tuần tự chuẩn GĐ1."""

    def __init__(
        self,
        client_id: int,
        train_subset: Subset,
        val_dataset: Dataset,
        class_count: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        local_epochs: int,
        target_device: torch.device,
        optimizer_name: str = "sgd",
        momentum: float = 0.0,
    ):
        if not isinstance(val_dataset, Sized):
            raise TypeError("Validation dataset must define __len__")
        self.train_samples = len(train_subset)
        self.validation_samples = len(val_dataset)
        self.client_id = client_id
        self.train_loader = DeadlineLoader(make_stage1_loader(train_subset, batch_size=batch_size, shuffle=True))
        self.val_loader = DeadlineLoader(make_stage1_loader(val_dataset, batch_size=batch_size, shuffle=False))
        self.class_count = class_count
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.device = target_device
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()

        # Khởi tạo mô hình client (không cần pretrained weights vì sẽ nhận global weights trước khi fit)
        self.model = create_mobilenetv3_stage1(num_classes=class_count, pretrained=False).to(self.device)
        self._upstream_eval = get_upstream_evaluate()

    def fit(self, parameters: list[np.ndarray]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        set_parameters(self.model, parameters)
        # Local optimizer: SGD bám sát Algorithm 1 (McMahan et al.) hoặc AdamW legacy
        if self.optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        losses = []
        for _ in range(self.local_epochs):
            loss = self._upstream_eval.train_one_epoch(
                self.model, self.train_loader, optimizer, self.criterion, self.device
            )
            losses.append(loss)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return get_parameters(self.model), self.train_samples, {"train_loss": avg_loss}

    def evaluate(self, parameters: list[np.ndarray]) -> tuple[float, int, dict[str, float]]:
        set_parameters(self.model, parameters)
        metrics = self._upstream_eval.evaluate(self.model, self.val_loader, self.criterion, self.device)
        return float(metrics["loss"]), self.validation_samples, {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }


def run_single_fedavg_job(
    job: JobConfig,
    train_set,
    val_set,
    test_set,
    class_names: list[str],
    partitions: list[list[int]],
    output_dir: Path,
    ledger: BudgetLedger,
    target_device: torch.device,
    resume: bool = True,
    calibration: bool = False,
) -> dict[str, Any]:
    """Thực thi một job FedAvg hoàn chỉnh với atomic commit và resume."""
    job_started = time.monotonic()
    job_id = job.job_id
    get_upstream_reproducibility().set_seed(job.seed)
    global_model = create_mobilenetv3_stage1(num_classes=len(class_names), pretrained=job.pretrained).to(target_device)
    initialization_sha256 = compute_state_dict_sha256(global_model.state_dict())
    expected_initialization = getattr(train_set, "initialization_sha256", None)
    if expected_initialization and initialization_sha256 != expected_initialization:
        raise ValueError("Initialization differs from frozen calibration")
    context = getattr(train_set, "identity_context", {"scope": "synthetic_cpu_test"})
    context = {**context, "partition_sha256": digest(partitions), "classes": class_names,
               "train_count": len(train_set), "validation_count": len(val_set), "test_count": len(test_set),
               "train_device": str(target_device), "evaluation_device": "cpu", "calibration": calibration, "initialization_sha256": initialization_sha256}
    identity = generate_job_identity(job, context=context)
    job_output_dir = output_dir / job_id
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
    history_rows: list[dict[str, Any]] = []
    best_parameters = copy.deepcopy(current_parameters)
    best_validation_accuracy = -1.0
    best_round = 0
    start_round = 1

    checkpoint_data = None
    # Kiểm tra Resume
    if resume and checkpoint_file.exists():
        print(f"[{job_id}] Tìm thấy checkpoint tại {checkpoint_file}. Đang nạp và xác minh...")
        checkpoint_data = load_checkpoint(checkpoint_file, expected_identity=identity, device="cpu")
        saved_round = checkpoint_data["server_round"]
        for idx, elapsed in checkpoint_data.get("round_timings", {}).items():
            ledger.record_round_progress(job_id, int(idx), elapsed, checkpoint_data["best_round"], checkpoint_data["best_validation_accuracy"])
        if saved_round >= job.rounds:
            print(f"[{job_id}] Checkpoint đã hoàn thành đủ {saved_round}/{job.rounds} rounds.")
            start_round = job.rounds + 1
        else:
            start_round = saved_round + 1
            print(f"[{job_id}] Khôi phục trạng thái từ round {saved_round}. Tiếp tục từ round {start_round}...")

        global_model.load_state_dict(checkpoint_data["global_model_state"])
        current_parameters = get_parameters(global_model)
        history_rows = checkpoint_data["history"]
        best_round = checkpoint_data["best_round"]
        best_validation_accuracy = checkpoint_data["best_validation_accuracy"]
        best_model_state = checkpoint_data["best_model_state"]
        best_model_temp = create_mobilenetv3_stage1(num_classes=len(class_names), pretrained=False)
        best_model_temp.load_state_dict(best_model_state)
        best_parameters = get_parameters(best_model_temp)

    # Khởi tạo clients
    clients = [
        Stage1PlantClient(
            client_id=client_id,
            train_subset=Subset(train_set, indices),
            val_dataset=val_set,
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

    # Model construction consumes RNG. Restore only after all clients exist.
    if checkpoint_data is not None:
        set_rng_states(checkpoint_data["rng_states"])

    round_duration_estimate = getattr(train_set, "round_seconds", 60.0)

    round_timings = dict(checkpoint_data.get("round_timings", {})) if checkpoint_data else {}

    # Vòng lặp huấn luyện FedAvg
    for server_round in range(start_round, job.rounds + 1):
        round_start_time = time.time()
        check_deadline()

        # Kiểm tra quota / deadline trước khi bắt đầu round
        can_run, reason = ledger.can_continue(estimated_next_round_seconds=round_duration_estimate)
        if not can_run:
            print(f"[{job_id}] DỪNG AN TOÀN TRƯỚC ROUND {server_round}: {reason}")
            return {
                "status": "PAUSED_QUOTA",
                "reason": reason,
                "job_id": job_id,
                "paused_at_round": server_round - 1,
            }

        print(f"[{job_id}] --- FedAvg Round {server_round}/{job.rounds} --- (alpha={job.alpha})", flush=True)

        # 1. Fit clients
        fitted_parameters, sample_counts, train_losses = [], [], []
        for client in clients:
            params, samples, metrics = client.fit(current_parameters)
            fitted_parameters.append(params)
            sample_counts.append(samples)
            train_losses.append(metrics["train_loss"])

        # 2. Weighted parameter aggregation
        current_parameters = weighted_parameters(fitted_parameters, sample_counts)

        # 3. Validation evaluation (client-level evaluation matching GĐ1 upstream)
        validation_results = []
        for client in clients:
            loss, samples, metrics = client.evaluate(current_parameters)
            validation_results.append((samples, {"loss": loss, **metrics}))
        total_validation = sum(samples for samples, _ in validation_results)
        validation_loss = float(sum(samples * values["loss"] for samples, values in validation_results) / max(total_validation, 1))
        aggregated = weighted_metrics([(samples, values) for samples, values in validation_results])
        val_acc = float(aggregated["accuracy"])
        val_f1 = float(aggregated["macro_f1"])

        total_train_samples = sum(sample_counts)
        round_train_loss = sum(loss * samples for loss, samples in zip(train_losses, sample_counts)) / max(total_train_samples, 1)

        row = {
            "epoch": server_round,
            "train_loss": float(round_train_loss),
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(val_acc),
            "validation_macro_f1": float(val_f1),
        }
        history_rows.append(row)

        # 4. Checkpoint selection policy: best validation accuracy (tie keeps earliest round)
        set_parameters(global_model, current_parameters)
        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            best_round = server_round
            best_parameters = [value.copy() for value in current_parameters]

        print(
            f"[{job_id} R{server_round}/{job.rounds}] "
            f"train_loss={round_train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | (best_val_acc={best_validation_accuracy:.4f} @ R{best_round})"
        )

        round_elapsed = time.time() - round_start_time
        round_duration_estimate = max(round_duration_estimate, round_elapsed * 1.5)  # cập nhật dự toán cho round sau

        round_timings[str(server_round)] = round_elapsed

        # 5. Atomic checkpoint commit
        best_model_for_saving = copy.deepcopy(global_model).cpu()
        set_parameters(best_model_for_saving, best_parameters)

        save_atomic_checkpoint(
            checkpoint_path=checkpoint_file,
            server_round=server_round,
            global_model_state=global_model.state_dict(),
            best_model_state=best_model_for_saving.state_dict(),
            best_round=best_round,
            best_validation_accuracy=best_validation_accuracy,
            history=history_rows,
            identity=identity,
            round_timings=round_timings,
        )

        # Cập nhật ledger
        ledger.record_round_progress(
            job_id=job_id,
            round_idx=server_round,
            elapsed_seconds=round_elapsed,
            best_round=best_round,
            best_val_acc=best_validation_accuracy,
        )

    if calibration:
        result = {"status": "CALIBRATED_NO_TEST", "job_id": job_id, "identity": identity.to_dict(),
                  "rounds": job.rounds, "elapsed_seconds": time.monotonic() - job_started,
                  "round_seconds": [ledger.jobs[job_id].round_events[str(i)] for i in range(1, job.rounds + 1)],
                  "test_evaluated": False}
        atomic_json(job_output_dir / "calibration_metrics.json", result)
        return result

    # Đánh giá Final Test với checkpoint tốt nhất
    print(f"[{job_id}] Hoàn thành toàn bộ {job.rounds} rounds. Đang nạp best checkpoint (Round {best_round}) để đánh giá test...")
    best_model = copy.deepcopy(global_model).cpu()
    set_parameters(best_model, best_parameters)

    test_loader = DeadlineLoader(make_stage1_loader(test_set, batch_size=job.batch_size, shuffle=False))
    upstream_eval = get_upstream_evaluate()
    test_metrics = upstream_eval.evaluate(best_model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    print(
        f"[{job_id}] FINAL TEST: acc={test_metrics['accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f} | loss={test_metrics['loss']:.4f}"
    )

    best_checkpoint_sha256 = compute_state_dict_sha256(best_model.state_dict())

    # Lưu best model checkpoint riêng biệt cho bàn giao
    best_model_target = job_output_dir / "best_model.pth"
    atomic_torch_save(best_model_target, {
        "model_state": best_model.state_dict(),
        "state_sha256": best_checkpoint_sha256,
        "class_names": class_names,
        "best_round": best_round,
        "best_validation_accuracy": best_validation_accuracy,
        "test_metrics": test_metrics,
        "identity": identity.to_dict(),
    })

    # Lưu fedavg_metrics.json
    metrics_payload = {
        "experiment": "fedavg",
        "profile": "stage1_compat",
        "scientific_stage2_complete": False,
        "job_id": job_id,
        "seed": job.seed,
        "alpha": job.alpha,
        "rounds": job.rounds,
        "local_epochs": job.local_epochs,
        "clients": job.num_clients,
        "batch_size": job.batch_size,
        "optimizer": job.optimizer,
        "lr": job.lr,
        "weight_decay": job.weight_decay,
        "split": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "selection_metric": "validation_accuracy",
        "best_round": best_round,
        "best_validation_accuracy": best_validation_accuracy,
        "test_metrics": test_metrics,
        "checkpoint_file": str(best_model_target),
        "checkpoint_sha256": best_checkpoint_sha256,
        "checkpoint_file_sha256": file_hash(best_model_target),
        "elapsed_seconds": max(ledger.jobs[job_id].elapsed_seconds, time.monotonic() - job_started),
        "resolved_config": job.to_dict(),
        "payload": {"kind": "estimated", "scope": "global weights download + client upload; no transport overhead",
                    "bytes": 2 * len(partitions) * job.rounds * sum(p.nbytes for p in current_parameters)},
        "sample_count": sum(len(ids) for ids in partitions) * job.rounds * job.local_epochs,
        "step_count": sum((len(ids) + job.batch_size - 1) // job.batch_size for ids in partitions) * job.rounds * job.local_epochs,
        "identity": identity.to_dict(),
        "history": history_rows,
        "status": "COMPLETED",
    }
    metrics_file = job_output_dir / "fedavg_metrics.json"
    atomic_json(metrics_file, metrics_payload)
    validate_completed(job_output_dir, identity)

    ledger.record_job_completion(
        job_id=job_id,
        test_accuracy=test_metrics["accuracy"],
        test_macro_f1=test_metrics["macro_f1"],
    )

    return metrics_payload


def run_stage1_compat(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    quota_hours: float = 0.0,
    target_alpha: float | None = None,
    seed: int = SEED,
    rounds: int = DEFAULT_ROUNDS,
    device_str: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Hàm điều phối chạy toàn bộ hoặc một phần các job trong profile stage1_compat."""
    import os
    if not os.environ.get("STAGE1_SUPERVISED"):
        raise RuntimeError("Use python -m stage1_compat run for durable accounting and deadlines")
    profile_cfg = get_default_profile_config(data_dir, output_dir, quota_hours)
    out_dir = Path(profile_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = BudgetLedger(out_dir, user_quota_hours=quota_hours)

    if seed != 42 or rounds != 10 or target_alpha not in (None, 1.0, 100.0):
        raise ValueError("Frozen profile requires seed42, round10 and alpha1/100")

    # Quyết định target device
    if device_str:
        target_device = torch.device(device_str)
    else:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[stage1_compat] Khởi động runner | device={target_device} | quota={quota_hours}h")

    # Giữ process lock
    with ledger.get_lock():
        ledger._load_or_init_ledger()
        from stage1_compat.preflight import run_preflight
        checked = run_preflight(profile_cfg.data_dir, out_dir)
        if checked.status != "PASSED":
            raise ValueError("Preflight failed")
        from stage1_compat.calibrate import validate_calibration
        frozen = read_json(checked.frozen_manifest_path)
        calibration = validate_calibration(out_dir, frozen["identity_hash"], str(target_device))
        # Thiết lập reproducibility seed
        get_upstream_reproducibility().set_seed(seed)

        # Nạp dataset chuẩn GĐ1
        print(f"[stage1_compat] Đang nạp dataset từ: {profile_cfg.data_dir}")
        train_set, val_set, test_set, class_names = load_stage1_datasets(profile_cfg.data_dir, seed=seed)

        remaining_budget = max(0.0, float(os.environ["STAGE1_HARD_DEADLINE"]) - time.time())
        remaining_rounds = sum(max(0, 10 - ledger.jobs[j.job_id].current_round) if j.job_id in ledger.jobs else 10 for j in profile_cfg.jobs)
        forecast = remaining_rounds * calibration["safe_round_seconds"] + calibration["final_reserve_seconds"]
        # Total budget insufficiency pauses; session duration is handled per round below.
        total_remaining = max(0.0, float(os.environ["STAGE1_TOTAL_DEADLINE"]) - time.time())
        if forecast > total_remaining:
            atomic_json(out_dir / "budget_insufficient.json", {"forecast_seconds": forecast, "remaining_seconds": total_remaining})
            return {"status": "PAUSED_QUOTA", "reason": "Full remaining matrix does not fit locked budget"}
        train_set.identity_context = {"dataset_sha256": frozen["identity_hash"], "budget_sha256": calibration["budget_sha256"]}
        train_set.round_seconds = calibration["safe_round_seconds"]
        train_set.initialization_sha256 = calibration["initialization_sha256"]

        # Lọc danh sách job cần chạy
        jobs_to_run = []
        for job in profile_cfg.jobs:
            if target_alpha is not None and abs(job.alpha - target_alpha) > 1e-4:
                continue
            job_copy = copy.deepcopy(job)
            job_copy.seed = seed
            job_copy.rounds = rounds
            jobs_to_run.append(job_copy)

        results = {}
        for job in jobs_to_run:
            print(f"[stage1_compat] Chuẩn bị job: {job.job_id} (alpha={job.alpha}, rounds={job.rounds})")
            # Nạp partition đã pin
            partitions = load_stage1_partition(
                alpha=job.alpha,
                train_labels=train_set.targets,
                seed=seed,
                num_clients=job.num_clients,
            )

            try:
                job_res = run_single_fedavg_job(
                    job=job,
                    train_set=train_set,
                    val_set=val_set,
                    test_set=test_set,
                    class_names=class_names,
                    partitions=partitions,
                    output_dir=out_dir,
                    ledger=ledger,
                    target_device=target_device,
                    resume=resume,
                )
            except BudgetExhausted as error:
                atomic_json(out_dir / job.job_id / "pause.json", {"reason": str(error)})
                raise
            except Exception as error:
                atomic_json(out_dir / job.job_id / "failure.json", {"error": str(error)})
                raise
            from stage1_compat.collector import run_stage1_collect
            run_stage1_collect(out_dir)
            results[job.job_id] = job_res
            if job_res.get("status") == "PAUSED_QUOTA":
                print("[stage1_compat] Runner tạm dừng do đạt giới hạn quota.")
                return {
                    "status": "PAUSED_QUOTA",
                    "reason": job_res.get("reason"),
                    "results": results,
                }

    print("[stage1_compat] Hoàn thành toàn bộ các job được chỉ định.")
    return {
        "status": "COMPLETED",
        "results": results,
        "output_dir": str(out_dir),
    }

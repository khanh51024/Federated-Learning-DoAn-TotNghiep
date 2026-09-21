"""Centralized and Local-only Stage-2 baselines using the shared training primitives."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .checkpoint import (load_checkpoint, save_inference_model, save_round_checkpoint,
                         save_checkpoint_atomic, restore_rng_state)
from .budget import BudgetExhausted, check_deadline
from .budget_state import atomic_json
from .config import TrainingConfig, load_training_config
from .control import EarlyStoppingController, LRSchedulerController
from .data import (
    build_centralized_training_loader,
    build_evaluation_loader,
    build_training_loader,
    get_client_manifest_path,
)
from .model import create_mobilenet_v3_small, get_model_state_dict_cpu
from .reporting import generate_report
from .reproducibility import derive_seed, seed_everything
from .task import evaluate_model, train_local
from .progress import EventLogger, ProgressRenderer
from .evaluation_artifacts import bind_evaluation, write_evaluation
from .server_app import _append_jsonl


def _write_eval(run_dir: Path, result: Dict[str, Any]) -> None:
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    pd.DataFrame(result["confusion_matrix"]).to_csv(run_dir / "confusion_matrix.csv", index=False)


def _train_one(
    cfg: TrainingConfig,
    run_dir: Path,
    train_loader: Any,
    seed: int,
    mode: str,
    client_id: Optional[int] = None,
    progress: Optional[ProgressRenderer] = None,
    progress_logger: Optional[EventLogger] = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    def phase(round_num: int, name: str) -> None:
        if progress_logger is not None and progress is not None:
            progress_logger.log_phase(round_num, name)
            progress.poll()

    resuming = getattr(cfg, "resume_baselines", False)
    if resuming and (run_dir / "summary.json").exists():
        previous = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if previous.get("status") == "completed" and (run_dir / "test_metrics.json").exists():
            _validate_resume_config(cfg, run_dir, mode, client_id)
            from .sweep import _result_row
            _result_row(run_dir, {"scenario": cfg.data.scenario}, seed,
                        "local-client" if client_id is not None else "centralized")
            return previous
    check_deadline()
    phase(0, "prepare")
    seed_everything(seed)
    model = create_mobilenet_v3_small(38, cfg.model.weights)
    model.to(cfg.runtime.client_device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    # Keep the optimizer's parameters and momentum on the training device.
    # Moving that model CPU<->CUDA for validation can invalidate optimizer refs.
    evaluation_model = create_mobilenet_v3_small(38, None)
    evaluation_model.load_state_dict(get_model_state_dict_cpu(model))
    val_loader = build_evaluation_loader(
        cfg.data.partition_dir / "global_val.csv", cfg.data.dataset_root,
        cfg.training.eval_batch_size, cfg.training.num_workers,
    )
    test_loader = build_evaluation_loader(
        cfg.data.partition_dir / "global_test.csv", cfg.data.dataset_root,
        cfg.training.eval_batch_size, cfg.training.num_workers,
    )
    early = EarlyStoppingController(
        cfg.early_stopping.enabled, cfg.early_stopping.min_delta,
        cfg.early_stopping.patience_rounds, cfg.early_stopping.warmup_rounds,
    )
    scheduler = LRSchedulerController(
        cfg.lr_scheduler.enabled, cfg.training.lr, cfg.lr_scheduler.factor,
        cfg.lr_scheduler.patience_rounds, cfg.lr_scheduler.threshold,
        cfg.lr_scheduler.min_lr,
    )
    resolved = cfg.to_dict()
    resolved["mode"] = mode
    resolved["client_id"] = client_id
    resolved["output"]["run_dir"] = str(run_dir)
    resume_path = run_dir / "last.pt"
    payload = None
    if resuming and resume_path.exists():
        payload = load_checkpoint(resume_path)
        if (payload.get("semantic_config_hash") != cfg.semantic_config_hash
                or payload["config"].get("mode") != mode
                or payload["config"].get("client_id") != client_id
                or payload["config"]["federation"]["max_rounds"] != cfg.federation.max_rounds):
            raise ValueError("Baseline checkpoint identity/budget mismatch")
        if "optimizer_state" not in payload:
            raise ValueError("Legacy baseline checkpoint lacks optimizer state; cannot resume faithfully")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        best_state = payload["best_model_state_dict"]
        best_loss, best_round = payload["best_loss"], payload["best_round"]
        best_metrics = payload.get("best_metrics", {})
        history = list(payload["history"])
        early.load_state_dict(payload["early_stopping_state"])
        scheduler.load_state_dict(payload["lr_scheduler_state"])
        start_round = int(payload["round"]) + 1
        # Repair best/history from the atomic commit, including a crash during publication.
        save_checkpoint_atomic({"format_version": 2, "model_state_dict": best_state,
            "best_round": best_round, "best_loss": best_loss, "config": resolved,
            "metrics": best_metrics,
            "semantic_config_hash": cfg.semantic_config_hash}, run_dir / "best.pt")
        if history:
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        (run_dir / "rounds.jsonl").write_text("".join(
            json.dumps({"run_id": cfg.run_id, **row}) + "\n" for row in history), encoding="utf-8")
        epochs_path = run_dir / "client_epochs.jsonl"
        if epochs_path.exists():
            committed = []
            for line in epochs_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if int(row["round"]) < start_round:
                        committed.append(row)
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"Corrupt client epoch ledger: {epochs_path}") from exc
            epochs_path.write_text("".join(json.dumps(r) + "\n" for r in committed), encoding="utf-8")
        restore_rng_state(payload["rng_state"])
    else:
        phase(0, "validation")
        initial = evaluate_model(evaluation_model, val_loader, cfg.runtime.server_device, 38)
        best_metrics = initial
        early.init_round_0(initial["loss"])
        scheduler.init_round_0(initial["loss"])
        best_state = get_model_state_dict_cpu(model)
        best_loss, best_round = float(initial["loss"]), 0
        history, start_round = [], 1
        save_round_checkpoint(
            run_dir, 0, best_state, best_state, 0, best_loss, cfg.training.lr,
            early.state_dict(), scheduler.state_dict(), history, resolved, is_best=True,
            extra_metrics=initial, optimizer_state=optimizer.state_dict(),
        )
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(resolved, file, sort_keys=False)
    round_num = start_round - 1
    rounds = [] if early.stopped else range(start_round, cfg.federation.max_rounds + 1)
    for round_num in rounds:
        check_deadline()
        round_started = time.monotonic()
        phase(round_num, "training")
        for group in optimizer.param_groups:
            group["lr"] = scheduler.lr
        state, train_metrics = train_local(
            model, train_loader, optimizer, cfg.training.local_epochs,
            cfg.runtime.client_device, cfg.training.amp, round_num,
            on_epoch_end=(lambda row: (
                progress_logger.log_client_epoch({**row, "client_id": client_id}), progress.poll()
            )) if progress is not None and progress_logger is not None else None,
        )
        evaluation_model.load_state_dict(state)
        phase(round_num, "validation")
        val = evaluate_model(evaluation_model, val_loader, cfg.runtime.server_device, 38)
        should_stop = early.step(round_num, val["loss"])
        next_lr = scheduler.step(round_num, val["loss"])
        if early.is_last_step_best:
            best_state, best_loss, best_round = state, float(val["loss"]), round_num
            best_metrics = val
        row = {
            "round": round_num,
            "train_loss": train_metrics["loss-sum"] / train_metrics["processed-examples"],
            "train_accuracy": train_metrics["correct"] / train_metrics["processed-examples"],
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"], "lr": float(optimizer.param_groups[0]["lr"]),
            "next_lr": next_lr, "bad_rounds": early.bad_rounds,
            "is_best": early.is_last_step_best,
            "duration_seconds": time.monotonic() - round_started,
            "processed_examples": train_metrics["processed-examples"],
            "optimizer_steps": train_metrics["optimizer-steps"],
            "skipped_optimizer_steps": train_metrics["skipped-optimizer-steps"],
        }
        history.append(row)
        phase(round_num, "checkpoint")
        save_round_checkpoint(
            run_dir, round_num, state, best_state, best_round, best_loss, next_lr,
            early.state_dict(), scheduler.state_dict(), history, resolved,
            is_best=early.is_last_step_best, extra_metrics=val,
            checkpoint_every_n_rounds=cfg.checkpoint.every_n_rounds,
            keep_last_n=cfg.checkpoint.keep_last_n,
            optimizer_state=optimizer.state_dict(),
            best_metrics=best_metrics,
        )
        temp = run_dir / "history.csv.tmp"
        pd.DataFrame(history).to_csv(temp, index=False)
        os.replace(temp, run_dir / "history.csv")
        _append_jsonl(run_dir / "rounds.jsonl", {"run_id": cfg.run_id, **row})
        for epoch in train_metrics["epoch-metrics"]:
            _append_jsonl(run_dir / "client_epochs.jsonl", {
                "run_id": cfg.run_id, "client_id": client_id, **epoch,
            })
        if progress is not None and progress_logger is not None:
            offset = (client_id or 0) * cfg.federation.max_rounds
            progress_logger.log_round_completed(
                offset + round_num, val["loss"], best_loss, row["lr"], early.bad_rounds,
                cfg.early_stopping.patience_rounds,
            )
            progress.poll()
        if should_stop:
            break

    best_payload = load_checkpoint(run_dir / "best.pt")
    phase(round_num, "evaluate")
    evaluation_model.load_state_dict(best_payload["model_state_dict"])
    test = {}
    if cfg.output.evaluate_test_after_train:
        test = evaluate_model(evaluation_model, test_loader, cfg.runtime.server_device, 38)
        test["class_names"] = cfg.data.class_names
        for row in test["per_class"]:
            row["class_name"] = cfg.data.class_names[int(row["class_id"])]
        bind_evaluation(test, best_payload, run_dir / "best.pt",
                        cfg.data.partition_dir / "global_test.csv", cfg.data.protocol_fingerprint)
        write_evaluation(test, run_dir)
    save_inference_model(run_dir, best_state, resolved, best_round, best_loss)
    summary = {
        "schema_version": 2,
        "status": "completed" if cfg.output.evaluate_test_after_train else "calibration_completed",
        "artifact_scope": "main_evaluated" if cfg.output.evaluate_test_after_train else "calibration_no_test",
        "run_id": cfg.run_id,
        "mode": mode, "client_id": client_id, "last_round": history[-1]["round"] if history else 0,
        "completed_rounds": len(history), "best_round": best_round,
        "best_val_loss": best_loss, "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"), "protocol_fingerprint": cfg.data.protocol_fingerprint,
        "optimizer_state_policy": "persistent within this baseline run",
        "amp_scaler_policy": "reset_each_round; scaler state is not carried across train_local calls",
        "stop_reason": early.stop_reason if early.stopped else "max_rounds",
        "duration_seconds": sum(row["duration_seconds"] for row in history),
        "training_seed": cfg.training.seed,
        "scientific_stage2_complete": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    phase(round_num, "plot")
    if cfg.output.evaluate_test_after_train:
        generate_report(run_dir)
    return summary


def _validate_resume_config(cfg, directory, mode, client_id=None):
    old = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    if (old.get("semantic_config_hash") != cfg.semantic_config_hash
            or old.get("mode") != mode
            or old.get("client_id") != client_id
            or old["federation"]["max_rounds"] != cfg.federation.max_rounds):
        raise ValueError("Baseline resume config, mode, client or budget mismatch")


def run_baseline(config_path: str | Path, mode: str, resume_dir: str | Path | None = None) -> Path:
    if mode not in {"centralized", "local-only"}:
        raise ValueError("mode must be 'centralized' or 'local-only'")
    config_file = Path(config_path).resolve()
    cfg = load_training_config(config_file, base_dir=Path(__file__).resolve().parents[1], mode=mode)
    from .prepare import audit_manifest_directory
    audit_manifest_directory(cfg.data.partition_dir, require_all_classes=cfg.data.bundle_path is None, dataset_root=cfg.data.dataset_root)
    base = cfg.output.root.parent / ("centralized" if mode == "centralized" else "local_only") / cfg.run_id
    if resume_dir is not None:
        base = Path(resume_dir).resolve()
        _validate_resume_config(cfg, base, mode)
        cfg.run_id = base.name
    cfg.resume_baselines = resume_dir is not None
    base.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(base / "resolved_config.yaml")
    progress_logger = EventLogger(base / "events.jsonl", cfg.run_id)
    with_progress = ProgressRenderer(
        base / "events.jsonl", cfg.federation.max_rounds * (cfg.federation.num_clients if mode == "local-only" else 1),
        description=mode, enabled=cfg.output.progress,
    )
    try:
        return _run_baseline(cfg, mode, base, with_progress, progress_logger)
    except BaseException as exc:
        progress_logger.log_failed(0, f"{type(exc).__name__}: {exc}")
        atomic_json(base / "summary.json", {
            "status": "paused" if isinstance(exc, BudgetExhausted) else "failed",
            "run_id": cfg.run_id, "mode": mode, "error": str(exc),
            "scientific_stage2_complete": False,
        })
        raise
    finally:
        with_progress.close()


def _run_baseline(cfg, mode, base, progress, progress_logger):
    if mode == "centralized":
        loader = build_centralized_training_loader(
            cfg.data.partition_dir, cfg.data.dataset_root, cfg.training.batch_size,
            cfg.training.seed, cfg.training.num_workers,
            cfg.runtime.client_device == "cuda", cfg.data.feature_skew,
        )
        _train_one(cfg, base, loader, cfg.training.seed, mode,
                   progress=progress, progress_logger=progress_logger)
        return base

    summaries = []
    for client_id in range(cfg.federation.num_clients):
        loader = build_training_loader(
            get_client_manifest_path(cfg.data.partition_dir, client_id), cfg.data.dataset_root,
            cfg.training.batch_size, derive_seed(cfg.training.seed, "loader", client_id),
            cfg.training.num_workers, cfg.runtime.client_device == "cuda",
            client_id=client_id, feature_skew=cfg.data.feature_skew,
            partition_dir=cfg.data.partition_dir,
        )
        summaries.append(_train_one(
            cfg, base / f"client_{client_id:02d}", loader,
            cfg.training.seed, mode, client_id,
            progress, progress_logger,
        ))
    aggregate = {
        "schema_version": 2,
        "status": "completed" if cfg.output.evaluate_test_after_train else "calibration_completed",
        "artifact_scope": "main_evaluated" if cfg.output.evaluate_test_after_train else "calibration_no_test",
        "mode": mode,
        "run_id": cfg.run_id, "clients": summaries,
        "evaluation_scope": "each local-only model on the same global test; not site-domain fairness",
        "protocol_fingerprint": cfg.data.protocol_fingerprint,
        "training_seed": cfg.training.seed,
        "scientific_stage2_complete": False,
    }
    if cfg.output.evaluate_test_after_train:
        accuracies = np.asarray([row["test_accuracy"] for row in summaries], dtype=float)
        macro_f1 = np.asarray([row["test_macro_f1"] for row in summaries], dtype=float)
        aggregate.update({
            "test_accuracy_mean": float(accuracies.mean()),
            "test_accuracy_std": float(accuracies.std(ddof=0)),
            "test_accuracy_min": float(accuracies.min()),
            "test_accuracy_max": float(accuracies.max()),
            "test_macro_f1_mean": float(macro_f1.mean()),
            "test_macro_f1_std": float(macro_f1.std(ddof=0)),
            "test_macro_f1_min": float(macro_f1.min()),
            "test_macro_f1_max": float(macro_f1.max()),
            "client_model_dispersion_ddof": 0,
        })
    else:
        aggregate["evaluation_pending"] = True
    base.mkdir(parents=True, exist_ok=True)
    atomic_json(base / "summary.json", aggregate)
    return base

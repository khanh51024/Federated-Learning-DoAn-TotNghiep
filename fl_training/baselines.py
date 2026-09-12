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

from .checkpoint import load_checkpoint, save_inference_model, save_round_checkpoint
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
    phase(0, "validation")
    initial = evaluate_model(evaluation_model, val_loader, cfg.runtime.server_device, 38)
    early.init_round_0(initial["loss"])
    scheduler.init_round_0(initial["loss"])
    best_state = get_model_state_dict_cpu(model)
    best_loss, best_round = float(initial["loss"]), 0
    history = []
    resolved = cfg.to_dict()
    resolved["mode"] = mode
    resolved["client_id"] = client_id
    resolved["output"]["run_dir"] = str(run_dir)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(resolved, file, sort_keys=False)
    save_round_checkpoint(
        run_dir, 0, best_state, best_state, 0, best_loss, cfg.training.lr,
        early.state_dict(), scheduler.state_dict(), history, resolved, is_best=True,
        extra_metrics=initial,
    )

    for round_num in range(1, cfg.federation.max_rounds + 1):
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
    test = evaluate_model(evaluation_model, test_loader, cfg.runtime.server_device, 38)
    test["class_names"] = cfg.data.class_names
    for row in test["per_class"]:
        row["class_name"] = cfg.data.class_names[int(row["class_id"])]
    bind_evaluation(test, best_payload, run_dir / "best.pt",
                    cfg.data.partition_dir / "global_test.csv", cfg.data.protocol_fingerprint)
    write_evaluation(test, run_dir)
    save_inference_model(run_dir, best_state, resolved, best_round, best_loss)
    summary = {
        "schema_version": 2, "status": "completed", "run_id": cfg.run_id,
        "mode": mode, "client_id": client_id, "last_round": history[-1]["round"],
        "completed_rounds": len(history), "best_round": best_round,
        "best_val_loss": best_loss, "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"], "protocol_fingerprint": cfg.data.protocol_fingerprint,
        "optimizer_state_policy": "persistent within this baseline run",
        "stop_reason": early.stop_reason if early.stopped else "max_rounds",
        "duration_seconds": sum(row["duration_seconds"] for row in history),
        "training_seed": cfg.training.seed,
        "scientific_stage2_complete": False,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    phase(round_num, "plot")
    generate_report(run_dir)
    return summary


def run_baseline(config_path: str | Path, mode: str) -> Path:
    if mode not in {"centralized", "local-only"}:
        raise ValueError("mode must be 'centralized' or 'local-only'")
    config_file = Path(config_path).resolve()
    cfg = load_training_config(config_file, base_dir=Path(__file__).resolve().parents[1], mode=mode)
    from .prepare import audit_manifest_directory
    audit_manifest_directory(cfg.data.partition_dir, require_all_classes=cfg.data.bundle_path is None)
    base = cfg.output.root.parent / ("centralized" if mode == "centralized" else "local_only") / cfg.run_id
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
        (base / "summary.json").write_text(json.dumps({
            "status": "failed", "run_id": cfg.run_id, "mode": mode, "error": str(exc),
        }), encoding="utf-8")
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
    accuracies = np.asarray([row["test_accuracy"] for row in summaries], dtype=float)
    macro_f1 = np.asarray([row["test_macro_f1"] for row in summaries], dtype=float)
    aggregate = {
        "schema_version": 1, "status": "completed", "mode": mode,
        "run_id": cfg.run_id, "clients": summaries,
        "test_accuracy_mean": float(accuracies.mean()),
        "test_accuracy_std": float(accuracies.std()),
        "test_macro_f1_mean": float(macro_f1.mean()),
        "test_macro_f1_std": float(macro_f1.std()),
        "test_accuracy_min": float(accuracies.min()),
        "test_accuracy_max": float(accuracies.max()),
        "test_accuracy_gap_pp": float((accuracies.max() - accuracies.min()) * 100),
        "evaluation_scope": "each local-only model on the same global test; not site-domain fairness",
        "protocol_fingerprint": cfg.data.protocol_fingerprint,
        "training_seed": cfg.training.seed,
        "scientific_stage2_complete": False,
    }
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "summary.json", "w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)
    return base

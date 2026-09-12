"""Flower ServerApp orchestration for deterministic, validated FedAvg rounds."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, RecordDict
from flwr.common import Context, Message
from flwr.serverapp import Grid, ServerApp

from .checkpoint import (
    load_checkpoint,
    materialize_best_checkpoint,
    restore_rng_state,
    save_inference_model,
    save_round_checkpoint,
    verify_checkpoint_compatibility,
)
from .config import TrainingConfig, load_training_config
from .control import EarlyStoppingController, LRSchedulerController
from .data import build_evaluation_loader, get_client_sample_count
from .model import create_mobilenet_v3_small, get_model_state_dict_cpu, set_model_state_dict
from .progress import EventLogger
from .reproducibility import derive_seed, seed_everything
from .strategy import aggregate_fedavg_state_dicts, summarize_client_updates
from .task import evaluate_model

app = ServerApp()


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def _write_history(path: Path, history: List[Dict[str, Any]]) -> None:
    temp_path = path.with_suffix(".csv.tmp")
    pd.DataFrame(history).to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _discover_node_mapping(grid: Grid, node_ids: List[int], timeout: int) -> Dict[int, int]:
    messages = [
        Message(
            content=RecordDict({}), message_type="query", dst_node_id=node_id,
            group_id="identity",
        )
        for node_id in node_ids
    ]
    replies = list(grid.send_and_receive(messages, timeout=timeout))
    mapping: Dict[int, int] = {}
    for reply in replies:
        if reply.has_error():
            raise RuntimeError(f"Identity query failed for node {reply.metadata.src_node_id}: {reply.error}")
        source = int(reply.metadata.src_node_id)
        identity = reply.content.get("identity")
        if identity is None:
            raise ValueError(f"Identity query reply from node {source} has no identity record")
        if int(identity.get("node_id", source)) != source:
            raise ValueError(f"Identity query source mismatch for node {source}")
        client_id = int(identity["partition_id"])
        if client_id in mapping:
            raise ValueError(f"Duplicate partition/client id discovered: {client_id}")
        mapping[client_id] = source
    return mapping


def _parse_and_validate_replies(
    replies: List[Message],
    expected_node_to_client: Dict[int, int],
    expected_counts: Dict[int, int],
    expected_round: int,
    local_epochs: int,
    batch_size: int | None = None,
) -> List[Dict[str, Any]]:
    def integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    def finite(value: Any, name: str, minimum: float = 0.0, maximum: float = float("inf")) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
        return number

    parsed: List[Dict[str, Any]] = []
    seen_sources: set[int] = set()
    for reply in replies:
        source = int(reply.metadata.src_node_id)
        if reply.has_error():
            raise RuntimeError(f"Client node {source} returned an error: {reply.error}")
        if source not in expected_node_to_client:
            raise ValueError(f"Unexpected reply source node {source}")
        if source in seen_sources:
            raise ValueError(f"Duplicate reply from source node {source}")
        seen_sources.add(source)
        if str(reply.metadata.group_id) != str(expected_round):
            raise ValueError(
                f"Reply from node {source} has round/group {reply.metadata.group_id}, expected {expected_round}"
            )
        for record_name in ("arrays", "metrics", "identity"):
            if record_name not in reply.content:
                raise ValueError(f"Reply from node {source} is missing '{record_name}'")
        identity = reply.content["identity"]
        client_id = integer(identity.get("client_id", -1), "client_id")
        if client_id != expected_node_to_client[source]:
            raise ValueError(f"Node {source} claimed client_id={client_id}")
        if integer(identity.get("round", -1), "round") != expected_round:
            raise ValueError(f"Client {client_id} replied for the wrong round")
        metrics = reply.content["metrics"]
        n_k = integer(metrics.get("num-examples", -1), "num-examples")
        if n_k <= 0 or n_k != expected_counts[client_id] or integer(identity.get("n_k", -1), "n_k") != n_k:
            raise ValueError(
                f"Client {client_id} sample count {n_k} != manifest count {expected_counts[client_id]}"
            )
        processed = integer(metrics.get("processed-examples", -1), "processed-examples")
        if processed != n_k * local_epochs:
            raise ValueError(
                f"Client {client_id} processed {processed}, expected {n_k * local_epochs}"
            )
        state = {key: torch.from_numpy(value.numpy()) for key, value in reply.content["arrays"].items()}
        if integer(metrics.get("local-epochs", -1), "local-epochs") != local_epochs:
            raise ValueError("Reply local-epochs differs from requested local epochs")
        loss_sum = finite(metrics.get("loss-sum"), "loss-sum")
        correct = integer(metrics.get("correct"), "correct")
        steps = integer(metrics.get("optimizer-steps"), "optimizer-steps")
        skipped = integer(metrics.get("skipped-optimizer-steps", 0), "skipped-optimizer-steps")
        if not 0 <= correct <= processed or steps < 0 or skipped < 0 or steps + skipped <= 0:
            raise ValueError("Invalid total correct/optimizer steps")
        epoch_metrics = []
        for epoch in range(1, local_epochs + 1):
            required_keys = [
                f"epoch-{epoch}-loss", f"epoch-{epoch}-accuracy", f"epoch-{epoch}-examples",
                f"epoch-{epoch}-steps", f"epoch-{epoch}-lr", f"epoch-{epoch}-duration",
                f"epoch-{epoch}-started", f"epoch-{epoch}-ended",
            ]
            if any(key not in metrics for key in required_keys):
                raise ValueError(f"Client {client_id} is missing epoch {epoch} metrics")
            examples = integer(metrics[f"epoch-{epoch}-examples"], "epoch examples")
            epoch_steps = integer(metrics[f"epoch-{epoch}-steps"], "epoch steps")
            epoch_skipped = integer(metrics.get(f"epoch-{epoch}-skipped-steps", 0), "epoch skipped steps")
            if examples != n_k or epoch_steps < 0 or epoch_skipped < 0 or epoch_steps + epoch_skipped <= 0:
                raise ValueError("Epoch sample count or optimizer steps is invalid")
            if batch_size is not None and epoch_steps + epoch_skipped != math.ceil(n_k / batch_size):
                raise ValueError("Epoch optimizer steps differ from the requested batch size")
            for suffix in ("loss", "lr", "duration", "started", "ended"):
                finite(metrics[f"epoch-{epoch}-{suffix}"], f"epoch {suffix}")
            finite(metrics[f"epoch-{epoch}-accuracy"], "epoch accuracy", maximum=1.0)
            if metrics[f"epoch-{epoch}-ended"] < metrics[f"epoch-{epoch}-started"]:
                raise ValueError("Epoch ended before it started")
            epoch_metrics.append({
                "local_epoch": epoch,
                "loss": float(metrics[f"epoch-{epoch}-loss"]),
                "accuracy": float(metrics[f"epoch-{epoch}-accuracy"]),
                "processed_examples": int(metrics[f"epoch-{epoch}-examples"]),
                "optimizer_steps": int(metrics[f"epoch-{epoch}-steps"]),
                "skipped_optimizer_steps": epoch_skipped,
                "lr": float(metrics[f"epoch-{epoch}-lr"]),
                "duration_seconds": float(metrics[f"epoch-{epoch}-duration"]),
                "started_at_unix": float(metrics[f"epoch-{epoch}-started"]),
                "ended_at_unix": float(metrics[f"epoch-{epoch}-ended"]),
            })
        if not math.isclose(sum(e["loss"] * e["processed_examples"] for e in epoch_metrics), loss_sum, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Epoch loss sums differ from total loss-sum")
        if not math.isclose(sum(e["accuracy"] * e["processed_examples"] for e in epoch_metrics), correct, abs_tol=1e-6):
            raise ValueError("Epoch accuracies differ from total correct count")
        if sum(e["optimizer_steps"] for e in epoch_metrics) != steps:
            raise ValueError("Epoch steps differ from total optimizer-steps")
        if sum(e["skipped_optimizer_steps"] for e in epoch_metrics) != skipped:
            raise ValueError("Epoch skipped steps differ from total skipped-optimizer-steps")
        parsed.append({
            "client_id": client_id, "node_id": source, "round": expected_round,
            "num-examples": n_k, "state_dict": state,
            "loss-sum": loss_sum,
            "correct": correct, "processed-examples": processed,
            "optimizer-steps": steps,
            "skipped-optimizer-steps": skipped,
            "local-epochs": int(metrics.get("local-epochs", 0)),
            "duration-seconds": float(metrics.get("duration-seconds", 0.0)),
            "epoch_metrics": epoch_metrics,
        })
    missing = set(expected_node_to_client) - seen_sources
    if missing:
        raise RuntimeError(f"Missing required replies from node(s): {sorted(missing)}")
    return parsed


def _select_clients(cfg: TrainingConfig, round_num: int) -> List[int]:
    count = max(1, math.ceil(cfg.federation.num_clients * cfg.federation.fraction_train))
    ids = np.arange(cfg.federation.num_clients)
    if count == len(ids):
        return ids.tolist()
    rng = np.random.default_rng(derive_seed(cfg.training.seed, "participation", round_num))
    return sorted(int(value) for value in rng.choice(ids, size=count, replace=False))


def _max_observed_concurrency(intervals: List[tuple[float, float]]) -> int:
    events = [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals]
    active = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _run(grid: Grid, context: Context, cfg: TrainingConfig, run_dir: Path, logger: EventLogger) -> None:
    del context
    seed_everything(cfg.training.seed)
    model = create_mobilenet_v3_small(num_classes=38, weights=cfg.model.weights)
    es = EarlyStoppingController(
        enabled=cfg.early_stopping.enabled, min_delta=cfg.early_stopping.min_delta,
        patience_rounds=cfg.early_stopping.patience_rounds,
        warmup_rounds=cfg.early_stopping.warmup_rounds,
    )
    ls = LRSchedulerController(
        enabled=cfg.lr_scheduler.enabled, initial_lr=cfg.training.lr,
        factor=cfg.lr_scheduler.factor, patience_rounds=cfg.lr_scheduler.patience_rounds,
        threshold=cfg.lr_scheduler.threshold, min_lr=cfg.lr_scheduler.min_lr,
    )

    history: List[Dict[str, Any]] = []
    start_round, current_lr, best_round, best_loss = 1, cfg.training.lr, 0, float("inf")
    best_model_state = get_model_state_dict_cpu(model)
    parent_run_id = None
    epoch_intervals: List[tuple[float, float]] = []
    epoch_seq = weight_seq = round_seq = 0
    attempt_id = int(os.environ.get("FL_TRAINING_ATTEMPT_ID", "1"))
    resume_payload = None
    resume_from_raw = os.environ.get("FL_TRAINING_RESUME_CHECKPOINT")
    if resume_from_raw:
        resume_payload = load_checkpoint(resume_from_raw)
        if resume_payload.get("artifact_type") == "inference_model" or resume_payload.get("config", {}).get("mode") in {"centralized", "local-only"}:
            raise ValueError("FedAvg resume requires a FedAvg training checkpoint, not a baseline/inference artifact")
        verify_checkpoint_compatibility(resume_payload, cfg.to_dict())
        if resume_payload.get("early_stopping_state", {}).get("stopped") and not os.environ.get("FL_TRAINING_FORCE_RESUME_STOPPED"):
            raise RuntimeError("Checkpoint is already early-stopped; pass --force-resume-stopped to continue explicitly")
        set_model_state_dict(model, resume_payload["model_state_dict"], strict=True)
        best_model_state = resume_payload["best_model_state_dict"]
        start_round = int(resume_payload["round"]) + 1
        current_lr = float(resume_payload["next_lr"])
        best_round = int(resume_payload["best_round"])
        best_loss = float(resume_payload["best_loss"])
        es.load_state_dict(resume_payload["early_stopping_state"])
        if es.stopped:
            es.stopped, es.stop_reason, es.bad_rounds = False, None, 0
        ls.load_state_dict(resume_payload["lr_scheduler_state"])
        history = list(resume_payload.get("history", []))
        parent_run_id = resume_payload.get("config", {}).get("run_id")
        materialize_best_checkpoint(resume_payload, run_dir)
        _write_history(run_dir / "history.csv", history)
        # A continuation with no new rounds is still a valid, resumable run.
        from .checkpoint import save_checkpoint_atomic
        continuation = {**resume_payload, "config": cfg.to_dict(), "parent_run_id": parent_run_id,
                        "semantic_config_hash": cfg.semantic_config_hash,
                        "early_stopping_state": es.state_dict()}
        save_checkpoint_atomic(continuation, run_dir / "last.pt")

    val_loader = build_evaluation_loader(
        cfg.data.partition_dir / "global_val.csv", cfg.data.dataset_root,
        batch_size=cfg.training.eval_batch_size, num_workers=cfg.training.num_workers,
    )
    if resume_payload and resume_payload.get("rng_state"):
        restore_rng_state(resume_payload["rng_state"])

    if start_round == 1:
        logger.log_phase(0, "validation")
        initial = evaluate_model(model, val_loader, cfg.runtime.server_device, 38)
        best_loss, best_round = float(initial["loss"]), 0
        best_model_state = get_model_state_dict_cpu(model)
        es.init_round_0(best_loss)
        ls.init_round_0(best_loss)
        save_round_checkpoint(
            run_dir, 0, best_model_state, best_model_state, 0, best_loss, current_lr,
            es.state_dict(), ls.state_dict(), history, cfg.to_dict(), is_best=True,
            extra_metrics=initial, parent_run_id=parent_run_id,
        )
        logger.log_round_completed(0, best_loss, best_loss, current_lr, 0,
                                   cfg.early_stopping.patience_rounds, is_best=True)

    node_ids = list(grid.get_node_ids())
    logger.log_phase(start_round - 1, "discover clients")
    mapping = _discover_node_mapping(grid, node_ids, cfg.federation.timeout_seconds)
    required_ids = set(range(cfg.federation.num_clients))
    if not required_ids.issubset(mapping):
        raise RuntimeError(f"Missing client partitions: {sorted(required_ids - set(mapping))}")
    expected_counts = {
        client_id: get_client_sample_count(cfg.data.partition_dir, client_id)
        for client_id in required_ids
    }

    for round_num in range(start_round, cfg.federation.max_rounds + 1):
        round_t0 = time.time()
        logger.log_phase(round_num, "client train")
        current_state = get_model_state_dict_cpu(model)
        arrays = ArrayRecord({key: Array(value.numpy()) for key, value in current_state.items()})
        selected_clients = _select_clients(cfg, round_num)
        selected_node_to_client = {mapping[cid]: cid for cid in selected_clients}
        messages = []
        for client_id in selected_clients:
            train_cfg = {
                "round": round_num, "local_epochs": cfg.training.local_epochs,
                "lr": current_lr, "momentum": cfg.training.momentum,
                "weight_decay": cfg.training.weight_decay, "seed": cfg.training.seed,
                "device": cfg.runtime.client_device, "amp": cfg.training.amp,
                "batch_size": cfg.training.batch_size, "num_workers": cfg.training.num_workers,
                "feature_skew": cfg.data.feature_skew,
                "partition_dir": str(cfg.data.partition_dir),
                "dataset_root": str(cfg.data.dataset_root), "client_id": client_id,
                "run_dir": str(run_dir), "run_id": cfg.run_id,
                "attempt_id": int(os.environ.get("FL_TRAINING_ATTEMPT_ID", "1")),
            }
            messages.append(Message(
                content=RecordDict({"arrays": arrays, "config": ConfigRecord(train_cfg)}),
                message_type="train", dst_node_id=mapping[client_id], group_id=str(round_num),
            ))
        replies = list(grid.send_and_receive(messages, timeout=cfg.federation.timeout_seconds))
        parsed = _parse_and_validate_replies(
            replies, selected_node_to_client, expected_counts, round_num, cfg.training.local_epochs,
            cfg.training.batch_size,
        )

        logger.log_phase(round_num, "aggregate")
        shapes = {key: tuple(value.shape) for key, value in current_state.items()}
        dtypes = {key: value.dtype for key, value in current_state.items()}
        new_state, train_metrics = aggregate_fedavg_state_dicts(parsed, shapes, dtypes)
        weight_rows = summarize_client_updates(current_state, parsed)
        total_n = sum(item["num-examples"] for item in parsed)
        for item in weight_rows:
            weight_seq += 1
            item.update({
                "schema_version": 1, "run_id": cfg.run_id, "round": round_num,
                "aggregation_weight": item["n_k"] / total_n,
                "attempt_id": attempt_id, "seq": weight_seq,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "phase": "aggregate",
            })
        epoch_rows = []
        for reply in parsed:
            for epoch in reply["epoch_metrics"]:
                epoch_intervals.append((epoch["started_at_unix"], epoch["ended_at_unix"]))
                epoch_seq += 1
                epoch_rows.append({
                    "schema_version": 1, "run_id": cfg.run_id, "round": round_num,
                    "client_id": reply["client_id"], "attempt_id": attempt_id,
                    "seq": epoch_seq, "phase": "client_train",
                    "time_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch["ended_at_unix"])
                    ), **epoch,
                })

        set_model_state_dict(model, new_state, strict=True)
        logger.log_phase(round_num, "validation")
        val = evaluate_model(model, val_loader, cfg.runtime.server_device, 38)
        val_loss = float(val["loss"])
        should_stop = es.step(round_num, val_loss)
        is_best = es.is_last_step_best
        next_lr = ls.step(round_num, val_loss)
        if is_best:
            best_model_state, best_round, best_loss = get_model_state_dict_cpu(model), round_num, val_loss

        record = {
            "round": round_num, "train_loss": train_metrics["train-loss"],
            "train_accuracy": train_metrics["train-accuracy"], "val_loss": val_loss,
            "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "lr": current_lr, "next_lr": next_lr, "bad_rounds": es.bad_rounds,
            "is_best": is_best, "duration_seconds": time.time() - round_t0,
            "participating_clients": len(parsed),
            "processed_examples": train_metrics["processed-examples"],
            "optimizer_steps": train_metrics["optimizer-steps"],
            "skipped_optimizer_steps": train_metrics["skipped-optimizer-steps"],
            "payload_bytes": sum(row["tensor_bytes"] for row in weight_rows) * 2,
        }
        history.append(record)
        round_seq += 1
        round_event = {
            "schema_version": 1, "run_id": cfg.run_id, "attempt_id": attempt_id,
            "seq": round_seq, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": "round_commit", **record,
        }
        logger.log_phase(round_num, "checkpoint")
        last_path, best_path = save_round_checkpoint(
            run_dir, round_num, new_state, best_model_state, best_round, best_loss,
            next_lr, es.state_dict(), ls.state_dict(), history, cfg.to_dict(),
            is_best=is_best, extra_metrics=val,
            checkpoint_every_n_rounds=cfg.checkpoint.every_n_rounds,
            keep_last_n=cfg.checkpoint.keep_last_n, parent_run_id=parent_run_id,
            attempt_id=attempt_id,
        )
        # last.pt is the commit point. Never publish a completed round first.
        _append_jsonl(run_dir / "rounds.jsonl", round_event)
        for item in weight_rows:
            _append_jsonl(run_dir / "weights.jsonl", item)
        for item in epoch_rows:
            _append_jsonl(run_dir / "client_epochs.jsonl", item)
        _write_history(run_dir / "history.csv", history)
        logger.log_checkpoint(round_num, {
            "last": str(last_path), "best": str(best_path) if best_path else "",
        })
        current_lr = next_lr
        logger.log_round_completed(
            round_num, val_loss, best_loss, record["lr"], es.bad_rounds,
            cfg.early_stopping.patience_rounds, train_metrics["train-loss"],
            train_metrics["train-accuracy"], record["duration_seconds"], is_best,
        )
        if should_stop:
            logger.log_stopped(round_num, es.stop_reason or "Early stopped")
            break

    save_inference_model(run_dir, best_model_state, cfg.to_dict(), best_round, best_loss)
    summary = {
        "schema_version": 2, "status": "completed", "run_id": cfg.run_id,
        "parent_run_id": parent_run_id, "completed_rounds": len(history),
        "last_round": int(history[-1]["round"]) if history else start_round - 1,
        "best_round": best_round, "best_val_loss": best_loss,
        "stop_reason": es.stop_reason if es.stopped else "max_rounds",
        "semantic_config_hash": cfg.semantic_config_hash,
        "protocol_fingerprint": cfg.data.protocol_fingerprint,
        "duration_seconds": sum(row["duration_seconds"] for row in history),
        "estimated_payload_bytes": sum(row.get("payload_bytes", 0) for row in history),
        "training_seed": cfg.training.seed,
        "runtime_concurrency": {
            "configured_max": cfg.runtime.max_concurrent_clients,
            "observed_max_overlapping_local_epochs": _max_observed_concurrency(epoch_intervals),
            "method": "maximum overlap of client-reported local-epoch wall-clock intervals",
        },
        "scientific_stage2_complete": False,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


@app.main()
def main(grid: Grid, context: Context) -> None:
    config_path = os.environ.get("FL_TRAINING_CONFIG_PATH") or str(
        context.run_config.get("config_path", "configs/train_smoke.yaml")
    )
    pkg_root = Path(__file__).resolve().parents[1]
    cfg = load_training_config(
        config_path, base_dir=pkg_root,
        run_id_override=os.environ.get("FL_TRAINING_RUN_ID"),
    )
    run_dir = Path(os.environ.get("FL_TRAINING_RUN_DIR", str(cfg.output.run_dir))).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(
        run_dir / "events.jsonl", cfg.run_id,
        attempt_id=int(os.environ.get("FL_TRAINING_ATTEMPT_ID", "1")),
    )
    (run_dir / "server_started.json").write_text(
        json.dumps({"run_id": cfg.run_id, "pid": os.getpid()}), encoding="utf-8",
    )
    try:
        _run(grid, context, cfg, run_dir, logger)
    except BaseException as exc:
        logger.log_failed(0, f"{type(exc).__name__}: {exc}")
        with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump({
                "schema_version": 2, "status": "failed", "run_id": cfg.run_id,
                "error_type": type(exc).__name__, "error": str(exc),
                "scientific_stage2_complete": False,
            }, file, indent=2)
        raise

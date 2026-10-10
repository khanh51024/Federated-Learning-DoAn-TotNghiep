"""
fl_training.client_app: Flower ClientApp handling train and query messages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Context, Message

from .budget import check_deadline
from .data import build_training_loader, get_client_manifest_path, get_client_sample_count
from .model import create_mobilenet_v3_small, set_model_state_dict
from .progress import EventLogger
from .reproducibility import derive_seed, seed_everything
from .task import train_local

app = ClientApp()


def _apply_deadline_config(config: Any) -> None:
    """Install runner deadlines passed explicitly in the Flower train message."""
    for field, environment_name in (
        ("soft_deadline_unix", "FL_TRAINING_SOFT_DEADLINE_UNIX"),
        ("hard_deadline_unix", "FL_TRAINING_HARD_DEADLINE_UNIX"),
    ):
        value = config.get(field, "")
        if value is None or str(value).strip() == "":
            os.environ.pop(environment_name, None)
        else:
            os.environ[environment_name] = str(value)
    check_deadline()


@app.query()
def query(msg: Message, context: Context) -> Message:
    """
    Lightweight node query to establish node_id -> partition_id mapping.
    Does not load images or initialize neural network models.
    """
    node_id = context.node_id
    partition_id = int(context.node_config.get("partition-id", node_id))

    identity_record = ConfigRecord({
        "node_id": node_id,
        "partition_id": partition_id,
        "status": "ready",
    })
    reply_dict = RecordDict({"identity": identity_record})
    return Message(content=reply_dict, reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Client training handler:
    1. Unpack global model weights from ArrayRecord.
    2. Extract round configuration (epochs, lr, seed, paths).
    3. Train locally with fresh SGD optimizer and set_epoch.
    4. Return updated parameters and metrics weighted by n_k.
    """
    cfg_record = msg.content["config"]
    _apply_deadline_config(cfg_record)

    round_num = int(cfg_record.get("round", 1))
    local_epochs = int(cfg_record.get("local_epochs", 1))
    lr = float(cfg_record.get("lr", 0.01))
    momentum = float(cfg_record.get("momentum", 0.9))
    weight_decay = float(cfg_record.get("weight_decay", 0.0001))
    training_seed = int(cfg_record.get("seed", 42))
    client_device = str(cfg_record.get("device", "cpu"))
    amp = bool(cfg_record.get("amp", False))
    batch_size = int(cfg_record.get("batch_size", 16))
    num_workers = int(cfg_record.get("num_workers", 0))
    feature_skew = str(cfg_record.get("feature_skew", "none"))

    partition_dir = Path(str(cfg_record["partition_dir"]))
    dataset_root = Path(str(cfg_record["dataset_root"]))

    # Stable client ID determination
    if "client_id" in cfg_record:
        client_id = int(cfg_record["client_id"])
    elif "partition-id" in context.node_config:
        client_id = int(context.node_config["partition-id"])
    else:
        client_id = int(context.node_id)

    manifest_path = get_client_manifest_path(partition_dir, client_id)
    n_k = get_client_sample_count(partition_dir, client_id)

    client_seed = derive_seed(training_seed, "client", client_id, "round", round_num)
    seed_everything(client_seed)

    # 1. Instantiate model and set incoming weights
    model = create_mobilenet_v3_small(num_classes=38, weights=None)

    incoming_arrays = msg.content["arrays"]
    state_dict_to_load: Dict[str, torch.Tensor] = {}
    for k, v in incoming_arrays.items():
        state_dict_to_load[k] = torch.from_numpy(v.numpy())

    set_model_state_dict(model, state_dict_to_load, strict=True)

    # 2. Build local training DataLoader
    loader_seed = derive_seed(training_seed, "loader", client_id)
    train_loader = build_training_loader(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        batch_size=batch_size,
        seed=loader_seed,
        num_workers=num_workers,
        pin_memory=client_device == "cuda",
        client_id=client_id,
        feature_skew=feature_skew,
        partition_dir=partition_dir,
    )

    # 3. Create fresh SGD optimizer
    model.to(client_device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # 4. Perform local training
    epoch_logger = None
    run_dir_raw = cfg_record.get("run_dir")
    if run_dir_raw:
        epoch_logger = EventLogger(
            Path(str(run_dir_raw)) / "client_logs" / f"client_{client_id:02d}.jsonl",
            run_id=str(cfg_record.get("run_id", "unknown")),
            attempt_id=int(cfg_record.get("attempt_id", 1)),
            client_id=client_id,
        )

    state_dict_cpu, metrics = train_local(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        epochs=local_epochs,
        device=client_device,
        amp=amp,
        round_num=round_num,
        on_epoch_end=epoch_logger.log_client_epoch if epoch_logger else None,
    )

    # 5. Pack reply message
    # Arrays
    array_dict: Dict[str, Array] = {}
    for k, t in state_dict_cpu.items():
        array_dict[k] = Array(t.numpy())

    # Metrics
    reply_metrics = {
        "num-examples": int(n_k),
        "loss-sum": float(metrics["loss-sum"]),
        "correct": int(metrics["correct"]),
        "processed-examples": int(metrics["processed-examples"]),
        "optimizer-steps": int(metrics["optimizer-steps"]),
        "skipped-optimizer-steps": int(metrics["skipped-optimizer-steps"]),
        "local-epochs": int(metrics["local-epochs"]),
        "duration-seconds": float(metrics["duration-seconds"]),
    }
    for epoch in metrics["epoch-metrics"]:
        number = int(epoch["local_epoch"])
        reply_metrics.update({
            f"epoch-{number}-loss": float(epoch["loss"]),
            f"epoch-{number}-accuracy": float(epoch["accuracy"]),
            f"epoch-{number}-examples": int(epoch["processed_examples"]),
            f"epoch-{number}-steps": int(epoch["optimizer_steps"]),
            f"epoch-{number}-skipped-steps": int(epoch["skipped_optimizer_steps"]),
            f"epoch-{number}-lr": float(epoch["lr"]),
            f"epoch-{number}-duration": float(epoch["duration_seconds"]),
            f"epoch-{number}-started": float(epoch["started_at_unix"]),
            f"epoch-{number}-ended": float(epoch["ended_at_unix"]),
        })
    metrics_record = MetricRecord(reply_metrics)

    # Identity
    identity_record = ConfigRecord({
        "client_id": client_id,
        "round": round_num,
        "n_k": n_k,
        "node_id": int(context.node_id),
    })

    reply_content = RecordDict({
        "arrays": ArrayRecord(array_dict),
        "metrics": metrics_record,
        "identity": identity_record,
    })

    return Message(content=reply_content, reply_to=msg)

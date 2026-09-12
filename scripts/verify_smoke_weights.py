"""Replay a small CPU round and compare every aggregated tensor independently.

Usage: python -m scripts.verify_smoke_weights --config <small-config> --run-dir <run>
Only for a weights=null, CPU, full-participation first round with <=256 train images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, RecordDict
from flwr.common import Message

from fl_training.checkpoint import load_checkpoint, model_state_sha256
from fl_training.client_app import train
from fl_training.config import load_training_config
from fl_training.data import get_client_sample_count
from fl_training.model import create_mobilenet_v3_small
from fl_training.reproducibility import seed_everything


def verify(config_path: Path, run_dir: Path):
    cfg = load_training_config(config_path)
    counts = [get_client_sample_count(cfg.data.partition_dir, cid) for cid in range(cfg.federation.num_clients)]
    if sum(counts) > 256 or cfg.model.weights is not None or cfg.runtime.client_device != "cpu" or cfg.federation.fraction_train != 1:
        raise ValueError("This verifier is bounded to small CPU full-participation runs initialized without pretrained weights")
    seed_everything(cfg.training.seed)
    initial = create_mobilenet_v3_small(38, None).state_dict()
    replies = []
    for cid in range(cfg.federation.num_clients):
        local_config = {"round": 1, "local_epochs": cfg.training.local_epochs, "lr": cfg.training.lr,
                        "momentum": cfg.training.momentum, "weight_decay": cfg.training.weight_decay,
                        "seed": cfg.training.seed, "device": "cpu", "amp": False,
                        "batch_size": cfg.training.batch_size, "num_workers": 0,
                        "partition_dir": str(cfg.data.partition_dir), "dataset_root": str(cfg.data.dataset_root),
                        "client_id": cid, "feature_skew": cfg.data.feature_skew}
        message = Message(content=RecordDict({
            "arrays": ArrayRecord({key: Array(value.numpy()) for key, value in initial.items()}),
            "config": ConfigRecord(local_config),
        }), dst_node_id=cid + 1, message_type="train", group_id="1")
        reply = train(message, SimpleNamespace(node_id=cid + 1, node_config={"partition-id": cid}))
        assert reply.content["metrics"]["num-examples"] == counts[cid]
        replies.append({key: value.numpy() for key, value in reply.content["arrays"].items()})
    checkpoint = load_checkpoint(run_dir / "round_0001.pt")
    if checkpoint["config"]["data"]["protocol_fingerprint"] != cfg.data.protocol_fingerprint:
        raise ValueError("Verifier data differs from the recorded run")
    actual = checkpoint["model_state_dict"]
    assert set(actual) == set(initial)
    maximum = 0.
    for key, tensor in actual.items():
        if tensor.is_floating_point():
            expected = sum(state[key].astype(np.float64) * n for state, n in zip(replies, counts)) / sum(counts)
            np.testing.assert_allclose(tensor.numpy(), expected, rtol=2e-6, atol=5e-7, err_msg=key)
            maximum = max(maximum, float(np.max(np.abs(tensor.numpy() - expected))))
        else:
            np.testing.assert_array_equal(tensor.numpy(), np.maximum.reduce([state[key] for state in replies]))
    result = {"round": 1, "tensor_entries": len(actual), "client_counts": counts,
              "aggregation_weights": [n / sum(counts) for n in counts], "max_abs_error": maximum,
              "checkpoint_model_sha256": model_state_sha256(actual),
              "method": "replay real client SGD then NumPy float64 weighted mean; BN counters use max"}
    (run_dir / "independent_weight_verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.run_dir), indent=2))

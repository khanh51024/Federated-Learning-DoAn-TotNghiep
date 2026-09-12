"""
Tests for Gate D: Flower Runtime Integration and Launcher Progress Renderer.
"""

import json
import os
import shutil
from pathlib import Path
import pytest
import torch

from fl_training.config import load_training_config
from fl_training.prepare import create_smoke_bundle
from fl_training.progress import EventLogger, ProgressRenderer

ROOT = Path(__file__).resolve().parents[1]


def test_gate_d_smoke_bundle_creation(tmp_path):
    src_partition = ROOT / "data" / "partitions_train_v1" / "label_skew" / "label_skew__seed_42__alpha_0_1__feature_none__b3b5ed03170d"
    bundle_dir = tmp_path / "smoke_bundle"

    create_smoke_bundle(
        source_partition_dir=src_partition,
        bundle_dir=bundle_dir,
        target_client_images=32,
        target_val_images=64,
        seed=42,
    )

    assert (bundle_dir / "clients" / "client_00.csv").exists()
    assert (bundle_dir / "clients" / "client_01.csv").exists()
    assert (bundle_dir / "global_val.csv").exists()
    assert (bundle_dir / "centralized_train.csv").exists()
    assert (bundle_dir / "fedavg_meta.json").exists()

    with open(bundle_dir / "fedavg_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["num_clients"] == 2
    assert meta["smoke"] is True
    assert len(meta["class_names"]) == 38
    assert set(meta["client_sample_counts"]) == set(meta["client_aggregation_weights"])
    assert sum(meta["client_aggregation_weights"].values()) == pytest.approx(1.0)


def test_gate_d_progress_renderer_non_tty(tmp_path):
    events_file = tmp_path / "events.jsonl"
    logger = EventLogger(events_file, run_id="test_run", attempt_id=1)

    renderer = ProgressRenderer(
        event_log_path=events_file,
        max_rounds=5,
        description="Test Progress",
        enabled=True,
        is_tty=False,
    )

    logger.log_phase(0, "validation round 0")
    logger.log_round_completed(0, val_loss=3.5, best_loss=3.5, lr=0.01, bad_rounds=0, patience=10, is_best=True)
    status = renderer.poll()
    assert status is None
    assert renderer.completed_rounds == 0

    logger.log_phase(1, "client train")
    logger.log_round_completed(1, val_loss=3.4, best_loss=3.4, lr=0.01, bad_rounds=0, patience=10, train_loss=3.45, train_acc=0.1, duration=1.2, is_best=True)
    status = renderer.poll()
    assert status is None
    assert renderer.completed_rounds == 1

    logger.log_stopped(1, "Early stopping test")
    status = renderer.poll()
    assert status == "stopped"

    renderer.close()


def test_gate_d_flower_simulation_integration_smoke(tmp_path):
    """
    Real 2-client 2-round Flower integration smoke test.
    """
    from flwr.simulation import run_simulation
    from fl_training.client_app import app as client_app
    from fl_training.server_app import app as server_app

    run_dir = tmp_path / "smoke_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure smoke bundle exists
    bundle_dir = ROOT / "data" / "partitions_train_v1" / "smoke_bundle_2c"
    if not bundle_dir.exists():
        src_partition = ROOT / "data" / "partitions_train_v1" / "label_skew" / "label_skew__seed_42__alpha_0_1__feature_none__b3b5ed03170d"
        create_smoke_bundle(src_partition, bundle_dir)

    os.environ["FL_TRAINING_CONFIG_PATH"] = str(ROOT / "configs" / "train_smoke.yaml")
    os.environ["FL_TRAINING_RUN_DIR"] = str(run_dir)

    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": 0},
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=2,
        backend_name="ray",
        backend_config=backend_config,
    )

    assert (run_dir / "last.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "summary.json").exists()

    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["completed_rounds"] == 2

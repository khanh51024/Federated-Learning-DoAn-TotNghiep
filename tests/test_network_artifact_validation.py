from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.model.mobilenetv3 import compute_model_hash


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_network_artifact.py"


def _write_valid_artifacts(tmp_path: Path):
    state = {
        "layer.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "counter": torch.tensor(0, dtype=torch.int64),
    }
    model_hash = compute_model_hash(state)
    checkpoint = tmp_path / "model.pt"
    metrics = tmp_path / "metrics.json"
    torch.save(
        {
            "model_state": state,
            "source": "flower_network_fedavg",
            "rounds": 1,
            "num_clients": 2,
            "local_epochs": 1,
            "seed": 42,
            "model_hash": model_hash,
            "test_metrics": {"accuracy": 0.5, "macro_f1": 0.4, "loss": 1.0},
        },
        checkpoint,
    )
    metrics.write_text(
        json.dumps(
            {
                "server_address": "127.0.0.1:9999",
                "num_clients_required": 2,
                "require_real_clients": True,
                "rounds": 1,
                "seed": 42,
                "evaluation_dataset_root": "D:/PlantVillage",
                "evaluation_partition_dir": "data/partitions_stage2_scratch_v3/label100",
                "require_real_evaluation": True,
                "final_test_metrics": {"accuracy": 0.5, "macro_f1": 0.4, "loss": 1.0},
                "checkpoint_file": str(checkpoint),
                "audit_rounds": [
                    {
                        "round": 1,
                        "clients": [
                            {"client_id": 0, "data_mode": "real", "status": "ok"},
                            {"client_id": 1, "data_mode": "real", "status": "ok"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return metrics, checkpoint


def _run(metrics: Path, checkpoint: Path, output: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics",
            str(metrics),
            "--checkpoint",
            str(checkpoint),
            "--expected-clients",
            "2",
            "--expected-rounds",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_valid_network_artifact_passes(tmp_path: Path):
    metrics, checkpoint = _write_valid_artifacts(tmp_path)
    output = tmp_path / "validation.json"
    proc = _run(metrics, checkpoint, output)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["checks"]["checkpoint_model_hash_matches"] is True


def test_tampered_model_state_fails_hash_check(tmp_path: Path):
    metrics, checkpoint = _write_valid_artifacts(tmp_path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["model_state"]["layer.weight"][0, 0] += 9.0
    # Keep the original model_hash deliberately unchanged.
    torch.save(ckpt, checkpoint)

    output = tmp_path / "validation.json"
    proc = _run(metrics, checkpoint, output)
    assert proc.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["checks"]["checkpoint_model_hash_matches"] is False

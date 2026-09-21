"""
Tests for Gate E: CLI Commands (preflight, prepare-data, smoke, evaluate, resume).
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_gate_e_preflight_cli():
    cmd = [
        sys.executable,
        "-m",
        "fl_training.cli",
        "preflight",
        "--config",
        str(ROOT / "configs" / "train_fedavg.yaml"),
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"Preflight failed:\n{res.stderr}\n{res.stdout}"
    assert "PREFLIGHT STATUS: READY FOR SIMULATION / TRAINING" in res.stdout


def test_gate_e_smoke_and_evaluate_cli(tmp_path):
    raw = yaml.safe_load((ROOT / 'configs/train_smoke.yaml').read_text())
    raw['output']['root'] = str(tmp_path / 'runs/fedavg')
    raw['data']['bundle_path'] = str(tmp_path / 'new_smoke_bundle')
    config = tmp_path / 'train.yaml'
    config.write_text(yaml.safe_dump(raw))
    # 1. Run smoke
    smoke_cmd = [
        sys.executable,
        "-m",
        "fl_training.cli",
        "smoke",
        "--config",
        str(config),
        "--no-progress",
    ]
    res_smoke = subprocess.run(smoke_cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert res_smoke.returncode == 0, f"Smoke CLI failed:\n{res_smoke.stderr}\n{res_smoke.stdout}"
    assert "Simulation completed successfully" in res_smoke.stdout

    # Find the newly created run dir
    runs_dir = tmp_path / "runs" / "fedavg"
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    assert len(run_dirs) > 0
    latest_run = run_dirs[0]

    best_pt = latest_run / "best.pt"
    assert best_pt.exists()

    # 2. Run evaluate
    eval_out = tmp_path / "eval_metrics"
    eval_out.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        sys.executable,
        "-m",
        "fl_training.cli",
        "evaluate",
        "--checkpoint",
        str(best_pt),
        "--output",
        str(eval_out),
        "--device",
        "cpu",
    ]
    res_eval = subprocess.run(eval_cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert res_eval.returncode == 0, f"Evaluate CLI failed:\n{res_eval.stderr}\n{res_eval.stdout}"
    assert (eval_out / "test_metrics.json").exists()
    assert (eval_out / "confusion_matrix.csv").exists()

    with open(eval_out / "test_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics

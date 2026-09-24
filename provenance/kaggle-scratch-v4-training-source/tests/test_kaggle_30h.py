from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from fl_training import baselines
from fl_training.budget import BudgetExhausted, check_deadline
from fl_training.budget_state import (
    BudgetSession, apply_usage_observations, choose_rounds, quota_remaining_from_state,
)
from fl_training.checkpoint import (
    load_checkpoint, model_state_sha256, save_checkpoint_atomic, verify_checkpoint_compatibility,
)
from fl_training.client_app import _apply_deadline_config
from fl_training.kaggle_budget import acquire_lock, collect_jobs, main as budget_main, release_lock
from fl_training.sweep import _result_row, _write_comparison
import fl_training.kaggle_budget as budget_runner


class FakeClock:
    def __init__(self, monotonic=0.0, wall=1000.0):
        self.monotonic = monotonic
        self.wall = wall

    def mono(self):
        return self.monotonic

    def unix(self):
        return self.wall

    def advance(self, seconds):
        self.monotonic += seconds
        self.wall += seconds


def test_budget_round_selection_and_invalid_inputs():
    timing = {
        "centralized": {"seconds_per_round": 10, "fixed_seconds": 5},
        "local-only": {"seconds_per_round": 100, "fixed_seconds": 50},
        "fedavg": {"seconds_per_round": 20, "fixed_seconds": 10},
    }
    modes = [mode for _ in range(6) for mode in timing]
    rounds, estimate = choose_rounds(timing, modes, 12285, 10, 20, 1.5)
    assert rounds == 10
    assert estimate == pytest.approx(12285)
    with pytest.raises(RuntimeError, match="cannot fit"):
        choose_rounds(timing, modes, 12284, 10, 20, 1.5)
    for bad in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            choose_rounds(timing, modes, bad, 10, 20, 1.5)
    with pytest.raises(ValueError):
        choose_rounds(timing, modes, 99999, 3, 20, 1.5)


def test_budget_ledger_preserves_usage_lease_and_quota():
    clock = FakeClock()
    state = {"used_seconds": 7200.0}
    saved = []
    session = BudgetSession(
        state, lambda: saved.append(dict(state)), 30, 3, 8, 15,
        clock=clock.mono, wall_clock=clock.unix, quota_remaining_hours=10,
    )
    assert session.total_remaining() == pytest.approx(7 * 3600)
    session.begin_lease(100)
    clock.advance(20)
    session.tick()
    assert state["used_seconds"] == pytest.approx(7220)
    session.end_lease()
    used = state["used_seconds"]

    interrupted = {
        "used_seconds": used,
        "active_lease": {"lease_id": "old", "reserved_seconds": 100,
                         "charged_seconds": 20, "heartbeat_unix": clock.wall - 30},
    }
    recovered = BudgetSession(
        interrupted, lambda: None, 30, 3, 8, 15,
        clock=clock.mono, wall_clock=clock.unix,
    )
    assert interrupted["used_seconds"] == pytest.approx(used + 30)
    assert interrupted["recovered_uncertain_lease"]["charged_seconds"] == 30
    assert recovered.total_remaining() < (30 - 3) * 3600


def test_usage_observations_never_reset_or_double_charge():
    state = {"used_seconds": 2 * 3600}
    apply_usage_observations(state, already_used_hours=1, charge_hours=.5,
                             charge_id="setup-a", quota_remaining_hours=10,
                             now=lambda: 100)
    assert state["used_seconds"] == pytest.approx(2.5 * 3600)
    apply_usage_observations(state, already_used_hours=2, charge_hours=.5,
                             charge_id="setup-a")
    assert state["used_seconds"] == pytest.approx(2.5 * 3600)
    state["used_seconds"] += 3600
    assert quota_remaining_from_state(state) == pytest.approx(9)
    with pytest.raises(ValueError, match="different"):
        apply_usage_observations(state, charge_hours=.25, charge_id="setup-a")


def test_fixed_matrix_plan_has_18_jobs_and_local_only_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert budget_main([
        "--config", str(root / "configs" / "kaggle_30h.yaml"),
        "--action", "plan", "--output-root", str(tmp_path),
    ]) == 0
    plan = json.loads((tmp_path / "requested_plan.json").read_text())
    assert len(plan["jobs"]) == 18
    assert len({job.split("/")[1] for job in plan["jobs"]}) == 6
    configs = list((tmp_path / "configs").glob("*.yaml"))
    assert len(configs) == 18
    local_cfg = yaml.safe_load(next(path for path in configs if path.stem.endswith("local-only")).read_text())
    assert Path(local_cfg["output"]["root"]).name == "local_only"
    assert local_cfg["training"]["seed"] == 42
    assert local_cfg["federation"]["num_clients"] == 10
    assert local_cfg["federation"]["fraction_train"] == 1.0
    assert local_cfg["training"]["local_epochs"] == 1
    assert local_cfg["early_stopping"]["enabled"] is False


def test_calibrate_freezes_protocol_without_starting_main(tmp_path, monkeypatch):
    calls = []

    def fake_run_job(job, session, state, save, calibration=False):
        assert calibration
        calls.append(job["mode"])
        run = tmp_path / "fake" / job["mode"]
        if job["mode"] == "local-only":
            histories = [run / f"client_{client_id:02d}" / "history.csv"
                         for client_id in range(10)]
        else:
            histories = [run / "history.csv"]
        for history in histories:
            history.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"duration_seconds": 1.0}]).to_csv(history, index=False)
        state["jobs"][job["id"]] = {"wall_seconds": 20.0}
        save()
        return run

    monkeypatch.setattr(budget_runner, "run_job", fake_run_job)
    monkeypatch.setattr(
        budget_runner, "collect_jobs",
        lambda jobs, output: {"completed_jobs": 0, "expected_jobs": len(jobs)},
    )
    monkeypatch.setattr(budget_runner.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("calibrate launched subprocess"))
    config = Path(__file__).resolve().parents[1] / "configs" / "kaggle_30h.yaml"
    code = budget_runner.main([
        "--config", str(config), "--action", "calibrate", "--output-root", str(tmp_path / "output"),
    ])
    assert code == 0
    assert calls == ["centralized", "local-only", "fedavg"]
    state = json.loads((tmp_path / "output" / "budget_state.json").read_text())
    assert state["status"] == "protocol_frozen_main_not_started"
    assert 10 <= state["rounds"] <= 20
    assert state["calibration"]["local-only"]["client_models_measured"] == 10


def test_collect_cli_uses_restored_ledger_and_never_starts_training(tmp_path, monkeypatch):
    def fake_calibration(job, session, state, save, calibration=False):
        assert calibration
        run = tmp_path / "calibration-artifacts" / job["mode"]
        histories = ([run / f"client_{client_id:02d}" / "history.csv" for client_id in range(10)]
                     if job["mode"] == "local-only" else [run / "history.csv"])
        for history in histories:
            history.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"duration_seconds": 1.0}]).to_csv(history, index=False)
        state["jobs"][job["id"]] = {"wall_seconds": 20.0}
        save()
        return run

    monkeypatch.setattr(budget_runner, "run_job", fake_calibration)
    monkeypatch.setattr(budget_runner.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("collect launched subprocess"))
    config = Path(__file__).resolve().parents[1] / "configs" / "kaggle_30h.yaml"
    output = tmp_path / "restored-output"
    assert budget_runner.main([
        "--config", str(config), "--action", "calibrate", "--output-root", str(output),
    ]) == 0
    assert budget_runner.main([
        "--config", str(config), "--action", "collect", "--output-root", str(output),
    ]) == 0
    status = json.loads((output / "comparison" / "collection_status.json").read_text())
    assert {key: status[key] for key in ("completed", "missing", "failed", "invalid")} == {
        "completed": 0, "missing": 18, "failed": 0, "invalid": 0,
    }
    assert status["matrix_complete"] is False
    assert len(status["jobs"]) == 18
    assert all(job["status"] == "missing" for job in status["jobs"])
    assert (output / "comparison" / "comparison.csv").is_file()
    assert (output / "comparison" / "comparison.json").is_file()
    assert (output / "comparison" / "comparison.md").is_file()


def test_deadline_validation_and_lock_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv("FL_TRAINING_SOFT_DEADLINE_UNIX", "100")
    monkeypatch.setenv("FL_TRAINING_HARD_DEADLINE_UNIX", "110")
    check_deadline(now=99)
    with pytest.raises(BudgetExhausted, match="Soft"):
        check_deadline(now=100)
    monkeypatch.setenv("FL_TRAINING_SOFT_DEADLINE_UNIX", "nan")
    with pytest.raises(ValueError, match="finite"):
        check_deadline(now=1)

    _apply_deadline_config({"soft_deadline_unix": "4102444800", "hard_deadline_unix": "4102444860"})
    assert os.environ["FL_TRAINING_SOFT_DEADLINE_UNIX"] == "4102444800"
    assert os.environ["FL_TRAINING_HARD_DEADLINE_UNIX"] == "4102444860"
    _apply_deadline_config({})
    assert "FL_TRAINING_SOFT_DEADLINE_UNIX" not in os.environ
    assert "FL_TRAINING_HARD_DEADLINE_UNIX" not in os.environ

    lock = tmp_path / "runner.lock"
    token = acquire_lock(lock)
    with pytest.raises(RuntimeError, match="Another runner"):
        acquire_lock(lock)
    with pytest.raises(RuntimeError, match="ownership"):
        release_lock(lock, "wrong")
    release_lock(lock, token)
    assert not lock.exists()


def test_resume_rejects_source_mode_and_fixed_budget_mismatch():
    payload = {
        "semantic_config_hash": "same",
        "config": {"source_fingerprint": "source-a", "mode": "train",
                   "federation": {"max_rounds": 10}},
    }
    current = {"semantic_config_hash": "same", "source_fingerprint": "source-b",
               "mode": "train", "federation": {"max_rounds": 10}}
    with pytest.raises(ValueError, match="source fingerprint"):
        verify_checkpoint_compatibility(payload, current)
    current["source_fingerprint"] = "source-a"
    current["mode"] = "centralized"
    with pytest.raises(ValueError, match="mode mismatch"):
        verify_checkpoint_compatibility(payload, current)
    current["mode"] = "train"
    current["federation"]["max_rounds"] = 11
    with pytest.raises(ValueError, match="round budget mismatch"):
        verify_checkpoint_compatibility(payload, current)


def _tiny_cfg(tmp_path: Path, run_id: str, rounds: int = 3):
    partition = tmp_path / "partition"
    partition.mkdir(exist_ok=True)
    for name in ("global_val.csv", "global_test.csv"):
        (partition / name).write_text("relative_path,label\n", encoding="utf-8")
    cfg = SimpleNamespace(
        run_id=run_id, semantic_config_hash="tiny-semantic", resume_baselines=False,
        model=SimpleNamespace(weights=None),
        training=SimpleNamespace(seed=42, lr=0.05, momentum=0.9, weight_decay=0.0,
                                 local_epochs=1, eval_batch_size=4, num_workers=0, amp=False),
        runtime=SimpleNamespace(client_device="cpu", server_device="cpu"),
        early_stopping=SimpleNamespace(enabled=False, min_delta=0.0, patience_rounds=2,
                                       warmup_rounds=0),
        lr_scheduler=SimpleNamespace(enabled=False, factor=0.5, patience_rounds=2,
                                     threshold=0.0, min_lr=0.0),
        federation=SimpleNamespace(max_rounds=rounds),
        checkpoint=SimpleNamespace(every_n_rounds=1, keep_last_n=2),
        output=SimpleNamespace(evaluate_test_after_train=True),
        data=SimpleNamespace(partition_dir=partition, dataset_root=tmp_path,
                             protocol_fingerprint="tiny-protocol",
                             class_names=[str(i) for i in range(38)], scenario="iid"),
    )
    cfg.to_dict = lambda: {
        "schema_version": 1, "mode": "centralized", "run_id": cfg.run_id,
        "semantic_config_hash": cfg.semantic_config_hash,
        "source_fingerprint": "tiny-source",
        "data": {"protocol_fingerprint": cfg.data.protocol_fingerprint,
                 "partition_dir": str(cfg.data.partition_dir), "class_names": cfg.data.class_names},
        "training": {"seed": 42}, "federation": {"max_rounds": cfg.federation.max_rounds},
        "output": {"run_dir": ""},
    }
    return cfg


def test_baseline_uninterrupted_equals_interrupted_resume_cpu(tmp_path, monkeypatch):
    data = TensorDataset(
        torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
        torch.tensor([0, 1, 2, 3]),
    )
    loader = DataLoader(data, batch_size=2, shuffle=False)
    monkeypatch.setattr(baselines, "create_mobilenet_v3_small",
                        lambda *_args, **_kwargs: torch.nn.Linear(4, 38))
    monkeypatch.setattr(baselines, "build_evaluation_loader", lambda *_args, **_kwargs: loader)
    monkeypatch.setattr(baselines, "generate_report", lambda *_args, **_kwargs: {})

    cfg_full = _tiny_cfg(tmp_path, "full")
    full_dir = tmp_path / "full"
    baselines._train_one(cfg_full, full_dir, loader, 42, "centralized")

    cfg_resume = _tiny_cfg(tmp_path, "resume")
    resume_dir = tmp_path / "resume"
    calls = {"value": 0}
    real_check = baselines.check_deadline

    def interrupt_before_round_two():
        calls["value"] += 1
        if calls["value"] == 3:
            raise BudgetExhausted("forced test interruption")
        real_check()

    monkeypatch.setattr(baselines, "check_deadline", interrupt_before_round_two)
    with pytest.raises(BudgetExhausted):
        baselines._train_one(cfg_resume, resume_dir, loader, 42, "centralized")
    assert load_checkpoint(resume_dir / "last.pt")["round"] == 1

    monkeypatch.setattr(baselines, "check_deadline", real_check)
    cfg_resume.resume_baselines = True
    baselines._train_one(cfg_resume, resume_dir, loader, 42, "centralized")
    full = load_checkpoint(full_dir / "last.pt")
    resumed = load_checkpoint(resume_dir / "last.pt")
    assert full["round"] == resumed["round"] == 3
    for name in full["model_state_dict"]:
        assert torch.equal(full["model_state_dict"][name], resumed["model_state_dict"][name])
    for left, right in zip(full["history"], resumed["history"]):
        assert {k: v for k, v in left.items() if k != "duration_seconds"} == {
            k: v for k, v in right.items() if k != "duration_seconds"
        }
    assert full["optimizer_state"]["param_groups"] == resumed["optimizer_state"]["param_groups"]
    for key in full["optimizer_state"]["state"]:
        assert torch.equal(full["optimizer_state"]["state"][key]["momentum_buffer"],
                           resumed["optimizer_state"]["state"][key]["momentum_buffer"])
    assert full["amp_scaler_policy"] == "reset_each_train_local_call"
    assert full["best_metrics"] == resumed["best_metrics"]
    # A second resume of an evaluated Centralized model must skip as Centralized.
    cfg_full.resume_baselines = True
    monkeypatch.setattr(baselines, "train_local", lambda **kwargs: pytest.fail("completed baseline retrained"))
    baselines._train_one(cfg_full, full_dir, loader, 42, "centralized")


def _write_result(run: Path, mode: str, accuracy: float, macro_f1: float,
                  client_id=None, semantic="same", protocol="protocol"):
    run.mkdir(parents=True)
    config = {
        "run_id": run.name, "mode": mode, "client_id": client_id,
        "source_fingerprint": "source", "semantic_config_hash": semantic,
        "training": {"seed": 42}, "federation": {"max_rounds": 10, "num_clients": 2},
        "data": {"protocol_fingerprint": protocol},
    }
    yaml.safe_dump(config, (run / "resolved_config.yaml").open("w", encoding="utf-8"))
    state = {"weight": torch.tensor([accuracy], dtype=torch.float32)}
    payload = {"model_state_dict": state, "best_round": 2, "config": config}
    save_checkpoint_atomic(payload, run / "best.pt")
    save_checkpoint_atomic({**payload, "round": 10}, run / "last.pt")
    metrics = {
        "accuracy": accuracy, "macro_f1": macro_f1, "checkpoint": "best.pt",
        "run_id": run.name, "protocol_fingerprint": protocol,
        "model_state_sha256": model_state_sha256(state), "checkpoint_round": 2,
    }
    metrics["evaluation_id"] = hashlib.sha256(
        json.dumps(metrics, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:24]
    (run / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame([{"round": 10, "duration_seconds": 1.0, "processed_examples": 8,
                   "optimizer_steps": 2, "skipped_optimizer_steps": 0,
                   "payload_bytes": 100}]).to_csv(run / "history.csv", index=False)
    summary = {"status": "completed", "artifact_scope": "main_evaluated", "mode": mode,
               "protocol_fingerprint": protocol, "best_round": 2, "completed_rounds": 10,
               "stop_reason": "max_rounds", "client_id": client_id}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_collector_gaps_local_distribution_corruption_and_cleanup(tmp_path, monkeypatch):
    condition = {"scenario": "label_skew", "alpha": 0.1, "feature_skew": "none"}
    central = tmp_path / "central"
    fedavg = tmp_path / "fedavg"
    local = tmp_path / "local"
    _write_result(central, "centralized", .80, .70)
    _write_result(fedavg, "train", .85, .75)
    fed_summary = json.loads((fedavg / "summary.json").read_text())
    fed_summary.pop("mode")
    (fedavg / "summary.json").write_text(json.dumps(fed_summary), encoding="utf-8")
    local.mkdir()
    config = {"run_id": "local", "mode": "local-only", "source_fingerprint": "source",
              "semantic_config_hash": "same", "training": {"seed": 42},
              "federation": {"max_rounds": 10, "num_clients": 2},
              "data": {"protocol_fingerprint": "protocol"}}
    yaml.safe_dump(config, (local / "resolved_config.yaml").open("w", encoding="utf-8"))
    clients = [_write_result(local / "client_00", "local-only", .50, .40, 0),
               _write_result(local / "client_01", "local-only", .70, .60, 1)]
    (local / "summary.json").write_text(json.dumps({
        "status": "completed", "artifact_scope": "main_evaluated", "mode": "local-only",
        "protocol_fingerprint": "protocol", "clients": clients,
    }), encoding="utf-8")

    rows = [_result_row(central, condition, 42, "centralized"),
            _result_row(fedavg, condition, 42, "fedavg"),
            _result_row(local, condition, 42, "local-only")]
    output = tmp_path / "comparison"
    _write_comparison(rows, output)
    frame = pd.read_csv(output / "comparison.csv")
    fed = frame[frame["mode"] == "fedavg"].iloc[0]
    assert fed["centralized_minus_accuracy_pp"] == pytest.approx(-5.0)
    assert fed["fedavg_minus_local_accuracy_pp"] == pytest.approx(25.0)
    loc = frame[frame["mode"] == "local-only"].iloc[0]
    assert loc["local_accuracy_std"] == pytest.approx(.1)
    assert (output / "comparison_macro_f1.png").exists()

    metrics_path = fedavg / "test_metrics.json"
    bad = json.loads(metrics_path.read_text())
    bad["accuracy"] = .1
    metrics_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        _result_row(fedavg, condition, 42, "fedavg")

    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("collect launched training"))
    (output / "accuracy_vs_alpha.png").write_bytes(b"stale")
    status = collect_jobs([], output)
    assert status["expected_jobs"] == 0
    assert not (output / "accuracy_vs_alpha.png").exists()

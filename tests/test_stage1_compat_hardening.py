"""Real failure cases for accounting, locks, identities, checkpoints and collection."""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from stage1_compat.budget import BudgetLedger, RunnerLock, LockError, DeadlineLoader, BudgetExhausted
from stage1_compat.accounting import BudgetSession, apply_usage_observations
from stage1_compat.integrity import read_json, atomic_json, validate_indices
from stage1_compat.identity import generate_job_identity
from stage1_compat.config import JobConfig
from stage1_compat.checkpoint import save_atomic_checkpoint, load_checkpoint, CheckpointError
from stage1_compat.collector import run_stage1_collect
from tests.test_stage1_compat_runner import actual_runner_fixture


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, 31])
def test_reject_invalid_quota(value, tmp_path):
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path, user_quota_hours=value)


def test_corrupt_ledger_is_not_overwritten(tmp_path):
    path = tmp_path / "stage1_ledger.json"
    path.write_text("{broken")
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path, user_quota_hours=30)
    assert path.read_text() == "{broken"


def test_round_events_idempotent_and_reload(tmp_path):
    ledger = BudgetLedger(tmp_path, user_quota_hours=30)
    ledger.record_job_start("j", 1., 42, 10, "identity")
    for _ in range(2):
        ledger.record_round_progress("j", 1, 15., 1, .5)
    restored = BudgetLedger(tmp_path, user_quota_hours=30)
    assert restored.total_used_seconds == 15
    with pytest.raises(ValueError):
        restored.record_job_start("j", 1., 42, 20, "identity")


def test_lock_real_second_process_and_crash_release(tmp_path):
    path = tmp_path / "runner.lock"
    code = "from pathlib import Path; from stage1_compat.budget import RunnerLock; import sys,time; lock=RunnerLock(Path(sys.argv[1])); lock.acquire(); print('LOCKED',flush=True); time.sleep(30)"
    child = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(LockError):
            RunnerLock(path).acquire()
    finally:
        child.kill(); child.wait(timeout=10)
    with RunnerLock(path):
        pass


def test_lease_recovery_counts_uncertain_time_once():
    now = [100.]
    state = {"used_seconds": 0.}
    save = lambda: None
    session = BudgetSession(state, save, 30, 3, 8, 15, clock=lambda: now[0], wall_clock=lambda: now[0])
    session.begin_lease(100)
    now[0] += 20; session.tick()
    now[0] += 30  # process dies without ending lease
    recovered = BudgetSession(state, save, 30, 3, 8, 15, clock=lambda: now[0], wall_clock=lambda: now[0])
    assert state["used_seconds"] == 50
    recovered_again = BudgetSession(state, save, 30, 3, 8, 15, clock=lambda: now[0], wall_clock=lambda: now[0])
    assert state["used_seconds"] == 50
    apply_usage_observations(state, already_used_hours=1)
    apply_usage_observations(state, already_used_hours=1)
    assert state["used_seconds"] == 3600


def test_identity_initialization_context_and_source():
    first = generate_job_identity(JobConfig("j", pretrained=True), context={"dataset": "a"})
    assert first != generate_job_identity(JobConfig("j", pretrained=False), context={"dataset": "a"})
    assert first != generate_job_identity(JobConfig("j", pretrained=True), context={"dataset": "b"})


@pytest.mark.parametrize("field", ["best_model_state", "history", "rng_states"])
def test_full_checkpoint_payload_corruption(field, tmp_path):
    path = tmp_path / "c.pth"
    state = {"weight": torch.tensor([1.])}
    identity = generate_job_identity(JobConfig("j"))
    save_atomic_checkpoint(path, 1, state, copy.deepcopy(state), 1, .5, [{"epoch": 1}], identity)
    raw = torch.load(path, weights_only=False)
    if field == "history": raw[field][0]["epoch"] = 2
    elif field == "rng_states": raw[field]["torch_cpu"][0] += 1
    else: raw[field]["weight"][0] += 1
    torch.save(raw, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, identity)


def test_deadline_before_batch(monkeypatch):
    monkeypatch.setenv("STAGE1_SOFT_DEADLINE", str(time.time()-1))
    with pytest.raises(BudgetExhausted):
        next(iter(DeadlineLoader(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor([1,2]))))))


def test_collector_rejects_json_without_checkpoint_and_clears_stale(tmp_path):
    reports = tmp_path / "reports"; reports.mkdir()
    stale = reports / "stage1_alpha_gap.png"; stale.write_bytes(b"old")
    job = tmp_path / "fedavg_alpha1_seed42"; job.mkdir()
    atomic_json(job / "fedavg_metrics.json", {"test_metrics": {"accuracy": .99}})
    run_stage1_collect(tmp_path)
    status = read_json(reports / "collection_status.json")
    assert status["invalid"] == [1.0]
    assert status["completed"] == []
    assert not stale.exists()
    row = read_json(reports / "comparison.json")["table"][1]
    assert row["fedavg_new_acc"] is None
    assert row["reference_gap_cent_minus_fed_pp"] is None


def test_restore_collect_and_skip_verify_checkpoint(actual_runner_fixture, tmp_path):
    import shutil
    runner, _, train, val, parts, job = actual_runner_fixture
    job.job_id = "fedavg_alpha1_seed42"
    root = tmp_path / "original"
    runner.run_single_fedavg_job(job, train, val, val, [str(i) for i in range(4)], parts,
                                root, BudgetLedger(root, user_quota_hours=30), torch.device("cpu"))
    restored = tmp_path / "restored"; shutil.copytree(root, restored)
    run_stage1_collect(restored)
    assert read_json(restored / "reports/collection_status.json")["completed"] == [1.]
    (restored / job.job_id / "best_model.pth").write_bytes(b"broken")
    with pytest.raises(ValueError):
        runner.run_single_fedavg_job(job, train, val, val, [str(i) for i in range(4)], parts,
                                    restored, BudgetLedger(restored, user_quota_hours=30), torch.device("cpu"))
    run_stage1_collect(restored)
    assert read_json(restored / "reports/collection_status.json")["invalid"] == [1.]


def test_out_of_bounds_partition_rejected():
    for groups in ([[0,1],[-1,3]], [[0,1],[2,4]], [[0,1],[2,True]]):
        with pytest.raises(ValueError): validate_indices(groups, 4)


def test_calibration_arithmetic_real_function():
    from stage1_compat.calibrate import calibration_forecast
    rate, reserve, forecast = calibration_forecast([{"elapsed_seconds":100.,"rounds":1},
                                                    {"elapsed_seconds":120.,"rounds":1}], 10.)
    assert rate == 180 and reserve == 735 and forecast == 4335


def test_collect_entrypoint_does_not_import_runner(tmp_path):
    code = "import sys; from stage1_compat.cli import main; main(['collect','--output-dir',sys.argv[1]]); assert 'stage1_compat.runner' not in sys.modules"
    result = subprocess.run([sys.executable,"-c",code,str(tmp_path)], capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_supervisor_hard_timeout_owns_only_its_child(tmp_path):
    from stage1_compat.launcher import supervise
    class ShortSession:
        def __init__(self): self.end = time.monotonic() + .5
        def tick(self): pass
        def hard_remaining(self): return max(0., self.end-time.monotonic())
    unrelated = subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
    try:
        result = supervise([sys.executable,"-c","import time;time.sleep(30)"], os.environ.copy(), tmp_path / "log", ShortSession())
        assert result["timed_out"]
        assert unrelated.poll() is None
    finally:
        unrelated.kill(); unrelated.wait(timeout=10)


def test_calibration_never_reads_test_dataset(actual_runner_fixture, tmp_path):
    runner, _, train, val, parts, job = actual_runner_fixture
    class UntouchableTest(torch.utils.data.Dataset):
        def __len__(self): return 12
        def __getitem__(self, index): raise AssertionError("Calibration touched test")
    result = runner.run_single_fedavg_job(job, train, val, UntouchableTest(), [str(i) for i in range(4)], parts,
        tmp_path, BudgetLedger(tmp_path, user_quota_hours=30), torch.device("cpu"), calibration=True)
    assert result["status"] == "CALIBRATED_NO_TEST"
    assert not (tmp_path / job.job_id / "fedavg_metrics.json").exists()


def test_crash_after_commit_before_ledger(actual_runner_fixture, tmp_path, monkeypatch):
    runner, _, train, val, parts, job = actual_runner_fixture
    ledger = BudgetLedger(tmp_path, user_quota_hours=30)
    original = ledger.record_round_progress
    def crash(*args, **kwargs): raise RuntimeError("crash after checkpoint")
    monkeypatch.setattr(ledger, "record_round_progress", crash)
    with pytest.raises(RuntimeError, match="crash after checkpoint"):
        runner.run_single_fedavg_job(job, train, val, val, [str(i) for i in range(4)], parts,
                                    tmp_path, ledger, torch.device("cpu"))
    restored = BudgetLedger(tmp_path, user_quota_hours=30)
    result = runner.run_single_fedavg_job(job, train, val, val, [str(i) for i in range(4)], parts,
                                        tmp_path, restored, torch.device("cpu"))
    assert [r["epoch"] for r in result["history"]] == [1,2,3]
    assert sorted(restored.jobs[job.job_id].round_events) == ["1","2","3"]


def test_actual_client_optimizer_reset(actual_runner_fixture, tmp_path, monkeypatch):
    runner, _, train, val, parts, job = actual_runner_fixture
    original = runner.optim.AdamW
    states = []
    def tracked(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        states.append(optimizer)
        assert not optimizer.state
        return optimizer
    monkeypatch.setattr(runner.optim, "AdamW", tracked)
    runner.run_single_fedavg_job(job, train, val, val, [str(i) for i in range(4)], parts,
                                tmp_path, BudgetLedger(tmp_path, user_quota_hours=30), torch.device("cpu"))
    assert len(states) == job.rounds * len(parts)
    assert len({id(x) for x in states}) == len(states)
    assert all(x.state for x in states)


def test_atomic_checkpoint_failure_preserves_previous(tmp_path, monkeypatch):
    import stage1_compat.integrity as integrity
    path = tmp_path / "c.pth"
    state = {"weight": torch.tensor([1.])}
    identity = generate_job_identity(JobConfig("j"))
    save_atomic_checkpoint(path,1,state,state,1,.5,[{"epoch":1}],identity)
    before = path.read_bytes()
    def fail(*args): raise OSError("replace failed")
    monkeypatch.setattr(integrity.os, "replace", fail)
    with pytest.raises(OSError):
        save_atomic_checkpoint(path,2,state,state,1,.5,[{"epoch":1},{"epoch":2}],identity)
    assert path.read_bytes() == before
    assert load_checkpoint(path,identity)["server_round"] == 1


def test_calibration_lock_rejects_source_and_budget_mismatch(tmp_path):
    from stage1_compat.calibrate import validate_calibration
    from stage1_compat.identity import source_fingerprint
    from stage1_compat.integrity import digest
    report = {"source_sha256": source_fingerprint(), "dataset_sha256": "dataset", "device": "cpu",
              "rounds_per_job": 10, "safety_margin": 1.5, "test_evaluated": False, "safe_round_seconds": 100.}
    report["budget_sha256"] = digest(report)
    path = tmp_path / "calibration/calibration_report.json"; atomic_json(path, report)
    assert validate_calibration(tmp_path,"dataset","cpu")["rounds_per_job"] == 10
    with pytest.raises(ValueError): validate_calibration(tmp_path,"dataset","cuda")
    with pytest.raises(ValueError): validate_calibration(tmp_path,"different","cpu")
    report["rounds_per_job"] = 3; atomic_json(path,report)
    with pytest.raises(ValueError): validate_calibration(tmp_path,"dataset","cpu")


def test_zero_quota_cli_never_starts_worker(tmp_path):
    result = subprocess.run([sys.executable,"-m","stage1_compat","run","--output-dir",str(tmp_path),
                             "--quota-hours","0"],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert read_json(tmp_path / "launch_status.json")["status"] == "PAUSED_QUOTA"
    assert not (tmp_path / "stage1_ledger.json").exists()


def test_worker_failure_is_not_reported_completed(tmp_path):
    result = subprocess.run([sys.executable,"-m","stage1_compat","run","--output-dir",str(tmp_path),
                             "--quota-hours","30","--data-dir",str(tmp_path / "missing")],capture_output=True,text=True)
    assert result.returncode != 0
    status = read_json(tmp_path / "launch_status.json")
    assert status["status"] == "FAILED"
    state = read_json(tmp_path / "account_budget.v2.json")
    assert state["used_seconds"] > 0
    assert "active_lease" not in state
    assert (tmp_path / "reports/collection_status.json").exists()


def test_chart_failure_invalidates_partial_report(tmp_path, monkeypatch):
    import stage1_compat.collector as collector
    def fail(*args): raise RuntimeError("plot failed")
    monkeypatch.setattr(collector,"plot_charts",fail)
    with pytest.raises(RuntimeError): collector.run_stage1_collect(tmp_path)
    assert read_json(tmp_path / "reports/collection_status.json")["status"] == "FAILED"
    assert not (tmp_path / "reports/comparison.json").exists()

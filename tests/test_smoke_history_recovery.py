"""A web restart may show only a completed, clearly historical smoke run."""

from __future__ import annotations

import json

from web import app as dashboard
from web import state as dashboard_state


def _run(root, name: str, *, status: str, accepted: int = 5):
    run = root / name
    run.mkdir()
    (run / "session.json").write_text(json.dumps({
        "kind": "flower_localhost_synthetic_smoke",
        "status": status,
        "scientific_valid": False,
        "published_result_reproduction": False,
        "client_processes": 5,
        "rounds": 2,
    }), encoding="utf-8")
    (run / "history.json").write_text(json.dumps({
        "audit_rounds": [
            {"round": 1, "accepted_clients": accepted},
            {"round": 2, "accepted_clients": accepted},
        ],
        "metrics_distributed_fit": {"accepted_clients": [[1, accepted], [2, accepted]]},
    }), encoding="utf-8")
    (run / "global.pt").write_bytes(b"checkpoint marker")
    return run


def test_latest_completed_smoke_run_skips_failed_and_incomplete(monkeypatch, tmp_path):
    valid = _run(tmp_path, "20261005T010000Z_valid", status="completed")
    _run(tmp_path, "20261005T020000Z_failed", status="failed")
    _run(tmp_path, "20261005T030000Z_wrong_audit", status="completed", accepted=4)
    monkeypatch.setattr(dashboard_state, "SMOKE_RUNS_ROOT", tmp_path)
    assert dashboard.latest_completed_smoke_run() == valid


def test_network_metrics_marks_recovered_run_historical(monkeypatch, tmp_path):
    valid = _run(tmp_path, "20261005T010000Z_valid", status="completed")
    monkeypatch.setattr(dashboard_state, "SMOKE_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(dashboard.NETWORK_DEMO, "run_dir", None)
    result = dashboard.get_federated_network_metrics()
    assert result["available"] is True
    assert result["historical_run"] is True
    assert result["scientific_valid"] is False
    assert result["run_dir"] == str(valid)
    assert [row["accepted_clients"] for row in result["rounds"]] == [5, 5]

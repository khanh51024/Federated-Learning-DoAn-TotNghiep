"""Snapshot của phiên Flower trên web: dữ liệu bằng chứng tính toán và việc bỏ liên kết phiên cũ."""

from __future__ import annotations

import json
from pathlib import Path

from web.network_demo import NetworkDemoManager, _brief_rounds

ROUND = {
    "round": 1, "accepted_clients": 5, "suspicious_clients": 0, "rejected_clients": 0,
    "round_fit_elapsed_ms": 1234.5, "mean_probe_rtt_ms": 14.0,
    "parent_model_hash": "a" * 64, "global_model_hash": "b" * 64, "mean_received_learnable_update_norm": 0.003,
    "server_validation_metrics": None,
    "clients": [{
        "client_id": 0, "num_examples": 22, "local_train_time_s": 1.9, "received_learnable_update_norm": 0.002,
        "learnable_update_relative": 1e-5, "status": "ok", "data_mode": "synthetic",
        "partition_config_sha256": "không đưa ra giao diện",
    }],
}


def _manager_with_progress(tmp_path: Path) -> NetworkDemoManager:
    manager = NetworkDemoManager()
    manager.run_dir = tmp_path
    (tmp_path / "progress.json").write_text(json.dumps({
        "round_in_progress": 2, "completed_round": 1, "connected_clients": 5, "accepted_clients": 5,
        "parent_model_hash": "b" * 64, "rounds": [ROUND],
    }), encoding="utf-8")
    return manager


def test_brief_rounds_keeps_only_what_the_page_needs():
    brief = _brief_rounds([ROUND])
    assert brief[0]["global_model_hash"] == "b" * 64 and brief[0]["round"] == 1
    assert "server_validation_metrics" not in brief[0]
    assert brief[0]["clients"][0]["received_learnable_update_norm"] == 0.002
    assert "partition_config_sha256" not in brief[0]["clients"][0]
    assert _brief_rounds(None) == [] and _brief_rounds("x") == []


def test_snapshot_exposes_rounds_and_processes_after_the_session_closed(tmp_path):
    manager = _manager_with_progress(tmp_path)
    manager._final_exit_codes = {"server": 0, "client_00": 0}
    snapshot = manager.snapshot()
    assert snapshot["rounds_brief"][0]["parent_model_hash"] == "a" * 64
    assert snapshot["active"] is False
    assert snapshot["processes"] == [
        {"name": "client_00", "running": False, "exit_code": 0},
        {"name": "server", "running": False, "exit_code": 0},
    ]


def test_detach_run_forgets_a_finished_session_but_never_a_running_one(tmp_path):
    manager = _manager_with_progress(tmp_path)
    manager._running = True
    manager.detach_run()
    assert manager.run_dir == tmp_path, "đang chạy thì không được bỏ liên kết"

    manager._running = False
    manager._final_exit_codes = {"server": 0}
    manager.detach_run()
    snapshot = manager.snapshot()
    assert manager.run_dir is None and snapshot["rounds_brief"] == [] and snapshot["processes"] == []
    assert snapshot["round_completed"] == 0 and snapshot["mean_probe_rtt_ms"] is None

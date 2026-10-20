"""Trang Flower: kết luận chỉ "hoạt động" khi có phiên hoàn tất với bằng chứng đủ; thiếu flwr hay mã băm lệch thì không."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.services import flower_health as fh

H0, H1, H2 = "a" * 64, "b" * 64, "c" * 64


def _write_run(root: Path, name: str, *, status: str = "completed", rounds: int = 2, accepted: int = 5, chain_ok: bool = True,
               final_hash: str = H2, error: str | None = None, with_checkpoint: bool = True, log: str | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    session = {"kind": fh.SMOKE_KIND, "status": status, "scientific_valid": False, "published_result_reproduction": False,
               "client_processes": 5, "rounds": rounds, "started_at_utc": "2026-10-20T01:00:00Z", "finished_at_utc": "2026-10-20T01:01:00Z"}
    if error:
        session["error"] = error
    (run / "session.json").write_text(json.dumps(session), encoding="utf-8")
    if log:
        (run / "server.log").write_text(log, encoding="utf-8")
    if status == "completed":
        hashes = [(H0, H1), (H1 if chain_ok else H0, H2)]
        audits = [{"round": i + 1, "clients": [{"client_id": c} for c in range(5)], "accepted_clients": accepted, "suspicious_clients": 0,
                   "rejected_clients": 0, "round_fit_elapsed_ms": 1500.0, "mean_probe_rtt_ms": 12.0, "max_probe_rtt_ms": 20.0,
                   "parent_model_hash": hashes[i][0], "global_model_hash": hashes[i][1]} for i in range(rounds)]
        (run / "history.json").write_text(json.dumps({"audit_rounds": audits, "final_model_hash": final_hash, "rounds": rounds}), encoding="utf-8")
        (run / "progress.json").write_text(json.dumps({"connected_clients": 5, "completed_round": rounds}), encoding="utf-8")
        if with_checkpoint:
            (run / "global.pt").write_bytes(b"checkpoint-bytes")
    return run


@pytest.fixture()
def runs(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "network_smoke"
    root.mkdir()
    monkeypatch.setattr(fh, "runs_root", lambda: root)
    monkeypatch.setattr(fh, "flwr_status", lambda: {"installed": True, "version": "1.39.0", "python": "3.11.9", "error": None})
    return root


def _health(**kwargs):
    return fh.build_health(snapshot={}, port_open=lambda port: False, read_checkpoint_hash=lambda path: H2, **kwargs)


def _by_id(result):
    return {item["id"]: item for item in result["items"]}


def test_a_completed_run_with_matching_hashes_is_reported_as_working(runs):
    _write_run(runs, "20261020T010000Z_aaaa1111")
    result = _health()
    items = _by_id(result)
    assert result["verdict"] == fh.WORKING and "5 client" in result["summary"]
    assert items["flwr"]["ok"] and items["clients"]["ok"] and items["rounds"]["ok"] and items["hashes"]["ok"]
    evidence = result["evidence"]
    assert evidence["hash_chain_ok"] is True and evidence["final_matches_checkpoint"] is True
    assert evidence["checkpoint"]["bytes"] == len(b"checkpoint-bytes") and len(evidence["checkpoint"]["sha256"]) == 64
    assert "participant" not in json.dumps(result) and str(runs) not in json.dumps(result), "không lộ đường dẫn tuyệt đối"


def test_missing_flwr_is_broken_even_if_an_old_run_exists(runs, monkeypatch):
    _write_run(runs, "20261012T092936Z_bbbb2222")
    monkeypatch.setattr(fh, "flwr_status", lambda: {"installed": False, "version": None, "python": "3.13.1", "error": "thiếu flwr"})
    result = _health()
    assert result["verdict"] == fh.BROKEN and "thiếu flwr" in result["summary"] and _by_id(result)["flwr"]["ok"] is False


def test_a_failed_newest_run_is_broken_and_shows_the_log_tail(runs):
    _write_run(runs, "20261012T092936Z_bbbb2222")
    _write_run(runs, "20261014T142841Z_cccc3333", status="failed", log="Traceback\nModuleNotFoundError: No module named 'flwr'\n")
    result = _health()
    assert result["verdict"] == fh.BROKEN
    assert "ModuleNotFoundError" in result["latest_attempt"]["log_tail"] and _by_id(result)["attempt"]["ok"] is False


def test_broken_hash_chain_or_checkpoint_is_not_working(runs):
    _write_run(runs, "20261020T010000Z_dddd4444", chain_ok=False)
    result = _health()
    assert _by_id(result)["hashes"]["ok"] is False and result["verdict"] == fh.BROKEN
    other = runs / "20261020T020000Z_eeee5555"
    _write_run(runs, other.name, final_hash=H1)  # mã băm cuối trong history khác vòng cuối
    assert _by_id(_health())["hashes"]["ok"] is False


def test_missing_cluster_updates_are_not_working(runs):
    _write_run(runs, "20261020T010000Z_ffff6666", accepted=4)
    result = _health()
    assert _by_id(result)["rounds"]["ok"] is False and result["verdict"] == fh.BROKEN


def test_old_run_without_hashes_is_working_but_hashes_are_unverified(runs):
    run = _write_run(runs, "20261012T092936Z_0000aaaa")
    audits = json.loads((run / "history.json").read_text(encoding="utf-8"))
    for audit in audits["audit_rounds"]:
        audit.pop("parent_model_hash"), audit.pop("global_model_hash")
    audits.pop("final_model_hash")
    (run / "history.json").write_text(json.dumps(audits), encoding="utf-8")
    result = _health()
    assert _by_id(result)["hashes"]["ok"] is None and result["verdict"] == fh.WORKING


def test_no_runs_at_all_is_not_verified(runs):
    result = _health()
    assert result["verdict"] == fh.NOT_VERIFIED and result["evidence"] is None and result["latest_attempt"] is None


def test_endpoint_is_public_read_only_and_never_starts_anything():
    response = TestClient(app).get("/api/flower/health")
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in (fh.WORKING, fh.NOT_VERIFIED, fh.BROKEN) and {"flwr", "items", "evidence", "live"} <= set(body)
    assert TestClient(app).post("/api/flower/health").status_code == 405

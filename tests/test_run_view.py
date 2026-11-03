"""Một chỗ tính cho web và terminal: run_view đọc đúng mọi trạng thái, không lộ IP, và số trên web trùng số terminal in ra."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from scripts import demo_show as ds
from src.federated import run_view as rv
from src.federated.verification_view import speed_view, sweep_view, weight_view

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def _client(client_id: int, machine: str) -> dict:
    return {
        "client_id": client_id, "machine_id": machine, "peer_hash": "pp" + machine, "same_host_as_server": False, "num_examples": 380, "status": "ok", "data_mode": "real",
        "probe_rtt_persistent_ms": 44.5, "probe_up_mb_per_s": 2.3, "probe_down_mb_per_s": 12.5, "response_ms": 20500.0, "local_train_time_s": 10.5,
        "local_train_time_source": "server_estimate", "update_norm_reported": 8.4, "received_update_norm": 8.4, "update_norm_matches": True,
        "received_learnable_update_norm": 0.77, "upload_hash_ok": True, "client_reported_hash": "a" * 64, "server_recomputed_hash": "a" * 64,
        "payload_bytes_up": 6307176, "payload_bytes_down": 6307176, "flower_cid": "cid-bi-mat",
    }


def _audit(round_num: int, parent: str, merged: str, accuracy: float = 0.5) -> dict:
    return {
        "round": round_num, "accepted_clients": 2, "rejected_clients": 0, "suspicious_clients": 0, "round_fit_elapsed_ms": 26000.0, "server_validate_ms": 100.0,
        "server_aggregate_ms": 50.0, "server_eval_ms": 850.0, "transfer_estimate_ms": 5000.0, "parent_model_hash": parent, "global_model_hash": merged,
        "server_validation_metrics": {"accuracy": accuracy, "loss": 2.5, "macro_f1": 0.4}, "clients": [_client(1, "m2"), _client(0, "m1")],
    }


def _progress(status: str = "running", age_s: int = 5, **extra) -> dict:
    return {"status": status, "updated_at_utc": (NOW - timedelta(seconds=age_s)).strftime(STAMP), "connected_clients": 2, "round_in_progress": 2, "round_participants": 2,
            "round_replies": [{"client_id": 0, "response_ms": 12000.0}], "rounds": [_audit(1, "p1", "g1")],
            "meta": {"num_clients": 2, "min_clients": 2, "rounds": 20, "seed": 42, "local_epochs": 1, "pretrained_init": True, "started_at_utc": "2026-11-04T11:00:00Z"}, **extra}


def test_running_view_counts_rounds_replies_and_checks():
    view = rv.run_view({"run_id": "r1", "progress": _progress(), "history": None}, now=NOW)
    assert view["state"] == "running" and view["completed_rounds"] == 1 and view["total_rounds"] == 20
    assert view["round_replies"] == [{"client_id": 0, "response_ms": 12000.0}]
    assert view["rounds"][0]["accuracy"] == 0.5 and [c["client_id"] for c in view["rounds"][0]["clients"]] == [0, 1]
    checks = view["checks"]
    assert checks["hash_ok"] == 2 and checks["hash_total"] == 2 and checks["norm_ok"] == 2 and checks["full_rounds"] == 1
    assert checks["machines"] == 2 and checks["clients_on_server_host"] == 0 and checks["bytes_up"] == 2 * 6307176
    assert view["caveat"] == rv.CAVEAT_PRETRAINED


def test_round_time_parts_add_up_to_the_whole_round():
    row = rv.run_view({"progress": _progress(), "history": None}, now=NOW)["rounds"][0]
    assert row["elapsed_s"] == 26.0 and row["server_s"] == 1.0 and row["transfer_s"] == 5.0 and row["learn_and_wait_s"] == 20.0
    assert row["learn_and_wait_s"] + row["transfer_s"] + row["server_s"] == row["elapsed_s"]


def test_machine_names_follow_smallest_client_and_stay_stable():
    view = rv.run_view({"progress": _progress(), "history": None}, now=NOW)
    assert {c["client_id"]: c["machine"] for c in view["rounds"][0]["clients"]} == {0: "Máy A", 1: "Máy B"}
    assert [m["label"] for m in view["machines"]] == ["Máy A", "Máy B"] and view["machines"][0]["client_ids"] == [0]


def test_waiting_view_before_any_round():
    progress = {"status": "waiting", "updated_at_utc": NOW.strftime(STAMP), "connected_clients": 6, "rounds": [], "meta": {"num_clients": 10, "min_clients": 10, "rounds": 20}}
    view = rv.run_view({"progress": progress, "history": None}, now=NOW)
    assert view["state"] == "waiting" and view["connected_clients"] == 6 and view["rounds"] == [] and view["checks"]["hash_total"] == 0


def test_dead_server_is_reported_as_stale_not_running():
    assert rv.run_view({"progress": _progress(age_s=rv.STALE_AFTER_S + 5), "history": None}, now=NOW)["state"] == "stale"
    assert rv.run_view({"progress": _progress(age_s=rv.STALE_AFTER_S - 5), "history": None}, now=NOW)["state"] == "running"
    waiting_dead = {"status": "waiting", "updated_at_utc": (NOW - timedelta(seconds=rv.STALE_AFTER_S + 5)).strftime(STAMP), "meta": {"num_clients": 10}}
    assert rv.run_view({"progress": waiting_dead, "history": None}, now=NOW)["state"] == "stale"


def test_failed_run_keeps_the_reason_even_when_old():
    view = rv.run_view({"progress": _progress("failed", age_s=9999, error="Vòng 3: nhận 1/2 client", failed_round={"round": 3}), "history": None}, now=NOW)
    assert view["state"] == "failed" and view["failed_round"] == 3 and "Vòng 3" in view["error"]


def test_history_means_finished_even_if_progress_is_old():
    history = {"num_clients_required": 2, "rounds": 1, "seed": 42, "final_model_hash": "g1", "audit_rounds": [_audit(1, "p1", "g1")],
               "final_test_metrics": {"accuracy": 0.61234, "loss": 1.2, "macro_f1": 0.5}}
    view = rv.run_view({"progress": _progress(age_s=99999), "history": history}, now=NOW)
    assert view["state"] == "finished" and view["final_test"]["accuracy"] == 0.6123 and view["checks"]["chain_ok"] is True


def test_broken_hash_chain_is_detected():
    history = {"num_clients_required": 2, "rounds": 2, "seed": 42, "final_model_hash": "g2", "audit_rounds": [_audit(1, "p1", "g1"), _audit(2, "KHAC", "g2")]}
    assert rv.run_view({"progress": None, "history": history})["checks"]["chain_ok"] is False


def test_old_history_without_new_fields_still_reads():
    client = {"client_id": 0, "num_examples": 10, "local_train_time_s": 1.5, "probe_rtt_ms": 3.0, "upload_hash_ok": True, "client_reported_hash": "x", "server_recomputed_hash": "x"}
    history = {"num_clients_required": 1, "rounds": 1, "seed": 42, "audit_rounds": [{"round": 1, "accepted_clients": 1, "rejected_clients": 0, "round_fit_elapsed_ms": 2000.0, "server_aggregate_ms": 1.0, "clients": [client]}]}
    view = rv.run_view({"progress": None, "history": history})
    row = view["rounds"][0]["clients"][0]
    assert view["state"] == "finished" and row["rtt_ms"] == 3.0 and row["machine"] == "chưa rõ" and row["train_is_server_estimate"] is False
    assert view["checks"]["host_check_available"] is False and view["checks"]["machines"] == 0


def test_no_data_at_all():
    assert rv.run_view({"run_id": "x", "progress": None, "history": None}) == {"available": False, "run_id": "x", "state": "none"}


def test_half_written_file_is_treated_as_missing(tmp_path):
    (tmp_path / rv.PROGRESS_FILE).write_text('{"status": "runn', encoding="utf-8")
    assert rv.load_run(tmp_path)["progress"] is None


def test_terminal_and_web_print_the_same_numbers():
    view = rv.run_view({"progress": _progress(), "history": None}, now=NOW)
    text = "\n".join(ds.view_lines(view))
    assert "độ chính xác 50.00%" in text and "RTT 44.5 ms" in text and "lên 2.30 MiB/s" in text and "phản hồi 20.50 giây" in text
    assert "(Máy A)" in text and "client khai 8.4000 / máy chủ tự tính 8.4000 -> KHỚP" in text and "2/2 lượt băm khớp" in text
    client = view["rounds"][0]["clients"][0]
    assert (client["rtt_ms"], client["up_mb_s"], client["response_s"]) == (44.5, 2.3, 20.5)


def test_view_never_carries_cid_or_full_hash_or_ip():
    dumped = json.dumps(rv.run_view({"progress": _progress(), "history": None}, now=NOW))
    assert "cid-bi-mat" not in dumped and "a" * 13 not in dumped and "peer_hash" not in dumped


def _write_run(root, name, payload, file_name=rv.PROGRESS_FILE):
    (root / name).mkdir(parents=True)
    (root / name / file_name).write_text(json.dumps(payload), encoding="utf-8")


def test_live_api_reads_latest_run_and_refuses_unknown_names(tmp_path, monkeypatch):
    from src import registry as reg
    from web.app import app
    from web.routers import flower as router

    monkeypatch.setattr(reg, "NETWORK_LIVE_DIR", str(tmp_path))
    router._live_cache.clear()
    client = TestClient(app)
    empty = client.get("/api/flower/live").json()
    assert empty == {"available": False, "state": "none", "runs": []}

    router._live_cache.clear()
    _write_run(tmp_path, "20261103T100000Z_cu", _progress("finished"))
    _write_run(tmp_path, "20261104T100000Z_moi", _progress())
    latest = client.get("/api/flower/live").json()
    assert latest["run_id"] == "20261104T100000Z_moi" and latest["runs"] == ["20261104T100000Z_moi", "20261103T100000Z_cu"]
    assert client.get("/api/flower/live", params={"run": "20261103T100000Z_cu"}).json()["run_id"] == "20261103T100000Z_cu"
    for bad in ("../../etc", "..", "khong-co", "a/b", "x" * 81):
        assert client.get("/api/flower/live", params={"run": bad}).status_code in (404, 422), bad


def test_sweep_view_marks_cells_not_yet_measured(tmp_path):
    table = tmp_path / "t.csv"
    header = "scheme,alpha,variant,seeds,part,n,top1_mean_%,top1_std_%,macro_f1_mean_%,gap_top1_points"
    rows = [header, ",,centralized,3,all_test,100,97.0,0.1,96.0,", "label_alpha_0_1,0.1,fedavg,3,all_test,100,90.0,0.5,88.0,7.0", "label_alpha_0_1,0.1,local,3,all_test,100,40.0,1.0,20.0,57.0"]
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    view = sweep_view(table)
    assert view["centralized_top1"] == 97.0 and [r["gap"] for r in view["rows"]] == [None, 7.0, 57.0]
    assert view["by_alpha"]["labels"][-1] == "α=0,1" and view["by_alpha"]["fedavg"][-1]["top1"] == 90.0 and view["by_alpha"]["fedavg"][0]["top1"] is None
    assert "Tự học một mình, α=1" in view["not_measured"] and "Phân tán, lệch nguồn ảnh" in view["not_measured"]


def test_real_verification_files_have_no_sealed_split_and_keep_float32_apart():
    weights = weight_view("artifacts/distributed_verification", "artifacts/fedsgd_equivalence.json")
    dumped = json.dumps(weights, ensure_ascii=False)
    assert "calibration" not in dumped
    transport = weights["transport"]
    assert transport["passed"] == transport["runs"] > 0
    assert transport["comparisons"] == transport["rounds"] * 10, "mỗi vòng 5 client, mỗi client so hai chiều: số vòng phải tính đúng trên các run có so mã băm"
    assert weights["fedsgd"]["max_l2_float64"] < 1e-9, "float32 không được trộn vào con số 'trùng nhau'"
    assert any(row["dtype"] == "float32" for row in weights["fedsgd"]["rows"]), "dòng float32 vẫn phải hiện ở bảng"


def test_speed_view_flags_client_on_server_host():
    speed = speed_view("artifacts/network_speedtest")
    assert speed["available"]
    by_label = {run["label"]: run for run in speed["runs"]}
    assert by_label["wan_vm_oracle"]["via_ssh_tunnel"] is True and by_label["wan_vm_oracle"]["client_on_server_host"] is True
    vm = next(machine for machine in by_label["wan_vm_oracle"]["machines"] if machine["name"] == "VM Oracle")
    assert vm["on_server_host"] is True


def test_clients_on_one_machine_get_a_plain_warning_and_notes_are_shown_everywhere(tmp_path):
    progress = _progress()
    for client in progress["rounds"][0]["clients"]:
        client["machine_id"] = "cung-may"
    (tmp_path / rv.PROGRESS_FILE).write_text(json.dumps(progress), encoding="utf-8")
    assert rv.run_view(rv.load_run(tmp_path), now=NOW)["note"] is None
    (tmp_path / rv.NOTE_FILE).write_text("  Laptop đang mở game lúc đo.\n", encoding="utf-8")
    view = rv.run_view(rv.load_run(tmp_path), now=NOW)
    assert view["shared_machine_note"] == rv.SHARED_MACHINE_NOTE and view["note"] == "Laptop đang mở game lúc đo."
    text = "\n".join(ds.view_lines(view))
    assert rv.SHARED_MACHINE_NOTE in text and "Laptop đang mở game lúc đo." in text


def test_two_machines_do_not_get_the_shared_machine_warning():
    assert rv.run_view({"progress": _progress(), "history": None}, now=NOW)["shared_machine_note"] is None

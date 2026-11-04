"""Một chỗ tính cho web và terminal: run_view đọc đúng mọi trạng thái, không lộ IP, và số trên web trùng số terminal in ra."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from scripts import demo_show as ds
from src.federated import run_detail as rd
from src.federated import run_view as rv
from src.federated.verification_view import drift_view, fairness_view, speed_view, sweep_view, weight_view

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


ALL_ITEMS = ["client:0", "client:1", "machine:A", "machine:B", "server", *rd.PROOF_ITEMS]


def _detail_run(tmp_path, *, finished: bool = False):
    progress = _progress()
    if finished:
        progress = {**progress, "status": "finished", "calibration_metrics": {"accuracy": 0.123456},
                    "final_test_metrics": {"accuracy": 0.9, "loss": 0.3, "macro_f1": 0.89, "per_class_metrics": {"0": {"f1": 0.5, "support": 5}, "1": {"f1": 1.0, "support": 5}}}}
    (tmp_path / rv.PROGRESS_FILE).write_text(json.dumps(progress), encoding="utf-8")
    (tmp_path / rd.LOG_FILE).write_text(
        "\x1b[92mINFO \x1b[0m: nghe 0.0.0.0:18089 tu 203.0.113.9\n[!] Flower client 0123456789abcdef0123456789abcdef rejected\nluu /home/ubuntu/a/global.pt\n", encoding="utf-8")
    return tmp_path


def test_every_detail_item_builds_and_terminal_prints_the_same_lines_as_the_api(tmp_path):
    run = _detail_run(tmp_path, finished=True)
    for item in ALL_ITEMS:
        built = rd.detail_for_run(run, item, now=NOW)
        assert built is not None, item
        assert built["how"] and built["lines"] and built["title"]
        printed = rd.detail_lines(built)
        assert all(line in printed for line in built["lines"]), item
        assert built["command"].endswith(f"--detail {item}")


def test_detail_shows_real_numbers_of_the_chosen_round(tmp_path):
    built = rd.detail_for_run(_detail_run(tmp_path), "client:1", 1, now=NOW)
    text = "\n".join(built["lines"])
    assert "Vòng 1, client 1 (Máy B)" in text and "6307176 byte" in text and "KHỚP" in text and "phản hồi 20.50 giây" in text
    assert built["command"].endswith("--detail client:1 --round 1")
    assert "2/2 lượt mã băm khớp" in rd.detail_for_run(tmp_path, "hash", now=NOW)["lines"][0]


def test_detail_never_exposes_ip_cid_paths_or_calibration(tmp_path):
    run = _detail_run(tmp_path, finished=True)
    dumped = json.dumps([rd.detail_for_run(run, item, now=NOW) for item in ALL_ITEMS], ensure_ascii=False)
    for secret in ("203.0.113.9", "0123456789abcdef0123456789abcdef", "/home/", "cid-bi-mat", "calibration", "0.123456", "\\u001b", "ppm1"):
        assert secret not in dumped, secret
    log = rd.detail_for_run(run, "server", now=NOW)["log"]
    assert log["total_lines"] == 3 and log["lines"][0].startswith("INFO") and "0.0.0.0:18089" in log["lines"][0]
    assert rd.detail_for_run(run, "hash", now=NOW)["log"] is None


def test_detail_rejects_unknown_item_or_round(tmp_path):
    run = _detail_run(tmp_path)
    for item, round_no in (("client:99", None), ("machine:Z", None), ("../etc", None), ("server;ls", None), ("server", 99), ("client:0", 99)):
        assert rd.detail_for_run(run, item, round_no, now=NOW) is None, (item, round_no)


def test_detail_says_no_numbers_yet_while_waiting_and_shows_running_state(tmp_path):
    (tmp_path / rv.PROGRESS_FILE).write_text(json.dumps(_progress("waiting", rounds=[], round_replies=[])), encoding="utf-8")
    for item in ("client:0", "server", *rd.PROOF_ITEMS):
        assert rd.detail_for_run(tmp_path, item, now=NOW) is not None, item
    assert "Chưa xong vòng nào" in "\n".join(rd.detail_for_run(tmp_path, "client:0", now=NOW)["lines"])
    assert rd.detail_for_run(tmp_path, "machine:A", now=NOW) is None
    (tmp_path / rv.PROGRESS_FILE).write_text(json.dumps(_progress()), encoding="utf-8")
    assert rd.detail_for_run(tmp_path, "client:0", now=NOW)["lines"][0] == "Vòng 2 đang chạy: đã gửi về sau 12.0 giây."
    assert "chưa nhận trọng số" in rd.detail_for_run(tmp_path, "client:1", now=NOW)["lines"][0]


def test_detail_api_matches_terminal_and_rejects_bad_input(tmp_path, monkeypatch):
    from src import registry as reg
    from web.app import app
    from web.routers import flower as router
    monkeypatch.setattr(reg, "NETWORK_LIVE_DIR", str(tmp_path))
    router._detail_cache.clear()
    run = tmp_path / "20261104T000000Z_2client"
    run.mkdir()
    _detail_run(run)
    client = TestClient(app)
    reply = client.get("/api/flower/detail", params={"item": "hash"})
    assert reply.status_code == 200
    assert reply.json()["lines"] == rd.detail_for_run(run, "hash")["lines"]
    assert reply.json()["run_id"] == run.name
    for params in ({"item": "khong-co"}, {"item": "hash", "run": "../../etc"}, {"item": "hash", "run": "khong_co"}, {"item": "client:99"}, {"item": "server", "round": 99}):
        assert client.get("/api/flower/detail", params=params).status_code == 404, params
    assert client.get("/api/flower/detail").status_code == 422


def test_log_tail_drops_only_the_flower_summary_and_says_so(tmp_path):
    (tmp_path / rd.LOG_FILE).write_text("INFO : [ROUND 1]\nINFO : [SUMMARY]\nINFO : bang 1\nINFO : bang 2\n[+] xong\n", encoding="utf-8")
    log = rd.read_log_tail(tmp_path)
    assert log["lines"] == ["INFO : [ROUND 1]", "[+] xong"] and log["skipped_summary_lines"] == 3 and log["total_lines"] == 5
    assert "đã bỏ 3 dòng" in rd.log_title(log)


FAIRNESS_HEAD = "scheme,variant,split,domain,seeds,mean_%,std_between_clients_%,best_%,worst_%,gap_best_worst_points,smallest_client_%\n"


def test_fairness_view_reads_only_the_requested_split_and_terminal_prints_the_same_numbers(tmp_path):
    table = tmp_path / "fair.csv"
    table.write_text(FAIRNESS_HEAD
                     + "label_alpha_0_1,fedavg,val,all,3,90.79,3.60,94.59,84.29,10.30,84.29\n"
                     + "label_alpha_0_1,fedavg,legacy_diagnostic_test,all,3,88.58,4.67,93.44,79.90,13.54,79.90\n"
                     + "label_alpha_0_1,fedavg,calibration,all,3,11.11,1.00,1.00,1.00,1.00,1.00\n"
                     + "label_alpha_0_1,fedavg,val,field,3,22.22,1.00,1.00,1.00,1.00,1.00\n", encoding="utf-8")
    view = fairness_view(table)
    assert [row["mean"] for row in view["rows"]] == [90.79] and view["rows"][0]["worst"] == 84.29 and view["rows"][0]["best"] == 94.59
    text = "\n".join(ds.fairness_lines(table, "val"))
    assert "90.79" in text and "84.29" in text and "88.58" not in text and "11.11" not in text and "22.22" not in text
    assert "11.11" not in json.dumps(view)
    try:
        fairness_view(table, "calibration")
    except ValueError:
        pass
    else:
        raise AssertionError("fairness_view không được nhận split calibration")


def _drift_history(path, deltas):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"method": "fedavg", "scheme": "label_alpha_0_1", "runtime": "flower", "seed": 1}), encoding="utf-8")
    lines = [json.dumps({"train": {"clients": [{"delta_l2_without_bn": 2.0, "l2_to_merged_without_bn": value}, {"delta_l2_without_bn": 4.0, "l2_to_merged_without_bn": value + 2}]}}) for value in deltas]
    (path / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_client_drift_is_averaged_over_clients_rounds_and_seeds_and_skips_old_logs(tmp_path):
    from scripts import report_client_drift as rcd
    _drift_history(tmp_path / "a", [100.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _drift_history(tmp_path / "b", [100.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({"method": "fedavg", "scheme": "label_alpha_0_1"}), encoding="utf-8")
    (old / "history.jsonl").write_text(json.dumps({"train": {"clients": [{"loss": 1.0}]}}) + "\n", encoding="utf-8")
    rows, skipped = rcd.collect(tmp_path)
    assert skipped == ["old"] and len(rows) == 1
    row = rows[0]
    # Mỗi vòng: trung bình 2 client là giá trị + 1. 5 vòng cuối bỏ vòng đầu 100. Hai seed: 2 và 4.
    assert row["seeds"] == 2 and row["rounds"] == 6 and row["to_merged_last5"] == 3.0 and row["changed_mean"] == 3.0
    csv_path = tmp_path / "drift.csv"
    import csv as csv_module
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=rcd.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    view = drift_view(csv_path)
    assert view["rows"][0]["level"] == "α=0,1" and view["rows"][0]["to_merged_last5"] == 3.0
    assert "3.000" in "\n".join(ds.drift_lines(csv_path))


def test_real_fairness_and_drift_files_show_the_gap_widening():
    fair = {(row["variant"], row["scheme"]): row for row in fairness_view("artifacts/sweep_v4/table_fairness.csv")["rows"]}
    assert fair[("fedavg", "label_alpha_100_0")]["std"] < fair[("fedavg", "label_alpha_0_1")]["std"]
    drift = drift_view("artifacts/sweep_v4/client_drift.csv")["rows"]
    values = [row["to_merged_last5"] for row in drift]
    assert [row["level"] for row in drift] == ["chia đều", "α=100", "α=10", "α=1", "α=0,5", "α=0,1"] and values == sorted(values)

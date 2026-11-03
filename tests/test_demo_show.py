"""Công cụ in kết quả khi demo (scripts/demo_show.py): phải báo đúng khi băm khớp và khi băm sai, không in "khớp" cho mọi trường hợp."""
from __future__ import annotations

import json

import torch

from scripts import demo_show as ds


def _history(client_hash: str = "abc", server_hash: str = "abc", ok: bool = True) -> dict:
    client = {"client_id": 0, "num_examples": 10, "local_train_time_s": 1.5, "probe_rtt_ms": 3.0, "received_learnable_update_norm": 0.1, "upload_hash_ok": ok,
              "client_reported_hash": client_hash, "server_recomputed_hash": server_hash}
    return {"num_clients_required": 1, "rounds": 1, "seed": 42, "server_address": "x", "final_model_hash": "f" * 20,
            "audit_rounds": [{"round": 1, "accepted_clients": 1, "rejected_clients": 0, "round_fit_elapsed_ms": 2000.0, "server_aggregate_ms": 1.0, "transfer_estimate_ms": 100.0,
                              "parent_model_hash": "p" * 20, "global_model_hash": "g" * 20, "clients": [client]}]}


def test_flower_table_reports_matching_and_mismatching_hashes():
    good = "\n".join(ds.flower_lines(_history()))
    assert "KHỚP" in good and "1/1 lượt băm khớp" in good and "học 1.50 giây" in good and "RTT 3.0 ms" in good
    bad = "\n".join(ds.flower_lines(_history(server_hash="zzz")))
    assert "SAI" in bad and "0/1 lượt băm khớp" in bad
    flagged = "\n".join(ds.flower_lines(_history(ok=False)))
    assert "SAI" in flagged


def test_flower_table_uses_same_rtt_field_as_report_and_shows_speed():
    history = _history()
    client = history["audit_rounds"][0]["clients"][0]
    client.update(probe_rtt_persistent_ms=1425.4, probe_up_mb_per_s=2.51, probe_down_mb_per_s=0.45)
    history["losses_distributed"] = [[1, 3.271], [2, 2.304]]
    text = "\n".join(ds.flower_lines(history))
    assert "3.27 -> 2.30" in text
    assert "RTT 1425.4 ms" in text and "lên 2.51 MiB/s, xuống 0.45 MiB/s" in text and "RTT 3.0 ms" not in text


def test_sweep_reads_only_all_test_rows_and_computes_gap(tmp_path):
    table = tmp_path / "table.csv"
    names = [("centralized", ""), ("fedavg+flower", "iid"), ("fedavg", "label_alpha_100_0"), ("fedavg", "label_alpha_10_0"), ("fedavg", "label_alpha_1_0"), ("fedavg", "label_alpha_0_5"), ("fedavg", "label_alpha_0_1"), ("local", "label_alpha_0_1")]
    rows = ["scheme,variant,seeds,part,n,top1_mean_%,gap_top1_points"] + [f"{scheme},{variant},3,all_test,100,{90 - i}.0,{i}.01" for i, (variant, scheme) in enumerate(names)] + [",centralized,3,plantvillage_test,50,11.0,"]
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    text = "\n".join(ds.sweep_lines(table))
    assert " 90.00%" in text and "11.00" not in text and "thấp hơn tập trung  7.01 điểm" in text


def test_compare_separates_learnable_from_other_and_counts_identical():
    a = {"w": torch.zeros(4), "bn": torch.zeros(2), "same": torch.ones(1), "n": torch.tensor(1)}
    b = {"w": torch.tensor([3.0, 4.0, 0.0, 0.0]), "bn": torch.ones(2), "same": torch.ones(1), "n": torch.tensor(9)}
    result = ds.compare_states(a, b, {"w", "same"})
    assert abs(result["l2_learnable"] - 5.0) < 1e-9
    assert abs(result["l2_other"] - 2 ** 0.5) < 1e-9
    assert (result["identical_tensors"], result["float_tensors"]) == (1, 3)
    assert result["largest"][0][1] == "w"


def test_run_table_counts_transport_checks_and_skips_centralized_fields(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"method": "fedavg", "runtime": "flower", "scheme": "s", "seed": 1, "gpu": None}), encoding="utf-8")
    client = {"client": 0, "loss": 1.0, "steps": 2, "train_seconds": 1.0, "delta_l2_without_bn": 0.5, "l2_to_merged_without_bn": 0.2, "bytes_up": 1000000, "bytes_down": 2000000, "transport_down_exact": True, "transport_up_exact": True}
    row = {"step": 0, "lr": 0.1, "val": {"top1": 0.5}, "seconds": 1.0, "train": {"clients": [client], "aggregation_check": {"max_abs_diff": 1e-8}}}
    (run / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    text = "\n".join(ds.run_lines(run, 1))
    assert "2/2 lượt khớp" in text and "sai số gộp 1.00e-08" in text and "cách mô hình chung 0.200" in text and "lên 1.0 MB" in text

    central = tmp_path / "c"
    central.mkdir()
    (central / "config.json").write_text(json.dumps({"method": "centralized", "scheme": None, "seed": 1, "gpu": None}), encoding="utf-8")
    (central / "history.jsonl").write_text(json.dumps({"step": 0, "lr": 0.1, "val": {"top1": 0.5}, "seconds": 1.0, "train": {"loss": 1.0}}) + "\n", encoding="utf-8")
    text = "\n".join(ds.run_lines(central, 1))
    assert "băm" not in text.lower() and "một tiến trình" in text

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


def test_sweep_keeps_parts_apart_and_prints_gap_from_the_table(tmp_path):
    table = tmp_path / "table.csv"
    rows = ["scheme,variant,seeds,part,n,top1_mean_%,top1_std_%,macro_f1_mean_%,gap_top1_points"]
    parts = {"all_test": 90, "plantvillage_test": 80, "field_same_source_test": 70, "plantdoc_holdout_v3": 60}
    for i, (variant, scheme, _) in enumerate(ds.SWEEP_ROWS):
        for part, base in parts.items():
            rows.append(f"{scheme},{variant},3,{part},100,{base - i}.0,0.5,{base - i - 1}.0,{i}.01")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    text = "\n".join(ds.sweep_lines(table))
    assert "90.00 ± 0.50%" in text and "thấp hơn tập trung  2.01 điểm" in text
    assert "80.00" in text and "70.00" in text and "60.00" in text
    central = next(line for line in text.splitlines() if "Tập trung, học bằng AdamW" in line)
    assert "thấp hơn" not in central


def test_fairness_uses_only_the_test_split_for_all_domains(tmp_path):
    table = tmp_path / "fair.csv"
    table.write_text("scheme,variant,split,domain,mean_%,std_between_clients_%,gap_best_worst_points,smallest_client_%\n"
                     "label_alpha_0_1,fedavg,legacy_diagnostic_test,all,88.58,4.67,13.54,79.90\n"
                     "label_alpha_0_1,fedavg,val,all,11.00,1.00,1.00,1.00\n"
                     "label_alpha_0_1,fedavg,legacy_diagnostic_test,field,22.00,1.00,1.00,1.00\n", encoding="utf-8")
    text = "\n".join(ds.fairness_lines(table))
    assert "88.58" in text and "13.54" in text and "79.90" in text and "11.00" not in text and "22.00" not in text


def _partition_block(n_k, field=None, disjoint=True):
    return {"n_k": n_k, "size_ratio_max_over_min": max(n_k) / min(n_k), "audit": {"disjoint": disjoint, "union_equals_train": True}, "heterogeneity": {"field_fraction": field or {}}}


def test_partitions_report_sizes_field_share_and_failed_audit(tmp_path):
    field = {"client_00": 0.0, "client_01": 0.8264}
    schemes = {key: _partition_block([10, 20]) for key in ("label_alpha_100_0", "label_alpha_10_0", "label_alpha_1_0", "label_alpha_0_5")}
    schemes["quantity_alpha_0_5"] = _partition_block([2413, 18554])
    schemes["source_skew"] = _partition_block([50, 50], field)
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schemes": schemes}), encoding="utf-8")
    text = "\n".join(ds.partition_lines(path))
    assert "gấp 7.7 lần" in text and "0%, 83%" in text and "CẢNH BÁO" not in text
    schemes["source_skew"]["audit"]["disjoint"] = False
    path.write_text(json.dumps({"schemes": schemes}), encoding="utf-8")
    assert "CẢNH BÁO" in "\n".join(ds.partition_lines(path))


def test_compression_prints_measured_numbers_and_says_estimate(tmp_path):
    row = {"payload_bytes": 1000000, "payload_ratio": 0.5, "top1": 0.93, "predictions_changed": 7}
    onnx_row = {"top1": 0.9}
    report = {"count": 100, "models": {"fedavg": {"weights_transport": {"fp32": {**row, "payload_ratio": 1.0}, "fp16": row, "delta_int8_per_tensor": row},
                                                     "latency": {"onnx_fp32_ms": 6.0, "onnx_int8_se_gemm_pw_late_ms": 5.0, "onnx_int8_all_ms": 8.0},
                                                     "onnx": {"fp32_bytes": 6000000, "variants": {"int8_se_gemm_pw_late": {"bytes": 2000000}, "int8_all": {"bytes": 1000000}},
                                                              "onnx_fp32": onnx_row, "onnx_int8_se_gemm_pw_late": onnx_row, "onnx_int8_all": {"top1": 0.1}}}}}
    path = tmp_path / "gd4.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    text = "\n".join(ds.compression_lines(path))
    assert "200 MB" in text and "ước tính" in text and "CHƯA học với nén" in text and "10.00%" in text


def test_speed_table_takes_medians_and_flags_mismatch():
    history = {"audit_rounds": []}
    for round_id, (slow, fast, rtt) in enumerate([(20.0, 1.0, 900.0), (18.0, 0.9, 1400.0), (15.0, 0.8, 1500.0)]):
        common = {"num_examples": 380, "probe_up_mb_per_s": 2.5, "probe_down_mb_per_s": 0.5, "upload_hash_ok": True, "client_reported_hash": "a", "server_recomputed_hash": "a"}
        bad = {**common, "server_recomputed_hash": "b"} if round_id == 0 else common
        history["audit_rounds"].append({"clients": [{"client_id": 0, "local_train_time_s": slow, "probe_rtt_persistent_ms": rtt, **common}, {"client_id": 1, "local_train_time_s": fast, "probe_rtt_persistent_ms": 0.1, **bad}]})
    text = "\n".join(ds.speed_lines(history, {0: "Laptop 1"}))
    assert "Laptop 1" in text and "client 1" in text and "18.00 giây" in text and "1400.0 ms" in text
    assert "khớp 3/3" in text and "khớp 2/3" in text and "gấp 20.0 lần" in text


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


def test_flower_table_shows_server_measured_numbers_and_norm_check():
    history = _history()
    audit = history["audit_rounds"][0]
    audit["server_validation_metrics"] = {"accuracy": 0.4321, "loss": 2.345, "macro_f1": 0.3}
    client = audit["clients"][0]
    client.update(local_train_time_source="server_estimate", response_ms=12500.0, received_update_norm=3.1416, update_norm_reported=3.1416, update_norm_matches=True)
    good = "\n".join(ds.flower_lines(history))
    assert "độ chính xác 43.21%" in good and "máy chủ ước lượng" in good and "phản hồi 12.50 giây" in good
    assert "client khai 3.1416 / máy chủ tự tính 3.1416 -> KHỚP" in good
    client.update(update_norm_reported=6.2832, update_norm_matches=False)
    assert "-> SAI" in "\n".join(ds.flower_lines(history))


def test_flower_table_survives_a_round_without_transfer_estimate():
    history = _history()
    history["audit_rounds"][0]["transfer_estimate_ms"] = None
    assert "chưa ước tính" in "\n".join(ds.flower_lines(history))

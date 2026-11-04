"""Trao đổi trọng số giữa server và client: không đổi một byte nào, và băm chiều client -> server được kiểm."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from src.federated.server_app import AuditedFlowerFedAvg, expected_client_contract
from src.model.mobilenetv3 import compute_model_hash, create_model, get_parameters, set_parameters
from web.network_demo import PARTITION


def _trained_like_state() -> dict[str, torch.Tensor]:
    """State dict có đủ kiểu dữ liệu thật: float32 (trọng số, BN mean/var) và int64 vô hướng (num_batches_tracked)."""
    torch.manual_seed(7)
    model = create_model(num_classes=38, pretrained=False)
    model.train()
    for _ in range(2):  # chạy BN để running_mean/var và num_batches_tracked khác giá trị khởi tạo
        model(torch.randn(4, 3, 64, 64))
    return get_parameters(model)


def test_flower_serialisation_round_trip_is_byte_identical():
    state = _trained_like_state()
    arrays = [v.numpy() for v in state.values()]
    back = parameters_to_ndarrays(ndarrays_to_parameters(arrays))
    assert len(back) == len(arrays) == 244
    for key, before, after in zip(state, arrays, back):
        assert before.dtype == after.dtype, key
        assert before.shape == after.shape, key
        assert before.tobytes() == after.tobytes(), key
    kinds = {a.dtype.str for a in arrays}
    assert kinds == {"<f4", "<i8"}  # chỉ float32 và int64: không có float16/nén ở đâu trong đường đi
    assert compute_model_hash(dict(zip(state, back))) == compute_model_hash(state)


def test_get_then_set_parameters_restores_model_exactly():
    state = _trained_like_state()
    fresh = create_model(num_classes=38, pretrained=False)
    set_parameters(fresh, {k: v.numpy() for k, v in state.items()})
    restored = get_parameters(fresh)
    assert all(torch.equal(state[k], restored[k]) for k in state)
    assert compute_model_hash(restored) == compute_model_hash(state)


def test_hash_changes_when_a_single_weight_bit_changes_or_key_order_swaps():
    state = _trained_like_state()
    reference = compute_model_hash(state)
    tampered = {k: v.clone() for k, v in state.items()}
    first = next(k for k, v in tampered.items() if v.is_floating_point())
    flat = tampered[first].view(-1).view(torch.int32)
    flat[0] ^= 1  # đổi đúng 1 bit
    assert compute_model_hash(tampered) != reference
    # hai tensor cùng hình dạng bị đổi chỗ cho nhau (cùng shape nên qua được kiểm shape) phải đổi băm
    same_shape = [k for k, v in state.items() if v.shape == state[first].shape and k != first and v.is_floating_point()]
    if same_shape:
        swapped = dict(state)
        swapped[first], swapped[same_shape[0]] = state[same_shape[0]], state[first]
        assert compute_model_hash(swapped) != reference


def _round(strategy_kwargs: dict, client_hash_for: dict[int, str | None]):
    model = create_model(num_classes=38, pretrained=False)
    parent = get_parameters(model)
    arrays = [v.detach().cpu().numpy() for v in parent.values()]
    expected = expected_client_contract(PARTITION, 2)
    # Số đo mạng do SERVER thực hiện (ProbeServer.take_measurement); thay bằng giả lập có số cố định để kiểm cách server ghi.
    measured = {"rtt_ms_p50": 2.0, "up_mb_per_s_net": 11.5, "down_mb_per_s_net": 20.0, "payload_bytes": 6_275_944, "probe_total_ms": 100.0}
    probe_stub = SimpleNamespace(take_measurement=lambda round_num, client_id: dict(measured))
    strategy = AuditedFlowerFedAvg.build(
        num_clients=2, initial_parameters=ndarrays_to_parameters(arrays),
        expected_clients=expected, expected_smoke=True, probe_server=probe_stub, **strategy_kwargs,
    )
    strategy._parent_by_round[1] = {key: value.copy() for key, value in zip(parent, arrays)}
    parent_hash = compute_model_hash(parent)
    strategy._parent_hash_by_round[1] = parent_hash
    strategy._down_payload_bytes_by_round[1] = sum(len(t) for t in ndarrays_to_parameters(arrays).tensors)
    results = []
    for cid in range(2):
        client_arrays = [v + 1e-4 if v.dtype.kind == "f" else v for v in arrays]  # có học một chút, không bị coi là no_train
        metadata = {
            "client_id": cid, "round": 1, "model_version": 0, "local_epochs": 1, "training_seed": 42,
            "parent_model_hash": parent_hash,
            "partition_id": expected[cid]["partition_id"],
            "partition_config_sha256": expected[cid]["partition_config_sha256"],
            "client_manifest_sha256": expected[cid]["client_manifest_sha256"],
            "smoke": True, "data_mode": "synthetic",
            # Số do client tự khai: server phải BỎ QUA hết (nó tự đo thời gian, tốc độ mạng và không nhận độ chính xác từ client).
            "client_wall_ms": 1.0, "probe_up_mb_per_s": 999.0, "probe_down_mb_per_s": 999.0, "probe_rtt_ms": 0.001,
            "local_train_time_s": 0.001, "accuracy": 1.0, "loss": 0.0,
        }
        sent_hash = client_hash_for.get(cid, "honest")
        if sent_hash == "honest":
            metadata["hash"] = compute_model_hash(dict(zip(parent, client_arrays)))
        elif sent_hash is not None:
            metadata["hash"] = sent_hash
        fit = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(client_arrays), expected[cid]["smoke_examples"], metadata)
        results.append((SimpleNamespace(cid=f"transport-{cid}"), fit))
        strategy._response_ms[(1, f"transport-{cid}")] = 1500.0 + cid       # server bấm giờ phản hồi của từng client
    return strategy, results


def test_server_accepts_matching_upload_hash_and_records_real_payload_bytes():
    strategy, results = _round({}, {})
    strategy.aggregate_fit(1, results, [])
    record = strategy.audit_rounds[0]
    for client in record["clients"]:
        assert client["upload_hash_ok"] is True
        assert client["client_reported_hash"] == client["server_recomputed_hash"]
        assert client["payload_bytes_up"] > 6_000_000  # ~6,3 MB thật trên dây, không phải số tự suy
        assert client["payload_bytes_down"] == client["payload_bytes_up"]  # cùng kiến trúc, cùng dtype
        assert client["probe_up_mb_per_s"] == 11.5 and client["probe_down_mb_per_s"] == 20.0       # số server đo, không phải 999 client khai
        assert client["probe_rtt_persistent_ms"] == 2.0 and client["probe_rtt_ms"] == 2.0
        assert client["local_train_time_s"] != 0.001 and client["local_train_time_source"] == "server_estimate"
        assert client["response_ms"] in (1500.0, 1501.0)
        assert not {"client_wall_ms", "client_decode_ms", "client_encode_ms", "accuracy", "loss"} & set(client)
    assert record["slowest_response_ms"] == 1501.0
    assert record["server_aggregate_ms"] > 0 and record["transfer_estimate_ms"] is not None


def test_server_rejects_upload_whose_hash_does_not_match_the_bytes_it_received(capsys):
    strategy, results = _round({}, {1: "0" * 64})
    with pytest.raises(RuntimeError, match="requires exactly 2 accepted clients; received=2 accepted=1"):
        strategy.aggregate_fit(1, results, [])
    assert "upload_hash_mismatch" in capsys.readouterr().out


def test_hash_with_wrong_format_is_rejected_and_not_echoed(capsys):
    strategy, results = _round({}, {1: "x" * 5000 + "\n[!] giả log"})
    with pytest.raises(RuntimeError, match="accepted=1"):
        strategy.aggregate_fit(1, results, [])
    out = capsys.readouterr().out
    assert "upload_hash_mismatch" in out and "giả log" not in out and "xxxxx" not in out


def test_missing_client_hash_is_recorded_as_unknown_not_as_verified():
    strategy, results = _round({}, {0: None, 1: None})
    strategy.aggregate_fit(1, results, [])
    assert [c["upload_hash_ok"] for c in strategy.audit_rounds[0]["clients"]] == [None, None]


def test_server_records_download_check_and_independent_merge_check():
    strategy, results = _round({}, {})
    strategy.aggregate_fit(1, results, [])
    record = strategy.audit_rounds[0]
    for client in record["clients"]:
        assert client["parent_hash_ok"] is True
        assert client["reported_parent_hash"] == record["parent_model_hash"]
    check = record["aggregation_check"]
    assert check["tensors_compared"] > 0 and 0 <= check["max_abs_diff"] <= 1e-6


def test_independent_merge_check_reports_a_wrong_merge():
    from src.federated.algorithms import aggregation_error
    updates = [{"num_examples": 1, "weights": {"w": torch.tensor([1.0, 3.0])}}, {"num_examples": 3, "weights": {"w": torch.tensor([5.0, 3.0])}}]
    assert aggregation_error(updates, {"w": torch.tensor([4.0, 3.0])})["max_abs_diff"] == 0.0
    assert aggregation_error(updates, {"w": torch.tensor([3.0, 3.0])})["max_abs_diff"] == 1.0   # trung bình thường, không theo số ảnh: phải bị phát hiện


def test_client_sending_wrong_parent_hash_is_rejected_and_recorded():
    strategy, results = _round({"reject_suspicious_updates": True}, {})
    results[0][1].metrics["parent_model_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="accepted=1"):
        strategy.aggregate_fit(1, results, [])
    clients = strategy.failed_rounds[-1]["clients"]
    assert sorted(client["parent_hash_ok"] for client in clients) == [False, True]


def test_tamper_after_hash_changes_exactly_one_number_so_the_hash_no_longer_matches():
    from src.federated.faulty_client import TAMPER_AFTER_HASH, apply_fault_mode, tamper_after_hash
    weights = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    assert apply_fault_mode(TAMPER_AFTER_HASH, weights, weights) is weights
    changed = tamper_after_hash(weights)
    assert compute_model_hash(changed) != compute_model_hash(weights)
    assert int((changed["a"] != weights["a"]).sum()) == 1 and torch.equal(changed["b"], weights["b"])
    assert weights["a"][0].item() == 1.0

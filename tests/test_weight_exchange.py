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
    strategy = AuditedFlowerFedAvg.build(
        num_clients=2, initial_parameters=ndarrays_to_parameters(arrays),
        expected_clients=expected, expected_smoke=True, **strategy_kwargs,
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
            "client_wall_ms": 1500.0 + cid, "probe_up_mb_per_s": 11.5, "probe_down_mb_per_s": -1.0,
        }
        sent_hash = client_hash_for.get(cid, "honest")
        if sent_hash == "honest":
            metadata["hash"] = compute_model_hash(dict(zip(parent, client_arrays)))
        elif sent_hash is not None:
            metadata["hash"] = sent_hash
        fit = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(client_arrays), expected[cid]["smoke_examples"], metadata)
        results.append((SimpleNamespace(cid=f"transport-{cid}"), fit))
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
        assert client["probe_up_mb_per_s"] == 11.5
        assert client["probe_down_mb_per_s"] is None  # -1 = không đo, không ghi thành 0
    assert record["slowest_client_wall_ms"] == 1501.0
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

"""Server rejects malformed Flower updates before any aggregate is committed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from src.federated.server_app import AuditedFlowerFedAvg, expected_client_contract
from src.model.mobilenetv3 import compute_model_hash, create_model, get_parameters
from web.network_demo import PARTITION


@pytest.mark.parametrize("fault", ["duplicate", "wrong_round", "wrong_hash", "wrong_samples", "missing", "dtype", "nan"])
def test_bad_update_aborts_whole_round(fault: str):
    model = create_model(num_classes=38, pretrained=False)
    parent = get_parameters(model)
    arrays = [value.detach().cpu().numpy() for value in parent.values()]
    initial = ndarrays_to_parameters(arrays)
    expected = expected_client_contract(PARTITION, 2)
    strategy = AuditedFlowerFedAvg.build(
        num_clients=2,
        initial_parameters=initial,
        expected_clients=expected,
        expected_smoke=True,
    )
    strategy._parent_by_round[1] = {key: value.copy() for key, value in zip(parent, arrays)}
    parent_hash = compute_model_hash(parent)
    results = []
    for cid in range(2):
        metadata = {
            "client_id": cid,
            "round": 1,
            "model_version": 0,
            "local_epochs": 1,
            "training_seed": 42,
            "parent_model_hash": parent_hash,
            "partition_id": expected[cid]["partition_id"],
            "partition_config_sha256": expected[cid]["partition_config_sha256"],
            "client_manifest_sha256": expected[cid]["client_manifest_sha256"],
            "smoke": True,
            "data_mode": "synthetic",
        }
        samples = expected[cid]["smoke_examples"]
        client_arrays = arrays
        if cid == 1:
            if fault == "duplicate": metadata["client_id"] = 0
            if fault == "wrong_round": metadata["round"] = 2
            if fault == "wrong_hash": metadata["parent_model_hash"] = "0" * 64
            if fault == "wrong_samples": samples += 1
            if fault in {"dtype", "nan"}:
                client_arrays = [value.copy() for value in arrays]
                if fault == "dtype": client_arrays[0] = client_arrays[0].astype(np.float64)
                if fault == "nan": client_arrays[0].flat[0] = np.nan
        fit = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(client_arrays), samples, metadata)
        results.append((SimpleNamespace(cid=f"transport-{cid}"), fit))
    if fault == "missing":
        results.pop()

    with pytest.raises(RuntimeError):
        strategy.aggregate_fit(1, results, [])
    assert strategy.audit_rounds == []
    assert strategy.final_parameters is initial


def test_reject_policy_aborts_round_instead_of_aggregating_the_rest():
    """Hợp đồng strict-round: update dị thường bị loại => không đủ N client hợp lệ => hủy vòng, không gộp phần còn lại."""
    model = create_model(num_classes=38, pretrained=False)
    parent = get_parameters(model)
    arrays = [value.detach().cpu().numpy() for value in parent.values()]
    initial = ndarrays_to_parameters(arrays)
    expected = expected_client_contract(PARTITION, 2)
    strategy = AuditedFlowerFedAvg.build(
        num_clients=2,
        initial_parameters=initial,
        expected_clients=expected,
        expected_smoke=True,
        reject_suspicious_updates=True,
        max_update_norm=1.0,
    )
    strategy._parent_by_round[1] = {key: value.copy() for key, value in zip(parent, arrays)}
    parent_hash = compute_model_hash(parent)
    results = []
    for cid in range(2):
        metadata = {
            "client_id": cid, "round": 1, "model_version": 0, "local_epochs": 1, "training_seed": 42,
            "parent_model_hash": parent_hash,
            "partition_id": expected[cid]["partition_id"],
            "partition_config_sha256": expected[cid]["partition_config_sha256"],
            "client_manifest_sha256": expected[cid]["client_manifest_sha256"],
            "smoke": True, "data_mode": "synthetic",
        }
        # client 0: update nhỏ nhưng khác 0 (không bị coi là no_train); client 1: update khổng lồ vượt ngưỡng norm
        shift = 50.0 if cid == 1 else 1e-4
        client_arrays = [v + shift if v.dtype.kind == "f" else v for v in arrays]
        fit = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(client_arrays), expected[cid]["smoke_examples"], metadata)
        results.append((SimpleNamespace(cid=f"transport-{cid}"), fit))

    with pytest.raises(RuntimeError, match="requires exactly 2 accepted clients; received=2 accepted=1"):
        strategy.aggregate_fit(1, results, [])
    assert strategy.final_parameters is initial


def _round_results(arrays, expected, parent_hash, shifts):
    """Kết quả fit cho một vòng; `shifts[cid]` là hàm (mảng gốc, chỉ số khóa) -> mảng client."""
    results = []
    for cid, shift in enumerate(shifts):
        metadata = {
            "client_id": cid, "round": 1, "model_version": 0, "local_epochs": 1, "training_seed": 42,
            "parent_model_hash": parent_hash,
            "partition_id": expected[cid]["partition_id"],
            "partition_config_sha256": expected[cid]["partition_config_sha256"],
            "client_manifest_sha256": expected[cid]["client_manifest_sha256"],
            "smoke": True, "data_mode": "synthetic",
        }
        client_arrays = [shift(v, index) for index, v in enumerate(arrays)]
        fit = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(client_arrays), expected[cid]["smoke_examples"], metadata)
        results.append((SimpleNamespace(cid=f"transport-{cid}"), fit))
    return results


def test_audit_records_hashes_and_separates_learnable_update_from_batchnorm_drift(tmp_path):
    """Bằng chứng "máy đã tính thật": mỗi vòng có mã băm mô hình cha/chung và mức đổi của THAM SỐ HỌC ĐƯỢC.

    Client 0 chỉ dịch tham số học được; client 1 chỉ dịch bộ đệm BatchNorm (giống ảnh nhiễu làm running_var trôi).
    Chuẩn update cũ vẫn cộng cả hai, còn chuẩn tham số học được của client 1 phải bằng 0."""
    import json

    from flwr.common import parameters_to_ndarrays

    model = create_model(num_classes=38, pretrained=False)
    learnable = {name for name, _ in model.named_parameters()}
    parent = get_parameters(model)
    keys = list(parent)
    arrays = [value.detach().cpu().numpy() for value in parent.values()]
    expected = expected_client_contract(PARTITION, 2)
    progress = tmp_path / "progress.json"
    strategy = AuditedFlowerFedAvg.build(
        num_clients=2, initial_parameters=ndarrays_to_parameters(arrays),
        expected_clients=expected, expected_smoke=True, progress_file=progress,
    )
    strategy._parent_by_round[1] = {key: value.copy() for key, value in zip(parent, arrays)}
    parent_hash = compute_model_hash(parent)
    strategy._parent_hash_by_round[1] = parent_hash

    def shift_learnable(v, index):
        return v + 1e-4 if v.dtype.kind == "f" and keys[index] in learnable else v

    def shift_buffers(v, index):
        return v + 1e-2 if v.dtype.kind == "f" and keys[index] not in learnable else v

    returned, _ = strategy.aggregate_fit(1, _round_results(arrays, expected, parent_hash, [shift_learnable, shift_buffers]), [])

    record = strategy.audit_rounds[0]
    averaged = dict(zip(keys, parameters_to_ndarrays(returned)))
    assert record["parent_model_hash"] == parent_hash
    assert record["global_model_hash"] == compute_model_hash(averaged)
    assert record["global_model_hash"] != parent_hash, "vòng đã gộp thì mô hình chung phải có mã băm mới"

    by_id = {client["client_id"]: client for client in record["clients"]}
    assert by_id[0]["received_learnable_update_norm"] > 0
    assert by_id[1]["received_learnable_update_norm"] == 0.0 and by_id[1]["received_update_norm"] > 0
    assert by_id[0]["received_learnable_update_norm"] <= by_id[0]["received_update_norm"] + 1e-9
    assert record["mean_received_learnable_update_norm"] == pytest.approx(
        by_id[0]["received_learnable_update_norm"] / 2)

    written = json.loads(progress.read_text(encoding="utf-8"))
    assert written["rounds"][0]["global_model_hash"] == record["global_model_hash"]

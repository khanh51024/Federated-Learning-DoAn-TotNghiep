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

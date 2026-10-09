from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import flwr as fl

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.server_app import start_flower_server


FAULTS = ["normal", "random_weights", "scaled_update", "noise_update"]


def _server(port: int, reject: bool, metrics: str, checkpoint: str) -> None:
    start_flower_server(
        server_address=f"127.0.0.1:{port}",
        num_clients=4,
        rounds=1,
        local_epochs=1,
        reject_suspicious_updates=reject,
        max_update_norm=5.0,
        metrics_file=metrics,
        checkpoint_file=checkpoint,
    )


def _client(client_id: int, port: int, fault_mode: str) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(
        client_id=client_id,
        smoke=True,
        local_epochs=1,
        fault_mode=fault_mode,
        probe_host="127.0.0.1",
    )
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=FlowerClientAdapter(core).to_client(),
    )


def _run_case(port: int, reject: bool, stem: str):
    metrics = Path(f"artifacts/metrics/{stem}.json")
    checkpoint = Path(f"artifacts/checkpoints/{stem}.pt")
    for path in (metrics, checkpoint):
        if path.exists():
            path.unlink()

    server = multiprocessing.Process(target=_server, args=(port, reject, str(metrics), str(checkpoint)))
    clients = [
        multiprocessing.Process(target=_client, args=(cid, port, fault))
        for cid, fault in enumerate(FAULTS)
    ]
    server.start()
    for proc in clients:
        proc.start()
    for proc in clients:
        proc.join(timeout=120)
    server.join(timeout=40)

    try:
        for proc in clients:
            assert proc.exitcode == 0, f"client {proc.pid} exit={proc.exitcode}"
        assert server.exitcode == 0
        assert metrics.exists()
        assert checkpoint.exists()
        return json.loads(metrics.read_text(encoding="utf-8"))
    finally:
        for proc in [*clients, server]:
            if proc.is_alive():
                proc.terminate()


def test_attack_matrix_observe_then_reject():
    observed = _run_case(18120, False, "attack_matrix_observe")
    row = observed["audit_rounds"][0]
    assert row["accepted_clients"] == 4
    assert row["suspicious_clients"] == 3
    assert row["rejected_clients"] == 0

    by_id = {r["client_id"]: r for r in row["clients"]}
    assert by_id[0]["status"] == "ok"
    for cid in (1, 2, 3):
        assert by_id[cid]["status"] == "suspicious"
        assert "update_norm_above_threshold" in by_id[cid]["validation_reason"]
        assert by_id[cid]["received_update_norm"] > 5.0

    rejected = _run_case(18122, True, "attack_matrix_reject")
    row = rejected["audit_rounds"][0]
    assert row["accepted_clients"] == 1
    assert row["suspicious_clients"] == 3
    assert row["rejected_clients"] == 3
    by_id = {r["client_id"]: r for r in row["clients"]}
    assert by_id[0]["status"] == "ok"
    for cid in (1, 2, 3):
        assert by_id[cid]["status"] == "rejected_suspicious"

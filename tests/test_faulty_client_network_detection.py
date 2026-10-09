"""Real-network verification that the audited server detects a no-train client."""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import flwr as fl

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.server_app import start_flower_server


METRICS = Path("artifacts/metrics/network_fault_detection_test.json")


def _server(port: int) -> None:
    start_flower_server(
        server_address=f"127.0.0.1:{port}",
        num_clients=2,
        rounds=1,
        local_epochs=1,
        metrics_file=METRICS,
    )


def _client(client_id: int, port: int, fault_mode: str) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(
        client_id=client_id,
        smoke=True,
        local_epochs=1,
        fault_mode=fault_mode,
    )
    adapter = FlowerClientAdapter(core)
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=adapter.to_client(),
    )


def test_server_flags_no_train_client_on_real_network():
    if METRICS.exists():
        METRICS.unlink()

    port = 18106
    server = multiprocessing.Process(target=_server, args=(port,))
    normal = multiprocessing.Process(target=_client, args=(0, port, "normal"))
    no_train = multiprocessing.Process(target=_client, args=(1, port, "no_train"))

    server.start()
    normal.start()
    no_train.start()
    normal.join(timeout=90)
    no_train.join(timeout=90)
    server.join(timeout=30)

    try:
        assert normal.exitcode == 0
        assert no_train.exitcode == 0
        assert server.exitcode == 0
        assert METRICS.exists()
        payload = json.loads(METRICS.read_text(encoding="utf-8"))
        suspicious = payload["metrics_distributed_fit"]["suspicious_clients"]
        accepted = payload["metrics_distributed_fit"]["accepted_clients"]
        assert suspicious[-1] == [1, 1]
        assert accepted[-1] == [1, 2]
    finally:
        for proc in (normal, no_train, server):
            if proc.is_alive():
                proc.terminate()

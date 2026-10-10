"""Production-path smoke test: 1 audited Flower server + 5 real client processes."""

from __future__ import annotations

import multiprocessing
import time

import flwr as fl

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.server_app import start_flower_server


def _server(port: int, metrics: str, checkpoint: str) -> None:
    start_flower_server(
        server_address=f"127.0.0.1:{port}",
        num_clients=5,
        rounds=2,
        local_epochs=1,
        metrics_file=metrics,
        checkpoint_file=checkpoint,
    )


def _client(client_id: int, port: int) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(client_id=client_id, smoke=True, local_epochs=1)
    adapter = FlowerClientAdapter(core)
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=adapter.to_client(),
    )


def test_five_clients_two_rounds_over_real_flower_network(tmp_path):
    port = 18105
    server = multiprocessing.Process(
        target=_server, args=(port, str(tmp_path / "five_clients.json"), str(tmp_path / "five_clients.pt"))
    )
    clients = [multiprocessing.Process(target=_client, args=(cid, port)) for cid in range(5)]
    server.start()
    for proc in clients:
        proc.start()

    for proc in clients:
        proc.join(timeout=120)
    server.join(timeout=40)

    try:
        for proc in clients:
            assert not proc.is_alive(), f"Client pid={proc.pid} timed out"
            assert proc.exitcode == 0, f"Client pid={proc.pid} exited with {proc.exitcode}"
        assert not server.is_alive(), "Audited Flower server timed out"
        assert server.exitcode == 0
    finally:
        for proc in clients:
            if proc.is_alive():
                proc.terminate()
        if server.is_alive():
            server.terminate()

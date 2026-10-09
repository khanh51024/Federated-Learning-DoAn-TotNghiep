"""End-to-end test for the production audited Flower server path."""

from __future__ import annotations

import multiprocessing
import time

import flwr as fl

from src.federated.client_app import PlantVillageClient, FlowerClientAdapter
from src.federated.server_app import start_flower_server


def _run_audited_server(port: int) -> None:
    start_flower_server(server_address=f"127.0.0.1:{port}", num_clients=1, rounds=2)


def _run_client(port: int) -> None:
    time.sleep(2)
    core = PlantVillageClient(client_id=0, smoke=True)
    adapter = FlowerClientAdapter(core)
    fl.client.start_client(server_address=f"127.0.0.1:{port}", client=adapter.to_client())


def test_audited_server_validates_two_network_rounds():
    port = 18100
    server = multiprocessing.Process(target=_run_audited_server, args=(port,))
    client = multiprocessing.Process(target=_run_client, args=(port,))
    server.start()
    client.start()
    client.join(timeout=70)
    server.join(timeout=30)
    try:
        assert not client.is_alive(), "Audited Flower client timed out"
        assert not server.is_alive(), "Audited Flower server timed out"
        assert client.exitcode == 0
        assert server.exitcode == 0
    finally:
        if client.is_alive():
            client.terminate()
        if server.is_alive():
            server.terminate()

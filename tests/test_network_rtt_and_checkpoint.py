from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import flwr as fl
import torch

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.server_app import start_flower_server
from src.model.mobilenetv3 import create_model, set_parameters


def _server(port: int, metrics: str, checkpoint: str) -> None:
    start_flower_server(
        server_address=f"127.0.0.1:{port}",
        num_clients=2,
        rounds=1,
        local_epochs=1,
        metrics_file=metrics,
        checkpoint_file=checkpoint,
    )


def _client(client_id: int, port: int) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(
        client_id=client_id,
        smoke=True,
        local_epochs=1,
        probe_host="127.0.0.1",
    )
    adapter = FlowerClientAdapter(core)
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=adapter.to_client(),
    )


def test_per_client_probe_rtt_and_network_checkpoint_are_real(tmp_path):
    METRICS = tmp_path / "network_rtt_checkpoint_test.json"
    CHECKPOINT = tmp_path / "network_rtt_checkpoint_test.pt"

    port = 18110
    server = multiprocessing.Process(target=_server, args=(port, str(METRICS), str(CHECKPOINT)))
    clients = [multiprocessing.Process(target=_client, args=(cid, port)) for cid in range(2)]
    server.start()
    for proc in clients:
        proc.start()
    for proc in clients:
        proc.join(timeout=90)
    server.join(timeout=30)

    try:
        for proc in clients:
            assert proc.exitcode == 0
        assert server.exitcode == 0
        assert METRICS.exists()
        assert CHECKPOINT.exists()

        payload = json.loads(METRICS.read_text(encoding="utf-8"))
        assert payload["probe_port"] == port + 1
        audit = payload["audit_rounds"]
        assert len(audit) == 1
        rows = sorted(audit[0]["clients"], key=lambda row: row["client_id"])
        assert [row["client_id"] for row in rows] == [0, 1]
        for row in rows:
            assert row["probe_rtt_ms"] is not None
            assert row["probe_rtt_ms"] > 0.0
            assert row["local_train_time_s"] >= 0.0
        assert audit[0]["mean_probe_rtt_ms"] > 0.0

        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        assert ckpt["source"] == "flower_network_fedavg"
        assert ckpt["rounds"] == 1
        assert ckpt["num_clients"] == 2
        model = create_model(num_classes=38, pretrained=False)
        set_parameters(model, ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 224, 224))
        assert tuple(logits.shape) == (1, 38)
    finally:
        for proc in [*clients, server]:
            if proc.is_alive():
                proc.terminate()

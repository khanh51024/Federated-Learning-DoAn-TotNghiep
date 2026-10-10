from __future__ import annotations

import csv
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import flwr as fl
import pytest
from PIL import Image

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.server_app import start_flower_server


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)


def _server(port: int, eval_root: str, eval_part: str, metrics: str, checkpoint: str) -> None:
    start_flower_server(
        server_address=f"127.0.0.1:{port}",
        num_clients=1,
        rounds=1,
        local_epochs=1,
        seed=42,
        evaluation_dataset_root=eval_root,
        evaluation_partition_dir=eval_part,
        require_real_evaluation=True,
        evaluation_batch_size=2,
        metrics_file=metrics,
        checkpoint_file=checkpoint,
    )


def _client(port: int) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(client_id=0, smoke=True, seed=42, probe_host="127.0.0.1")
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=FlowerClientAdapter(core).to_client(),
    )


def test_real_heldout_evaluator_produces_round_and_final_metrics(tmp_path: Path):
    eval_root = tmp_path / "heldout_images"
    (eval_root / "ClassA").mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(eval_root / "ClassA" / "val.jpg")
    Image.new("RGB", (32, 32), "yellow").save(eval_root / "ClassA" / "test.jpg")

    # Server ràng buộc client_id với partition đã công bố (partition_config.json + clients/client_XX.csv);
    # client smoke dùng partition label100 cục bộ nên eval_partition phải là bản sao cùng tên.
    real = Path(__file__).resolve().parents[1] / "data/partitions_stage2_scratch_v3/label100"
    if not (real / "partition_config.json").is_file() or not (real / "clients/client_00.csv").is_file():
        pytest.skip("cần partition label100 cục bộ (data/partitions_stage2_scratch_v3) để dựng hợp đồng client")
    eval_part = tmp_path / "label100"
    (eval_part / "clients").mkdir(parents=True)
    shutil.copy(real / "partition_config.json", eval_part / "partition_config.json")
    shutil.copy(real / "clients/client_00.csv", eval_part / "clients/client_00.csv")
    _write_manifest(
        eval_part / "global_val.csv",
        [{"relative_path": "ClassA/val.jpg", "label": 0, "class_name": "ClassA"}],
    )
    _write_manifest(
        eval_part / "global_test.csv",
        [{"relative_path": "ClassA/test.jpg", "label": 0, "class_name": "ClassA"}],
    )

    metrics = tmp_path / "network_eval.json"
    checkpoint = tmp_path / "network_eval.pt"
    port = 18141
    server = multiprocessing.Process(
        target=_server,
        args=(port, str(eval_root), str(eval_part), str(metrics), str(checkpoint)),
    )
    client = multiprocessing.Process(target=_client, args=(port,))
    server.start()
    client.start()
    client.join(timeout=90)
    server.join(timeout=40)

    try:
        assert client.exitcode == 0
        assert server.exitcode == 0
        assert metrics.exists()
        assert checkpoint.exists()

        payload = json.loads(metrics.read_text(encoding="utf-8"))
        assert payload["require_real_evaluation"] is True
        assert isinstance(payload["final_test_metrics"], dict)
        assert set(("accuracy", "loss", "macro_f1")).issubset(payload["final_test_metrics"])

        fit_metrics = payload["metrics_distributed_fit"]
        assert "server_val_accuracy" in fit_metrics
        assert "server_val_loss" in fit_metrics
        assert "server_val_macro_f1" in fit_metrics

        audit = payload["audit_rounds"][0]
        assert isinstance(audit["server_validation_metrics"], dict)
        assert "accuracy" in audit["server_validation_metrics"]
    finally:
        for proc in (client, server):
            if proc.is_alive():
                proc.terminate()

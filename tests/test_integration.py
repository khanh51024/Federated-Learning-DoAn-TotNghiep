"""
Integration test toàn bộ vòng đời huấn luyện Federated Learning (test_integration.py).
Kiểm tra phối hợp: 1 Server + 5 Clients -> 1 FL round -> FedAvg -> Cập nhật Global Model -> Lưu Checkpoint.
"""

from pathlib import Path
import pytest
import torch

from src.federated.server_app import run_federated_simulation


def test_federated_integration_smoke_5_clients(tmp_path):
    """Kiểm tra tích hợp 1 round mô phỏng 5 clients với chế độ smoke chạy nhanh trên CPU."""
    out_dir = tmp_path / "fl_test_output"
    results = run_federated_simulation(
        partition_dir="data/partitions_stage2_scratch_v3/label100",
        num_clients=5,
        rounds=1,
        local_epochs=1,
        smoke=True,
        output_dir=out_dir,
    )

    assert results["experiment"] == "fedavg"
    assert results["num_clients"] == 5
    assert results["rounds"] == 1
    assert "best_accuracy" in results
    assert len(results["history"]) == 1

    # Kiểm tra file checkpoint và metrics được tạo
    ckpt_file = out_dir / "checkpoints" / "fedavg_best.pt"
    metrics_file = out_dir / "metrics" / "fedavg_results.json"
    telemetry_file = out_dir / "metrics" / "telemetry_history.json"

    assert ckpt_file.exists(), "File checkpoint fedavg_best.pt phải được lưu"
    assert metrics_file.exists(), "File metrics fedavg_results.json phải được lưu"
    assert telemetry_file.exists(), "File telemetry_history.json phải được lưu"

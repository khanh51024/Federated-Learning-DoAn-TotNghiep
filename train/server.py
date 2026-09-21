"""
train.server: Flower ServerApp & FedAvg Aggregation Server.
Điều phối toàn bộ quá trình Federated Learning:
- Khởi tạo global model -> Phát tán weights -> Nhận client updates -> Kiểm tra hash SHA-256
- FedAvg aggregation theo n_k -> Đánh giá global test set -> Lưu checkpoint -> Ghi telemetry.
Hỗ trợ cả:
  1. Network Deployment Engine (server lắng nghe cổng IP:Port cho 5 process client độc lập).
  2. Simulation Engine (chạy trực tiếp 5 client phục vụ thử nghiệm nhanh và sweep).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .centralized import evaluate
from .client import PlantVillageClient
from .data import PlantVillageDataset
from .model import (
    compute_model_hash,
    create_model,
    get_parameters,
    set_parameters,
)
from .strategy import aggregate_fedavg
from .telemetry import ServerRoundTelemetry, TelemetryManager


def run_federated_simulation(
    partition_dir: str | Path = "data/partitions_stage2_scratch_v3/label100",
    dataset_root: Optional[str | Path] = None,
    num_clients: int = 5,
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    fault_modes: Optional[Dict[int, str]] = None,
    smoke: bool = False,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Chạy Federated Learning mô phỏng đa client có kiểm định toán học đầy đủ.
    Được dùng cho sweep tham số, kiểm thử tự động và giao diện Web.
    """
    part_path = Path(partition_dir)
    test_csv = part_path / "global_test.csv"
    output_path = Path(output_dir) if output_dir else Path("artifacts")
    checkpoints_dir = output_path / "checkpoints"
    metrics_dir = output_path / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    telemetry_mgr = TelemetryManager(metrics_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Khởi tạo Global Model
    global_model = create_model(num_classes=38, pretrained=False).to(device)
    global_weights = get_parameters(global_model)
    criterion = nn.CrossEntropyLoss()

    # Dữ liệu test chung
    test_ds = PlantVillageDataset(test_csv, dataset_root=dataset_root, is_train=False)
    if smoke:
        test_ds.samples = test_ds.samples[:38]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Khởi tạo 5 Clients
    fault_modes = fault_modes or {}
    clients: List[PlantVillageClient] = []
    for cid in range(num_clients):
        f_mode = fault_modes.get(cid, "normal")
        c = PlantVillageClient(
            client_id=cid,
            partition_dir=part_path,
            dataset_root=dataset_root,
            batch_size=batch_size,
            lr=lr,
            local_epochs=local_epochs,
            fault_mode=f_mode,
            device=device,
            smoke=smoke,
        )
        clients.append(c)

    print(f"[*] Starting Federated Simulation: {num_clients} clients, {rounds} rounds, local_epochs={local_epochs}")

    best_acc = -1.0
    best_weights = None
    round_history = []
    total_start = time.time()

    for r in range(1, rounds + 1):
        round_t_start = time.time()
        print(f"\n--- Round {r}/{rounds} ---")

        # Phát tán model và nhận updates từ từng client
        client_replies = []
        client_telemetries = []

        for c in clients:
            reply = c.fit(global_weights, round_num=r)
            client_replies.append(reply)
            client_telemetries.append(reply["telemetry"])

            # Kiểm tra tính toàn vẹn SHA-256
            computed_hash = compute_model_hash(reply["weights"])
            assert computed_hash == reply["hash"], f"SHA-256 mismatch tại client {c.client_id}!"

            print(f"Client {c.client_id:02d}: loss={reply['loss']:.4f}, acc={reply['accuracy']*100:.1f}%, UpdateNorm={reply['update_norm']:.4f}, Time={reply['train_duration_s']:.2f}s")

        # FedAvg Aggregation
        agg_t_start = time.perf_counter()
        global_weights = aggregate_fedavg(client_replies)
        agg_duration_ms = (time.perf_counter() - agg_t_start) * 1000

        # Cập nhật global model
        set_parameters(global_model, global_weights)

        # Đánh giá toàn cục trên global_test
        eval_metrics = evaluate(global_model, test_loader, criterion, device)
        total_round_duration = time.time() - round_t_start

        server_telemetry = ServerRoundTelemetry(
            round_num=r,
            aggregation_time_ms=round(agg_duration_ms, 2),
            total_round_time_s=round(total_round_duration, 2),
            clients_participating=num_clients,
            global_loss=eval_metrics["loss"],
            global_accuracy=eval_metrics["accuracy"],
            global_macro_f1=eval_metrics["macro_f1"],
            client_metrics=client_telemetries,
        )
        telemetry_mgr.record_round(server_telemetry)

        round_history.append({
            "round": r,
            "global_loss": eval_metrics["loss"],
            "global_accuracy": eval_metrics["accuracy"],
            "global_macro_f1": eval_metrics["macro_f1"],
            "aggregation_time_ms": round(agg_duration_ms, 2),
            "round_duration_s": round(total_round_duration, 2),
        })

        print(f"[+] Round {r} Global Eval: Acc={eval_metrics['accuracy']*100:.2f}%, F1={eval_metrics['macro_f1']:.4f}, Loss={eval_metrics['loss']:.4f}")

        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]
            best_weights = copy.deepcopy(global_weights)

    # Lưu best và last checkpoint
    torch.save({"model_state": best_weights or global_weights, "test_accuracy": best_acc}, checkpoints_dir / "fedavg_best.pt")
    torch.save({"model_state": global_weights, "last_accuracy": eval_metrics["accuracy"]}, checkpoints_dir / "fedavg_last.pt")

    results_payload = {
        "experiment": "fedavg",
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "best_accuracy": best_acc,
        "total_time_s": round(time.time() - total_start, 2),
        "history": round_history,
        "telemetry_summary": telemetry_mgr.get_summary(),
    }

    with open(metrics_dir / "fedavg_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[+] Federated Learning Hoàn Tất! Best Global Accuracy: {best_acc*100:.2f}%")
    return results_payload


# Flower Network Server Deployment
def start_flower_server(
    server_address: str = "0.0.0.0:8080",
    num_clients: int = 5,
    rounds: int = 5,
) -> None:
    """Khởi chạy Flower Server lắng nghe kết nối từ các client process qua IP:Port."""
    import flwr as fl
    from flwr.server.strategy import FedAvg

    print(f"[*] Starting Flower Server listening on {server_address}")
    print(f"[*] Waiting for {num_clients} clients to connect...")

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FedAvg Server")
    parser.add_argument("--mode", type=str, choices=["simulation", "deployment"], default="simulation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--port", type=str, default="8080")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.mode == "deployment":
        start_flower_server(server_address=f"0.0.0.0:{args.port}", num_clients=args.num_clients, rounds=args.rounds)
    else:
        run_federated_simulation(num_clients=args.num_clients, rounds=args.rounds, smoke=args.smoke)

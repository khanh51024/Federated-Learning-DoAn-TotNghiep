"""
train.local_only: Huấn luyện độc lập từng client (Local-only baseline).
Mỗi cơ sở/bệnh viện/trang trại chỉ tự học trên dữ liệu của mình, không chia sẻ trọng số.
Dùng làm mốc so sánh (lower bound) để chứng minh giá trị gia tăng của Federated Learning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .centralized import evaluate, train_one_epoch
from .data import PlantVillageDataset
from .model import create_model, get_parameters


def run_local_only_training(
    partition_dir: Optional[str | Path] = None,
    dataset_root: Optional[str | Path] = None,
    num_clients: int = 5,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    smoke: bool = False,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    part_path = Path(partition_dir) if partition_dir else Path("data/partitions_stage2_scratch_v3/label100")
    clients_dir = part_path / "clients"
    test_csv = part_path / "global_test.csv"

    output_path = Path(output_dir) if output_dir else Path("artifacts")
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Local-only Training: {num_clients} clients, epochs={epochs}, device={device}")

    test_ds = PlantVillageDataset(test_csv, dataset_root=dataset_root, is_train=False)
    if smoke:
        test_ds.samples = test_ds.samples[:38]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    client_results = []
    accuracies = []
    f1_scores = []
    start_time = time.time()

    for cid in range(num_clients):
        client_csv = clients_dir / f"client_{cid:02d}.csv"
        train_ds = PlantVillageDataset(client_csv, dataset_root=dataset_root, is_train=True)
        if smoke:
            train_ds.samples = train_ds.samples[:32]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = create_model(num_classes=38, pretrained=False).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        for ep in range(1, epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Đánh giá model của client trên tập global test chung
        metrics = evaluate(model, test_loader, criterion, device)
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["macro_f1"])

        client_results.append({
            "client_id": cid,
            "sample_count": len(train_ds),
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_loss": metrics["loss"],
        })
        print(f"Client {cid:02d} ({len(train_ds)} samples) -> Test Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['macro_f1']:.4f}")

    total_duration = round(time.time() - start_time, 2)
    summary_metrics = {
        "experiment": "local_only",
        "num_clients": num_clients,
        "epochs": epochs,
        "total_train_time_s": total_duration,
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "std_accuracy": round(float(np.std(accuracies)), 4),
        "min_accuracy": round(float(np.min(accuracies)), 4),
        "max_accuracy": round(float(np.max(accuracies)), 4),
        "mean_macro_f1": round(float(np.mean(f1_scores)), 4),
        "clients": client_results,
    }

    with open(metrics_dir / "local_only_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"[+] Local-only hoàn tất! Mean Acc: {summary_metrics['mean_accuracy']*100:.2f}% (std={summary_metrics['std_accuracy']*100:.2f}%)")
    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local-only Training")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_local_only_training(
        num_clients=args.num_clients,
        epochs=args.epochs,
        smoke=args.smoke,
    )

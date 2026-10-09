"""
train.client: Flower ClientApp và vòng đời huấn luyện cục bộ (Client Lifecycle).
- Nhận w_global -> Nạp model -> Train E epochs (SGD) -> Tính ||Delta_k||_2 -> Hash SHA-256 -> Gửi trả server.
- Không bao giờ gửi ảnh thô về server. Hỗ trợ chạy thành process độc lập qua IP:Port hoặc simulation.
"""

from __future__ import annotations

import argparse
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

from .centralized import evaluate, train_one_epoch
from .data import PlantVillageDataset
from .faulty_client import apply_fault_mode
from .model import (
    check_parameters_finite,
    compute_model_hash,
    create_model,
    get_parameters,
    set_parameters,
)
from .telemetry import ClientRoundTelemetry
from .weight_inspector import inspect_layer_weights


class PlantVillageClient:
    """Core ML Logic của một client Federated Learning độc lập."""

    def __init__(
        self,
        client_id: int,
        partition_dir: str | Path = "data/partitions_stage2_scratch_v3/label100",
        dataset_root: Optional[str | Path] = None,
        batch_size: int = 32,
        lr: float = 0.01,
        local_epochs: int = 1,
        fault_mode: str = "normal",
        device: Optional[torch.device] = None,
        smoke: bool = False,
    ):
        self.client_id = client_id
        self.partition_dir = Path(partition_dir)
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.fault_mode = fault_mode
        self.smoke = smoke
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Nạp dữ liệu riêng của client
        client_csv = self.partition_dir / "clients" / f"client_{client_id:02d}.csv"
        self.dataset = PlantVillageDataset(client_csv, dataset_root=dataset_root, is_train=True)
        if self.smoke:
            self.dataset.samples = self.dataset.samples[:32]
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.num_examples = max(len(self.dataset), 1)

        self.model = create_model(num_classes=38, pretrained=False).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def fit(
        self,
        global_weights: Dict[str, torch.Tensor | np.ndarray],
        round_num: int = 1,
    ) -> Dict[str, Any]:
        """Thực thi vòng lặp huấn luyện cục bộ khi nhận được weights từ server."""
        # 1. Nạp global weights và kiểm tra hash
        set_parameters(self.model, global_weights)
        recv_hash = compute_model_hash(global_weights)

        # Lưu bản sao global weights để tính Delta_k
        w_global_cpu = get_parameters(self.model)

        # 2. Huấn luyện E epochs với fresh SGD
        t_start = time.perf_counter()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        loss_accum, acc_accum = 0.0, 0.0
        for ep in range(self.local_epochs):
            l, a = train_one_epoch(self.model, self.loader, optimizer, self.criterion, self.device)
            loss_accum += l
            acc_accum += a
        local_train_duration = time.perf_counter() - t_start

        avg_loss = round(loss_accum / self.local_epochs, 4)
        avg_acc = round(acc_accum / self.local_epochs, 4)

        # 3. Trích xuất updated weights và tính ||Delta_k||_2
        updated_weights = get_parameters(self.model)
        check_parameters_finite(updated_weights)

        # Tính toán update norm tổng thể
        delta_norms = []
        for k in updated_weights.keys():
            diff = (updated_weights[k].float() - w_global_cpu[k].float()).numpy().flatten()
            delta_norms.append(np.linalg.norm(diff) ** 2)
        total_update_norm = round(float(np.sqrt(sum(delta_norms))), 4)

        # 4. Áp dụng chế độ faulty client nếu có
        final_weights = apply_fault_mode(self.fault_mode, updated_weights, w_global_cpu)
        send_hash = compute_model_hash(final_weights)

        # Tính bytes truyền đi
        bytes_sent = sum(t.numel() * t.element_size() for t in final_weights.values())

        telemetry = ClientRoundTelemetry(
            client_id=self.client_id,
            round_num=round_num,
            local_train_time_s=round(local_train_duration, 3),
            bytes_uploaded=bytes_sent,
            bytes_downloaded=bytes_sent,
            loss=avg_loss,
            accuracy=avg_acc,
            num_examples=self.num_examples,
        )

        return {
            "client_id": self.client_id,
            "weights": final_weights,
            "num_examples": self.num_examples,
            "hash": send_hash,
            "recv_hash": recv_hash,
            "update_norm": total_update_norm,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "train_duration_s": round(local_train_duration, 3),
            "telemetry": telemetry,
        }


# Flower Deployment Adapter
try:
    import flwr as fl
    from flwr.client import NumPyClient

    class FlowerClientAdapter(NumPyClient):
        def __init__(self, core_client: PlantVillageClient):
            self.core = core_client

        def get_parameters(self, config):
            weights = get_parameters(self.core.model)
            return [v.detach().cpu().numpy() for v in weights.values()]

        def fit(self, parameters, config):
            param_keys = list(self.core.model.state_dict().keys())
            weights_dict = {k: torch.from_numpy(p) for k, p in zip(param_keys, parameters)}
            round_num = config.get("round", 1)

            res = self.core.fit(weights_dict, round_num=round_num)
            updated_ndarrays = [
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                for v in res["weights"].values()
            ]
            metrics = {
                "loss": float(res["loss"]),
                "accuracy": float(res["accuracy"]),
                "update_norm": float(res["update_norm"]),
                "hash": str(res["hash"]),
            }
            return updated_ndarrays, int(res["num_examples"]), metrics

        def evaluate(self, parameters, config):
            param_keys = list(self.core.model.state_dict().keys())
            weights_dict = {k: torch.from_numpy(p) for k, p in zip(param_keys, parameters)}
            set_parameters(self.core.model, weights_dict)
            metrics = evaluate(self.core.model, self.core.loader, self.core.criterion, self.core.device)
            return float(metrics["loss"]), int(self.core.num_examples), {"accuracy": float(metrics["accuracy"])}

except ImportError:
    FlowerClientAdapter = None


def start_client_process(
    server_address: str = "127.0.0.1:8080",
    client_id: int = 0,
    partition_dir: str = "data/partitions_stage2_scratch_v3/label100",
    smoke: bool = False,
) -> None:
    """Khởi chạy client process độc lập kết nối đến Server qua IP:Port."""
    print(f"[*] Starting Flower Client {client_id} -> Server: {server_address}")
    core_client = PlantVillageClient(client_id=client_id, partition_dir=partition_dir, smoke=smoke)
    adapter = FlowerClientAdapter(core_client)
    fl.client.start_client(server_address=server_address, client=adapter.to_client())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client Process")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--partition-dir", type=str, default="data/partitions_stage2_scratch_v3/label100")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    start_client_process(
        server_address=args.server,
        client_id=args.client_id,
        partition_dir=args.partition_dir,
        smoke=args.smoke,
    )

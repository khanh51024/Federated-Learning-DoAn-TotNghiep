"""
src.federated.client_app: Flower ClientApp và vòng đời huấn luyện cục bộ (Client Lifecycle).
- Nhận w_global -> Nạp model -> Train E epochs (SGD) -> Tính ||Delta_k||_2 -> Hash SHA-256 -> Gửi trả server.
- Không bao giờ gửi ảnh thô về server. Hỗ trợ chạy thành process độc lập qua IP:Port hoặc simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.centralized.evaluate import evaluate
from src.centralized.train import train_one_epoch
from src.data.plantvillage import PlantVillageDataset
from src.model.mobilenetv3 import (
    check_parameters_finite,
    compute_model_hash,
    create_model,
    get_parameters,
    set_parameters,
)
from .faulty_client import apply_fault_mode
from .network_probe import measure_probe_rtt_ms
from .telemetry import ClientRoundTelemetry


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
        seed: int = 42,
        probe_host: Optional[str] = None,
        device: Optional[torch.device] = None,
        smoke: bool = False,
        require_real_data: bool = False,
    ):
        self.client_id = client_id
        self.partition_dir = Path(partition_dir)
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.fault_mode = fault_mode
        self.seed = int(seed)
        self.probe_host = probe_host
        self.smoke = smoke
        self.require_real_data = require_real_data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.partition_id = self.partition_dir.name
        partition_config = self.partition_dir / "partition_config.json"
        self.partition_config_sha256 = (
            hashlib.sha256(partition_config.read_bytes()).hexdigest()
            if partition_config.exists()
            else "missing"
        )

        # Nạp dữ liệu riêng của client
        client_csv = self.partition_dir / "clients" / f"client_{client_id:02d}.csv"
        self.client_manifest_sha256 = (
            hashlib.sha256(client_csv.read_bytes()).hexdigest()
            if client_csv.exists()
            else "missing"
        )
        self.dataset = PlantVillageDataset(client_csv, dataset_root=dataset_root, is_train=True)
        self.data_mode = "real" if bool(getattr(self.dataset, "real_mode", False)) else "synthetic"
        if self.require_real_data and self.data_mode != "real":
            raise RuntimeError(
                f"Client {client_id} requires real PlantVillage images, but dataset_root={dataset_root!r} "
                "did not resolve manifest paths. Refusing synthetic fallback."
            )
        if self.smoke:
            # Smoke dùng đúng một mẫu mỗi lớp hiện diện để 5 client chạy được
            # trên máy demo ít RAM; đây không phải phép huấn luyện khoa học.
            buckets: Dict[int, list] = {}
            for s in self.dataset.samples:
                buckets.setdefault(s["label"], []).append(s)
            balanced_subset = []
            for lbl in sorted(buckets.keys()):
                balanced_subset.extend(buckets[lbl][:1])
            self.dataset.samples = balanced_subset if balanced_subset else self.dataset.samples[:32]
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.seed + int(client_id) * 1009)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
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

        # 3. Trích xuất updated weights và tính ||Delta_k||_2 trước fault injection
        updated_weights = get_parameters(self.model)
        check_parameters_finite(updated_weights)

        def _update_norm(weights: Dict[str, torch.Tensor | np.ndarray]) -> float:
            delta_sq = 0.0
            for key, value in weights.items():
                tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
                parent = w_global_cpu[key].detach().cpu()
                tensor = tensor.detach().cpu()
                # Update norm is defined over trainable/floating model tensors.
                # Integer buffers such as BatchNorm.num_batches_tracked must not
                # dominate the L2 norm or disagree with the server validator.
                if not (tensor.is_floating_point() or parent.is_floating_point()):
                    continue
                diff = (tensor.float() - parent.float()).numpy().ravel()
                delta_sq += float(np.linalg.norm(diff) ** 2)
            return round(float(np.sqrt(delta_sq)), 4)

        trained_update_norm = _update_norm(updated_weights)

        # 4. Áp dụng chế độ faulty client nếu có.  Mọi telemetry/anomaly check
        # phải phản ánh payload cuối cùng thực sự gửi về server, không phải bản
        # update hợp lệ trước khi bị biến đổi.
        final_weights = apply_fault_mode(self.fault_mode, updated_weights, w_global_cpu)
        check_parameters_finite(final_weights)
        sent_update_norm = _update_norm(final_weights)
        send_hash = compute_model_hash(final_weights)

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
            trained_update_norm=trained_update_norm,
            sent_update_norm=sent_update_norm,
            update_norm=sent_update_norm,
        )

        return {
            "client_id": self.client_id,
            "round": int(round_num),
            "model_version": int(round_num - 1),
            "local_epochs": int(self.local_epochs),
            "training_seed": int(self.seed),
            "partition_id": str(self.partition_id),
            "partition_config_sha256": str(self.partition_config_sha256),
            "client_manifest_sha256": str(self.client_manifest_sha256),
            "data_mode": str(self.data_mode),
            "smoke": bool(self.smoke),
            "weights": final_weights,
            "num_examples": self.num_examples,
            "hash": send_hash,
            "recv_hash": recv_hash,
            "parent_model_hash": recv_hash,
            "trained_update_norm": trained_update_norm,
            "sent_update_norm": sent_update_norm,
            "update_norm": sent_update_norm,
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
            round_num = int(config.get("round", 1))
            configured_local_epochs = int(config.get("local_epochs", self.core.local_epochs))
            if configured_local_epochs <= 0:
                raise ValueError("local_epochs received from server must be > 0")
            server_seed = int(config.get("seed", self.core.seed))
            if server_seed != self.core.seed:
                raise ValueError(
                    f"training seed mismatch: server={server_seed}, client={self.core.seed}. "
                    "Restart client with the same --seed as the server."
                )

            probe_rtt_ms = None
            probe_port = config.get("probe_port")
            if self.core.probe_host and probe_port is not None:
                try:
                    probe_rtt_ms = float(measure_probe_rtt_ms(self.core.probe_host, int(probe_port)))
                except Exception:
                    probe_rtt_ms = None

            # Server controls the local training budget for this round. Preserve
            # the previous client setting so the adapter remains reusable.
            previous_local_epochs = self.core.local_epochs
            self.core.local_epochs = configured_local_epochs
            try:
                res = self.core.fit(weights_dict, round_num=round_num)
            finally:
                self.core.local_epochs = previous_local_epochs
            updated_ndarrays = [
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                for v in res["weights"].values()
            ]
            metrics = {
                "loss": float(res["loss"]),
                "accuracy": float(res["accuracy"]),
                "trained_update_norm": float(res["trained_update_norm"]),
                "sent_update_norm": float(res["sent_update_norm"]),
                "update_norm": float(res["sent_update_norm"]),
                "hash": str(res["hash"]),
                "parent_model_hash": str(res["parent_model_hash"]),
                "round": int(round_num),
                "model_version": int(config.get("model_version", round_num - 1)),
                "local_epochs": int(configured_local_epochs),
                "local_train_time_s": float(res["train_duration_s"]),
                "client_id": int(self.core.client_id),
                "probe_rtt_ms": float(probe_rtt_ms) if probe_rtt_ms is not None else -1.0,
                "data_mode": str(self.core.data_mode),
                "training_seed": int(self.core.seed),
                "partition_id": str(self.core.partition_id),
                "partition_config_sha256": str(self.core.partition_config_sha256),
                "client_manifest_sha256": str(self.core.client_manifest_sha256),
                "smoke": bool(self.core.smoke),
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
    server_address: str = "127.0.0.1:8089",
    client_id: int = 0,
    partition_dir: str = "data/partitions_stage2_scratch_v3/label100",
    dataset_root: Optional[str | Path] = None,
    smoke: bool = False,
    local_epochs: int = 1,
    fault_mode: str = "normal",
    seed: int = 42,
    require_real_data: bool = False,
) -> None:
    """Khởi chạy client process độc lập kết nối đến Server qua IP:Port."""
    print(f"[*] Starting Flower Client {client_id} -> Server: {server_address}")
    probe_host = server_address.rsplit(":", 1)[0]
    core_client = PlantVillageClient(
        client_id=client_id,
        partition_dir=partition_dir,
        dataset_root=dataset_root,
        smoke=smoke,
        local_epochs=local_epochs,
        fault_mode=fault_mode,
        seed=seed,
        probe_host=probe_host,
        require_real_data=require_real_data or not smoke,
    )
    adapter = FlowerClientAdapter(core_client)
    fl.client.start_client(server_address=server_address, client=adapter.to_client())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client Process")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--server", type=str, default="127.0.0.1:8089")
    parser.add_argument("--partition-dir", type=str, default="data/partitions_stage2_scratch_v3/label100")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root folder containing the 38 PlantVillage class directories")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-real-data", action="store_true", help="Fail if PlantVillage image paths do not resolve; never use synthetic fallback")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fault-mode",
        type=str,
        default="normal",
        choices=["normal", "no_train", "random_weights", "scaled_update", "noise_update"],
    )
    args = parser.parse_args()

    start_client_process(
        server_address=args.server,
        client_id=args.client_id,
        partition_dir=args.partition_dir,
        dataset_root=args.dataset_root,
        smoke=args.smoke,
        local_epochs=args.local_epochs,
        fault_mode=args.fault_mode,
        seed=args.seed,
        require_real_data=args.require_real_data,
    )

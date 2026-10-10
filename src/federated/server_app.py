"""
src.federated.server_app: Flower ServerApp & FedAvg Aggregation Server.
Điều phối toàn bộ quá trình Federated Learning:
- Khởi tạo global model -> Phát tán weights -> Nhận client updates -> Kiểm tra hash SHA-256
- FedAvg aggregation theo n_k -> Đánh giá global test set -> Lưu checkpoint -> Ghi telemetry.
Hỗ trợ cả:
  1. Network Deployment Engine (server lắng nghe cổng IP:Port cho 5 process client độc lập).
  2. Simulation Engine (chạy trực tiếp 5 client phục vụ thử nghiệm nhanh và sweep).
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import random
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

from src import registry as reg
from src.centralized.evaluate import evaluate
from src.data.plantvillage import PlantVillageDataset
from src.model.mobilenetv3 import (
    compute_model_hash,
    create_model,
    get_parameters,
    set_parameters,
)
from .client_app import PlantVillageClient
from .network_probe import ProbeServer
from .strategy import aggregate_fedavg
from .telemetry import ServerRoundTelemetry, TelemetryManager
from .update_validator import validate_client_update


def expected_client_contract(partition_dir: str | Path, count: int) -> Dict[int, Dict[str, Any]]:
    """Bind each client ID to its published local CSV and expected sample counts."""
    directory = Path(partition_dir)
    config = directory / "partition_config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    expected: Dict[int, Dict[str, Any]] = {}
    for client_id in range(count):
        manifest = directory / "clients" / f"client_{client_id:02d}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        per_label: Dict[int, int] = {}
        for row in rows:
            label = int(row["label"])
            per_label[label] = per_label.get(label, 0) + 1
        expected[client_id] = {
            "partition_id": directory.name,
            "partition_config_sha256": config_sha,
            "client_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "full_examples": len(rows),
            "smoke_examples": sum(min(1, amount) for amount in per_label.values()),
        }
    return expected


def run_federated_simulation(
    partition_dir: str | Path = "data/partitions_stage2_scratch_v3/label100",
    dataset_root: Optional[str | Path] = None,
    num_clients: int = 5,
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 42,
    fault_modes: Optional[Dict[int, str]] = None,
    smoke: bool = False,
    require_real_data: bool = False,
    output_dir: Optional[str | Path] = None,
    step_callback: Optional[Any] = None,
    is_cancelled: Optional[Any] = None,
    reject_suspicious_updates: bool = False,
    max_update_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Chạy Federated Learning mô phỏng đa client có kiểm định toán học đầy đủ.
    Được dùng cho sweep tham số, kiểm thử tự động và giao diện Web.
    """
    if not smoke:
        require_real_data = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    part_path = Path(partition_dir)
    val_csv = part_path / "global_val.csv" if (part_path / "global_val.csv").exists() else (part_path / "global_test.csv")
    test_csv = part_path / "global_test.csv"
    output_path = Path(output_dir) if output_dir else Path(reg.ARTIFACTS_DIR)
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

    # Dữ liệu validation (dùng để chọn checkpoint giữa các vòng)
    val_ds = PlantVillageDataset(
        val_csv,
        dataset_root=dataset_root,
        is_train=False,
        require_real_images=require_real_data,
    )
    # Dữ liệu test độc lập (chỉ dùng đánh giá cuối cùng sau khi hoàn tất training)
    test_ds = PlantVillageDataset(
        test_csv,
        dataset_root=dataset_root,
        is_train=False,
        require_real_images=require_real_data,
    )

    if smoke:
        # Lấy 1 mẫu đại diện cho mỗi lớp để đánh giá đa dạng và nhanh
        buckets_val: Dict[int, list] = {}
        for s in val_ds.samples:
            buckets_val.setdefault(s["label"], []).append(s)
        val_ds.samples = [buckets_val[lbl][0] for lbl in sorted(buckets_val.keys()) if buckets_val[lbl]]

        buckets_test: Dict[int, list] = {}
        for s in test_ds.samples:
            buckets_test.setdefault(s["label"], []).append(s)
        test_ds.samples = [buckets_test[lbl][0] for lbl in sorted(buckets_test.keys()) if buckets_test[lbl]]

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
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
            seed=seed,
            device=device,
            smoke=smoke,
            require_real_data=require_real_data,
        )
        clients.append(c)

    expected_clients = expected_client_contract(part_path, num_clients)

    print(f"[*] Starting Federated Simulation: {num_clients} clients, {rounds} rounds, local_epochs={local_epochs}")

    best_acc = -1.0
    best_weights = None
    round_history = []
    total_start = time.time()

    for r in range(1, rounds + 1):
        if is_cancelled and is_cancelled():
            print(f"[!] Nhận được yêu cầu dừng tại Vòng {r}. Dừng mô phỏng FedAvg.")
            break

        round_t_start = time.time()
        print(f"\n--- Round {r}/{rounds} ---")

        # Phát tán model và nhận updates từ từng client
        client_replies = []
        client_telemetries = []
        seen_client_ids: set[int] = set()

        for c in clients:
            if is_cancelled and is_cancelled():
                break
            reply = c.fit(global_weights, round_num=r)
            client_telemetries.append(reply["telemetry"])

            client_id = int(reply.get("client_id", -1))
            if client_id != c.client_id or client_id in seen_client_ids:
                raise RuntimeError(f"Round {r}: duplicate or invalid client_id {client_id}")
            seen_client_ids.add(client_id)
            expected = expected_clients[client_id]
            expected_n = expected["smoke_examples" if smoke else "full_examples"]
            if (int(reply.get("round", -1)) != r
                    or int(reply.get("model_version", -2)) != r - 1
                    or int(reply.get("local_epochs", -1)) != local_epochs
                    or int(reply.get("training_seed", -1)) != seed):
                raise RuntimeError(f"Round {r}: client {client_id} returned stale/mismatched round metadata")
            if (bool(reply.get("smoke")) != smoke
                    or reply.get("partition_id") != expected["partition_id"]
                    or reply.get("partition_config_sha256") != expected["partition_config_sha256"]
                    or reply.get("client_manifest_sha256") != expected["client_manifest_sha256"]
                    or int(reply.get("num_examples", -1)) != expected_n):
                raise RuntimeError(f"Round {r}: client {client_id} partition/sample contract mismatch")
            if require_real_data and reply.get("data_mode") != "real":
                raise RuntimeError(f"Round {r}: client {client_id} did not use real images")

            # Kiểm tra tính toàn vẹn payload và lineage của global model.
            computed_hash = compute_model_hash(reply["weights"])
            if computed_hash != reply.get("hash"):
                raise RuntimeError(f"Round {r}: SHA-256 mismatch from client {client_id}")

            validation = validate_client_update(
                global_weights,
                reply["weights"],
                reported_parent_hash=reply.get("parent_model_hash"),
                max_update_norm=max_update_norm,
            )
            reply["validation"] = validation.to_dict()
            # Server-side value is authoritative; never trust a client-reported norm.
            reply["server_received_update_norm"] = validation.received_update_norm

            if not validation.accepted:
                raise ValueError(
                    f"Client {c.client_id} gửi update không hợp lệ: {validation.reason}"
                )
            if validation.suspicious and reject_suspicious_updates:
                print(
                    f"[!] Client {c.client_id:02d} rejected: {validation.reason}, "
                    f"server_norm={validation.received_update_norm:.4f}"
                )
                continue

            client_replies.append(reply)
            flag = "SUSPICIOUS" if validation.suspicious else "OK"
            print(
                f"Client {c.client_id:02d}: loss={reply['loss']:.4f}, "
                f"acc={reply['accuracy']*100:.1f}%, "
                f"SentNorm={reply['sent_update_norm']:.4f}, "
                f"ServerNorm={validation.received_update_norm:.4f}, "
                f"status={flag}, Time={reply['train_duration_s']:.2f}s"
            )

        if is_cancelled and is_cancelled():
            print(f"[!] Nhận được yêu cầu dừng sau vòng thu thập client tại Vòng {r}.")
            break

        if len(client_replies) != num_clients or len(seen_client_ids) != num_clients:
            raise RuntimeError(
                f"Round {r} requires exactly {num_clients} accepted clients; "
                f"received={len(seen_client_ids)} accepted={len(client_replies)}"
            )

        # FedAvg Aggregation
        agg_t_start = time.perf_counter()
        global_weights = aggregate_fedavg(client_replies, parent_weights=global_weights)
        agg_duration_ms = (time.perf_counter() - agg_t_start) * 1000

        # Cập nhật global model
        set_parameters(global_model, global_weights)

        # Đánh giá toàn cục trên tập validation độc lập (không dùng test set để chọn checkpoint)
        eval_metrics = evaluate(global_model, val_loader, criterion, device)

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

        round_record = {
            "round": r,
            "total_rounds": rounds,
            "global_loss": round(eval_metrics["loss"], 4),
            "global_accuracy": round(eval_metrics["accuracy"], 4),
            "global_macro_f1": round(eval_metrics["macro_f1"], 4),
            "aggregation_time_ms": round(agg_duration_ms, 2),
            "round_duration_s": round(total_round_duration, 2),
            "status": "completed",
        }
        round_history.append(round_record)

        print(f"[+] Round {r} Global Val: Acc={eval_metrics['accuracy']*100:.2f}%, F1={eval_metrics['macro_f1']:.4f}, Loss={eval_metrics['loss']:.4f}")

        if step_callback:
            try:
                step_callback(round_record)
            except Exception:
                pass

        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]
            best_weights = copy.deepcopy(global_weights)

    # Đánh giá cuối cùng trên tập Test độc lập (Held-out Test Evaluation - No Leakage)
    test_eval_model = create_model(num_classes=38, pretrained=False).to(device)
    set_parameters(test_eval_model, best_weights or global_weights)
    final_test_metrics = evaluate(test_eval_model, test_loader, criterion, device)
    print(f"[+] Final Held-out Test Evaluation: Acc={final_test_metrics['accuracy']*100:.2f}%, Loss={final_test_metrics['loss']:.4f}, F1={final_test_metrics['macro_f1']:.4f}")

    # Lưu best và last checkpoint. Smoke/không có output_dir riêng KHÔNG BAO GIỜ
    # được ghi vào slot canonical (bài học 25/09: smoke ghi đè fedavg_best.pt).
    if smoke and not output_dir:
        smoke_ckpt_dir = checkpoints_dir / "smoke"
        smoke_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_weights or global_weights, "val_accuracy": best_acc, "test_metrics": final_test_metrics, "seed": seed}, smoke_ckpt_dir / "fedavg_best_smoke.pt")
        torch.save({"model_state": global_weights, "last_accuracy": eval_metrics["accuracy"], "seed": seed}, smoke_ckpt_dir / "fedavg_last_smoke.pt")
    else:
        torch.save({"model_state": best_weights or global_weights, "val_accuracy": best_acc, "test_metrics": final_test_metrics, "seed": seed}, checkpoints_dir / "fedavg_best.pt")
        torch.save({"model_state": global_weights, "last_accuracy": eval_metrics["accuracy"], "seed": seed}, checkpoints_dir / "fedavg_last.pt")

    results_payload = {
        "experiment": "fedavg",
        "provenance": "in_silico_simulation" if smoke else "empirical_training",
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "seed": seed,
        "best_val_accuracy": best_acc,
        "test_metrics": final_test_metrics,
        "best_accuracy": final_test_metrics["accuracy"],
        "total_time_s": round(time.time() - total_start, 2),
        "history": round_history,
        "telemetry_summary": telemetry_mgr.get_summary(),
    }

    out_file = metrics_dir / "fedavg_results.json"
    if smoke and out_file.exists() and not output_dir:
        try:
            with open(out_file, "r", encoding="utf-8") as existing_f:
                existing_data = json.load(existing_f)
                if existing_data.get("rounds", 0) > 1:
                    out_file = metrics_dir / "smoke_fedavg_results.json"
        except Exception:
            pass

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[+] Federated Learning Hoàn Tất! Best Global Accuracy: {best_acc*100:.2f}%")
    return results_payload


# Flower Network Server Deployment
class AuditedFlowerFedAvg:
    """Factory-backed wrapper for Flower FedAvg with server-side update audit.

    Defined as a light proxy so importing this module does not require Flower
    for simulation-only workflows. ``build`` returns a real FedAvg subclass.
    """

    @staticmethod
    def build(
        *,
        num_clients: int,
        initial_parameters: Any,
        local_epochs: int = 1,
        seed: int = 42,
        probe_port: Optional[int] = None,
        evaluation_loader: Optional[DataLoader] = None,
        evaluation_device: Optional[torch.device] = None,
        require_real_clients: bool = False,
        reject_suspicious_updates: bool = False,
        max_update_norm: Optional[float] = None,
        expected_clients: Optional[Dict[int, Dict[str, Any]]] = None,
        expected_smoke: Optional[bool] = None,
        progress_file: Optional[str | Path] = None,
    ) -> Any:
        from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
        from flwr.server.strategy import FedAvg

        model_keys = list(create_model(num_classes=38, pretrained=False).state_dict().keys())

        class _AuditedFedAvg(FedAvg):
            def __init__(self) -> None:
                self._parent_by_round: Dict[int, Dict[str, np.ndarray]] = {}
                self._fit_started_at: Dict[int, float] = {}
                self.audit_rounds: List[Dict[str, Any]] = []
                self.final_parameters: Any = initial_parameters
                super().__init__(
                    fraction_fit=1.0,
                    min_fit_clients=num_clients,
                    min_available_clients=num_clients,
                    initial_parameters=initial_parameters,
                )

            def _write_progress(self, payload: Dict[str, Any]) -> None:
                if progress_file is None:
                    return
                progress_path = Path(progress_file)
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                pending = progress_path.with_suffix(progress_path.suffix + ".tmp")
                pending.write_text(json.dumps(payload), encoding="utf-8")
                pending.replace(progress_path)

            def configure_fit(self, server_round, parameters, client_manager):
                arrays = parameters_to_ndarrays(parameters)
                parent = {k: np.asarray(v).copy() for k, v in zip(model_keys, arrays)}
                self._parent_by_round[int(server_round)] = parent
                self._fit_started_at[int(server_round)] = time.perf_counter()
                parent_hash = compute_model_hash(parent)

                pairs = super().configure_fit(server_round, parameters, client_manager)
                self._write_progress({
                    "round_in_progress": int(server_round),
                    "completed_round": len(self.audit_rounds),
                    "connected_clients": int(client_manager.num_available()),
                    "accepted_clients": 0,
                    "rounds": self.audit_rounds,
                })
                configured = []
                for client, fit_ins in pairs:
                    config = dict(fit_ins.config)
                    config.update(
                        {
                            "round": int(server_round),
                            "model_version": int(server_round - 1),
                            "parent_model_hash": parent_hash,
                            "local_epochs": int(local_epochs),
                            "seed": int(seed),
                            "probe_port": int(probe_port) if probe_port is not None else -1,
                        }
                    )
                    configured.append((client, FitIns(fit_ins.parameters, config)))
                return configured

            def aggregate_fit(self, server_round, results, failures):
                parent = self._parent_by_round.get(int(server_round))
                if parent is None:
                    raise RuntimeError(f"Missing parent parameters for round {server_round}")

                accepted_results = []
                seen_ids: set[int] = set()
                suspicious_count = 0
                rejected_count = 0
                received_norms: List[float] = []
                local_train_times: List[float] = []
                client_records: List[Dict[str, Any]] = []

                for client_proxy, fit_res in results:
                    arrays = parameters_to_ndarrays(fit_res.parameters)
                    if len(arrays) != len(model_keys):
                        raise RuntimeError(f"Round {server_round}: wrong tensor count from {client_proxy.cid}")
                    client_weights = {k: np.asarray(v) for k, v in zip(model_keys, arrays)}
                    reported_parent_hash = str(fit_res.metrics.get("parent_model_hash", ""))
                    validation = validate_client_update(
                        parent,
                        client_weights,
                        reported_parent_hash=reported_parent_hash,
                        max_update_norm=max_update_norm,
                    )
                    fit_res.metrics["server_received_update_norm"] = float(validation.received_update_norm)
                    fit_res.metrics["server_validation_status"] = (
                        "rejected" if not validation.accepted else "suspicious" if validation.suspicious else "ok"
                    )
                    fit_res.metrics["server_validation_reason"] = validation.reason
                    received_norms.append(float(validation.received_update_norm))
                    local_train_time_s = float(fit_res.metrics.get("local_train_time_s", 0.0))
                    local_train_times.append(local_train_time_s)
                    probe_rtt_ms = float(fit_res.metrics.get("probe_rtt_ms", -1.0))
                    client_id = int(fit_res.metrics.get("client_id", -1))
                    data_mode = str(fit_res.metrics.get("data_mode", "unknown"))
                    training_seed = int(fit_res.metrics.get("training_seed", -1))
                    partition_id = str(fit_res.metrics.get("partition_id", "unknown"))
                    partition_config_sha256 = str(fit_res.metrics.get("partition_config_sha256", "unknown"))
                    client_manifest_sha256 = str(fit_res.metrics.get("client_manifest_sha256", "unknown"))
                    expected = (expected_clients or {}).get(client_id)
                    is_smoke = bool(fit_res.metrics.get("smoke", False))
                    record = {
                        "client_id": client_id,
                        "flower_cid": str(client_proxy.cid),
                        "num_examples": int(fit_res.num_examples),
                        "local_train_time_s": local_train_time_s,
                        "probe_rtt_ms": probe_rtt_ms if probe_rtt_ms >= 0 else None,
                        "data_mode": data_mode,
                        "training_seed": training_seed,
                        "partition_id": partition_id,
                        "partition_config_sha256": partition_config_sha256,
                        "client_manifest_sha256": client_manifest_sha256,
                        "received_update_norm": float(validation.received_update_norm),
                        "validation_reason": validation.reason,
                        "status": "ok",
                    }

                    if client_id in seen_ids or client_id < 0 or client_id >= num_clients:
                        raise RuntimeError(f"Round {server_round}: duplicate or invalid client_id {client_id}")
                    seen_ids.add(client_id)
                    if expected_smoke is not None and is_smoke != expected_smoke:
                        raise RuntimeError(f"Round {server_round}: client {client_id} smoke mode mismatch")
                    if (int(fit_res.metrics.get("round", -1)) != int(server_round)
                            or int(fit_res.metrics.get("model_version", -2)) != int(server_round) - 1
                            or int(fit_res.metrics.get("local_epochs", -1)) != int(local_epochs)
                            or training_seed != int(seed)):
                        raise RuntimeError(f"Round {server_round}: client {client_id} returned stale/mismatched round metadata")
                    if expected is not None:
                        expected_n = expected["smoke_examples" if is_smoke else "full_examples"]
                        if (partition_id != expected["partition_id"]
                                or partition_config_sha256 != expected["partition_config_sha256"]
                                or client_manifest_sha256 != expected["client_manifest_sha256"]
                                or int(fit_res.num_examples) != expected_n):
                            raise RuntimeError(f"Round {server_round}: client {client_id} partition/sample contract mismatch")
                    elif int(fit_res.num_examples) <= 0:
                        raise RuntimeError(f"Round {server_round}: client {client_id} has invalid n_k")

                    if require_real_clients and data_mode != "real":
                        rejected_count += 1
                        record["status"] = "rejected_synthetic_data"
                        record["validation_reason"] = "real_data_required"
                        client_records.append(record)
                        print(
                            f"[!] Flower client {client_proxy.cid} rejected: real_data_required (data_mode={data_mode})"
                        )
                        continue

                    if not validation.accepted:
                        rejected_count += 1
                        record["status"] = "rejected"
                        client_records.append(record)
                        print(
                            f"[!] Flower client {client_proxy.cid} rejected: {validation.reason}"
                        )
                        continue
                    if validation.suspicious:
                        suspicious_count += 1
                        record["status"] = "suspicious"
                        print(
                            f"[!] Flower client {client_proxy.cid} suspicious: {validation.reason}, "
                            f"server_norm={validation.received_update_norm:.4f}"
                        )
                        if reject_suspicious_updates:
                            rejected_count += 1
                            record["status"] = "rejected_suspicious"
                            client_records.append(record)
                            continue
                    client_records.append(record)
                    accepted_results.append((client_proxy, fit_res))

                if failures or len(results) != num_clients or len(accepted_results) != num_clients:
                    raise RuntimeError(
                        f"Round {server_round} requires exactly {num_clients} accepted clients; "
                        f"received={len(results)} accepted={len(accepted_results)} failures={len(failures)}"
                    )
                updates = [
                    {
                        "client_id": int(fit_res.metrics["client_id"]),
                        "num_examples": int(fit_res.num_examples),
                        "weights": {k: torch.from_numpy(np.asarray(v)) for k, v in zip(
                            model_keys, parameters_to_ndarrays(fit_res.parameters)
                        )},
                    }
                    for _, fit_res in accepted_results
                ]
                parent_tensors = {k: torch.from_numpy(v) for k, v in parent.items()}
                averaged = aggregate_fedavg(updates, parent_weights=parent_tensors)
                aggregated_parameters = ndarrays_to_parameters([
                    averaged[k].detach().cpu().numpy() for k in model_keys
                ])
                metrics: Dict[str, Any] = {}

                server_eval_metrics: Optional[Dict[str, Any]] = None
                if aggregated_parameters is not None and evaluation_loader is not None:
                    eval_device = evaluation_device or torch.device("cpu")
                    aggregated_arrays = parameters_to_ndarrays(aggregated_parameters)
                    eval_weights = {k: np.asarray(v) for k, v in zip(model_keys, aggregated_arrays)}
                    eval_model = create_model(num_classes=38, pretrained=False).to(eval_device)
                    set_parameters(eval_model, eval_weights)
                    server_eval_metrics = evaluate(
                        eval_model,
                        evaluation_loader,
                        nn.CrossEntropyLoss(),
                        eval_device,
                    )
                    metrics.update(
                        {
                            "server_val_accuracy": float(server_eval_metrics["accuracy"]),
                            "server_val_loss": float(server_eval_metrics["loss"]),
                            "server_val_macro_f1": float(server_eval_metrics["macro_f1"]),
                        }
                    )

                elapsed_ms = (
                    time.perf_counter() - self._fit_started_at.get(int(server_round), time.perf_counter())
                ) * 1000.0
                max_train_ms = max(local_train_times, default=0.0) * 1000.0
                valid_rtts = [float(r["probe_rtt_ms"]) for r in client_records if r.get("probe_rtt_ms") is not None]
                metrics.update(
                    {
                        "accepted_clients": int(len(accepted_results)),
                        "suspicious_clients": int(suspicious_count),
                        "rejected_clients": int(rejected_count),
                        "mean_received_update_norm": float(np.mean(received_norms)) if received_norms else 0.0,
                        "mean_probe_rtt_ms": float(np.mean(valid_rtts)) if valid_rtts else -1.0,
                        "max_probe_rtt_ms": float(np.max(valid_rtts)) if valid_rtts else -1.0,
                        "round_fit_elapsed_ms": float(elapsed_ms),
                        # This is an experiment-level estimate, not one-way latency.
                        "estimated_comm_overhead_ms": float(max(0.0, elapsed_ms - max_train_ms)),
                    }
                )
                if aggregated_parameters is not None:
                    self.final_parameters = aggregated_parameters
                self.audit_rounds.append(
                    {
                        "round": int(server_round),
                        "clients": client_records,
                        "accepted_clients": int(len(accepted_results)),
                        "suspicious_clients": int(suspicious_count),
                        "rejected_clients": int(rejected_count),
                        "round_fit_elapsed_ms": float(elapsed_ms),
                        "estimated_comm_overhead_ms": float(max(0.0, elapsed_ms - max_train_ms)),
                        "mean_probe_rtt_ms": float(np.mean(valid_rtts)) if valid_rtts else None,
                        "max_probe_rtt_ms": float(np.max(valid_rtts)) if valid_rtts else None,
                        "server_validation_metrics": server_eval_metrics,
                    }
                )
                self._write_progress({
                    "round_in_progress": int(server_round),
                    "completed_round": int(server_round),
                    "connected_clients": int(num_clients),
                    "accepted_clients": int(len(accepted_results)),
                    "rounds": self.audit_rounds,
                })
                return aggregated_parameters, metrics

        return _AuditedFedAvg()


def start_flower_server(
    server_address: str = "0.0.0.0:8089",
    num_clients: int = 5,
    rounds: int = 5,
    local_epochs: int = 1,
    seed: int = 42,
    require_real_clients: bool = False,
    reject_suspicious_updates: bool = False,
    max_update_norm: Optional[float] = None,
    evaluation_dataset_root: Optional[str | Path] = None,
    evaluation_partition_dir: str | Path = "data/partitions_stage2_scratch_v3/label100",
    require_real_evaluation: bool = False,
    evaluation_batch_size: int = 64,
    metrics_file: Optional[str | Path] = reg.NETWORK_FLOWER_HISTORY_JSON,
    checkpoint_file: Optional[str | Path] = reg.FEDERATED_NETWORK_GLOBAL_PT,
    progress_file: Optional[str | Path] = None,
    expected_smoke: Optional[bool] = None,
) -> Any:
    """Khởi chạy Flower Server lắng nghe kết nối từ các client process qua IP:Port."""
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    print(f"[*] Starting Flower Server listening on {server_address}")
    print(f"[*] Configured for {num_clients} clients, {rounds} round(s), seed={seed}...")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    init_model = create_model(num_classes=38, pretrained=False)
    init_weights = [v.cpu().numpy() for v in get_parameters(init_model).values()]
    initial_params = ndarrays_to_parameters(init_weights)

    evaluation_loader = None
    final_test_loader = None
    evaluation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if evaluation_dataset_root is not None:
        eval_part = Path(evaluation_partition_dir)
        val_csv = eval_part / "global_val.csv"
        test_csv = eval_part / "global_test.csv"
        if not val_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(
                f"Missing held-out evaluation manifests under {eval_part}: global_val.csv/global_test.csv"
            )
        try:
            val_ds = PlantVillageDataset(
                val_csv,
                dataset_root=evaluation_dataset_root,
                is_train=False,
                require_real_images=require_real_evaluation,
            )
            test_ds = PlantVillageDataset(
                test_csv,
                dataset_root=evaluation_dataset_root,
                is_train=False,
                require_real_images=require_real_evaluation,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Real held-out evaluation is required, but global_val/global_test image paths "
                f"did not fully resolve under evaluation_dataset_root: {exc}"
            ) from exc
        evaluation_loader = DataLoader(val_ds, batch_size=evaluation_batch_size, shuffle=False)
        final_test_loader = DataLoader(test_ds, batch_size=evaluation_batch_size, shuffle=False)
        print(
            f"[*] Held-out evaluator enabled: val={len(val_ds)}, test={len(test_ds)}, "
            f"data_mode={'real' if getattr(val_ds, 'real_mode', False) else 'synthetic'}"
        )

    bind_host, port_text = server_address.rsplit(":", 1)
    flower_port = int(port_text)
    probe_port = flower_port + 1
    probe_server = ProbeServer(bind_host, probe_port)
    probe_server.start()

    strategy = AuditedFlowerFedAvg.build(
        num_clients=num_clients,
        initial_parameters=initial_params,
        local_epochs=local_epochs,
        seed=seed,
        probe_port=probe_port,
        evaluation_loader=evaluation_loader,
        evaluation_device=evaluation_device,
        require_real_clients=require_real_clients,
        reject_suspicious_updates=reject_suspicious_updates,
        max_update_norm=max_update_norm,
        expected_clients=expected_client_contract(evaluation_partition_dir, num_clients),
        expected_smoke=expected_smoke,
        progress_file=progress_file,
    )

    try:
        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    finally:
        probe_server.stop()

    if len(strategy.audit_rounds) != rounds or any(
        audit.get("accepted_clients") != num_clients for audit in strategy.audit_rounds
    ):
        raise RuntimeError(
            f"Flower run incomplete: {len(strategy.audit_rounds)}/{rounds} audited rounds; "
            "no final checkpoint or metrics were written"
        )

    saved_checkpoint = None
    final_test_metrics: Optional[Dict[str, Any]] = None
    final_arrays = parameters_to_ndarrays(strategy.final_parameters)
    model_keys = list(init_model.state_dict().keys())
    final_weights = {k: np.asarray(v) for k, v in zip(model_keys, final_arrays)}
    final_model = create_model(num_classes=38, pretrained=False)
    set_parameters(final_model, final_weights)

    if final_test_loader is not None:
        final_model = final_model.to(evaluation_device)
        final_test_metrics = evaluate(
            final_model,
            final_test_loader,
            nn.CrossEntropyLoss(),
            evaluation_device,
        )
        print(
            f"[+] Final held-out network test: Acc={final_test_metrics['accuracy']*100:.2f}%, "
            f"F1={final_test_metrics['macro_f1']:.4f}, Loss={final_test_metrics['loss']:.4f}"
        )

    if checkpoint_file:
        checkpoint_path = Path(checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()},
                "source": "flower_network_fedavg",
                "rounds": int(rounds),
                "num_clients": int(num_clients),
                "local_epochs": int(local_epochs),
                "seed": int(seed),
                "model_hash": compute_model_hash(get_parameters(final_model)),
                "test_metrics": final_test_metrics,
            },
            checkpoint_path,
        )
        saved_checkpoint = str(checkpoint_path)

    if metrics_file:
        metrics_path = Path(metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "server_address": server_address,
            "probe_port": int(probe_port),
            "num_clients_required": int(num_clients),
            "require_real_clients": bool(require_real_clients),
            "rounds": int(rounds),
            "seed": int(seed),
            "evaluation_dataset_root": str(evaluation_dataset_root) if evaluation_dataset_root is not None else None,
            "evaluation_partition_dir": str(evaluation_partition_dir) if evaluation_dataset_root is not None else None,
            "require_real_evaluation": bool(require_real_evaluation),
            "final_test_metrics": final_test_metrics,
            "checkpoint_file": saved_checkpoint,
            "audit_rounds": strategy.audit_rounds,
            "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
            "losses_distributed": getattr(history, "losses_distributed", []),
            "losses_centralized": getattr(history, "losses_centralized", []),
            "metrics_distributed": getattr(history, "metrics_distributed", {}),
            "metrics_centralized": getattr(history, "metrics_centralized", {}),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print("[+] Flower Network Server completed successfully!")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FedAvg Server")
    parser.add_argument("--mode", choices=["simulation", "deployment"], default="simulation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-real-clients", action="store_true")
    parser.add_argument("--reject-suspicious-updates", action="store_true")
    parser.add_argument("--max-update-norm", type=float, default=None)
    parser.add_argument("--evaluation-dataset-root", type=str, default=None, help="Optional held-out PlantVillage image root for server-side global validation/test only")
    parser.add_argument("--evaluation-partition-dir", type=str, default="data/partitions_stage2_scratch_v3/label100")
    parser.add_argument("--require-real-evaluation", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--metrics-file", type=str, default=reg.NETWORK_FLOWER_HISTORY_JSON)
    parser.add_argument("--checkpoint-file", type=str, default=reg.FEDERATED_NETWORK_GLOBAL_PT)
    parser.add_argument("--progress-file", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--partition-dir", type=str, default="data/partitions_stage2_scratch_v3/label100")
    parser.add_argument("--dataset-root", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "deployment":
        start_flower_server(
            server_address=f"{args.host}:{args.port}",
            num_clients=args.num_clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            seed=args.seed,
            require_real_clients=args.require_real_clients,
            reject_suspicious_updates=args.reject_suspicious_updates,
            max_update_norm=args.max_update_norm,
            evaluation_dataset_root=args.evaluation_dataset_root,
            evaluation_partition_dir=args.evaluation_partition_dir,
            require_real_evaluation=args.require_real_evaluation,
            evaluation_batch_size=args.evaluation_batch_size,
            metrics_file=args.metrics_file,
            checkpoint_file=args.checkpoint_file,
            progress_file=args.progress_file,
            expected_smoke=args.smoke,
        )
    else:
        run_federated_simulation(
            partition_dir=args.partition_dir,
            dataset_root=args.dataset_root,
            num_clients=args.num_clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            seed=args.seed,
            smoke=args.smoke,
        )

"""Router giám sát: status, metrics, clients, weights inspect, artifact status."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter

from src import registry as reg

from web.services import model_service, training_guard
from web.state import NETWORK_DEMO, check_port_open

from src.data.partition import calculate_data_scarcity

router = APIRouter()


@router.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Lấy trạng thái hệ thống: CPU/GPU, Server Flower, Checkpoints và tác vụ hiện thời."""
    from web.state import CURRENT_TASK

    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    traced_ready = model_service.is_web_visible("plantvillage_traced_reference")
    flower_active = NETWORK_DEMO.snapshot()["server_running"]

    return {
        "server_status": "online",
        "device": dev_name,
        "cuda_available": cuda_avail,
        "flower_server_active": flower_active,
        "flower_daemon_status": "Port 8089 (Listening)" if flower_active else "Port 8089 (Idle / Standby)",
        "task": CURRENT_TASK,
        "network_processes": NETWORK_DEMO.snapshot(),
        "canonical_run": {
            "run_id": None,
            "scientific_valid": False,
            "status_label": "UNVERIFIED",
            "message": "Historical scratch artifacts are outside the published mixed-data comparison. See /api/demo/evidence.",
        },
        "models_available": model_service.status_models_available(),
        "checkpoints_info": model_service.status_checkpoints_info(),
        "traced_reference_available": traced_ready,
        # Giao diện biết trước có cần đăng nhập quản trị để bấm Bắt đầu/Dừng và các trần tham số của máy chủ này
        "training": training_guard.status_block(),
    }


@router.get("/api/metrics/{mode}")
def get_metrics(mode: str, source: str | None = Query(None)) -> Any:
    """
    Đọc dữ liệu metric:
    - Mặc định: trả về artifact canonical lịch sử, kèm cờ provenance/scientific_valid.
    - Nếu source=demo hoặc source=web_run: trả về kết quả phiên mô phỏng web và luôn đánh dấu không hợp lệ khoa học.
    """
    canonical_map = {
        "centralized": reg.abs_path(reg.CENTRALIZED_RESULTS_JSON),
        "federated": reg.abs_path(reg.FEDAVG_RESULTS_JSON),
        "local_only": reg.abs_path(reg.LOCAL_ONLY_RESULTS_JSON),
        "telemetry": reg.abs_path(reg.TELEMETRY_HISTORY_JSON),
    }
    if source in ("demo", "web_run"):
        stem = {
            "centralized": "centralized_results",
            "federated": "fedavg_results",
        }.get(mode, "telemetry_history")
        candidates = [
            reg.abs_path(reg.WEB_RUNS_DIR) / f"{mode}_demo" / "metrics" / f"{stem}.json",
            reg.abs_path(reg.WEB_RUNS_DIR) / f"{mode}_full" / "metrics" / f"{stem}.json",
            reg.abs_path(reg.DEMO_RUNS_DIR) / mode / "metrics" / f"{stem}.json",
        ]
        for target in candidates:
            if target.exists():
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data["provenance"] = "web_interactive_simulation"
                        data["source"] = "web_run"
                        data["scientific_valid"] = False
                        data["scientific_reason"] = "web demo/simulation artifact; not a verified real-data scientific run"
                    return data
        return JSONResponse(
            status_code=404,
            content={"available": False, "message": f"Chưa có phiên mô phỏng web nào cho '{mode}'. Hãy bấm Bắt Đầu Huấn Luyện để chạy thử."}
        )

    # Mặc định: Trả về canonical run để đảm bảo tính nhất quán khoa học toàn hệ thống
    target = canonical_map.get(mode)
    if target and target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("provenance", "canonical_stage2_run")
                    data["source"] = "canonical"
                    status = model_service.artifact_scientific_status("canonical")
                    data["scientific_valid"] = status["overall_scientific_valid"]
                    data["scientific_reason"] = (
                        status["centralized"]["reason"] if mode == "centralized"
                        else status["federated"]["reason"] if mode == "federated"
                        else "canonical artifact is unverified until linked provenance passes"
                    )
                return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi đọc file metric {target}: {str(e)}")

    return JSONResponse(
        status_code=404,
        content={"available": False, "message": f"Chưa có dữ liệu metric cho chế độ '{mode}'."}
    )


@router.get("/api/clients")
def get_clients_info(alpha: float = Query(1.0)) -> Dict[str, Any]:
    """
    Thông tin 5 Clients: đọc partition_config.json theo Dirichlet alpha,
    data-starvation từ manifest thật, telemetry vòng gần nhất.
    """
    from web.state import CURRENT_TASK  # noqa: F401  (giữ contract import cũ)

    flower_connected = check_port_open(port=8080) or check_port_open(port=8089)

    alpha_map = {
        0.1: f"{reg.PARTITIONS_DIR}/label01",
        1.0: f"{reg.PARTITIONS_DIR}/label1",
        100.0: f"{reg.PARTITIONS_DIR}/label100",
    }
    matched_key = min(alpha_map.keys(), key=lambda k: abs(k - alpha))
    part_dir = Path(alpha_map[matched_key])
    if not part_dir.exists():
        part_dir = Path(f"{reg.PARTITIONS_DIR}/label100")

    part_config_file = part_dir / "partition_config.json"
    partition_details: Dict[int, Any] = {}
    actual_alpha = matched_key
    total_samples_global = 38984

    if part_config_file.exists():
        try:
            with open(part_config_file, "r", encoding="utf-8") as f:
                p_cfg = json.load(f)
                actual_alpha = p_cfg.get("alpha", matched_key)
                total_samples_global = p_cfg.get("partition_metrics", {}).get("total_samples", 38984)
                for c_item in p_cfg.get("partition_metrics", {}).get("client_details", []):
                    partition_details[c_item["client_id"]] = c_item
        except Exception:
            pass

    telemetry_file = reg.abs_path(reg.TELEMETRY_HISTORY_JSON)
    if not telemetry_file.exists():
        telemetry_file = reg.abs_path(reg.DEMO_RUNS_DIR) / "fedavg" / "metrics" / "telemetry_history.json"

    last_round_telemetry: Dict[int, Any] = {}
    if telemetry_file.exists():
        try:
            with open(telemetry_file, "r", encoding="utf-8") as f:
                t_list = json.load(f)
                if isinstance(t_list, list) and len(t_list) > 0:
                    last_round = t_list[-1]
                    for cm in last_round.get("client_metrics", []):
                        last_round_telemetry[cm.get("client_id")] = cm
        except Exception:
            pass

    clients_data = []
    base_counts = [7568, 7699, 7817, 8032, 7868]
    base_entropies = [4.84, 4.88, 4.80, 4.78, 4.78]

    # Nhóm cây trồng thực tế được phân bổ theo đặc trưng non-IID
    CROP_ROLES = [
        "Vườn cây ăn quả (Táo, Nho, Đào, Anh đào)",
        "Nông hộ rau màu (Cà chua, Ớt chuông, Khoai tây)",
        "Vùng ngũ cốc & đậu (Ngô gỉ sắt/đốm xám, Đậu nành)",
        "Vườn trái cây có múi & quả mọng (Cam, Dâu tây, Mâm xôi)",
        "Vùng trồng đa canh (Bí ngô, Ớt, Cà chua, Khoai tây)",
    ]

    for cid in range(5):
        p_info = partition_details.get(cid, {})
        t_info = last_round_telemetry.get(cid, {})

        samples = p_info.get("sample_count", base_counts[cid])
        classes_present = p_info.get("num_classes", 38)
        shannon_entropy = round(p_info.get("shannon_entropy", base_entropies[cid]), 3)
        sample_ratio = p_info.get("sample_ratio", round(samples / max(total_samples_global, 1), 4))
        weight_pct = round(sample_ratio * 100, 2)

        # Data-starvation trực tiếp từ manifest thật của client
        scarcity = None
        client_csv = part_dir / "clients" / f"client_{cid:02d}.csv"
        if client_csv.exists():
            try:
                labels = []
                with open(client_csv, "r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        labels.append(int(row["label"]))
                scarcity = calculate_data_scarcity(labels, total_classes=38)
            except Exception:
                scarcity = None

        if scarcity is None:
            entropy_norm = float(p_info.get("entropy_normalized", 0.0))
            missing = int(p_info.get("num_classes_missing", max(38 - int(classes_present), 0)))
            scarcity = {
                "num_samples": int(samples),
                "class_count_present": int(classes_present),
                "class_count_missing": missing,
                "largest_class_ratio": None,
                "label_entropy": float(shannon_entropy),
                "entropy_normalized": entropy_norm,
                "starvation_score": None,
                "starvation_level": "unknown",
            }

        has_run = cid in last_round_telemetry
        local_train_time = round(t_info.get("local_train_time_s", 0.0), 2) if has_run else None
        raw_norm = t_info.get("update_norm")
        update_norm = round(float(raw_norm), 4) if (has_run and raw_norm is not None and float(raw_norm) > 0) else None

        status_str = "Daemon lắng nghe (gRPC)" if flower_connected else "Mô phỏng nội bộ (In-silico)"

        clients_data.append({
            "client_id": cid,
            "client_name": f"Trạm Nông Hộ #{cid:02d}",
            "role_badge": f"Node {cid} (Phân vùng {cid})",
            "execution_type": "Flower In-Silico Simulation",
            "data_partition_file": f"client_{cid:02d}.csv",
            "samples": samples,
            "total_dataset_pct": f"{weight_pct}%",
            "crop_focus": CROP_ROLES[cid % len(CROP_ROLES)],
            "classes_present": int(scarcity.get("class_count_present", classes_present)),
            "classes_missing": int(scarcity.get("class_count_missing", max(38 - int(classes_present), 0))),
            "label_entropy": float(scarcity.get("label_entropy", shannon_entropy)),
            "entropy_normalized": scarcity.get("entropy_normalized"),
            "largest_class_ratio": scarcity.get("largest_class_ratio"),
            "data_starvation_score": scarcity.get("starvation_score"),
            "data_starvation_level": scarcity.get("starvation_level"),
            "fedavg_weight": f"n_{cid} / N = {sample_ratio:.4f}",
            "local_train_time_s": local_train_time,
            "update_norm": update_norm,
            "status": status_str,
            "cli_command": f"python -m src.federated.client_app --server datn.tranngochung.id.vn:8080 --client-id {cid}",
        })

    return {
        "clients": clients_data,
        "total_clients": 5,
        "alpha": actual_alpha,
        "total_samples": total_samples_global,
        "flower_server_active": flower_connected,
        "flower_daemon_status": "Port 8080 (Listening)" if flower_connected else "Port 8080 (Idle / Standby)",
        "framework": "Flower Federated Learning (flwr)",
        "simulation_mode": True,
        "description": "5 Clients mô phỏng phân tán (In-silico simulation) theo phân phối Dirichlet non-IID. Hỗ trợ kết nối máy vật lý ngoài qua gRPC."
    }


@router.get("/api/weights/inspect")
def inspect_weights(layer: str | None = None) -> Dict[str, Any]:
    """Thống kê trọng số theo layer — ủy quyền cho model_service (registry)."""
    return model_service.inspect_weights(layer)


@router.get("/api/artifacts/status")
def get_artifact_status(federated_source: str = Query("canonical")) -> Dict[str, Any]:
    if federated_source not in {"canonical", "network"}:
        raise HTTPException(status_code=400, detail="federated_source chỉ nhận 'canonical' hoặc 'network'.")
    return model_service.artifact_scientific_status(federated_source)

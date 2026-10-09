"""
web.app: Máy chủ FastAPI cung cấp REST API và giao diện Web Dashboard cho hệ thống Federated Learning.
Web chỉ làm nhiệm vụ Presentation, API và Inference; tuyệt đối không chứa logic training nặng nề bên trong.
"""

from __future__ import annotations

import csv
import io
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import torch

from web.demo_evidence import PROTOCOL_ID, load_evidence, predict_published
from web.plantvillage_reference import predict_plantvillage
from web.network_demo import NetworkDemoManager

try:
    from src.centralized.inference import predict_image
    from src.centralized.train import run_centralized_training
    from src.data.partition import calculate_data_scarcity
    from src.model.mobilenetv3 import create_model, get_parameters
    from src.federated.server_app import run_federated_simulation
    from src.federated.weight_inspector import inspect_layer_weights, inspect_model_state
except ImportError:
    from train.centralized import predict_image, run_centralized_training
    from train.data import calculate_data_scarcity
    from train.model import create_model, get_parameters
    from train.server import run_federated_simulation
    from train.weight_inspector import inspect_layer_weights, inspect_model_state


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        NETWORK_DEMO.stop()


app = FastAPI(
    title="Federated Learning PlantVillage Dashboard",
    description="Hệ thống điều khiển, quan sát và suy luận (inference) Federated Learning trên MobileNetV3-Small",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Trạng thái tác vụ nền
CURRENT_TASK: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "completed" | "failed" | "cancelled"
    "experiment": None,
    "type": None,  # "centralized" | "federated"
    "progress": 0,
    "message": "Hệ thống sẵn sàng",
    "task_id": None,
    "config": None,
    "error": None,
    "live_progress": [],
    "cancel_requested": False,
}
NETWORK_DEMO = NetworkDemoManager()
SMOKE_RUNS_ROOT = Path(__file__).resolve().parents[1] / "artifacts/web_runs/network_smoke"


def latest_completed_smoke_run() -> Optional[Path]:
    """Recover only a fully audited local smoke run after a web restart."""
    if not SMOKE_RUNS_ROOT.is_dir():
        return None
    for run_dir in sorted(SMOKE_RUNS_ROOT.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        try:
            session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            audits = history["audit_rounds"]
            if (session.get("kind") == "flower_localhost_synthetic_smoke"
                    and session.get("status") == "completed"
                    and session.get("scientific_valid") is False
                    and session.get("published_result_reproduction") is False
                    and session.get("client_processes") == 5
                    and len(audits) == session.get("rounds") == 2
                    and [audit.get("round") for audit in audits] == [1, 2]
                    and all(audit.get("accepted_clients") == 5 for audit in audits)
                    and (run_dir / "global.pt").is_file()):
                return run_dir
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


class CentralizedTrainRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=100, description="Số epoch huấn luyện tập trung (1-100)")
    smoke: bool = Field(default=True, description="Chế độ demo chạy nhanh (True) hoặc huấn luyện đầy đủ (False)")
    dataset_root: Optional[str] = Field(default=None, description="Đường dẫn thư mục ảnh PlantVillage")
    seed: int = Field(default=42, ge=0, le=2_147_483_647, description="Seed cho model init/DataLoader")


class FederatedTrainRequest(BaseModel):
    rounds: int = Field(default=2, ge=1, le=100, description="Số vòng giao tiếp FedAvg (1-100)")
    local_epochs: int = Field(default=1, ge=1, le=20, description="Số epoch cục bộ mỗi client (1-20)")
    alpha: float = Field(default=0.1, description="Hệ số Dirichlet non-IID: 0.1, 1.0, hoặc 100.0")
    smoke: bool = Field(default=True, description="Chế độ demo chạy nhanh (True) hoặc mô phỏng đầy đủ (False)")
    dataset_root: Optional[str] = Field(default=None, description="Đường dẫn thư mục ảnh PlantVillage")
    engine: str = Field(default="simulation", description="simulation hoặc network")
    num_clients: int = Field(default=5, ge=1, le=50, description="Số Flower clients yêu cầu")
    server_address: str = Field(default="127.0.0.1:8089", description="Địa chỉ client kết nối Flower server khi engine=network")
    require_real_clients: bool = Field(default=False, description="Từ chối client synthetic/dummy trong network mode")
    require_real_evaluation: bool = Field(default=False, description="Bắt buộc held-out evaluator dùng ảnh PlantVillage thật")
    reject_suspicious_updates: bool = Field(default=False, description="Loại update đáng ngờ trước FedAvg")
    max_update_norm: Optional[float] = Field(default=None, gt=0, description="Ngưỡng L2 norm để gắn cờ update bất thường")
    seed: int = Field(default=42, ge=0, le=2_147_483_647, description="Seed cho model init/DataLoader để tái lập experiment")


def check_port_open(host: str = "127.0.0.1", port: int = 8080, timeout: float = 0.3) -> bool:
    """Kiểm tra xem Flower gRPC Server có đang mở port lắng nghe thật không."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_and_verify_dataset_root(candidate: Optional[str]) -> Optional[Path]:
    """Kiểm tra sự tồn tại của thư mục ảnh thật PlantVillage."""
    candidates = []
    if candidate:
        candidates.append(Path(candidate))
    if os.environ.get("DATASET_ROOT"):
        candidates.append(Path(os.environ["DATASET_ROOT"]))
    if os.environ.get("PLANTVILLAGE_DATASET_ROOT"):
        candidates.append(Path(os.environ["PLANTVILLAGE_DATASET_ROOT"]))
    candidates.extend([
        Path("data/PlantVillage"),
        Path("data/plantvillage"),
        Path("../PlantVillage-Dataset/raw/color"),
        Path("PlantVillage-Dataset/raw/color"),
        Path("/home/ubuntu/fl-project/data/PlantVillage"),
    ])
    for c in candidates:
        if c.exists() and c.is_dir():
            subdirs = [p for p in c.iterdir() if p.is_dir()]
            if len(subdirs) >= 10:
                return c.resolve()
    return None


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Federated Learning API is running. static/index.html is being prepared."}


@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Lấy trạng thái hệ thống: CPU/GPU, Server Flower, Checkpoints và tác vụ hiện thời."""
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    # Kiểm tra các checkpoint có sẵn
    centralized_ready = Path("artifacts/checkpoints/centralized_best.pt").exists()
    fedavg_ready = Path("artifacts/checkpoints/fedavg_best.pt").exists()
    traced_ready = Path("artifacts/checkpoints/plantvillage_traced.pt").exists()

    demo_cent_ready = Path("artifacts/demo_runs/centralized/checkpoints/centralized_best.pt").exists()
    demo_fed_ready = Path("artifacts/demo_runs/fedavg/checkpoints/fedavg_best.pt").exists()

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
        "models_available": {
            "centralized": False,
            "fedavg": False,
            "traced_reference": False,
        },
        "checkpoints_info": {
            "centralized_canonical": centralized_ready,
            "fedavg_canonical": fedavg_ready,
            "centralized_demo": demo_cent_ready,
            "fedavg_demo": demo_fed_ready,
        }
    }


@app.get("/api/metrics/{mode}")
def get_metrics(mode: str, source: Optional[str] = Query(None)) -> Any:
    """
    Đọc dữ liệu metric:
    - Mặc định: trả về artifact canonical lịch sử, kèm cờ provenance/scientific_valid.
    - Nếu source=demo hoặc source=web_run: trả về kết quả phiên mô phỏng web và luôn đánh dấu không hợp lệ khoa học.
    """
    canonical_map = {
        "centralized": Path("artifacts/metrics/centralized_results.json"),
        "federated": Path("artifacts/metrics/fedavg_results.json"),
        "local_only": Path("artifacts/metrics/local_only_results.json"),
        "telemetry": Path("artifacts/metrics/telemetry_history.json"),
    }
    if source in ("demo", "web_run"):
        candidates = [
            Path(f"artifacts/web_runs/{mode}_demo/metrics/{'centralized_results' if mode=='centralized' else ('fedavg_results' if mode=='federated' else 'telemetry_history')}.json"),
            Path(f"artifacts/web_runs/{mode}_full/metrics/{'centralized_results' if mode=='centralized' else ('fedavg_results' if mode=='federated' else 'telemetry_history')}.json"),
            Path(f"artifacts/demo_runs/{mode}/metrics/{'centralized_results' if mode=='centralized' else ('fedavg_results' if mode=='federated' else 'telemetry_history')}.json"),
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
                    status = _artifact_scientific_status("canonical")
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


@app.get("/api/clients")
def get_clients_info(alpha: float = Query(1.0)) -> Dict[str, Any]:
    """
    Cung cấp thông tin thực tế, khoa học và minh bạch về 5 Clients trong mạng Federated Learning:
    - Phản ánh đúng 5 Phân Vùng Dữ Liệu Cục Bộ (Partitions client_00 -> client_04).
    - Đọc trực tiếp từ partition_config.json theo Dirichlet alpha.
    - Cung cấp lệnh CLI để kết nối thiết bị vật lý bên ngoài qua Flower gRPC socket (port 8080/8089).
    """
    flower_connected = check_port_open(port=8080) or check_port_open(port=8089)

    # 1. Chọn thư mục partition tương ứng alpha
    alpha_map = {
        0.1: "data/partitions_stage2_scratch_v3/label01",
        1.0: "data/partitions_stage2_scratch_v3/label1",
        100.0: "data/partitions_stage2_scratch_v3/label100",
    }
    matched_key = min(alpha_map.keys(), key=lambda k: abs(k - alpha))
    part_dir = Path(alpha_map[matched_key])
    if not part_dir.exists():
        part_dir = Path("data/partitions_stage2_scratch_v3/label100")

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

    # 2. Đọc telemetry từ vòng huấn luyện FedAvg gần nhất (nếu có)
    telemetry_file = Path("artifacts/metrics/telemetry_history.json")
    if not telemetry_file.exists():
        telemetry_file = Path("artifacts/demo_runs/fedavg/metrics/telemetry_history.json")

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

        # Tính data-starvation trực tiếp từ manifest thật của client, không suy
        # diễn chỉ từ sample count. Nếu manifest không đọc được thì fallback về
        # các thống kê đã lưu trong partition_config.json.
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


@app.get("/api/weights/inspect")
def inspect_weights(layer: Optional[str] = None) -> Dict[str, Any]:
    """
    Trích xuất thống kê trọng số chi tiết cho layer được chọn.
    - Nạp từ fedavg_best.pt thực tế.
    - Sử dụng reference tensor (nếu có fedavg_last.pt hoặc baseline) để tính Update Norm ||Δ_k||_2.
    - Không fallback âm thầm sang model ngẫu nhiên.
    """
    fedavg_ckpt = Path("artifacts/checkpoints/fedavg_best.pt")
    if not fedavg_ckpt.exists():
        fedavg_ckpt = Path("artifacts/demo_runs/fedavg/checkpoints/fedavg_best.pt")

    if not fedavg_ckpt.exists():
        return {
            "available": False,
            "message": "Chưa có checkpoint FedAvg (fedavg_best.pt) để kiểm tra trọng số.",
            "available_layers": [],
            "selected_layer_stats": None,
        }

    try:
        ckpt = torch.load(fedavg_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        weights = {k: v.cpu().detach() if hasattr(v, "cpu") else torch.tensor(v) for k, v in state.items()}
    except Exception as e:
        return {
            "available": False,
            "message": f"Không thể nạp checkpoint: {str(e)}",
            "available_layers": [],
            "selected_layer_stats": None,
        }

    # Tìm reference checkpoint để tính update norm
    ref_weights = None
    ref_ckpt = Path("artifacts/checkpoints/fedavg_last.pt")
    if not ref_ckpt.exists():
        ref_ckpt = Path("artifacts/checkpoints/centralized_best.pt")
    if ref_ckpt.exists():
        try:
            r_ckpt = torch.load(ref_ckpt, map_location="cpu", weights_only=False)
            r_state = r_ckpt.get("model_state", r_ckpt)
            ref_weights = {k: v.cpu().detach() if hasattr(v, "cpu") else torch.tensor(v) for k, v in r_state.items()}
        except Exception:
            ref_weights = None

    all_layers = sorted(weights.keys())
    selected_layer = layer if layer and layer in weights else all_layers[0]

    ref_tensor = ref_weights.get(selected_layer) if ref_weights else None
    stats = inspect_layer_weights(weights[selected_layer], reference_tensor=ref_tensor, layer_name=selected_layer)

    return {
        "available": True,
        "available_layers": all_layers,
        "selected_layer_stats": stats,
        "has_reference": ref_tensor is not None,
        "checkpoint_source": fedavg_ckpt.name,
    }


@app.get("/api/demo/evidence")
def get_demo_evidence() -> Dict[str, Any]:
    return load_evidence()


@app.get("/api/compare")
def get_comparison_table(protocol_id: str = PROTOCOL_ID) -> Dict[str, Any]:
    """Return comparison data without inventing or silently trusting stale metrics."""
    if protocol_id != PROTOCOL_ID:
        raise HTTPException(status_code=400, detail="Protocol không có bộ đối chứng được kiểm hash.")
    evidence = load_evidence()
    if not evidence["available"]:
        return {"available": False, "reason": evidence["reason"], "rows": [], "models": []}
    rows = [
        {
            "model_id": model["model_id"],
            "model": model["name"],
            "phase": model["phase"],
            "method": model["method"],
            "source": source,
            "protocol_id": evidence["protocol_id"],
            "release_manifest_sha256": evidence["release_manifest_sha256"],
            "validation_manifest_sha256": evidence["validation_manifest_sha256"],
            "seed": evidence["seed"],
            "metric_source": evidence["metrics_source"],
            "evidence_level": evidence["evidence_level"],
            "branch": model["branch"],
            "commit": model["commit"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            **model["metrics"][source],
        }
        for model in evidence["models"]
        for source in ("plantdoc", "plantvillage")
    ]
    return {
        "available": True,
        "protocol_id": evidence["protocol_id"],
        "evidence_level": evidence["evidence_level"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "seed": evidence["seed"],
        "rows": rows,
        "models": evidence["models"],
        "scientific_stage2_complete": False,
    }


@app.get("/api/federated/network-metrics")
def get_federated_network_metrics() -> Dict[str, Any]:
    """Return persisted metrics from the real Flower network server path."""
    current_run = NETWORK_DEMO.run_dir
    historical_run = current_run is None
    if historical_run:
        current_run = latest_completed_smoke_run()
    metrics_path = current_run / "history.json" if current_run else SMOKE_RUNS_ROOT / "none/history.json"
    if not metrics_path.exists():
        return {
            "available": False,
            "message": "Chưa có lịch sử Flower network thật.",
            "rounds": [],
        }

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không đọc được network metrics: {exc}")

    distributed_fit = payload.get("metrics_distributed_fit", {}) or {}
    round_map: Dict[int, Dict[str, Any]] = {}
    for metric_name, values in distributed_fit.items():
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            round_num, value = item
            try:
                round_num = int(round_num)
            except Exception:
                continue
            round_map.setdefault(round_num, {"round": round_num})[metric_name] = value

    rounds = [round_map[k] for k in sorted(round_map)]
    return {
        "available": True,
        "scientific_valid": False,
        "provenance": "flower_localhost_synthetic_smoke",
        "historical_run": historical_run,
        "run_dir": str(current_run),
        "server_address": payload.get("server_address"),
        "probe_port": payload.get("probe_port"),
        "num_clients_required": payload.get("num_clients_required"),
        "configured_rounds": payload.get("rounds"),
        "seed": payload.get("seed"),
        "require_real_clients": payload.get("require_real_clients", False),
        "evaluation_dataset_root": payload.get("evaluation_dataset_root"),
        "require_real_evaluation": payload.get("require_real_evaluation", False),
        "final_test_metrics": payload.get("final_test_metrics"),
        "checkpoint_file": payload.get("checkpoint_file"),
        "rounds": rounds,
        "audit_rounds": payload.get("audit_rounds", []),
    }


@app.post("/api/train/stop")
def stop_training() -> Dict[str, Any]:
    """Hủy/Dừng tác vụ huấn luyện đang chạy (Centralized hoặc Federated)."""
    if CURRENT_TASK["status"] != "running":
        return {
            "status": CURRENT_TASK["status"],
            "message": "Không có tác vụ huấn luyện nào đang chạy để dừng.",
            "stopped": False,
        }

    CURRENT_TASK["cancel_requested"] = True
    if CURRENT_TASK.get("type") == "federated-network":
        NETWORK_DEMO.stop()
    CURRENT_TASK["message"] = "Đang gửi lệnh dừng tới tiến trình huấn luyện..."
    return {
        "status": "stopping",
        "message": "Đã gửi tín hiệu dừng khẩn cấp tới tiến trình huấn luyện.",
        "stopped": True,
    }


def _run_centralized_bg(epochs: int, smoke: bool, dataset_root: Optional[str], seed: int):
    CURRENT_TASK["status"] = "running"
    CURRENT_TASK["experiment"] = "Centralized"
    CURRENT_TASK["type"] = "centralized"
    CURRENT_TASK["progress"] = 5
    CURRENT_TASK["live_progress"] = []
    CURRENT_TASK["cancel_requested"] = False
    mode_text = "demo" if smoke else "thực tế"
    CURRENT_TASK["message"] = f"Đang khởi động huấn luyện Centralized ({epochs} epochs, {mode_text})..."
    CURRENT_TASK["error"] = None

    def is_cancelled() -> bool:
        return CURRENT_TASK.get("cancel_requested", False)

    def step_cb(step_data: Dict[str, Any]):
        CURRENT_TASK["live_progress"].append(step_data)
        ep = step_data.get("epoch", 1)
        CURRENT_TASK["progress"] = int((ep / max(epochs, 1)) * 100)
        val_acc = step_data.get("val_accuracy", 0.0)
        val_acc_pct = val_acc * 100.0 if val_acc <= 1.0 else val_acc
        CURRENT_TASK["message"] = f"Centralized: Epoch {ep}/{epochs} - Val Acc: {val_acc_pct:.2f}%"

    try:
        output_dir = Path("artifacts/web_runs/centralized_demo") if smoke else Path("artifacts/web_runs/centralized_full")
        output_dir.mkdir(parents=True, exist_ok=True)

        res = run_centralized_training(
            epochs=epochs,
            smoke=smoke,
            require_real_data=not smoke,
            dataset_root=dataset_root,
            seed=seed,
            output_dir=output_dir,
            step_callback=step_cb,
            is_cancelled=is_cancelled,
        )

        if CURRENT_TASK.get("cancel_requested"):
            CURRENT_TASK["status"] = "cancelled"
            CURRENT_TASK["message"] = f"Đã dừng huấn luyện Centralized tại Epoch {len(CURRENT_TASK['live_progress'])}/{epochs} theo yêu cầu."
        else:
            CURRENT_TASK["progress"] = 100
            CURRENT_TASK["status"] = "completed"
            acc = res.get("test_metrics", {}).get("accuracy", 0.0)
            acc_pct = acc * 100.0 if acc <= 1.0 else acc
            CURRENT_TASK["message"] = f"Centralized ({mode_text}) hoàn tất {epochs}/{epochs} epochs! Test Acc: {acc_pct:.2f}%"
    except Exception as e:
        if CURRENT_TASK.get("cancel_requested"):
            CURRENT_TASK["status"] = "cancelled"
            CURRENT_TASK["message"] = "Đã dừng huấn luyện Centralized theo yêu cầu."
        else:
            CURRENT_TASK["status"] = "failed"
            CURRENT_TASK["message"] = f"Huấn luyện Centralized thất bại: {str(e)}"
            CURRENT_TASK["error"] = str(e)


@app.post("/api/centralized/train")
def trigger_centralized_train(
    background_tasks: BackgroundTasks,
    req: CentralizedTrainRequest = CentralizedTrainRequest(),
):
    """
    Kích hoạt huấn luyện Centralized:
    - Xác thực số epoch > 0.
    - Tách biệt rõ ràng chế độ demo (smoke=True) và huấn luyện thật (smoke=False).
    - Bảo vệ checkpoint/metrics chính: demo run được lưu vào artifacts/demo_runs/centralized/.
    """
    if CURRENT_TASK["status"] == "running":
        raise HTTPException(status_code=400, detail="Một tác vụ huấn luyện khác đang chạy. Vui lòng chờ tác vụ hiện tại hoàn tất.")

    if not req.smoke:
        raise HTTPException(status_code=400, detail="Chế độ ảnh thật chưa khả dụng: thiếu release mixed pv_pd_v3, manifest và W0.")

    CURRENT_TASK["config"] = req.model_dump()
    background_tasks.add_task(_run_centralized_bg, req.epochs, req.smoke, None, req.seed)
    return {
        "status": "started",
        "message": f"Bắt đầu huấn luyện Centralized ({req.epochs} epochs, {'demo' if req.smoke else 'ảnh thật'})",
        "config": req.model_dump()
    }


def _run_federated_bg(rounds: int, local_epochs: int, partition_dir: str, smoke: bool, dataset_root: Optional[str], seed: int):
    CURRENT_TASK["status"] = "running"
    CURRENT_TASK["experiment"] = "Federated"
    CURRENT_TASK["type"] = "federated"
    CURRENT_TASK["progress"] = 5
    CURRENT_TASK["live_progress"] = []
    CURRENT_TASK["cancel_requested"] = False
    mode_text = "demo" if smoke else "thực tế"
    CURRENT_TASK["message"] = f"Đang điều phối FedAvg ({rounds} rounds, E={local_epochs}, {mode_text})..."
    CURRENT_TASK["error"] = None

    def is_cancelled() -> bool:
        return CURRENT_TASK.get("cancel_requested", False)

    def step_cb(step_data: Dict[str, Any]):
        CURRENT_TASK["live_progress"].append(step_data)
        r = step_data.get("round", 1)
        CURRENT_TASK["progress"] = int((r / max(rounds, 1)) * 100)
        g_acc = step_data.get("global_accuracy", 0.0)
        g_acc_pct = g_acc if g_acc > 1.0 else g_acc * 100.0
        CURRENT_TASK["message"] = f"FedAvg: Vòng {r}/{rounds} - Global Acc: {g_acc_pct:.2f}%"

    try:
        output_dir = Path("artifacts/web_runs/fedavg_demo") if smoke else Path("artifacts/web_runs/fedavg_full")
        output_dir.mkdir(parents=True, exist_ok=True)

        res = run_federated_simulation(
            partition_dir=partition_dir,
            rounds=rounds,
            local_epochs=local_epochs,
            seed=seed,
            smoke=smoke,
            require_real_data=not smoke,
            dataset_root=dataset_root,
            output_dir=output_dir,
            step_callback=step_cb,
            is_cancelled=is_cancelled,
        )

        if CURRENT_TASK.get("cancel_requested"):
            CURRENT_TASK["status"] = "cancelled"
            CURRENT_TASK["message"] = f"Đã dừng huấn luyện FedAvg tại Vòng {len(CURRENT_TASK['live_progress'])}/{rounds} theo yêu cầu."
        else:
            CURRENT_TASK["progress"] = 100
            CURRENT_TASK["status"] = "completed"
            acc = res.get("best_accuracy", 0.0)
            acc_pct = acc if acc > 1.0 else acc * 100.0
            CURRENT_TASK["message"] = f"FedAvg ({mode_text}) hoàn tất {rounds}/{rounds} rounds! Best Acc: {acc_pct:.2f}%"
    except Exception as e:
        if CURRENT_TASK.get("cancel_requested"):
            CURRENT_TASK["status"] = "cancelled"
            CURRENT_TASK["message"] = "Đã dừng điều phối FedAvg theo yêu cầu."
        else:
            CURRENT_TASK["status"] = "failed"
            CURRENT_TASK["message"] = f"Điều phối FedAvg thất bại: {str(e)}"
            CURRENT_TASK["error"] = str(e)


def _run_federated_network_bg(
    rounds: int, local_epochs: int, port: int, seed: int,
    reject_suspicious_updates: bool, max_update_norm: Optional[float],
):
    """Launch and monitor one Flower server plus five owned client processes."""
    def update(snapshot: Dict[str, Any]) -> None:
        CURRENT_TASK["network_processes"] = snapshot
        completed = int(snapshot["round_completed"])
        CURRENT_TASK["progress"] = min(99, int(completed / max(rounds, 1) * 100))
        CURRENT_TASK["message"] = (
            f"Flower smoke: {snapshot['connected_clients']}/5 client đã kết nối, "
            f"{snapshot['client_processes_running']}/5 tiến trình đang chạy; "
            f"vòng hiện tại {snapshot['round_in_progress']}/{rounds}, hoàn tất {completed}/{rounds}; "
            f"{snapshot['accepted_clients_last_round']}/5 update hợp lệ ở vòng cuối."
        )
    try:
        result = NETWORK_DEMO.run(
            port=port,
            rounds=rounds,
            local_epochs=local_epochs,
            seed=seed,
            reject_suspicious_updates=reject_suspicious_updates,
            max_update_norm=max_update_norm,
            status_callback=update,
        )
        CURRENT_TASK["progress"] = 100
        CURRENT_TASK["status"] = "completed"
        CURRENT_TASK["message"] = f"Flower smoke hoàn tất {rounds} vòng với 5 client; đây là demo kỹ thuật trên ảnh tổng hợp."
        CURRENT_TASK["network_history_available"] = True
        CURRENT_TASK["run_dir"] = result["run_dir"]
    except Exception as e:
        if isinstance(e, InterruptedError) or CURRENT_TASK.get("cancel_requested"):
            CURRENT_TASK["status"] = "cancelled"
            CURRENT_TASK["message"] = "Đã dừng toàn bộ tiến trình Flower smoke."
        else:
            CURRENT_TASK["status"] = "failed"
            CURRENT_TASK["message"] = f"Flower network thất bại: {e}"
            CURRENT_TASK["error"] = str(e)


@app.post("/api/federated/train")
def trigger_federated_train(
    background_tasks: BackgroundTasks,
    req: FederatedTrainRequest = FederatedTrainRequest(),
):
    """
    Kích hoạt huấn luyện FedAvg:
    - Ràng buộc tham số: rounds >= 1, local_epochs >= 1.
    - Ánh xạ hệ số alpha tới thư mục phân hoạch non-IID tương ứng.
    - Tách biệt rõ ràng chế độ demo (smoke=True) và huấn luyện thật (smoke=False).
    - Bảo vệ checkpoint/metrics chính: demo run được lưu vào artifacts/demo_runs/fedavg/.
    """
    if CURRENT_TASK["status"] == "running":
        raise HTTPException(status_code=400, detail="Một tác vụ huấn luyện khác đang chạy. Vui lòng chờ tác vụ hiện tại hoàn tất.")

    engine = req.engine.strip().lower()
    if engine not in {"simulation", "network"}:
        raise HTTPException(status_code=400, detail="engine chỉ nhận 'simulation' hoặc 'network'.")

    if not req.smoke:
        raise HTTPException(status_code=400, detail="Chế độ ảnh thật chưa khả dụng: thiếu release mixed pv_pd_v3, manifest và W0.")

    if engine == "network":
        if req.rounds != 2 or req.local_epochs != 1:
            raise HTTPException(status_code=400, detail="Flower smoke demo cố định 2 vòng và 1 local epoch")
        if ":" not in req.server_address:
            raise HTTPException(status_code=400, detail="server_address phải có dạng HOST:PORT, ví dụ 127.0.0.1:8089")
        host, port_text = req.server_address.rsplit(":", 1)
        if host not in {"127.0.0.1", "localhost"} or not port_text.isdecimal():
            raise HTTPException(status_code=400, detail="Demo chỉ chấp nhận địa chỉ client localhost HOST:PORT")
        port = int(port_text)
        if not 1024 <= port <= 65534:
            raise HTTPException(status_code=400, detail="Port phải nằm trong khoảng 1024..65534")
        if not req.smoke or req.require_real_clients or req.require_real_evaluation or req.dataset_root:
            raise HTTPException(status_code=400, detail="Chế độ ảnh thật cần release mixed pv_pd_v3 và manifest; hiện web chỉ cho Flower smoke tổng hợp.")
        if req.num_clients != 5:
            raise HTTPException(status_code=400, detail="Demo Flower yêu cầu đúng 5 client")
        if abs(req.alpha - 0.1) > 1e-9:
            raise HTTPException(status_code=400, detail="Demo Flower dùng manifest phân vùng nhãn alpha=0.1")
        if check_port_open(port=port) or check_port_open(port=port + 1):
            raise HTTPException(status_code=409, detail="Cổng Flower hoặc cổng RTT đang được sử dụng")
        NETWORK_DEMO.prepare()
        CURRENT_TASK["config"] = req.model_dump()
        CURRENT_TASK.update({
            "status": "running", "experiment": "Flower localhost smoke",
            "type": "federated-network", "progress": 0,
            "message": "Đang khởi động Flower server và 5 client localhost...",
            "error": None, "cancel_requested": False, "live_progress": [],
            "network_history_available": False,
        })
        background_tasks.add_task(
            _run_federated_network_bg,
            req.rounds,
            req.local_epochs,
            port,
            req.seed,
            req.reject_suspicious_updates,
            req.max_update_norm,
        )
        return {
            "status": "started",
            "engine": "network",
            "message": (
                f"Đang khởi động Flower server 0.0.0.0:{port} và 5 client 127.0.0.1:{port}; "
                f"chạy {req.rounds} vòng smoke tổng hợp."
            ),
            "config": req.model_dump(),
        }

    # Ánh xạ alpha tới partition directory cho simulation
    alpha_map = {
        0.1: "data/partitions_stage2_scratch_v3/label01",
        1.0: "data/partitions_stage2_scratch_v3/label1",
        100.0: "data/partitions_stage2_scratch_v3/label100",
    }
    matched_key = next((alpha for alpha in alpha_map if abs(alpha - req.alpha) < 1e-9), None)
    if matched_key is None:
        raise HTTPException(status_code=400, detail="Simulation chỉ hỗ trợ alpha=0.1, 1.0 hoặc 100.0")
    if req.dataset_root:
        raise HTTPException(status_code=400, detail="Simulation web chỉ dùng smoke tổng hợp; dataset_root ảnh thật chưa khả dụng")
    part_dir = Path(alpha_map[matched_key])
    if not part_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Thư mục phân hoạch cho alpha={req.alpha} không tồn tại: {part_dir}"
        )

    CURRENT_TASK["config"] = req.model_dump()
    background_tasks.add_task(
        _run_federated_bg,
        req.rounds,
        req.local_epochs,
        str(part_dir),
        req.smoke,
        None,
        req.seed,
    )
    return {
        "status": "started",
        "message": f"Bắt đầu huấn luyện FedAvg ({req.rounds} rounds, E={req.local_epochs}, α={req.alpha}, {'demo' if req.smoke else 'ảnh thật'})",
        "config": req.model_dump()
    }


def _artifact_scientific_status(federated_source: str) -> Dict[str, Any]:
    """Return whether checkpoints are suitable for scientific reporting.

    A checkpoint being loadable is not enough.  Canonical artifacts must pass
    provenance validation; network artifacts must come from a run that required
    real clients and whose audit confirms every participating client used real
    PlantVillage images.
    """
    provenance_path = Path("artifacts/artifact_provenance_validation.json")
    provenance = {}
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except Exception:
            provenance = {}

    centralized_valid = provenance.get("status") == "verified"
    centralized_reason = (
        "canonical provenance verified"
        if centralized_valid
        else "canonical checkpoint/metrics provenance is unverified; see artifacts/artifact_provenance_validation.json"
    )

    if federated_source == "canonical":
        fed_valid = provenance.get("status") == "verified"
        fed_reason = centralized_reason
    elif federated_source == "network":
        history_path = Path("artifacts/metrics/network_flower_history.json")
        history = {}
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = {}
        require_real = bool(history.get("require_real_clients", False))
        require_real_evaluation = bool(history.get("require_real_evaluation", False))
        final_test_metrics = history.get("final_test_metrics")
        audits = history.get("audit_rounds", []) or []
        client_rows = [c for rnd in audits for c in (rnd.get("clients", []) or [])]
        all_real = bool(client_rows) and all(c.get("data_mode") == "real" for c in client_rows)
        has_real_test = (
            require_real_evaluation
            and isinstance(final_test_metrics, dict)
            and all(k in final_test_metrics for k in ("accuracy", "loss", "macro_f1"))
        )
        fed_valid = require_real and all_real and has_real_test
        if not history_path.exists():
            fed_reason = "no Flower network history found"
        elif not require_real:
            fed_reason = "network run did not enforce require_real_clients"
        elif not client_rows:
            fed_reason = "network run has no per-client data-mode audit"
        elif not all_real:
            fed_reason = "one or more Flower clients used synthetic/unknown data"
        elif not require_real_evaluation:
            fed_reason = "network run did not enforce a real held-out evaluator"
        elif not isinstance(final_test_metrics, dict):
            fed_reason = "network run has no final held-out test metrics"
        else:
            fed_reason = "network run used real clients and a real held-out evaluator with final test metrics"
    else:
        fed_valid = False
        fed_reason = "unknown federated source"

    return {
        "centralized": {"scientific_valid": centralized_valid, "reason": centralized_reason},
        "federated": {"scientific_valid": fed_valid, "reason": fed_reason},
        "overall_scientific_valid": centralized_valid and fed_valid,
    }


@app.get("/api/artifacts/status")
def get_artifact_status(federated_source: str = Query("canonical")) -> Dict[str, Any]:
    if federated_source not in {"canonical", "network"}:
        raise HTTPException(status_code=400, detail="federated_source chỉ nhận 'canonical' hoặc 'network'.")
    return _artifact_scientific_status(federated_source)


@app.post("/api/inference/plantvillage")
async def run_plantvillage_reference(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Standalone PlantVillage reference; independent from FedAvg comparison."""
    contents = await file.read()
    try:
        with Image.open(io.BytesIO(contents)) as image:
            return predict_plantvillage(image)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Ảnh hoặc model PlantVillage không hợp lệ: {exc}") from exc


@app.post("/api/inference")
async def run_dual_inference(
    file: UploadFile = File(...),
    federated_source: str = Query("fedavg"),
) -> Dict[str, Any]:
    """
    Inference song song độc lập (Dual Model):
    - Cột Centralized: Nạp model từ artifacts/checkpoints/centralized_best.pt.
    - Cột Federated: Nạp model từ artifacts/checkpoints/fedavg_best.pt.
    - Tuyệt đối không tráo đổi hay dùng chung một file TorchScript.
    - Nếu checkpoint chưa sẵn sàng, trả về status: not_ready để giao diện thông báo rõ ràng.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File tải lên không phải là ảnh hợp lệ")

    if federated_source not in {"fedavg", *(f"local_{i:02d}" for i in range(5))}:
        raise HTTPException(status_code=400, detail="Chỉ chọn FedAvg v7 hoặc một trong năm Local-only v7 cùng protocol.")
    try:
        central = predict_published(image, "centralized")
        selected = predict_published(image, federated_source)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Artifact suy luận chưa sẵn sàng: {exc}") from exc
    selected_name = "FedAvg v7" if federated_source == "fedavg" else f"Local {federated_source[-2:]}"
    return {
        "filename": file.filename,
        "federated_source": federated_source,
        "selected_model_name": selected_name,
        "protocol_id": PROTOCOL_ID,
        "scientific_status": {
            "evidence_level": "published_artifact_hash_verified_not_locally_reevaluated",
            "overall_scientific_valid": False,
            "reason": "Cùng validation nội bộ; chưa tái đánh giá tại máy demo hoặc trên ảnh ngoài mạng.",
        },
        "centralized": central,
        "federated": selected,
    }



# Mount thư mục static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8088))
    uvicorn.run("web.app:app", host="127.0.0.1", port=port, reload=False)

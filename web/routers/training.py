"""Router training: điều khiển centralized/federated demo + network metrics."""
from __future__ import annotations

import functools
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from src import registry as reg

from web import state
from web.services import admin_auth, training_guard
from web.state import CURRENT_TASK, NETWORK_DEMO, check_port_open

from src.centralized.train import run_centralized_training
from src.federated.server_app import run_federated_simulation

router = APIRouter()

# Chỉ một tác vụ huấn luyện cùng lúc: khối "kiểm trạng thái rồi đặt running" phải nguyên tử, vì các endpoint `def` chạy
# trên threadpool nên hai yêu cầu đồng thời có thể cùng thấy status != running.
_TASK_LOCK = threading.Lock()


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


@router.get("/api/federated/network-metrics")
def get_federated_network_metrics() -> Dict[str, Any]:
    """Return persisted metrics from the real Flower network server path."""
    current_run = NETWORK_DEMO.run_dir
    historical_run = current_run is None
    if historical_run:
        current_run = state.latest_completed_smoke_run()
    metrics_path = current_run / "history.json" if current_run else state.SMOKE_RUNS_ROOT / "none/history.json"
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


def _mark_task_running(experiment: str, task_type: str, message: str) -> None:
    """Đặt tác vụ nền thành "running" NGAY trong endpoint, không đợi BackgroundTask (chạy sau khi đã trả lời).

    Nếu để BackgroundTask đặt, lệnh Dừng gửi sát sau Bắt đầu sẽ thấy status != running và bị từ chối,
    còn cancel_requested=False đặt muộn sẽ xóa luôn lệnh dừng đến sớm."""
    CURRENT_TASK.update({
        "status": "running", "experiment": experiment, "type": task_type, "progress": 5,
        "message": message, "error": None, "cancel_requested": False, "live_progress": [],
        "job_id": uuid.uuid4().hex[:12],
    })


def _expire_job(job_id: str) -> None:
    """Quá thời gian tối đa: dừng đúng job đó (bỏ qua nếu job đã xong hoặc đã là job khác)."""
    if CURRENT_TASK.get("status") != "running" or CURRENT_TASK.get("job_id") != job_id:
        return
    CURRENT_TASK["cancel_requested"] = True
    CURRENT_TASK["message"] = f"Đã quá {training_guard.limits()['max_seconds']} giây: tự dừng để bảo vệ máy chủ."
    if CURRENT_TASK.get("type") == "federated-network":
        NETWORK_DEMO.stop()
    training_guard.audit("train_timeout", None, job_id=job_id)


def _supervised(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bọc hàm chạy nền: hẹn giờ dừng khi quá thời gian và ghi nhật ký kết thúc (giữ nguyên __name__ cho test)."""
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def inner(*args: Any, **kwargs: Any) -> Any:
            job_id = str(CURRENT_TASK.get("job_id") or "")
            timer = training_guard.arm_watchdog(job_id, _expire_job)
            try:
                return function(*args, **kwargs)
            finally:
                timer.cancel()
                training_guard.audit("train_end", None, kind=kind, job_id=job_id,
                                     status=CURRENT_TASK.get("status"), message=CURRENT_TASK.get("message"))
        return inner
    return decorate


def _guard_start(request: Optional[Request]) -> None:
    """Hạn mức số lần bấm Bắt đầu theo IP (chỉ áp cho yêu cầu HTTP thật)."""
    if request is None:
        return
    allowed, wait = admin_auth.LIMITER.allow(
        f"trainstart:{admin_auth.client_ip(request)}", training_guard.limits()["starts_per_10_min"], 600)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn bấm Bắt đầu quá nhiều lần. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})


def _cap(name: str, value: int, label: str) -> None:
    cap = training_guard.limits()[name]
    if value > cap:
        raise HTTPException(status_code=422, detail=f"{label}: máy chủ này chỉ cho tối đa {cap} (bạn chọn {value}).")


@router.post("/api/train/stop", dependencies=[Depends(admin_auth.require_admin)])
def stop_training(request: Request = None) -> Dict[str, Any]:  # type: ignore[assignment]
    """Hủy/Dừng tác vụ huấn luyện đang chạy (Centralized hoặc Federated). Cần quyền quản trị khi máy chủ đã cấu hình."""
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
    training_guard.audit("train_stop", request, job_id=CURRENT_TASK.get("job_id"), kind=CURRENT_TASK.get("type"))
    return {
        "status": "stopping",
        "message": "Đã gửi tín hiệu dừng khẩn cấp tới tiến trình huấn luyện.",
        "stopped": True,
    }


@_supervised("centralized")
def _run_centralized_bg(epochs: int, smoke: bool, dataset_root: Optional[str], seed: int):
    # Trạng thái running/cancel_requested đã được endpoint đặt đồng bộ (_mark_task_running); không đặt lại ở đây.
    mode_text = "demo" if smoke else "thực tế"

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
        output_dir = reg.abs_path(reg.WEB_RUNS_DIR) / ("centralized_demo" if smoke else "centralized_full")
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


@router.post("/api/centralized/train", dependencies=[Depends(admin_auth.require_admin)])
def trigger_centralized_train(
    background_tasks: BackgroundTasks,
    req: CentralizedTrainRequest = CentralizedTrainRequest(),
    request: Request = None,  # type: ignore[assignment]
):
    """
    Kích hoạt huấn luyện Centralized (chỉ smoke demo):
    - Demo run lưu vào artifacts/web_runs/centralized_demo/, không đụng slot canonical.
    - Cần quyền quản trị khi máy chủ đã cấu hình; chỉ một tác vụ một lúc (409 nếu đang chạy).
    """
    with _TASK_LOCK:
        _guard_start(request)
        return _start_centralized(background_tasks, req, request)


def _start_centralized(background_tasks: BackgroundTasks, req: CentralizedTrainRequest, request: Optional[Request]):
    if CURRENT_TASK["status"] == "running":
        raise HTTPException(status_code=409, detail="Một tác vụ huấn luyện khác đang chạy. Vui lòng chờ tác vụ hiện tại hoàn tất.")

    if not req.smoke:
        raise HTTPException(status_code=400, detail="Chế độ ảnh thật chưa khả dụng: thiếu release mixed pv_pd_v3, manifest và W0.")
    _cap("max_centralized_epochs", req.epochs, "Số epoch")
    training_guard.require_memory("inprocess")

    CURRENT_TASK["config"] = req.model_dump()
    _mark_task_running(
        "Centralized", "centralized",
        f"Đang khởi động huấn luyện Centralized ({req.epochs} epochs, {'demo' if req.smoke else 'thực tế'})...",
    )
    training_guard.audit("train_start", request, kind="centralized", job_id=CURRENT_TASK["job_id"], config=req.model_dump())
    background_tasks.add_task(_run_centralized_bg, req.epochs, req.smoke, None, req.seed)
    return {
        "status": "started",
        "message": f"Bắt đầu huấn luyện Centralized ({req.epochs} epochs, {'demo' if req.smoke else 'ảnh thật'})",
        "config": req.model_dump()
    }


@_supervised("federated")
def _run_federated_bg(rounds: int, local_epochs: int, partition_dir: str, smoke: bool, dataset_root: Optional[str], seed: int):
    # Trạng thái running/cancel_requested đã được endpoint đặt đồng bộ (_mark_task_running); không đặt lại ở đây.
    mode_text = "demo" if smoke else "thực tế"

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
        output_dir = reg.abs_path(reg.WEB_RUNS_DIR) / ("fedavg_demo" if smoke else "fedavg_full")
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


@_supervised("federated-network")
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
            f"Chạy thử Flower: {snapshot['connected_clients']}/5 client đã kết nối, "
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
        CURRENT_TASK["message"] = f"Chạy thử Flower hoàn tất {rounds} vòng với 5 client; đây là demo kỹ thuật trên ảnh tổng hợp."
        CURRENT_TASK["network_history_available"] = True
        CURRENT_TASK["run_dir"] = result["run_dir"]
    except Exception as e:
        if isinstance(e, InterruptedError) or CURRENT_TASK.get("cancel_requested"):
            CURRENT_TASK["status"] = "cancelled"
            CURRENT_TASK["message"] = "Đã dừng toàn bộ tiến trình chạy thử Flower."
        else:
            CURRENT_TASK["status"] = "failed"
            CURRENT_TASK["message"] = f"Flower network thất bại: {e}"
            CURRENT_TASK["error"] = str(e)


@router.post("/api/federated/train", dependencies=[Depends(admin_auth.require_admin)])
def trigger_federated_train(
    background_tasks: BackgroundTasks,
    req: FederatedTrainRequest = FederatedTrainRequest(),
    request: Request = None,  # type: ignore[assignment]
):
    """
    Kích hoạt huấn luyện FedAvg (chỉ smoke demo, simulation hoặc network).
    Demo run lưu vào artifacts/web_runs/, không đụng slot canonical.
    Cần quyền quản trị khi máy chủ đã cấu hình; chỉ một tác vụ một lúc (409 nếu đang chạy).
    """
    with _TASK_LOCK:
        _guard_start(request)
        return _start_federated(background_tasks, req, request)


def _start_federated(background_tasks: BackgroundTasks, req: FederatedTrainRequest, request: Optional[Request]):
    if CURRENT_TASK["status"] == "running":
        raise HTTPException(status_code=409, detail="Một tác vụ huấn luyện khác đang chạy. Vui lòng chờ tác vụ hiện tại hoàn tất.")

    engine = req.engine.strip().lower()
    if engine not in {"simulation", "network"}:
        raise HTTPException(status_code=400, detail="engine chỉ nhận 'simulation' hoặc 'network'.")

    if not req.smoke:
        raise HTTPException(status_code=400, detail="Chế độ ảnh thật chưa khả dụng: thiếu release mixed pv_pd_v3, manifest và W0.")

    if engine == "network":
        if req.rounds != 2 or req.local_epochs != 1:
            raise HTTPException(status_code=400, detail="Chạy thử Flower cố định 2 vòng và 1 epoch cục bộ")
        if ":" not in req.server_address:
            raise HTTPException(status_code=400, detail="server_address phải có dạng HOST:PORT, ví dụ 127.0.0.1:8089")
        host, port_text = req.server_address.rsplit(":", 1)
        if host not in {"127.0.0.1", "localhost"} or not port_text.isdecimal():
            raise HTTPException(status_code=400, detail="Demo chỉ chấp nhận địa chỉ client localhost HOST:PORT")
        port = int(port_text)
        if not 1024 <= port <= 65534:
            raise HTTPException(status_code=400, detail="Port phải nằm trong khoảng 1024..65534")
        if not req.smoke or req.require_real_clients or req.require_real_evaluation or req.dataset_root:
            raise HTTPException(status_code=400, detail="Chế độ ảnh thật cần release mixed pv_pd_v3 và manifest; hiện web chỉ cho chạy thử Flower trên ảnh tổng hợp.")
        if req.num_clients != 5:
            raise HTTPException(status_code=400, detail="Demo Flower yêu cầu đúng 5 client")
        if abs(req.alpha - 0.1) > 1e-9:
            raise HTTPException(status_code=400, detail="Demo Flower dùng manifest phân vùng nhãn alpha=0.1")
        if check_port_open(port=port) or check_port_open(port=port + 1):
            raise HTTPException(status_code=409, detail="Cổng Flower hoặc cổng RTT đang được sử dụng")
        training_guard.require_memory("network")
        NETWORK_DEMO.prepare()
        CURRENT_TASK["config"] = req.model_dump()
        CURRENT_TASK.update({
            "status": "running", "experiment": "Chạy thử Flower trên máy này",
            "type": "federated-network", "progress": 0,
            "message": "Đang khởi động Flower server và 5 client localhost...",
            "error": None, "cancel_requested": False, "live_progress": [],
            "network_history_available": False, "job_id": uuid.uuid4().hex[:12],
        })
        training_guard.audit("train_start", request, kind="federated-network", job_id=CURRENT_TASK["job_id"],
                             config=req.model_dump())
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
                f"chạy thử {req.rounds} vòng trên ảnh tổng hợp."
            ),
            "config": req.model_dump(),
        }

    # Ánh xạ alpha tới partition directory cho simulation
    alpha_map = {
        0.1: f"{reg.PARTITIONS_DIR}/label01",
        1.0: f"{reg.PARTITIONS_DIR}/label1",
        100.0: f"{reg.PARTITIONS_DIR}/label100",
    }
    matched_key = next((alpha for alpha in alpha_map if abs(alpha - req.alpha) < 1e-9), None)
    if matched_key is None:
        raise HTTPException(status_code=400, detail="Simulation chỉ hỗ trợ alpha=0.1, 1.0 hoặc 100.0")
    if req.dataset_root:
        raise HTTPException(status_code=400, detail="Simulation web chỉ dùng smoke tổng hợp; dataset_root ảnh thật chưa khả dụng")
    _cap("max_rounds", req.rounds, "Số vòng")
    _cap("max_local_epochs", req.local_epochs, "Epoch cục bộ")
    training_guard.require_memory("inprocess")
    part_dir = Path(alpha_map[matched_key])
    if not part_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Thư mục phân hoạch cho alpha={req.alpha} không tồn tại: {part_dir}"
        )

    CURRENT_TASK["config"] = req.model_dump()
    _mark_task_running(
        "Federated", "federated",
        f"Đang điều phối FedAvg ({req.rounds} rounds, E={req.local_epochs}, {'demo' if req.smoke else 'thực tế'})...",
    )
    training_guard.audit("train_start", request, kind="federated", job_id=CURRENT_TASK["job_id"], config=req.model_dump())
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

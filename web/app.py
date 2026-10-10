"""
web.app: Máy chủ FastAPI cung cấp REST API và giao diện Web Dashboard cho hệ thống Federated Learning.

Kiến trúc (từ 2026-10-09, xem docs/MODEL_REGISTRY.md):
- web/routers/monitoring.py  — status, metrics, clients, weights inspect, artifact status
- web/routers/inference.py   — dual inference, PlantVillage reference, demo evidence, compare
- web/routers/training.py    — điều khiển centralized/federated demo + network metrics
- web/services/model_service.py — service registry/inference duy nhất (đọc artifacts/registry.json)
- web/state.py               — trạng thái background task + NetworkDemoManager dùng chung

Web chỉ làm nhiệm vụ Presentation, API và Inference; tuyệt đối không chứa logic
training nặng nề bên trong và không tự hardcode đường dẫn checkpoint.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.routers import inference as _inference_router
from web.routers import monitoring as _monitoring_router
from web.routers import training as _training_router
from web.state import CURRENT_TASK, NETWORK_DEMO, latest_completed_smoke_run  # noqa: F401

# Re-export các tên public để import cũ (`from web.app import ...`) tiếp tục hoạt động
from web.routers.training import (  # noqa: F401
    CentralizedTrainRequest,
    FederatedTrainRequest,
    _run_centralized_bg,
    _run_federated_bg,
    _run_federated_network_bg,
    get_federated_network_metrics,
    stop_training,
    trigger_centralized_train,
    trigger_federated_train,
)
from web.routers.monitoring import (  # noqa: F401
    get_artifact_status,
    get_clients_info,
    get_metrics,
    get_system_status,
    inspect_weights,
)
from web.services.model_service import artifact_scientific_status as _artifact_scientific_status  # noqa: F401
from web.routers.inference import (  # noqa: F401
    get_comparison_table,
    get_demo_evidence,
    run_dual_inference,
    run_plantvillage_reference,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        NETWORK_DEMO.stop()


app = FastAPI(
    title="Federated Learning PlantVillage Dashboard",
    description="Hệ thống điều khiển, quan sát và suy luận (inference) Federated Learning trên MobileNetV3-Small",
    version="1.2.0",
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


app.include_router(_monitoring_router.router)
app.include_router(_inference_router.router)
app.include_router(_training_router.router)

# Mount thư mục static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8088))
    uvicorn.run("web.app:app", host="127.0.0.1", port=port, reload=False)

"""
web.app: Máy chủ FastAPI cung cấp REST API và giao diện Web Dashboard cho hệ thống Federated Learning.

Kiến trúc (từ 2026-10-09, xem docs/MODEL_REGISTRY.md):
- web/routers/monitoring.py  — status, metrics, clients, weights inspect, artifact status
- web/routers/inference.py   — dual inference, PlantVillage reference, demo evidence, compare
- web/routers/training.py    — điều khiển centralized/federated demo + network metrics
- web/routers/community_fl.py — học phân tán cộng đồng: mỗi người vào web là một client (xem docs/COMMUNITY_FL.md)
- web/services/model_service.py — service registry/inference duy nhất (đọc artifacts/registry.json)
- web/state.py               — trạng thái background task + NetworkDemoManager dùng chung

Web phục vụ giao diện, API và suy luận. Huấn luyện thử Centralized và FedAvg mô phỏng chạy TRONG tiến trình web
(BackgroundTasks) dưới khóa một tác vụ, trần tham số, hẹn giờ và kiểm RAM (web/services/training_guard.py); phiên Flower
chạy NGOÀI tiến trình (1 server + 5 client, web/network_demo.py). Không tự hardcode đường dẫn checkpoint.
"""
from __future__ import annotations

import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from web.routers import community_fl as _community_router
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

# Giao diện và API cùng một origin nên mặc định không cần CORS cho trang khác. `CORS_ALLOW_ORIGINS` (danh sách phân tách bằng
# dấu phẩy) mở thêm origin cụ thể khi cần; không còn `*` kèm credentials. Token cộng đồng đi bằng header Authorization.
_CORS_ORIGINS = [item.strip() for item in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if item.strip()]
if _CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


LONG_CACHE_PREFIXES = ("/static/fonts/", "/static/vendor/")


def cache_control_for(path: str, status: int) -> str | None:
    """Chính sách cache theo đường dẫn + mã trạng thái (hàm thuần để test được).

    - Phông và thư viện ghim phiên bản hiếm khi đổi: giữ một ngày, KỂ CẢ khi trình duyệt hỏi lại và nhận 304
      (trước đây 304 rơi vào nhánh no-store làm trình duyệt vứt bản đã giữ).
    - Mã nguồn giao diện: `no-cache` = luôn hỏi lại máy chủ nhưng được nhận 304 (ETag) thay vì tải lại cả tệp. Các module JS
      import lẫn nhau không có ?v= nên KHÔNG được dùng max-age cho chúng, nếu không bản cũ sẽ còn lại sau khi phát hành.
    - 404/lỗi không bao giờ được cache lâu."""
    if path.startswith(LONG_CACHE_PREFIXES) and status in (200, 304):
        return "public, max-age=86400"
    if path == "/" or path.startswith("/static/"):
        return "no-cache"
    return None


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    policy = cache_control_for(request.url.path, response.status_code)
    if policy is not None:
        response.headers["Cache-Control"] = policy
    return response


# ───────── Tiêu đề bảo mật cho trang chính ─────────
# CSP cho mọi trang HTML (trừ /docs, /redoc của FastAPI nạp Swagger từ CDN và /quan-tri đã có chính sách riêng chặt hơn).
# - script-src 'self' + 'wasm-unsafe-eval': onnxruntime-web biên dịch WebAssembly trong Web Worker; KHÔNG có 'unsafe-eval'/'unsafe-inline'.
# - style-src 'self': giao diện không dùng style="" (dùng data-style → CSSOM, xem core/dom.js applyStyleData).
# - img-src data:/blob: cho favicon, mũi tên của <select>, ảnh Grad-CAM từ API và ảnh xem trước của người dùng.
# Không gắn CSP lên tệp JS của worker: chính sách của worker lấy từ phản hồi của nó, bỏ trống thì wasm biên dịch được ở mọi trình duyệt.
PAGE_CSP = (
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data: blob:; "
    "font-src 'self'; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
CSP_MODE_ENV = "CROPFED_CSP_MODE"  # enforce (mặc định) | report | off — công tắc lùi nhanh không cần sửa mã
CSP_EXEMPT_PREFIXES = ("/docs", "/redoc", "/openapi.json")


def security_headers_for(path: str, content_type: str, csp_mode: str | None = None) -> dict[str, str]:
    """Tiêu đề bảo mật theo đường dẫn và loại nội dung (hàm thuần để test được). Dùng setdefault khi gắn để /quan-tri giữ chính sách riêng."""
    headers = {"X-Content-Type-Options": "nosniff"}
    if not content_type.startswith("text/html") or path.startswith(CSP_EXEMPT_PREFIXES):
        return headers
    headers["Referrer-Policy"] = "same-origin"
    headers["X-Frame-Options"] = "DENY"
    mode = (csp_mode if csp_mode is not None else os.environ.get(CSP_MODE_ENV, "enforce")).strip().lower()
    if mode == "off":
        return headers
    headers["Content-Security-Policy-Report-Only" if mode == "report" else "Content-Security-Policy"] = PAGE_CSP
    return headers


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in security_headers_for(request.url.path, response.headers.get("content-type", "")).items():
        response.headers.setdefault(name, value)
    return response


# Windows/Python thường không biết .woff2 và trả text/plain; khai báo tường minh để không phụ thuộc registry của OS.
mimetypes.add_type("font/woff2", ".woff2")
# Worker module và onnxruntime-web: trình duyệt từ chối script module/wasm nếu MIME sai (Windows hay trả text/plain).
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)


# Cùng một đối tượng StaticFiles vừa gắn ở /static vừa phục vụ trang chủ: có ETag/Last-Modified và trả 304 cho "/".
_STATIC_FILES = StaticFiles(directory=str(STATIC_DIR))


@app.get("/")
async def serve_index(request: Request):
    try:
        return await _STATIC_FILES.get_response("index.html", request.scope)
    except StarletteHTTPException:
        return {"message": "Federated Learning API is running. static/index.html is being prepared."}


ADMIN_PAGE = Path(__file__).resolve().parent / "static" / "admin" / "admin.html"
# Trang quản trị học phân tán cộng đồng: tệp tĩnh riêng, KHÔNG nằm trong menu của trang chính và không có liên kết từ menu công khai.
# Quyền thật nằm ở token quản trị mà các API /api/community/admin/* kiểm; các tiêu đề dưới đây chỉ làm trang khó bị nhúng/lập chỉ mục.
ADMIN_PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; font-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
}


@app.get("/quan-tri", include_in_schema=False)
def serve_admin_page():
    if not ADMIN_PAGE.is_file():
        return JSONResponse({"detail": "Trang quản trị chưa có trên máy chủ này."}, status_code=404, headers=ADMIN_PAGE_HEADERS)
    return FileResponse(ADMIN_PAGE, headers=ADMIN_PAGE_HEADERS)


app.include_router(_monitoring_router.router)
app.include_router(_inference_router.router)
app.include_router(_training_router.router)
app.include_router(_community_router.router)

# Mount thư mục static assets
app.mount("/static", _STATIC_FILES, name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8088))
    uvicorn.run("web.app:app", host="127.0.0.1", port=port, reload=False)

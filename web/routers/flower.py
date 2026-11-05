"""Hai endpoint đọc của trang Flower: /api/flower/health (kết luận) và /api/flower/evidence/{id} (bằng chứng thô theo mục).

Logic ở web/services/flower_health.py và flower_evidence.py. Cả hai đều chỉ đọc và chỉ nhận danh sách mục đã định sẵn,
không nhận đường dẫn hay lệnh từ client.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src import registry as reg
from src.federated.run_detail import ITEM, detail_for_run, tamper_view
from src.federated.run_view import list_runs, load_run, run_summary, run_view
from web import state
from web.services import admin_auth, client_package, flower_demo, flower_evidence, flower_health

router = APIRouter(prefix="/api/flower", tags=["flower"])

PER_MINUTE = 30
EVIDENCE_PER_MINUTE = 20
PACKAGE_PER_HOUR = 20       # một gói vài trăm MB, cho phép tải lại và tải tiếp nhưng không để một IP tải mãi
LIVE_PER_MINUTE = 120       # trang hỏi mỗi 3 giây; cả lớp xem chung một IP vẫn đủ
LIVE_CACHE_S = 2.0          # nhiều người cùng xem chỉ đọc tệp một lần mỗi 2 giây
RUNS_CACHE_S = 5.0          # phải đọc tệp của mọi lần đo, và cả ba trang demo đều hỏi
RUN_NAME = re.compile(r"^[0-9A-Za-z_-]{1,80}$")
_live_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_detail_cache: Dict[str, tuple[float, Optional[Dict[str, Any]]]] = {}
_runs_cache: Optional[tuple[float, Dict[str, Any]]] = None
_live_lock = threading.Lock()


def _live(run: Optional[str]) -> Dict[str, Any]:
    root = reg.abs_path(reg.NETWORK_LIVE_DIR)
    runs = list_runs(root)
    names = [path.name for path in runs]
    if run is not None and run not in names:     # chỉ nhận đúng tên đã liệt kê, không ghép đường dẫn từ dữ liệu người dùng
        raise HTTPException(status_code=404, detail="Không có lần đo này.")
    if not runs:
        return {"available": False, "state": "none", "runs": []}
    tamper = tamper_view(reg.abs_path(reg.NETWORK_TAMPER_DIR))
    summary = None if tamper is None else {"run_id": tamper["run_id"], "client_id": tamper["rejected"]["client_id"], "client_hash": tamper["rejected"]["client_hash"], "server_hash": tamper["rejected"]["server_hash"]}
    return {**run_view(load_run(root / (run or names[0]))), "runs": names, "tamper_check": summary}


def _cached_live(run: Optional[str]) -> Dict[str, Any]:
    key = run or ""
    now = time.monotonic()
    with _live_lock:
        cached = _live_cache.get(key)
        if cached and now - cached[0] < LIVE_CACHE_S:
            return cached[1]
    built = _live(run)
    with _live_lock:
        if len(_live_cache) > 64:
            _live_cache.clear()
        _live_cache[key] = (now, built)
    return built


def _build() -> Dict[str, Any]:
    return flower_health.build_health(
        snapshot=state.NETWORK_DEMO.snapshot(),
        port_open=lambda port: state.check_port_open("127.0.0.1", port),
    )


@router.get("/live")
async def get_flower_live(request: Request, run: Optional[str] = Query(default=None, max_length=80)) -> Dict[str, Any]:
    """Lần đo thật do terminal chạy (scripts/remote/run_network_server.sh). Cùng hàm tính với scripts/demo_show.py. Không chứa IP."""
    allowed, wait = admin_auth.LIMITER.allow(f"flower-live:{admin_auth.client_ip(request)}", LIVE_PER_MINUTE, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn thao tác quá nhanh. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})
    if run is not None and not RUN_NAME.fullmatch(run):
        raise HTTPException(status_code=404, detail="Không có lần đo này.")
    return await run_in_threadpool(_cached_live, run)


def _cached_runs() -> Dict[str, Any]:
    global _runs_cache
    now = time.monotonic()
    with _live_lock:
        if _runs_cache and now - _runs_cache[0] < RUNS_CACHE_S:
            return _runs_cache[1]
    built = {"runs": [run_summary(run_view(load_run(path))) for path in list_runs(reg.abs_path(reg.NETWORK_LIVE_DIR))]}
    with _live_lock:
        _runs_cache = (now, built)
    return built


@router.get("/runs")
async def get_flower_runs(request: Request) -> Dict[str, Any]:
    """Mỗi lần đo một dòng, mới nhất trước, cho biểu đồ so các lần đo và dải nối ba trang demo. Không chứa IP."""
    allowed, wait = admin_auth.LIMITER.allow(f"flower-live:{admin_auth.client_ip(request)}", LIVE_PER_MINUTE, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn thao tác quá nhanh. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})
    return await run_in_threadpool(_cached_runs)


def _cached_detail(run: Optional[str], item: str, round_no: Optional[int]) -> Optional[Dict[str, Any]]:
    root = reg.abs_path(reg.NETWORK_LIVE_DIR)
    names = [path.name for path in list_runs(root)]
    if not names or (run is not None and run not in names):     # chỉ nhận đúng tên đã liệt kê, không ghép đường dẫn từ dữ liệu người dùng
        return None
    key = f"{run or names[0]}|{item}|{round_no}"
    now = time.monotonic()
    with _live_lock:
        cached = _detail_cache.get(key)
        if cached and now - cached[0] < LIVE_CACHE_S:
            return cached[1]
    built = detail_for_run(root / (run or names[0]), item, round_no, tamper_root=reg.abs_path(reg.NETWORK_TAMPER_DIR))
    with _live_lock:
        if len(_detail_cache) > 256:
            _detail_cache.clear()
        _detail_cache[key] = (now, built)
    return built


@router.get("/detail")
async def get_flower_detail(
    request: Request,
    item: str = Query(max_length=16),
    run: Optional[str] = Query(default=None, max_length=80),
    round: Optional[int] = Query(default=None, ge=1, le=100000),
) -> Dict[str, Any]:
    """Một mục trên trang: số đó làm ra thế nào, kèm các dòng terminal. Cùng hàm với scripts/demo_show.py flower --detail. Không chứa IP."""
    allowed, wait = admin_auth.LIMITER.allow(f"flower-detail:{admin_auth.client_ip(request)}", LIVE_PER_MINUTE, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn thao tác quá nhanh. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})
    if not ITEM.fullmatch(item) or (run is not None and not RUN_NAME.fullmatch(run)):
        raise HTTPException(status_code=404, detail="Không có mục này.")
    built = await run_in_threadpool(_cached_detail, run, item, round)
    if built is None:
        raise HTTPException(status_code=404, detail="Không có mục này.")
    return built


def _package_or_503() -> Dict[str, Any]:
    package = client_package.info()
    if package is None:
        raise HTTPException(status_code=503, detail="Máy chủ này chưa có gói client.")
    return package


def _limit_package(request: Request) -> None:
    allowed, wait = admin_auth.LIMITER.allow(f"client-package:{admin_auth.client_ip(request)}", PACKAGE_PER_HOUR, 3600)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Tải quá nhiều lần. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})


@router.get("/client-package/status")
async def get_client_package_status() -> Dict[str, Any]:
    """Trang dùng để biết có nên hiện nút tải bộ cài không."""
    package = client_package.info()
    return {"available": True, "bytes": package["bytes"]} if package else {"available": False}


@router.get("/client-package/install.bat")
async def get_client_package_installer(request: Request) -> Response:
    """File cài nhỏ, điền sẵn địa chỉ và mã băm của gói. Không cần đăng nhập: laptop cần cài thường chưa đăng nhập, và gói không có gì bí mật."""
    _limit_package(request)
    package = _package_or_503()
    try:
        body = client_package.installer(str(request.base_url), package)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Địa chỉ không hợp lệ.") from error
    return Response(content=body, media_type="application/octet-stream",
                    headers={"Content-Disposition": 'attachment; filename="CAI_CLIENT_CROPFED.bat"', "Cache-Control": "no-store"})


@router.get("/client-package/file")
async def get_client_package_file(request: Request) -> FileResponse:
    _limit_package(request)
    _package_or_503()
    return FileResponse(client_package.zip_path(), media_type="application/zip", filename=client_package.ZIP_NAME)


class DemoStart(BaseModel):
    clients: int = Field(default=7, ge=1, le=10)
    rounds: int = Field(default=3, ge=1, le=20)
    light: bool = False       # bộ ảnh nhẹ trên máy chủ; đường dẫn do máy chủ tự biết, không nhận từ trang


@router.get("/demo/status")
async def get_flower_demo_status(request: Request) -> Dict[str, Any]:
    """Nút "Bắt đầu lần đo" có dùng được ở máy chủ này không, và có lần đo nào đang chạy từ nút không. Không chứa đường dẫn."""
    allowed, wait = admin_auth.LIMITER.allow(f"flower-demo:{admin_auth.client_ip(request)}", LIVE_PER_MINUTE, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn thao tác quá nhanh. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})
    return flower_demo.status()


@router.post("/demo/start", dependencies=[Depends(admin_auth.require_admin)])
async def post_flower_demo_start(body: DemoStart) -> Dict[str, Any]:
    """Mở máy chủ đo. Cổng đo mở cho mọi IP trong lúc đo và tự đóng khi xong, nên chỉ quản trị viên được bấm."""
    try:
        return await run_in_threadpool(flower_demo.start, body.clients, body.rounds, body.light)
    except flower_demo.DemoError as error:
        raise HTTPException(status_code=error.status, detail=error.message) from error


@router.post("/demo/stop", dependencies=[Depends(admin_auth.require_admin)])
async def post_flower_demo_stop() -> Dict[str, Any]:
    try:
        return await run_in_threadpool(flower_demo.stop)
    except flower_demo.DemoError as error:
        raise HTTPException(status_code=error.status, detail=error.message) from error


@router.get("/health")
async def get_flower_health(request: Request) -> Dict[str, Any]:
    allowed, wait = admin_auth.LIMITER.allow(f"flower:{admin_auth.client_ip(request)}", PER_MINUTE, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn thao tác quá nhanh. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})
    # đọc tệp và nạp checkpoint nhỏ: chạy ngoài vòng lặp sự kiện để không chặn /api/status mà giao diện đang poll
    return await run_in_threadpool(_build)


@router.get("/evidence/{item_id}")
async def get_flower_evidence(item_id: str, request: Request) -> Dict[str, Any]:
    """Bằng chứng thô của một mục: lệnh đã chạy + output, log gốc, bản ghi. Gọi khi người xem bấm mở, không poll theo trang."""
    allowed, wait = admin_auth.LIMITER.allow(f"flower-ev:{admin_auth.client_ip(request)}", EVIDENCE_PER_MINUTE, 60)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Bạn mở bảng bằng chứng quá nhanh. Hãy thử lại sau {int(wait)} giây.",
                            headers={"Retry-After": str(int(wait))})
    built = await run_in_threadpool(flower_evidence.cached_build, item_id, state.NETWORK_DEMO.snapshot())
    if built is None:  # id ngoài danh sách: trả 404 thay vì 422 để không lộ rằng có những mục khác tồn tại
        raise HTTPException(status_code=404, detail="Không có mục kiểm tra này.")
    return built

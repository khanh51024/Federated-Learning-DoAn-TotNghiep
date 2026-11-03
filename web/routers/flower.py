"""Hai endpoint đọc của trang Flower: /api/flower/health (kết luận) và /api/flower/evidence/{id} (bằng chứng thô theo mục).

Logic ở web/services/flower_health.py và flower_evidence.py. Cả hai đều chỉ đọc và chỉ nhận danh sách mục đã định sẵn,
không nhận đường dẫn hay lệnh từ client.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from src import registry as reg
from src.federated.run_view import list_runs, load_run, run_view
from web import state
from web.services import admin_auth, flower_evidence, flower_health

router = APIRouter(prefix="/api/flower", tags=["flower"])

PER_MINUTE = 30
EVIDENCE_PER_MINUTE = 20
LIVE_PER_MINUTE = 120       # trang hỏi mỗi 3 giây; cả lớp xem chung một IP vẫn đủ
LIVE_CACHE_S = 2.0          # nhiều người cùng xem chỉ đọc tệp một lần mỗi 2 giây
RUN_NAME = re.compile(r"^[0-9A-Za-z_-]{1,80}$")
_live_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_live_lock = threading.Lock()


def _live(run: Optional[str]) -> Dict[str, Any]:
    root = reg.abs_path(reg.NETWORK_LIVE_DIR)
    runs = list_runs(root)
    names = [path.name for path in runs]
    if run is not None and run not in names:     # chỉ nhận đúng tên đã liệt kê, không ghép đường dẫn từ dữ liệu người dùng
        raise HTTPException(status_code=404, detail="Không có lần đo này.")
    if not runs:
        return {"available": False, "state": "none", "runs": []}
    return {**run_view(load_run(root / (run or names[0]))), "runs": names}


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

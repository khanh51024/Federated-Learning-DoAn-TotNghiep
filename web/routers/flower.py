"""Hai endpoint đọc của trang Flower: /api/flower/health (kết luận) và /api/flower/evidence/{id} (bằng chứng thô theo mục).

Logic ở web/services/flower_health.py và flower_evidence.py. Cả hai đều chỉ đọc và chỉ nhận danh sách mục đã định sẵn,
không nhận đường dẫn hay lệnh từ client.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from web import state
from web.services import admin_auth, flower_evidence, flower_health

router = APIRouter(prefix="/api/flower", tags=["flower"])

PER_MINUTE = 30
EVIDENCE_PER_MINUTE = 20


def _build() -> Dict[str, Any]:
    return flower_health.build_health(
        snapshot=state.NETWORK_DEMO.snapshot(),
        port_open=lambda port: state.check_port_open("127.0.0.1", port),
    )


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

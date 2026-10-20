"""GET /api/flower/health: Flower có chạy thật không (chỉ đọc, công khai). Logic ở web/services/flower_health.py."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from web import state
from web.services import admin_auth, flower_health

router = APIRouter(prefix="/api/flower", tags=["flower"])

PER_MINUTE = 30


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

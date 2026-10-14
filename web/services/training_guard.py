"""Bảo vệ tài nguyên của các API huấn luyện trên máy chủ nhỏ (EC2 2 GiB): trần tham số, giới hạn thời gian, nhật ký thao tác.

Trần mặc định bảo thủ và đổi được bằng biến môi trường (xem `limits()`); /api/status trả các trần này để giao diện đặt
thuộc tính max của ô nhập, nên người dùng không phải gửi lên rồi mới biết bị từ chối.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request

from src import registry as reg
from web.services import admin_auth

AUDIT_MAX_BYTES = 5 * 1024 * 1024
_AUDIT_LOCK = threading.Lock()

# (tên trường trong /api/status, biến môi trường, mặc định, giới hạn trên cứng)
_LIMIT_SPECS = {
    "max_centralized_epochs": ("TRAIN_MAX_CENTRALIZED_EPOCHS", 10, 100),
    "max_rounds": ("TRAIN_MAX_ROUNDS", 10, 100),
    "max_local_epochs": ("TRAIN_MAX_LOCAL_EPOCHS", 3, 20),
    "max_seconds": ("TRAIN_MAX_SECONDS", 900, 7200),
    "starts_per_10_min": ("TRAIN_STARTS_PER_10_MIN", 6, 60),
}


def limits() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, (env, default, hard) in _LIMIT_SPECS.items():
        try:
            value = int(os.environ.get(env, default))
        except ValueError:
            value = default
        out[key] = max(1, min(value, hard))
    return out


def available_mb() -> Optional[int]:
    """RAM còn dùng được (MemAvailable, MB) trên Linux; None nếu không đọc được (Windows/macOS: không kiểm)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _min_available(kind: str) -> int:
    # Đo trên EC2 2 GiB (2026-10-14): cây tiến trình Flower (server + 5 client) chiếm ~1,6 GB; khi đó bản web không trả lời trong
    # 30 giây. Máy rảnh còn ~1,5 GB trống nên ngưỡng 1400 chỉ cho chạy khi máy hoàn toàn rảnh. Job trong tiến trình (Centralized,
    # simulation) đỉnh ~1,05 GB RSS nên chỉ cần còn ~500 MB.
    env, default = ("TRAIN_MIN_AVAILABLE_MB_NETWORK", 1400) if kind == "network" else ("TRAIN_MIN_AVAILABLE_MB", 500)
    try:
        return max(0, int(os.environ.get(env, default)))
    except ValueError:
        return default


def require_memory(kind: str) -> None:
    """Từ chối bắt đầu job khi RAM trống quá thấp: trên máy chủ 2 GB, job Flower 5 client + ứng dụng làm cả máy swap nặng và
    kéo cả trang xuống. Trả 503 kèm số liệu thật để người vận hành biết vì sao."""
    free = available_mb()
    need = _min_available(kind)
    if free is not None and free < need:
        raise HTTPException(
            status_code=503,
            detail=f"Máy chủ chưa đủ RAM trống để chạy phiên này (còn {free} MB, cần tối thiểu {need} MB). "
                   "Hãy thử lại sau ít phút hoặc đợi tác vụ khác của máy chủ kết thúc.")


def status_block() -> Dict[str, Any]:
    return {"requires_admin": admin_auth.auth_required(), "limits": limits()}


def actor(request: Optional[Request]) -> Dict[str, str]:
    if request is None:
        return {"user": "local", "ip": "local"}
    return {"user": getattr(request.state, "admin_user", "local"), "ip": admin_auth.client_ip(request)}


def audit(event: str, request: Optional[Request] = None, **fields: Any) -> None:
    """Một dòng JSON mỗi thao tác: ai (tài khoản + IP), lúc nào, cấu hình và kết quả. Lỗi ghi nhật ký không làm hỏng thao tác."""
    record = {"ts": round(time.time(), 3), "event": event, **actor(request), **fields}
    try:
        path = reg.abs_path(reg.TRAINING_AUDIT_JSONL)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK:
            if path.exists() and path.stat().st_size > AUDIT_MAX_BYTES:
                path.replace(path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def arm_watchdog(job_id: str, expire: Callable[[str], None]) -> threading.Timer:
    """Hẹn giờ dừng job quá `max_seconds`. Gọi `.cancel()` khi job kết thúc; `expire(job_id)` tự kiểm job còn là job đó."""
    timer = threading.Timer(limits()["max_seconds"], expire, args=(job_id,))
    timer.daemon = True
    timer.start()
    return timer

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

from fastapi import Request

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

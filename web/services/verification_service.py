"""Dữ liệu cho tab "Phân tán và tập trung" (/api/evidence/verification). Số do src/federated/verification_view.py tính, cùng hàm với scripts/demo_show.py.

Mỗi phần hỏng riêng: thiếu tệp ở một phần thì phần đó báo lý do, các phần khác vẫn hiện.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict

from src import registry as reg
from src.federated.verification_view import drift_view, fairness_view, speed_view, sweep_view, weight_view

logger = logging.getLogger(__name__)
CACHE_S = 30.0
_cache: Dict[str, Any] = {"at": -CACHE_S, "value": None}


def _guard(name: str, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return build()
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("Không đọc được phần %s của trang kiểm chứng", name, exc_info=True)
        return {"available": False, "reason": f"Chưa đọc được số liệu ({type(error).__name__})."}


def build() -> Dict[str, Any]:
    now = time.monotonic()
    if _cache["value"] is not None and now - _cache["at"] < CACHE_S:
        return _cache["value"]
    value = {
        "sweep": _guard("sweep", lambda: sweep_view(reg.abs_path(reg.SWEEP_V4_DIR) / "table_main.csv")),
        "fairness": _guard("fairness", lambda: fairness_view(reg.abs_path(reg.SWEEP_V4_DIR) / "table_fairness.csv")),
        "drift": _guard("drift", lambda: drift_view(reg.abs_path(reg.SWEEP_V4_DIR) / "client_drift.csv")),
        "speed": _guard("speed", lambda: speed_view(reg.abs_path(reg.NETWORK_SPEEDTEST_DIR))),
        "weights": _guard("weights", lambda: weight_view(reg.abs_path(reg.DISTRIBUTED_VERIFICATION_DIR), reg.abs_path(reg.FEDSGD_EQUIVALENCE_JSON))),
    }
    _cache.update(at=now, value=value)
    return value

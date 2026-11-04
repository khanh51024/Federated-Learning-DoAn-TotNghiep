"""Nút "Bắt đầu lần đo" trên trang #flower: mở máy chủ đo qua mạng ngay trên máy chủ web, bằng đúng script chạy tay.

An toàn: chỉ router kiểm quyền quản trị mới gọi được; lệnh là danh sách tham số cố định, không qua shell, số client và số vòng đã kiểm là số nguyên;
tiến trình con không nhận biến môi trường của web nên không thấy mật khẩu quản trị. Cổng đo tự đóng khi xong, khi lỗi, khi bấm dừng, hoặc sau MAX_HOURS giờ
(xem scripts/remote/run_network_server.sh). Mỗi lúc chỉ chạy một lần đo.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Optional

from src import registry as reg
from src.federated.run_view import PROGRESS_FILE, list_runs

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "remote" / "run_network_server.sh"
MAX_HOURS = 1
EARLY_EXIT_WAIT_S = 2.0     # script báo lỗi tham số, cổng bận, thiếu dữ liệu ngay từ đầu: đợi chừng này để báo lại trên trang
STOP_WAIT_S = 20.0
STOPPED_MESSAGE = "Đã dừng bằng nút trên trang."


class DemoError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


_lock = threading.Lock()
_process: Optional[subprocess.Popen[bytes]] = None
_output: Optional[IO[bytes]] = None


def _data_dir() -> Path:
    return Path.home() / "net10-data"


def _blocked_reason() -> Optional[str]:
    if os.name != "posix":
        return "Chỉ chạy được trên máy chủ Linux."
    if shutil.which("bash") is None or not SCRIPT.is_file():
        return "Máy chủ này không có script đo."
    if not (_data_dir() / "partition").is_dir():
        return "Máy chủ này chưa có dữ liệu đo."
    return None


def _running() -> bool:
    return _process is not None and _process.poll() is None


def status() -> dict[str, Any]:
    with _lock:
        running = _running()
    return {"can_start": _blocked_reason() is None, "running": running}


def _child_env() -> dict[str, str]:
    """Chỉ những gì script cần. Môi trường của web có mật khẩu quản trị nên không truyền xuống."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path.home()),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONUNBUFFERED": "1",
        "RELEASE": str(ROOT),
    }


def _last_line(stream: IO[bytes]) -> str:
    stream.seek(0)
    lines = [line.strip() for line in stream.read().decode("utf-8", "replace").splitlines() if line.strip()]
    return lines[-1][:300] if lines else "Script dừng mà không báo gì."


def start(clients: int, rounds: int) -> dict[str, Any]:
    global _process, _output
    reason = _blocked_reason()
    if reason is not None:
        raise DemoError(503, reason)
    with _lock:
        if _running():
            raise DemoError(409, "Đang có một lần đo chạy từ trang này.")
        if _output is not None:
            _output.close()
        _output = tempfile.TemporaryFile()
        argv = ["bash", str(SCRIPT), "--public", "--clients", str(int(clients)), "--rounds", str(int(rounds)), "--max-hours", str(MAX_HOURS)]
        _process = subprocess.Popen(argv, cwd=str(ROOT), env=_child_env(), stdin=subprocess.DEVNULL, stdout=_output, stderr=subprocess.STDOUT, start_new_session=True)
        logger.info("Bắt đầu lần đo từ trang web: %d client, %d vòng, pid %d", clients, rounds, _process.pid)
        deadline = time.monotonic() + EARLY_EXIT_WAIT_S
        while time.monotonic() < deadline:
            if _process.poll() is not None:
                break
            time.sleep(0.1)
        if _process.poll() is not None:
            raise DemoError(409, _last_line(_output))
    return {"started": True, "clients": clients, "rounds": rounds, "closes_after_hours": MAX_HOURS}


def _mark_newest_run_stopped() -> None:
    """Máy chủ bị dừng không kịp ghi gì. Ghi vào lần đo mới nhất để trang báo đã dừng ngay, không phải đợi 90 giây mới báo mất liên lạc."""
    runs = list_runs(reg.abs_path(reg.NETWORK_LIVE_DIR))
    if not runs:
        return
    path = runs[0] / PROGRESS_FILE
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if progress.get("status") not in ("waiting", "running"):
        return
    progress.update(status="failed", error=STOPPED_MESSAGE, updated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    temporary = path.with_suffix(".stop.tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def stop() -> dict[str, Any]:
    with _lock:
        if not _running():
            raise DemoError(409, "Không có lần đo nào chạy từ trang này.")
        process = _process
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(STOP_WAIT_S)
        except subprocess.TimeoutExpired as error:
            # Không dùng SIGKILL: script cần chạy nốt phần đóng cổng tường lửa.
            raise DemoError(500, "Lần đo chưa dừng sau 20 giây. Thử lại, hoặc xem trên máy chủ.") from error
        _mark_newest_run_stopped()
    logger.info("Đã dừng lần đo từ trang web")
    return {"stopped": True}

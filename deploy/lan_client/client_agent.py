"""Chạy nền trên máy đã cài bộ client: báo danh với web, và khi quản trị bắt đầu lần đo thì tự mở đúng các client được giao.

Chỉ dùng thư viện chuẩn để lúc rảnh tốn rất ít RAM; torch và flwr chỉ nạp trong cửa sổ client.
Web chỉ giao số client đầu và số lượng. Máy chủ đo và thư mục ảnh đã ghi lúc cài, nên web không làm máy này nối sang chỗ khác hay chạy lệnh khác được.
Dừng bằng cách tạo tệp agent.stop trong thư mục trạng thái (GO_NUT_WEB.bat làm việc này).
Việc ghi khóa tự chạy và dừng bản cũ cũng nằm ở đây thay vì trong file .bat: tên thư mục có &, ) hay dấu cách không làm sai dòng lệnh nào.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import msvcrt
except ImportError:     # chỉ máy Windows mới chạy bản thật; test trên hệ khác vẫn nạp được tệp này
    msvcrt = None

KEY = re.compile(r"^[0-9a-f]{32}$")
JOB_ID = re.compile(r"^[0-9a-f]{16}$")
SAFE_WEB = re.compile(r"^https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?$")
SAFE_ARG = re.compile(r"^[A-Za-z0-9.:_\\-]+$")     # máy chủ và thư mục ảnh được ghép vào dòng lệnh cmd
MAX_CLIENTS = 10
POLL_S = 5
RETRY_MAX_S = 30
HTTP_TIMEOUT_S = 5
STOP_CHECK_S = 0.5
STOP_WAIT_S = 30        # phải dài hơn lúc tệ nhất: đang chờ mạng, rồi đang dừng các cửa sổ client
REPLY_MAX_BYTES = 1 << 16
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "CropFedClient"
FLOWER_PORT = 18089
SERVER_WAIT_S = 180     # máy chủ đo cần một lúc để nạp mô hình và mở cổng
LOG_MAX_BYTES = 1 << 20
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def default_state_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "CropFedClient"


def app_root() -> Path:
    """Thư mục có src: ngay cạnh tệp này khi chạy từ gói, hai cấp trên khi chạy từ deploy/lan_client của dự án."""
    here = Path(__file__).resolve().parent
    return here if (here / "src").is_dir() else here.parents[1]


def log(state_dir: Path, text: str) -> None:
    path = state_dir / "agent.log"
    if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
        path.unlink()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def load_key(state_dir: Path) -> str:
    """Khóa riêng của máy, tạo một lần. Giữ nguyên qua các lần cài lại nên mã của máy trên trang không đổi."""
    path = state_dir / "agent_key.txt"
    if path.exists():
        key = path.read_text(encoding="ascii").strip()
        if KEY.fullmatch(key):
            return key
    key = secrets.token_hex(16)
    path.write_text(key, encoding="ascii")
    return key


def code_of(key: str) -> str:
    return hashlib.sha256(key.encode("ascii")).hexdigest()[:6]


def ram_gb() -> tuple[Optional[float], Optional[float]]:
    """(tổng, còn trống). Quản trị viên nhìn số này để biết máy có chịu nổi không; mỗi client tốn khoảng 0,6–0,9 GB."""
    if os.name != "nt":
        return None, None

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return round(status.total / 1024 ** 3, 1), round(status.free / 1024 ** 3, 1)


def parse_job(raw: Any) -> Optional[dict[str, Any]]:
    """Phần việc do web gửi. Sai dạng thì bỏ qua, không chạy gì."""
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not JOB_ID.fullmatch(raw["id"]):
        return None
    first, count = raw.get("first"), raw.get("count")
    if type(first) is not int or type(count) is not int or first < 0 or count < 1 or first + count > MAX_CLIENTS:
        return None
    return {"id": raw["id"], "first": first, "count": count}


def port_open(host: str, port: int = FLOWER_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def launch_clients(root: Path, state_dir: Path, first: int, count: int, server: str, data: str) -> subprocess.Popen:
    """Dùng đúng CHAY_CLIENT.bat như khi bấm tay. Nó mở mỗi client một cửa sổ, đợi hết rồi trả mã thoát."""
    bat = Path(__file__).resolve().parent / "CHAY_CLIENT.bat"
    with (state_dir / "client_launch.log").open("wb") as output:
        return subprocess.Popen(f'cmd /s /c ""{bat}" {first} {count} {server} {data}"', cwd=str(root), env={**os.environ, "DEMO_NOPAUSE": "1"},
                                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)


def kill_tree(process: subprocess.Popen) -> None:
    """Giết cả các cửa sổ client. Máy chủ bị dừng thì client Flower tự thử nối lại mãi nên không tự thoát."""
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, check=False)
    process.wait(timeout=10)


class Agent:
    def __init__(self, web: str, server: str, data: str, state_dir: Path, root: Path, *,
                 post: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
                 launch: Callable[..., subprocess.Popen] = launch_clients, kill: Callable[[subprocess.Popen], None] = kill_tree,
                 server_ready: Callable[[str], bool] = port_open, clock: Callable[[], float] = time.monotonic) -> None:
        self.web, self.server, self.data, self.state_dir, self.root = web, server, data, state_dir, root
        self.key = load_key(state_dir)
        self.post = post or self._post
        self.launch, self.kill, self.server_ready, self.clock = launch, kill, server_ready, clock
        self.job: Optional[dict[str, Any]] = None
        self.process: Optional[subprocess.Popen] = None
        self.deadline = 0.0
        self.result: Optional[str] = None
        self.result_job: Optional[str] = None
        self.last_job_id: Optional[str] = None

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.web}/api/flower/agents/heartbeat", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "User-Agent": "cropfed-client-agent"}, method="POST")
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as reply:
            return json.loads(reply.read(REPLY_MAX_BYTES).decode("utf-8"))

    def _note(self, text: str) -> None:
        log(self.state_dir, text)

    def _has_images(self, first: int, count: int) -> bool:
        data = self.root / self.data
        return (data / "partition").is_dir() and all((data / f"images_client_{client:02d}").is_dir() for client in range(first, first + count))

    def _finish(self, result: str) -> None:
        self._note(f"phần việc {self.job['id']} kết thúc: {result}")
        self.result, self.result_job = result, self.job["id"]
        self.job, self.process = None, None

    def _advance(self) -> None:
        """Đưa phần việc đang có đi tiếp một bước: chờ máy chủ mở cổng, mở client, hoặc ghi nhận đã xong."""
        if self.job is None:
            return
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self._finish("ok" if code == 0 else "failed")
            return
        if self.server_ready(self.server):
            self.process = self.launch(self.root, self.state_dir, self.job["first"], self.job["count"], self.server, self.data)
            self._note(f"đã mở client {self.job['first']} đến {self.job['first'] + self.job['count'] - 1}")
        elif self.clock() > self.deadline:
            self._note(f"sau {SERVER_WAIT_S} giây vẫn không nối được {self.server}:{FLOWER_PORT}")
            self._finish("failed")

    def _accept(self, raw: Any) -> None:
        job = parse_job(raw)
        if raw is not None and job is None:
            self._note("bỏ qua phần việc sai dạng")
        if job is None or self.job is not None or job["id"] == self.last_job_id:
            return
        self.last_job_id = job["id"]
        self.job, self.deadline = job, self.clock() + SERVER_WAIT_S
        self.result, self.result_job = None, None
        if not self._has_images(job["first"], job["count"]):
            self._note(f"không thấy ảnh của client {job['first']} đến {job['first'] + job['count'] - 1} trong {self.data}")
            self._finish("failed")
            return
        self._note(f"nhận phần việc {job['id']}: client {job['first']} đến {job['first'] + job['count'] - 1}")

    def stop_job(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.kill(self.process)
        if self.job is not None:
            self._note(f"đã dừng phần việc {self.job['id']}")
        self.job, self.process = None, None

    def tick(self) -> dict[str, Any]:
        """Một nhịp: tiến phần việc, báo danh, làm theo trả lời. Lỗi mạng để hàm run xử lý."""
        self._advance()
        total, free = ram_gb()
        reply = self.post({"key": self.key, "cores": os.cpu_count(), "ram_gb": total, "ram_free_gb": free,
                           "running": self.job["id"] if self.job else None, "result": self.result, "result_job": self.result_job})
        if not isinstance(reply, dict):
            raise ValueError("web trả về không phải JSON object")
        if reply.get("stop"):
            self.stop_job()
        self._accept(reply.get("job"))
        return reply

    def run(self) -> None:
        stop_file = self.state_dir / "agent.stop"
        delay: float = POLL_S
        online = None
        while not stop_file.exists():
            try:
                reply = self.tick()
                delay = POLL_S if self.process is not None or not isinstance(reply.get("poll_s"), int) else max(1, min(reply["poll_s"], RETRY_MAX_S))
                if self.job is not None and self.process is None:
                    delay = 2        # đang chờ cổng máy chủ: báo nhận ngay và mở client sớm nhất có thể
                if online is not True:
                    self._note(f"đã nối {self.web}, mã máy {code_of(self.key)}")
                online = True
            except Exception as error:      # mất mạng, web đang khởi động lại, 429, trả lời đứt giữa chừng. Chạy nền không ai thấy lỗi: ghi nhật ký rồi thử lại, không được chết
                if online is not False:
                    self._note(f"chưa nối được {self.web}: {error}")
                online = False
                delay = min(delay * 2, RETRY_MAX_S)
            wake = self.clock() + delay
            while self.clock() < wake and not stop_file.exists():
                time.sleep(STOP_CHECK_S)
        self.stop_job()
        stop_file.unlink(missing_ok=True)
        self._note("đã dừng theo agent.stop")


def single_instance(state_dir: Path):
    """Giữ khóa suốt đời tiến trình. Trả None nếu máy này đã có một agent đang chạy."""
    handle = (state_dir / "agent.lock").open("a+b")
    if msvcrt is None:
        return handle
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def stop_running(state_dir: Path, wait_s: float = STOP_WAIT_S) -> bool:
    """Dừng agent đang chạy và đợi nó nhả hẳn khóa, nhờ vậy xóa được thư mục cài và agent mới không tự thoát vì tưởng đã có agent."""
    stop_file = state_dir / "agent.stop"
    deadline = time.monotonic() + wait_s
    while True:
        lock = single_instance(state_dir)
        if lock is not None:
            lock.close()
            stop_file.unlink(missing_ok=True)
            return True
        if time.monotonic() > deadline:
            return False
        stop_file.touch()
        time.sleep(STOP_CHECK_S)


def background_command(web: str, server: str, data: str, state_dir: Path) -> list[str]:
    python = Path(sys.executable).with_name("pythonw.exe")
    return [str(python), str(Path(__file__).resolve()), "--web", web, "--server", server, "--data", data, "--state-dir", str(state_dir)]


def register_autostart(command: list[str], subkey: str = RUN_KEY) -> None:
    import winreg
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, subkey, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, subprocess.list2cmdline(command))


def install(web: str, server: str, data: str, state_dir: Path) -> int:
    """Dừng bản cũ, ghi khóa tự chạy khi đăng nhập, chạy luôn, in mã của máy."""
    if not stop_running(state_dir):
        print(f"[LỖI] Không dừng được bản đang chạy sau {STOP_WAIT_S} giây. Khởi động lại máy rồi chạy lại file này.")
        return 1
    key = load_key(state_dir)       # tạo khóa trước khi chạy nền để hai bên không cùng tạo
    command = background_command(web, server, data, state_dir)
    register_autostart(command)
    subprocess.Popen(command, cwd=str(state_dir), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Python trong gói bỏ qua PYTHONUTF8; file .bat đã chcp 65001
    print(f"Máy này đã nối với {web}, mã trên trang: {code_of(key)}.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", help="địa chỉ trang, ví dụ https://datn.tranngochung.id.vn")
    parser.add_argument("--server", help="máy chủ đo")
    parser.add_argument("--data", help="thư mục ảnh nhẹ, tính từ thư mục có src")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--install", action="store_true", help="ghi khóa tự chạy khi đăng nhập, chạy nền, in mã của máy")
    parser.add_argument("--stop", action="store_true", help="dừng agent đang chạy và đợi nó thoát hẳn")
    args = parser.parse_args(argv)
    state_dir = Path(args.state_dir) if args.state_dir else default_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    if args.stop:
        return 0 if stop_running(state_dir) else 1
    if not args.web or not SAFE_WEB.fullmatch(args.web) or not args.server or not SAFE_ARG.fullmatch(args.server) or not args.data or not SAFE_ARG.fullmatch(args.data):
        log(state_dir, f"tham số không hợp lệ: {sys.argv[1:]}")
        return 2
    if args.install:
        return install(args.web, args.server, args.data, state_dir)
    lock = single_instance(state_dir)
    if lock is None:
        log(state_dir, "đã có agent khác đang chạy, thoát")
        return 0
    try:
        Agent(args.web, args.server, args.data, state_dir, app_root()).run()
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

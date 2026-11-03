"""Small TCP probe used only for per-client network measurements.

This is deliberately separate from Flower/gRPC so the metric is not confused
with model-transfer or training time.  The server listens on a dedicated port
(typically Flower port + 1).

Two protocols share the port, chosen by the first 4 bytes of a connection:

* ``PING`` -> ``PONG`` then close.  One-shot RTT that INCLUDES the TCP handshake
  (about two round trips); kept for backward compatibility (`measure_probe_rtt_ms`).
* ``PNG2`` / ``UPLD`` / ``DNLD`` / ``QUIT`` on ONE persistent connection
  (`measure_link`): handshake time, round-trip time without the handshake, and
  payload throughput in each direction, measured entirely on the client's own clock
  so two machines do not need synchronized clocks.
"""

from __future__ import annotations

import hashlib
import secrets
import socket
import struct
import threading
import time
from contextlib import closing
from typing import Any, Dict, List, Optional

_CHUNK = 64 * 1024
_FILLER = b"\xa5" * _CHUNK


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = size
    while remaining > 0:
        block = conn.recv(min(remaining, _CHUNK))
        if not block:
            raise ConnectionError("probe connection closed early")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _drain(conn: socket.socket, size: int) -> None:
    remaining = size
    while remaining > 0:
        block = conn.recv(min(remaining, _CHUNK))
        if not block:
            raise ConnectionError("probe connection closed early")
        remaining -= len(block)


def _send_filler(conn: socket.socket, size: int) -> None:
    remaining = size
    while remaining > 0:
        take = min(remaining, _CHUNK)
        conn.sendall(_FILLER[:take])
        remaining -= take


class ProbeServer:
    # Upper bound for one upload/download test so a stray client cannot make the server stream forever.
    MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
    MAX_CONNECTIONS = 16  # kết nối đồng thời tối đa; kết nối thứ 17 bị đóng ngay
    MAX_COMMANDS_PER_CONNECTION = 64
    MAX_BYTES_PER_CONNECTION = 256 * 1024 * 1024  # tổng byte nhận + gửi trong một kết nối
    CONNECTION_DEADLINE_S = 120.0  # tuổi thọ tối đa của một kết nối (chống kết nối nhỏ giọt giữ luồng mãi)

    MAX_STORED_MEASUREMENTS = 4096

    def __init__(self, host: str, port: int, allow_payload: bool = True, measure_payload_bytes: int = 0):
        self.host = host
        self.port = int(port)
        # allow_payload=False: chỉ trả lời PING/PNG2 (RTT); UPLD/DNLD (khối byte lớn) bị từ chối. Máy chủ Flower chỉ bật khi
        # người vận hành yêu cầu đo băng thông (--probe-payload-bytes >= 0).
        self.allow_payload = bool(allow_payload)
        # Kích thước khối byte do SERVER chọn cho phép đo SMEA (client không được đề xuất); 0 = chỉ đo RTT.
        if not 0 <= int(measure_payload_bytes) <= self.MAX_PAYLOAD_BYTES:
            raise ValueError(f"measure_payload_bytes phải nằm trong [0, {self.MAX_PAYLOAD_BYTES}]")
        self.measure_payload_bytes = int(measure_payload_bytes)
        self._measurements: Dict[tuple, Dict[str, Any]] = {}
        # Muối riêng từng lần chạy: mã băm IP chỉ để biết hai client có cùng nguồn không; không dò ngược được ra IP (IPv4 đếm hết được nếu không có muối).
        self._peer_salt = secrets.token_bytes(16)
        self._measurements_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.MAX_CONNECTIONS)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._serve, name=f"probe-{self.port}", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self._sock = s
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(64)
            s.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    conn, peer = s.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    time.sleep(0.05)  # hết file descriptor tạm thời: thử lại, không để bộ lắng nghe chết
                    continue
                if not self._slots.acquire(blocking=False):  # quá số kết nối đồng thời
                    conn.close()
                    continue
                try:
                    threading.Thread(target=self._handle, args=(conn, peer), daemon=True).start()
                except Exception:  # noqa: BLE001 — không tạo được luồng: nhả chỗ và đóng kết nối
                    self._slots.release()
                    conn.close()

    def _handle(self, conn: socket.socket, peer: Any = None) -> None:
        try:
            with conn:
                self._serve_connection(conn, peer)
        finally:
            self._slots.release()

    def _peer_facts(self, conn: socket.socket, peer: Any) -> Dict[str, Any]:
        """Client này có nằm trên chính máy chủ không, và nguồn của nó (mã băm có muối, KHÔNG lưu IP để trang công khai không lộ IP)."""
        peer_ip = str(peer[0]) if peer else ""
        if peer_ip.startswith("::ffff:"):
            peer_ip = peer_ip[len("::ffff:"):]
        local_ip = str(conn.getsockname()[0])
        same_host = peer_ip.startswith("127.") or peer_ip == "::1" or peer_ip == local_ip
        return {"peer_hash": hashlib.sha256(self._peer_salt + peer_ip.encode()).hexdigest()[:8], "same_host_as_server": same_host}

    def _serve_connection(self, conn: socket.socket, peer: Any = None) -> None:
        deadline = time.monotonic() + self.CONNECTION_DEADLINE_S
        moved = 0
        conn.settimeout(10.0)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            command = _recv_exact(conn, 4)
            if command == b"PING":  # legacy one-shot exchange
                conn.sendall(b"PONG")
                return
            if command == b"SMEA":  # server-driven measurement: the server times everything on its own clock
                self._run_measurement(conn, peer)
                return
            for _ in range(self.MAX_COMMANDS_PER_CONNECTION):
                if time.monotonic() > deadline or moved > self.MAX_BYTES_PER_CONNECTION:
                    return
                if command == b"PNG2":
                    conn.sendall(b"PONG")
                elif command in (b"UPLD", b"DNLD") and self.allow_payload:
                    (size,) = struct.unpack("!Q", _recv_exact(conn, 8))
                    if size > self.MAX_PAYLOAD_BYTES:
                        return
                    moved += size
                    if command == b"UPLD":
                        _drain(conn, size)
                        conn.sendall(b"OKAY")
                    else:
                        _send_filler(conn, size)
                else:  # QUIT, lệnh lạ, hoặc UPLD/DNLD khi phép đo tải đang tắt
                    return
                command = _recv_exact(conn, 4)
        except (OSError, ConnectionError, struct.error):
            pass

    def _run_measurement(self, conn: socket.socket, peer: Any = None) -> None:
        """Phép đo do server điều khiển: server gửi lệnh, client chỉ làm theo và KHÔNG gửi con số nào.

        Mọi thời gian đo bằng đồng hồ của server. Client chỉ khai (số vòng, số thứ tự client) để server biết ghi kết quả vào đâu;
        server đối chiếu hai số này với metadata của lượt fit rồi mới dùng.
        """
        client_id, round_num = struct.unpack("!II", _recv_exact(conn, 8))
        started = time.perf_counter()
        rtts: List[float] = []
        for _ in range(7):
            t = time.perf_counter()
            conn.sendall(b"SPNG")
            if _recv_exact(conn, 4) != b"SPNG":
                return
            rtts.append((time.perf_counter() - t) * 1000.0)
        ordered = sorted(rtts)
        p50 = _percentile(ordered, 0.5)
        result: Dict[str, Any] = {
            "client_id": int(client_id), "round": int(round_num),
            "rtt_ms_min": ordered[0], "rtt_ms_p50": p50, "rtt_ms_max": ordered[-1],
            "rtt_samples_ms": [round(v, 3) for v in rtts], "payload_bytes": 0,
            **self._peer_facts(conn, peer),
        }
        size = self.measure_payload_bytes if self.allow_payload else 0
        if size > 0:
            # Lên: tính từ lúc server gửi lệnh đến lúc nhận byte cuối (có nửa RTT của lệnh). Xuống: từ lúc gửi đến lúc nhận OKAY (có nửa RTT của OKAY).
            t = time.perf_counter()
            conn.sendall(b"SUPL" + struct.pack("!Q", size))
            _drain(conn, size)
            up_s = time.perf_counter() - t
            t = time.perf_counter()
            conn.sendall(b"SDNL" + struct.pack("!Q", size))
            _send_filler(conn, size)
            if _recv_exact(conn, 4) != b"OKAY":
                return
            down_s = time.perf_counter() - t
            half_rtt_s = p50 / 2000.0
            mb = size / (1024 * 1024)
            result.update({
                "payload_bytes": int(size), "up_s": up_s, "down_s": down_s,
                "up_mb_per_s": mb / max(up_s, 1e-9), "down_mb_per_s": mb / max(down_s, 1e-9),
                "up_mb_per_s_net": mb / max(up_s - half_rtt_s, 1e-9), "down_mb_per_s_net": mb / max(down_s - half_rtt_s, 1e-9),
            })
        conn.sendall(b"DONE")
        result["probe_total_ms"] = (time.perf_counter() - started) * 1000.0
        with self._measurements_lock:
            self._measurements[(int(round_num), int(client_id))] = result
            while len(self._measurements) > self.MAX_STORED_MEASUREMENTS:
                self._measurements.pop(next(iter(self._measurements)))

    def take_measurement(self, round_num: int, client_id: int) -> Optional[Dict[str, Any]]:
        """Kết quả phép đo SMEA của (vòng, client), hoặc None nếu client không đo (hoặc đo hỏng)."""
        with self._measurements_lock:
            return self._measurements.pop((int(round_num), int(client_id)), None)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def measure_probe_rtt_ms(host: str, port: int, timeout_s: float = 2.0) -> float:
    """Measure TCP application RTT for one PING/PONG exchange (includes the TCP handshake)."""
    t0 = time.perf_counter()
    with closing(socket.create_connection((host, int(port)), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(b"PING")
        data = s.recv(16)
        if data != b"PONG":
            raise RuntimeError(f"Unexpected probe response: {data!r}")
    return (time.perf_counter() - t0) * 1000.0


def obey_server_measurement(host: str, port: int, client_id: int, round_num: int, timeout_s: float = 30.0) -> bool:
    """Phía client của phép đo SMEA: làm theo lệnh của server (trả lời ping, gửi/nhận khối byte) mà KHÔNG đo và KHÔNG gửi con số nào.

    Trả True nếu server báo DONE. Khối byte tối đa theo ProbeServer.MAX_PAYLOAD_BYTES nên server lạ không bắt được client gửi vô hạn.
    """
    with closing(socket.create_connection((host, int(port)), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(b"SMEA" + struct.pack("!II", int(client_id), int(round_num)))
        for _ in range(ProbeServer.MAX_COMMANDS_PER_CONNECTION):
            command = _recv_exact(s, 4)
            if command == b"DONE":
                return True
            if command == b"SPNG":
                s.sendall(b"SPNG")
            elif command in (b"SUPL", b"SDNL"):
                (size,) = struct.unpack("!Q", _recv_exact(s, 8))
                if size > ProbeServer.MAX_PAYLOAD_BYTES:
                    raise ValueError(f"server đòi khối {size} byte, vượt trần {ProbeServer.MAX_PAYLOAD_BYTES}")
                if command == b"SUPL":
                    _send_filler(s, size)
                else:
                    _drain(s, size)
                    s.sendall(b"OKAY")
            else:
                raise RuntimeError(f"lệnh lạ từ server: {command!r}")
    return False


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def measure_link(
    host: str,
    port: int,
    payload_bytes: int = 0,
    pings: int = 7,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Measure one link on a persistent connection (client clock only).

    Returns connect_ms (TCP handshake), rtt_ms_{min,p50,max} (round trips WITHOUT handshake) and, when
    ``payload_bytes`` > 0, upload/download time and MB/s.  The upload time ends when the server acknowledges the
    last byte, so it includes one round trip; ``*_mbps_net`` subtracts the median RTT for a cleaner throughput.
    """
    out: Dict[str, Any] = {"payload_bytes": int(payload_bytes)}
    t0 = time.perf_counter()
    with closing(socket.create_connection((host, int(port)), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        out["connect_ms"] = (time.perf_counter() - t0) * 1000.0
        rtts: List[float] = []
        for _ in range(max(1, int(pings))):
            t = time.perf_counter()
            s.sendall(b"PNG2")
            if _recv_exact(s, 4) != b"PONG":
                raise RuntimeError("Unexpected probe response")
            rtts.append((time.perf_counter() - t) * 1000.0)
        ordered = sorted(rtts)
        out["rtt_ms_min"] = ordered[0]
        out["rtt_ms_p50"] = _percentile(ordered, 0.5)
        out["rtt_ms_max"] = ordered[-1]
        out["rtt_samples_ms"] = [round(v, 3) for v in rtts]
        if payload_bytes > 0:
            size = int(payload_bytes)
            t = time.perf_counter()
            s.sendall(b"UPLD" + struct.pack("!Q", size))
            _send_filler(s, size)
            if _recv_exact(s, 4) != b"OKAY":
                raise RuntimeError("Unexpected probe response")
            up_s = time.perf_counter() - t
            t = time.perf_counter()
            s.sendall(b"DNLD" + struct.pack("!Q", size))
            _drain(s, size)
            down_s = time.perf_counter() - t
            rtt_s = out["rtt_ms_p50"] / 1000.0
            out.update(
                {
                    "up_s": up_s,
                    "down_s": down_s,
                    "up_mb_per_s": size / (1024 * 1024) / max(up_s, 1e-9),
                    "down_mb_per_s": size / (1024 * 1024) / max(down_s, 1e-9),
                    "up_mb_per_s_net": size / (1024 * 1024) / max(up_s - rtt_s, 1e-9),
                    "down_mb_per_s_net": size / (1024 * 1024) / max(down_s - rtt_s, 1e-9),
                }
            )
        try:
            s.sendall(b"QUIT")
        except OSError:
            pass
    return out

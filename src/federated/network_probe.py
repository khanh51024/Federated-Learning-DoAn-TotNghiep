"""Small TCP echo probe used only for per-client RTT measurement.

This is deliberately separate from Flower/gRPC so the metric is not confused
with model-transfer or training time.  The server listens on a dedicated port
(typically Flower port + 1), and clients measure PING->PONG round-trip time.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import closing
from typing import Optional


class ProbeServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
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
                    conn, _ = s.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    conn.settimeout(2.0)
                    try:
                        data = conn.recv(16)
                        if data == b"PING":
                            conn.sendall(b"PONG")
                    except OSError:
                        pass

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
    """Measure TCP application RTT for one PING/PONG exchange."""
    t0 = time.perf_counter()
    with closing(socket.create_connection((host, int(port)), timeout=timeout_s)) as s:
        s.settimeout(timeout_s)
        s.sendall(b"PING")
        data = s.recv(16)
        if data != b"PONG":
            raise RuntimeError(f"Unexpected probe response: {data!r}")
    return (time.perf_counter() - t0) * 1000.0

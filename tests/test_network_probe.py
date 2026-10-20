"""Probe mạng: RTT một lần (cũ) và đo tuyến trên một kết nối giữ sẵn (mới), chạy trên loopback."""
from __future__ import annotations

import socket
import time

import pytest

from src.federated.network_probe import ProbeServer, measure_link, measure_probe_rtt_ms


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def probe():
    port = _free_port()
    server = ProbeServer("127.0.0.1", port)
    server.start()
    time.sleep(0.3)
    yield port
    server.stop()


def test_legacy_one_shot_rtt_still_works(probe):
    assert measure_probe_rtt_ms("127.0.0.1", probe) > 0


def test_measure_link_without_payload_reports_connect_and_rtt(probe):
    out = measure_link("127.0.0.1", probe, payload_bytes=0, pings=5)
    assert out["connect_ms"] >= 0 and 0 < out["rtt_ms_min"] <= out["rtt_ms_p50"] <= out["rtt_ms_max"]
    assert len(out["rtt_samples_ms"]) == 5 and "up_s" not in out


def test_measure_link_with_payload_reports_both_directions(probe):
    size = 2 * 1024 * 1024
    out = measure_link("127.0.0.1", probe, payload_bytes=size, pings=3)
    assert out["payload_bytes"] == size
    assert out["up_s"] > 0 and out["down_s"] > 0
    assert out["up_mb_per_s"] > 0 and out["down_mb_per_s"] > 0
    assert out["up_mb_per_s_net"] >= out["up_mb_per_s"] * 0.99  # trừ RTT chỉ làm tốc độ tăng


def test_payload_tests_are_refused_when_disabled():
    port = _free_port()
    server = ProbeServer("127.0.0.1", port, allow_payload=False)
    server.start()
    time.sleep(0.3)
    try:
        assert measure_probe_rtt_ms("127.0.0.1", port) > 0  # RTT vẫn dùng được
        with pytest.raises((ConnectionError, OSError)):
            measure_link("127.0.0.1", port, payload_bytes=1024, pings=1)  # UPLD bị đóng kết nối
    finally:
        server.stop()


def test_connection_cap_closes_the_extra_connection(probe):
    held = []
    try:
        for _ in range(ProbeServer.MAX_CONNECTIONS):
            s = socket.create_connection(("127.0.0.1", probe), timeout=5)
            s.sendall(b"PNG2")
            assert s.recv(4) == b"PONG"  # kết nối đang được phục vụ, giữ mở
            held.append(s)
        extra = socket.create_connection(("127.0.0.1", probe), timeout=5)
        extra.settimeout(5)
        try:
            extra.sendall(b"PNG2")
            closed = extra.recv(4) == b""  # kết nối thứ 17 bị đóng ngay, không được phục vụ
        except ConnectionError:
            closed = True  # Windows báo đóng bằng RST thay vì EOF
        assert closed
        extra.close()
    finally:
        for s in held:
            s.close()
    time.sleep(0.3)
    assert measure_probe_rtt_ms("127.0.0.1", probe) > 0  # chỗ đã được nhả sau khi các kết nối đóng


def test_server_survives_back_to_back_clients_and_oversize_request(probe):
    for _ in range(3):
        assert measure_link("127.0.0.1", probe, payload_bytes=1024, pings=1)["up_s"] > 0
    import struct

    with socket.create_connection(("127.0.0.1", probe), timeout=5) as s:
        s.sendall(b"DNLD" + struct.pack("!Q", 10**12))  # vượt trần: server đóng kết nối, không phát vô tận
        s.settimeout(5)
        assert s.recv(16) == b""
    assert measure_probe_rtt_ms("127.0.0.1", probe) > 0

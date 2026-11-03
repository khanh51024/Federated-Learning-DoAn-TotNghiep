"""Hợp đồng mới của học liên kết qua mạng: client chỉ gửi trọng số + số mẫu + độ biến thiên; thời gian, độ trễ và độ chính xác do SERVER tính.

Yêu cầu của giảng viên: client không gửi độ chính xác và độ trễ vì server tự tính. Test khóa điều đó ở ba chỗ: nội dung tin nhắn của client,
phép đo mạng do server điều khiển, và một run Flower thật (số tối thiểu client + số liệu trong nhật ký kiểm).
"""
from __future__ import annotations

import json
import multiprocessing
import socket
import time

import flwr as fl
import pytest
import torch

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.network_probe import ProbeServer, obey_server_measurement
from src.federated.server_app import AuditedFlowerFedAvg, start_flower_server
from src.model.mobilenetv3 import create_model, get_parameters

FORBIDDEN_CLIENT_KEYS = {
    "accuracy", "loss", "local_train_time_s", "train_duration_s", "client_wall_ms", "client_decode_ms", "client_encode_ms",
    "probe_rtt_ms", "probe_rtt_persistent_ms", "probe_connect_ms", "probe_up_mb_per_s", "probe_down_mb_per_s", "probe_payload_bytes",
}
REQUIRED_CLIENT_KEYS = {"update_norm", "hash", "parent_model_hash", "round", "model_version", "local_epochs", "client_id"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_client_sends_only_weights_sample_count_and_update_norm():
    core = PlantVillageClient(client_id=0, smoke=True, local_epochs=1)
    adapter = FlowerClientAdapter(core)
    parameters = [v.detach().cpu().numpy() for v in get_parameters(create_model(num_classes=38, pretrained=False)).values()]
    arrays, num_examples, metrics = adapter.fit(parameters, {"round": 1, "local_epochs": 1, "seed": 42})
    assert len(arrays) == len(parameters) and num_examples > 0
    assert not FORBIDDEN_CLIENT_KEYS & set(metrics), sorted(FORBIDDEN_CLIENT_KEYS & set(metrics))
    assert REQUIRED_CLIENT_KEYS <= set(metrics)
    assert metrics["update_norm"] > 0
    assert not hasattr(adapter, "evaluate") or type(adapter).evaluate is fl.client.NumPyClient.evaluate, "client không được tự đánh giá và gửi độ chính xác"


def test_server_driven_measurement_is_timed_by_the_server_and_client_sends_no_numbers():
    port = _free_port()
    server = ProbeServer("127.0.0.1", port, allow_payload=True, measure_payload_bytes=256 * 1024)
    server.start()
    time.sleep(0.3)
    try:
        assert obey_server_measurement("127.0.0.1", port, client_id=3, round_num=2) is True
        result = server.take_measurement(2, 3)
        assert result is not None and result["client_id"] == 3 and result["round"] == 2
        assert 0 < result["rtt_ms_min"] <= result["rtt_ms_p50"] <= result["rtt_ms_max"] and len(result["rtt_samples_ms"]) == 7
        assert result["payload_bytes"] == 256 * 1024
        assert result["up_mb_per_s"] > 0 and result["down_mb_per_s"] > 0
        assert result["up_mb_per_s_net"] >= result["up_mb_per_s"] * 0.99
        assert result["probe_total_ms"] > 0
        assert server.take_measurement(2, 3) is None, "mỗi kết quả chỉ được lấy một lần"
        assert server.take_measurement(2, 4) is None
    finally:
        server.stop()


def test_measurement_without_payload_only_measures_rtt():
    port = _free_port()
    server = ProbeServer("127.0.0.1", port, allow_payload=False)
    server.start()
    time.sleep(0.3)
    try:
        assert obey_server_measurement("127.0.0.1", port, client_id=0, round_num=1) is True
        result = server.take_measurement(1, 0)
        assert result["payload_bytes"] == 0 and "up_mb_per_s" not in result and result["rtt_ms_p50"] > 0
    finally:
        server.stop()


def test_client_refuses_a_server_that_asks_for_an_oversized_block():
    port = _free_port()
    listener = socket.socket()
    listener.bind(("127.0.0.1", port))
    listener.listen(1)

    def _evil_server() -> None:
        import struct
        conn, _ = listener.accept()
        with conn:
            conn.recv(12)
            conn.sendall(b"SUPL" + struct.pack("!Q", ProbeServer.MAX_PAYLOAD_BYTES + 1))
            time.sleep(0.5)

    import threading
    thread = threading.Thread(target=_evil_server, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError):
            obey_server_measurement("127.0.0.1", port, client_id=0, round_num=1, timeout_s=5.0)
    finally:
        listener.close()
        thread.join(timeout=2)


def test_min_clients_cannot_exceed_total_clients():
    with pytest.raises(ValueError):
        AuditedFlowerFedAvg.build(num_clients=3, min_clients=4, initial_parameters=None)
    with pytest.raises(ValueError):
        AuditedFlowerFedAvg.build(num_clients=3, min_clients=0, initial_parameters=None)


def _server(port: int, metrics: str, checkpoint: str) -> None:
    # Tổng 3 client dự kiến nhưng chỉ cần 2 để bắt đầu: run phải chạy được với đúng 2 client.
    start_flower_server(server_address=f"127.0.0.1:{port}", num_clients=3, min_clients=2, rounds=2, local_epochs=1,
                        metrics_file=metrics, checkpoint_file=checkpoint, probe_payload_bytes=64 * 1024)


def _client(client_id: int, port: int) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(client_id=client_id, smoke=True, local_epochs=1, probe_host="127.0.0.1")
    fl.client.start_client(server_address=f"127.0.0.1:{port}", client=FlowerClientAdapter(core).to_client())


def test_flower_run_with_min_clients_records_server_measured_numbers(tmp_path):
    metrics, checkpoint = tmp_path / "history.json", tmp_path / "global.pt"
    port = _free_port()
    server = multiprocessing.Process(target=_server, args=(port, str(metrics), str(checkpoint)))
    clients = [multiprocessing.Process(target=_client, args=(cid, port)) for cid in range(2)]
    server.start()
    for proc in clients:
        proc.start()
    try:
        for proc in clients:
            proc.join(timeout=120)
        server.join(timeout=30)
        assert server.exitcode == 0 and all(proc.exitcode == 0 for proc in clients)
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        assert payload["num_clients_required"] == 3 and payload["min_clients_required"] == 2
        assert len(payload["audit_rounds"]) == 2
        for audit in payload["audit_rounds"]:
            assert audit["accepted_clients"] == 2 and audit["min_clients_required"] == 2
            assert audit["slowest_response_ms"] and audit["slowest_response_ms"] > 0
            for row in audit["clients"]:
                assert row["response_ms"] > 0, "server phải tự bấm giờ phản hồi"
                assert row["probe_rtt_persistent_ms"] > 0 and row["probe_up_mb_per_s"] > 0 and row["probe_down_mb_per_s"] > 0
                assert row["probe_payload_bytes"] == 64 * 1024
                assert row["local_train_time_s"] is not None and row["local_train_time_source"] == "server_estimate"
                assert row["update_norm_matches"] is True and row["upload_hash_ok"] is True
                assert row["predicted_transfer_ms"] > 0
        for key in ("client_wall_ms", "client_decode_ms", "client_encode_ms"):
            assert all(key not in row for audit in payload["audit_rounds"] for row in audit["clients"]), key
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert ckpt["num_clients"] == 3
    finally:
        for proc in [*clients, server]:
            if proc.is_alive():
                proc.terminate()


class _LyingNormAdapter(FlowerClientAdapter):
    """Client khai độ biến thiên gấp đôi số thật: server tự tính lại từ trọng số nhận được nên phải phát hiện."""

    def fit(self, parameters, config):
        arrays, count, metrics = super().fit(parameters, config)
        metrics["update_norm"] = metrics["sent_update_norm"] = metrics["update_norm"] * 2.0
        return arrays, count, metrics


def _lying_server(port: int, progress: str) -> None:
    start_flower_server(server_address=f"127.0.0.1:{port}", num_clients=2, rounds=1, local_epochs=1, metrics_file=None,
                        checkpoint_file=None, progress_file=progress)


def _lying_client(client_id: int, port: int) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(client_id=client_id, smoke=True, local_epochs=1, probe_host="127.0.0.1")
    adapter = _LyingNormAdapter(core) if client_id == 1 else FlowerClientAdapter(core)
    fl.client.start_client(server_address=f"127.0.0.1:{port}", client=adapter.to_client())


def test_server_rejects_a_client_whose_reported_update_norm_disagrees_with_the_weights(tmp_path):
    progress = tmp_path / "progress.json"
    port = _free_port()
    server = multiprocessing.Process(target=_lying_server, args=(port, str(progress)))
    clients = [multiprocessing.Process(target=_lying_client, args=(cid, port)) for cid in range(2)]
    server.start()
    for proc in clients:
        proc.start()
    try:
        # Vòng thất bại làm server báo lỗi nhưng tiến trình Flower không tự thoát khi client còn nối (hành vi sẵn có):
        # chờ server ghi bằng chứng vào tệp tiến độ rồi tự tắt, không dựa vào mã thoát.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if progress.exists() and "failed_round" in json.loads(progress.read_text(encoding="utf-8")):
                break
            time.sleep(1.0)
        else:
            pytest.fail("server không ghi bằng chứng vòng thất bại trong 120 giây")
        failed = json.loads(progress.read_text(encoding="utf-8"))["failed_round"]
        status = {row["client_id"]: row["status"] for row in failed["clients"]}
        assert status[1] == "rejected_update_norm_mismatch" and status[0] == "ok", status
        liar = next(row for row in failed["clients"] if row["client_id"] == 1)
        assert liar["update_norm_matches"] is False and liar["update_norm_reported"] > 1.9 * (liar["received_update_norm"] or 0) * 0.99
    finally:
        for proc in [*clients, server]:
            if proc.is_alive():
                proc.terminate()

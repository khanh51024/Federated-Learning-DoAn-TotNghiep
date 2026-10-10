from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import flwr as fl

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient
from src.federated.server_app import start_flower_server


FAULTS = ["normal", "random_weights", "scaled_update", "noise_update"]


def _server(port: int, reject: bool, metrics: str, checkpoint: str) -> None:
    start_flower_server(
        server_address=f"127.0.0.1:{port}",
        num_clients=4,
        rounds=1,
        local_epochs=1,
        reject_suspicious_updates=reject,
        max_update_norm=5.0,
        metrics_file=metrics,
        checkpoint_file=checkpoint,
    )


def _client(client_id: int, port: int, fault_mode: str) -> None:
    time.sleep(2.0)
    core = PlantVillageClient(
        client_id=client_id,
        smoke=True,
        local_epochs=1,
        fault_mode=fault_mode,
        probe_host="127.0.0.1",
    )
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=FlowerClientAdapter(core).to_client(),
    )


def _run_case(port: int, reject: bool, stem: str, out_dir: Path, expect_abort: bool = False):
    metrics = out_dir / f"{stem}.json"
    checkpoint = out_dir / f"{stem}.pt"

    server = multiprocessing.Process(target=_server, args=(port, reject, str(metrics), str(checkpoint)))
    clients = [
        multiprocessing.Process(target=_client, args=(cid, port, fault))
        for cid, fault in enumerate(FAULTS)
    ]
    server.start()
    for proc in clients:
        proc.start()
    for proc in clients:
        proc.join(timeout=30 if expect_abort else 120)
    server.join(timeout=40)

    try:
        if expect_abort:
            # Hợp đồng strict-round: update bị loại làm vòng không đủ N client hợp lệ -> server hủy vòng,
            # không lặng lẽ gộp phần còn lại và không ghi checkpoint.
            # Thông điệp/mã lỗi chính xác được khóa ở tests/test_demo_server_contract.py::test_reject_policy_aborts_round_...
            # Ở đây chỉ chứng minh hệ quả đầu-cuối: vòng bị hủy nên KHÔNG có checkpoint/metrics nào được ghi.
            # (Không assert exitcode: server có thể chưa thoát khi client còn giữ kết nối gRPC -> test phụ thuộc thời điểm.)
            assert server.exitcode != 0, "server không được kết thúc thành công khi có update bị loại"
            assert not checkpoint.exists()
            assert not metrics.exists()
            return None
        for proc in clients:
            assert proc.exitcode == 0, f"client {proc.pid} exit={proc.exitcode}"
        assert server.exitcode == 0
        assert metrics.exists()
        assert checkpoint.exists()
        return json.loads(metrics.read_text(encoding="utf-8"))
    finally:
        for proc in [*clients, server]:
            if proc.is_alive():
                proc.terminate()


def test_attack_matrix_observe_then_reject(tmp_path):
    observed = _run_case(18120, False, "attack_matrix_observe", tmp_path)
    row = observed["audit_rounds"][0]
    assert row["accepted_clients"] == 4
    assert row["suspicious_clients"] == 3
    assert row["rejected_clients"] == 0

    by_id = {r["client_id"]: r for r in row["clients"]}
    assert by_id[0]["status"] == "ok"
    for cid in (1, 2, 3):
        assert by_id[cid]["status"] == "suspicious"
        assert "update_norm_above_threshold" in by_id[cid]["validation_reason"]
        assert by_id[cid]["received_update_norm"] > 5.0

    # reject-suspicious + strict round: 3 update dị thường bị loại => chỉ còn 1/4 hợp lệ => server hủy vòng
    assert _run_case(18122, True, "attack_matrix_reject", tmp_path, expect_abort=True) is None

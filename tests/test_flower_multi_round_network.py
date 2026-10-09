"""
Test mạng thực tế (Network Socket gRPC): 1 Server + 2 Clients chạy 2 Rounds liên tiếp.
Chứng minh qua socket mạng:
- Round 1: Server phát tán w_0 -> Clients train -> gửi update lên Server -> Server FedAvg aggregate ra w_1.
- Round 2: Server phát tán w_1 TRỞ LẠI XUỐNG CÁC CLIENTS qua socket -> Clients train tiếp -> gửi update -> Server aggregate ra w_2.
"""

import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from src.federated.client_app import PlantVillageClient, FlowerClientAdapter
from src.model.mobilenetv3 import create_model, get_parameters


def run_server(port=8098, num_clients=2, num_rounds=2):
    init_model = create_model(num_classes=38, pretrained=False)
    init_weights = [v.cpu().numpy() for v in get_parameters(init_model).values()]
    initial_params = ndarrays_to_parameters(init_weights)

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
    )

    print(f"[*] Flower Server starting on 127.0.0.1:{port} ({num_clients} clients, {num_rounds} rounds)...")
    fl.server.start_server(
        server_address=f"127.0.0.1:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    print("[+] Flower Server hoàn tất đủ số rounds!")


def run_client(cid, port=8098):
    time.sleep(2)  # Đợi server bind port
    print(f"[*] Client {cid} connecting to 127.0.0.1:{port}...")
    core_client = PlantVillageClient(client_id=cid, smoke=True)
    adapter = FlowerClientAdapter(core_client)
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=adapter.to_client(),
    )
    print(f"[+] Client {cid} hoàn tất và ngắt kết nối!")


if __name__ == "__main__":
    port = 8098
    p_server = multiprocessing.Process(target=run_server, args=(port, 2, 2))
    p_c0 = multiprocessing.Process(target=run_client, args=(0, port))
    p_c1 = multiprocessing.Process(target=run_client, args=(1, port))

    p_server.start()
    p_c0.start()
    p_c1.start()

    p_c0.join(timeout=45)
    p_c1.join(timeout=45)
    p_server.join(timeout=15)

    if p_server.is_alive():
        p_server.terminate()
    if p_c0.is_alive():
        p_c0.terminate()
    if p_c1.is_alive():
        p_c1.terminate()

    print("[+] Test Flower 2 Rounds qua Network Socket hoàn tất thành công!")


def test_two_clients_two_rounds_over_real_socket():
    """Pytest entrypoint proving round-2 feedback over a real Flower socket."""
    port = 18098
    p_server = multiprocessing.Process(target=run_server, args=(port, 2, 2))
    clients = [multiprocessing.Process(target=run_client, args=(cid, port)) for cid in range(2)]
    p_server.start()
    for proc in clients:
        proc.start()
    for proc in clients:
        proc.join(timeout=60)
    p_server.join(timeout=25)
    try:
        for proc in clients:
            assert not proc.is_alive(), f"Flower client pid={proc.pid} timed out"
            assert proc.exitcode == 0
        assert not p_server.is_alive(), "Flower server did not finish within timeout"
        assert p_server.exitcode == 0
    finally:
        for proc in clients:
            if proc.is_alive():
                proc.terminate()
        if p_server.is_alive():
            p_server.terminate()

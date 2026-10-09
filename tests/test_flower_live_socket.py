"""
Test trực tiếp kết nối Flower qua Network Socket (gRPC IP:Port).
Khởi chạy Flower Server lắng nghe port 8099, client kết nối thật qua socket, hoàn tất 1 round.
"""

import multiprocessing
import sys
import time
from pathlib import Path

# Thêm root dir vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from src.federated.client_app import PlantVillageClient, FlowerClientAdapter
from src.model.mobilenetv3 import create_model, get_parameters



def run_server(port=8099):
    init_model = create_model(num_classes=38, pretrained=False)
    init_weights = [v.cpu().numpy() for v in get_parameters(init_model).values()]
    initial_params = ndarrays_to_parameters(init_weights)

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        initial_parameters=initial_params,
    )

    print(f"[*] Server starting on 127.0.0.1:{port}...")
    fl.server.start_server(
        server_address=f"127.0.0.1:{port}",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
    print("[+] Server completed!")


def run_client(port=8099):
    # Đợi server mở socket
    time.sleep(2)
    print(f"[*] Client connecting to 127.0.0.1:{port}...")
    core_client = PlantVillageClient(client_id=0, smoke=True)
    adapter = FlowerClientAdapter(core_client)
    fl.client.start_client(
        server_address=f"127.0.0.1:{port}",
        client=adapter.to_client(),
    )
    print("[+] Client completed!")


if __name__ == "__main__":
    p_server = multiprocessing.Process(target=run_server, args=(8099,))
    p_client = multiprocessing.Process(target=run_client, args=(8099,))

    p_server.start()
    p_client.start()

    p_client.join(timeout=30)
    p_server.join(timeout=10)

    if p_server.is_alive():
        p_server.terminate()
    if p_client.is_alive():
        p_client.terminate()

    print("[+] Test Flower Network Socket hoàn tất!")


def test_live_flower_socket_round_completes():
    """Pytest entrypoint: fail if either real gRPC process cannot finish."""
    port = 18099
    p_server = multiprocessing.Process(target=run_server, args=(port,))
    p_client = multiprocessing.Process(target=run_client, args=(port,))
    p_server.start()
    p_client.start()
    p_client.join(timeout=45)
    p_server.join(timeout=20)
    try:
        assert not p_client.is_alive(), "Flower client did not finish within timeout"
        assert not p_server.is_alive(), "Flower server did not finish within timeout"
        assert p_client.exitcode == 0
        assert p_server.exitcode == 0
    finally:
        if p_client.is_alive():
            p_client.terminate()
        if p_server.is_alive():
            p_server.terminate()

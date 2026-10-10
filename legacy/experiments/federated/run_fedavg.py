"""
Thực nghiệm Huấn luyện Federated Learning (FedAvg 5 clients, 10 rounds).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.federated.server_app import run_federated_simulation

if __name__ == "__main__":
    print("[*] Starting Federated Learning FedAvg Experiment...")
    results = run_federated_simulation(
        partition_dir="data/partitions_stage2_scratch_v3/label100",
        num_clients=5,
        rounds=10,
        local_epochs=1,
    )
    print(f"[+] Hoàn tất Federated Experiment! Best Accuracy: {results['best_accuracy']*100:.2f}%")

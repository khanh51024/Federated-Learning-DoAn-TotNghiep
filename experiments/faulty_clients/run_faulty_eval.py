"""
Thực nghiệm Đo lường Độ bền vững của FedAvg trước Client Lỗi / Tấn công (Faulty Clients).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.federated.server_app import run_federated_simulation

MODES = ["normal", "no_train", "random_weights", "scaled_update"]

if __name__ == "__main__":
    print("[*] Starting Faulty Client Impact Evaluation...")
    summary = []
    for mode in MODES:
        print(f"\n>>> Evaluating Client 0 in mode: '{mode}'...")
        res = run_federated_simulation(
            partition_dir="data/partitions_stage2_scratch_v3/label100",
            num_clients=5,
            rounds=3,
            fault_modes={0: mode},
            smoke=True,
        )
        summary.append({
            "mode": mode,
            "best_accuracy": res["best_accuracy"],
        })
    print("\n[+] Hoàn tất đánh giá Faulty Clients!")
    for item in summary:
        print(f"- Mode '{item['mode']}': Best Acc = {item['best_accuracy']*100:.2f}%")

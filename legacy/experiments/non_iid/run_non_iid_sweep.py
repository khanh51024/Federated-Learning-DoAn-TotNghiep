"""
Thực nghiệm Khảo sát Tác động của Mức độ Non-IID Dirichlet (alpha = 100, 1.0, 0.1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.federated.server_app import run_federated_simulation

CONDITIONS = [
    {"name": "IID (alpha=100.0)", "path": "data/partitions_stage2_scratch_v3/label100"},
    {"name": "Moderate non-IID (alpha=1.0)", "path": "data/partitions_stage2_scratch_v3/label1"},
    {"name": "Severe non-IID (alpha=0.1)", "path": "data/partitions_stage2_scratch_v3/label01"},
]

if __name__ == "__main__":
    print("[*] Starting Non-IID Dirichlet Sweep...")
    summary = []
    for cond in CONDITIONS:
        print(f"\n>>> Running condition: {cond['name']}...")
        res = run_federated_simulation(
            partition_dir=cond["path"],
            num_clients=5,
            rounds=5,
            smoke=True,
        )
        summary.append({
            "condition": cond["name"],
            "best_accuracy": res["best_accuracy"],
        })
    print("\n[+] Hoàn tất khảo sát Non-IID Sweep!")
    for item in summary:
        print(f"- {item['condition']}: Best Acc = {item['best_accuracy']*100:.2f}%")

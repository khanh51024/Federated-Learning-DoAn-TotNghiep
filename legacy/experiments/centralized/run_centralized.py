"""
Thực nghiệm Huấn luyện Tập trung (Centralized Baseline Experiment).
Đánh giá trên tập test cố định: global_test.csv (10.921 ảnh).
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.centralized.train import run_centralized_training

if __name__ == "__main__":
    print("[*] Starting Centralized Baseline Experiment...")
    results = run_centralized_training(
        epochs=10,
        batch_size=32,
        lr=0.01,
        pretrained=False,
    )
    print(f"[+] Hoàn tất Centralized Experiment! Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")

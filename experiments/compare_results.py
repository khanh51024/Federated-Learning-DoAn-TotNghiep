"""Tổng hợp ba baseline sau khi các lệnh train đã chạy xong."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import RESULTS_DIR


def read(name: str) -> dict | None:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    centralized, fedavg, local = read("centralized_metrics.json"), read("fedavg_metrics.json"), read("local_only_metrics.json")
    rows = []
    if centralized:
        value = centralized["test_metrics"]
        rows.append(("Centralized (best val)", value["accuracy"] if "accuracy" in value else value["test_accuracy"],
                     value["macro_f1"] if "macro_f1" in value else value["test_macro_f1"]))
    if fedavg and fedavg["history"]:
        value = fedavg["test_metrics"]
        rows.append(("FedAvg (best val)", value["accuracy"], value["macro_f1"]))
    if local:
        rows.append(("Local-only avg", local["average_accuracy"], local["average_macro_f1"]))
    if not rows:
        raise FileNotFoundError("Chưa có metrics. Hãy chạy ít nhất một thí nghiệm.")
    report = "| Cấu hình | Accuracy | Macro-F1 |\n|---|---:|---:|\n" + "\n".join(f"| {name} | {acc:.4f} | {f1:.4f} |" for name, acc, f1 in rows)
    (RESULTS_DIR / "comparison.md").write_text(report + "\n", encoding="utf-8")
    labels = [row[0] for row in rows]; x = range(len(rows))
    fig, axis = plt.subplots(figsize=(7, 4)); width = 0.35
    axis.bar([i - width / 2 for i in x], [row[1] for row in rows], width, label="Accuracy")
    axis.bar([i + width / 2 for i in x], [row[2] for row in rows], width, label="Macro-F1")
    axis.set_xticks(list(x), labels); axis.set_ylim(0, 1); axis.legend(); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "comparison.png", dpi=150); plt.close(fig)
    print(report)


if __name__ == "__main__":
    main()

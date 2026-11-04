"""Client trôi xa mô hình chung bao nhiêu theo mức lệch dữ liệu, tính từ nhật ký từng vòng của scripts/train_v4.py.

Mỗi vòng, nhật ký ghi `l2_to_merged_without_bn` (khoảng cách giữa mô hình của client và mô hình vừa gộp, không tính BatchNorm)
và `delta_l2_without_bn` (client đã đổi bao nhiêu so với mô hình nhận được). Không dùng ảnh, không dùng GPU.
Thư mục runs/ không có trên máy chủ web, nên kết quả ghi thành CSV để web và terminal đọc.

Chạy: python scripts/report_client_drift.py --runs runs/v4_sweep --out artifacts/sweep_v4/client_drift.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Optional

LAST_ROUNDS = 5
COLUMNS = ("scheme", "runtime", "seeds", "rounds", "changed_mean", "to_merged_mean", "to_merged_last5", "to_merged_last5_std")


def _round_means(history: Path) -> Optional[list[tuple[float, float]]]:
    """(client đổi bao nhiêu, cách mô hình chung bao xa) của từng vòng, trung bình các client. Nhật ký cũ thiếu trường thì trả None."""
    rounds: list[tuple[float, float]] = []
    for line in history.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clients = (json.loads(line).get("train") or {}).get("clients") or []
        if not clients or any("l2_to_merged_without_bn" not in c or "delta_l2_without_bn" not in c for c in clients):
            return None
        rounds.append((statistics.mean(c["delta_l2_without_bn"] for c in clients), statistics.mean(c["l2_to_merged_without_bn"] for c in clients)))
    return rounds or None


def collect(runs: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Gộp theo cách chia và đường chạy. Trả về (các hàng, các run bị bỏ vì nhật ký thiếu trường)."""
    groups: dict[tuple[str, str], list[list[tuple[float, float]]]] = {}
    skipped: list[str] = []
    for config_path in sorted(Path(runs).glob("*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("method") != "fedavg":
            continue
        rounds = _round_means(config_path.parent / "history.jsonl")
        if rounds is None:
            skipped.append(config_path.parent.name)
            continue
        groups.setdefault((str(config.get("scheme")), str(config.get("runtime") or "tu_cai")), []).append(rounds)
    rows = []
    for (scheme, runtime), runs_of_group in sorted(groups.items()):
        last = [statistics.mean(value[1] for value in rounds[-LAST_ROUNDS:]) for rounds in runs_of_group]
        rows.append({
            "scheme": scheme,
            "runtime": runtime,
            "seeds": len(runs_of_group),
            "rounds": len(runs_of_group[0]),
            "changed_mean": round(statistics.mean(value[0] for rounds in runs_of_group for value in rounds), 4),
            "to_merged_mean": round(statistics.mean(value[1] for rounds in runs_of_group for value in rounds), 4),
            "to_merged_last5": round(statistics.mean(last), 4),
            "to_merged_last5_std": round(statistics.stdev(last), 4) if len(last) > 1 else None,
        })
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, skipped = collect(args.runs)
    if not rows:
        raise SystemExit(f"Không có run FedAvg đủ nhật ký trong {args.runs}")
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[drift] đã ghi {args.out}: {len(rows)} hàng; bỏ {len(skipped)} run thiếu trường nhật ký {skipped}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402

def _safe_stats(values: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate report experiment summaries across training seeds.")
    ap.add_argument("--input", default=reg.REPORT_EXPERIMENTS_DIR)
    ap.add_argument("--mode", choices=["smoke", "real"], required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 2026])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = ROOT / args.input
    wanted = set(args.seeds)
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seed = payload.get("seed")
        if payload.get("mode") != args.mode or seed not in wanted:
            continue
        if args.mode == "real" and not payload.get("scientific_accuracy_valid", False):
            continue
        runs.append((summary_path.parent, payload))

    latest_by_seed: Dict[int, tuple[Path, Dict[str, Any]]] = {}
    for run_dir, payload in runs:
        seed = int(payload["seed"])
        latest_by_seed[seed] = (run_dir, payload)

    missing = sorted(wanted - set(latest_by_seed))
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    source_runs = []
    for seed in sorted(latest_by_seed):
        run_dir, payload = latest_by_seed[seed]
        source_runs.append({"seed": seed, "run_dir": str(run_dir)})
        for row in payload.get("results", []):
            exp = str(row.get("experiment"))
            for metric in ("test_accuracy", "test_macro_f1", "test_loss", "elapsed_s"):
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    grouped[exp][metric].append(float(value))

    aggregated = []
    for experiment in sorted(grouped):
        metrics = grouped[experiment]
        aggregated.append({
            "experiment": experiment,
            "accuracy": _safe_stats(metrics.get("test_accuracy", [])),
            "macro_f1": _safe_stats(metrics.get("test_macro_f1", [])),
            "loss": _safe_stats(metrics.get("test_loss", [])),
            "elapsed_s": _safe_stats(metrics.get("elapsed_s", [])),
        })

    report = {
        "mode": args.mode,
        "requested_seeds": sorted(wanted),
        "source_runs": source_runs,
        "missing_seeds": missing,
        "scientific_valid": args.mode == "real" and not missing,
        "partition_variance_included": False,
        "note": "These repeats vary training/model-init seed on the existing fixed partitions. They do not measure partition-seed variance.",
        "aggregated": aggregated,
    }

    output = Path(args.output) if args.output else root / f"multiseed_{args.mode}_summary.json"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment",
            "accuracy_mean", "accuracy_std",
            "macro_f1_mean", "macro_f1_std",
            "loss_mean", "loss_std",
            "elapsed_s_mean", "elapsed_s_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in aggregated:
            writer.writerow({
                "experiment": row["experiment"],
                "accuracy_mean": row["accuracy"]["mean"],
                "accuracy_std": row["accuracy"]["std"],
                "macro_f1_mean": row["macro_f1"]["mean"],
                "macro_f1_std": row["macro_f1"]["std"],
                "loss_mean": row["loss"]["mean"],
                "loss_std": row["loss"]["std"],
                "elapsed_s_mean": row["elapsed_s"]["mean"],
                "elapsed_s_std": row["elapsed_s"]["std"],
            })

    print(json.dumps(report, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())

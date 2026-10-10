from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.centralized.train import run_centralized_training
from src.federated.server_app import run_federated_simulation

PART_ROOT = ROOT / "data" / "partitions_stage2_scratch_v3"


def _real_dataset_root(cli_value: str | None) -> Path | None:
    raw = cli_value or os.environ.get("DATASET_ROOT") or os.environ.get("PLANTVILLAGE_DATASET_ROOT")
    return Path(raw) if raw else None


def _require_real_images(dataset_root: Path | None) -> None:
    if dataset_root is None or not dataset_root.exists():
        raise SystemExit(
            "REAL mode requires --dataset-root or DATASET_ROOT/PLANTVILLAGE_DATASET_ROOT pointing to PlantVillage images."
        )
    probe_csv = PART_ROOT / "label100" / "centralized_train.csv"
    with probe_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))[:16]
    missing = [row["relative_path"] for row in rows if not (dataset_root / row["relative_path"]).exists()]
    if missing:
        raise SystemExit(f"REAL mode aborted: {len(missing)}/16 probe images are missing under {dataset_root}.")


def _scenario_rows() -> List[Dict[str, Any]]:
    suite = json.loads((PART_ROOT / "suite.json").read_text(encoding="utf-8"))
    rows = []
    for name in ("label100", "label1", "label01", "quantity01", "label_quantity01"):
        item = suite["conditions"][name]
        counts = list(item["n_k"])
        rows.append({
            "name": name,
            "scenario": item["scenario"],
            "alpha": item.get("alpha"),
            "quantity_alpha": item.get("quantity_alpha"),
            "n_k": counts,
            "min_n": min(counts),
            "max_n": max(counts),
            "total_n": sum(counts),
        })
    return rows


def _compact_fed_result(name: str, result: Dict[str, Any], elapsed: float, seed: int) -> Dict[str, Any]:
    test = result.get("test_metrics", {}) or {}
    history = result.get("history", result.get("round_history", [])) or []
    last = history[-1] if history else {}
    return {
        "experiment": name,
        "type": "federated",
        "seed": seed,
        "elapsed_s": round(elapsed, 3),
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_loss": test.get("loss"),
        "last_round": last.get("round"),
        "last_val_accuracy": last.get("global_accuracy", last.get("val_accuracy")),
        "last_val_loss": last.get("global_loss", last.get("val_loss")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run reproducible report experiments for Centralized vs Federated Learning.")
    ap.add_argument("--mode", choices=["smoke", "real"], default="smoke")
    ap.add_argument("--dataset-root")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42, help="Training/model-init seed; partition manifests remain fixed unless regenerated separately")
    ap.add_argument("--output", default=reg.REPORT_EXPERIMENTS_DIR)
    args = ap.parse_args()

    dataset_root = _real_dataset_root(args.dataset_root)
    if args.mode == "real":
        _require_real_images(dataset_root)

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "mode": args.mode,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "rounds": args.rounds,
        "epochs": args.epochs,
        "seed": args.seed,
        "partition_note": "Training seed varies model initialization/shuffle only; existing partition manifests are fixed and must be regenerated separately to study partition-seed variance.",
        "scenarios": _scenario_rows(),
        "warning": None if args.mode == "real" else "Smoke mode may use deterministic synthetic tensors when raw PlantVillage images are unavailable. Do not report its accuracy as scientific performance.",
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    results: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    central = run_centralized_training(
        dataset_root=dataset_root,
        epochs=args.epochs,
        batch_size=32,
        smoke=(args.mode == "smoke"),
        require_real_data=(args.mode == "real"),
        seed=args.seed,
        output_dir=run_dir / "centralized",
    )
    results.append({
        "experiment": "centralized",
        "type": "centralized",
        "seed": args.seed,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "test_accuracy": central.get("test_metrics", {}).get("accuracy"),
        "test_macro_f1": central.get("test_metrics", {}).get("macro_f1"),
        "test_loss": central.get("test_metrics", {}).get("loss"),
    })

    for scenario in ("label100", "label1", "label01", "quantity01"):
        t0 = time.perf_counter()
        fed = run_federated_simulation(
            partition_dir=PART_ROOT / scenario,
            dataset_root=dataset_root,
            num_clients=5,
            rounds=args.rounds,
            local_epochs=1,
            batch_size=32,
            seed=args.seed,
            smoke=(args.mode == "smoke"),
            require_real_data=(args.mode == "real"),
            output_dir=run_dir / scenario,
        )
        results.append(_compact_fed_result(scenario, fed, time.perf_counter() - t0, args.seed))

    summary = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "seed": args.seed,
        "results": results,
        "scientific_accuracy_valid": args.mode == "real",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["experiment", "type", "seed", "elapsed_s", "test_accuracy", "test_macro_f1", "test_loss", "last_round", "last_val_accuracy", "last_val_loss"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

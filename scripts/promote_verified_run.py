from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_meta(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out: Dict[str, Any] = {"file_sha256": sha256_file(path)}
    if isinstance(payload, dict):
        for key in ("seed", "epochs", "rounds", "num_clients", "local_epochs", "source"):
            value = payload.get(key)
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
        tm = payload.get("test_metrics")
        if isinstance(tm, dict):
            out["test_metrics"] = {
                k: v for k, v in tm.items() if isinstance(v, (int, float, str, bool))
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote one verified REAL report run into immutable verified artifacts.")
    ap.add_argument("run_dir", type=Path, help="One artifacts/report_experiments/<timestamp> directory")
    ap.add_argument("--scenario", default="label1", choices=["label100", "label1", "label01", "quantity01"])
    ap.add_argument("--preflight", type=Path, default=Path("artifacts/data_preflight.json"))
    ap.add_argument("--verified-root", type=Path, default=Path("artifacts/verified"))
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    preflight_path = args.preflight if args.preflight.is_absolute() else ROOT / args.preflight
    verified_root = args.verified_root if args.verified_root.is_absolute() else ROOT / args.verified_root

    required = [run_dir / "plan.json", run_dir / "summary.json", preflight_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Promotion refused: missing required files: {missing}")

    plan = load_json(run_dir / "plan.json")
    summary = load_json(run_dir / "summary.json")
    preflight = load_json(preflight_path)

    if plan.get("mode") != "real" or summary.get("mode") != "real":
        raise SystemExit("Promotion refused: run is not mode=real")
    if not summary.get("scientific_accuracy_valid", False):
        raise SystemExit("Promotion refused: summary.scientific_accuracy_valid is false")
    if not preflight.get("ok", False):
        raise SystemExit("Promotion refused: PlantVillage preflight did not pass")

    central_ckpt = run_dir / "centralized" / "checkpoints" / "centralized_best.pt"
    central_metrics = run_dir / "centralized" / "metrics" / "centralized_results.json"
    fed_ckpt = run_dir / args.scenario / "checkpoints" / "fedavg_best.pt"
    fed_metrics = run_dir / args.scenario / "metrics" / "fedavg_results.json"
    needed_artifacts = [central_ckpt, central_metrics, fed_ckpt, fed_metrics]
    missing_artifacts = [str(p) for p in needed_artifacts if not p.exists()]
    if missing_artifacts:
        raise SystemExit(f"Promotion refused: missing experiment artifacts: {missing_artifacts}")

    seed = int(summary.get("seed", plan.get("seed", -1)))
    promotion_id = f"real_seed{seed}_{run_dir.name}_{args.scenario}"
    dest = verified_root / promotion_id
    if dest.exists():
        raise SystemExit(f"Promotion refused: destination already exists: {dest}")
    (dest / "checkpoints").mkdir(parents=True, exist_ok=False)
    (dest / "metrics").mkdir(parents=True, exist_ok=False)

    copied = {
        "centralized_checkpoint": dest / "checkpoints" / "centralized_best.pt",
        "centralized_metrics": dest / "metrics" / "centralized_results.json",
        "federated_checkpoint": dest / "checkpoints" / f"fedavg_{args.scenario}_best.pt",
        "federated_metrics": dest / "metrics" / f"fedavg_{args.scenario}_results.json",
    }
    shutil.copy2(central_ckpt, copied["centralized_checkpoint"])
    shutil.copy2(central_metrics, copied["centralized_metrics"])
    shutil.copy2(fed_ckpt, copied["federated_checkpoint"])
    shutil.copy2(fed_metrics, copied["federated_metrics"])
    shutil.copy2(run_dir / "plan.json", dest / "plan.json")
    shutil.copy2(run_dir / "summary.json", dest / "summary.json")
    shutil.copy2(preflight_path, dest / "data_preflight.json")

    manifest = {
        "status": "verified_real_run",
        "promotion_id": promotion_id,
        "source_run_dir": str(run_dir),
        "scenario": args.scenario,
        "training_seed": seed,
        "dataset": {
            "root_at_preflight": preflight.get("dataset_root"),
            "total_images": preflight.get("actual_total_images"),
            "num_classes": preflight.get("actual_num_classes"),
            "class_count_fingerprint_sha256": preflight.get("class_count_fingerprint_sha256"),
        },
        "artifacts": {
            "centralized": {
                "checkpoint": str(copied["centralized_checkpoint"].relative_to(ROOT)),
                "metrics": str(copied["centralized_metrics"].relative_to(ROOT)),
                "checkpoint_meta": checkpoint_meta(copied["centralized_checkpoint"]),
                "metrics_sha256": sha256_file(copied["centralized_metrics"]),
            },
            "federated": {
                "checkpoint": str(copied["federated_checkpoint"].relative_to(ROOT)),
                "metrics": str(copied["federated_metrics"].relative_to(ROOT)),
                "checkpoint_meta": checkpoint_meta(copied["federated_checkpoint"]),
                "metrics_sha256": sha256_file(copied["federated_metrics"]),
            },
        },
        "note": "Promoted only after REAL-mode summary and PlantVillage preflight passed. This manifest does not by itself replace multi-seed statistics.",
    }
    manifest_path = dest / "verified_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_pointer = verified_root / "LATEST_VERIFIED.json"
    latest_pointer.write_text(
        json.dumps({"promotion_id": promotion_id, "manifest": str(manifest_path.relative_to(ROOT))}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

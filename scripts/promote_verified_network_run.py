from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote a scientifically valid Flower network run into immutable verified artifacts.")
    ap.add_argument("--metrics", default=reg.NETWORK_FLOWER_HISTORY_JSON)
    ap.add_argument("--checkpoint", default=reg.FEDERATED_NETWORK_GLOBAL_PT)
    ap.add_argument("--expected-clients", type=int, default=5)
    ap.add_argument("--expected-rounds", type=int, required=True)
    ap.add_argument("--tag", default="network")
    ap.add_argument("--output-root", default=reg.VERIFIED_NETWORK_DIR)
    args = ap.parse_args()

    metrics = (ROOT / args.metrics).resolve() if not Path(args.metrics).is_absolute() else Path(args.metrics).resolve()
    checkpoint = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()
    validation_path = reg.abs_path(reg.NETWORK_ARTIFACT_VALIDATION_JSON)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "validate_network_artifact.py"),
        "--metrics", str(metrics),
        "--checkpoint", str(checkpoint),
        "--expected-clients", str(args.expected_clients),
        "--expected-rounds", str(args.expected_rounds),
        "--output", str(validation_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print("Promotion refused: network artifact validation did not pass.", file=sys.stderr)
        return 2

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("ok") is not True:
        print("Promotion refused: validation report is not OK.", file=sys.stderr)
        return 2

    metrics_payload: Dict[str, Any] = json.loads(metrics.read_text(encoding="utf-8"))
    seed = int(metrics_payload.get("seed", -1))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    promotion_id = f"{stamp}_{args.tag}_seed{seed}_r{args.expected_rounds}"
    out_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    dest = out_root / promotion_id
    if dest.exists():
        raise SystemExit(f"Destination already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=False)

    metrics_dest = dest / "network_metrics.json"
    checkpoint_dest = dest / "global_model.pt"
    validation_dest = dest / "validation.json"
    shutil.copy2(metrics, metrics_dest)
    shutil.copy2(checkpoint, checkpoint_dest)
    shutil.copy2(validation_path, validation_dest)

    manifest = {
        "promotion_id": promotion_id,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scientific_valid": True,
        "source": "flower_network_fedavg",
        "seed": seed,
        "rounds": int(metrics_payload.get("rounds", args.expected_rounds)),
        "num_clients": int(metrics_payload.get("num_clients_required", args.expected_clients)),
        "require_real_clients": metrics_payload.get("require_real_clients"),
        "require_real_evaluation": metrics_payload.get("require_real_evaluation"),
        "evaluation_dataset_root": metrics_payload.get("evaluation_dataset_root"),
        "evaluation_partition_dir": metrics_payload.get("evaluation_partition_dir"),
        "final_test_metrics": metrics_payload.get("final_test_metrics"),
        "files": {
            "network_metrics.json": sha256_file(metrics_dest),
            "global_model.pt": sha256_file(checkpoint_dest),
            "validation.json": sha256_file(validation_dest),
        },
    }
    manifest_path = dest / "verified_network_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    latest = out_root / "LATEST.txt"
    latest.write_text(promotion_id + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "promotion_dir": str(dest), "manifest": manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

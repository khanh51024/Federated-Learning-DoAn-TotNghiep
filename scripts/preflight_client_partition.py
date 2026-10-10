from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
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
    ap = argparse.ArgumentParser(description="Validate only one Flower client's local PlantVillage partition.")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--partition-dir", type=Path, required=True)
    ap.add_argument("--client-id", type=int, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    partition_dir = args.partition_dir if args.partition_dir.is_absolute() else ROOT / args.partition_dir
    client_csv = partition_dir / "clients" / f"client_{args.client_id:02d}.csv"
    partition_config = partition_dir / "partition_config.json"

    result: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "partition_dir": str(partition_dir),
        "partition_id": partition_dir.name,
        "client_id": args.client_id,
        "client_manifest": str(client_csv),
        "ok": False,
    }

    if not dataset_root.is_dir():
        result["error"] = "dataset_root does not exist or is not a directory"
    elif not client_csv.is_file():
        result["error"] = "client manifest does not exist"
    else:
        with client_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        missing = []
        labels = Counter()
        class_names = Counter()
        for row in rows:
            labels[int(row["label"])] += 1
            class_names[row.get("class_name", "")] += 1
            path = dataset_root / row["relative_path"]
            if not path.is_file() and len(missing) < 50:
                missing.append(row["relative_path"])

        # Need the exact missing count, not only examples.
        missing_count = sum(1 for row in rows if not (dataset_root / row["relative_path"]).is_file())
        result.update(
            {
                "num_examples": len(rows),
                "num_classes_present": len(labels),
                "labels_present": sorted(labels),
                "class_distribution": {str(k): labels[k] for k in sorted(labels)},
                "missing_count": missing_count,
                "missing_examples": missing[:20],
                "client_manifest_sha256": sha256_file(client_csv),
                "partition_config_sha256": sha256_file(partition_config) if partition_config.exists() else None,
            }
        )
        result["ok"] = len(rows) > 0 and missing_count == 0

    output = args.output
    if output is None:
        output = reg.abs_path(reg.CLIENT_PREFLIGHT_DIR) / f"{partition_dir.name}_client_{args.client_id:02d}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402

MANIFEST = ROOT / "data" / "dataset_manifest.json"
PART_ROOT = ROOT / "data" / "partitions_stage2_scratch_v3" / "label100"


def _load_expected() -> Dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _collect_partition_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name in ("centralized_train.csv", "centralized_val.csv", "centralized_test.csv"):
        p = PART_ROOT / name
        if p.exists():
            rows.extend(_read_rows(p))
    return rows


def _class_counts(root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        n = 0
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
            n += sum(1 for _ in child.glob(ext))
        if n:
            counts[child.name] = n
    return counts


def _sample_integrity(root: Path, rows: List[Dict[str, str]], limit: int = 128) -> Tuple[int, List[str]]:
    missing: List[str] = []
    checked = 0
    if not rows:
        return checked, missing
    stride = max(1, len(rows) // max(1, limit))
    for row in rows[::stride][:limit]:
        checked += 1
        rel = row["relative_path"]
        if not (root / rel).is_file():
            missing.append(rel)
    return checked, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a PlantVillage image root against the locked thesis protocol.")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--full-path-check", action="store_true", help="Check every manifest path, not only a deterministic sample.")
    ap.add_argument("--output", default=reg.DATA_PREFLIGHT_JSON)
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    expected = _load_expected()
    rows = _collect_partition_rows()
    expected_total = int(expected.get("total_images", 54305))
    expected_classes = int(expected.get("num_classes", 38))

    result: Dict = {
        "dataset_root": str(root),
        "exists": root.exists(),
        "expected_total_images": expected_total,
        "expected_num_classes": expected_classes,
        "checks": {},
        "ok": False,
    }

    if not root.exists() or not root.is_dir():
        result["error"] = "dataset root does not exist or is not a directory"
    else:
        counts = _class_counts(root)
        total = sum(counts.values())
        result["actual_total_images"] = total
        result["actual_num_classes"] = len(counts)
        result["checks"]["class_count_matches"] = len(counts) == expected_classes
        result["checks"]["total_image_count_matches"] = total == expected_total

        class_names = sorted(counts)
        manifest_class_names = sorted({row["class_name"] for row in rows})
        result["checks"]["class_names_match_manifests"] = class_names == manifest_class_names
        result["class_names_missing"] = sorted(set(manifest_class_names) - set(class_names))
        result["class_names_unexpected"] = sorted(set(class_names) - set(manifest_class_names))

        check_rows = rows if args.full_path_check else rows[:]
        if args.full_path_check:
            missing_paths = [row["relative_path"] for row in check_rows if not (root / row["relative_path"]).is_file()]
            checked = len(check_rows)
        else:
            checked, missing_paths = _sample_integrity(root, check_rows, limit=128)
        result["path_check"] = {
            "mode": "full" if args.full_path_check else "sample",
            "checked": checked,
            "missing_count": len(missing_paths),
            "missing_examples": missing_paths[:20],
        }
        result["checks"]["manifest_paths_resolve"] = len(missing_paths) == 0
        result["ok"] = all(result["checks"].values())

        # Stable fingerprint of class names/counts for experiment logs. This is
        # not a byte-level dataset checksum, but catches accidental folder/count
        # drift without hashing 54k image files.
        summary_blob = json.dumps({k: counts[k] for k in sorted(counts)}, separators=(",", ":"), ensure_ascii=False)
        result["class_count_fingerprint_sha256"] = hashlib.sha256(summary_blob.encode("utf-8")).hexdigest()
        result["class_counts"] = {k: counts[k] for k in sorted(counts)}

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

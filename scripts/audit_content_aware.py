#!/usr/bin/env python
"""Audit path, group and exact-byte disjointness for a generated index."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fl_training.prepare import audit_manifest_directory
from fl_training.content_audit import audit_image_content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dataset-root", type=Path)
    args = ap.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8"))["images"]
    if not index.get('partitions'):
        raise ValueError('Cannot audit an empty partition index')
    dataset = (args.dataset_root or ROOT / index['dataset_path']).resolve()
    def hash_file(key):
        return key, hashlib.sha256((dataset / key).read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(hash_file, cache))
    stale = [key for key in hashes if hashes[key] != cache[key]['sha256']]
    if stale:
        raise ValueError(f'Stale content cache for {len(stale)} images; rebuild with --strict')

    def load(path: Path):
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def key(row: dict) -> str:
        return "/".join(row["relative_path"].replace("\\", "/").split("/")[-2:])

    report = {"index": str(args.index), "partitions": [], "all_ok": True,
              "freshly_hashed_images": len(hashes), "source": "actual image bytes"}
    # index entries store paths relative to the package root, not to the
    # partition directory containing index.json.
    root = ROOT
    for entry in index["partitions"].values():
        part = root / entry["relative_dir"]
        structure = audit_manifest_directory(part, total_source_images=index['total_source_images'])
        content = audit_image_content(part, dataset, hashes=hashes)
        split_rows = {
            "train": load(part / "centralized_train.csv"),
            "val": load(part / "global_val.csv"),
            "test": load(part / "global_test.csv"),
        }
        paths = {name: {r["relative_path"] for r in rows} for name, rows in split_rows.items()}
        groups = {name: {r["group_id"] for r in rows} for name, rows in split_rows.items()}
        byte_hashes = {name: {hashes[key(r)] for r in rows} for name, rows in split_rows.items()}
        clients = [load(part / "clients" / f"client_{i:02d}.csv") for i in range(entry["num_clients"])]
        client_hashes = [{hashes[key(r)] for r in rows} for rows in clients]
        split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        rec = {
            "structure": structure,
            "content": content,
            "scenario": entry["scenario"],
            "alpha": entry.get("alpha"),
            "quantity_alpha": entry.get("quantity_alpha"),
            "feature_skew": entry["feature_skew"],
            "counts": {k: len(v) for k, v in paths.items()},
            "path_overlaps": {f"{a}__{b}": len(paths[a] & paths[b]) for a, b in split_pairs},
            "group_overlaps": {f"{a}__{b}": len(groups[a] & groups[b]) for a, b in split_pairs},
            "byte_hash_overlaps": {f"{a}__{b}": len(byte_hashes[a] & byte_hashes[b]) for a, b in split_pairs},
            "client_byte_hash_overlap_pairs": sum(
                bool(client_hashes[i] & client_hashes[j])
                for i in range(len(client_hashes)) for j in range(i)
            ),
        }
        rec["ok"] = all(not values for values in rec["path_overlaps"].values()) and all(
            not values for values in rec["group_overlaps"].values()
        ) and all(not values for values in rec["byte_hash_overlaps"].values()) and rec["client_byte_hash_overlap_pairs"] == 0
        report["partitions"].append(rec)
        report["all_ok"] &= rec["ok"]

    output = args.output or args.index.parent / "content_aware_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"], "partitions": len(report["partitions"]), "output": str(output)}))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

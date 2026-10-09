#!/usr/bin/env python
"""Decode every PlantVillage source image exactly once and write an audit JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.leaf_groups import IMAGE_EXTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode each source image once.")
    parser.add_argument("--dataset-root", default="../PlantVillage-Dataset/raw/color")
    parser.add_argument("--output", default="data/source_integrity.json")
    args = parser.parse_args()

    dataset_root = (ROOT / args.dataset_root).resolve()
    paths = sorted(p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    failures = []
    digest = hashlib.sha256()
    started = time.time()
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(dataset_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\n")
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": relative, "error": str(exc)})
        if index % 5000 == 0:
            print(f"decoded {index}/{len(paths)} images; failures={len(failures)}", flush=True)

    report = {
        "dataset_root": str(dataset_root),
        "image_count": len(paths),
        "decoded_ok": len(paths) - len(failures),
        "failures": failures,
        "relative_paths_sha256": digest.hexdigest(),
        "seconds": round(time.time() - started, 2),
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

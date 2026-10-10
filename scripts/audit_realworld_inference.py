"""Auditable inference smoke check on user-supplied images.

Usage:
    python scripts/audit_realworld_inference.py --images web/static/samples --output artifacts/inference_audit.json

Image filenames are NOT ground-truth labels. This script never computes accuracy
without an externally verified set of labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402
from web.demo_evidence import predict_published, load_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=reg.abs_path(reg.INFERENCE_AUDIT_JSON))
    args = parser.parse_args()
    image_path = args.images.resolve()
    files = sorted(p for p in (image_path.iterdir() if image_path.is_dir() else [image_path])
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        parser.error("No images found")
    evidence = load_evidence()
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Unlabelled images: predictions only, NOT ground-truth accuracy.",
        "evidence_available": evidence.get("available", False),
        "images": [],
    }
    for path in files:
        row = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                row["size"] = [rgb.width, rgb.height]
                for model in ("centralized", "fedavg"):
                    pred = predict_published(rgb, model)
                    row[model] = {
                        "predicted_class": pred["predicted_class"],
                        "confidence_pct": pred["confidence_pct"],
                        "top3": pred["top3"],
                        "is_uncertain": pred["is_uncertain"],
                        "confidence_calibrated": pred["confidence_calibrated"],
                        "checkpoint_sha256": pred["checkpoint_sha256"],
                    }
                row["models_disagree"] = row["centralized"]["predicted_class"] != row["fedavg"]["predicted_class"]
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            row["error"] = str(exc)
        output["images"].append(row)
        print(f"{path.name}: " + (str(row["error"]) if "error" in row else
              f"C={row['centralized']['predicted_class']} ({row['centralized']['confidence_pct']}%), "
              f"F={row['fedavg']['predicted_class']} ({row['fedavg']['confidence_pct']}%)"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit saved to {args.output}")
    return 0 if all("error" not in row for row in output["images"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize published validation accuracy separately for PlantVillage and PlantDoc.

Does not claim local image-level re-evaluation. Uses the pinned comparison CSV.
Run: python scripts/report_domain_gap.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402

SOURCE = reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/fedavg/validation_comparison.csv")
OUTPUT = reg.abs_path(reg.DOMAIN_GAP_REPORT_JSON)


def build_report(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_model = {}
    for row in rows:
        if row["model"] not in {"Centralized v6", "FedAvg v7"}:
            continue
        by_model.setdefault(row["model"], {})[row["source"]] = {
            "count": int(row["count"]),
            "accuracy_pct": round(float(row["top1_accuracy"]) * 100, 2),
            "macro_f1_pct": round(float(row["macro_f1_supported"]) * 100, 2),
            "crop_accuracy_pct": round(float(row["crop_accuracy"]) * 100, 2),
        }
    for model, sources in by_model.items():
        if set(sources) != {"plantdoc", "plantvillage"}:
            raise ValueError(f"Missing dataset metric for {model}")
        sources["plantvillage_minus_plantdoc_accuracy_pp"] = round(
            sources["plantvillage"]["accuracy_pct"] - sources["plantdoc"]["accuracy_pct"], 2)
    return {
        "source_csv": path.relative_to(ROOT).as_posix(),
        "evidence": "published comparison artifact; not a locally repeated validation",
        "models": by_model,
        "note": "Crop accuracy is a separately reported metric; do not interpret as full 38-class top-1 accuracy.",
        "recommendation": "Evaluate and improve real-world/domain robustness before optimizing confidence display.",
    }


if __name__ == "__main__":
    report = build_report(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, m in report["models"].items():
        print(f"{name}: PlantVillage={m['plantvillage']['accuracy_pct']}%, "
              f"PlantDoc={m['plantdoc']['accuracy_pct']}%, "
              f"gap={m['plantvillage_minus_plantdoc_accuracy_pp']} pp")
    print(f"Saved: {OUTPUT}")

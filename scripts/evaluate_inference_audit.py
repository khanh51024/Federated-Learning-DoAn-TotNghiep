"""Evaluate external-photo predictions only against independently verified labels.

CSV schema: filename,true_class (true_class must match canonical 38-class taxonomy).
Example:
  python scripts/evaluate_inference_audit.py --audit artifacts/inference_audit.json --labels my_labels.csv
No accuracy is reported for missing labels or files without a successful prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def evaluate(audit: dict, rows: list[dict], valid_classes: set[str]) -> dict:
    truth = {}
    for row in rows:
        name = Path(row["filename"]).name
        label = row["true_class"].strip()
        if label not in valid_classes:
            raise ValueError(f"Unknown ground-truth class for {name}: {label!r}")
        if name in truth:
            raise ValueError(f"Duplicate filename in label CSV: {name}")
        truth[name] = label
    entries = [item for item in audit["images"] if "error" not in item and Path(item["file"]).name in truth]
    counts = Counter(Path(item["file"]).name for item in audit["images"])
    if any(counts[Path(item["file"]).name] > 1 for item in entries):
        raise ValueError("Ambiguous duplicate image filename in audit; use unique filenames")
    results = {"labeled_count": len(entries), "labels_missing_from_audit": sorted(set(truth) - set(counts)), "models": {}}
    for model in ("centralized", "fedavg"):
        correct = 0
        high_conf_wrong = []
        errors = []
        for item in entries:
            pred = item[model]
            actual = truth[Path(item["file"]).name]
            guessed = pred["predicted_class"]
            if guessed == actual:
                correct += 1
            else:
                err = {"filename": Path(item["file"]).name, "actual": actual,
                       "predicted": guessed, "confidence_pct": pred["confidence_pct"]}
                errors.append(err)
                if pred["confidence_pct"] >= 90:
                    high_conf_wrong.append(err)
        results["models"][model] = {
            "accuracy": round(correct / len(entries), 4) if entries else None,
            "correct": correct,
            "wrong": len(errors),
            "high_confidence_wrong_count": len(high_conf_wrong),
            "high_confidence_wrong": high_conf_wrong,
            "errors": errors,
        }
    results["warning"] = "Small or non-representative labeled samples cannot validate deployment accuracy."
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/labeled_inference_evaluation.json"))
    args = parser.parse_args()
    from sys import path as syspath
    syspath.insert(0, str(Path(__file__).resolve().parents[1]))
    from web.demo_evidence import load_evidence
    evidence = load_evidence()
    if not evidence.get("available"):
        raise SystemExit("Evidence unavailable: " + evidence.get("reason", "unknown"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    with args.labels.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"filename", "true_class"}.issubset(reader.fieldnames or []):
            parser.error("CSV must contain filename,true_class")
        labels = list(reader)
    report = evaluate(audit, labels, set(evidence["classes"]))
    if report["labels_missing_from_audit"]:
        raise SystemExit("Missing images in audit: " + ", ".join(report["labels_missing_from_audit"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"labeled_count": report["labeled_count"], "models": {
        k: {key: v[key] for key in ("accuracy", "correct", "wrong", "high_confidence_wrong_count")}
        for k, v in report["models"].items()}}, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

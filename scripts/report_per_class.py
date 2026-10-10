"""Phân tích theo từng lớp trên số liệu validation ĐÃ CÔNG BỐ trong bundle (không cần ảnh).

Nguồn: `<bundle>/<model>/validation_metrics.json` (quality.by_source.<nguồn>.per_class_recall, đã kiểm hash bởi
scripts/import_branch_demo.py). Đây là phân tích trên số công bố, KHÔNG phải tái đánh giá cục bộ:
không có ảnh validation nên không dựng được confusion matrix đầy đủ hay đo calibration.

Chạy: python scripts/report_per_class.py   ->  artifacts/plantdoc_per_class_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

MODELS = {"Centralized v6": "centralized", "FedAvg v7": "fedavg"}
WEAK_RECALL = 0.5
MIN_SUPPORT = 5  # lớp support nhỏ hơn dễ nhiễu, vẫn liệt kê nhưng đánh dấu


def _load(model_dir: str) -> dict[str, Any]:
    path = reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/{model_dir}/validation_metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))["validation"]


def _source(validation: dict[str, Any], source: str) -> dict[str, Any]:
    quality = validation["quality"]["by_source"][source]
    summary = validation["by_source"][source]
    per_class = quality["per_class_recall"]
    count = int(quality["count"])
    correct = count - int(quality["wrong_crop_count"]) - int(quality["right_crop_wrong_disease_count"])
    wrong = count - correct
    return {
        "count": count,
        "classes_with_support": len(per_class),
        "top1_accuracy": summary["top1_accuracy"],
        "crop_accuracy": quality["crop_accuracy"],
        "error_breakdown": {
            "correct": correct,
            "wrong_crop": int(quality["wrong_crop_count"]),
            "right_crop_wrong_disease": int(quality["right_crop_wrong_disease_count"]),
        },
        "high_confidence_wrong": {
            "threshold": validation["quality"]["high_confidence_threshold"],
            "count": int(quality["high_confidence_wrong_count"]),
            "rate_of_all": quality["high_confidence_wrong_rate"],
            "share_of_errors": round(int(quality["high_confidence_wrong_count"]) / wrong, 4) if wrong else None,
        },
        "per_class_recall": {k: {"support": int(v["support"]), "recall": v["recall"]} for k, v in sorted(per_class.items())},
        "consistency": {
            # recall có trọng số support phải trùng top-1; phân rã lỗi phải cộng đúng tổng
            "support_sum_equals_count": sum(int(v["support"]) for v in per_class.values()) == count,
            "weighted_recall_equals_top1": abs(
                sum(v["support"] * v["recall"] for v in per_class.values()) / count - summary["top1_accuracy"]
            ) < 1e-9,
            "correct_matches_top1": correct == round(count * summary["top1_accuracy"]),
        },
    }


def build_report() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, model_dir in MODELS.items():
        validation = _load(model_dir)
        models[name] = {src: _source(validation, src) for src in ("plantdoc", "plantvillage")}

    fed, cen = models["FedAvg v7"], models["Centralized v6"]
    pd_fed = fed["plantdoc"]["per_class_recall"]
    pd_cen = cen["plantdoc"]["per_class_recall"]
    pv_fed = fed["plantvillage"]["per_class_recall"]

    rows = []
    for cls, info in pd_fed.items():
        pv = pv_fed.get(cls, {}).get("recall")
        rows.append({
            "class": cls,
            "plantdoc_support": info["support"],
            "fedavg_plantdoc_recall": info["recall"],
            "centralized_plantdoc_recall": pd_cen.get(cls, {}).get("recall"),
            "fedavg_plantvillage_recall": pv,
            "fedavg_domain_gap_pp": None if pv is None else round((pv - info["recall"]) * 100, 2),
            "low_support": info["support"] < MIN_SUPPORT,
        })
    rows.sort(key=lambda r: (r["fedavg_plantdoc_recall"], -r["plantdoc_support"], r["class"]))

    weak = [r for r in rows if r["fedavg_plantdoc_recall"] < WEAK_RECALL]
    worse = [r for r in rows if r["centralized_plantdoc_recall"] is not None
             and r["fedavg_plantdoc_recall"] < r["centralized_plantdoc_recall"]]
    better = [r for r in rows if r["centralized_plantdoc_recall"] is not None
              and r["fedavg_plantdoc_recall"] > r["centralized_plantdoc_recall"]]
    pd = fed["plantdoc"]
    pv_classes = fed["plantvillage"]["classes_with_support"]

    return {
        "evidence": "published validation_metrics.json (hash-verified bundle); not locally re-evaluated — no validation images on this machine",
        "source_bundle": reg.DEMO_BUNDLE_DIR,
        "protocol_id": reg.DEMO_PROTOCOL_ID,
        "weak_recall_threshold": WEAK_RECALL,
        "min_support_for_reliable_class": MIN_SUPPORT,
        "models": models,
        "fedavg_vs_centralized_plantdoc": {
            "classes_compared": len(rows),
            "fedavg_worse_in": len(worse),
            "fedavg_better_in": len(better),
            "fedavg_worse_classes": [r["class"] for r in worse],
        },
        "fedavg_plantdoc_class_table": rows,
        "summary": {
            "plantdoc_classes_covered": f"{pd['classes_with_support']}/{pv_classes}",
            "weak_classes_fedavg": [r["class"] for r in weak],
            "weak_images_fedavg": sum(r["plantdoc_support"] for r in weak),
            "error_breakdown_fedavg_plantdoc": pd["error_breakdown"],
            "high_confidence_wrong_fedavg_plantdoc": pd["high_confidence_wrong"],
        },
        "not_available_without_images": [
            "confusion matrix đầy đủ trên ảnh PlantDoc",
            "danh sách từng ảnh sai với confidence cao",
            "calibration (ECE / temperature scaling) trên tập validation độc lập",
            "tỷ lệ PlantDoc trong dữ liệu huấn luyện (bundle chỉ công bố validation_manifest, không kèm manifest train)",
        ],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1258) không in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build_report()
    out = reg.abs_path(reg.PER_CLASS_REPORT_JSON)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[report_per_class] wrote {reg.PER_CLASS_REPORT_JSON}")
    s = report["summary"]
    print(f"[report_per_class] PlantDoc phủ {s['plantdoc_classes_covered']} lớp; {len(s['weak_classes_fedavg'])} lớp recall<{WEAK_RECALL:.0%} "
          f"({s['weak_images_fedavg']} ảnh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

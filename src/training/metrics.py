"""Chỉ số đánh giá theo nguồn ảnh và điểm chọn checkpoint cân bằng giữa các nguồn."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

NUM_CLASSES = 38
MIN_SOURCE_IMAGES = 30   # nguồn có ít ảnh hơn mức này trong tập kiểm tra không tham gia điểm chọn (quá nhiễu)
LAB_DOMAIN = "plantvillage"


def macro_f1(targets: np.ndarray, preds: np.ndarray) -> float:
    """Macro-F1 trên các lớp CÓ mặt trong nhãn thật của tập đang đo (lớp vắng mặt không kéo điểm về 0)."""
    classes = np.unique(targets)
    if len(classes) == 0:
        return float("nan")
    scores = []
    for c in classes:
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def source_key(row: dict[str, Any]) -> str:
    """Nhóm nguồn để báo cáo: PlantVillage, PlantDoc, hoặc tên bộ dữ liệu ngoài đồng."""
    if row["source_domain"] == LAB_DOMAIN:
        return "PlantVillage"
    return "PlantDoc" if row["source_dataset"] in ("PlantDoc", "plantdoc") else row["source_dataset"]


def summarize(logits: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """top-1 và macro-F1 toàn tập, theo miền (phòng thí nghiệm / ngoài đồng), theo từng nguồn; cùng điểm chọn checkpoint.

    selection_score = trung bình macro-F1 của các nguồn có >= MIN_SOURCE_IMAGES ảnh: mỗi nguồn nặng như nhau, nên bộ ảnh nhiều (ngô, cà chua có ảnh tăng cường)
    không quyết định việc chọn checkpoint thay cho PlantDoc hay PlantVillage."""
    targets = np.array([int(r["class_id"]) for r in rows])
    preds = logits.argmax(axis=1)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[source_key(row)].append(i)
    by_source = {}
    for name, idx in sorted(groups.items()):
        t, p = targets[idx], preds[idx]
        by_source[name] = {"n": len(idx), "top1": round(float(np.mean(t == p)), 4), "macro_f1": round(macro_f1(t, p), 4)}
    eligible = [v["macro_f1"] for v in by_source.values() if v["n"] >= MIN_SOURCE_IMAGES]
    lab = np.array([r["source_domain"] == LAB_DOMAIN for r in rows])
    out = {
        "n": len(rows), "top1": round(float(np.mean(targets == preds)), 4) if len(rows) else None, "macro_f1": round(macro_f1(targets, preds), 4),
        "plantvillage_top1": round(float(np.mean(targets[lab] == preds[lab])), 4) if lab.any() else None,
        "field_top1": round(float(np.mean(targets[~lab] == preds[~lab])), 4) if (~lab).any() else None,
        "field_macro_f1": round(macro_f1(targets[~lab], preds[~lab]), 4) if (~lab).any() else None,
        "by_source": by_source,
        "selection_score": round(float(np.mean(eligible)), 4) if eligible else None,
        "selection_sources": sorted(k for k, v in by_source.items() if v["n"] >= MIN_SOURCE_IMAGES),
    }
    return out

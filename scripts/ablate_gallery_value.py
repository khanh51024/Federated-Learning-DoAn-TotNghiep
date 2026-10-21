"""Phân tích thêm của scripts/eval_gallery_value.py: thư viện v3 cộng từng nguồn ảnh mới, chạy trên đặc trưng DINOv2 ĐÃ LƯU.

Không nạp mô hình và không trích đặc trưng mới (chỉ numpy), nên nhẹ về bộ nhớ. Cần `data/reeval/gallery_v4_dinov2.npz` do eval_gallery_value.py sinh;
ảnh nào chưa có đặc trưng trong tệp đó (ví dụ nguồn thêm sau lần đo) bị bỏ qua và được đếm vào `skipped_without_features`.

Mục đích: tách tác động của (a) trần ảnh mỗi lớp trong thư viện v4 làm mất lá PlantDoc khỏi (b) từng nguồn mới. Đây là phân tích CHẨN ĐOÁN trên tập thử đã khóa,
không dùng để chọn nguồn (chọn nguồn theo PlantWild là học trên tập thử).

Chạy:
    python scripts/ablate_gallery_value.py        # ghi artifacts/dataset_v4_value_ablation.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_foundation_models import build_items, knn_scores, read_gallery_features_v3, _jsonl  # noqa: E402
from scripts.eval_gallery_value import GALLERY_NPZ, V4_DIR, select_field_gallery  # noqa: E402
from src import registry as reg  # noqa: E402

PLANTDOC_NAMES = ("PlantDoc", "plantdoc")
K = 10
FIELD_CAP = 120


def top1(gallery: np.ndarray, labels: np.ndarray, sets: dict[str, tuple[np.ndarray, np.ndarray]], num_classes: int) -> dict[str, float]:
    return {name: round(100 * float((knn_scores(x, gallery, labels, K, num_classes).argmax(1) == t).mean()), 1) for name, (x, t) in sets.items()}


def main() -> int:
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    eval_sets_raw, gallery_v3 = build_items(0)
    order = [(n, i) for n, items in eval_sets_raw.items() for i in items]
    eval_x, gal3 = read_gallery_features_v3(len(order), len(gallery_v3))
    gal3_labels = np.array([g["target"] for g in gallery_v3])
    names = np.array([n for n, _ in order])
    targets = np.array([i["target"] for _, i in order])

    data = np.load(reg.abs_path(GALLERY_NPZ), allow_pickle=False)
    cache = {str(k): v for k, v in zip(data["keys"], data["feats"])}
    v4 = reg.abs_path(V4_DIR)
    train = _jsonl(v4 / "manifests" / "train.jsonl")
    test = _jsonl(v4 / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    field_all = select_field_gallery(train, FIELD_CAP)
    field = [r for r in field_all if r["sample_id"] in cache]
    field_test = [r for r in test if r["source_domain"] == "plantdoc" and r["source_dataset"] not in (*PLANTDOC_NAMES, "own_photos") and r["sample_id"] in cache]

    sets = {n: (eval_x[names == n], targets[names == n]) for n in ("ext_strict", "ext_healthy", "pd_val", "pd_test")}
    sets["v4_field_test"] = (np.stack([cache[r["sample_id"]] for r in field_test]), np.array([int(r["class_id"]) for r in field_test]))

    def add(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            return gal3, gal3_labels
        return (np.concatenate([gal3, np.stack([cache[r["sample_id"]] for r in rows])]),
                np.concatenate([gal3_labels, np.array([int(r["class_id"]) for r in rows])]))

    external = [r for r in field if r["source_dataset"] not in PLANTDOC_NAMES]
    result: dict[str, Any] = {"v3 (PlantVillage 40 mỗi lớp + mọi lá PlantDoc của tập học)": top1(gal3, gal3_labels, sets, len(classes))}
    result["v3 + mọi ảnh ngoài-PlantDoc trong mẫu v4"] = top1(*add(external), sets, len(classes))
    per_source = {src: top1(*add([r for r in external if r["source_dataset"] == src]), sets, len(classes))
                  for src in sorted({r["source_dataset"] for r in external})}
    pv_idx = [i for i, g in enumerate(gallery_v3) if g["source"] == "plantvillage"]
    v4_gallery = np.concatenate([gal3[pv_idx], np.stack([cache[r["sample_id"]] for r in field])])
    v4_labels = np.concatenate([gal3_labels[pv_idx], np.array([int(r["class_id"]) for r in field])])
    result["v4 như scripts/eval_gallery_value.py (PlantVillage 40 + mẫu ngoài đồng tối đa 120 mỗi lớp)"] = top1(v4_gallery, v4_labels, sets, len(classes))
    report = {
        "evidence": f"kNN k={K} (top-1, %) trên đặc trưng DINOv2-small đã lưu; không huấn luyện lớp nào",
        "release_manifest_sha256": hashlib.sha256((v4 / "release_manifest.json").read_bytes()).hexdigest(),
        "sets": {n: int(len(t)) for n, (_, t) in sets.items()},
        "gallery_field_images": {"in_sample": len(field_all), "used": len(field), "skipped_without_features": len(field_all) - len(field),
                                 "plantdoc_in_sample": sum(1 for r in field if r["source_dataset"] in PLANTDOC_NAMES),
                                 "by_source": dict(Counter(r["source_dataset"] for r in field))},
        "galleries": result, "add_one_source_to_v3": per_source,
        "caveat": "Chẩn đoán, không dùng chọn nguồn. v4_field_test cùng nguồn với thư viện nên chỉ đo nhận cùng nguồn; ext_* và pd_* là nguồn lạ.",
    }
    out = reg.abs_path(reg.DATASET_V4_ABLATION_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for label, scores in result.items():
        print(f"{label[:70]:70s} {scores}")
    for src, scores in per_source.items():
        print(f"  v3 + {src:40s} {scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

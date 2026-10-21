"""Đo giá trị của dữ liệu v4 KHÔNG cần huấn luyện: kNN trên đặc trưng đóng băng DINOv2-small, thư viện v3 so với thư viện v4.

Cùng các tập đo đã khóa với scripts/eval_foundation_models.py (PlantWild bệnh và khỏe, PlantDoc cắt kiểm tra và giữ riêng) cộng hai tập MỚI chỉ có ở v4:
  v4_field_test : ảnh ngoài đồng thuộc legacy_diagnostic_test của release v4 (cùng nguồn với dữ liệu học nhưng khác nhóm/cây)
  own_test      : ảnh tự chụp phần thử (nếu đã có)
Thư viện v3: lá PlantDoc trong tập học + 40 ảnh PlantVillage mỗi lớp (đặc trưng đã lưu ở data/reeval/foundation_features.npz).
Thư viện v4: mọi ảnh ngoài đồng trong train của release v4 (tối đa `--field-cap` mỗi lớp, chọn theo mã băm) + cùng 40 ảnh PlantVillage mỗi lớp.
Độ chính xác này đo MỨC DỮ LIỆU giúp ích (nếu thư viện v4 tốt hơn nhiều thì dữ liệu mới có giá trị), không phải mô hình huấn luyện của đồ án.

Cần môi trường có transformers/torch (tmp/venv-fm):
    tmp/venv-fm/Scripts/python.exe -X utf8 scripts/eval_gallery_value.py [--field-cap 120]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_foundation_models import (  # noqa: E402
    K_VALUES, build_items, crop_allowed, knn_scores, l2_normalize, read_gallery_features_v3, _jsonl,
)
from src import registry as reg  # noqa: E402

V4_DIR = "data/dataset_sources/mixed/pv_pd_v4"
GALLERY_NPZ = "data/reeval/gallery_v4_dinov2.npz"
OWN_MANIFEST = "data/own_photos/manifest.jsonl"


def select_field_gallery(rows: list[dict[str, Any]], cap: int, seed: int = 20261022) -> list[dict[str, Any]]:
    """Ảnh ngoài đồng (source_domain == plantdoc) của train, tối đa `cap` mỗi lớp, chọn theo mã băm cố định."""
    by_class: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if r["source_domain"] == "plantdoc":
            by_class.setdefault(int(r["class_id"]), []).append(r)
    out: list[dict[str, Any]] = []
    for cid in sorted(by_class):
        pool = sorted(by_class[cid], key=lambda r: hashlib.sha256(f"{seed}|{r['sample_id']}".encode()).hexdigest())
        out.extend(pool[:cap])
    return out


def dinov2_features(paths: list[Path], threads: int, batch: int = 32) -> np.ndarray:
    import importlib.util

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(max(1, threads))
    spec = importlib.util.spec_from_file_location("image_io_standalone", ROOT / "src" / "common" / "image_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    feats = []
    started = time.time()
    for start in range(0, len(paths), batch):
        images = []
        for p in paths[start:start + batch]:
            with Image.open(p) as handle:
                images.append(module.to_rgb(handle).copy())
        with torch.inference_mode():
            feats.append(model(**processor(images=images, return_tensors="pt")).last_hidden_state[:, 0].numpy())
        done = start + len(images)
        if done % (batch * 8) < batch or done == len(paths):
            rate = done / max(time.time() - started, 1e-9)
            print(f"[value] {done}/{len(paths)} ({rate:.1f} ảnh/s, còn ~{(len(paths) - done) / max(rate, 1e-9) / 60:.1f} phút)", flush=True)
    return np.concatenate(feats)


def summarize(scores: np.ndarray, targets: np.ndarray, oracle_scores: np.ndarray) -> dict[str, Any]:
    top1 = scores.argmax(1) == targets
    top3 = (np.argsort(-scores, axis=1)[:, :3] == targets[:, None]).any(1)
    return {"top1": round(100 * float(top1.mean()), 2), "top3": round(100 * float(top3.mean()), 2),
            "top1_oracle_crop": round(100 * float((oracle_scores.argmax(1) == targets).mean()), 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field-cap", type=int, default=120)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    sets, gallery_v3 = build_items(0)
    order = [(n, i) for n, items in sets.items() for i in items]
    n_eval = len(order)
    v3_feats = read_gallery_features_v3(n_eval, len(gallery_v3))  # (eval[n_eval, d], gallery_v3[m, d]) của DINOv2 đã lưu

    v4 = reg.abs_path(V4_DIR)
    root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    train = _jsonl(v4 / "manifests" / "train.jsonl")
    test = _jsonl(v4 / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    field_train = select_field_gallery(train, args.field_cap)
    pv_gallery_idx = [i for i, g in enumerate(gallery_v3) if g["source"] == "plantvillage"]
    v4_field_test = [r for r in test if r["source_domain"] == "plantdoc" and r["source_dataset"] not in ("PlantDoc", "plantdoc")]
    own_test = [r for r in v4_field_test if r["source_dataset"] == "own_photos"]
    v4_field_test = [r for r in v4_field_test if r["source_dataset"] != "own_photos"]

    # đặc trưng của ảnh mới (thư viện v4 + tập thử mới), lưu theo sample_id
    cache: dict[str, np.ndarray] = {}
    cache_path = reg.abs_path(GALLERY_NPZ)
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        cache = {str(k): v for k, v in zip(data["keys"], data["feats"])}
    need = [r for r in [*field_train, *v4_field_test, *own_test] if r["sample_id"] not in cache]
    print(f"[value] cần trích đặc trưng {len(need)} ảnh mới", flush=True)
    if need:
        feats = dinov2_features([root / r["relative_path"] for r in need], args.threads)
        cache.update({r["sample_id"]: f for r, f in zip(need, feats)})
        np.savez_compressed(cache_path, keys=np.array(list(cache)), feats=np.stack(list(cache.values())))

    eval_x, gal3 = v3_feats
    gal3_labels = np.array([g["target"] for g in gallery_v3])
    gal4 = np.concatenate([gal3[pv_gallery_idx], np.stack([cache[r["sample_id"]] for r in field_train])])
    gal4_labels = np.concatenate([gal3_labels[pv_gallery_idx], np.array([int(r["class_id"]) for r in field_train])])
    names = np.array([n for n, _ in order])
    targets = np.array([i["target"] for _, i in order])
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {n: (eval_x[names == n], targets[names == n]) for n in sets}
    if v4_field_test:
        eval_sets["v4_field_test"] = (np.stack([cache[r["sample_id"]] for r in v4_field_test]), np.array([int(r["class_id"]) for r in v4_field_test]))
    if own_test:
        eval_sets["own_test"] = (np.stack([cache[r["sample_id"]] for r in own_test]), np.array([int(r["class_id"]) for r in own_test]))

    results: dict[str, Any] = {}
    for set_name, (x, t) in eval_sets.items():
        oracle = crop_allowed(classes, None, t)
        entry: dict[str, Any] = {"n": int(len(t))}
        for label, gal, gal_labels in (("gallery_v3", gal3, gal3_labels), ("gallery_v4", gal4, gal4_labels)):
            entry[label] = {}
            for k in K_VALUES:
                free = knn_scores(x, gal, gal_labels, k, len(classes))
                restricted = knn_scores(x, gal, gal_labels, k, len(classes), allowed=oracle)
                entry[label][f"k{k}"] = summarize(free, t, restricted)
        results[set_name] = entry
    classes_with_field = {c: int(sum(1 for r in field_train if int(r["class_id"]) == i)) for i, c in enumerate(classes)}
    report = {
        "evidence": "kNN trên đặc trưng DINOv2-small đóng băng; không huấn luyện lớp nào",
        "release_manifest_sha256": hashlib.sha256((v4 / "release_manifest.json").read_bytes()).hexdigest(),
        "gallery": {"v3": {"plantvillage_per_class": 40, "plantdoc_train": int(len(gal3_labels) - len(pv_gallery_idx))},
                    "v4": {"plantvillage_per_class": 40, "field_images": len(field_train), "field_cap_per_class": args.field_cap,
                           "field_per_class": classes_with_field}},
        "k_values": list(K_VALUES), "sets": results,
        "caveat": "Thư viện v4 gồm cả ảnh cùng nguồn với tập 'v4_field_test' (khác nhóm/cây); tập đó đo khả năng nhận cùng nguồn, không phải ảnh nguồn lạ. "
                  "PlantWild và PlantDoc là nguồn không nằm trong thư viện: đo nguồn lạ.",
    }
    out = reg.abs_path(reg.DATASET_V4_VALUE_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for set_name, e in results.items():
        print(f"{set_name:14s} n={e['n']:5d} v3 k10 top1 {e['gallery_v3']['k10']['top1']:6.2f} -> v4 {e['gallery_v4']['k10']['top1']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

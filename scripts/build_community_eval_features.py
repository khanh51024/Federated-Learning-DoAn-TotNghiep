"""Tạo tập đặc trưng GIÁM SÁT cho cổng chấp nhận của học phân tán cộng đồng.

Máy chủ web không có ảnh, nên cổng "head mới không được kém head cũ" không thể chạy backbone. Script này (chạy trên máy có
dataset) trích đặc trưng 1024 chiều của toàn bộ ảnh validation của release đã ghim bằng đúng file ONNX mà trình duyệt dùng, rồi
lưu cùng nhãn. Trên máy chủ, đánh giá một head chỉ là một phép nhân ma trận.

Chạy (sau scripts/export_community_feature_model.py):
    python scripts/build_community_eval_features.py

Lưu ý khoa học: tập validation này đã được dùng để chọn mô hình gốc, nên độ chính xác tính trên nó chỉ là tín hiệu GIÁM SÁT của
hệ thống (theo dõi từng vòng), không được trộn với số công bố của bundle; số mức báo cáo lấy từ scripts/simulate_community.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402
from src.federated.community import (  # noqa: E402
    BIAS_KEY, FEATURE_DIM, SOURCE_PLANTDOC, SOURCE_PLANTVILLAGE, WEIGHT_KEY, EvalSet, evaluate_head, file_sha256,
)

IMAGE_TOLERANCE = 1  # chấp nhận lệch tối đa một ảnh so với số đã công bố (khác phần cứng/thư viện), giống reeval của repo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="Chỉ N ảnh đầu (thử nhanh; KHÔNG ghi manifest)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader

    from scripts.evaluate_published_models import ManifestImages, _load_contract_transforms, _verify_inputs
    from web.demo_evidence import load_evidence

    manifest_path = reg.abs_path(reg.COMMUNITY_MANIFEST_JSON)
    if not manifest_path.is_file():
        print("[community] chưa có community_manifest.json — chạy scripts/export_community_feature_model.py trước", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    onnx_path = reg.abs_path(reg.COMMUNITY_FEATURE_MODEL_ONNX)
    if file_sha256(onnx_path) != manifest["feature_model"]["sha256"]:
        print("[community] file ONNX khác SHA trong manifest — chạy lại export_community_feature_model.py", file=sys.stderr)
        return 1

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[community] bundle không hợp lệ: {evidence.get('reason')}", file=sys.stderr)
        return 1
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    if not (release / "manifests" / "val.jsonl").is_file():
        print(f"[community] thiếu dataset ở {reg.DATASET_SOURCES_DIR}", file=sys.stderr)
        return 1
    val_rows, _calibration = _verify_inputs(evidence, release, dataset_root)
    if args.limit:
        val_rows = val_rows[: args.limit]

    transform = _load_contract_transforms(reg.abs_path(reg.DEMO_BUNDLE_DIR))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    loader = DataLoader(ManifestImages(val_rows, dataset_root, transform), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    features = np.zeros((len(val_rows), FEATURE_DIM), dtype=np.float32)
    started, done = time.time(), 0
    with torch.inference_mode():
        for images, index in loader:
            features[index.numpy()] = session.run(None, {"input": images.numpy()})[0]
            done += len(index)
            if done % (args.batch_size * 10) < args.batch_size or done == len(val_rows):
                rate = done / max(time.time() - started, 1e-9)
                print(f"    {done}/{len(val_rows)} ảnh ({rate:.1f} ảnh/s)", flush=True)

    labels = np.array([row["class_id"] for row in val_rows], dtype=np.int64)
    sources = np.array(
        [SOURCE_PLANTDOC if str(row["source_dataset"]).lower() == "plantdoc" else SOURCE_PLANTVILLAGE for row in val_rows],
        dtype=np.int8,
    )
    # Máy chủ dùng float16 để tệp nhỏ (~9,5 MB): chấm điểm head khởi điểm trên chính bản đã làm tròn đó
    stored = features.astype(np.float16)
    evalset = EvalSet(stored.astype(np.float32), labels, sources)
    with np.load(reg.abs_path(reg.COMMUNITY_BASE_HEAD_NPZ), allow_pickle=False) as data:
        base_head = {WEIGHT_KEY: data["weight"].astype(np.float32), BIAS_KEY: data["bias"].astype(np.float32)}
    measured = evaluate_head(base_head, evalset)

    published = {m["model_id"]: m for m in evidence["models"]}["fedavg"]["metrics"]
    problems: list[str] = []
    for source, label in (("plantvillage", "PlantVillage"), ("plantdoc", "PlantDoc")):
        count = measured[f"{source}_count"]
        got = measured[f"{source}_accuracy"]
        want = float(published[source]["top1_accuracy"])
        if not args.limit and abs(got - want) * count > IMAGE_TOLERANCE + 1e-6:
            problems.append(f"{label}: đo {got:.4f} khác số công bố {want:.4f} quá {IMAGE_TOLERANCE} ảnh")
    print(f"[community] head khởi điểm trên {measured['count']} ảnh: acc {measured['accuracy']:.4f}, "
          f"PlantVillage {measured['plantvillage_accuracy']:.4f}, PlantDoc {measured['plantdoc_accuracy']:.4f}, "
          f"macro-recall {measured['macro_recall']:.4f}")
    if problems:
        print("[community] ĐẶC TRƯNG KHÔNG TÁI HIỆN SỐ ĐÃ CÔNG BỐ: " + "; ".join(problems), file=sys.stderr)
        return 1
    if args.limit:
        print("[community] --limit: không ghi file.")
        return 0

    out = reg.abs_path(reg.COMMUNITY_EVAL_FEATURES_NPZ)
    np.savez(out, features=stored, labels=labels, sources=sources)
    manifest["eval_features"] = {
        "file": out.name, "sha256": file_sha256(out), "bytes": out.stat().st_size, "count": int(len(labels)),
        "dtype": "float16", "split": "validation", "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "counts": {"plantvillage": measured["plantvillage_count"], "plantdoc": measured["plantdoc_count"]},
        "feature_model_sha256": manifest["feature_model"]["sha256"],
    }
    manifest["base_eval"] = {
        key: measured[key] for key in
        ("count", "accuracy", "macro_recall", "plantvillage_accuracy", "plantdoc_accuracy", "plantvillage_count", "plantdoc_count")
    }
    manifest["base_eval"]["published_reference"] = {
        source: {"count": int(published[source]["count"]), "top1_accuracy": float(published[source]["top1_accuracy"])}
        for source in ("plantvillage", "plantdoc")
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[community] đã ghi {out.name}: {out.stat().st_size / 1e6:.2f} MB, sha256 {manifest['eval_features']['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Trích đặc trưng 1024 chiều (phần đóng băng của FedAvg v7) cho thí nghiệm mô phỏng học phân tán cộng đồng.

Kho "người tham gia" là `calibration.jsonl` (không giao với train/validation theo group_id, đã kiểm). Tập báo cáo là test giữ riêng
`legacy_diagnostic_test.jsonl` (không dùng để chọn mô hình). Đặc trưng đi đường PyTorch (`.venv` không có onnxruntime) và được đối
chiếu với `eval_features.npz` (trích bằng ONNX, đúng đường của trình duyệt) trên 200 ảnh validation: cosine phải >= 0,999.

Chạy:
    python scripts/build_community_sim_features.py            # ghi data/community_sim/features.npz (gitignored)
    python scripts/build_community_sim_features.py --limit 64 # thử nhanh, không ghi
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_published_models as ev  # noqa: E402
from src import registry as reg  # noqa: E402
from src.federated.community import FEATURE_DIM, SOURCE_PLANTDOC, SOURCE_PLANTVILLAGE  # noqa: E402

OUT = "data/community_sim/features.npz"
PARITY_IMAGES = 200
MIN_COSINE = 0.999
SOURCE_CODE = {"plantvillage": SOURCE_PLANTVILLAGE, "plantdoc": SOURCE_PLANTDOC}


def extract(rows, dataset_root, transform, frozen, batch_size: int, workers: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ev.ManifestImages(rows, dataset_root, transform), batch_size=batch_size, shuffle=False, num_workers=workers)
    out = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    started, done = time.time(), 0
    with torch.inference_mode():
        for images, index in loader:
            out[index.numpy()] = frozen(images).numpy()
            done += len(index)
            if done % (batch_size * 10) < batch_size or done == len(rows):
                rate = done / max(time.time() - started, 1e-9)
                print(f"    {done}/{len(rows)} ảnh ({rate:.1f} ảnh/s, còn ~{(len(rows) - done) / max(rate, 1e-9) / 60:.1f} phút)", flush=True)
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    import torch

    from scripts.export_community_feature_model import build_frozen
    from web.demo_evidence import STORE, _published_model, load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[sim-features] bundle không hợp lệ: {evidence.get('reason')}", file=sys.stderr)
        return 1
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    val_rows, cal_rows = ev._verify_inputs(evidence, release, dataset_root)
    test_rows = ev._verify_holdout(evidence, release)
    test_sha = ev._sha256(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    if args.limit:
        cal_rows, test_rows = cal_rows[: args.limit], test_rows[: args.limit]

    torch.set_num_threads(max(1, min(10, torch.get_num_threads())))
    _info, model = _published_model(evidence, "fedavg", STORE)
    frozen = build_frozen(model)
    transform = ev._load_contract_transforms(reg.abs_path(reg.DEMO_BUNDLE_DIR))

    # đối chiếu với đặc trưng ONNX của trình duyệt (cùng ảnh validation, cùng thứ tự manifest)
    print(f"[sim-features] đối chiếu PyTorch ↔ ONNX trên {PARITY_IMAGES} ảnh validation")
    eval_path = reg.abs_path("artifacts/community_fl/eval_features.npz")
    with np.load(eval_path, allow_pickle=False) as data:
        onnx_features = data["features"][:PARITY_IMAGES].astype(np.float32)
    torch_features = extract(val_rows[:PARITY_IMAGES], dataset_root, transform, frozen, args.batch_size, args.workers)
    cosine = (torch_features * onnx_features).sum(1) / (np.linalg.norm(torch_features, axis=1) * np.linalg.norm(onnx_features, axis=1) + 1e-12)
    print(f"[sim-features] cosine thấp nhất {cosine.min():.6f}, trung bình {cosine.mean():.6f}")
    if cosine.min() < MIN_COSINE:
        print("[sim-features] đặc trưng PyTorch lệch ONNX quá ngưỡng — dừng", file=sys.stderr)
        return 1

    print(f"[sim-features] calibration ({len(cal_rows)} ảnh):")
    cal_x = extract(cal_rows, dataset_root, transform, frozen, args.batch_size, args.workers)
    print(f"[sim-features] test giữ riêng ({len(test_rows)} ảnh):")
    test_x = extract(test_rows, dataset_root, transform, frozen, args.batch_size, args.workers)

    def pack(rows):
        return (np.array([r["class_id"] for r in rows], dtype=np.int64),
                np.array([SOURCE_CODE[r["source_dataset"].lower()] for r in rows], dtype=np.int8))

    cal_y, cal_s = pack(cal_rows)
    test_y, test_s = pack(test_rows)
    print(f"[sim-features] calibration: {len(cal_rows)} ảnh, {len(set(cal_y.tolist()))} nhóm, PlantDoc {int((cal_s == SOURCE_PLANTDOC).sum())}")
    print(f"[sim-features] test: {len(test_rows)} ảnh, PlantDoc {int((test_s == SOURCE_PLANTDOC).sum())}")
    if args.limit:
        print("[sim-features] --limit: không ghi")
        return 0
    out = reg.abs_path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({
        "fedavg": next(m["checkpoint_sha256"] for m in evidence["models"] if m["model_id"] == "fedavg"),
        "calibration": evidence["release_manifest_sha256"], "test": test_sha, "n": [len(cal_rows), len(test_rows)],
    }, sort_keys=True).encode()).hexdigest()
    np.savez_compressed(out, fingerprint=fingerprint, cal_x=cal_x.astype(np.float16), cal_y=cal_y, cal_s=cal_s,
                        test_x=test_x.astype(np.float16), test_y=test_y, test_s=test_s,
                        parity_min_cosine=np.float32(cosine.min()), test_manifest_sha256=test_sha)
    print(f"[sim-features] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Xuất phần ĐÓNG BĂNG của FedAvg v7 (backbone + classifier[0..2]) sang ONNX cho học liên kết cộng đồng trong trình duyệt.

Trình duyệt chạy file ONNX này (onnxruntime-web, wasm) để biến ảnh thành vector đặc trưng 1024 chiều; chỉ lớp
`classifier.3` (38.950 tham số) được huấn luyện và gửi về máy chủ. Script này:

1. nạp checkpoint FedAvg v7 qua đường đã kiểm SHA-256 của `web.demo_evidence` (không đọc file chưa băm);
2. xuất ONNX (opset 17, batch động) và kiểm khớp đặc trưng PyTorch ↔ onnxruntime trên ảnh gallery;
3. ghi head khởi điểm (vòng 0) và `community_manifest.json` (SHA của mọi file, 38 tên lớp, kết quả kiểm khớp).

Chạy:
    python scripts/export_community_feature_model.py

Cần thêm gói `onnx` và `onnxruntime` (khai báo trong pyproject.toml, nhóm test). Máy chủ web KHÔNG cần hai gói này: chỉ
đọc file kết quả. Chạy trên máy có checkpoint, rồi chép thư mục artifacts/community_fl/ lên máy chủ.
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
    BIAS_KEY, FEATURE_DIM, NUM_CLASSES, WEIGHT_KEY, file_sha256, head_sha,
)

MIN_COSINE = 0.999
OPSET = 17


def build_frozen(model: Any) -> Any:
    """features -> avgpool -> flatten -> classifier[0] (Linear 576→1024) -> classifier[1] (Hardswish) -> classifier[2] (Dropout: eval = identity)."""
    import torch
    from torch import nn

    class Frozen(nn.Module):
        def __init__(self, source: nn.Module):
            super().__init__()
            self.features, self.avgpool = source.features, source.avgpool
            self.fc, self.act, self.drop = source.classifier[0], source.classifier[1], source.classifier[2]

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.drop(self.act(self.fc(torch.flatten(self.avgpool(self.features(x)), 1))))

    return Frozen(model).eval()


def parity_images(limit_errors: int = 30) -> list[Path]:
    gallery = sorted(reg.abs_path(reg.DEMO_GALLERY_DIR).glob("*.jpg"))
    errors = sorted(reg.abs_path(reg.ERROR_GALLERY_DIR).glob("sample_*.jpg"))[:limit_errors]
    return gallery + errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-parity", action="store_true", help="Bỏ bước so khớp PyTorch ↔ onnxruntime (không khuyến nghị)")
    args = parser.parse_args()

    import torch
    from PIL import Image, ImageOps

    from web.demo_evidence import STORE, _input_tensor, _published_model, load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[community] bundle không hợp lệ: {evidence.get('reason')}", file=sys.stderr)
        return 1
    model_info, model = _published_model(evidence, "fedavg", STORE)
    model.eval()
    frozen = build_frozen(model)
    head_layer = model.classifier[3]
    weight = head_layer.weight.detach().cpu().numpy().astype(np.float32)
    bias = head_layer.bias.detach().cpu().numpy().astype(np.float32)
    assert weight.shape == (NUM_CLASSES, FEATURE_DIM), weight.shape
    with torch.inference_mode():
        probe = torch.randn(2, 3, 224, 224)
        gap = float((head_layer(frozen(probe)) - model(probe)).abs().max())
    if gap > 1e-5:
        print(f"[community] head(frozen(x)) khác model(x): {gap:.3e}", file=sys.stderr)
        return 1

    out_dir = reg.abs_path(reg.COMMUNITY_FL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = reg.abs_path(reg.COMMUNITY_FEATURE_MODEL_ONNX)
    example = torch.randn(1, 3, 224, 224)
    export_kwargs: dict[str, Any] = dict(
        input_names=["input"], output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}}, opset_version=OPSET,
    )
    try:
        torch.onnx.export(frozen, example, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:  # torch cũ chưa có tham số `dynamo`
        torch.onnx.export(frozen, example, str(onnx_path), **export_kwargs)

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    parity: dict[str, Any] = {"checked": False}
    if not args.skip_parity:
        images = parity_images()
        if not images:
            print("[community] không có ảnh gallery để so khớp", file=sys.stderr)
            return 1
        cosines, max_diff, agree = [], 0.0, 0
        started = time.perf_counter()
        for path in images:
            with Image.open(path) as image:
                tensor = _input_tensor(ImageOps.exif_transpose(image).convert("RGB"))
            with torch.inference_mode():
                reference = frozen(tensor).numpy()[0]
            produced = session.run(None, {"input": tensor.numpy()})[0][0]
            cosines.append(float(reference @ produced / (np.linalg.norm(reference) * np.linalg.norm(produced) + 1e-12)))
            max_diff = max(max_diff, float(np.abs(reference - produced).max()))
            agree += int((weight @ reference + bias).argmax() == (weight @ produced + bias).argmax())
        parity = {
            "checked": True, "images": len(images), "min_cosine": min(cosines), "max_abs_feature_diff": max_diff,
            "argmax_agree": agree, "seconds": round(time.perf_counter() - started, 2),
        }
        if min(cosines) < MIN_COSINE or agree != len(images):
            print(f"[community] ONNX KHÔNG khớp PyTorch: {parity}", file=sys.stderr)
            return 1

    head = {WEIGHT_KEY: weight, BIAS_KEY: bias}
    head_path = reg.abs_path(reg.COMMUNITY_BASE_HEAD_NPZ)
    np.savez(head_path, weight=weight, bias=bias)

    manifest_path = reg.abs_path(reg.COMMUNITY_MANIFEST_JSON)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    previous_feature = (manifest.get("feature_model") or {}).get("sha256")
    manifest.update({
        "schema": 1,
        "base_model": {
            "model_id": reg.COMMUNITY_MODEL_ID,
            "checkpoint_sha256": model_info["checkpoint_sha256"],
            "protocol_id": evidence["protocol_id"],
            "release_manifest_sha256": evidence["release_manifest_sha256"],
            "validation_manifest_sha256": evidence["validation_manifest_sha256"],
            "transform": evidence["transform"],
        },
        "classes": list(evidence["classes"]),
        "feature_model": {
            "file": onnx_path.name, "sha256": file_sha256(onnx_path), "bytes": onnx_path.stat().st_size,
            "opset": OPSET, "input": "input", "output": "features", "dim": FEATURE_DIM, "transform": evidence["transform"],
        },
        "base_head": {"file": head_path.name, "sha256": file_sha256(head_path), "head_sha": head_sha(head)},
        "parity": parity,
    })
    if previous_feature and previous_feature != manifest["feature_model"]["sha256"]:
        # ONNX đổi => đặc trưng giám sát cũ không còn đúng: bắt buộc build lại
        manifest.pop("eval_features", None)
        manifest.pop("base_eval", None)
        print("[community] ONNX mới: đã xóa mục eval_features — chạy lại scripts/build_community_eval_features.py")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[community] ONNX {onnx_path.stat().st_size / 1e6:.2f} MB, sha256 {manifest['feature_model']['sha256'][:16]}…")
    print(f"[community] head khởi điểm {manifest['base_head']['head_sha'][:16]}…; kiểm khớp: {parity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

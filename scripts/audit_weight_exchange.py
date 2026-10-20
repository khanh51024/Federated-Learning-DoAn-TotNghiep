"""Kiểm trọng số đi qua lại giữa server và client có bị đổi / mất độ chính xác không.

Ba phép đo trên 7 checkpoint của bundle published (Centralized, FedAvg, 5 Local-only):

1. ĐƯỜNG TRUYỀN. Mỗi state_dict (244 tensor) đi qua đúng đường mà Flower dùng: ndarrays -> Parameters (np.save) -> protobuf
   (serde của Flower, đúng thứ ghi lên dây gRPC) -> ngược lại, rồi get_parameters/set_parameters của mô hình. So từng byte,
   dtype, shape, thứ tự khóa và SHA-256 `compute_model_hash`. Đếm byte thật trên dây.
2. LÀM TRÒN KHI GỘP. Gộp 5 mô hình Local-only theo n_k thật của FedAvg v7 bằng (a) float64 rồi làm tròn một lần về float32
   (đang dùng ở src/federated/strategy.py), (b) cộng thuần float32. Đo chênh trọng số (đơn vị ulp float32) và số ảnh validation
   đổi dự đoán. Đây là phép đo SỐ HỌC của phép gộp, không phải mô hình FedAvg thật.
3. ĐỘ NHẠY CỦA PHÉP KIỂM. Nếu trọng số bị nén khi truyền (float16, bfloat16, int8 mỗi tensor) thì bao nhiêu ảnh đổi dự đoán và
   độ chính xác đổi bao nhiêu. Chứng minh phép so byte ở mục 1 đủ nhạy để bắt lỗi, và cho biết nén an toàn tới đâu.

Ảnh validation (4.613) đọc đúng như lúc chấm của nhóm train (EXIF, RGBA -> nền đen, canonical_v1); cổng: dự đoán của FedAvg và
Centralized trên ảnh này phải trùng logits đã lưu ở data/reeval/logits.npz.

Chạy:
    python scripts/audit_weight_exchange.py            # đủ phép đo (vài phút CPU), ghi artifacts/weight_exchange_audit.json
    python scripts/audit_weight_exchange.py --limit 64 # thử nhanh, KHÔNG ghi artifacts/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

MODEL_ORDER = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
LOGITS_NPZ = "data/reeval/logits.npz"  # gitignored (data/*)


# --------------------------------------------------------------------------- phép tính thuần (có test)
def flower_wire_round_trip(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """ndarrays -> Parameters -> protobuf (bytes trên dây gRPC) -> Parameters -> ndarrays. Trả (mảng, số byte trên dây)."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.common.serde import parameters_from_proto, parameters_to_proto

    proto = parameters_to_proto(ndarrays_to_parameters(arrays))
    wire = proto.SerializeToString()
    proto_back = type(proto)()
    proto_back.ParseFromString(wire)
    return parameters_to_ndarrays(parameters_from_proto(proto_back)), len(wire)


def compare_arrays(before: list[np.ndarray], after: list[np.ndarray], keys: list[str]) -> dict[str, Any]:
    mismatched = [
        key for key, a, b in zip(keys, before, after)
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
    ]
    return {
        "tensors": len(before), "mismatched_tensors": mismatched,
        "all_bytes_equal": not mismatched and len(before) == len(after),
        "dtypes": sorted({a.dtype.str for a in before}),
    }


def ulp_distance_float32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Khoảng cách theo số ulp float32 giữa hai mảng (cùng dấu: hiệu của biểu diễn nguyên)."""
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-(2**31)) - ia, ia)  # đưa về thứ tự đơn điệu của số thực
    ib = np.where(ib < 0, np.int64(-(2**31)) - ib, ib)
    return np.abs(ia - ib)


def average_float64(tensors: list[np.ndarray], weights: list[int]) -> np.ndarray:
    total = float(sum(weights))
    acc = np.zeros(tensors[0].shape, dtype=np.float64)
    for t, n in zip(tensors, weights):
        acc += t.astype(np.float64) * n
    return (acc / total).astype(np.float32)


def average_float32(tensors: list[np.ndarray], weights: list[int]) -> np.ndarray:
    total = np.float32(sum(weights))
    acc = np.zeros(tensors[0].shape, dtype=np.float32)
    for t, n in zip(tensors, weights):
        acc += t.astype(np.float32) * np.float32(n)
    return acc / total


def quantize(array: np.ndarray, mode: str) -> np.ndarray:
    """Giả lập nén khi truyền rồi giải nén về float32. Tensor không phải float giữ nguyên."""
    if array.dtype.kind != "f":
        return array
    import torch

    t = torch.from_numpy(array)
    if mode == "float16":
        return t.half().float().numpy()
    if mode == "bfloat16":
        return t.bfloat16().float().numpy()
    if mode == "int8_per_tensor":
        scale = float(np.abs(array).max()) / 127.0 or 1.0
        return (np.clip(np.round(array / scale), -127, 127) * scale).astype(np.float32)
    raise ValueError(mode)


# --------------------------------------------------------------------------- dữ liệu / mô hình
def _load_val_uint8(limit: int, workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ảnh validation đã resize 224x224 (uint8, chưa chuẩn hóa), nhãn, nguồn. Bít-chính-xác với transform canonical_v1."""
    from PIL import Image
    from torchvision import transforms as T

    from scripts.evaluate_published_models import _jsonl
    from src.common.image_io import to_rgb

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    rows = _jsonl(release / "manifests" / "val.jsonl")
    if limit:
        rows = rows[:limit]
    resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True)

    def load(row: dict[str, Any]) -> np.ndarray:
        with Image.open(root / row["relative_path"]) as handle:
            return np.asarray(resize(to_rgb(handle)), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(pool.map(load, rows))
    return (np.stack(images), np.array([r["class_id"] for r in rows], dtype=np.int64),
            np.array([r["source_domain"] for r in rows]))


def _logits(model: Any, images: np.ndarray, batch: int = 128) -> np.ndarray:
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            x = torch.from_numpy(images[start:start + batch]).permute(0, 3, 1, 2).float() / 255.0
            out.append(model((x - mean) / std).numpy())
    return np.concatenate(out)


def _model_with(state: dict[str, np.ndarray]) -> Any:
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}, strict=True)
    return model.eval()


def _scores(logits: np.ndarray, targets: np.ndarray, sources: np.ndarray) -> dict[str, Any]:
    pred = logits.argmax(axis=1)
    out = {"top1": round(100.0 * float((pred == targets).mean()), 3)}
    for src in ("plantvillage", "plantdoc"):
        mask = sources == src
        if mask.any():
            out[src] = round(100.0 * float((pred[mask] == targets[mask]).mean()), 3)
    return out


# --------------------------------------------------------------------------- báo cáo
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="chỉ N ảnh validation đầu (thử nhanh, không ghi artifacts/)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    from scripts.evaluate_published_models import _load_model
    from src.model.mobilenetv3 import compute_model_hash, get_parameters, set_parameters
    from web.demo_evidence import load_evidence

    torch.set_num_threads(max(1, args.threads))
    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[audit] bundle chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    paths = {"centralized": bundle / "centralized" / "checkpoint_best.pt", "fedavg": bundle / "fedavg" / "checkpoint_best.pt",
             **{f"local_{i:02d}": bundle / "local" / f"client_{i:02d}" / "checkpoint_best.pt" for i in range(5)}}
    models = {name: _load_model(path) for name, path in paths.items()}
    states = {name: {k: v.detach().cpu().numpy().copy() for k, v in m.state_dict().items()} for name, m in models.items()}
    keys = list(states["fedavg"])

    # ---- 1. đường truyền
    transport: dict[str, Any] = {}
    for name in MODEL_ORDER:
        arrays = [states[name][k] for k in keys]
        back, wire_bytes = flower_wire_round_trip(arrays)
        result = compare_arrays(arrays, back, keys)
        before_hash = compute_model_hash(dict(zip(keys, arrays)))
        after_hash = compute_model_hash(dict(zip(keys, back)))
        fresh = _model_with({k: v for k, v in zip(keys, back)})
        restored = get_parameters(fresh)
        exact_set_get = all(torch.equal(restored[k], torch.from_numpy(states[name][k])) for k in keys)
        probe = _model_with(states[name])
        set_parameters(probe, {k: torch.from_numpy(v.copy()) for k, v in zip(keys, back)})
        reloaded_hash = compute_model_hash(get_parameters(probe))
        result.update({
            "sha256_before": before_hash, "sha256_after_wire": after_hash, "sha256_equal": before_hash == after_hash == reloaded_hash,
            "set_get_parameters_exact": bool(exact_set_get),
            "raw_tensor_bytes": int(sum(a.nbytes for a in arrays)), "wire_bytes_protobuf": int(wire_bytes),
            "wire_overhead_bytes": int(wire_bytes - sum(a.nbytes for a in arrays)),
        })
        transport[name] = result
        print(f"[audit] đường truyền {name}: bytes_equal={result['all_bytes_equal']} sha_equal={result['sha256_equal']}", flush=True)

    # ---- dữ liệu để đo dự đoán
    started = time.time()
    images, targets, sources = _load_val_uint8(args.limit, workers=8)
    print(f"[audit] {len(images)} ảnh validation sẵn sàng ({time.time() - started:.0f}s)", flush=True)
    base_logits = {n: _logits(models[n], images) for n in ("centralized", "fedavg")}
    gate: dict[str, Any] = {}
    saved = reg.abs_path(LOGITS_NPZ)
    if saved.is_file() and not args.limit:
        z = np.load(saved)
        for n in ("centralized", "fedavg"):
            same = int((base_logits[n].argmax(1) == z[f"val_{n}"].argmax(1)).sum())
            gate[n] = {"same_prediction": same, "of": len(images), "max_abs_logit_diff": float(np.abs(base_logits[n] - z[f"val_{n}"]).max())}
    print("[audit] cổng tái lập so với logits đã lưu:", gate, flush=True)
    if gate and any(g["same_prediction"] != g["of"] for g in gate.values()):
        print("[audit] cổng tái lập KHÔNG đạt (dự đoán khác logits đã lưu): dừng, không ghi artifacts/", file=sys.stderr)
        return 2

    # ---- 2. làm tròn khi gộp
    n_k = [int(json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["clients"][f"client_{i:02d}"]["n_k"])
           for i in range(5)]
    locals_ = [f"local_{i:02d}" for i in range(5)]
    agg64: dict[str, np.ndarray] = {}
    agg32: dict[str, np.ndarray] = {}
    ulp_stats: dict[str, Any] = {"max_ulp": 0, "tensors_with_any_diff": 0, "elements": 0, "elements_diff": 0}
    for k in keys:
        tensors = [states[n][k] for n in locals_]
        if tensors[0].dtype.kind != "f":
            agg64[k] = agg32[k] = tensors[0]
            continue
        agg64[k], agg32[k] = average_float64(tensors, n_k), average_float32(tensors, n_k)
        ulps = ulp_distance_float32(agg64[k], agg32[k])
        ulp_stats["elements"] += int(ulps.size)
        ulp_stats["elements_diff"] += int((ulps > 0).sum())
        ulp_stats["max_ulp"] = max(ulp_stats["max_ulp"], int(ulps.max()))
        ulp_stats["tensors_with_any_diff"] += int(ulps.max() > 0)
    l64, l32 = _logits(_model_with(agg64), images), _logits(_model_with(agg32), images)
    rounding = {
        "clients": locals_, "n_k": n_k, "float_tensors": sum(1 for k in keys if states["fedavg"][k].dtype.kind == "f"),
        **ulp_stats,
        "elements_diff_share": round(ulp_stats["elements_diff"] / max(1, ulp_stats["elements"]), 5),
        "max_abs_weight_diff": float(max(np.abs(agg64[k].astype(np.float64) - agg32[k].astype(np.float64)).max()
                                          for k in keys if agg64[k].dtype.kind == "f")),
        "images_changed_prediction": int((l64.argmax(1) != l32.argmax(1)).sum()), "of": len(images),
        "max_abs_logit_diff": float(np.abs(l64 - l32).max()),
        "scores_float64_accumulate": _scores(l64, targets, sources), "scores_float32_accumulate": _scores(l32, targets, sources),
        "note": "Gộp 5 mô hình Local-only theo n_k của FedAvg v7: phép đo số học, KHÔNG phải mô hình FedAvg thật.",
    }
    print("[audit] làm tròn khi gộp:", {k: rounding[k] for k in ("max_ulp", "elements_diff_share", "images_changed_prediction")}, flush=True)

    # ---- 3. độ nhạy: nếu trọng số bị nén khi truyền
    sensitivity: dict[str, Any] = {}
    for name in ("fedavg", "centralized"):
        sensitivity[name] = {"baseline": _scores(base_logits[name], targets, sources)}
        for mode in ("float16", "bfloat16", "int8_per_tensor"):
            q = {k: quantize(states[name][k], mode) for k in keys}
            logits = _logits(_model_with(q), images)
            diffs = [float(np.abs(q[k].astype(np.float64) - states[name][k].astype(np.float64)).max())
                     for k in keys if states[name][k].dtype.kind == "f"]
            sensitivity[name][mode] = {
                "max_abs_weight_diff": max(diffs),
                "images_changed_prediction": int((logits.argmax(1) != base_logits[name].argmax(1)).sum()), "of": len(images),
                "scores": _scores(logits, targets, sources),
                "wire_bytes_ratio": {"float16": 0.5, "bfloat16": 0.5, "int8_per_tensor": 0.25}[mode],
            }
            print(f"[audit] nén {mode} {name}: đổi {sensitivity[name][mode]['images_changed_prediction']}/{len(images)} ảnh", flush=True)

    report = {
        "evidence": "đo cục bộ trên CPU từ checkpoint đã kiểm hash của bundle published",
        "protocol_id": evidence["protocol_id"],
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "flwr_version": __import__("flwr").__version__,
        "validation_images": int(len(images)),
        "baseline_reproduces_saved_logits": gate,
        "transport_round_trip": transport,
        "all_transport_exact": all(r["all_bytes_equal"] and r["sha256_equal"] and r["set_get_parameters_exact"] for r in transport.values()),
        "aggregation_rounding": rounding,
        "lossy_transport_sensitivity": sensitivity,
        "seconds": round(time.time() - started, 1),
    }
    if args.limit:
        print(json.dumps({"all_transport_exact": report["all_transport_exact"]}))
        print("[audit] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.WEIGHT_EXCHANGE_AUDIT_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[audit] đã ghi {reg.WEIGHT_EXCHANGE_AUDIT_JSON}; all_transport_exact={report['all_transport_exact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""So sánh trọng số và hành vi giữa Centralized, FedAvg và các Local-only đã công bố.

Trả lời: FedAvg lệch Centralized bao nhiêu (theo từng nhóm lớp), mỗi mô hình đã rời khỏi trọng số ImageNet bao xa, hai mô hình
nhìn ảnh có giống nhau không (CKA trên đặc trưng 1024 chiều, tỉ lệ dự đoán trùng, KL), và nội suy trọng số giữa hai mô hình
(chỉ là phép thăm dò: BatchNorm không được tính lại, xem `caveat` trong báo cáo).

Cảnh báo đọc số (ghi cả vào báo cáo): Centralized dùng AdamW lr 1e-3, FedAvg dùng SGD lr 0,01 không momentum; hai mô hình KHÔNG
chỉ khác nhau ở chỗ có/không liên kết. Khoảng cách trọng số vì vậy là cận trên của phần do học phân tán gây ra.

Chạy:
    python scripts/compare_model_weights.py            # ghi artifacts/model_weight_comparison.json (vài phút CPU)
    python scripts/compare_model_weights.py --limit 64 # thử nhanh, KHÔNG ghi artifacts/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

MODEL_ORDER = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
LOGITS_NPZ = "data/reeval/logits.npz"  # gitignored (data/*)
IMAGENET_FILE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
IMAGENET_SHA_PREFIX = "047dcff4"  # torchvision ghi 8 ký tự đầu SHA-256 vào tên file
INTERPOLATION = (0.0, 0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------------- phép tính thuần numpy (có test)
def is_bn_stat(name: str) -> bool:
    return name.endswith(("running_mean", "running_var", "num_batches_tracked"))


def layer_group(name: str) -> str:
    """Nhóm lớp: features thấp (khối 0-3), giữa (4-8), cao (9-12) và classifier đầu/cuối."""
    match = re.match(r"features\.(\d+)\.", name)
    if match:
        block = int(match.group(1))
        return "features_low_0_3" if block <= 3 else "features_mid_4_8" if block <= 8 else "features_high_9_12"
    if name.startswith("classifier.0"):
        return "classifier_hidden_576_to_1024"
    if name.startswith("classifier.3"):
        return "classifier_output_1024_to_38"
    return "other"


def vector_distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    diff = float(np.linalg.norm(a64 - b64))
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    cos = float(a64 @ b64 / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return {"l2": diff, "relative": diff / max(na, nb, 1e-30), "cosine": cos, "norm_a": na, "norm_b": nb}


def grouped_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray], skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Khoảng cách theo nhóm lớp, tách trọng số học được khỏi thống kê BatchNorm (running_mean/var)."""
    out: dict[str, Any] = {}
    buckets: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for name in a:
        if a[name].dtype.kind != "f" or any(name.startswith(s) for s in skip):
            continue
        kind = "bn_stats" if is_bn_stat(name) else "learnable"
        for key in (f"{kind}:all", f"{kind}:{layer_group(name)}"):
            buckets.setdefault(key, ([], []))
            buckets[key][0].append(a[name].ravel())
            buckets[key][1].append(b[name].ravel())
    for key, (xs, ys) in sorted(buckets.items()):
        out[key] = vector_distance(np.concatenate(xs), np.concatenate(ys))
    return out


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith 2019) giữa hai ma trận đặc trưng [n, d1], [n, d2] của cùng n ảnh."""
    x = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.astype(np.float64).mean(axis=0, keepdims=True)
    cross = np.linalg.norm(y.T @ x, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / denom) if denom > 0 else float("nan")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    return float(0.5 * (np.sum(p * np.log((p + eps) / (q + eps)), axis=1) + np.sum(q * np.log((q + eps) / (p + eps)), axis=1)).mean())


def interpolate(a: dict[str, np.ndarray], b: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    out = {}
    for name in a:
        if a[name].dtype.kind == "f":
            out[name] = ((1.0 - alpha) * a[name].astype(np.float64) + alpha * b[name].astype(np.float64)).astype(np.float32)
        else:
            out[name] = a[name] if alpha < 0.5 else b[name]  # num_batches_tracked: bộ đếm, không nội suy
    return out


# --------------------------------------------------------------------------- mô hình
def _features_and_logits(model: Any, images: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats, outs = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            x = torch.from_numpy(images[start:start + batch]).permute(0, 3, 1, 2).float() / 255.0
            x = model.avgpool(model.features((x - mean) / std)).flatten(1)
            f = model.classifier[:3](x)  # 1024 chiều, sau Hardswish; Dropout ở eval là phép đồng nhất
            feats.append(f.numpy())
            outs.append(model.classifier[3](f).numpy())
    return np.concatenate(feats), np.concatenate(outs)


def _imagenet_state() -> tuple[dict[str, np.ndarray] | None, str]:
    import torch

    if not IMAGENET_FILE.is_file():
        return None, "missing"
    digest = hashlib.sha256(IMAGENET_FILE.read_bytes()).hexdigest()
    if not digest.startswith(IMAGENET_SHA_PREFIX):
        return None, f"sha_mismatch:{digest[:8]}"
    state = torch.load(IMAGENET_FILE, map_location="cpu", weights_only=True)
    return {k: v.numpy() for k, v in state.items()}, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="chỉ N ảnh validation đầu (thử nhanh, không ghi artifacts/)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    from scripts.audit_weight_exchange import _load_val_uint8, _model_with
    from scripts.evaluate_published_models import _load_model
    from web.demo_evidence import load_evidence

    torch.set_num_threads(max(1, args.threads))
    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[compare] bundle chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    paths = {"centralized": bundle / "centralized" / "checkpoint_best.pt", "fedavg": bundle / "fedavg" / "checkpoint_best.pt",
             **{f"local_{i:02d}": bundle / "local" / f"client_{i:02d}" / "checkpoint_best.pt" for i in range(5)}}
    models = {name: _load_model(path) for name, path in paths.items()}
    states = {n: {k: v.detach().cpu().numpy().copy() for k, v in m.state_dict().items()} for n, m in models.items()}

    # ---- 1. khoảng cách trọng số
    pairs = [("centralized", "fedavg")] + [("centralized", f"local_{i:02d}") for i in range(5)] + [("fedavg", f"local_{i:02d}") for i in range(5)]
    weight_distance = {f"{a}__{b}": grouped_distance(states[a], states[b]) for a, b in pairs}
    imagenet, imagenet_digest = _imagenet_state()
    from_imagenet: dict[str, Any] = {"sha256": imagenet_digest, "note": "Bỏ classifier.3 (đầu 1000 lớp ImageNet khác đầu 38 lớp)."}
    if imagenet is not None:
        for name in MODEL_ORDER:
            from_imagenet[name] = grouped_distance(states[name], imagenet, skip=("classifier.3",))
    print("[compare] khoảng cách trọng số xong", flush=True)

    # ---- 2. hành vi trên ảnh validation
    started = time.time()
    images, targets, sources = _load_val_uint8(args.limit, workers=8)
    feats, logits = {}, {}
    for name in MODEL_ORDER:
        feats[name], logits[name] = _features_and_logits(models[name], images)
        print(f"[compare] đặc trưng {name} ({time.time() - started:.0f}s)", flush=True)
    saved = reg.abs_path(LOGITS_NPZ)
    gate: dict[str, Any] = {}
    if saved.is_file() and not args.limit:
        z = np.load(saved)
        for n in ("centralized", "fedavg"):
            gate[n] = {"same_prediction": int((logits[n].argmax(1) == z[f"val_{n}"].argmax(1)).sum()), "of": len(images)}
    if gate and any(g["same_prediction"] != g["of"] for g in gate.values()):
        print("[compare] cổng tái lập KHÔNG đạt (dự đoán khác logits đã lưu): dừng, không ghi artifacts/", file=sys.stderr)
        return 2
    preds = {n: logits[n].argmax(1) for n in MODEL_ORDER}
    probs = {n: softmax(logits[n]) for n in MODEL_ORDER}
    behaviour: dict[str, Any] = {}
    for a, b in pairs:
        row: dict[str, Any] = {
            "cka_features_1024": round(linear_cka(feats[a], feats[b]), 5),
            "prediction_agreement": round(float((preds[a] == preds[b]).mean()), 5),
            "symmetric_kl": round(symmetric_kl(probs[a], probs[b]), 5),
        }
        for src in ("plantvillage", "plantdoc"):
            mask = sources == src
            if mask.any():
                row[f"prediction_agreement_{src}"] = round(float((preds[a][mask] == preds[b][mask]).mean()), 5)
        behaviour[f"{a}__{b}"] = row
    classes = evidence["classes"]
    recall_gap = []
    for cid, name in enumerate(classes):
        mask = targets == cid
        if mask.any():
            rc, rf = float((preds["centralized"][mask] == cid).mean()), float((preds["fedavg"][mask] == cid).mean())
            recall_gap.append({"class": name, "n": int(mask.sum()), "recall_centralized": round(rc, 4), "recall_fedavg": round(rf, 4),
                               "gap_points": round(100 * (rc - rf), 2)})
    recall_gap.sort(key=lambda r: -abs(r["gap_points"]))

    # ---- 3. nội suy trọng số Centralized -> FedAvg (tập con: toàn bộ PlantDoc + mẫu PlantVillage cố định)
    rng = np.random.default_rng(20261020)
    pd_idx = np.flatnonzero(sources == "plantdoc")
    pv_idx = np.flatnonzero(sources == "plantvillage")
    pick = np.concatenate([pd_idx, rng.choice(pv_idx, size=min(731, len(pv_idx)), replace=False)]) if len(pv_idx) else pd_idx
    path_scores = []
    for alpha in INTERPOLATION:
        model = _model_with(interpolate(states["centralized"], states["fedavg"], alpha))
        _, out = _features_and_logits(model, images[pick])
        hit = out.argmax(1) == targets[pick]
        entry: dict[str, Any] = {"alpha_toward_fedavg": alpha, "top1": round(100 * float(hit.mean()), 2)}
        for src in ("plantvillage", "plantdoc"):
            mask = sources[pick] == src
            if mask.any():
                entry[src] = round(100 * float(hit[mask].mean()), 2)
        path_scores.append(entry)
        print(f"[compare] nội suy alpha={alpha}: {entry}", flush=True)

    report = {
        "evidence": "đo cục bộ trên CPU từ checkpoint đã kiểm hash của bundle published",
        "protocol_id": evidence["protocol_id"],
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "validation_images": int(len(images)),
        "baseline_reproduces_saved_logits": gate,
        "caveat": "Centralized: AdamW lr 1e-3 (83 epoch, chọn epoch 22). FedAvg: SGD lr 0,01 không momentum (32 vòng, chọn vòng 27). "
                  "Khoảng cách trọng số gồm cả phần do khác bộ tối ưu, không chỉ do học phân tán.",
        "weight_distance": weight_distance,
        "distance_from_imagenet": from_imagenet,
        "behaviour_on_validation": behaviour,
        "recall_gap_centralized_minus_fedavg": recall_gap,
        "interpolation_centralized_to_fedavg": {
            "images": int(len(pick)), "scores": path_scores,
            "caveat": "Thống kê BatchNorm (running mean/var) được nội suy tuyến tính, KHÔNG tính lại theo từng alpha, nên rào cản ở giữa đường "
                      "có thể bị phóng đại; không dùng để khẳng định hai mô hình cùng hay khác một thung lũng.",
        },
        "seconds": round(time.time() - started, 1),
    }
    if args.limit:
        print(json.dumps({"pairs": len(weight_distance), "interp": path_scores}, ensure_ascii=False)[:600])
        print("[compare] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.MODEL_WEIGHT_COMPARISON_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[compare] đã ghi {reg.MODEL_WEIGHT_COMPARISON_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

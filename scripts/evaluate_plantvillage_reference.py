"""Đánh giá lại mô hình PlantVillage tham chiếu (KHÔNG huấn luyện lại) trên cùng tập kiểm tra của đồ án.

Mô hình `artifacts/checkpoints/plantvillage_traced.pt` không do đồ án huấn luyện: đó là
`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (Hugging Face, MobileNetV2) đã được `torch.jit.trace`.
Bộ huấn luyện của tác giả là bản Kaggle "New Plant Diseases Dataset", một bản tăng cường của chính PlantVillage, nên số đo trên ảnh
PlantVillage của đồ án CÓ THỂ BỊ RÒ RỈ (không kiểm được bằng băm byte vì ảnh đã được mã hóa lại). Chỉ PlantDoc (mô hình chưa từng học)
là phép thử công bằng. Báo cáo ghi rõ điều này và không dùng số PlantVillage để so sánh với FedAvg.

Quy trình (bám `scripts/evaluate_published_models.py`): manifest val / calibration / test giữ riêng của release pv_pd_v3 đã kiểm hash,
EXIF transpose, RGBA -> nền đen, softmax; nhiệt độ T học trên calibration, KHÔNG học trên val/test.

Chạy:
    python scripts/evaluate_plantvillage_reference.py            # suy luận (vài phút-chục phút, CPU) rồi ghi báo cáo
    python scripts/evaluate_plantvillage_reference.py --reuse    # dùng logits đã lưu ở data/reeval/ nếu còn khớp
    python scripts/evaluate_plantvillage_reference.py --limit 64 # thử nhanh (KHÔNG ghi artifacts/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_published_models as ev  # noqa: E402
from scripts.report_statistics import cluster_bootstrap_ci, mcnemar_exact  # noqa: E402
from src import registry as reg  # noqa: E402

REPORT_JSON = "artifacts/plantvillage_reference_eval.json"
LOGITS_NPZ = "data/reeval/plantvillage_logits.npz"
SOURCES = ev.SOURCES
HF_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
HF_SNAPSHOT = "c1861579a670fb6232258805b801cd4137cb7176"
SEED = 20261020
BOOTSTRAP = 2000

PREPROCESSINGS = {
    # cách web dùng trước đây (get_default_transform): chuẩn hóa ImageNet
    "imagenet_256_224": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    # cách của tác giả mô hình (preprocessor_config.json): chuẩn hóa 0,5
    "hf_half_256_224": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}
CHOSEN = "hf_half_256_224"


def make_transform(name: str):
    from torchvision import transforms as T

    spec = PREPROCESSINGS[name]
    return T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize(spec["mean"], spec["std"])])


class _Wrapped:
    """Cho `ev.collect_logits` một callable trả logits [N, 38] từ mô hình TorchScript."""

    def __init__(self, module):
        self.module = module

    def __call__(self, images):
        return self.module(images)


def _stats(values: np.ndarray, clusters: list[str], rng: np.random.Generator) -> dict[str, Any]:
    return cluster_bootstrap_ci(clusters, values.astype(np.float64), rng, BOOTSTRAP)


def evaluate_split(logits: np.ndarray, rows, cal_logits: np.ndarray, cal_targets: np.ndarray, classes: list[str],
                   rng: np.random.Generator) -> dict[str, Any]:
    targets = np.array([r["class_id"] for r in rows], dtype=np.int64)
    sources = np.array([r["source_dataset"].lower() for r in rows])
    result = ev.analyze_model(logits, targets, sources, cal_logits, cal_targets, classes)
    preds = logits.argmax(axis=1)
    correct = preds == targets
    for source in SOURCES:
        mask = sources == source
        if not mask.any():
            continue
        clusters = [rows[i]["group_id"] for i in np.flatnonzero(mask)]
        result["by_source"][source]["top1_ci95"] = _stats(correct[mask], clusters, rng)
    return result


def compare_with_fedavg(rows, pv_logits: np.ndarray, fedavg_logits: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    targets = np.array([r["class_id"] for r in rows], dtype=np.int64)
    sources = np.array([r["source_dataset"].lower() for r in rows])
    out: dict[str, Any] = {}
    for source in SOURCES:
        mask = sources == source
        if not mask.any():
            continue
        pv = pv_logits[mask].argmax(axis=1) == targets[mask]
        fed = fedavg_logits[mask].argmax(axis=1) == targets[mask]
        out[source] = {
            "count": int(mask.sum()),
            "plantvillage_model_top1": float(pv.mean()),
            "fedavg_top1": float(fed.mean()),
            "mcnemar": mcnemar_exact(pv, fed),
        }
    return out


def _fedavg_holdout_logits(evidence: dict[str, Any], manifest_sha: str, count: int) -> np.ndarray | None:
    path = reg.abs_path("data/reeval/holdout_logits.npz")
    if not path.is_file():
        return None
    checkpoints = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}
    fingerprint = hashlib.sha256(json.dumps([manifest_sha, count, checkpoints], sort_keys=True).encode()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint:
            return None
        return data["test_fedavg"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    import torch

    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[pv-eval] bundle không hợp lệ: {evidence.get('reason')}", file=sys.stderr)
        return 1
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    classes = evidence["classes"]
    val_rows, cal_rows = ev._verify_inputs(evidence, release, dataset_root)
    test_rows = ev._verify_holdout(evidence, release)
    test_sha = ev._sha256(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    if args.limit:
        val_rows, cal_rows, test_rows = val_rows[: args.limit], cal_rows[: args.limit], test_rows[: args.limit]

    model_path = reg.abs_path(reg.PLANTVILLAGE_TRACED_PT)
    model_sha = ev._sha256(model_path)
    if model_sha != reg.PLANTVILLAGE_TRACED_SHA256:
        print("[pv-eval] checkpoint không khớp SHA-256 đã ghim", file=sys.stderr)
        return 1

    fingerprint = hashlib.sha256(json.dumps(
        [model_sha, evidence["validation_manifest_sha256"], test_sha, len(val_rows), len(cal_rows), len(test_rows),
         PREPROCESSINGS, CHOSEN], sort_keys=True).encode()).hexdigest()
    cache = reg.abs_path(LOGITS_NPZ)
    logits: dict[str, np.ndarray] | None = None
    if args.reuse and cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) == fingerprint:
                logits = {k: data[k] for k in data.files if k != "fingerprint"}
                print("[pv-eval] dùng lại logits đã lưu")
    if logits is None:
        torch.set_num_threads(max(1, min(10, torch.get_num_threads())))
        module = torch.jit.load(str(model_path), map_location="cpu").eval()
        models = {"pv": _Wrapped(module)}
        logits = {}
        # so hai cách chuẩn hóa chỉ trên val (đủ để chọn cách đúng); các tập còn lại chạy cách đã chọn
        for name in PREPROCESSINGS:
            print(f"[pv-eval] val với chuẩn hóa {name} ({len(val_rows)} ảnh):")
            logits[f"val_{name}"] = ev.collect_logits(val_rows, dataset_root, make_transform(name), models,
                                                      args.batch_size, args.workers)["pv"]
        transform = make_transform(CHOSEN)
        print(f"[pv-eval] calibration ({len(cal_rows)} ảnh):")
        logits["cal"] = ev.collect_logits(cal_rows, dataset_root, transform, models, args.batch_size, args.workers)["pv"]
        print(f"[pv-eval] test giữ riêng ({len(test_rows)} ảnh):")
        logits["test"] = ev.collect_logits(test_rows, dataset_root, transform, models, args.batch_size, args.workers)["pv"]
        if not args.limit:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, fingerprint=fingerprint, **logits)

    rng = np.random.default_rng(SEED)
    cal_targets = np.array([r["class_id"] for r in cal_rows], dtype=np.int64)
    val_targets = np.array([r["class_id"] for r in val_rows], dtype=np.int64)
    comparison = {}
    for name in PREPROCESSINGS:
        top1 = float((logits[f"val_{name}"].argmax(axis=1) == val_targets).mean())
        comparison[name] = {"val_top1": round(top1, 5), **PREPROCESSINGS[name]}
    val_result = evaluate_split(logits[f"val_{CHOSEN}"], val_rows, logits["cal"], cal_targets, classes, rng)
    test_result = evaluate_split(logits["test"], test_rows, logits["cal"], cal_targets, classes, rng)

    fed = _fedavg_holdout_logits(evidence, test_sha, len(test_rows))
    versus = compare_with_fedavg(test_rows, logits["test"], fed, rng) if fed is not None and not args.limit else None

    report = {
        "evidence": ("đánh giá lại cục bộ trên CPU, không huấn luyện lại; ảnh và manifest đã kiểm hash theo release pv_pd_v3; "
                     "nhiệt độ T học trên calibration"),
        "model": {
            "registry_id": "plantvillage_traced_reference",
            "origin": f"Hugging Face {HF_ID}",
            "hf_snapshot": HF_SNAPSHOT,
            "architecture": "MobileNetV2 (transformers MobileNetV2ForImageClassification), 38 lớp",
            "trained_by_project": False,
            "training_data": "Kaggle 'New Plant Diseases Dataset' (bản tăng cường của PlantVillage), theo model card",
            "published_accuracy_note": "model card tự báo 0,9541 trên tập đánh giá riêng của tác giả; không dùng ở đây",
            "checkpoint": model_path.name,
            "checkpoint_sha256": model_sha,
            "class_order": "id2label 0..37 cùng thứ tự với taxonomy của đồ án (kiểm bằng độ chính xác theo lớp)",
        },
        "preprocessing_chosen": CHOSEN,
        "preprocessing_comparison_on_val": comparison,
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "holdout_manifest_sha256": test_sha,
        "validation_count": len(val_rows),
        "calibration_count": len(cal_rows),
        "holdout_count": len(test_rows),
        "classes": classes,
        "leakage": {
            "plantvillage": "likely",
            "plantdoc": "none",
            "note": ("Tập huấn luyện của tác giả là bản tăng cường của chính PlantVillage nên ảnh PlantVillage trong val/test của đồ án "
                     "gần như chắc nằm trong dữ liệu mô hình đã học (ảnh đã mã hóa lại, băm byte không phát hiện được). "
                     "Số PlantVillage chỉ là mức trần; PlantDoc là phép thử công bằng vì mô hình chưa từng thấy bộ này."),
        },
        "validation": val_result,
        "holdout": test_result,
        "vs_fedavg_on_holdout": versus,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "cluster": "group_id"},
    }
    for label, result in (("val", val_result), ("test", test_result)):
        by = result["by_source"]
        print(f"[pv-eval] {label}: PV {by.get('plantvillage', {}).get('top1_accuracy', float('nan')):.4f} "
              f"PD {by.get('plantdoc', {}).get('top1_accuracy', float('nan')):.4f} "
              f"T={result['temperature_fitted_on_calibration']}")
    print("[pv-eval] chuẩn hóa:", {k: v["val_top1"] for k, v in comparison.items()})
    if args.limit:
        print("[pv-eval] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(REPORT_JSON)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[pv-eval] wrote {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tái đánh giá CỤC BỘ 7 checkpoint của bundle published trên ảnh validation của release mixed_pv_pd_v3.

Mục đích: xác nhận (hoặc bác bỏ) số liệu mà nhóm train công bố, và đo thứ chưa có trong bundle:
confusion matrix đầy đủ, ảnh sai với confidence cao, calibration (ECE) trước/sau temperature scaling.

Quy trình bám sát nhóm train (`evaluate_three_workflows.py` ở nhánh train-*): cùng manifest val, transform `canonical_v1`
của hợp đồng đi kèm bundle (đã kiểm hash), EXIF transpose, RGBA -> nền đen, softmax, top-1.
Temperature được fit trên `calibration.jsonl` (tách biệt hoàn toàn với train và val), KHÔNG fit trên val.

Chạy:
    python scripts/evaluate_published_models.py            # chạy suy luận (vài phút-chục phút, CPU) rồi phân tích
    python scripts/evaluate_published_models.py --reuse    # dùng logits đã lưu ở data/reeval/ nếu còn khớp
    python scripts/evaluate_published_models.py --limit 64 # thử nhanh (KHÔNG ghi vào artifacts/)

Cần dataset ở data/dataset_sources (nhánh dataset-sources của GitHub). Không có -> thoát mã 1, không đoán.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

BINS = 15
HIGH_CONFIDENCE = 0.9
SOURCES = ("plantdoc", "plantvillage")
NPZ_DIR = "data/reeval"  # gitignored (data/*)
# `reproduced` = khớp TUYỆT ĐỐI số ảnh đúng và metric (sai số float <= 5e-4).
# `within_tolerance` = chấp nhận tối đa MỘT ảnh đổi nhãn so với số công bố (ví dụ ảnh sát ngưỡng quyết định khi khác phần cứng/
# thư viện): |chênh số ảnh đúng| <= 1, |chênh số lỗi-tự-tin| <= 2, metric lệch <= 2e-3. Tiêu chí này được ghi trong báo cáo,
# không nới ngầm: báo cáo liệt kê cả số lượng khớp tuyệt đối lẫn số lượng chỉ nằm trong ngưỡng.
FLOAT_TOLERANCE = 5e-4
WITHIN_ONE_IMAGE = {"correct_images": 1, "high_confidence_wrong_images": 2, "metric": 2e-3}


# --------------------------------------------------------------------------- phân tích (thuần numpy, có test)
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = BINS) -> tuple[float, list[dict[str, Any]]]:
    """Expected Calibration Error (độ rộng bin đều). Trả (ECE, các bin có ảnh)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    value, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
        value += abs(acc - conf) * n / total
        table.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4), "n": n,
                      "mean_confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return float(value), table


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """T tối thiểu NLL trên tập hiệu chuẩn (tìm theo log T)."""
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(lambda log_t: nll(logits, targets, float(np.exp(log_t))),
                             bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def group_scores(targets: np.ndarray, preds: np.ndarray, conf: np.ndarray, classes: list[str]) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    support = sorted(set(targets.tolist()))
    if not support:
        return {"count": 0}
    crops = [name.split("___", 1)[0] for name in classes]
    return {
        "count": int(len(targets)),
        "correct": int((targets == preds).sum()),
        "supported_classes": len(support),
        "top1_accuracy": float((targets == preds).mean()),
        "macro_f1_supported": float(f1_score(targets, preds, labels=support, average="macro", zero_division=0)),
        "crop_accuracy": float(np.mean([crops[t] == crops[p] for t, p in zip(targets, preds)])),
        "high_confidence_wrong_count_0_9": int(((targets != preds) & (conf >= HIGH_CONFIDENCE)).sum()),
    }


def per_class_recall(targets: np.ndarray, preds: np.ndarray, classes: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for index, name in enumerate(classes):
        mask = targets == index
        if mask.any():
            out[name] = {"support": int(mask.sum()), "recall": float((preds[mask] == index).mean())}
    return out


def confusion(targets: np.ndarray, preds: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (targets, preds), 1)
    return matrix


def compare_with_published(own: dict[str, Any], published: dict[str, Any]) -> dict[str, Any]:
    """So số tái đo với số công bố: số ảnh đúng phải bằng nhau; metric float trong ngưỡng."""
    if not published:
        return {"available": False}
    pub_correct = round(float(published["top1_accuracy"]) * int(published["count"]))
    diffs = {k: abs(float(own[k]) - float(published[k]))
             for k in ("top1_accuracy", "macro_f1_supported", "crop_accuracy") if k in published}
    pub_hcw = published.get("high_confidence_wrong_count_0_9")
    max_diff = max(diffs.values()) if diffs else 0.0
    hcw_diff = None if pub_hcw is None else own["high_confidence_wrong_count_0_9"] - int(pub_hcw)
    return {
        "available": True,
        "count_equal": int(own["count"]) == int(published["count"]),
        "correct_own": own["correct"],
        "correct_published": pub_correct,
        "correct_diff_images": own["correct"] - pub_correct,
        "max_abs_metric_diff": max(diffs.values()) if diffs else None,
        "high_confidence_wrong_own": own["high_confidence_wrong_count_0_9"],
        "high_confidence_wrong_published": pub_hcw,
        "high_confidence_wrong_diff_images": hcw_diff,
        "reproduced": bool(
            int(own["count"]) == int(published["count"])
            and own["correct"] == pub_correct
            and max_diff <= FLOAT_TOLERANCE
            and hcw_diff in (None, 0)
        ),
        "within_tolerance": bool(
            int(own["count"]) == int(published["count"])
            and abs(own["correct"] - pub_correct) <= WITHIN_ONE_IMAGE["correct_images"]
            and (hcw_diff is None or abs(hcw_diff) <= WITHIN_ONE_IMAGE["high_confidence_wrong_images"])
            and max_diff <= WITHIN_ONE_IMAGE["metric"]
        ),
    }


def analyze_model(
    val_logits: np.ndarray, val_targets: np.ndarray, val_sources: np.ndarray,
    cal_logits: np.ndarray, cal_targets: np.ndarray, classes: list[str],
) -> dict[str, Any]:
    probs = softmax(val_logits)
    preds, conf = probs.argmax(axis=1), probs.max(axis=1)
    correct = (preds == val_targets).astype(np.float64)
    temperature = fit_temperature(cal_logits, cal_targets)
    probs_t = softmax(val_logits, temperature)
    conf_t = probs_t.max(axis=1)
    out: dict[str, Any] = {
        "temperature_fitted_on_calibration": round(temperature, 4),
        "calibration_set_size": int(len(cal_targets)),
        "overall": group_scores(val_targets, preds, conf, classes),
        "by_source": {}, "calibration": {},
    }
    for scope, mask in (("overall", np.ones(len(val_targets), dtype=bool)),
                        *((s, val_sources == s) for s in SOURCES)):
        if not mask.any():
            continue
        e_raw, bins_raw = ece(conf[mask], correct[mask])
        e_ts, _ = ece(conf_t[mask], correct[mask])
        out["calibration"][scope] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "mean_confidence_raw": float(conf[mask].mean()),
            "mean_confidence_temperature_scaled": float(conf_t[mask].mean()),
            "ece_raw": round(e_raw, 5),
            "ece_temperature_scaled": round(e_ts, 5),
            "nll_raw": round(nll(val_logits[mask], val_targets[mask]), 5),
            "nll_temperature_scaled": round(nll(val_logits[mask], val_targets[mask], temperature), 5),
            "reliability_raw": bins_raw,
        }
        if scope != "overall":
            out["by_source"][scope] = group_scores(val_targets[mask], preds[mask], conf[mask], classes)
            out["by_source"][scope]["per_class_recall"] = per_class_recall(val_targets[mask], preds[mask], classes)
    return out


# --------------------------------------------------------------------------- suy luận
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ManifestImages:
    """Dataset mở ảnh đúng như CanonicalClassificationDataset của nhóm train (EXIF, RGBA -> nền đen)."""

    def __init__(self, rows: list[dict[str, Any]], root: Path, transform: Any):
        self.rows, self.root, self.transform = rows, root, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image, ImageOps

        path = self.root / self.rows[index]["relative_path"]
        with Image.open(path) as image:
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:  # noqa: BLE001 — giống normalize_exif_orientation của nhóm train
                pass
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (0, 0, 0))
                background.paste(rgba, mask=rgba.split()[3])
                rgb = background
            else:
                rgb = image.convert("RGB")
        return self.transform(rgb), index


def _load_contract_transforms(bundle: Path):
    spec = importlib.util.spec_from_file_location("bundle_contract_transforms", bundle / "contract" / "transforms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # file này đã được web.demo_evidence.load_evidence kiểm hash
    return module.get_transforms("canonical_v1", is_train=False)


def _load_model(path: Path):
    import torch
    from numpy._core.multiarray import _reconstruct
    from torch import nn
    from torch.serialization import safe_globals
    from torchvision.models import mobilenet_v3_small

    numpy_types = [type(np.dtype(kind)) for kind in ("float64", "int64", "float32", "uint32", "uint64", "int32", "uint8")]
    with safe_globals([_reconstruct, np.ndarray, np.dtype, *numpy_types]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload.get("model_state_dict"))
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict(state, strict=True)
    return model.eval()


def collect_logits(rows, root: Path, transform, models: dict[str, Any], batch_size: int, workers: int) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ManifestImages(rows, root, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, persistent_workers=False)
    out = {key: np.zeros((len(rows), 38), dtype=np.float32) for key in models}
    started = time.time()
    done = 0
    with torch.inference_mode():
        for images, index in loader:
            for key, model in models.items():
                out[key][index.numpy()] = model(images).numpy()
            done += len(index)
            if done % (batch_size * 10) < batch_size or done == len(rows):
                rate = done / max(time.time() - started, 1e-9)
                print(f"    {done}/{len(rows)} ảnh ({rate:.1f} ảnh/s, còn ~{(len(rows) - done) / max(rate, 1e-9) / 60:.1f} phút)", flush=True)
    return out


# --------------------------------------------------------------------------- báo cáo
def _published_for(model_key: str, source: str) -> dict[str, Any]:
    rel = {"centralized": "centralized", "fedavg": "fedavg"}.get(model_key) or f"local/client_{model_key[-2:]}"
    path = reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/{rel}/validation_metrics.json")
    validation = json.loads(path.read_text(encoding="utf-8"))["validation"]
    block = dict(validation["by_source"][source])
    block["per_class_recall"] = (validation.get("quality", {}).get("by_source", {}).get(source, {}) or {}).get("per_class_recall")
    return block


def build_report(
    evidence: dict[str, Any], val_rows, cal_rows,
    val_logits: dict[str, np.ndarray], cal_logits: dict[str, np.ndarray],
) -> dict[str, Any]:
    classes = evidence["classes"]
    val_targets = np.array([r["class_id"] for r in val_rows], dtype=np.int64)
    cal_targets = np.array([r["class_id"] for r in cal_rows], dtype=np.int64)
    sources = np.array([r["source_dataset"].lower() for r in val_rows])
    names = {m["model_id"]: m["name"] for m in evidence["models"]}

    models: dict[str, Any] = {}
    for key in val_logits:
        result = analyze_model(val_logits[key], val_targets, sources, cal_logits[key], cal_targets, classes)
        result["display_name"] = names[key]
        result["vs_published"] = {}
        for source in SOURCES:
            if source not in result["by_source"]:  # --limit có thể làm một nguồn không còn ảnh
                continue
            published = _published_for(key, source)
            comparison = compare_with_published(result["by_source"][source], published)
            pub_recall = published.get("per_class_recall")
            if pub_recall:
                own_recall = result["by_source"][source]["per_class_recall"]
                comparison["per_class_recall_equal"] = all(
                    abs(own_recall.get(c, {"recall": None})["recall"] - v["recall"]) <= 1e-9 for c, v in pub_recall.items()
                ) and set(own_recall) == set(pub_recall)
                comparison["reproduced"] = comparison["reproduced"] and comparison["per_class_recall_equal"]
                if not comparison["per_class_recall_equal"]:  # recall theo lớp lệch nhiều nhất bao nhiêu ảnh (phải <= 1 ảnh)
                    comparison["per_class_max_correct_diff_images"] = max(
                        abs(round(own_recall[c]["recall"] * own_recall[c]["support"]) - round(v["recall"] * v["support"]))
                        for c, v in pub_recall.items() if c in own_recall
                    )
                    comparison["within_tolerance"] = comparison["within_tolerance"] and comparison["per_class_max_correct_diff_images"] <= 1
            result["vs_published"][source] = comparison
        models[key] = result

    # Confusion matrix đầy đủ + ảnh sai tự tin cho hai model so sánh chính
    detail: dict[str, Any] = {}
    for key in ("fedavg", "centralized"):
        probs = softmax(val_logits[key])
        preds, conf = probs.argmax(axis=1), probs.max(axis=1)
        wrong_confident = [
            {
                "sample_id": val_rows[i]["sample_id"],
                "relative_path": val_rows[i]["relative_path"],
                "source": sources[i],
                "target": classes[val_targets[i]],
                "predicted": classes[preds[i]],
                "confidence": round(float(conf[i]), 4),
            }
            for i in np.argsort(-conf)
            if preds[i] != val_targets[i] and conf[i] >= HIGH_CONFIDENCE
        ]
        detail[key] = {
            "confusion_matrix": {s: confusion(val_targets[sources == s], preds[sources == s], len(classes)).tolist() for s in SOURCES},
            "top_confusions": {
                s: [{"target": classes[t], "predicted": classes[p], "count": int(c)}
                    for (t, p), c in Counter(
                        (int(t), int(p)) for t, p in zip(val_targets[sources == s], preds[sources == s]) if t != p
                    ).most_common(10)]
                for s in SOURCES
            },
            "high_confidence_wrong": wrong_confident,
        }

    comparisons = [c for m in models.values() for c in m["vs_published"].values()]
    all_reproduced = all(c.get("reproduced") for c in comparisons)
    main_exact = all(models[k]["vs_published"][s].get("reproduced") for k in ("centralized", "fedavg")
                     for s in SOURCES if s in models[k]["vs_published"])
    return {
        "evidence": "tái đánh giá cục bộ trên CPU từ checkpoint đã kiểm hash + ảnh validation đã kiểm byte_sha256",
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "validation_count": len(val_rows),
        "calibration_count": len(cal_rows),
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "classes": classes,
        "bins": BINS,
        "high_confidence_threshold": HIGH_CONFIDENCE,
        "all_published_numbers_reproduced": all_reproduced,
        "all_within_tolerance": all(c.get("within_tolerance") for c in comparisons),
        "main_models_exact": main_exact,
        "comparisons_total": len(comparisons),
        "comparisons_exact": sum(1 for c in comparisons if c.get("reproduced")),
        "tolerance_rule": WITHIN_ONE_IMAGE,
        "known_discrepancy_note": (
            "Chênh lệch (nếu có) chỉ ở Local-only và <= 1 ảnh. Mỗi chênh đều có ảnh ứng viên với độ lệch top-1/top-2 rất nhỏ "
            "(< 0,01); phù hợp với sai số số học giữa môi trường chạy, CHƯA được chứng minh."
        ),
        "models": models,
        "detail": detail,
    }


def _verify_inputs(evidence: dict[str, Any], release: Path, dataset_root: Path) -> tuple[list, list]:
    val_path, cal_path = release / "manifests" / "val.jsonl", release / "manifests" / "calibration.jsonl"
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    for path, key in ((val_path, "val"), (cal_path, "calibration")):
        if _sha256(path) != manifest["manifest_hashes"][key]:
            raise ValueError(f"{path.name}: hash khác release_manifest.json")
    if manifest["manifest_hashes"]["val"] != evidence["validation_manifest_sha256"]:
        raise ValueError("val manifest không phải bộ validation mà bundle đã công bố")
    val_rows, cal_rows = _jsonl(val_path), _jsonl(cal_path)
    for name, rows in (("val", val_rows), ("calibration", cal_rows)):
        for row in rows:
            if row["class_id"] < 0 or evidence["classes"][row["class_id"]] != row["target_class"]:
                raise ValueError(f"{name}: class_id không khớp taxonomy ({row['sample_id']})")
    train_groups = {r["group_id"] for r in _jsonl(release / "manifests" / "train.jsonl")}
    overlap = train_groups & {r["group_id"] for r in cal_rows + val_rows}
    if overlap:
        raise ValueError(f"train giao với val/calibration theo group_id: {len(overlap)} nhóm")
    if {r["group_id"] for r in cal_rows} & {r["group_id"] for r in val_rows}:
        raise ValueError("calibration giao với validation — không dùng được để hiệu chuẩn")
    return val_rows, cal_rows


# --------------------------------------------------------------------------- test giữ riêng (legacy_diagnostic_test)
def _verify_holdout(evidence: dict[str, Any], release: Path) -> list[dict[str, Any]]:
    """Tập test của release: hash khớp release_manifest, nhãn khớp taxonomy, không giao train/val/calibration theo nhóm."""
    path = release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl"
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["manifest_hashes"].get(reg.HOLDOUT_SPLIT)
    if not expected or _sha256(path) != expected:
        raise ValueError(f"{path.name}: hash khác release_manifest.json")
    rows = _jsonl(path)
    if len(rows) != int(manifest["counts"]["legacy_diagnostic_test_samples"]):
        raise ValueError("số ảnh test khác release_manifest.json")
    for row in rows:
        if row["class_id"] < 0 or evidence["classes"][row["class_id"]] != row["target_class"]:
            raise ValueError(f"test: class_id không khớp taxonomy ({row['sample_id']})")
    groups = {r["group_id"] for r in rows}
    for name in ("train", "val", "calibration"):
        if groups & {r["group_id"] for r in _jsonl(release / "manifests" / f"{name}.jsonl")}:
            raise ValueError(f"test giao với {name} theo group_id — không còn là tập giữ riêng")
    return rows


def _recorded_holdout(source: str) -> dict[str, Any] | None:
    """Số Centralized v6 tự ghi trên tập test (centralized/summary.json); các mô hình khác không có."""
    path = reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/centralized/summary.json")
    block = (json.loads(path.read_text(encoding="utf-8")).get(reg.HOLDOUT_SPLIT) or {}).get("per_source", {}).get(source)
    if not block:
        return None
    return {"count": int(block["count"]), "accuracy": float(block["accuracy"])}


def compare_with_recorded(own: dict[str, Any], recorded: dict[str, Any] | None) -> dict[str, Any]:
    """Số ghi trong summary làm tròn 4 chữ số, nên ứng với một KHOẢNG số ảnh đúng (vd 0,9923 × 10.843 ảnh: 10.759–10.760).
    Khớp khi cùng số ảnh và số ảnh đúng của ta làm tròn ra đúng con số đã ghi; lệch thì đo tới đầu gần nhất của khoảng."""
    if not recorded:
        return {"available": False}
    count = recorded["count"]
    consistent = [c for c in range(count + 1) if round(c / count, 4) == round(recorded["accuracy"], 4)] if count else []
    own_correct = int(own["correct"])
    count_equal = int(own["count"]) == count
    if not consistent:
        diff = None
    elif consistent[0] <= own_correct <= consistent[-1]:
        diff = 0
    else:
        diff = own_correct - (consistent[0] if own_correct < consistent[0] else consistent[-1])
    return {
        "available": True,
        "count_equal": count_equal,
        "accuracy_recorded": recorded["accuracy"],
        "correct_own": own_correct,
        "correct_recorded_range": [consistent[0], consistent[-1]] if consistent else None,
        "correct_diff_images": diff,
        "reproduced": count_equal and diff == 0,
    }


def build_holdout_report(
    evidence: dict[str, Any], rows, logits: dict[str, np.ndarray], cal_rows, cal_logits: dict[str, np.ndarray],
    manifest_sha256: str,
) -> dict[str, Any]:
    classes = evidence["classes"]
    targets = np.array([r["class_id"] for r in rows], dtype=np.int64)
    cal_targets = np.array([r["class_id"] for r in cal_rows], dtype=np.int64)
    sources = np.array([r["source_dataset"].lower() for r in rows])
    names = {m["model_id"]: m["name"] for m in evidence["models"]}
    models: dict[str, Any] = {}
    for key in logits:
        result = analyze_model(logits[key], targets, sources, cal_logits[key], cal_targets, classes)
        result["display_name"] = names[key]
        if key == "centralized":
            result["vs_recorded"] = {s: compare_with_recorded(result["by_source"][s], _recorded_holdout(s))
                                     for s in SOURCES if s in result["by_source"]}
        models[key] = result
    detail: dict[str, Any] = {}
    for key in ("fedavg", "centralized"):
        preds = softmax(logits[key]).argmax(axis=1)
        detail[key] = {
            "confusion_matrix": {s: confusion(targets[sources == s], preds[sources == s], len(classes)).tolist() for s in SOURCES},
            "top_confusions": {
                s: [{"target": classes[t], "predicted": classes[p], "count": int(c)}
                    for (t, p), c in Counter(
                        (int(t), int(p)) for t, p in zip(targets[sources == s], preds[sources == s]) if t != p
                    ).most_common(10)]
                for s in SOURCES
            },
        }
    counts = Counter(sources.tolist())
    return {
        "evidence": ("chấm cục bộ trên CPU 7 checkpoint đã kiểm hash, trên tập test giữ riêng của release "
                     "(không dùng để chọn checkpoint theo dataset_card); ảnh đã kiểm byte_sha256 qua audit release"),
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "split": reg.HOLDOUT_SPLIT,
        "holdout_manifest_sha256": manifest_sha256,
        "holdout_count": len(rows),
        "holdout_counts_by_source": {s: int(counts.get(s, 0)) for s in SOURCES},
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "classes": classes,
        "bins": BINS,
        "high_confidence_threshold": HIGH_CONFIDENCE,
        "caveat": ("PlantVillage trong tập này có cùng vấn đề nhóm lá như val (một phần ảnh không được gom nhóm theo lá); "
                   "PlantDoc chỉ có 172 ảnh — khoảng tin cậy rộng, xem artifacts/statistics_report.json."),
        "models": models,
        "detail": detail,
    }


def main_holdout(args, evidence: dict[str, Any], release: Path, dataset_root: Path) -> int:
    rows = _verify_holdout(evidence, release)
    val_rows, cal_rows = _verify_inputs(evidence, release, dataset_root)
    manifest_sha = _sha256(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    store = reg.abs_path(NPZ_DIR)
    store.mkdir(parents=True, exist_ok=True)
    keys = [m["model_id"] for m in evidence["models"]]
    checkpoints = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}

    # logits calibration: dùng lại cache của lần tái đánh giá val nếu còn khớp (T phải fit trên CÙNG tập calibration)
    cal_logits = None
    val_cache = store / "logits.npz"
    val_fingerprint = hashlib.sha256(
        json.dumps([evidence["validation_manifest_sha256"], len(val_rows), len(cal_rows), checkpoints], sort_keys=True).encode()
    ).hexdigest()  # cùng công thức với cache của lần tái đánh giá val
    if val_cache.is_file():
        with np.load(val_cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) == val_fingerprint and all(f"cal_{k}" in data for k in keys):
                cal_logits = {k: data[f"cal_{k}"] for k in keys}
                print("[holdout] dùng lại logits calibration của lần tái đánh giá val")
    holdout_cache = store / "holdout_logits.npz"
    fingerprint = hashlib.sha256(json.dumps([manifest_sha, len(rows), checkpoints], sort_keys=True).encode()).hexdigest()
    logits = None
    if args.reuse and holdout_cache.is_file():
        with np.load(holdout_cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) == fingerprint:
                logits = {k: data[f"test_{k}"] for k in keys}
                print("[holdout] dùng lại logits đã lưu")
    if logits is None or cal_logits is None:
        import torch

        torch.set_num_threads(max(1, min(10, torch.get_num_threads())))
        transform = _load_contract_transforms(reg.abs_path(reg.DEMO_BUNDLE_DIR))
        models = {}
        for item in evidence["models"]:
            path = reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/{item['checkpoint_relative_path']}")
            if _sha256(path) != item["checkpoint_sha256"]:
                raise ValueError(f"{item['model_id']}: checkpoint đổi sau khi kiểm hash")
            models[item["model_id"]] = _load_model(path)
        if cal_logits is None:
            print(f"[holdout] calibration ({len(cal_rows)} ảnh):")
            cal_logits = collect_logits(cal_rows, dataset_root, transform, models, args.batch_size, args.workers)
        if logits is None:
            print(f"[holdout] {reg.HOLDOUT_SPLIT} ({len(rows)} ảnh):")
            logits = collect_logits(rows, dataset_root, transform, models, args.batch_size, args.workers)
            if not args.limit:
                np.savez_compressed(holdout_cache, fingerprint=fingerprint, **{f"test_{k}": v for k, v in logits.items()})

    report = build_holdout_report(evidence, rows, logits, cal_rows, cal_logits, manifest_sha)
    for key, model in report["models"].items():
        by = model["by_source"]
        line = (f"[holdout] {model['display_name']:15} PV {by.get('plantvillage', {}).get('top1_accuracy', float('nan')):.4f} "
                f"PD {by.get('plantdoc', {}).get('top1_accuracy', float('nan')):.4f}")
        if "vs_recorded" in model:
            line += " | so với summary.json: " + ", ".join(
                f"{s}={v.get('reproduced')} ({v.get('correct_diff_images')} ảnh)" for s, v in model["vs_recorded"].items())
        print(line)
    if args.limit:
        print("[holdout] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.HOLDOUT_EVAL_JSON)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[holdout] wrote {reg.HOLDOUT_EVAL_JSON}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", action="store_true",
                        help=f"Chấm trên tập test giữ riêng {reg.HOLDOUT_SPLIT} thay vì validation")
    parser.add_argument("--reuse", action="store_true", help="Dùng logits đã lưu ở data/reeval/ nếu manifest không đổi")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ N ảnh đầu mỗi tập (thử nhanh; không ghi artifacts/)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[reeval] bundle không hợp lệ: {evidence.get('reason')}", file=sys.stderr)
        return 1
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    if not (release / "manifests" / "val.jsonl").is_file():
        print(f"[reeval] thiếu dataset ở {reg.DATASET_SOURCES_DIR} (nhánh dataset-sources của GitHub)", file=sys.stderr)
        return 1
    if args.holdout:
        return main_holdout(args, evidence, release, dataset_root)
    val_rows, cal_rows = _verify_inputs(evidence, release, dataset_root)
    if args.limit:
        val_rows, cal_rows = val_rows[: args.limit], cal_rows[: args.limit]

    store = reg.abs_path(NPZ_DIR)
    store.mkdir(parents=True, exist_ok=True)
    cache = store / "logits.npz"
    fingerprint = hashlib.sha256(
        json.dumps([evidence["validation_manifest_sha256"], len(val_rows), len(cal_rows),
                    {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}], sort_keys=True).encode()
    ).hexdigest()
    keys = [m["model_id"] for m in evidence["models"]]
    val_logits = cal_logits = None
    if args.reuse and cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) == fingerprint:
                val_logits = {k: data[f"val_{k}"] for k in keys}
                cal_logits = {k: data[f"cal_{k}"] for k in keys}
                print("[reeval] dùng lại logits đã lưu")
    if val_logits is None:
        import torch

        torch_threads = max(1, min(10, torch.get_num_threads()))
        torch.set_num_threads(torch_threads)
        transform = _load_contract_transforms(reg.abs_path(reg.DEMO_BUNDLE_DIR))
        models = {}
        for item in evidence["models"]:
            path = reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/{item['checkpoint_relative_path']}")
            if _sha256(path) != item["checkpoint_sha256"]:  # load_evidence đã kiểm; kiểm lại ngay trước khi unpickle
                raise ValueError(f"{item['model_id']}: checkpoint đổi sau khi kiểm hash")
            models[item["model_id"]] = _load_model(path)
        print(f"[reeval] 7 model đã nạp; val={len(val_rows)} ảnh, calibration={len(cal_rows)} ảnh, CPU threads={torch_threads}")
        print("  validation:")
        val_logits = collect_logits(val_rows, dataset_root, transform, models, args.batch_size, args.workers)
        print("  calibration:")
        cal_logits = collect_logits(cal_rows, dataset_root, transform, models, args.batch_size, args.workers)
        np.savez_compressed(cache, fingerprint=fingerprint,
                            **{f"val_{k}": v for k, v in val_logits.items()}, **{f"cal_{k}": v for k, v in cal_logits.items()})

    report = build_report(evidence, val_rows, cal_rows, val_logits, cal_logits)
    for key, model in report["models"].items():
        cal = model["calibration"]["overall"]
        vs = model["vs_published"]
        by = model["by_source"]
        pv = by.get("plantvillage", {}).get("top1_accuracy", float("nan"))
        pd = by.get("plantdoc", {}).get("top1_accuracy", float("nan"))
        print(f"[reeval] {model['display_name']:15} PV {pv:.4f} PD {pd:.4f} | "
              f"tái lập: PV={vs.get('plantvillage', {}).get('reproduced')} PD={vs.get('plantdoc', {}).get('reproduced')} | "
              f"ECE {cal['ece_raw']:.4f} -> {cal['ece_temperature_scaled']:.4f} (T={model['temperature_fitted_on_calibration']})")
    print(f"[reeval] khớp tuyệt đối {report['comparisons_exact']}/{report['comparisons_total']} phép so; "
          f"trong ngưỡng ≤1 ảnh: {report['all_within_tolerance']}; hai model chính khớp tuyệt đối: {report['main_models_exact']}")
    if args.limit:
        print("[reeval] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.LOCAL_REEVAL_JSON)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[reeval] wrote {reg.LOCAL_REEVAL_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

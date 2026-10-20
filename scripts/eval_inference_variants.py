"""Đo các cách nhận ảnh KHÔNG đổi trọng số: TTA, giữ tỉ lệ ảnh, chọn loại cây, gộp hai mô hình, từ chối khi không chắc.

Câu hỏi: trước khi train lại, có cách đưa ảnh vào cùng checkpoint đã công bố để nhận ảnh ngoài đồng tốt hơn không?
Mỗi ảnh được suy luận qua 6 góc nhìn (nguyên khung/lật/cắt giữa 80%/lật, giữ tỉ lệ/lật) với Centralized và FedAvg,
logits lưu vào data/reeval/variant_logits.npz; mọi biến thể sau đó chỉ là phép tính trên logits (numpy), có thể chạy lại.

Tập đo (không tập nào nằm trong train):
  pd_val / pd_test : lá PlantDoc đã cắt (val 269, giữ riêng 172)
  ff_val / ff_test : ảnh PlantDoc NGUYÊN KHUNG, chỉ cảnh mà mọi khung cùng một lớp (giống ảnh người dùng tải lên)
  pv_val / pv_test : mẫu phân tầng 1.000 ảnh PlantVillage mỗi tập (kiểm không giảm trên ảnh phòng thí nghiệm)

NGƯỠNG CHẤP NHẬN (đặt trước khi chạy, không đổi sau khi thấy số):
  Một biến thể được phép lên web nếu so với "gốc" CÙNG mô hình:
    1. ảnh nguyên khung: tăng >= +2,0 điểm top-1 trên CẢ ff_val lẫn ff_test;
    2. ảnh cắt: không giảm quá 2,0 điểm trên pd_val và pd_test;
    3. PlantVillage: không giảm quá 0,3 điểm trên pv_val và pv_test.
  Biến thể "chọn loại cây thật" dùng người dùng làm nguồn thông tin nên báo riêng là cận trên, không xét ngưỡng.

Chạy:
    python scripts/eval_inference_variants.py            # suy luận (vài phút CPU) rồi ghi artifacts/inference_variants_eval.json
    python scripts/eval_inference_variants.py --reuse    # dùng logits đã lưu nếu còn khớp
    python scripts/eval_inference_variants.py --limit 8  # thử nhanh, KHÔNG ghi artifacts/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SEED = 20261020
BOOTSTRAP = 2000
PV_SAMPLE = 1000
MODELS = ("centralized", "fedavg")
VIEWS = ("squash", "squash_flip", "center80", "center80_flip", "letterbox", "letterbox_flip")
NPZ = "data/reeval/variant_logits.npz"  # gitignored (data/*)
# Ngưỡng chấp nhận (xem docstring): điểm phần trăm, so với "gốc" cùng mô hình
FRAME_GAIN_MIN = 2.0
CROP_DROP_MAX = 2.0
PV_DROP_MAX = 0.3

# biến thể -> các góc nhìn được lấy trung bình xác suất
VARIANTS: dict[str, tuple[str, ...]] = {
    "base": ("squash",),
    "flip_tta": ("squash", "squash_flip"),
    "multi_tta": ("squash", "squash_flip", "center80", "center80_flip"),
    "letterbox": ("letterbox",),
    "letterbox_tta": ("letterbox", "letterbox_flip"),
    "all_views": VIEWS,
}


# --------------------------------------------------------------------------- phép tính thuần numpy (có test)
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def crop_of(class_name: str) -> str:
    return class_name.split("___", 1)[0]


def crop_masks(classes: list[str]) -> tuple[list[str], np.ndarray]:
    """(tên loại cây, ma trận [số cây, số lớp] 0/1)."""
    names = sorted({crop_of(c) for c in classes})
    return names, np.array([[crop_of(c) == n for c in classes] for n in names], dtype=bool)


def class_names_crop_index(classes: list[str]) -> tuple[np.ndarray, list[str]]:
    """(chỉ số loại cây của từng lớp, danh sách tên loại cây)."""
    names = sorted({crop_of(c) for c in classes})
    index = {n: i for i, n in enumerate(names)}
    return np.array([index[crop_of(c)] for c in classes]), names


def mean_over_views(probs: np.ndarray, views: tuple[str, ...], all_views: tuple[str, ...]) -> np.ndarray:
    """Trung bình xác suất [N, V, C] trên các góc nhìn được chọn."""
    return probs[:, [all_views.index(v) for v in views]].mean(axis=1)


def restrict_to_crop(probs: np.ndarray, classes: list[str], crops: np.ndarray | None = None) -> np.ndarray:
    """Chỉ giữ các lớp của MỘT loại cây rồi chuẩn hóa lại.

    `crops` = chỉ số loại cây cho từng ảnh (người dùng chọn); None = tự đoán loại cây có tổng xác suất lớn nhất."""
    _, masks = crop_masks(classes)
    if crops is None:
        mass = probs @ masks.T.astype(np.float64)
        crops = mass.argmax(axis=1)
    keep = masks[crops]
    out = np.where(keep, probs, 0.0)
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def topk_correct(probs: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    top = np.argsort(-probs, axis=1)[:, :k]
    return (top == targets[:, None]).any(axis=1)


def macro_f1(targets: np.ndarray, preds: np.ndarray) -> float | None:
    from sklearn.metrics import f1_score

    labels = sorted(set(targets.tolist()))
    return float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)) if labels else None


def paired_cluster_ci(groups: list[str], a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                      resamples: int = BOOTSTRAP) -> dict[str, Any]:
    """CI 95% của mean(b) - mean(a) (điểm phần trăm) khi lấy lại mẫu theo cụm, cùng cụm cho cả hai."""
    index: dict[str, int] = {}
    ids = np.array([index.setdefault(g, len(index)) for g in groups], dtype=np.int64)
    k = len(index)
    n = np.bincount(ids, minlength=k).astype(np.float64)
    sa = np.bincount(ids, weights=a.astype(np.float64), minlength=k)
    sb = np.bincount(ids, weights=b.astype(np.float64), minlength=k)
    est = 100.0 * (sb.sum() - sa.sum()) / n.sum()
    if k < 2:
        return {"delta": est, "low": None, "high": None, "clusters": k}
    stats = np.empty(resamples)
    for start in range(0, resamples, 250):
        stop = min(resamples, start + 250)
        idx = rng.integers(0, k, size=(stop - start, k))
        stats[start:stop] = 100.0 * (sb[idx].sum(axis=1) - sa[idx].sum(axis=1)) / n[idx].sum(axis=1)
    low, high = np.quantile(stats, [0.025, 0.975])
    return {"delta": float(est), "low": float(low), "high": float(high), "clusters": int(k)}


def accuracy_ci(groups: list[str], correct: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    from scripts.report_statistics import cluster_bootstrap_ci

    ci = cluster_bootstrap_ci(groups, correct.astype(np.float64), rng, resamples=BOOTSTRAP)
    return {k: (round(100.0 * v, 2) if isinstance(v, float) else v) for k, v in ci.items()}


def selective_table(conf: np.ndarray, correct: np.ndarray, coverages: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)) -> dict[str, Any]:
    """Độ chính xác khi chỉ trả lời phần ảnh tự tin nhất; AUROC của độ tin cậy trong việc bắt ảnh sai."""
    order = np.argsort(-conf, kind="stable")
    c = correct[order].astype(np.float64)
    table = {}
    for cov in coverages:
        take = max(1, int(round(cov * len(c))))
        table[f"coverage_{int(cov * 100)}"] = {"answered": int(take), "accuracy": round(100.0 * float(c[:take].mean()), 2),
                                                "min_confidence": round(float(conf[order][take - 1]), 4)}
    out: dict[str, Any] = {"table": table}
    if 0 < correct.sum() < len(correct):
        from sklearn.metrics import roc_auc_score

        out["auroc_confidence_detects_wrong"] = round(float(roc_auc_score(~correct.astype(bool), 1.0 - conf)), 4)
    return out


# --------------------------------------------------------------------------- dữ liệu
def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_datasets(limit: int) -> dict[str, list[dict[str, Any]]]:
    """Mỗi mục: {key, path, target(class_id), group}. Ảnh nguyên khung chỉ lấy cảnh một lớp, đã kiểm SHA-256."""
    from scripts import eval_fullframe_plantdoc as ff

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    cache = reg.abs_path(ff.CACHE_DIR)
    annotations, images, rows_ff = ff.load_inputs(release)
    sets: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(SEED)
    for split, name in (("val", "val.jsonl"), (reg.HOLDOUT_SPLIT, f"{reg.HOLDOUT_SPLIT}.jsonl")):
        short = "val" if split == "val" else "test"
        rows = _jsonl(release / "manifests" / name)
        pd = [r for r in rows if r["source_domain"] == "plantdoc"]
        pv = [r for r in rows if r["source_domain"] == "plantvillage"]
        sets[f"pd_{short}"] = [{"key": r["sample_id"], "path": root / r["relative_path"], "target": int(r["class_id"]),
                                "group": r["group_id"]} for r in pd]
        by_class: dict[int, list[dict[str, Any]]] = {}
        for r in pv:
            by_class.setdefault(int(r["class_id"]), []).append(r)
        per_class = max(1, -(-PV_SAMPLE // len(by_class)))
        picked = []
        for cid in sorted(by_class):
            pool = sorted(by_class[cid], key=lambda r: r["sample_id"])
            picked += rng.sample(pool, min(per_class, len(pool)))
        sets[f"pv_{short}"] = [{"key": r["sample_id"], "path": root / r["relative_path"], "target": int(r["class_id"]),
                                "group": r["group_id"]} for r in picked]
        single = ff.single_class_scenes(rows_ff[short], annotations)
        scenes = []
        for scene, cid in sorted(single.items()):
            path = ff.scene_path(cache, images[scene])
            if path.is_file() and ff.sha256_file(path) == images[scene]["byte_sha256"]:
                scenes.append({"key": scene, "path": path, "target": int(cid), "group": scene})
        sets[f"ff_{short}"] = scenes
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
    return sets


# --------------------------------------------------------------------------- suy luận
def view_images(img):
    """6 góc nhìn của một ảnh RGB (đã qua giải mã chuẩn): trả về dict tên -> ảnh PIL."""
    from PIL import Image, ImageOps

    w, h = img.size
    box = (int(round(0.1 * w)), int(round(0.1 * h)), int(round(0.9 * w)), int(round(0.9 * h)))
    centre = img.crop(box)
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))  # cùng màu nền đen với ảnh trong suốt lúc train
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return {
        "squash": img, "squash_flip": ImageOps.mirror(img),
        "center80": centre, "center80_flip": ImageOps.mirror(centre),
        "letterbox": canvas, "letterbox_flip": ImageOps.mirror(canvas),
    }


def fingerprint(evidence: dict[str, Any], sets: dict[str, list[dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for model_id in MODELS:
        digest.update(next(m["checkpoint_sha256"] for m in evidence["models"] if m["model_id"] == model_id).encode())
    for name in sorted(sets):
        digest.update(name.encode())
        for item in sets[name]:
            digest.update(item["key"].encode())
    digest.update(("v1|" + ",".join(VIEWS)).encode())
    return digest.hexdigest()


def run_inference(sets: dict[str, list[dict[str, Any]]], threads: int) -> dict[str, np.ndarray]:
    import torch
    from PIL import Image

    from scripts.evaluate_published_models import _load_contract_transforms, _load_model
    from src.common.image_io import to_rgb

    torch.set_num_threads(max(1, threads))
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    transform = _load_contract_transforms(bundle)
    paths = {"centralized": bundle / "centralized" / "checkpoint_best.pt", "fedavg": bundle / "fedavg" / "checkpoint_best.pt"}
    models = {name: _load_model(path) for name, path in paths.items()}

    out: dict[str, np.ndarray] = {}
    started = time.time()
    for name, items in sets.items():
        logits = np.zeros((len(MODELS), len(items), len(VIEWS), 38), dtype=np.float32)
        for i, item in enumerate(items):
            with Image.open(item["path"]) as handle:
                img = to_rgb(handle)
            views = view_images(img)
            batch = torch.stack([transform(views[v]) for v in VIEWS])
            with torch.inference_mode():
                for m, model_name in enumerate(MODELS):
                    logits[m, i] = models[model_name](batch).numpy()
            if (i + 1) % 100 == 0:
                print(f"[variants] {name} {i + 1}/{len(items)} {time.time() - started:.0f}s", flush=True)
        out[name] = logits
        print(f"[variants] {name}: {len(items)} ảnh xong ({time.time() - started:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------- phân tích
def analyze(sets: dict[str, list[dict[str, Any]]], logits: dict[str, np.ndarray], classes: list[str],
            temperatures: dict[str, float]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    crop_names, _ = crop_masks(classes)
    crop_index = {n: i for i, n in enumerate(crop_names)}
    class_crop = np.array([crop_index[crop_of(c)] for c in classes])
    results: dict[str, Any] = {}
    correct_store: dict[tuple[str, str, str], np.ndarray] = {}

    def mean_probs(probs_by_model: dict[str, np.ndarray], model_name: str, idx: list[int]) -> np.ndarray:
        if model_name == "ensemble":
            return np.mean([probs_by_model[m][:, idx].mean(axis=1) for m in MODELS], axis=0)
        return probs_by_model[model_name][:, idx].mean(axis=1)

    for set_name, items in sets.items():
        targets = np.array([it["target"] for it in items], dtype=np.int64)
        groups = [it["group"] for it in items]
        true_crops = class_crop[targets]
        probs_by_model = {m: softmax(logits[set_name][i]) for i, m in enumerate(MODELS)}  # [N, V, C]
        entry: dict[str, Any] = {"n": len(items), "clusters": len(set(groups)), "models": {}}
        for model_name in (*MODELS, "ensemble"):
            model_entry: dict[str, Any] = {}
            for variant, views in VARIANTS.items():
                p = mean_probs(probs_by_model, model_name, [VIEWS.index(v) for v in views])
                for crop_mode in ("free", "predicted_crop", "oracle_crop"):
                    if crop_mode == "free":
                        q = p
                    elif crop_mode == "predicted_crop":
                        q = restrict_to_crop(p, classes)
                    else:
                        q = restrict_to_crop(p, classes, true_crops)
                    pred = q.argmax(axis=1)
                    correct = pred == targets
                    key = variant if crop_mode == "free" else f"{variant}+{crop_mode}"
                    correct_store[(set_name, model_name, key)] = correct
                    model_entry[key] = {
                        "top1": round(100.0 * float(correct.mean()), 2),
                        "top3": round(100.0 * float(topk_correct(q, targets, 3).mean()), 2),
                        "crop_accuracy": round(100.0 * float((class_crop[pred] == true_crops).mean()), 2),
                        "macro_f1_supported": round(100.0 * (macro_f1(targets, pred) or 0.0), 2),
                        "top1_ci95": accuracy_ci(groups, correct, rng),
                    }
            # từ chối khi không chắc: độ tin cậy = xác suất lớn nhất, hiệu chuẩn bằng T cho mô hình đơn
            temperature = temperatures.get(model_name)
            selective = {}
            for variant in ("base", "multi_tta"):
                p = mean_probs(probs_by_model, model_name, [VIEWS.index(v) for v in VARIANTS[variant]])
                if temperature:
                    p = softmax(np.log(np.clip(p, 1e-12, 1.0)) / temperature)
                selective[variant] = selective_table(p.max(axis=1), p.argmax(axis=1) == targets)
            model_entry["selective_answering"] = selective
            entry["models"][model_name] = model_entry
        results[set_name] = entry

    # so với "gốc" cùng mô hình + CI cặp theo cụm + cổng chấp nhận
    verdicts: dict[str, Any] = {}
    for model_name in (*MODELS, "ensemble"):
        for variant in VARIANTS:
            for key in (variant, f"{variant}+predicted_crop"):
                if key == "base":
                    continue
                deltas: dict[str, Any] = {}
                for set_name, items in sets.items():
                    groups = [it["group"] for it in items]
                    deltas[set_name] = paired_cluster_ci(groups, correct_store[(set_name, model_name, "base")],
                                                         correct_store[(set_name, model_name, key)], rng)
                ok_frame = all(deltas[s]["delta"] >= FRAME_GAIN_MIN for s in ("ff_val", "ff_test") if s in deltas)
                ok_crop = all(deltas[s]["delta"] >= -CROP_DROP_MAX for s in ("pd_val", "pd_test") if s in deltas)
                ok_pv = all(deltas[s]["delta"] >= -PV_DROP_MAX for s in ("pv_val", "pv_test") if s in deltas)
                verdicts[f"{model_name}:{key}"] = {
                    "delta_vs_base_points": {s: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}
                                             for s, d in deltas.items()},
                    "frame_gain_ok": bool(ok_frame), "crop_not_worse_ok": bool(ok_crop), "pv_not_worse_ok": bool(ok_pv),
                    "accepted_for_web": bool(ok_frame and ok_crop and ok_pv),
                }
    return {"sets": results, "verdicts": verdicts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse", action="store_true", help="dùng logits đã lưu nếu còn khớp")
    parser.add_argument("--limit", type=int, default=0, help="chỉ N ảnh đầu mỗi tập (thử nhanh, không ghi artifacts/)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[variants] bundle chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    if not (reg.abs_path(reg.RELEASE_V3_DIR) / "annotations" / "detection.json").is_file():
        print("[variants] thiếu data/dataset_sources/mixed/pv_pd_v3", file=sys.stderr)
        return 1
    sets = build_datasets(args.limit)
    print("[variants] số ảnh:", {k: len(v) for k, v in sets.items()})
    fp = fingerprint(evidence, sets)
    npz = reg.abs_path(NPZ)
    logits: dict[str, np.ndarray] | None = None
    if args.reuse and npz.is_file() and not args.limit:
        data = np.load(npz, allow_pickle=False)
        if str(data["fingerprint"]) == fp:
            logits = {k: data[k] for k in sets}
            print("[variants] dùng lại logits đã lưu")
    if logits is None:
        logits = run_inference(sets, args.threads)
        if not args.limit:
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, fingerprint=np.array(fp), **logits)

    classes = evidence["classes"]
    reeval = json.loads(reg.abs_path(reg.LOCAL_REEVAL_JSON).read_text(encoding="utf-8"))
    temperatures = {m: float(reeval["models"][m]["temperature_fitted_on_calibration"]) for m in MODELS}

    # cổng tái lập: "gốc" trên PlantDoc val phải khớp số ảnh đúng đã công bố
    gate: dict[str, Any] = {}
    if "pd_val" in sets and not args.limit:
        targets = np.array([it["target"] for it in sets["pd_val"]])
        for i, m in enumerate(MODELS):
            correct = int((logits["pd_val"][i][:, 0].argmax(axis=1) == targets).sum())
            published = int(reeval["models"][m]["by_source"]["plantdoc"]["correct"])
            gate[m] = {"correct": correct, "published_correct": published, "equal": correct == published, "of": len(targets)}
    if gate and not all(g["equal"] for g in gate.values()):
        print(f"[variants] cổng tái lập KHÔNG đạt, không ghi artifacts/: {gate}", file=sys.stderr)
        return 2
    analysis = analyze(sets, logits, classes, temperatures)
    report = {
        "protocol_id": evidence["protocol_id"],
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"] if m["model_id"] in MODELS},
        "seed": SEED, "bootstrap": BOOTSTRAP,
        "acceptance_rule": {
            "frame_gain_min_points_on_ff_val_and_ff_test": FRAME_GAIN_MIN,
            "crop_drop_max_points_on_pd_val_and_pd_test": CROP_DROP_MAX,
            "plantvillage_drop_max_points_on_pv_val_and_pv_test": PV_DROP_MAX,
            "note": "Đặt trước khi chạy. Biến thể '+oracle_crop' dùng loại cây thật do người dùng chọn: cận trên, không xét ngưỡng.",
        },
        "views": list(VIEWS), "variants": {k: list(v) for k, v in VARIANTS.items()},
        "datasets": {
            "pd_val": "lá PlantDoc đã cắt, tập kiểm tra", "pd_test": "lá PlantDoc đã cắt, tập giữ riêng",
            "ff_val": "ảnh PlantDoc nguyên khung, cảnh một lớp, tập kiểm tra",
            "ff_test": "ảnh PlantDoc nguyên khung, cảnh một lớp, tập giữ riêng",
            "pv_val": f"mẫu phân tầng ~{PV_SAMPLE} ảnh PlantVillage, tập kiểm tra",
            "pv_test": f"mẫu phân tầng ~{PV_SAMPLE} ảnh PlantVillage, tập giữ riêng",
        },
        "baseline_reproduces_published_pd_val": gate,
        "temperatures": temperatures,
        "caveat": "PlantDoc nhỏ (vài trăm ảnh, theo cụm cảnh): khoảng tin cậy rộng. Không phải độ chính xác ngoài thực địa.",
        **analysis,
    }
    if args.limit:
        print(json.dumps({"sets": {k: v["n"] for k, v in analysis["sets"].items()}}, ensure_ascii=False))
        print("[variants] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.INFERENCE_VARIANTS_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[variants] đã ghi {reg.INFERENCE_VARIANTS_JSON}; cổng tái lập: {gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Đo hai mô hình nền (BioCLIP, DINOv2) trên đúng các tập ảnh ngoài đồng đã khóa: mốc "phương pháp mạnh hơn" có số đo thật.

Câu hỏi: nếu thay MobileNetV3-Small bằng mô hình nền cỡ lớn, KHÔNG huấn luyện thêm lớp nào, ảnh ngoài đồng nhận được bao nhiêu?
Hai cách dùng, đều không có bước học:
  * BioCLIP zero-shot: so ảnh với câu mô tả 38 lớp (không dùng ảnh huấn luyện).
  * kNN trên đặc trưng đóng băng (BioCLIP và DINOv2-small): thư viện tham chiếu = mọi lá PlantDoc trong tập học (2.115) cộng 40 ảnh
    PlantVillage mỗi lớp (cố định bằng seed); bỏ phiếu k láng giềng theo độ tương đồng cosin. Không có lan truyền ngược.

Tập đo giống các phép đo trước để so trực tiếp: PlantWild tầng bệnh (ext_strict), PlantWild tầng lá khỏe ngầm định (ext_healthy),
PlantDoc đã cắt (kiểm tra và giữ riêng). Cùng khóa SHA-256 manifest PlantWild.

Cần môi trường riêng (không đụng .venv của đồ án): tmp/venv-fm có torch, open_clip_torch, transformers.

Chạy:
    tmp/venv-fm/Scripts/python.exe -X utf8 scripts/eval_foundation_models.py            # ghi artifacts/foundation_model_eval.json
    tmp/venv-fm/Scripts/python.exe -X utf8 scripts/eval_foundation_models.py --limit 24 # thử nhanh, KHÔNG ghi artifacts/
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
GALLERY_PV_PER_CLASS = 40
K_VALUES = (1, 5, 10)
NPZ = "data/reeval/foundation_features.npz"  # gitignored (data/*)
BIOCLIP = "hf-hub:imageomics/bioclip"
DINO = "facebook/dinov2-small"


def _load_to_rgb():
    """Nạp `to_rgb` trực tiếp từ tệp để không kéo theo các gói khác của `src.common`."""
    spec = importlib.util.spec_from_file_location("image_io_standalone", ROOT / "src" / "common" / "image_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.to_rgb


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- văn bản mô tả lớp (thuần, có test)
def natural_names(class_name: str) -> tuple[str, str]:
    """('tomato', 'late blight') từ 'Tomato___Late_blight'; lớp khỏe có điều kiện là 'healthy'."""
    crop, _, condition = class_name.partition("___")
    crop = crop.replace("Corn_(maize)", "corn").replace("Cherry_(including_sour)", "cherry").replace("Pepper,_bell", "bell pepper")
    crop = crop.replace("_", " ").strip().lower()
    cond = condition.replace("Cercospora_leaf_spot Gray_leaf_spot", "gray leaf spot").replace("Haunglongbing_(Citrus_greening)", "citrus greening")
    cond = cond.replace("Spider_mites Two-spotted_spider_mite", "two-spotted spider mite").replace("Esca_(Black_Measles)", "esca black measles")
    cond = cond.replace("Leaf_blight_(Isariopsis_Leaf_Spot)", "leaf blight").replace("Common_rust_", "common rust")
    cond = cond.replace("_", " ").strip().lower()
    return crop, cond


def prompts_for(class_name: str) -> list[str]:
    crop, cond = natural_names(class_name)
    if cond == "healthy":
        return [f"a photo of a healthy {crop} leaf.", f"healthy {crop} plant leaf without disease."]
    return [f"a photo of a {crop} leaf with {cond}.", f"{crop} plant disease: {cond}."]


# --------------------------------------------------------------------------- phân loại trên đặc trưng (thuần, có test)
def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def knn_scores(query: np.ndarray, gallery: np.ndarray, gallery_labels: np.ndarray, k: int, classes: int,
               allowed: np.ndarray | None = None) -> np.ndarray:
    """Điểm theo lớp [n, classes]: tổng độ tương đồng cosin của k láng giềng gần nhất. `allowed` [n, classes] (bool) giới hạn lớp được xét."""
    sims = l2_normalize(query) @ l2_normalize(gallery).T
    out = np.zeros((len(query), classes), dtype=np.float64)
    for i in range(len(query)):
        row = sims[i]
        if allowed is not None:  # chỉ láng giềng thuộc các lớp được phép
            row = np.where(allowed[i][gallery_labels], row, -np.inf)
        k_eff = min(k, int(np.isfinite(row).sum()))
        if k_eff <= 0:
            continue
        top = np.argpartition(-row, k_eff - 1)[:k_eff]
        np.add.at(out[i], gallery_labels[top], np.maximum(row[top], 0.0))
    return out


def zero_shot_scores(image_feats: np.ndarray, text_feats: np.ndarray, temperature: float = 100.0) -> np.ndarray:
    logits = temperature * l2_normalize(image_feats) @ l2_normalize(text_feats).T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def crop_allowed(classes: list[str], crop_index: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """[n, classes] bool: lớp cùng loại cây với nhãn thật (người dùng chọn đúng cây)."""
    class_crop = np.array([c.split("___", 1)[0] for c in classes])
    return class_crop[None, :] == class_crop[targets][:, None]


# --------------------------------------------------------------------------- dữ liệu
def build_items(limit: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    from scripts.build_external_eval_set import EXTERNAL_DIR, MANIFEST  # chỉ hằng số; không nạp gói nặng

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    base = reg.abs_path(EXTERNAL_DIR)
    manifest_path = reg.abs_path(MANIFEST)
    locked = json.loads(reg.abs_path(reg.EXTERNAL_EVAL_SET_JSON).read_text(encoding="utf-8"))["manifest_sha256"]
    if sha256_file(manifest_path) != locked:
        raise SystemExit("manifest PlantWild khác mã băm đã khóa")
    sets: dict[str, list[dict[str, Any]]] = {"ext_strict": [], "ext_healthy": [], "pd_val": [], "pd_test": []}
    for row in json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]:
        key = "ext_strict" if row["tier"] == "strict" else "ext_healthy"
        sets[key].append({"path": base / row["relative_path"], "target": int(row["class_id"]), "key": row["byte_sha256"]})
    for split, name in (("pd_val", "val.jsonl"), ("pd_test", f"{reg.HOLDOUT_SPLIT}.jsonl")):
        for r in _jsonl(release / "manifests" / name):
            if r["source_domain"] == "plantdoc":
                sets[split].append({"path": root / r["relative_path"], "target": int(r["class_id"]), "key": r["sample_id"]})
    rng = random.Random(SEED)
    train = _jsonl(release / "manifests" / "train.jsonl")
    gallery = [{"path": root / r["relative_path"], "target": int(r["class_id"]), "key": r["sample_id"], "source": "plantdoc"}
               for r in train if r["source_domain"] == "plantdoc"]
    by_class: dict[int, list[dict[str, Any]]] = {}
    for r in train:
        if r["source_domain"] == "plantvillage":
            by_class.setdefault(int(r["class_id"]), []).append(r)
    for cid in sorted(by_class):
        pool = sorted(by_class[cid], key=lambda r: r["sample_id"])
        for r in rng.sample(pool, min(GALLERY_PV_PER_CLASS, len(pool))):
            gallery.append({"path": root / r["relative_path"], "target": cid, "key": r["sample_id"], "source": "plantvillage"})
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
        gallery = gallery[: limit * 3]
    return sets, gallery


# --------------------------------------------------------------------------- trích đặc trưng
def extract_features(paths: list[Path], threads: int, batch: int = 32) -> dict[str, np.ndarray]:
    import open_clip
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(max(1, threads))
    to_rgb = _load_to_rgb()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(BIOCLIP)
    clip_model.eval()
    dino_processor = AutoImageProcessor.from_pretrained(DINO)
    dino = AutoModel.from_pretrained(DINO).eval()
    clip_feats, dino_feats = [], []
    started = time.time()
    for start in range(0, len(paths), batch):
        images = []
        for p in paths[start:start + batch]:
            with Image.open(p) as handle:
                images.append(to_rgb(handle).copy())
        with torch.inference_mode():
            clip_feats.append(clip_model.encode_image(torch.stack([clip_preprocess(i) for i in images])).numpy())
            inputs = dino_processor(images=images, return_tensors="pt")
            dino_feats.append(dino(**inputs).last_hidden_state[:, 0].numpy())  # token CLS
        done = start + len(images)
        if done % (batch * 8) < batch or done == len(paths):
            rate = done / max(time.time() - started, 1e-9)
            print(f"[fm] {done}/{len(paths)} ({rate:.1f} ảnh/s, còn ~{(len(paths) - done) / max(rate, 1e-9) / 60:.1f} phút)", flush=True)
    return {"bioclip": np.concatenate(clip_feats), "dinov2_small": np.concatenate(dino_feats)}


def text_features(classes: list[str]) -> np.ndarray:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(BIOCLIP)
    model.eval()
    tokenizer = open_clip.get_tokenizer(BIOCLIP)
    rows = []
    with torch.inference_mode():
        for name in classes:
            emb = model.encode_text(tokenizer(prompts_for(name))).numpy()
            rows.append(l2_normalize(emb).mean(axis=0))
    return np.stack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    sets, gallery = build_items(args.limit)
    order = [(name, item) for name, items in sets.items() for item in items]
    paths = [item["path"] for _, item in order] + [g["path"] for g in gallery]
    fp = hashlib.sha256(("|".join(str(i["key"]) for _, i in order) + "#" + "|".join(g["key"] for g in gallery)).encode()).hexdigest()
    npz = reg.abs_path(NPZ)
    feats = None
    if args.reuse and npz.is_file() and not args.limit:
        data = np.load(npz, allow_pickle=False)
        if str(data["fingerprint"]) == fp:
            feats = {k: data[k] for k in ("bioclip", "dinov2_small", "text")}
            print("[fm] dùng lại đặc trưng đã lưu")
    if feats is None:
        print(f"[fm] {len(order)} ảnh cần đo + {len(gallery)} ảnh thư viện")
        feats = extract_features(paths, args.threads)
        feats["text"] = text_features(classes)
        if not args.limit:
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, fingerprint=np.array(fp), **feats)

    n_eval = len(order)
    gal_labels = np.array([g["target"] for g in gallery])
    names = np.array([name for name, _ in order])
    targets = np.array([item["target"] for _, item in order])
    results: dict[str, Any] = {}
    for set_name in sets:
        mask = names == set_name
        if not mask.any():
            continue
        t = targets[mask]
        oracle = crop_allowed(classes, None, t)
        entry: dict[str, Any] = {"n": int(mask.sum()), "methods": {}}

        def summarize(scores: np.ndarray, scores_oracle: np.ndarray) -> dict[str, Any]:
            top1 = scores.argmax(1) == t
            top3 = (np.argsort(-scores, axis=1)[:, :3] == t[:, None]).any(1)
            return {"top1": round(100 * float(top1.mean()), 2), "top3": round(100 * float(top3.mean()), 2),
                    "top1_oracle_crop": round(100 * float((scores_oracle.argmax(1) == t).mean()), 2)}

        zs = zero_shot_scores(feats["bioclip"][:n_eval][mask], feats["text"])
        zs_oracle = np.where(oracle, zs, 0.0)
        entry["methods"]["bioclip_zero_shot"] = summarize(zs, zs_oracle)
        for model in ("bioclip", "dinov2_small"):
            query, gal = feats[model][:n_eval][mask], feats[model][n_eval:]
            for k in K_VALUES:
                free = knn_scores(query, gal, gal_labels, k, len(classes))
                restricted = knn_scores(query, gal, gal_labels, k, len(classes), allowed=oracle)
                entry["methods"][f"{model}_knn_k{k}"] = summarize(free, restricted)
        results[set_name] = entry

    report = {
        "evidence": "đo cục bộ trên CPU; không huấn luyện lớp nào (zero-shot và kNN trên đặc trưng đóng băng)",
        "models": {"bioclip": BIOCLIP, "dinov2_small": DINO},
        "gallery": {"plantdoc_train_leaves": int(sum(1 for g in gallery if g["source"] == "plantdoc")),
                    "plantvillage_per_class": GALLERY_PV_PER_CLASS, "total": len(gallery), "seed": SEED},
        "k_values": list(K_VALUES),
        "external_manifest_sha256": json.loads(reg.abs_path(reg.EXTERNAL_EVAL_SET_JSON).read_text(encoding="utf-8"))["manifest_sha256"],
        "prompts_example": {c: prompts_for(c) for c in classes[:3]},
        "caveat": "Không so trực tiếp với MobileNetV3 về công bằng: mô hình nền có số tham số lớn hơn nhiều và được tiền huấn luyện trên dữ liệu rất khác; "
                  "đây là mốc tham khảo, không thay mô hình của đồ án. Thư viện kNN dùng ảnh PlantDoc/PlantVillage đã nằm trong tập học của MobileNetV3.",
        "sets": results,
    }
    if args.limit:
        print(json.dumps({s: {m: v["top1"] for m, v in e["methods"].items()} for s, e in results.items()}, ensure_ascii=False))
        print("[fm] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.FOUNDATION_MODEL_EVAL_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[fm] đã ghi {reg.FOUNDATION_MODEL_EVAL_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

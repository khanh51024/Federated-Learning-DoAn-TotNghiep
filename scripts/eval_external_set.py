"""Đo 7 mô hình đã công bố trên tập ảnh ngoài đồng độc lập PlantWild (đã loại ảnh trùng PlantDoc/PlantVillage).

Đây là phép đo gần nhất với "ảnh bất kỳ trên mạng" mà đồ án có: ảnh do người dùng Google/Bing chụp ngoài đồng, nhãn do nhóm PlantWild
gán, chưa từng nằm trong train/val/test của đồ án. Hai tầng báo riêng (xem scripts/build_external_eval_set.py):
  ext_strict   : ảnh bệnh, nhãn từ PlantWild v2 (chuyên gia nông nghiệp đã rà)
  ext_healthy  : thư mục "<cây> leaf" của PlantWild v1 hiểu là lá khỏe (nhãn NGẦM ĐỊNH, không chắc là khỏe thật)

Mỗi ảnh đi qua 6 góc nhìn với Centralized và FedAvg (TTA, giữ tỉ lệ, chọn loại cây... như scripts/eval_inference_variants.py) và một
góc nhìn với 5 Local-only. Khóa manifest bằng SHA-256: kết quả chỉ hợp lệ nếu manifest trùng mã băm đã ghi.

Chạy:
    python scripts/eval_external_set.py            # vài chục phút CPU; ghi artifacts/external_eval_results.json
    python scripts/eval_external_set.py --reuse    # dùng logits đã lưu nếu còn khớp
    python scripts/eval_external_set.py --limit 40 # thử nhanh, KHÔNG ghi artifacts/
"""
from __future__ import annotations

import argparse
import hashlib
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

NPZ = "data/reeval/external_logits.npz"  # gitignored (data/*)
SEED = 20261020
LOCALS = tuple(f"local_{i:02d}" for i in range(5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_inference(items: list[dict[str, Any]], threads: int, cache: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """logits[model] = [N, V, 38]; Centralized/FedAvg có đủ 6 góc nhìn, Local-only chỉ góc nhìn 'squash' (V=1)."""
    import torch
    from PIL import Image

    from scripts.eval_inference_variants import VIEWS, view_images
    from scripts.evaluate_published_models import _load_contract_transforms, _load_model
    from src.common.image_io import to_rgb

    torch.set_num_threads(max(1, threads))
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    transform = _load_contract_transforms(bundle)
    paths = {"centralized": bundle / "centralized" / "checkpoint_best.pt", "fedavg": bundle / "fedavg" / "checkpoint_best.pt",
             **{f"local_{i:02d}": bundle / "local" / f"client_{i:02d}" / "checkpoint_best.pt" for i in range(5)}}
    models = {name: _load_model(path) for name, path in paths.items()}
    out = {"centralized": np.zeros((len(items), len(VIEWS), 38), np.float32), "fedavg": np.zeros((len(items), len(VIEWS), 38), np.float32),
           **{n: np.zeros((len(items), 1, 38), np.float32) for n in LOCALS}}
    started = time.time()
    for i, item in enumerate(items):
        try:
            with Image.open(item["path"]) as handle:
                img = to_rgb(handle)
        except Exception:  # noqa: BLE001 — ảnh hỏng: giữ logits 0, ghi lại ở nơi gọi
            item["unreadable"] = True
            continue
        views = view_images(img)
        batch = torch.stack([transform(views[v]) for v in VIEWS])
        with torch.inference_mode():
            for name in ("centralized", "fedavg"):
                out[name][i] = models[name](batch).numpy()
            for name in LOCALS:
                out[name][i, 0] = models[name](batch[:1]).numpy()[0]
        if (i + 1) % 250 == 0:
            rate = (i + 1) / max(time.time() - started, 1e-9)
            print(f"[external] {i + 1}/{len(items)} ({rate:.1f} ảnh/s, còn ~{(len(items) - i - 1) / max(rate, 1e-9) / 60:.1f} phút)", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    from scripts.build_external_eval_set import EXTERNAL_DIR, MANIFEST
    from scripts.eval_inference_variants import (
        VARIANTS, VIEWS, accuracy_ci, class_names_crop_index, macro_f1, mean_over_views, restrict_to_crop, selective_table,
        softmax, topk_correct,
    )
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[external] bundle chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    manifest_path = reg.abs_path(MANIFEST)
    if not manifest_path.is_file():
        print("[external] thiếu manifest; chạy scripts/build_external_eval_set.py", file=sys.stderr)
        return 1
    manifest_sha = sha256_file(manifest_path)
    locked = json.loads(reg.abs_path(reg.EXTERNAL_EVAL_SET_JSON).read_text(encoding="utf-8"))["manifest_sha256"]
    if manifest_sha != locked:
        print("[external] manifest khác mã băm đã khóa trong artifacts/external_eval_set.json: dừng", file=sys.stderr)
        return 1
    classes = evidence["classes"]
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
    base = reg.abs_path(EXTERNAL_DIR)
    items = [{"key": r["byte_sha256"], "path": base / r["relative_path"], "target": int(r["class_id"]), "tier": r["tier"],
              "name": r["target_class"]} for r in rows]
    for item, row in zip(items, rows):  # kiểm byte từng ảnh trước khi dùng
        if args.limit:
            continue
        if sha256_file(item["path"]) != row["byte_sha256"]:
            print(f"[external] ảnh đổi byte: {row['relative_path']}", file=sys.stderr)
            return 1
    if args.limit:
        rng = np.random.default_rng(SEED)
        items = [items[i] for i in sorted(rng.choice(len(items), size=min(args.limit, len(items)), replace=False))]

    digest = hashlib.sha256()
    for m in evidence["models"]:
        digest.update(m["checkpoint_sha256"].encode())
    digest.update(manifest_sha.encode())
    digest.update(str(len(items)).encode())
    fp = digest.hexdigest()
    npz = reg.abs_path(NPZ)
    logits = None
    if args.reuse and npz.is_file() and not args.limit:
        data = np.load(npz, allow_pickle=False)
        if str(data["fingerprint"]) == fp:
            logits = {k: data[k] for k in data.files if k != "fingerprint"}
            print("[external] dùng lại logits đã lưu")
    if logits is None:
        logits = run_inference(items, args.threads)
        if not args.limit:
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, fingerprint=np.array(fp), **logits)

    targets = np.array([it["target"] for it in items])
    tiers = np.array([it["tier"] for it in items])
    class_crop, crop_names = class_names_crop_index(classes)
    true_crops = class_crop[targets]
    healthy_ids = np.array([c.endswith("___healthy") for c in classes])
    rng = np.random.default_rng(SEED)
    results: dict[str, Any] = {}
    for tier in ("strict", "healthy_assumed"):
        mask = tiers == tier
        if not mask.any():
            continue
        t, groups = targets[mask], [str(i) for i in np.flatnonzero(mask)]
        entry: dict[str, Any] = {"n": int(mask.sum()), "classes": int(len(set(t.tolist()))), "models": {}}
        names = ("centralized", "fedavg", "ensemble", *LOCALS)
        for name in names:
            per_variant: dict[str, Any] = {}
            for variant, views in VARIANTS.items():
                if name in LOCALS:
                    if variant != "base":
                        continue
                    p = softmax(logits[name][mask][:, 0])
                elif name == "ensemble":
                    p = np.mean([mean_over_views(softmax(logits[m][mask]), views, VIEWS) for m in ("centralized", "fedavg")], axis=0)
                else:
                    p = mean_over_views(softmax(logits[name][mask]), views, VIEWS)
                for mode in ("free", "oracle_crop"):
                    q = p if mode == "free" else restrict_to_crop(p, classes, true_crops[mask])
                    pred = q.argmax(1)
                    correct = pred == t
                    key = variant if mode == "free" else f"{variant}+oracle_crop"
                    per_variant[key] = {
                        "top1": round(100 * float(correct.mean()), 2), "top3": round(100 * float(topk_correct(q, t, 3).mean()), 2),
                        "crop_accuracy": round(100 * float((class_crop[pred] == true_crops[mask]).mean()), 2),
                        "macro_f1_supported": round(100 * (macro_f1(t, pred) or 0.0), 2),
                        "top1_ci95": accuracy_ci(groups, correct, rng),
                    }
                    if variant == "base" and mode == "free":
                        per_variant[key]["selective_answering"] = selective_table(q.max(1), correct)
                        if tier == "healthy_assumed":  # lá khỏe bị báo là bệnh = báo động giả
                            per_variant[key]["predicted_as_disease_share"] = round(100 * float((~healthy_ids[pred]).mean()), 2)
            entry["models"][name] = per_variant
        # từng lớp cho FedAvg và Centralized (gốc)
        per_class = {}
        for name in ("centralized", "fedavg"):
            pred = softmax(logits[name][mask][:, 0]).argmax(1)
            per_class[name] = {classes[c]: {"n": int((t == c).sum()), "recall": round(float((pred[t == c] == c).mean()), 4)}
                               for c in sorted(set(t.tolist()))}
        entry["per_class_recall_base"] = per_class
        pred_c = softmax(logits["centralized"][mask][:, 0]).argmax(1)
        entry["top_confusions_centralized"] = [
            {"target": classes[a], "predicted": classes[b], "count": int(n)}
            for (a, b), n in Counter(zip(t.tolist(), pred_c.tolist())).most_common(40) if a != b][:12]
        results[tier] = entry

    report = {
        "evidence": "tập ngoài đồng PlantWild đã loại ảnh trùng; chưa từng nằm trong train/val/test của đồ án",
        "manifest": MANIFEST, "manifest_sha256": manifest_sha, "images": len(items),
        "unreadable_images": int(sum(1 for i in items if i.get("unreadable"))),
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "license_note": "PlantWild CC BY-NC-ND 4.0: chỉ dùng để đánh giá.",
        "caveat": "ext_healthy dùng nhãn ngầm định (thư mục '<cây> leaf'). PlantWild lấy từ Google nên vẫn có thể còn ảnh gần trùng với "
                  "dữ liệu train mà dHash 64 bit không bắt được; số này là cận trên của độ chính xác ngoài đồng.",
        "tiers": results,
    }
    if args.limit:
        print(json.dumps({t: {m: v["base"]["top1"] for m, v in e["models"].items()} for t, e in results.items()}, ensure_ascii=False))
        print("[external] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(reg.EXTERNAL_EVAL_RESULTS_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[external] đã ghi {reg.EXTERNAL_EVAL_RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

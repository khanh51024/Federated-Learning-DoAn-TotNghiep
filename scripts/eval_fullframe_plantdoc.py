"""Đo mô hình đã công bố khi nhận ảnh PlantDoc NGUYÊN KHUNG (chưa cắt) so với ảnh đã cắt lá.

Câu hỏi: nếu người dùng tải cả ảnh (cả cây, nhiều lá, nền rối) thì mô hình còn đúng bao nhiêu, và một bước
cắt lá (detector) có đáng làm không? Script này đo "giới hạn trên" của mọi detector: cắt đúng theo khung do người gán.

Bốn cách đưa ảnh vào CÙNG 7 checkpoint, cùng transform canonical_v1 của bundle (thêm detector_top1 khi có --detector):
  published_crop : ảnh cắt đã có sẵn trong release (số công bố; dùng để kiểm chéo)
  annotator_crop : tự cắt lại từ ảnh gốc theo khung + lề 8% (đúng crop_geometry của nhánh train)
  full_frame     : cả ảnh gốc
  center_square  : hình vuông ở giữa ảnh gốc (cách người dùng hay chụp)
  detector_top1  : (--detector) detector lá tìm khung điểm cao nhất rồi cắt lề 8%; không thấy lá thì dùng cả ảnh

Chỉ đánh giá ảnh thuộc tập kiểm tra (val) và tập giữ riêng (legacy_diagnostic_test) của release: các cảnh này không
nằm trong train. Ảnh gốc tải từ kho PlantDoc công khai và PHẢI khớp byte_sha256 trong detection.json.

Chạy:
    python scripts/eval_fullframe_plantdoc.py --download   # tải ảnh gốc còn thiếu (kiểm SHA-256)
    python scripts/eval_fullframe_plantdoc.py              # đo và ghi artifacts/fullframe_eval.json
    python scripts/eval_fullframe_plantdoc.py --detector data/leaf_detector/leaf_detector.pt  # thêm chế độ detector
    python scripts/eval_fullframe_plantdoc.py --limit 20   # thử nhanh (KHÔNG ghi artifacts/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SOURCE_URL = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/master"
CACHE_DIR = "data/fullframe_plantdoc"  # gitignored (data/*)
OUT_JSON = "artifacts/fullframe_eval.json"
MARGIN = 0.08  # crop_policy bbox_margin_8pct_v1
MODES = ("published_crop", "annotator_crop", "full_frame", "center_square")
MODEL_ORDER = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
SPLITS = {"val": "val.jsonl", "test": "legacy_diagnostic_test.jsonl"}


# --------------------------------------------------------------------------- hình học (thuần, có test)
def crop_box(width: int, height: int, bbox_xyxy: list[float], margin: float = MARGIN) -> tuple[int, int, int, int]:
    """Giống `crop_bounding_box` của nhánh train (plant_data_contract/crop_geometry.py): lề mỗi phía, kẹp trong ảnh."""
    xmin, ymin, xmax, ymax = bbox_xyxy
    dx = max(0.0, xmax - xmin) * margin
    dy = max(0.0, ymax - ymin) * margin
    left = max(0, int(math.floor(xmin - dx)))
    top = max(0, int(math.floor(ymin - dy)))
    right = min(width, int(math.ceil(xmax + dx)))
    bottom = min(height, int(math.ceil(ymax + dy)))
    if right <= left:
        right = min(width, left + 1)
        if right <= left and left > 0:
            left = right - 1
    if bottom <= top:
        bottom = min(height, top + 1)
        if bottom <= top and top > 0:
            top = bottom - 1
    return left, top, right, bottom


def center_square(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def single_class_scenes(rows: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, int]:
    """image_id -> class_id cho những cảnh mà MỌI khung (trong tập đang xét) cùng một lớp.
    Chỉ với các cảnh này, nhãn của "cả ảnh" hay "hình vuông giữa" mới xác định được."""
    by_scene: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        ann = annotations[row["annotation_id"]]
        by_scene[ann["image_id"]].add(int(row["class_id"]))
    return {scene: next(iter(classes)) for scene, classes in by_scene.items() if len(classes) == 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- dữ liệu
def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_inputs(release: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    detection = json.loads((release / "annotations" / "detection.json").read_text(encoding="utf-8"))
    annotations = {a["annotation_id"]: a for a in detection["annotations"]}
    images = {i["image_id"]: i for i in detection["images"]}
    rows = {
        split: [r for r in _jsonl(release / "manifests" / name) if r["source_domain"] == "plantdoc"]
        for split, name in SPLITS.items()
    }
    return annotations, images, rows


def scene_path(cache: Path, image: dict[str, Any]) -> Path:
    # relative_path: PlantDoc-Object-Detection-Dataset/{TRAIN|TEST}/<file>
    parts = image["relative_path"].split("/")
    return cache / parts[-2] / parts[-1]


def download_missing(cache: Path, images: dict[str, dict[str, Any]], wanted: set[str]) -> dict[str, int]:
    """Tải ảnh gốc còn thiếu vào thư mục riêng; chỉ giữ tệp khớp byte_sha256 (nguồn bên ngoài = dữ liệu không tin cậy)."""
    stats = {"present": 0, "downloaded": 0, "rejected_sha": 0, "failed": 0}
    for image_id in sorted(wanted):
        image = images[image_id]
        target = scene_path(cache, image)
        if target.is_file() and sha256_file(target) == image["byte_sha256"]:
            stats["present"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        folder, name = image["relative_path"].split("/")[-2:]
        url = f"{SOURCE_URL}/{folder}/{urllib.parse.quote(name)}"
        tmp = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response, tmp.open("wb") as out:
                    out.write(response.read(64 << 20))
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(1.5 * (attempt + 1))
        else:
            stats["failed"] += 1
            tmp.unlink(missing_ok=True)
            continue
        if sha256_file(tmp) != image["byte_sha256"]:
            stats["rejected_sha"] += 1
            tmp.unlink(missing_ok=True)
            continue
        tmp.replace(target)
        stats["downloaded"] += 1
    return stats


# --------------------------------------------------------------------------- đánh giá
def _open_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:  # noqa: BLE001 — như normalize_exif_orientation của nhóm train
            pass
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            from PIL import Image as _Image

            background = _Image.new("RGB", rgba.size, (0, 0, 0))
            background.paste(rgba, mask=rgba.split()[3])
            return background
        return image.convert("RGB")


def evaluate(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from scripts.evaluate_published_models import _load_contract_transforms, _load_model
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[fullframe] bundle chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    release = reg.abs_path(reg.DATASET_SOURCES_DIR) / "mixed" / "pv_pd_v3"
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    cache = reg.abs_path(CACHE_DIR)
    if not (release / "annotations" / "detection.json").is_file():
        print("[fullframe] thiếu data/dataset_sources/mixed/pv_pd_v3 (nhánh dataset-sources)", file=sys.stderr)
        return 1

    annotations, images, rows_by_split = load_inputs(release)
    if args.limit:
        rows_by_split = {k: v[: args.limit] for k, v in rows_by_split.items()}
    wanted = {annotations[r["annotation_id"]]["image_id"] for rows in rows_by_split.values() for r in rows}

    if args.download:
        stats = download_missing(cache, images, wanted)
        print(f"[fullframe] ảnh gốc: {stats}")
    missing = {i for i in wanted if not scene_path(cache, images[i]).is_file()}
    if missing and not args.download:
        print(f"[fullframe] thiếu {len(missing)}/{len(wanted)} ảnh gốc (chạy với --download để thử tải)", file=sys.stderr)
    excluded = {"scenes": len(missing), "annotations": 0, "image_ids": sorted(missing)}
    if missing:
        # vài tệp không còn ở kho công khai (404): bỏ hẳn các khung của cảnh đó và ghi rõ, không đoán
        kept = {split: [r for r in rows if annotations[r["annotation_id"]]["image_id"] not in missing] for split, rows in rows_by_split.items()}
        excluded["annotations"] = sum(len(rows_by_split[k]) - len(kept[k]) for k in kept)
        rows_by_split, wanted = kept, wanted - missing
        print(f"[fullframe] bỏ {excluded['scenes']} cảnh ({excluded['annotations']} khung) vì ảnh gốc không còn ở kho công khai")
    # kiểm lại SHA-256 từng ảnh gốc trước khi dùng
    bad = [i for i in wanted if sha256_file(scene_path(cache, images[i])) != images[i]["byte_sha256"]]
    if bad:
        print(f"[fullframe] {len(bad)} ảnh gốc sai SHA-256: dừng", file=sys.stderr)
        return 1

    torch.set_num_threads(max(1, args.threads))
    transform = _load_contract_transforms(bundle)
    detector = None
    detector_meta: dict[str, Any] = {}
    if args.detector:
        from scripts.train_leaf_detector import load_detector

        detector, payload = load_detector(Path(args.detector))
        threshold = float(args.detector_threshold or payload.get("score_threshold", 0.5))
        detector_meta = {"checkpoint_sha256": sha256_file(Path(args.detector)), "score_threshold": threshold,
                         "train_images": payload.get("train_images"), "epochs": payload.get("epochs"),
                         "pretrained_backbone": payload.get("pretrained_backbone")}
        detector_fallbacks = 0
    files = {
        "centralized": bundle / "centralized" / "checkpoint_best.pt",
        "fedavg": bundle / "fedavg" / "checkpoint_best.pt",
        **{f"local_{i:02d}": bundle / "local" / f"client_{i:02d}" / "checkpoint_best.pt" for i in range(5)},
    }
    models = {name: _load_model(path) for name, path in files.items()}
    expected_sha = {m["model_id"]: m.get("checkpoint_sha256") for m in evidence["models"]}
    for name, path in files.items():
        if expected_sha.get(name) and sha256_file(path) != expected_sha[name]:
            print(f"[fullframe] checkpoint {name} sai SHA-256", file=sys.stderr)
            return 1

    def predict(batch: list[Any]) -> dict[str, np.ndarray]:
        with torch.inference_mode():
            x = torch.stack([transform(img) for img in batch])
            return {name: model(x).argmax(1).numpy() for name, model in models.items()}

    results: dict[str, Any] = {}
    cross: dict[str, Any] = {}
    started = time.time()
    for split, rows in rows_by_split.items():
        single = single_class_scenes(rows, annotations)
        # dự đoán theo (chế độ, khóa): published/annotator theo annotation_id; full/center theo image_id
        preds: dict[tuple[str, str], dict[str, int]] = {}
        sizes_mismatch = 0

        def run(mode: str, key: str, image) -> None:
            out = predict([image])
            preds[(mode, key)] = {name: int(value[0]) for name, value in out.items()}

        scene_cache: dict[str, Any] = {}
        for index, row in enumerate(rows):
            ann = annotations[row["annotation_id"]]
            published = _open_rgb(dataset_root / row["relative_path"])
            run("published_crop", row["annotation_id"], published)
            scene = ann["image_id"]
            if scene not in scene_cache:
                scene_cache = {scene: _open_rgb(scene_path(cache, images[scene]))}  # giữ một cảnh trong bộ nhớ
            full = scene_cache[scene]
            width, height = full.size
            box = crop_box(width, height, ann["bbox_xyxy"])
            crop = full.crop(box)
            if abs(crop.size[0] - published.size[0]) > 1 or abs(crop.size[1] - published.size[1]) > 1:
                sizes_mismatch += 1
            run("annotator_crop", row["annotation_id"], crop)
            if scene in single and ("full_frame", scene) not in preds:
                run("full_frame", scene, full)
                run("center_square", scene, full.crop(center_square(width, height)))
                if detector is not None:
                    import torchvision.transforms.functional as TF

                    with torch.inference_mode():
                        found = detector([TF.to_tensor(full)])[0]
                    keep = [i for i in range(len(found["scores"])) if float(found["scores"][i]) >= threshold]
                    if keep:
                        top = max(keep, key=lambda i: float(found["scores"][i]))
                        run("detector_top1", scene, full.crop(crop_box(width, height, found["boxes"][top].tolist())))
                    else:
                        detector_fallbacks += 1
                        preds[("detector_top1", scene)] = dict(preds[("full_frame", scene)])
            if (index + 1) % 50 == 0:
                print(f"[fullframe] {split} {index + 1}/{len(rows)} {time.time() - started:.0f}s", flush=True)

        def accuracy(mode: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
            entry: dict[str, Any] = {"n": len(subset)}
            for name in MODEL_ORDER:
                correct = 0
                for row in subset:
                    key = row["annotation_id"] if mode in ("published_crop", "annotator_crop") else annotations[row["annotation_id"]]["image_id"]
                    correct += int(preds[(mode, key)][name] == int(row["class_id"]))
                entry[name] = {"correct": correct, "top1": round(correct / len(subset), 4) if subset else None}
            return entry

        subset_single = [r for r in rows if annotations[r["annotation_id"]]["image_id"] in single]
        results[split] = {
            "annotations": len(rows),
            "scenes": len({annotations[r["annotation_id"]]["image_id"] for r in rows}),
            "single_class_scenes": len(single),
            "annotations_in_single_class_scenes": len(subset_single),
            "all_annotations": {m: accuracy(m, rows) for m in ("published_crop", "annotator_crop")},
            "single_class_scenes_only": {m: accuracy(m, subset_single) for m in (MODES + (("detector_top1",) if detector is not None else ()))},
            "crop_size_mismatch_vs_published": sizes_mismatch,
        }
        # kiểm chéo: cắt lại có cho CÙNG dự đoán như crop đã công bố không?
        same = {
            name: sum(preds[("published_crop", r["annotation_id"])][name] == preds[("annotator_crop", r["annotation_id"])][name] for r in rows)
            for name in MODEL_ORDER
        }
        cross[split] = {name: {"same_prediction": same[name], "of": len(rows)} for name in ("centralized", "fedavg")}

    report = {
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "crop_policy": "bbox_margin_8pct_v1",
        "margin": MARGIN,
        "source": SOURCE_URL,
        "scenes_verified_sha256": len(wanted),
        "excluded_unavailable": excluded,
        "modes": {
            "published_crop": "ảnh cắt đã có trong release (số công bố)",
            "annotator_crop": "tự cắt lại từ ảnh gốc theo khung người gán + lề 8%",
            "full_frame": "cả ảnh gốc (chỉ cảnh mà mọi khung cùng một lớp)",
            "center_square": "hình vuông giữa ảnh gốc (chỉ cảnh mà mọi khung cùng một lớp)",
        },
        "caveat": "Giới hạn trên của detector: khung do người gán. full_frame/center_square chỉ đo trên cảnh một lớp; mỗi "
                  "khung trong cảnh thừa hưởng dự đoán của cả ảnh. Không phải độ chính xác ngoài thực địa.",
        "splits": results,
        "cross_check_recrop_vs_published": cross,
        "detector": ({**detector_meta, "fallback_to_full_frame_scenes": detector_fallbacks} if detector is not None else None),
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1)[:6000])
    if args.limit:
        print("[fullframe] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(OUT_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[fullframe] đã ghi {OUT_JSON}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="tải ảnh gốc PlantDoc còn thiếu (kiểm SHA-256)")
    parser.add_argument("--limit", type=int, default=0, help="chỉ xét N khung đầu mỗi split (thử nhanh)")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--detector", default="", help="checkpoint detector lá (scripts/train_leaf_detector.py)")
    parser.add_argument("--detector-threshold", type=float, default=0.0, help="ngưỡng điểm (mặc định: theo checkpoint)")
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

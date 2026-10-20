"""Dựng tập ảnh ngoài đồng độc lập từ PlantWild để đo nhận diện ảnh "bất kỳ trên mạng".

Vì sao PlantWild: ảnh Google/Bing do người dùng chụp ngoài đồng, được nhóm UQ (ACM MM 2024) gán nhãn, nhiều bệnh trùng với
38 lớp của đồ án. Giấy phép CC BY-NC-ND 4.0: chỉ dùng để đánh giá, không đưa ảnh vào git, không phân phối lại.

Quy tắc đặt trước khi nhìn kết quả mô hình:
  * Chỉ giữ lớp PlantWild KHỚP ĐÚNG NGHĨA với một lớp trong 38 lớp (bảng DISEASE_MAP/HEALTHY_MAP bên dưới). Lớp mơ hồ
    (ví dụ "grape leaf spot" so với Grape Leaf blight) bị bỏ, không đoán.
  * Tầng `strict`  = bệnh, từ PlantWild v2 (bản chuyên gia nông nghiệp đã rà lại nhãn).
    Tầng `healthy_assumed` = thư mục "<cây> leaf" của PlantWild v1 hiểu là lá khỏe; nhãn NGẦM ĐỊNH nên báo riêng.
  * Loại ảnh trùng/gần trùng với PlantDoc (mọi cảnh, kể cả train/val/test) và PlantVillage (dHash 64 bit, Hamming <= 6),
    vì PlantWild cũng lấy từ Google nên có thể trùng ảnh mà mô hình đã học. Loại ảnh trùng nhau trong PlantWild.
  * Khóa manifest bằng SHA-256; mọi lần train sau này dùng đúng manifest này làm tập test ngoài đồng.

Chạy:
    python scripts/build_external_eval_set.py --download-scenes   # tải thêm cảnh PlantDoc (kiểm SHA-256) để so trùng
    python scripts/build_external_eval_set.py                     # dựng manifest + báo cáo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

EXTERNAL_DIR = "data/external"
PLANTWILD_V1 = f"{EXTERNAL_DIR}/plantwild/v1/plantwild/images"
PLANTWILD_V2 = f"{EXTERNAL_DIR}/plantwild/v2/plantwild_v2"
MANIFEST = f"{EXTERNAL_DIR}/plantwild_overlap_manifest.json"
DHASH_CACHE = f"{EXTERNAL_DIR}/dhash_cache.npz"
HAMMING_MAX = 6  # <= 6/64 bit khác nhau = coi là cùng ảnh (nén/resize lại)

# PlantWild v2 (bệnh, nhãn đã rà) -> lớp trong taxonomy của đồ án. Chỉ các cặp khớp đúng nghĩa.
DISEASE_MAP = {
    "apple black rot": "Apple___Black_rot",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "bell pepper bacterial spot": "Pepper,_bell___Bacterial_spot",
    "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
# PlantWild v1: thư mục "<cây> leaf" = lá khỏe (NGẦM ĐỊNH, người gán không ghi "healthy").
HEALTHY_MAP = {
    "apple leaf": "Apple___healthy",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "corn leaf": "Corn_(maize)___healthy",
    "grape leaf": "Grape___healthy",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "peach leaf": "Peach___healthy",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soybean leaf": "Soybean___healthy",
    "strawberry leaf": "Strawberry___healthy",
    "tomato leaf": "Tomato___healthy",
}


# --------------------------------------------------------------------------- thuần (có test)
def dhash64(image: Any) -> int:
    """dHash 64 bit: so sánh độ sáng điểm kề nhau trên ảnh xám 9x8."""
    from PIL import Image

    gray = image.convert("L").resize((9, 8), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_all(query: int, refs: np.ndarray) -> np.ndarray:
    """Khoảng cách Hamming từ `query` tới từng phần tử của mảng uint64 `refs`."""
    xor = np.bitwise_xor(refs, np.uint64(query))
    return _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def nearest_distance(query: int, refs: np.ndarray) -> int:
    return int(hamming_to_all(query, refs).min()) if len(refs) else 64


def class_names_in_maps() -> list[str]:
    return sorted(set(DISEASE_MAP.values()) | set(HEALTHY_MAP.values()))


# --------------------------------------------------------------------------- I/O
def _hash_file(path_text: str) -> tuple[str, int | None]:
    from PIL import Image

    try:
        with Image.open(path_text) as image:
            if image.format == "JPEG":
                image.draft("L", (72, 72))  # giải mã JPEG ở độ phân giải thấp: nhanh hơn nhiều, đủ cho dHash 9x8
            return path_text, dhash64(image)
    except Exception:  # noqa: BLE001 — ảnh hỏng thì bỏ, ghi số lượng
        return path_text, None


def hash_many(paths: list[Path], cache: dict[str, int], workers: int) -> dict[str, int]:
    todo = [str(p) for p in paths if str(p) not in cache]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (path_text, value) in enumerate(pool.map(_hash_file, todo, chunksize=64), 1):
                if value is not None:
                    cache[path_text] = value
                if index % 5000 == 0:
                    print(f"[external] dHash {index}/{len(todo)}", flush=True)
    return cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_scenes(workers: int) -> dict[str, int]:
    """Tải mọi cảnh gốc PlantDoc (kiểm SHA-256) vào cache của eval_fullframe_plantdoc: dùng để so trùng với PlantWild."""
    from scripts import eval_fullframe_plantdoc as ff

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    detection = json.loads((release / "annotations" / "detection.json").read_text(encoding="utf-8"))
    images = {i["image_id"]: i for i in detection["images"]}
    cache = reg.abs_path(ff.CACHE_DIR)
    ids = sorted(images)
    chunks = [set(ids[i::workers]) for i in range(workers)]
    total = {"present": 0, "downloaded": 0, "rejected_sha": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for stats in pool.map(lambda chunk: ff.download_missing(cache, images, chunk), chunks):
            for key, value in stats.items():
                total[key] += value
    return total


def plantdoc_scene_paths() -> list[Path]:
    from scripts import eval_fullframe_plantdoc as ff

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    detection = json.loads((release / "annotations" / "detection.json").read_text(encoding="utf-8"))
    cache = reg.abs_path(ff.CACHE_DIR)
    return [p for p in (ff.scene_path(cache, i) for i in detection["images"]) if p.is_file()]


def collect_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v2 = reg.abs_path(PLANTWILD_V2)
    for folder, target in sorted(DISEASE_MAP.items()):
        for path in sorted((v2 / folder).glob("*.jpg")):
            rows.append({"path": path, "plantwild_class": folder, "target_class": target, "tier": "strict", "source": "plantwild_v2"})
    v1 = reg.abs_path(PLANTWILD_V1)
    for folder, target in sorted(HEALTHY_MAP.items()):
        for path in sorted((v1 / folder).glob("*.jpg")):
            rows.append({"path": path, "plantwild_class": folder, "target_class": target, "tier": "healthy_assumed", "source": "plantwild_v1"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-scenes", action="store_true", help="tải cảnh PlantDoc còn thiếu (kiểm SHA-256)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from web.demo_evidence import load_evidence

    classes = load_evidence()["classes"]
    unknown = [c for c in class_names_in_maps() if c not in classes]
    if unknown:
        print(f"[external] lớp không thuộc taxonomy: {unknown}", file=sys.stderr)
        return 1
    if not (reg.abs_path(PLANTWILD_V2).is_dir() and reg.abs_path(PLANTWILD_V1).is_dir()):
        print("[external] thiếu data/external/plantwild/{v1,v2} (tải từ huggingface.co/datasets/uqtwei2/PlantWild)", file=sys.stderr)
        return 1
    if args.download_scenes:
        print("[external] cảnh PlantDoc:", download_scenes(args.workers))

    cache_path = reg.abs_path(DHASH_CACHE)
    cache: dict[str, int] = {}
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=False)
        cache = {str(k): int(v) for k, v in zip(data["paths"], data["hashes"])}
    candidates = collect_candidates()
    scenes = plantdoc_scene_paths()
    pv_root = reg.abs_path(reg.DATASET_SOURCES_DIR) / "PlantVillage-Dataset" / "raw" / "color"
    pv = sorted(p for p in pv_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[external] ứng viên {len(candidates)}, cảnh PlantDoc {len(scenes)}, PlantVillage {len(pv)}")
    hash_many([c["path"] for c in candidates] + scenes + pv, cache, args.workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, paths=np.array(list(cache)), hashes=np.array(list(cache.values()), dtype=np.uint64))

    refs_pd = np.array([cache[str(p)] for p in scenes if str(p) in cache], dtype=np.uint64)
    refs_pv = np.array([cache[str(p)] for p in pv if str(p) in cache], dtype=np.uint64)
    seen: list[int] = []
    kept: list[dict[str, Any]] = []
    excluded = {"unreadable": 0, "near_duplicate_of_plantdoc": 0, "near_duplicate_of_plantvillage": 0, "duplicate_inside_plantwild": 0}
    nearest_pd: list[int] = []
    for row in candidates:
        key = str(row["path"])
        if key not in cache:
            excluded["unreadable"] += 1
            continue
        h = cache[key]
        d_pd, d_pv = nearest_distance(h, refs_pd), nearest_distance(h, refs_pv)
        nearest_pd.append(d_pd)
        if d_pd <= HAMMING_MAX:
            excluded["near_duplicate_of_plantdoc"] += 1
            continue
        if d_pv <= HAMMING_MAX:
            excluded["near_duplicate_of_plantvillage"] += 1
            continue
        if seen and nearest_distance(h, np.array(seen, dtype=np.uint64)) <= HAMMING_MAX:
            excluded["duplicate_inside_plantwild"] += 1
            continue
        seen.append(h)
        row_out = {k: v for k, v in row.items() if k != "path"}
        row_out.update({
            "relative_path": Path(key).resolve().relative_to(reg.abs_path(EXTERNAL_DIR).resolve()).as_posix(),
            "class_id": classes.index(row["target_class"]),
            "byte_sha256": sha256_file(row["path"]),
            "nearest_plantdoc_hamming": d_pd, "nearest_plantvillage_hamming": d_pv,
        })
        kept.append(row_out)

    by_class: dict[str, dict[str, int]] = {}
    for row in kept:
        by_class.setdefault(row["target_class"], {"count": 0})["count"] += 1
    manifest = {
        "name": "plantwild_external_eval_v1",
        "license_note": "PlantWild: CC BY-NC-ND 4.0. Chỉ để đánh giá; không phân phối lại ảnh.",
        "hamming_max_for_duplicate": HAMMING_MAX,
        "samples": kept,
    }
    manifest_path = reg.abs_path(MANIFEST)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report = {
        "manifest": MANIFEST,
        "manifest_sha256": sha256_file(manifest_path),
        "candidates": len(candidates),
        "kept": len(kept),
        "kept_by_tier": {t: sum(1 for r in kept if r["tier"] == t) for t in ("strict", "healthy_assumed")},
        "classes_covered": len(by_class),
        "classes_with_field_photos_in_training_before": 14,
        "excluded": excluded,
        "reference_images": {"plantdoc_scenes": int(len(refs_pd)), "plantvillage": int(len(refs_pv))},
        "nearest_plantdoc_hamming_histogram": {str(k): int(v) for k, v in zip(*np.unique(np.clip(nearest_pd, 0, 20), return_counts=True))},
        "per_class": by_class,
        "map": {"disease_v2": DISEASE_MAP, "healthy_v1_assumed": HEALTHY_MAP},
        "rule": "Manifest khóa theo SHA-256 trước mọi lần train mới; ảnh trùng PlantDoc/PlantVillage đã loại.",
    }
    out = reg.abs_path(reg.EXTERNAL_EVAL_SET_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidates", "kept", "kept_by_tier", "classes_covered", "excluded", "manifest_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

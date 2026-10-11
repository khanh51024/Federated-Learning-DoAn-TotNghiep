"""Dựng thư viện ảnh mẫu có nhãn thật cho trang Chẩn đoán của web.

Nguồn: data/demo_images/labels.csv + 19 ảnh (gitignored, lấy từ tập VALIDATION của release mixed_pv_pd_v3).
Đích:  web/static/samples/demo/  (đi theo git và lên server; server không có dataset)
  - <file>.jpg|png   ảnh gốc, SAO CHÉP NGUYÊN BYTE (đây là ảnh được gửi vào mô hình)
  - thumb/<id>.jpg   ảnh xem trước nhỏ (chỉ để hiển thị lưới; không dùng để suy luận)
  - catalog.json     nhãn thật, độ khó, nguồn, recall của lớp đó, sha256, kích thước, kết quả kiểm thuộc validation

Kiểm trước khi chép (không đạt thì dừng, không ghi gì):
  - nhãn thuộc đúng 38 lớp của taxonomy (artifacts/client_shard_class_counts.json)
  - sha256 của ảnh khớp cột sha256_12
  - nếu có release trên đĩa: sample_id của ảnh nằm trong manifest validation và KHÔNG nằm trong train

Chạy: python scripts/build_demo_gallery.py
Không có data/demo_images (vd. server): giữ nguyên catalog đã sinh; chưa có catalog thì thoát mã 1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

THUMB_MAX = 176
DIFFICULTY_KEYS = {"Rất dễ": "very_easy", "Dễ": "easy", "Vừa": "medium", "Khó": "hard", "Rất khó": "very_hard"}
SOURCE_KEYS = {"PlantVillage": "plantvillage", "PlantDoc": "plantdoc"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_difficulty(text: str) -> tuple[str, str, bool]:
    """'Rất khó, ngoài đồng' -> ('Rất khó', 'very_hard', True)."""
    base = text.split(",")[0].strip()
    if base not in DIFFICULTY_KEYS:
        raise ValueError(f"độ khó không nhận ra: {text!r}")
    return base, DIFFICULTY_KEYS[base], "ngoài đồng" in text


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validation_membership(entries: list[tuple[str, str, str]], classes: list[str]) -> dict[str, Any]:
    """Đối chiếu từng ảnh (sample_id, sha256 byte của file, nhãn) với manifest release nếu có trên đĩa.

    "verified" chỉ khi: sample_id nằm trong val.jsonl VÀ byte_sha256 của dòng đó đúng bằng sha256 của file VÀ nhãn trùng
    class_id của dòng đó VÀ sample_id/byte_sha256 không xuất hiện trong train.jsonl."""
    manifests = reg.abs_path(reg.RELEASE_V3_DIR) / "manifests"
    val_path, train_path = manifests / "val.jsonl", manifests / "train.jsonl"
    if not (val_path.is_file() and train_path.is_file()):
        return {"status": "not_verified", "reason": "không có release trên máy dựng; chưa đối chiếu manifest", "in_val": None}
    val_rows = {r["sample_id"]: r for r in _jsonl_rows(val_path)}
    train_rows = _jsonl_rows(train_path)
    train_ids = {r["sample_id"] for r in train_rows}
    train_hashes = {r["byte_sha256"] for r in train_rows}
    problems = []
    for sample_id, digest, label in entries:
        row = val_rows.get(sample_id)
        if row is None:
            problems.append(f"{sample_id}: không nằm trong val.jsonl")
            continue
        if row.get("byte_sha256") != digest:
            problems.append(f"{sample_id}: byte_sha256 trong manifest khác file")
        class_id = row.get("class_id")
        if not isinstance(class_id, int) or not 0 <= class_id < len(classes) or classes[class_id] != label:
            problems.append(f"{sample_id}: nhãn khác class_id của manifest")
        if sample_id in train_ids or digest in train_hashes:
            problems.append(f"{sample_id}: xuất hiện trong train.jsonl")
    if problems:
        raise ValueError("ảnh mẫu không khớp manifest validation: " + "; ".join(problems))
    return {
        "status": "verified",
        "reason": "sample_id, byte_sha256 và nhãn khớp val.jsonl; không có trong train.jsonl",
        "in_val": True,
    }


def build(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    csv_path = source_dir / "labels.csv"
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError("labels.csv rỗng")
    shard = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))
    classes = set(shard.get("classes") or [])
    if len(classes) != 38:
        raise ValueError("không lấy được 38 lớp từ báo cáo shard")

    plan = []
    for row in rows:
        label = row["nhan_that"]
        if label not in classes:
            raise ValueError(f"{row['file']}: nhãn {label!r} không thuộc 38 lớp")
        if row["nguon"] not in SOURCE_KEYS:
            raise ValueError(f"{row['file']}: nguồn không nhận ra {row['nguon']!r}")
        path = source_dir / row["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        prefix = row["sha256_12"].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{12}", prefix) or not digest.startswith(prefix):
            raise ValueError(f"{row['file']}: sha256_12 thiếu hoặc không khớp file")
        plan.append((row, path, digest))

    membership = validation_membership(
        [(row["sample_id"], digest, row["nhan_that"]) for row, _, digest in plan],
        list(shard.get("classes") or []),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = out_dir / "thumb"
    thumbs.mkdir(exist_ok=True)
    items = []
    for row, path, digest in plan:
        difficulty, difficulty_key, field = parse_difficulty(row["do_kho"])
        target = out_dir / path.name
        shutil.copyfile(path, target)
        if _sha256(target) != digest:
            raise ValueError(f"{path.name}: bản chép không khớp byte với nguồn")
        with Image.open(target) as image:
            width, height = image.size
            thumb = image.convert("RGB")
            thumb.thumbnail((THUMB_MAX, THUMB_MAX))
            thumb_name = f"{int(row['stt']):02d}.jpg"
            thumb.save(thumbs / thumb_name, "JPEG", quality=82, optimize=True)
        items.append({
            "id": int(row["stt"]),
            "file": path.name,
            "thumb": f"thumb/{thumb_name}",
            "label": row["nhan_that"],
            "source": SOURCE_KEYS[row["nguon"]],
            "source_name": row["nguon"],
            "difficulty": difficulty,
            "difficulty_key": difficulty_key,
            "field_photo": field,
            "class_val_count": int(row["so_anh_cung_lop_trong_val"]),
            "recall_fedavg_pct": float(row["fedavg_recall_%"]),
            "recall_centralized_pct": float(row["centralized_recall_%"]),
            "sample_id": row["sample_id"],
            "sha256": digest,
            "bytes": target.stat().st_size,
            "width": width,
            "height": height,
        })
    items.sort(key=lambda item: item["id"])
    keep = {item["file"] for item in items}
    keep_thumbs = {Path(item["thumb"]).name for item in items}
    for stale in out_dir.iterdir():
        if stale.is_file() and stale.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and stale.name not in keep:
            stale.unlink()
    for stale in thumbs.iterdir():
        if stale.is_file() and stale.name not in keep_thumbs:
            stale.unlink()
    return {
        "evidence": (
            "Ảnh lấy từ tập VALIDATION của release mixed_pv_pd_v3 (không dùng để huấn luyện), xếp từ dễ đến khó theo "
            "recall của lớp đó ở FedAvg v7. recall_*_pct là recall của LỚP trên validation cùng nguồn, không phải kết quả của riêng ảnh."
        ),
        "source": reg.DEMO_IMAGES_LABELS_CSV,
        "validation_membership": membership,
        "count": len(items),
        "items": items,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    source_dir = reg.abs_path(reg.DEMO_IMAGES_DIR)
    out_dir = reg.abs_path(reg.DEMO_GALLERY_DIR)
    catalog = reg.abs_path(reg.DEMO_GALLERY_CATALOG_JSON)
    if not (source_dir / "labels.csv").is_file():
        if catalog.is_file():
            print(f"[demo_gallery] thiếu {reg.DEMO_IMAGES_DIR} — giữ nguyên {reg.DEMO_GALLERY_CATALOG_JSON} đã sinh")
            return 0
        print(f"[demo_gallery] thiếu {reg.DEMO_IMAGES_DIR} và chưa có {reg.DEMO_GALLERY_CATALOG_JSON}", file=sys.stderr)
        return 1
    result = build(source_dir, out_dir)
    catalog.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[demo_gallery] {result['count']} ảnh | thuộc validation: {result['validation_membership']['status']} | "
          f"wrote {reg.DEMO_GALLERY_CATALOG_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

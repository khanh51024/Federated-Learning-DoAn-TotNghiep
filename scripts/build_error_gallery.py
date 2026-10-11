"""Thư viện ảnh "sai mà rất tự tin" cho bước 3 của web.

Nguồn: `detail.{fedavg,centralized}.high_confidence_wrong` trong artifacts/local_reeval_report.json — các ảnh VALIDATION mà
mô hình đoán sai với xác suất ≥ 0,9. Ảnh gốc nằm trong data/dataset_sources (gitignored); script tạo thumbnail 160 px vào
web/static/samples/errors/ (đi theo git, lên server) và catalog.json.

Kiểm trước khi ghi (không đạt thì dừng): báo cáo tái đánh giá khớp checkpoint/validation của bundle đã kiểm hash; mỗi ảnh có
trong val.jsonl và byte_sha256 của file khớp manifest; nhãn thật khớp class_id.

Chạy: python scripts/build_error_gallery.py
Không có dataset: giữ nguyên catalog đã sinh; chưa có thì thoát mã 1.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

THUMB = 160
MODELS = ("fedavg", "centralized")
ATTRIBUTION = ("Ảnh thuộc tập validation của release mixed_pv_pd_v3, lấy từ PlantVillage (Hughes & Salathé, 2015) và "
               "PlantDoc (Singh và cộng sự, 2020); chỉ dùng minh họa lỗi của mô hình trong đồ án.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thumb_name(sample_id: str) -> str:
    safe = "".join(ch for ch in sample_id if ch.isalnum() or ch in "_-")
    return f"{safe}.jpg"


def merge_wrong(detail: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Gộp danh sách của hai mô hình theo sample_id: mỗi ảnh ghi mô hình nào sai, đoán gì, tự tin bao nhiêu."""
    items: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for row in (detail.get(model) or {}).get("high_confidence_wrong") or []:
            item = items.setdefault(row["sample_id"], {
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "source": row["source"],
                "target": row["target"],
                "wrong_by": {},
            })
            if item["target"] != row["target"]:
                raise ValueError(f"{row['sample_id']}: nhãn thật khác nhau giữa hai mô hình")
            item["wrong_by"][model] = {"predicted": row["predicted"], "confidence": float(row["confidence"])}
    return items


def build(out_dir: Path) -> dict[str, Any]:
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        raise ValueError(f"bundle không hợp lệ: {evidence.get('reason')}")
    report = json.loads(reg.abs_path(reg.LOCAL_REEVAL_JSON).read_text(encoding="utf-8"))
    checkpoints = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}
    if (report.get("validation_manifest_sha256") != evidence["validation_manifest_sha256"]
            or report.get("checkpoint_sha256") != checkpoints):
        raise ValueError("báo cáo tái đánh giá không khớp bundle đã kiểm hash")

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    val_rows = {}
    with (release / "manifests" / "val.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                val_rows[row["sample_id"]] = row
    classes = evidence["classes"]
    items = merge_wrong(report.get("detail") or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_items = []
    for sample_id, item in sorted(items.items(), key=lambda kv: -max(v["confidence"] for v in kv[1]["wrong_by"].values())):
        row = val_rows.get(sample_id)
        if row is None:
            raise ValueError(f"{sample_id}: không có trong val.jsonl")
        if classes[row["class_id"]] != item["target"]:
            raise ValueError(f"{sample_id}: nhãn thật khác manifest")
        source_path = reg.abs_path(reg.DATASET_SOURCES_DIR) / row["relative_path"]
        if _sha256(source_path) != row["byte_sha256"]:
            raise ValueError(f"{sample_id}: byte_sha256 khác manifest")
        name = _thumb_name(sample_id)
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((THUMB, THUMB))
            image.save(out_dir / name, "JPEG", quality=80, optimize=True)
        catalog_items.append({
            "sample_id": sample_id,
            "thumb": name,
            "source": item["source"],
            "target": item["target"],
            "wrong_by": item["wrong_by"],
        })
    keep = {i["thumb"] for i in catalog_items}
    for stale in out_dir.glob("*.jpg"):
        if stale.name not in keep:
            stale.unlink()
    return {
        "evidence": "ảnh validation mà FedAvg v7 hoặc Centralized v6 đoán SAI với xác suất >= 0,9 (từ báo cáo tái đánh giá cục bộ)",
        "attribution": ATTRIBUTION,
        "threshold": report.get("high_confidence_threshold"),
        "protocol_id": evidence["protocol_id"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "checkpoint_sha256": {m: checkpoints[m] for m in MODELS},
        "count": len(catalog_items),
        "items": catalog_items,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = reg.abs_path(reg.ERROR_GALLERY_DIR)
    catalog = reg.abs_path(reg.ERROR_GALLERY_CATALOG_JSON)
    if not (reg.abs_path(reg.RELEASE_V3_DIR) / "manifests" / "val.jsonl").is_file():
        if catalog.is_file():
            print(f"[error_gallery] thiếu dataset — giữ nguyên {reg.ERROR_GALLERY_CATALOG_JSON}")
            return 0
        print(f"[error_gallery] thiếu dataset ở {reg.DATASET_SOURCES_DIR}", file=sys.stderr)
        return 1
    result = build(out_dir)
    catalog.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out_dir.glob("*.jpg"))
    print(f"[error_gallery] {result['count']} ảnh, {size / 1024:.0f} KB | wrote {reg.ERROR_GALLERY_CATALOG_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Nạp ảnh ngoài đồng ứng viên (lấy từ nơi khác, đã sàng lọc từng ảnh) vào thư mục nguồn của release v4.

Vì sao có: hai lớp (Cherry powdery mildew, Strawberry leaf scorch) không có bộ công khai nào; ảnh iNaturalist theo loài nấm `Podosphaera cerasi` và
`Diplocarpon fragariae` mà quy trình tải tự động không lấy (tên khoa học khác với cấu hình, và bộ lọc lá loại gần hết). Mỗi ảnh ứng viên được xem bằng mắt
qua ảnh lưới; những ảnh được nhận ghi bằng `photo_id` trong configs/field_candidate_review.json (người kiểm: `ai_screening`, KHÔNG phải chuyên gia).

Kết quả:
  * data/external/sources/<nguồn>/{images,manifest.jsonl,import_report.json}: cùng định dạng với nguồn iNaturalist đã có, nhãn `community`, `unreviewed`;
  * data/label_review_cand/status.json: `accepted_samples` (scripts/build_release_v4.py và scripts/report_data_gaps.py đọc tệp này như đọc status của kiểm nhãn).

Ảnh vẫn đi qua mọi cổng của scripts/build_release_v4.py (trùng với tập khóa, gom nhóm theo quan sát và đặc trưng, cổng rò rỉ).

    python scripts/import_field_candidates.py --candidates-root <thư mục chứa inat_cerasi/, inat_scorch/ cùng manifest.jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SPEC = "configs/field_candidate_review.json"
EXT_SOURCES = "data/external/sources"
CANDIDATE_STATUS = "data/label_review_cand/status.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_accepted(rows: list[dict[str, Any]], photo_ids: list[int], target_class: str) -> list[dict[str, Any]]:
    """Giữ đúng các ảnh đã nhận (theo photo_id). Thiếu ảnh hoặc sai lớp là lỗi: danh sách nhận phải khớp với bộ ứng viên đã xem."""
    by_photo = {int(r["photo_id"]): r for r in rows}
    missing = sorted(set(photo_ids) - set(by_photo))
    if missing:
        raise SystemExit(f"thiếu {len(missing)} ảnh đã nhận trong bộ ứng viên: {missing[:5]}")
    picked = [by_photo[p] for p in sorted(set(photo_ids))]
    wrong = [r["sample_id"] for r in picked if r.get("target_class") != target_class]
    if wrong:
        raise SystemExit(f"ảnh không thuộc lớp {target_class}: {wrong[:3]}")
    return picked


def manifest_row(source: str, class_id: int, candidate: dict[str, Any], digest: str, width: int, height: int, direct: int, flipped: int) -> dict[str, Any]:
    return {
        "sample_id": f"ext_{source}_{digest[:16]}",
        "relative_path": f"{source}/images/{candidate['target_class']}/{digest[:16]}.jpg",
        "byte_sha256": digest,
        "dhash64": f"{direct:016x}",
        "dhash64_flip": f"{flipped:016x}",
        "group_id": f"{source}:obs{candidate['observation_id']}",
        "class_id": class_id,
        "target_class": candidate["target_class"],
        "source_dataset": source,
        "source_domain": "plantdoc",
        "mapping_status": "exact_match",
        "quarantine_status": "clean",
        "label_trust": "community",
        "label_review_status": "unreviewed",
        "capture_note": "ngoài đồng/vườn",
        "original_file": f"{candidate['target_class']}/{candidate['observation_id']}_{candidate['photo_id']}.jpg",
        "width": width,
        "height": height,
        "license": candidate["license"],
        "author": candidate.get("author"),
        "url": candidate["observation_url"],
        "observation_id": candidate["observation_id"],
        "taxon": candidate["taxon"],
        "leaf_score": None,
    }


def check_existing(sample_ids: list[str]) -> list[str]:
    """Ảnh đã nằm sẵn trong thư mục nguồn khác: chỉ cần chắc là còn tồn tại trong một manifest."""
    known = {r["sample_id"] for m in reg.abs_path(EXT_SOURCES).glob("*/manifest.jsonl") for r in read_jsonl(m)}
    missing = [i for i in sample_ids if i not in known]
    if missing:
        raise SystemExit(f"ảnh đã nhận không có trong manifest nào: {missing[:3]}")
    return list(sample_ids)


def import_source(source: str, entry: dict[str, Any], candidates_root: Path, classes: list[str]) -> list[dict[str, Any]]:
    import io

    from PIL import Image, ImageOps

    from scripts.build_external_eval_set import dhash64
    from scripts.import_external_source import normalize_image_bytes

    rows = select_accepted(read_jsonl(candidates_root / source / "manifest.jsonl"), entry["accepted_photo_ids"], entry["target_class"])
    class_id = classes.index(entry["target_class"])
    base = reg.abs_path(EXT_SOURCES) / source
    out_rows, seen = [], set()
    for candidate in rows:
        # cùng cách chuẩn hóa như scripts/import_external_source.py: JPEG RGB đã xoay thẳng, cạnh dài giới hạn; sha tính trên byte đã chuẩn hóa
        with Image.open(candidates_root / candidate["relative_path"]) as handle:
            handle.load()
            data = normalize_image_bytes(handle)
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:      # hai ảnh trùng byte sau chuẩn hóa sẽ trùng sample_id
            continue
        seen.add(digest)
        with Image.open(io.BytesIO(data)) as small:
            width, height = small.size
            direct, flipped = dhash64(small), dhash64(ImageOps.mirror(small.convert("RGB")))
        row = manifest_row(source, class_id, candidate, digest, width, height, direct, flipped)
        dst = reg.abs_path(EXT_SOURCES) / row["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        out_rows.append(row)
    out_rows.sort(key=lambda r: r["sample_id"])
    (base / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    report = {"source": source, "title": f"iNaturalist {entry['taxon']} (ứng viên đã sàng lọc từng ảnh)", "license": "theo từng ảnh",
              "images_imported": len(out_rows), "by_class": dict(Counter(r["target_class"] for r in out_rows)),
              "candidates_total": entry["candidates_total"], "observations": len({r["observation_id"] for r in out_rows}),
              "label_review_status": {"unreviewed": len(out_rows)}, "reviewer": "ai_screening"}
    (base / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-root", required=True, type=Path)
    parser.add_argument("--spec", default=SPEC)
    args = parser.parse_args()
    spec = json.loads(reg.abs_path(args.spec).read_text(encoding="utf-8"))
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    accepted: list[str] = []
    for source, entry in spec["sources"].items():
        rows = import_source(source, entry, args.candidates_root, classes)
        accepted += [r["sample_id"] for r in rows]
        print(f"[candidates] {source}: {len(rows)} ảnh ({len({r['observation_id'] for r in rows})} quan sát) -> {entry['target_class']}")
    existing = [i for key, ids in spec.get("existing_sample_ids", {}).items() if key != "note" for i in ids]
    accepted += check_existing(existing)
    status = {"reviewer": spec["reviewer"], "note": spec["note"], "spec": args.spec, "accepted_samples": sorted(accepted)}
    path = reg.abs_path(CANDIDATE_STATUS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[candidates] {len(accepted)} ảnh được nhận -> {CANDIDATE_STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

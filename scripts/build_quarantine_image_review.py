"""Lập bộ kiểm TỪNG LÁ cho các lớp lá PlantDoc đang cách ly mà kiểm theo nhóm không đạt (nhiều lá nhỏ, mờ nên "không rõ").

Khác scripts/build_label_review.py (kiểm theo nhóm: nhận hoặc loại cả lớp), ở đây mỗi lá được quyết định riêng; chỉ lá được đánh dấu "đúng" mới rời
cách ly. Lá vẫn giữ nguyên split dự kiến và nhóm cảnh của v3, nên không có rò rỉ giữa các lá cùng một ảnh gốc.

Chọn lá: thuộc `--classes`, trạng thái cách ly `clean`, cạnh ngắn >= `--min-side` px (lá nhỏ hơn thường không nhận ra được loài), lấy mẫu cố định
tối đa `--per-class` lá mỗi lớp. Ghi data/label_review_q2/samples.json (định dạng như bộ kiểm ngoài đồng, mọi nhóm là nhóm kiểm từng ảnh) và ảnh lưới
6 x 5 ô ở data/label_review_q2/sheets/. Sau khi xem: scripts/record_review_decisions.py --review-dir data/label_review_q2 ... rồi
scripts/apply_label_review.py --samples data/label_review_q2/samples.json ...; scripts/build_release_v4.py đọc `accepted_samples` của status.json.

Chạy:
    python scripts/build_quarantine_image_review.py --classes Blueberry___healthy --classes Raspberry___healthy
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_review_sheets import COLS, make_sheet, safe_name  # noqa: E402
from src import registry as reg  # noqa: E402

OUT_DIR = "data/label_review_q2"
SEED = 20261022


def extend_sample(previous: list[str], pool: list[str], per_class: int, rng: random.Random) -> list[str]:
    """Danh sách lá cần kiểm của một lớp sau lần chạy này: lá đã kiểm (`previous`) giữ nguyên số thứ tự, lá mới lấy ngẫu nhiên từ `pool` nối vào cuối
    cho tới tổng `per_class`. Nhờ vậy quyết định đã ghi theo số thứ tự vẫn đúng chỗ khi kiểm thêm."""
    seen = set(previous)
    remaining = sorted(i for i in pool if i not in seen)
    new = sorted(rng.sample(remaining, min(max(0, per_class - len(previous)), len(remaining))))
    return list(previous) + new


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", action="append", required=True)
    parser.add_argument("--per-class", type=int, default=150, help="tổng số lá mỗi lớp sau lần chạy này (chạy lại với số lớn hơn để kiểm thêm)")
    parser.add_argument("--min-side", type=int, default=150)
    args = parser.parse_args()

    root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    rows: dict[str, dict] = {}
    with (reg.abs_path(reg.RELEASE_V3_DIR) / "manifests" / "quarantine.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["sample_id"], r)
    pools: dict[str, list[dict]] = {c: [] for c in args.classes}
    for r in rows.values():
        if (r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean" and r.get("target_class") in pools
                and r.get("split") in ("train", "val", "calibration", "legacy_diagnostic_test")):
            with Image.open(root / r["relative_path"]) as image:
                if min(image.size) >= args.min_side:
                    pools[r["target_class"]].append(r)
    rng = random.Random(SEED)
    out = reg.abs_path(OUT_DIR)
    previous = json.loads((out / "samples.json").read_text(encoding="utf-8")) if (out / "samples.json").is_file() else {"classes": {}, "totals": {}}
    classes, totals = dict(previous["classes"]), dict(previous["totals"])
    for name, pool in pools.items():
        group = f"plantdoc_quarantine|{name}"
        classes[group] = extend_sample(classes.get(group, []), [r["sample_id"] for r in pool], args.per_class, rng)
        totals[group] = len(pool)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.json").write_text(json.dumps({
        "seed": SEED, "min_side": args.min_side, "per_class": max(len(v) for v in classes.values()), "accept_correct": 0.9, "accept_unsure_max": 0.2,
        "classes": classes, "totals": totals, "per_image_groups": sorted(classes),
        "note": "lá PlantDoc cách ly kiểm từng lá; chỉ lá 'đúng' được nhận, giữ split và nhóm cảnh của v3",
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    page = COLS * 5
    for group, ids in classes.items():
        for number, start in enumerate(range(0, len(ids), page), start=1):
            chunk = ids[start:start + page]
            suffix = "" if len(ids) <= page else f"__p{number}"
            make_sheet(chunk, rows, f"{group}  (lá {start + 1}-{start + len(chunk)} / {len(ids)})", out / "sheets" / f"{safe_name(group)}{suffix}.jpg",
                       number_from=start + 1, base=root)
    print(f"[q2] {sum(len(v) for v in classes.values())} lá cần kiểm ({ {g: len(v) for g, v in classes.items()} }); kho >= {args.min_side}px: {totals} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

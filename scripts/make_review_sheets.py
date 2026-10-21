"""Tạo ảnh lưới (contact sheet) cho từng nhóm kiểm nhãn trong data/label_review_ext/samples.json để xem nhanh bằng mắt.

Mỗi nhóm một ảnh `sheets/<nhóm>.jpg`: lưới 6 x 5 = 30 ô, mỗi ô có số thứ tự (1..30) theo đúng thứ tự trong samples.json. Người kiểm (hoặc AI sàng lọc)
ghi các số "sai" và "không rõ"; script `scripts/record_review_decisions.py` đổi danh sách đó thành decisions.json (số còn lại là "đúng").

Chạy:
    python scripts/make_review_sheets.py [--group "nguồn|lớp"]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

REVIEW_DIR = "data/label_review_ext"
EXT_SOURCES = "data/external/sources"
TILE = 200
COLS = 6


def safe_name(group: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", group)


def load_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for manifest in sorted(reg.abs_path(EXT_SOURCES).glob("*/manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def make_sheet(ids: list[str], rows: dict[str, dict[str, Any]], title: str, out: Path, number_from: int = 1) -> None:
    from PIL import Image, ImageDraw

    base = reg.abs_path(EXT_SOURCES)
    rows_n = (len(ids) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE, rows_n * TILE + 24), (245, 245, 240))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title[:150], fill=(20, 20, 20))
    for i, sid in enumerate(ids):
        row = rows[sid]
        with Image.open(base / row["relative_path"]) as handle:
            thumb = handle.convert("RGB")
            thumb.thumbnail((TILE - 4, TILE - 4))
        x, y = (i % COLS) * TILE + 2, (i // COLS) * TILE + 24 + 2
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + 26, y + 16], fill=(255, 255, 255))
        draw.text((x + 3, y + 2), str(number_from + i), fill=(200, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=82)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", default=[])
    args = parser.parse_args()
    samples = json.loads((reg.abs_path(REVIEW_DIR) / "samples.json").read_text(encoding="utf-8"))
    rows = load_rows()
    out_dir = reg.abs_path(REVIEW_DIR) / "sheets"
    page_size = COLS * 5
    for group, ids in samples["classes"].items():
        if args.group and group not in args.group:
            continue
        for page, start in enumerate(range(0, len(ids), page_size), start=1):
            chunk = ids[start:start + page_size]
            suffix = "" if len(ids) <= page_size else f"__p{page}"
            make_sheet(chunk, rows, f"{group}  (ảnh {start + 1}-{start + len(chunk)} / {len(ids)}; số trên ô = thứ tự trong nhóm)",
                       out_dir / f"{safe_name(group)}{suffix}.jpg", number_from=start + 1)
    print(f"[sheets] {len(samples['classes'])} nhóm -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

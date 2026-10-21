"""Đổi kết quả xem ảnh lưới thành `decisions_<người kiểm>.json` để scripts/apply_label_review.py áp quy tắc nhận nhóm.

Đầu vào: `--spec` là tệp JSON dạng {"<nhóm>": {"sai": [số...], "khong_ro": [số...], "ghi_chu": "..."}}; số là thứ tự ô (1..30) trên ảnh lưới do
scripts/make_review_sheets.py tạo. Mọi ô không nằm trong "sai"/"khong_ro" được tính "đúng". Nhóm không có trong tệp thì KHÔNG có quyết định (chưa kiểm).

`--reviewer`: ghi rõ ai kiểm ("ai_screening" = sàng lọc bằng mắt do AI, KHÔNG phải chuyên gia; "human" = người). Tệp của người nên đặt cuối trong
scripts/apply_label_review.py để ghi đè.

Chạy:
    python scripts/record_review_decisions.py --spec data/label_review_ext/ai_spec.json --reviewer ai_screening --out data/label_review_ext/decisions_ai.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

REVIEW_DIR = "data/label_review_ext"


def build_decisions(samples: dict[str, Any], spec: dict[str, Any]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for group, ids in samples["classes"].items():
        if group not in spec:
            continue
        wrong = set(spec[group].get("sai", []))
        unsure = set(spec[group].get("khong_ro", []))
        for index, sample_id in enumerate(ids, start=1):
            decisions[sample_id] = "sai" if index in wrong else "khong_ro" if index in unsure else "dung"
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--reviewer", default="ai_screening")
    parser.add_argument("--out", required=True)
    parser.add_argument("--review-dir", default=REVIEW_DIR, help="thư mục chứa samples.json (mặc định bộ kiểm ngoài đồng)")
    args = parser.parse_args()
    samples = json.loads((reg.abs_path(args.review_dir) / "samples.json").read_text(encoding="utf-8"))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    decisions = build_decisions(samples, spec)
    payload = {"reviewer": args.reviewer, "note": "sàng lọc bằng mắt qua ảnh lưới; không phải kiểm của chuyên gia" if args.reviewer == "ai_screening" else "",
               "groups_reviewed": sorted(g for g in samples["classes"] if g in spec), "decisions": decisions}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[review] {len(decisions)} quyết định, {len(payload['groups_reviewed'])} nhóm -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

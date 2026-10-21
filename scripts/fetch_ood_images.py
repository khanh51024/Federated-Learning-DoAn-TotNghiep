"""Tải ảnh KHÔNG phải lá cây bệnh từ Wikimedia Commons (giấy phép mở, ghi tác giả và URL từng ảnh) để đo cổng từ chối "ảnh bất kỳ".

Người dùng web có thể gửi ảnh xe, người, thức ăn, phong cảnh, hoa quả hoặc lá của loài khác; mô hình 38 lớp vẫn trả một tên bệnh cho các ảnh đó. Bộ ảnh này chỉ
dùng ĐỂ ĐO (không bao giờ đưa vào học) việc từ chối đúng. Hai nhóm:
  general    ảnh không phải lá cây (xe, người, động vật, nhà cửa, trong nhà, thức ăn, quả, hoa, phong cảnh, giấy tờ, đất đá)
  other_leaf lá cây của loài KHÔNG nằm trong 38 lớp (sồi, phong, hoa hồng, chuối, dương xỉ...): nhóm khó, cổng "có phải lá không" không loại được

Dùng lại bộ tìm kiếm và tải của scripts/fetch_wikimedia.py (cùng bộ lọc giấy phép). Ghi vào data/external/ood/ (ngoài git):
    images/<nhóm>/<chủ đề>/<mã trang>.jpg   metadata.jsonl (giấy phép, tác giả, URL trang tệp, truy vấn)

Chạy:
    python scripts/fetch_ood_images.py [--per-query 25]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_wikimedia import download, search_files, slug  # noqa: E402
from src import registry as reg  # noqa: E402

OOD_DIR = "data/external/ood"
QUERIES: dict[str, dict[str, list[str]]] = {
    "general": {
        "vehicle": ["car on street", "bus", "passenger aircraft", "bicycle", "truck"],
        "person": ["portrait of a man", "group of people", "woman smiling", "children playing"],
        "animal": ["dog", "cat", "horse", "bird on branch", "cow"],
        "building": ["house exterior", "church building", "city skyline", "bridge"],
        "indoor": ["living room interior", "kitchen interior", "office desk", "bedroom"],
        "food": ["pizza", "bread", "meal on plate", "cake", "noodles"],
        "fruit_vegetable": ["apple fruit", "tomato fruit", "banana fruit", "vegetables at market", "orange fruit"],
        "flower": ["rose flower", "sunflower", "tulip field", "orchid"],
        "landscape": ["mountain landscape", "beach sea", "river valley", "snow field", "desert"],
        "document": ["newspaper page", "road sign", "book cover", "poster printed"],
        "ground": ["soil texture", "rock surface", "sand", "wooden wall"],
    },
    "other_leaf": {
        "tree_leaf": ["oak leaf", "maple leaf", "birch leaf", "ginkgo leaf", "elm leaf"],
        "garden_leaf": ["rose leaf", "banana leaf", "fern leaf", "mint leaf", "basil leaf"],
    },
}


def fetch_topic(group: str, topic: str, queries: list[str], per_query: int) -> dict[str, int]:
    base = reg.abs_path(OOD_DIR)
    meta_path = base / "metadata.jsonl"
    have = {json.loads(l)["page_id"] for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()} if meta_path.is_file() else set()
    seen: dict[int, dict] = {}
    for query in queries:
        for item in search_files(query, per_query)[:per_query]:
            seen.setdefault(item["page_id"], {**item, "query": query})
    todo = [v for k, v in seen.items() if k not in have]
    jobs = [(v, base / "images" / group / topic / f"{v['page_id']}.jpg") for v in todo]
    with ThreadPoolExecutor(max_workers=6) as pool:
        ok = list(pool.map(lambda j: download(j[0]["thumb_url"], j[1]), jobs))
    added = 0
    for (v, dest), good in zip(jobs, ok):
        if not good:
            continue
        row = {"page_id": v["page_id"], "group": group, "topic": topic, "file": str(dest.relative_to(base)).replace("\\", "/"), "license": v["license"],
               "attribution": v["artist"], "page_url": v["page_url"], "title": v["title"], "query": v["query"], "source": "wikimedia_commons"}
        with meta_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        added += 1
    return {"candidates": len(seen), "downloaded": added}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-query", type=int, default=25)
    args = parser.parse_args()
    (reg.abs_path(OOD_DIR)).mkdir(parents=True, exist_ok=True)
    for group, topics in QUERIES.items():
        for topic, queries in topics.items():
            result = fetch_topic(group, topic, queries, args.per_query)
            print(f"[ood] {group}/{slug(topic)}: {result}", flush=True)
    meta = reg.abs_path(OOD_DIR) / "metadata.jsonl"
    rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()] if meta.is_file() else []
    print(f"[ood] tổng {len(rows)} ảnh: " + ", ".join(f"{g}={sum(1 for r in rows if r['group'] == g)}" for g in QUERIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

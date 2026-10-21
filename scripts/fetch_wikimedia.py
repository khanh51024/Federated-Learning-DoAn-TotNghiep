"""Tải ảnh Wikimedia Commons (giấy phép mở, ghi tác giả và URL từng ảnh) cho các lớp bệnh còn thiếu ảnh ngoài đồng.

Dùng API công khai của Commons (không cần tài khoản): tìm tệp ảnh (không gian tên File) theo từ khóa trong phần `wikimedia:` của configs/external_sources.yaml, lọc theo
giấy phép khai báo (CC0, CC BY, CC BY-SA, phạm vi công cộng), tải bản thu nhỏ 640 px và ghi `inat_metadata.jsonl` (cùng định dạng với scripts/fetch_inaturalist.py để dùng
chung bộ nhập): giấy phép, tác giả, URL trang tệp của TỪNG ảnh.

Kết quả tìm kiếm của Commons chứa nhiều ảnh không phải ảnh lá (sơ đồ, ảnh kính hiển vi, ảnh quả...). Vì vậy nguồn này luôn qua kiểm nhãn TỪNG ẢNH (xem docs/DATASET_V4.md);
bộ lọc lá (scripts/filter_leaf_photos.py) có thể chạy trước để giảm số ảnh phải xem.

Chạy:
    python scripts/fetch_wikimedia.py --count-only
    python scripts/fetch_wikimedia.py --all [--max-per-class 120]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CropFed-FL-research/1.0 (university thesis; image collection via public API)"
SOURCES_DIR = "data/external/sources"
ALLOWED_LICENSE = re.compile(r"^(cc0|cc[ -]?by|public domain|pd[ -]|attribution)", re.I)
BLOCKED_TITLE = re.compile(r"\b(diagram|map|logo|icon|drawing|illustration|svg|life ?cycle|microscop|spore|conidi|culture|petri|pcr|gel|graph|chart|poster|scheme)\b", re.I)
IMAGE_MIME = {"image/jpeg"}


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def license_ok(short_name: str) -> bool:
    return bool(short_name) and bool(ALLOWED_LICENSE.match(short_name.strip()))


def strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def api_get(params: dict[str, Any]) -> dict[str, Any]:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            time.sleep(0.6)
            return data
        except Exception:  # noqa: BLE001
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Commons API lỗi: {url}")


def search_files(query: str, limit: int) -> list[dict[str, Any]]:
    """Tệp ảnh khớp từ khóa, kèm giấy phép và URL bản 640 px. Phân trang tới khi đủ `limit` hoặc hết kết quả."""
    out: list[dict[str, Any]] = []
    offset = 0
    while len(out) < limit and offset <= 300:  # kết quả tìm kiếm xếp theo độ liên quan: các trang sau gần như toàn ảnh không liên quan
        try:
            data = api_get({
            "action": "query", "generator": "search", "gsrnamespace": 6, "gsrsearch": query, "gsrlimit": 50, "gsroffset": offset,
            "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 640,
            })
        except RuntimeError:
            break
        pages = (data.get("query") or {}).get("pages") or {}
        if not pages:
            break
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            info = (page.get("imageinfo") or [None])[0]
            if not info or info.get("mime") not in IMAGE_MIME or BLOCKED_TITLE.search(page.get("title", "")):
                continue
            meta = info.get("extmetadata") or {}
            lic = strip_tags((meta.get("LicenseShortName") or {}).get("value", ""))
            if not license_ok(lic) or min(info.get("width", 0), info.get("height", 0)) < 300:
                continue
            out.append({
                "page_id": int(page["pageid"]), "title": page["title"], "license": lic, "thumb_url": info.get("thumburl") or info["url"],
                "page_url": info.get("descriptionurl"), "artist": strip_tags((meta.get("Artist") or {}).get("value", ""))[:160],
                "description": strip_tags((meta.get("ImageDescription") or {}).get("value", ""))[:200],
            })
        nxt = (data.get("continue") or {}).get("gsroffset")
        if nxt is None:
            break
        offset = int(nxt)
    return out


def download(url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if len(data) < 5000:
                return False
            dest.write_bytes(data)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    return False


def fetch_class(entry: dict[str, Any], max_per_class: int) -> dict[str, Any]:
    target = entry["target_class"]
    base = reg.abs_path(SOURCES_DIR) / f"inat_wm_{slug(target)}"
    raw = base / "raw" / target
    meta_path = base / "inat_metadata.jsonl"
    have = {json.loads(l)["photo_id"] for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()} if meta_path.is_file() else set()
    seen: dict[int, dict[str, Any]] = {}
    for query in entry["queries"]:
        for item in search_files(query, max_per_class * 2):
            seen.setdefault(item["page_id"], {**item, "query": query})
    todo = [v for k, v in seen.items() if k not in have][: max(0, max_per_class - len(have))]
    jobs = [(v, raw / f"{v['page_id']}.jpg") for v in todo]
    with ThreadPoolExecutor(max_workers=6) as pool:
        ok = list(pool.map(lambda j: download(j[0]["thumb_url"], j[1]), jobs))
    added = 0
    for (v, dest), good in zip(jobs, ok):
        if not good:
            continue
        row = {"photo_id": v["page_id"], "observation_id": v["page_id"], "file": f"{target}/{dest.name}", "target_class": target, "taxon_query": v["query"],
               "taxon_name": v["title"], "license": v["license"], "attribution": v["artist"], "observation_url": v["page_url"], "observed_on": None,
               "place": v["description"], "user": None, "source": "wikimedia_commons"}
        with meta_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        added += 1
    return {"target_class": target, "candidates": len(seen), "downloaded": added, "total_in_metadata": len(have) + added}


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--class", dest="classes", action="append", default=[])
    parser.add_argument("--max-per-class", type=int, default=120)
    args = parser.parse_args()
    entries = yaml.safe_load(reg.abs_path("configs/external_sources.yaml").read_text(encoding="utf-8"))["wikimedia"]["classes"]
    entries = [e for e in entries if args.all or args.count_only or e["target_class"] in args.classes]
    for entry in entries:
        if args.count_only:
            n = {q: len(search_files(q, 100)) for q in entry["queries"]}
            print(f"{entry['target_class']:46s} {n}", flush=True)
            continue
        print(json.dumps(fetch_class(entry, args.max_per_class), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

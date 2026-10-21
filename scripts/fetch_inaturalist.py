"""Tải ảnh iNaturalist (cấp "research", giấy phép mở) theo tên tác nhân gây bệnh cho các lớp còn thiếu ảnh ngoài đồng.

Chỉ dùng API công khai của iNaturalist (không cần tài khoản), tôn trọng giới hạn tốc độ (không quá ~50 yêu cầu/phút). Ảnh lấy từ máy chủ ảnh mở
(kích thước "medium", cạnh dài 500 px). Mỗi ảnh ghi vào `inat_metadata.jsonl`: mã quan sát, mã ảnh, tác giả, giấy phép của TỪNG ảnh, URL quan sát,
tên loài, ngày, nơi; không có ảnh nào ngoài các giấy phép khai báo (`licenses`). Ảnh giữ lại ở data/external/ (gitignored), chỉ dùng cho đồ án.

Khác với bộ dữ liệu khoa học, nhãn iNaturalist là nhãn cộng đồng và ảnh gồm cả ảnh không phải lá (quả, cả cây, nấm trên thân). Vì vậy ảnh tải về mang
`label_trust: community` và PHẢI qua kiểm nhãn mẫu trước khi vào release (scripts/build_label_review.py).

Cấu hình: phần `inaturalist:` của configs/external_sources.yaml.

Chạy:
    python scripts/fetch_inaturalist.py --count-only            # chỉ đếm số quan sát từng taxon, không tải ảnh
    python scripts/fetch_inaturalist.py --class Tomato___Leaf_Mold --max-per-class 400
    python scripts/fetch_inaturalist.py --all
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
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

API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "CropFed-FL-research/1.0 (university thesis)"
SOURCES_DIR = "data/external/sources"
API_DELAY_S = 1.3  # ~46 yêu cầu/phút
IMAGE_DELAY_S = 0.15


def slug(target_class: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in target_class).strip("_")


def api_get(params: dict[str, Any]) -> dict[str, Any]:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            time.sleep(API_DELAY_S)
            return data
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"iNaturalist API lỗi: {url}")


def base_params(taxon_name: str, licenses: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    params = {
        "taxon_name": taxon_name, "quality_grade": "research", "photos": "true", "photo_license": ",".join(licenses),
        "per_page": 200, "order_by": "id", "order": "asc",
    }
    params.update(extra or {})
    return params


def count_observations(taxon_name: str, licenses: list[str], extra: dict[str, Any] | None = None) -> int:
    params = base_params(taxon_name, licenses, extra)
    params["per_page"] = 1
    return int(api_get(params).get("total_results", 0))


def photo_url(square_url: str, size: str = "medium") -> str:
    """URL ảnh theo kích thước từ URL ảnh vuông của API (…/square.jpg -> …/medium.jpg)."""
    for known in ("square", "small", "medium", "large", "original"):
        if f"/{known}." in square_url:
            return square_url.replace(f"/{known}.", f"/{size}.")
    return square_url


def download(url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if len(data) < 2000:
                return False
            dest.write_bytes(data)
            time.sleep(IMAGE_DELAY_S)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return False


def fetch_class(entry: dict[str, Any], licenses: list[str], max_per_class: int) -> dict[str, Any]:
    target = entry["target_class"]
    base = reg.abs_path(SOURCES_DIR) / f"inat_{slug(target)}"
    raw = base / "raw" / target
    meta_path = base / "inat_metadata.jsonl"
    have: dict[int, dict[str, Any]] = {}
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            have[row["photo_id"]] = row
    per_taxon = max(1, int(entry.get("max_raw", max_per_class)) // max(1, len(entry["taxa"])))
    stats = {"target_class": target, "downloaded": 0, "skipped_existing": 0, "taxa": {}}
    extra = dict(entry.get("api_extra") or {})
    for taxon in entry["taxa"]:
        got = 0
        page = 1
        total_available = None
        while got < per_taxon:
            data = api_get({**base_params(taxon, licenses, extra), "page": page})
            if total_available is None:
                total_available = int(data.get("total_results", 0))
            results = data.get("results", [])
            if not results:
                break
            pending: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
            for obs in results:
                if got + len(pending) >= per_taxon:
                    break
                photo = (obs.get("photos") or [None])[0]
                if not photo or not photo.get("license_code"):
                    continue
                pid = int(photo["id"])
                if pid in have:
                    stats["skipped_existing"] += 1
                    got += 1
                    continue
                ext = ".jpg" if ".jpg" in photo["url"].lower() or ".jpeg" in photo["url"].lower() else ".png"
                pending.append((obs, photo, raw / f"{obs['id']}_{pid}{ext}"))
            # Máy chủ ảnh mở trả ~100 KB/s mỗi kết nối: tải song song (8 luồng) trong một trang kết quả
            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(lambda item: download(photo_url(item[1]["url"]), item[2]), pending))
            for (obs, photo, dest), ok in zip(pending, outcomes):
                if not ok:
                    continue
                pid = int(photo["id"])
                row = {
                    "photo_id": pid, "observation_id": obs["id"], "file": f"{target}/{dest.name}", "target_class": target,
                    "taxon_query": taxon, "taxon_name": (obs.get("taxon") or {}).get("name"), "license": photo["license_code"],
                    "attribution": photo.get("attribution"), "observation_url": obs.get("uri"), "observed_on": obs.get("observed_on"),
                    "place": obs.get("place_guess"), "user": (obs.get("user") or {}).get("login"),
                }
                have[pid] = row
                with meta_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["downloaded"] += 1
                got += 1
            if len(results) < 200 or page >= 40:
                break
            page += 1
        stats["taxa"][taxon] = {"observations_available": total_available, "kept": got}
    stats["total_in_metadata"] = len(have)
    return stats


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--class", dest="classes", action="append", default=[])
    parser.add_argument("--max-per-class", type=int, default=400)
    args = parser.parse_args()

    config = yaml.safe_load(reg.abs_path("configs/external_sources.yaml").read_text(encoding="utf-8"))["inaturalist"]
    licenses = config["licenses"]
    entries = [e for e in config["classes"] if args.all or e["target_class"] in args.classes or args.count_only]
    if not entries:
        print("không có lớp nào được chọn (dùng --all hoặc --class)", file=sys.stderr)
        return 1
    summary = []
    for entry in entries:
        if args.count_only:
            counts = {t: count_observations(t, licenses, entry.get("api_extra")) for t in entry["taxa"]}
            print(f"{entry['target_class']:46s} {counts}", flush=True)
            summary.append({"target_class": entry["target_class"], "counts": counts})
            continue
        stats = fetch_class(entry, licenses, args.max_per_class)
        print(json.dumps(stats, ensure_ascii=False), flush=True)
        summary.append(stats)
    out = reg.abs_path(SOURCES_DIR) / "_inat_last_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "summary": summary}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

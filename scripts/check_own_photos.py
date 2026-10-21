"""Kiểm và chuẩn hóa ảnh lá do chính bạn chụp (nguồn quý nhất: đúng điều kiện dùng thật), rồi chia phần học / phần thử theo CÂY.

Thư mục đầu vào (không commit): data/own_photos/
    raw/                ảnh gốc từ điện thoại (jpg/png/heic đã đổi sang jpg), đặt tự do
    labels.csv          mỗi ảnh một dòng, cột:
        file        đường dẫn tương đối trong raw/ (vd tao_01/IMG_2041.jpg)
        class       một trong 38 lớp (tên đầy đủ, vd Tomato___Late_blight) hoặc tên ngắn không phân biệt hoa thường (vd "tomato late blight")
        plant_id    mã cây (cùng một cây = cùng mã; nhiều ảnh của một cây PHẢI cùng mã để không rò rỉ giữa học và thử)
        location    nơi chụp (vườn/ruộng, vd "vuon_A")
        date        YYYY-MM-DD
        device      tên điện thoại
        verified    "expert" nếu có người có chuyên môn xác nhận nhãn, "self" nếu tự nhận biết, "unsure" nếu chưa chắc (ảnh unsure KHÔNG dùng)

Quy tắc chia (đặt trước, không đổi sau khi thấy kết quả): mỗi cây thuộc đúng một phía; ~30% số cây của MỖI lớp vào phần thử (`split: test`), còn lại vào phần học.
Lớp chỉ có 1 cây: toàn bộ vào phần học và bị ghi cảnh báo (không đo được). Phần thử khóa bằng SHA-256 ở data/own_photos/test_lock.json trước khi bất kỳ ảnh học nào
được đưa vào release.

Chạy:
    python scripts/check_own_photos.py --template          # tạo labels.csv mẫu và thư mục
    python scripts/check_own_photos.py                     # kiểm, chuẩn hóa, ghi manifest.jsonl và báo cáo
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

OWN_DIR = "data/own_photos"
TEST_FRACTION = 0.30
MAX_SIDE = 640
HEADER = ["file", "class", "plant_id", "location", "date", "device", "verified"]


# --------------------------------------------------------------------------- phần thuần (có test)
def normalize_class(text: str, classes: list[str]) -> str | None:
    """Tên lớp đầy đủ, hoặc tên ngắn: 'tomato late blight', 'Tomato - Late blight', 'tomato___late_blight'."""
    def key(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    table = {key(c): c for c in classes}
    table.update({key(c.replace("___", " ")): c for c in classes})
    return table.get(key(text))


def plant_test_assignment(plants_by_class: dict[str, list[str]], fraction: float = TEST_FRACTION) -> dict[tuple[str, str], str]:
    """(lớp, plant_id) -> 'train' | 'test'. Ổn định theo mã băm của mã cây, không phụ thuộc thứ tự dòng.

    Lớp có >= 2 cây: ít nhất một cây vào test (cây có mã băm nhỏ nhất) và ít nhất một cây ở train; còn lại theo ngưỡng băm."""
    out: dict[tuple[str, str], str] = {}
    for cls, plants in plants_by_class.items():
        unique = sorted(set(plants), key=lambda p: hashlib.sha256(f"{cls}|{p}".encode()).hexdigest())
        if len(unique) == 1:
            out[(cls, unique[0])] = "train"
            continue
        n_test = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
        for i, plant in enumerate(unique):
            out[(cls, plant)] = "test" if i < n_test else "train"
    return out


def validate_row(row: dict[str, str], classes: list[str]) -> tuple[str | None, str | None]:
    """(lớp chuẩn hóa, lý do loại)."""
    cls = normalize_class(row.get("class", ""), classes)
    if cls is None:
        return None, "unknown_class"
    if not row.get("plant_id", "").strip():
        return None, "missing_plant_id"
    if row.get("verified", "").strip().lower() == "unsure":
        return None, "marked_unsure"
    return cls, None


# --------------------------------------------------------------------------- I/O
def make_template(base: Path) -> None:
    (base / "raw").mkdir(parents=True, exist_ok=True)
    labels = base / "labels.csv"
    if not labels.exists():
        with labels.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerow(["vuon_A/IMG_0001.jpg", "Tomato___Late_blight", "ca_chua_01", "vuon_A", "2026-10-22", "Samsung A54", "self"])
    print(f"[own] đã tạo {labels} (dòng ví dụ ở dưới tiêu đề; xóa dòng ví dụ khi nhập thật) và thư mục raw/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    base = reg.abs_path(OWN_DIR)
    if args.template:
        make_template(base)
        return 0
    labels = base / "labels.csv"
    if not labels.is_file():
        print(f"[own] chưa có {labels}; chạy với --template để tạo mẫu", file=sys.stderr)
        return 1
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    import importlib.util

    spec = importlib.util.spec_from_file_location("image_io_standalone", ROOT / "src" / "common" / "image_io.py")
    image_io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_io)

    with labels.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    excluded: Counter[str] = Counter()
    good: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    for row in rows:
        cls, reason = validate_row(row, classes)
        if reason:
            excluded[reason] += 1
            continue
        path = base / "raw" / row["file"].strip()
        if not path.is_file():
            excluded["missing_file"] += 1
            continue
        try:
            with Image.open(path) as handle:
                handle.load()
                if min(handle.size) < 224:
                    excluded["too_small"] += 1
                    continue
                rgb = image_io.to_rgb(handle)
                w, h = rgb.size
                if max(w, h) > MAX_SIDE:
                    scale = MAX_SIDE / max(w, h)
                    rgb = rgb.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
                buffer = io.BytesIO()
                rgb.save(buffer, format="JPEG", quality=90, optimize=True)
        except Exception:  # noqa: BLE001
            excluded["unreadable"] += 1
            continue
        data = buffer.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_sha:
            excluded["duplicate"] += 1
            continue
        seen_sha.add(sha)
        good.append({"cls": cls, "row": row, "data": data, "sha": sha, "size": rgb.size, "orig": row["file"].strip()})

    plants: dict[str, list[str]] = defaultdict(list)
    for g in good:
        plants[g["cls"]].append(g["row"]["plant_id"].strip())
    assignment = plant_test_assignment(plants)
    out_rows: list[dict[str, Any]] = []
    for g in good:
        cls, plant = g["cls"], g["row"]["plant_id"].strip()
        split = assignment[(cls, plant)]
        rel = f"own_photos/images/{cls}/{g['sha'][:16]}.jpg"
        dest = base / "images" / cls / f"{g['sha'][:16]}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(g["data"])
        out_rows.append({
            "sample_id": f"own_{g['sha'][:16]}", "relative_path": rel, "byte_sha256": g["sha"], "class_id": classes.index(cls), "target_class": cls,
            "source_dataset": "own_photos", "source_domain": "plantdoc", "mapping_status": "exact_match", "quarantine_status": "clean",
            "label_trust": g["row"].get("verified", "self") or "self", "label_review_status": "trusted",
            "group_id": f"own:{cls}:{plant}", "split": split, "plant_id": plant, "location": g["row"].get("location", ""),
            "date": g["row"].get("date", ""), "device": g["row"].get("device", ""), "original_file": g["orig"],
            "width": g["size"][0], "height": g["size"][1], "license": "own", "capture_note": "tự chụp",
        })
    (base / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    test_rows = sorted(r["byte_sha256"] for r in out_rows if r["split"] == "test")
    lock = {"test_images": len(test_rows), "test_sha256_digest": hashlib.sha256("".join(test_rows).encode()).hexdigest(),
            "note": "Khóa phần thử trước khi đưa phần học vào release; không đổi sau khi đã xem kết quả."}
    (base / "test_lock.json").write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for r in out_rows:
        by_split[r["target_class"]][r["split"]] += 1
    warnings = [c for c, s in by_split.items() if s["test"] == 0]
    report = {"images": len(out_rows), "excluded": dict(excluded), "classes": len(by_split),
              "plants": {c: len(set(p)) for c, p in plants.items()}, "by_class_split": {c: dict(s) for c, s in sorted(by_split.items())},
              "warn_no_test_plants": warnings, "test_lock": lock}
    (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[own] {len(out_rows)} ảnh hợp lệ, loại {dict(excluded)}; lớp không có cây để thử: {warnings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

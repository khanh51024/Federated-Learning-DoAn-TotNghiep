"""Nhập một nguồn ảnh ngoài đồng đã tải (scripts/fetch_external_sources.py, scripts/fetch_inaturalist.py) thành manifest theo schema release v3.

Với mỗi ảnh: đọc theo đúng cách lúc huấn luyện (xoay theo EXIF, nền trong suốt dán đen, RGB), thu nhỏ cạnh dài về <= 640 px, lưu JPEG q90
vào data/external/sources/<nguồn>/images/<lớp>/<sha16>.jpg; ghi một dòng manifest: lớp (class_id), nguồn, giấy phép, tác giả/URL, tình trạng
ánh xạ (`exact_match` hoặc `alias`), độ tin nhãn (`label_trust`) và tình trạng kiểm nhãn (`label_review_status`).

`label_review_status`:
  trusted     nhãn do tác giả bộ dữ liệu khoa học gán và ánh xạ khớp đúng nghĩa: dùng được ngay
  unreviewed  nhãn cộng đồng (iNaturalist) hoặc ánh xạ gần đúng (alias): chỉ dùng sau khi kiểm nhãn mẫu đạt (scripts/build_label_review.py)

Ảnh bị loại và ghi lý do trong import_report.json: không đọc được, quá nhỏ (cạnh ngắn < min_side), trùng byte trong nguồn, khớp `exclude_name_regex`,
nhãn không thuộc class_map/alias_map.

Chạy:
    python scripts/import_external_source.py --source mendeley_tomato_bangladesh
    python scripts/import_external_source.py --all            # mọi nguồn trong YAML đã tải + mọi thư mục inat_*
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SOURCES_DIR = "data/external/sources"
MAX_SIDE = 640
JPEG_QUALITY = 90
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- phần thuần (có test)
def normalize_key(text: str) -> str:
    """So khớp tên nhãn không phân biệt hoa thường, dấu cách/gạch dưới/gạch nối: 'Early_Blight' == 'early blight'."""
    return re.sub(r"[\s_\-]+", " ", text.strip().lower())


def label_key(path: Path, raw: Path, label_from: str) -> str:
    """Khóa nhãn của một ảnh: tên thư mục chứa nó, hoặc tiền tố tên tệp (bỏ số thứ tự cuối: 'Early Blight_12' -> 'Early Blight')."""
    if label_from == "filename_prefix":
        return re.sub(r"[\s_]*\(?\d+\)?$", "", path.stem).strip()
    if label_from == "filename_regex_first_token":
        return re.split(r"[_\- ]", path.stem)[0]
    return path.parent.name


_AUG_TAIL = re.compile(r"[_~.\s]*(aug[_~\s]*\d+|enhanced|flip\w*|rot\w*)$", re.I)


def group_of(source: str, target: str, path: Path, sha: str, extra: dict[str, Any], group_from: str) -> str:
    """Nhóm chia train/test: các ảnh cùng nguồn gốc phải ở cùng một phía để không rò rỉ.

    aug_prefix: ảnh tăng cường ngoại tuyến của tác giả ('<gốc>_aug_207') gom theo '<gốc>'; observation: ảnh iNaturalist theo mã quan sát; còn lại mỗi ảnh một nhóm."""
    if "observation_id" in extra:
        return f"{source}:obs{extra['observation_id']}"
    if group_from == "aug_prefix":
        return f"{source}:{target}:{_AUG_TAIL.sub('', path.stem).strip('_~. ')}"
    return f"{source}:{sha[:16]}"


def resolve_class(key: str, class_map: dict[str, str], alias_map: dict[str, str]) -> tuple[str | None, str]:
    """(lớp đích, tình trạng ánh xạ). Không thuộc bảng nào -> (None, 'unmapped')."""
    norm = normalize_key(key)
    exact = {normalize_key(k): v for k, v in class_map.items()}
    alias = {normalize_key(k): v for k, v in alias_map.items()}
    if norm in exact:
        return exact[norm], "exact_match"
    if norm in alias:
        return alias[norm], "alias"
    return None, "unmapped"


def normalize_image_bytes(image: Any) -> bytes:
    """Ảnh PIL -> JPEG RGB đã xoay thẳng, cạnh dài <= MAX_SIDE."""
    from PIL import Image

    rgb = _to_rgb(image)
    w, h = rgb.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _to_rgb(image: Any) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("image_io_standalone", ROOT / "src" / "common" / "image_io.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.to_rgb(image)


# --------------------------------------------------------------------------- nguồn
def iter_images(raw: Path) -> Iterator[Path]:
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def inat_rows(source_dir: Path) -> dict[str, dict[str, Any]]:
    meta = source_dir / "inat_metadata.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["file"]] = row
    return rows


def import_source(name: str, spec: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    from PIL import Image

    from scripts.build_external_eval_set import dhash64

    base = reg.abs_path(SOURCES_DIR) / name
    raw = base / "raw"
    if not raw.is_dir():
        raise SystemExit(f"{name}: chưa tải (thiếu {raw})")
    is_inat = spec.get("kind") == "inaturalist"
    class_map = spec.get("class_map") or {}
    alias_map = spec.get("alias_map") or {}
    label_from = spec.get("label_from", "folder")
    exclude = re.compile(spec["exclude_name_regex"]) if spec.get("exclude_name_regex") else None
    min_side = int(spec.get("min_side", 128))
    inat = inat_rows(base) if is_inat else {}
    top_k = int(spec.get("leaf_top_k", 300))  # nhóm iNaturalist lớn: chỉ giữ các ảnh có điểm "ảnh lá" cao nhất, để mẫu 30 ảnh đại diện cho phần được nhận
    leaf_path = base / "leaf_filter.json"
    leaf_filter = json.loads(leaf_path.read_text(encoding="utf-8")) if (is_inat and leaf_path.is_file()) else None
    out_dir = base / "images"
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    per_class_cap = spec.get("max_per_class")
    per_class_count: Counter[str] = Counter()
    seen_sha: set[str] = set()
    unmapped_keys: Counter[str] = Counter()
    ordered = list(iter_images(raw))
    if is_inat and (base / "leaf_filter.json").is_file():
        scores = json.loads((base / "leaf_filter.json").read_text(encoding="utf-8"))["scores"]
        ordered.sort(key=lambda p: -scores.get(p.relative_to(raw).as_posix(), 0.0))
        keep_files = {p for p in ordered if scores.get(p.relative_to(raw).as_posix(), 0.0) >= json.loads((base / "leaf_filter.json").read_text(encoding="utf-8"))["threshold"]}
        ordered = [p for p in ordered if p in keep_files][:top_k] + [p for p in ordered if p not in keep_files]
    for path in ordered:
        rel = path.relative_to(raw).as_posix()
        if exclude and exclude.search(path.name):
            excluded["exclude_name_regex"] += 1
            continue
        extra: dict[str, Any] = {}
        if is_inat:
            meta = inat.get(path.relative_to(raw).as_posix())
            if not meta:
                excluded["no_inat_metadata"] += 1
                continue
            if leaf_filter is not None and leaf_filter["scores"].get(meta["file"], 0.0) < leaf_filter["threshold"]:
                excluded["not_leaf_photo"] += 1
                continue
            target, status = meta["target_class"], "exact_match"
            extra = {"license": meta["license"], "author": meta.get("attribution"), "url": meta.get("observation_url"),
                     "observation_id": meta["observation_id"], "taxon": meta.get("taxon_name"),
                     "leaf_score": (leaf_filter or {}).get("scores", {}).get(meta["file"])}
        elif label_from == "path_rules":
            target, status = None, "unmapped"
            for rule in spec["path_rules"]:  # luật đầu tiên khớp đường dẫn tương đối thắng
                if re.search(rule["regex"], rel):
                    target, status = rule.get("class"), rule.get("status", "exact_match")
                    break
            if target is None:
                excluded["unmapped_label"] += 1
                continue
            extra = {"license": spec.get("license"), "author": None, "url": spec.get("url")}
        elif label_from == "constant":
            target, status = spec["constant_class"], spec.get("constant_status", "alias")
            extra = {"license": spec.get("license"), "author": None, "url": spec.get("url")}
        else:
            key = label_key(path, raw, label_from)
            target, status = resolve_class(key, class_map, alias_map)
            if target is None:
                unmapped_keys[key] += 1
                excluded["unmapped_label"] += 1
                continue
            extra = {"license": spec.get("license"), "author": None, "url": spec.get("url")}
        if per_class_cap and per_class_count[target] >= int(per_class_cap):
            excluded["max_per_class"] += 1
            continue
        try:
            with Image.open(path) as handle:
                handle.load()
                width, height = handle.size
                if min(width, height) < min_side:
                    excluded["too_small"] += 1
                    continue
                data = normalize_image_bytes(handle)
        except Exception:  # noqa: BLE001 — ảnh hỏng
            excluded["unreadable"] += 1
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_sha:
            excluded["duplicate_inside_source"] += 1
            continue
        seen_sha.add(sha)
        dest = out_dir / target / f"{sha[:16]}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        with Image.open(io.BytesIO(data)) as small:
            from PIL import ImageOps

            dh = dhash64(small)
            dh_flip = dhash64(ImageOps.mirror(small.convert("RGB")))
        trusted = (not is_inat) and status == "exact_match" and spec.get("label_trust", "expert") == "expert" and not spec.get("needs_review")
        rows.append({
            "sample_id": f"ext_{name}_{sha[:16]}",
            "relative_path": f"{name}/images/{target}/{sha[:16]}.jpg",
            "byte_sha256": sha, "dhash64": f"{dh:016x}", "dhash64_flip": f"{dh_flip:016x}",
            "group_id": group_of(name, target, path, sha, extra, spec.get("group_from", "image")),
            "class_id": classes.index(target), "target_class": target,
            "source_dataset": name, "source_domain": "plantdoc",  # "ngoài đồng": runner train chỉ biết hai miền, xem docs/DATASET_V4.md
            "mapping_status": status, "quarantine_status": "clean",
            "label_trust": "community" if is_inat else spec.get("label_trust", "expert"),
            "label_review_status": "trusted" if trusted else "unreviewed",
            "capture_note": ("nhà kính" if spec.get("greenhouse") else "nền đã xóa" if spec.get("background_removed") else "ngoài đồng/vườn"),
            "original_file": rel, "width": int(width), "height": int(height), **extra,
        })
        per_class_count[target] += 1
    manifest = base / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by_class = Counter(r["target_class"] for r in rows)
    report = {
        "source": name, "title": spec.get("title"), "license": spec.get("license"), "images_imported": len(rows),
        "by_class": dict(sorted(by_class.items())), "excluded": dict(excluded),
        "unmapped_labels": dict(unmapped_keys.most_common(20)),
        "label_review_status": dict(Counter(r["label_review_status"] for r in rows)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (base / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(reg.abs_path("configs/external_sources.yaml").read_text(encoding="utf-8"))
    sources = dict(config["sources"])
    base = reg.abs_path(SOURCES_DIR)
    for directory in sorted(base.glob("inat_*")):
        sources.setdefault(directory.name, {"kind": "inaturalist", "title": f"iNaturalist {directory.name[5:]}", "license": "theo từng ảnh"})
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    wanted = list(sources) if args.all else args.source
    for name in wanted:
        if sources[name].get("skip"):
            print(f"[import] bỏ qua {name}: {sources[name].get('skip_reason')}")
            continue
        if not (base / name / "raw").is_dir():
            print(f"[import] bỏ qua {name}: chưa tải")
            continue
        report = import_source(name, sources[name], classes)
        print(f"[import] {name}: {report['images_imported']} ảnh; loại {report['excluded']}; theo lớp {report['by_class']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

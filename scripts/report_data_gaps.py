"""Báo cáo khoảng trống dữ liệu ngoài đồng theo từng lớp trong 38 lớp: còn thiếu bao nhiêu ảnh để học và để đo.

Đếm ảnh ngoài đồng "dùng được để học" (field_ready) của từng lớp từ:
  * lá PlantDoc trong tập học release v3 (exact_match);
  * lá PlantDoc đang cách ly mà nhóm lớp đã được nhận ở bước kiểm nhãn (data/label_review/status.json);
  * ảnh nguồn ngoài đồng (data/external/sources/*/manifest.jsonl): `trusted`, hoặc `unreviewed` thuộc nhóm (nguồn, lớp) đã được nhận
    (data/label_review_ext/status.json);
  * ảnh tự chụp phần học (data/own_photos/manifest.jsonl, do scripts/check_own_photos.py sinh).
Ngưỡng: tối thiểu 100 ảnh để cho học, mục tiêu 300 ảnh từ >= 2 nguồn. Báo thêm số ảnh của các tập test đã khóa theo lớp, và trần độ chính xác của
Local-only (mỗi client chỉ thấy một phần lớp nên không thể đoán đúng lớp chưa từng học).

Chạy:
    python scripts/report_data_gaps.py        # ghi artifacts/data_gap_report.json và in bảng
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

MIN_READY = 100
TARGET_READY = 300
MIN_SOURCES = 2
EXT_SOURCES = "data/external/sources"
EXT_STATUS = "data/label_review_ext/status.json"
QUARANTINE_STATUS = "data/label_review/status.json"
QUARANTINE_IMAGE_STATUS = "data/label_review_q2/status.json"
OWN_PHOTOS = "data/own_photos/manifest.jsonl"
PLANTWILD_MANIFEST = "data/external/plantwild_overlap_manifest.json"
V4_PARTITION_SPEC = "data/dataset_sources/mixed/pv_pd_v4/partition_spec_v4.json"
CEILING_SCHEME = "label_alpha_0_1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _accepted(path: str, key: str = "accepted") -> set[str]:
    p = reg.abs_path(path)
    if not p.is_file():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get(key, []))


def readiness(ready: int, sources: int) -> str:
    if ready >= TARGET_READY and sources >= MIN_SOURCES:
        return "dat_muc_tieu"
    if ready >= MIN_READY:
        return "du_toi_thieu"
    return "thieu"


def shards_from_partition_spec(spec: dict[str, Any], classes: list[str], scheme: str = CEILING_SCHEME) -> dict[str, Any]:
    """Số ảnh mỗi lớp của từng client, đọc từ client_stats của partition_spec_v4.json: {client: {"per_class": {tên lớp: số ảnh}}}."""
    stats = spec["scenarios"][scheme]["client_stats"]
    return {client: {"per_class": {classes[int(i)]: n for i, n in s["class_histogram"].items()}} for client, s in stats.items()}


def local_only_ceilings(classes: list[str], counts_by_set: dict[str, Counter[str]], shards: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trần top-1 của từng client Local-only trên mỗi tập đo: tỉ lệ ảnh thuộc lớp mà shard của client có ảnh.

    `shards` mặc định là phân vùng của v3 (artifacts/client_shard_class_counts.json); build_report() truyền phân vùng của v4 nếu release v4 đã dựng."""
    if shards is None:
        shards = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["clients"]
    out: dict[str, Any] = {}
    for client, info in shards.items():
        seen = {name for name, count in info["per_class"].items() if (count if isinstance(count, int) else sum(count.values())) > 0}
        out[client] = {"classes_seen": len(seen)}
        for set_name, counts in counts_by_set.items():
            total = sum(counts.values())
            out[client][set_name] = round(100.0 * sum(v for k, v in counts.items() if k in seen) / total, 1) if total else None
    return out


def build_report() -> dict[str, Any]:
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    train = _jsonl(release / "manifests" / "train.jsonl")
    val = _jsonl(release / "manifests" / "val.jsonl")
    test = _jsonl(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    quarantine = _jsonl(release / "manifests" / "quarantine.jsonl")
    q_ok = _accepted(QUARANTINE_STATUS)
    q_samples_ok = _accepted(QUARANTINE_IMAGE_STATUS, "accepted_samples")
    ext_ok = _accepted(EXT_STATUS)
    ext_samples_ok = _accepted(EXT_STATUS, "accepted_samples")

    pv_train, pd_train = Counter(), Counter()
    for r in train:
        (pv_train if r["source_domain"] == "plantvillage" else pd_train)[r["target_class"]] += 1
    pd_quarantine: Counter[str] = Counter()
    pd_quarantine_ready: Counter[str] = Counter()
    for r in quarantine:
        if r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean" and r.get("target_class") and int(r.get("class_id", -1)) >= 0:
            pd_quarantine[r["target_class"]] += 1
            if r["target_class"] in q_ok or r["sample_id"] in q_samples_ok:
                pd_quarantine_ready[r["target_class"]] += 1

    ext_ready: dict[str, Counter[str]] = defaultdict(Counter)
    ext_pending: Counter[str] = Counter()
    ext_sources_total: dict[str, Counter[str]] = defaultdict(Counter)
    for manifest in sorted(reg.abs_path(EXT_SOURCES).glob("*/manifest.jsonl")):
        for r in _jsonl(manifest):
            cls, src = r["target_class"], r["source_dataset"]
            ext_sources_total[cls][src] += 1
            if r["label_review_status"] == "trusted" or f"{src}|{cls}" in ext_ok or r["sample_id"] in ext_samples_ok:
                ext_ready[cls][src] += 1
            else:
                ext_pending[cls] += 1
    own_train: Counter[str] = Counter()
    own_test: Counter[str] = Counter()
    for r in _jsonl(reg.abs_path(OWN_PHOTOS)):
        (own_test if r.get("split") == "test" else own_train)[r["target_class"]] += 1

    pd_val, pd_test = Counter(), Counter()
    for r in val:
        if r["source_domain"] == "plantdoc":
            pd_val[r["target_class"]] += 1
    for r in test:
        if r["source_domain"] == "plantdoc":
            pd_test[r["target_class"]] += 1
    wild = reg.abs_path(PLANTWILD_MANIFEST)
    plantwild: Counter[str] = Counter()
    if wild.is_file():
        for r in json.loads(wild.read_text(encoding="utf-8"))["samples"]:
            plantwild[r["target_class"]] += 1

    spec_path = reg.abs_path(V4_PARTITION_SPEC)
    ceiling_shards = shards_from_partition_spec(json.loads(spec_path.read_text(encoding="utf-8")), classes) if spec_path.is_file() else None

    per_class = []
    for name in classes:
        sources: set[str] = set()
        if pd_train[name]:
            sources.add("plantdoc_v3")
        if pd_quarantine_ready[name]:
            sources.add("plantdoc_quarantine")
        sources.update(ext_ready[name])
        if own_train[name]:
            sources.add("own_photos")
        ready = pd_train[name] + pd_quarantine_ready[name] + sum(ext_ready[name].values()) + own_train[name]
        per_class.append({
            "class": name, "pv_train": pv_train[name],
            "field_ready": ready, "field_sources": sorted(sources), "status": readiness(ready, len(sources)),
            "missing_to_target": max(0, TARGET_READY - ready),
            "parts": {"plantdoc_v3": pd_train[name], "plantdoc_quarantine_total": pd_quarantine[name],
                      "plantdoc_quarantine_ready": pd_quarantine_ready[name], "external_ready": dict(ext_ready[name]),
                      "external_pending_review": ext_pending[name], "own_photos_train": own_train[name]},
            "tests": {"plantdoc_val": pd_val[name], "plantdoc_test": pd_test[name], "plantwild": plantwild[name], "own_photos_test": own_test[name]},
        })
    summary = Counter(c["status"] for c in per_class)
    counts_by_set = {"plantvillage_val": Counter(r["target_class"] for r in val if r["source_domain"] == "plantvillage"),
                     "plantdoc_val": pd_val, "plantwild": plantwild}
    return {
        "thresholds": {"min_ready": MIN_READY, "target_ready": TARGET_READY, "min_sources": MIN_SOURCES},
        "summary": {"classes": len(classes), **dict(summary), "total_field_ready": sum(c["field_ready"] for c in per_class)},
        "classes": per_class,
        "local_only_ceiling_top1_percent": local_only_ceilings(classes, counts_by_set, ceiling_shards),
        "local_only_ceiling_source": "partition_spec_v4.json (label_alpha_0_1, release v4)" if ceiling_shards else "client_shard_class_counts.json (release v3)",
        "note": "field_ready chỉ đếm ảnh ngoài đồng đã qua kiểm nhãn hoặc nhãn khoa học tin cậy; PlantWild không được đếm vào học (chỉ để thử).",
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    report = build_report()
    out = reg.abs_path(reg.DATA_GAP_REPORT_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[gaps] {report['summary']}")
    print(f"{'lớp':46s} {'PV':>5s} {'sẵn sàng':>8s} {'nguồn':>5s} {'chờ kiểm':>8s}  trạng thái")
    for c in report["classes"]:
        print(f"{c['class'][:46]:46s} {c['pv_train']:5d} {c['field_ready']:8d} {len(c['field_sources']):5d} {c['parts']['external_pending_review']:8d}  {c['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

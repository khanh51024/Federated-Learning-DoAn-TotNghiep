"""Phân bố lớp trong shard huấn luyện của từng client (partition label_alpha_0_1 của release mixed_pv_pd_v3).

Dùng để GIẢI THÍCH kết quả Local-only trên web: một Local-only không nhận ra được bệnh mà shard của nó
không có ảnh. Nguồn: manifest release đã kiểm hash (data/dataset_sources/mixed/pv_pd_v3), đối chiếu với
`local/combined_summary.json` của bundle (n_k từng client) và với train.jsonl (hợp các shard = train).

Chạy: python scripts/report_client_shards.py   ->  artifacts/client_shard_class_counts.json

Máy không có dataset (vd. server demo): giữ nguyên file đã sinh, KHÔNG đoán; chưa có file thì thoát mã 1.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

NUM_CLIENTS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_report() -> dict[str, Any]:
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    shard_dir = release / "partitions" / reg.PARTITION_SCHEME
    needed = [shard_dir / f"client_{i:02d}.jsonl" for i in range(NUM_CLIENTS)] + [
        release / "manifests" / "train.jsonl",
        release / "release_manifest.json",
    ]
    missing = [p.relative_to(ROOT).as_posix() for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"thiếu dữ liệu release: {missing}")

    classes = json.loads(reg.abs_path(reg.DATASET_MANIFEST_JSON).read_text(encoding="utf-8"))["classes"]
    known = set(classes)
    release_manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    train_hash_ok = _sha256(release / "manifests" / "train.jsonl") == release_manifest["manifest_hashes"]["train"]
    train_ids = {row["sample_id"] for row in _jsonl(release / "manifests" / "train.jsonl")}

    summary = json.loads(reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/local/combined_summary.json").read_text(encoding="utf-8"))
    published_nk = {name: int(info["n_k"]) for name, info in summary["clients"].items()}

    clients: dict[str, Any] = {}
    union: set[str] = set()
    shard_hashes: dict[str, str] = {}
    totals = {name: 0 for name in classes}
    unknown_classes: set[str] = set()
    for index in range(NUM_CLIENTS):
        name = f"client_{index:02d}"
        path = shard_dir / f"{name}.jsonl"
        shard_hashes[name] = _sha256(path)
        counts = {cls: 0 for cls in classes}
        rows = _jsonl(path)
        for row in rows:
            cls = row["target_class"]
            if cls not in known:
                unknown_classes.add(cls)
                continue
            counts[cls] += 1
            totals[cls] += 1
        union.update(row["sample_id"] for row in rows)
        clients[name] = {
            "n_k": len(rows),
            "classes_with_images": sum(1 for v in counts.values() if v > 0),
            "per_class": counts,
        }

    checks = {
        "train_manifest_hash_matches_release": train_hash_ok,
        "union_of_shards_equals_train": union == train_ids,
        "shards_disjoint": sum(c["n_k"] for c in clients.values()) == len(union),
        "n_k_matches_bundle_summary": {n: clients[n]["n_k"] for n in clients} == published_nk,
        "all_classes_in_taxonomy": not unknown_classes,
        # release_manifest chỉ ghi hash shard của partition `iid`; label_alpha_0_1 được đối chiếu bằng n_k + hợp = train
        "release_manifest_has_shard_hashes_for_scheme": reg.PARTITION_SCHEME in (release_manifest.get("partition_shard_hashes") or {}),
    }
    return {
        "evidence": "đếm từ manifest release đã kiểm hash; đối chiếu n_k với local/combined_summary.json của bundle",
        "release": release_manifest.get("release_name"),
        "partition_scheme": reg.PARTITION_SCHEME,
        "protocol_id": reg.DEMO_PROTOCOL_ID,
        "classes": classes,
        "total_train_samples": len(train_ids),
        "clients": clients,
        "totals_all_clients": totals,
        "shard_files_sha256": shard_hashes,
        "checks": checks,
        "consistent": all(v for k, v in checks.items() if k != "release_manifest_has_shard_hashes_for_scheme"),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1258) không in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON)
    try:
        report = build_report()
    except FileNotFoundError as exc:
        if out.is_file():
            print(f"[report_client_shards] {exc} — giữ nguyên {reg.CLIENT_SHARD_COUNTS_JSON} đã sinh trước đó")
            return 0
        print(f"[report_client_shards] {exc} và chưa có {reg.CLIENT_SHARD_COUNTS_JSON}", file=sys.stderr)
        return 1
    if not report["consistent"]:
        print(f"[report_client_shards] KHÔNG nhất quán: {report['checks']}", file=sys.stderr)
        return 2
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sizes = ", ".join(f"{n}={c['n_k']}" for n, c in report["clients"].items())
    print(f"[report_client_shards] wrote {reg.CLIENT_SHARD_COUNTS_JSON} ({sizes}; tổng {report['total_train_samples']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

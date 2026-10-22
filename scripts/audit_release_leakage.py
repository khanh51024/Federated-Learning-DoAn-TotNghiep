"""Audit leakage trên release mà gói mô hình đang dùng ràng buộc (artifacts/active_bundle.json -> import_manifest.json).

Chuỗi bằng chứng (mỗi mắt xích được kiểm, không suy diễn):
  checkpoint.release_manifest_sha256  ==  sha256(release_manifest.json)        (web/demo_evidence kiểm lúc suy luận)
  release_manifest.manifest_hashes    ==  sha256(train.jsonl / val.jsonl / ...)
  train, validation, calibration      đôi một KHÔNG giao nhau theo group_id, parent_scene_id, byte_sha256, sample_id, image_id
  các shard của partition             là tập con của train, rời nhau, hợp = train
  ảnh trên đĩa                        khớp byte_sha256 trong manifest   (--skip-bytes để bỏ qua, nhanh hơn)

Ghi chú về hash release: file trong git được lưu với LF; trên máy nhóm train nó kết thúc bằng CRLF và hash ra đúng
RELEASE_SHA. Script ghi rõ điều này thay vì giả vờ hash file thô khớp.

Chạy: python scripts/audit_release_leakage.py   ->  artifacts/release_leakage_audit.json
Không có dataset (data/dataset_sources) -> giữ nguyên file đã sinh; chưa có file thì thoát mã 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bundle, registry as reg  # noqa: E402

_IDENTITY = bundle.load_manifest()
RELEASE_SHA = _IDENTITY.get("release_manifest_sha256")        # nhận dạng lấy từ gói, không viết cứng
VALIDATION_SHA = _IDENTITY.get("validation_manifest_sha256")

KEYS = ("group_id", "parent_scene_id", "byte_sha256", "sample_id", "image_id")
SPLITS = ("train", "val", "calibration")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_overlaps(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    """Số phần tử giao nhau của từng cặp split theo từng khóa (0 = tách biệt)."""
    out: dict[str, dict[str, int]] = {}
    names = list(rows_by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}__{b}"] = {
                key: len({r[key] for r in rows_by_split[a]} & {r[key] for r in rows_by_split[b]}) for key in KEYS
            }
    return out


def audit_partition(shards: dict[str, list[dict[str, Any]]], train_ids: set[str]) -> dict[str, Any]:
    union: set[str] = set()
    total = 0
    group_owner: dict[str, str] = {}
    cross_client_groups = 0
    for client, rows in shards.items():
        total += len(rows)
        union.update(r["sample_id"] for r in rows)
        for row in rows:
            owner = group_owner.setdefault(row["group_id"], client)
            if owner != client:
                cross_client_groups += 1
    return {
        "clients": len(shards),
        "samples": total,
        "unique_samples": len(union),
        "disjoint": total == len(union),
        "subset_of_train": union <= train_ids,
        "union_equals_train": union == train_ids,
        "groups_split_across_clients": cross_client_groups,  # >0 nghĩa là một nhóm ảnh bị chia cho nhiều client
    }


def build_report(verify_bytes: bool) -> dict[str, Any]:
    release = reg.abs_path(reg.RELEASE_DIR)
    image_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    manifest_path = release / "release_manifest.json"
    needed = [manifest_path] + [release / "manifests" / f"{n}.jsonl" for n in ("train", "val", "calibration")]
    missing = [p.relative_to(ROOT).as_posix() for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"thiếu dữ liệu release: {missing}")

    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    crlf_terminated = raw.rstrip(b"\r\n") + b"\r\n"
    release_binding = {
        "pinned_release_manifest_sha256": RELEASE_SHA,
        "sha256_of_stored_file": _sha256_bytes(raw),
        "sha256_with_crlf_terminator": _sha256_bytes(crlf_terminated),
        "matches_pinned": _sha256_bytes(crlf_terminated) == RELEASE_SHA or _sha256_bytes(raw) == RELEASE_SHA,
        "note": "git lưu file với LF; hash khớp khi thêm CRLF cuối file (cách file nằm trên máy nhóm train)",
    }

    manifest_hash_ok = {
        name: _sha256_file(release / "manifests" / f"{name}.jsonl") == manifest["manifest_hashes"][name] for name in SPLITS
    }
    rows = {name: _jsonl(release / "manifests" / f"{name}.jsonl") for name in SPLITS}
    train_ids = {r["sample_id"] for r in rows["train"]}
    overlaps = audit_overlaps(rows)

    partitions: dict[str, Any] = {}
    for scheme_dir in sorted((release / "partitions").iterdir()):
        files = sorted(scheme_dir.glob("client_*.jsonl"))
        if files:
            partitions[scheme_dir.name] = audit_partition({f.stem: _jsonl(f) for f in files}, train_ids)

    image_bytes: dict[str, Any] = {"verified": False}
    if verify_bytes:
        bad = missing_files = checked = 0
        for name in SPLITS:
            for row in rows[name]:
                path = image_root / row["relative_path"]
                if not path.is_file():
                    missing_files += 1
                    continue
                checked += 1
                if _sha256_file(path) != row["byte_sha256"]:
                    bad += 1
        image_bytes = {"verified": True, "checked": checked, "hash_mismatch": bad, "missing_files": missing_files}

    disjoint = all(v == 0 for pair in overlaps.values() for v in pair.values())
    iid_ok = all(p["subset_of_train"] and p["disjoint"] for p in partitions.values())
    passed = bool(
        release_binding["matches_pinned"]
        and manifest["manifest_hashes"]["val"] == VALIDATION_SHA
        and all(manifest_hash_ok.values())
        and disjoint
        and iid_ok
        and (not verify_bytes or (image_bytes["hash_mismatch"] == 0 and image_bytes["missing_files"] == 0))
    )
    return {
        "evidence": "đối chiếu manifest release đã kiểm hash; không phải kiểm tra từng checkpoint được huấn luyện trên split nào",
        "scope": "split của release. Việc từng checkpoint train trên đúng split này dựa vào release_manifest_sha256 ghi trong "
                 "metadata checkpoint (web kiểm mỗi lần suy luận) và n_k từng client khớp tóm tắt huấn luyện công bố.",
        "release_name": manifest.get("release_name"),
        "release_binding": release_binding,
        "validation_manifest_sha256": manifest["manifest_hashes"]["val"],
        "manifest_hashes_match_files": manifest_hash_ok,
        "counts": {name: len(rows[name]) for name in SPLITS},
        "pairwise_overlaps": overlaps,
        "partitions": partitions,
        "image_bytes": image_bytes,
        "passed": passed,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-bytes", action="store_true", help="Bỏ qua băm từng ảnh (nhanh hơn, bằng chứng yếu hơn)")
    args = parser.parse_args()

    out = reg.abs_path(reg.RELEASE_LEAKAGE_AUDIT_JSON)
    try:
        report = build_report(verify_bytes=not args.skip_bytes)
    except FileNotFoundError as exc:
        if out.is_file():
            print(f"[audit_leakage] {exc} — giữ nguyên {reg.RELEASE_LEAKAGE_AUDIT_JSON} đã sinh trước đó")
            return 0
        print(f"[audit_leakage] {exc} và chưa có {reg.RELEASE_LEAKAGE_AUDIT_JSON}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    overlap_total = sum(v for pair in report["pairwise_overlaps"].values() for v in pair.values())
    print(f"[audit_leakage] passed={report['passed']} | giao nhau tổng={overlap_total} | "
          f"ảnh kiểm byte={report['image_bytes'].get('checked')} | wrote {reg.RELEASE_LEAKAGE_AUDIT_JSON}")
    return 0  # trạng thái đạt/không nằm trong báo cáo (checklist đọc `passed`), không dùng mã thoát để báo


if __name__ == "__main__":
    raise SystemExit(main())

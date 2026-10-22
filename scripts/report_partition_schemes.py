"""Phân bố lớp theo client cho MỌI cách chia dữ liệu của release mà gói mô hình đang dùng khai báo (chỉ dữ liệu, không có số của mô hình).

Dùng cho trình xem non-IID ở bước 2 của web: so sánh IID, lệch nhãn (Dirichlet α), lệch số lượng, lệch đặc trưng và
"mixed_stress". Chỉ α=0,1 (label_alpha_0_1) có mô hình đã huấn luyện; các cách chia khác chỉ để minh họa dữ liệu.

Kiểm tra trước khi ghi (không đạt thì dừng):
  - hash từng file shard khớp partition_spec_v4.json (các kịch bản của release; v4 có 2: label_alpha_0_1 và iid) hoặc release_manifest.json (iid)
  - mỗi cách chia: các shard rời nhau, hợp lại đúng bằng train.jsonl
  - label_alpha_0_1 khớp n_k của bundle đã kiểm hash (local/combined_summary.json)

Chạy: python scripts/report_partition_schemes.py  ->  artifacts/partition_schemes_class_counts.json
Không có dataset: giữ nguyên file đã sinh; chưa có thì thoát mã 1.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

FAMILY = {
    "iid": "Đồng đều (IID)",
    "label": "Lệch nhãn (Dirichlet α)",
    "quantity": "Lệch số lượng ảnh",
    "feature": "Lệch nguồn ảnh (PlantVillage/PlantDoc)",
    "mixed": "Kết hợp nhiều kiểu lệch",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def describe(scheme: str) -> dict[str, Any]:
    """'label_alpha_0_1' -> họ 'label', α = 0.1."""
    if scheme == "iid":
        return {"family": "iid", "family_label": FAMILY["iid"], "alpha": None}
    if scheme == "mixed_stress":
        return {"family": "mixed", "family_label": FAMILY["mixed"], "alpha": None}
    family, _, value = scheme.partition("_alpha_")
    return {"family": family, "family_label": FAMILY.get(family, family), "alpha": float(value.replace("_", "."))}


def scheme_counts(shards: dict[str, list[dict[str, Any]]], classes: list[str]) -> dict[str, Any]:
    clients = {}
    for name, rows in sorted(shards.items()):
        per_class = Counter(r["target_class"] for r in rows)
        plantdoc = sum(1 for r in rows if r["source_dataset"] == "PlantDoc")
        clients[name] = {
            "n_k": len(rows),
            "classes_with_images": sum(1 for c in classes if per_class.get(c)),
            "plantdoc_share": round(plantdoc / len(rows), 4) if rows else 0.0,
            "per_class": [int(per_class.get(c, 0)) for c in classes],  # theo thứ tự `classes`
        }
    return clients


def build() -> dict[str, Any]:
    release = reg.abs_path(reg.RELEASE_DIR)
    spec = json.loads((release / "partition_spec_v4.json").read_text(encoding="utf-8"))
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    shard_report = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))
    classes = list(shard_report["classes"])
    train = _jsonl(release / "manifests" / "train.jsonl")
    train_ids = {r["sample_id"] for r in train}
    if _sha256(release / "manifests" / "train.jsonl") != manifest["manifest_hashes"]["train"]:
        raise ValueError("train.jsonl khác hash trong release_manifest.json")

    expected_hashes = {name: s["shard_hashes"] for name, s in spec["scenarios"].items()}
    expected_hashes.update(manifest.get("partition_shard_hashes") or {})
    schemes: dict[str, Any] = {}
    for scheme_dir in sorted(p for p in (release / "partitions").iterdir() if p.is_dir()):
        name = scheme_dir.name
        pinned = expected_hashes.get(name)
        if not pinned:
            raise ValueError(f"{name}: không có hash ghim trong partition_spec_v4.json/release_manifest.json")
        shards = {}
        for file_name, digest in sorted(pinned.items()):
            path = scheme_dir / file_name
            if _sha256(path) != digest:
                raise ValueError(f"{name}/{file_name}: hash khác bản ghim")
            shards[Path(file_name).stem] = _jsonl(path)
        ids = [r["sample_id"] for rows in shards.values() for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != train_ids:
            raise ValueError(f"{name}: các shard không rời nhau hoặc không phủ đúng train")
        spec_entry = spec["scenarios"].get(name, {})
        schemes[name] = {
            **describe(name),
            "has_trained_models": name == reg.PARTITION_SCHEME,
            "mean_js_divergence": spec_entry.get("mean_js_divergence"),
            "clients": scheme_counts(shards, classes),
        }

    published = {c: v["n_k"] for c, v in shard_report["clients"].items()}
    trained = {c: v["n_k"] for c, v in schemes[reg.PARTITION_SCHEME]["clients"].items()}
    if trained != published:
        raise ValueError(f"{reg.PARTITION_SCHEME}: n_k khác báo cáo shard đã kiểm")
    return {
        "evidence": "đếm từ manifest partition của release đã kiểm hash (chỉ dữ liệu; không có số của mô hình)",
        "release": manifest["release_name"],
        "protocol_id": shard_report["protocol_id"],
        "classes": classes,
        "total_train_samples": len(train),
        "trained_scheme": reg.PARTITION_SCHEME,
        "schemes": schemes,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = reg.abs_path(reg.PARTITION_SCHEMES_JSON)
    if not (reg.abs_path(reg.RELEASE_DIR) / "partition_spec_v4.json").is_file():
        if out.is_file():
            print(f"[partition_schemes] thiếu dataset — giữ nguyên {reg.PARTITION_SCHEMES_JSON}")
            return 0
        print(f"[partition_schemes] thiếu dataset ở {reg.RELEASE_DIR}", file=sys.stderr)
        return 1
    report = build()
    out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    for name, scheme in report["schemes"].items():
        sizes = [c["n_k"] for c in scheme["clients"].values()]
        classes = [c["classes_with_images"] for c in scheme["clients"].values()]
        print(f"[partition_schemes] {name:20} n_k={sizes} lớp/client={classes}")
    print(f"[partition_schemes] wrote {reg.PARTITION_SCHEMES_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

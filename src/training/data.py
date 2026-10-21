"""Đọc release v4 (manifest, phân vùng client) và tạo Dataset; trọng số mẫu cân bằng ảnh phòng thí nghiệm với ảnh ngoài đồng."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "calibration", "legacy_diagnostic_test")
NUM_CLIENTS = 5
LAB_DOMAIN = "plantvillage"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_release(release_dir: Path, expected_sha: str | None) -> dict[str, Any]:
    """Manifest các split và SHA-256 của release; dừng nếu SHA khác giá trị được ghim (release đã bị sửa hoặc nhầm bản)."""
    manifest_path = release_dir / "release_manifest.json"
    actual = sha256_file(manifest_path)
    if expected_sha and actual != expected_sha:
        raise SystemExit(f"release_manifest.json có SHA-256 {actual}, khác giá trị ghim {expected_sha}")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in meta["manifest_hashes"].items():
        path = release_dir / "manifests" / f"{name}.jsonl"
        if path.is_file() and sha256_file(path) != expected:
            raise SystemExit(f"manifests/{name}.jsonl khác mã băm trong release_manifest.json")
    return {"sha256": actual, "meta": meta, **{name: read_jsonl(release_dir / "manifests" / f"{name}.jsonl") for name in SPLITS}}


def load_shard(release_dir: Path, scheme: str, client: int, release: dict[str, Any]) -> list[dict[str, Any]]:
    path = release_dir / "partitions" / scheme / f"client_{client:02d}.jsonl"
    expected = release["meta"]["partition_shard_hashes"][scheme][path.name]
    if sha256_file(path) != expected:
        raise SystemExit(f"{path} khác mã băm trong release_manifest.json")
    return read_jsonl(path)


def domain_balanced_weights(rows: list[dict[str, Any]]) -> list[float]:
    """Trọng số mất mát từng mẫu sao cho tổng trọng số của ảnh phòng thí nghiệm (PlantVillage) bằng tổng trọng số ảnh ngoài đồng; trung bình bằng 1.

    Nếu tập chỉ có một miền thì mọi trọng số bằng 1. Tính trên chính tập được học (với FedAvg và Local-only là shard của client)."""
    counts = Counter(r["source_domain"] == LAB_DOMAIN for r in rows)
    if len(counts) < 2:
        return [1.0] * len(rows)
    n = len(rows)
    return [n / (2 * counts[r["source_domain"] == LAB_DOMAIN]) for r in rows]


class ManifestDataset:
    """Ảnh theo manifest; giải mã chuẩn của ứng dụng (xoay EXIF, nền trong suốt thành đen, RGB). Trả (ảnh đã biến đổi, nhãn, chỉ số)."""

    def __init__(self, rows: list[dict[str, Any]], root: Path, transform: Any):
        self.rows, self.root, self.transform = rows, Path(root), transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        from PIL import Image

        from src.common.image_io import to_rgb

        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as handle:
            image = to_rgb(handle)
        return self.transform(image), int(row["class_id"]), index

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


EXTRA_DIR = "partitions_extra"
EXTRA_MANIFEST = "extra_manifest.json"


def load_extra_manifest(release_dir: Path, release: dict[str, Any]) -> dict[str, Any] | None:
    """Cách chia bổ sung (scripts/build_extra_partitions.py) nằm ngoài release_manifest.json để mã băm release không đổi; manifest riêng của
    chúng phải ghi đúng mã băm release đang dùng. None khi không có."""
    path = release_dir / EXTRA_DIR / EXTRA_MANIFEST
    if not path.is_file():
        return None
    extra = json.loads(path.read_text(encoding="utf-8"))
    if extra.get("release_manifest_sha256") != release["sha256"]:
        raise SystemExit(f"{path} dựng cho release {extra.get('release_manifest_sha256')}, khác release đang dùng {release['sha256']}")
    return extra


def extra_manifest_sha256(release_dir: Path) -> str | None:
    path = release_dir / EXTRA_DIR / EXTRA_MANIFEST
    return sha256_file(path) if path.is_file() else None


def load_shard(release_dir: Path, scheme: str, client: int, release: dict[str, Any]) -> list[dict[str, Any]]:
    name = f"client_{client:02d}.jsonl"
    in_release = release["meta"]["partition_shard_hashes"].get(scheme)
    if in_release is not None:
        path, expected = release_dir / "partitions" / scheme / name, in_release[name]
    else:
        extra = load_extra_manifest(release_dir, release)
        if extra is None or scheme not in extra["schemes"]:
            raise SystemExit(f"cách chia '{scheme}' không có trong release_manifest.json cũng không có trong {EXTRA_DIR}/{EXTRA_MANIFEST}")
        path, expected = release_dir / EXTRA_DIR / scheme / name, extra["schemes"][scheme]["shard_sha256"][name]
    if sha256_file(path) != expected:
        raise SystemExit(f"{path} khác mã băm đã ghim")
    return read_jsonl(path)


def domain_balanced_weights(rows: list[dict[str, Any]]) -> list[float]:
    """Trọng số mất mát từng mẫu sao cho tổng trọng số của ảnh phòng thí nghiệm (PlantVillage) bằng tổng trọng số ảnh ngoài đồng; trung bình bằng 1.

    Nếu tập chỉ có một miền thì mọi trọng số bằng 1. Tính trên chính tập được học (với FedAvg và Local-only là shard của client)."""
    counts = Counter(r["source_domain"] == LAB_DOMAIN for r in rows)
    if len(counts) < 2:
        return [1.0] * len(rows)
    n = len(rows)
    return [n / (2 * counts[r["source_domain"] == LAB_DOMAIN]) for r in rows]


def balanced_sampling_weights(rows: list[dict[str, Any]], class_power: float = 1.0) -> list[float]:
    """Xác suất chọn từng ảnh (tổng bằng 1) cho bộ lấy mẫu có hoàn lại thay cho trọng số mất mát.

    Hai miền (phòng thí nghiệm, ngoài đồng) được chọn bằng nhau; trong mỗi miền, lớp c được chọn tỉ lệ với n_c^class_power rồi chia đều cho ảnh
    của lớp. class_power = 1 cho đúng tỉ lệ của `domain_balanced_weights`; 0,5 (căn bậc hai) nâng vừa phải các lớp ít ảnh; 0 là cân bằng lớp hoàn toàn."""
    groups: dict[bool, Counter[int]] = {}
    for r in rows:
        groups.setdefault(r["source_domain"] == LAB_DOMAIN, Counter())[int(r["class_id"])] += 1
    mass = {d: sum(n ** class_power for n in counts.values()) for d, counts in groups.items()}
    out = []
    for r in rows:
        d, c = r["source_domain"] == LAB_DOMAIN, int(r["class_id"])
        n = groups[d][c]
        out.append((1 / len(groups)) * (n ** class_power / mass[d]) / n)
    return out


def class_prior(rows: list[dict[str, Any]], weights: list[float], num_classes: int = 38) -> list[float]:
    """Tần suất lớp mà mô hình thực sự thấy khi học (theo xác suất lấy mẫu hoặc trọng số mất mát), tổng bằng 1; dùng cho logit adjustment."""
    prior = [0.0] * num_classes
    for r, w in zip(rows, weights):
        prior[int(r["class_id"])] += w
    total = sum(prior)
    return [p / total for p in prior]


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

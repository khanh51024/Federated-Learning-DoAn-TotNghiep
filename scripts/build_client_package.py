"""Dựng gói client cho nút "Tải bộ cài" trên trang #flower: có sẵn Python, không có gì cần cài thêm.

Python và thư viện lấy từ gói laptop 2 đã dựng (--source-zip). Mã lấy từ repo. Dữ liệu chỉ gồm ảnh của client, không có ảnh hay danh sách kiểm tra của máy chủ:
    partition/, images_client_05, images_client_06   client 5 và 6 của lần đo 7 client (tmp/net10)
    mini10/partition, mini10/images_client_00..09     bộ ảnh nhẹ (tmp/mini10)

Chạy:
    python scripts/build_client_package.py
Rồi chép <out>/goi_client.zip và <out>/manifest.json lên ~/client-package của máy chủ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREFIXES = ("python/", "thu_vien/")
NET_CLIENTS = (5, 6)
LIGHT_CLIENTS = range(10)


def _client_files(data: Path, prefix: str, clients) -> Iterator[tuple[str, Path]]:
    """Bản kê và ảnh của đúng các client đã chọn. Không lấy global_val, global_test, images_eval: đó là phần của máy chủ."""
    yield f"{prefix}partition/partition_config.json", data / "partition" / "partition_config.json"
    for client in clients:
        name = f"client_{client:02d}.csv"
        yield f"{prefix}partition/clients/{name}", data / "partition" / "clients" / name
        images = data / f"images_client_{client:02d}"
        for path in sorted(images.rglob("*")):
            if path.is_file():
                yield f"{prefix}images_client_{client:02d}/{path.relative_to(images).as_posix()}", path


def plain_members(root: Path, net10: Path, mini10: Path) -> list[tuple[str, Path]]:
    """Mọi tệp trong gói trừ Python và thư viện: (tên trong gói, tệp nguồn)."""
    members = [(path.relative_to(root).as_posix(), path) for path in sorted((root / "src").rglob("*.py"))]
    members += [(path.name, path) for path in sorted((root / "deploy" / "lan_client").iterdir()) if path.is_file()]
    members += _client_files(net10, "", NET_CLIENTS)
    members += _client_files(mini10, "mini10/", LIGHT_CLIENTS)
    return members


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source_zip: Path, root: Path, net10: Path, mini10: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = out / "goi_client.zip"
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(source_zip) as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
        for entry in source.infolist():
            if entry.filename.startswith(RUNTIME_PREFIXES) and not entry.is_dir():
                package.writestr(entry, source.read(entry), compress_type=zipfile.ZIP_DEFLATED)
        for name, path in plain_members(root, net10, mini10):
            package.write(path, name)
    temporary.replace(target)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"sha256": _sha256(target), "bytes": target.stat().st_size, "commit": commit}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-zip", default="tmp/LAPTOP_2_10_CLIENT.zip", help="gói laptop 2 có sẵn python/ và thu_vien/")
    parser.add_argument("--net10", default="tmp/net10")
    parser.add_argument("--mini10", default="tmp/mini10")
    parser.add_argument("--out", default="tmp/client_package")
    args = parser.parse_args()
    source_zip, net10, mini10 = ROOT / args.source_zip, ROOT / args.net10, ROOT / args.mini10
    for need in (source_zip, net10 / "partition", mini10 / "partition"):
        if not need.exists():
            print(f"[package] thiếu {need}", file=sys.stderr)
            return 1
    manifest = build(source_zip, ROOT, net10, mini10, ROOT / args.out)
    print(f"[package] {manifest['bytes'] / 1_000_000:.0f} MB, sha256 {manifest['sha256']}, commit {manifest['commit'][:9]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

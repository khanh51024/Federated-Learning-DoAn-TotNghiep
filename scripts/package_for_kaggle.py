"""Script đóng gói mã nguồn và partitions v5 phục vụ huấn luyện trên Kaggle.

Gói nén sẽ chứa:
- stage2_matched/
- stage1_compat/ (bao gồm upstream/)
- src/
- fl_training/
- data/partitions_stage2_matched_v5/
- requirements-stage1.txt, requirements-stage2-matched.txt, pyproject.toml
- kaggle_stage2_matched_v5.ipynb
- docs/MATCHED_STAGE2_V5.md

Loại bỏ hoàn toàn: .venv, __pycache__, checkpoint cũ, log, cache tạm.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = ROOT / "matched_stage2_v5_kaggle.zip"

INCLUDE_DIRS = [
    "stage2_matched",
    "stage1_compat",
    "src",
    "fl_training",
    "data/partitions_stage2_matched_v5",
]

INCLUDE_FILES = [
    "requirements-stage1.txt",
    "requirements-stage2-matched.txt",
    "pyproject.toml",
    "kaggle_stage2_matched_v5.ipynb",
    "docs/MATCHED_STAGE2_V5.md",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".git",
    ".venv",
    ".pytest_cache",
    ".runtime",
    ".stage2_matched_runtime",
    ".stage2_scratch_runtime",
    "runs",
    "output",
    ".pt",
    ".pth",
]


def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in ("__pycache__", ".venv", ".pytest_cache", ".git", ".runtime", "runs", "output"):
            return True
    for pat in EXCLUDE_PATTERNS:
        if pat.startswith(".") and path.suffix == pat:
            return True
        if pat in path.name:
            return True
    return False


def build_package():
    print(f"Đang quét các tệp từ {ROOT}...")
    files_to_pack = []

    # Thêm các thư mục
    for d in INCLUDE_DIRS:
        dir_path = ROOT / d
        if not dir_path.is_dir():
            print(f"CẢNH BÁO: Thư mục không tồn tại: {dir_path}")
            continue
        for p in dir_path.rglob("*"):
            if p.is_file() and not should_exclude(p):
                files_to_pack.append(p)

    # Thêm các tệp đơn lẻ
    for f in INCLUDE_FILES:
        file_path = ROOT / f
        if file_path.is_file() and not should_exclude(file_path):
            files_to_pack.append(file_path)
        else:
            print(f"CẢNH BÁO: Tệp không tồn tại: {file_path}")

    files_to_pack = sorted(set(files_to_pack), key=lambda p: p.relative_to(ROOT).as_posix())
    print(f"Tìm thấy {len(files_to_pack)} tệp hợp lệ để đóng gói.")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    total_bytes = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files_to_pack:
            arcname = p.relative_to(ROOT).as_posix()
            zf.write(p, arcname)
            total_bytes += p.stat().st_size

    zip_size = OUTPUT_ZIP.stat().st_size
    hasher = hashlib.sha256()
    with open(OUTPUT_ZIP, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest()

    summary = {
        "output_file": str(OUTPUT_ZIP),
        "file_count": len(files_to_pack),
        "uncompressed_bytes": total_bytes,
        "uncompressed_mb": round(total_bytes / (1024 * 1024), 2),
        "zip_bytes": zip_size,
        "zip_mb": round(zip_size / (1024 * 1024), 2),
        "zip_sha256": zip_sha256,
    }

    print("\n--- KẾT QUẢ ĐÓNG GÓI CHO KAGGLE ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    build_package()

"""Script đóng gói phát hành Stage 2 Scratch FL v3 (FedAvg-only).

Gói nén fl_package_v3.zip và manifest release_manifest_v3.json bao gồm:
- Toàn bộ mã nguồn FedAvg-only: stage1_compat, stage2_matched, stage2_scratch, src, fl_training, tests
- Bộ phân hoạch sạch v3: data/partitions_stage2_scratch_v3
- Review và verification fixtures: data/duplicate_review.json, data/visually_verified_pairs.json, data/four_visually_verified_pairs.json
- Notebook chạy Kaggle: kaggle_stage2_scratch_v3.ipynb
- Cấu hình & phụ thuộc: pyproject.toml, requirements-stage1.txt, requirements-stage2-matched.txt

Loại trừ:
- .venv, __pycache__, .git, .pytest_cache, logs, runs, weights/checkpoints (.pth, .pt)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = ROOT / "fl_package_v3.zip"
OUTPUT_MANIFEST = ROOT / "release_manifest_v3.json"

INCLUDE_DIRS = [
    "stage1_compat",
    "stage2_matched",
    "stage2_scratch",
    "src",
    "fl_training",
    "tests",
    "data/partitions_stage2_scratch_v3",
]

INCLUDE_FILES = [
    "data/duplicate_review.json",
    "data/visually_verified_pairs.json",
    "data/four_visually_verified_pairs.json",
    "kaggle_stage2_scratch_v3.ipynb",
    "requirements-stage1.txt",
    "requirements-stage2-matched.txt",
    "pyproject.toml",
]

EXCLUDE_PARTS = {
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".git",
    ".runtime",
    ".stage2_matched_runtime",
    ".stage2_scratch_runtime",
    "runs",
    "output",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pt",
    ".pth",
    ".log",
}


def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_PARTS:
            return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_release_v3():
    print(f"[1/4] Scanning files to package from {ROOT}...")
    files_to_pack: list[Path] = []

    for d in INCLUDE_DIRS:
        dir_path = ROOT / d
        if not dir_path.is_dir():
            print(f"  WARNING: Directory not found: {dir_path}")
            continue
        for p in dir_path.rglob("*"):
            if p.is_file() and not should_exclude(p):
                files_to_pack.append(p)

    for f in INCLUDE_FILES:
        file_path = ROOT / f
        if file_path.is_file() and not should_exclude(file_path):
            files_to_pack.append(file_path)
        else:
            print(f"  WARNING: File not found: {file_path}")

    files_to_pack = sorted(set(files_to_pack), key=lambda p: p.relative_to(ROOT).as_posix())
    print(f"  Found {len(files_to_pack)} valid files to package.")

    # Generate file-level manifest
    print("[2/4] Computing SHA256 hashes for manifest...")
    manifest_entries = {}
    total_uncompressed = 0
    for p in files_to_pack:
        rel = p.relative_to(ROOT).as_posix()
        size = p.stat().st_size
        sha = compute_sha256(p)
        manifest_entries[rel] = {
            "size_bytes": size,
            "sha256": sha,
        }
        total_uncompressed += size

    manifest_data = {
        "protocol": "stage2_scratch_clean_v3",
        "scope": "fedavg_only",
        "total_files": len(files_to_pack),
        "uncompressed_bytes": total_uncompressed,
        "files": manifest_entries,
    }

    OUTPUT_MANIFEST.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Manifest written to {OUTPUT_MANIFEST}")

    # Build zip archive
    print(f"[3/4] Creating zip package {OUTPUT_ZIP}...")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Include manifest inside the zip as well
        zf.write(OUTPUT_MANIFEST, "release_manifest_v3.json")
        for p in files_to_pack:
            arcname = p.relative_to(ROOT).as_posix()
            zf.write(p, arcname)

    zip_size = OUTPUT_ZIP.stat().st_size
    zip_sha = compute_sha256(OUTPUT_ZIP)

    # Verification: extract to clean temp directory
    print("[4/4] Verifying zip integrity in clean temp directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
            zf.extractall(temp_path)

        # Check extracted files against manifest
        extracted_manifest = json.loads((temp_path / "release_manifest_v3.json").read_text(encoding="utf-8"))
        for rel_path, meta in extracted_manifest["files"].items():
            extracted_file = temp_path / rel_path
            assert extracted_file.is_file(), f"Missing extracted file: {rel_path}"
            assert extracted_file.stat().st_size == meta["size_bytes"], f"Size mismatch for {rel_path}"
            extracted_sha = compute_sha256(extracted_file)
            assert extracted_sha == meta["sha256"], f"Hash mismatch for {rel_path}: {extracted_sha} vs {meta['sha256']}"

    summary = {
        "output_zip": str(OUTPUT_ZIP),
        "output_manifest": str(OUTPUT_MANIFEST),
        "file_count": len(files_to_pack),
        "uncompressed_bytes": total_uncompressed,
        "uncompressed_mb": round(total_uncompressed / (1024 * 1024), 2),
        "zip_bytes": zip_size,
        "zip_mb": round(zip_size / (1024 * 1024), 2),
        "zip_sha256": zip_sha,
        "verification": "PASSED (all extracted files matched size & sha256)",
    }
    print("\n--- PACKAGE RELEASE V3 SUCCESSFUL ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    build_release_v3()

"""Đóng gói dữ liệu và mã để huấn luyện release v4 trên máy GPU khác (không cần repo, không cần thư mục data/ đầy đủ).

Tạo trong `--out` (mặc định tmp/train_bundle/):
  train_data_v4.tar   dataset_sources/mixed/pv_pd_v4/ (manifest, phân vùng, partition_spec, quality_audit...) + MỌI ảnh mà train/val/calibration/
                      legacy_diagnostic_test tham chiếu, giữ nguyên đường dẫn tương đối (giải nén ra thư mục X thì --dataset-root X/dataset_sources)
  train_code.tar      `git archive HEAD` của src/, scripts/train_v4.py, scripts/remote/, requirements-train.txt, kèm CODE_VERSION (mã commit) và
                      train_contract/ (bản chép 4 tệp kiểm của nhánh train để scripts/verify_release_contract.py chạy không cần git)
  bundle.json         SHA-256 hai tệp tar, SHA release, số ảnh, mã commit
Không đóng gói PlantWild hay ảnh không phải lá (chỉ dùng để đo, đo ở máy này sau khi tải checkpoint về).

Chạy (sau khi commit để mã trong tar đúng với HEAD):
    python scripts/pack_train_bundle.py
    python scripts/pack_train_bundle.py --code-only   # chỉ dựng lại train_code.tar khi dữ liệu không đổi (giữ train_data_v4.tar, kiểm lại SHA release)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

RELEASE_REL = "mixed/pv_pd_v4"
CODE_PATHS = ("src", "scripts/train_v4.py", "scripts/summarize_ablation.py", "scripts/remote", "scripts/verify_release_contract.py", "requirements-train.txt")
CONTRACT_BRANCH = "origin/train-centralized"
CONTRACT_FILES = ("plant_data_contract/plant_data_contract/integrity.py", "plant_data_contract/plant_data_contract/partitions.py",
                  "plant_data_contract/plant_data_contract/full_gate.py", "training-workflows/shared/data_selection.py")
SPLITS = ("train", "val", "calibration", "legacy_diagnostic_test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "tmp" / "train_bundle")
    parser.add_argument("--code-only", action="store_true", help="giữ train_data_v4.tar đã có (release không đổi), chỉ dựng lại train_code.tar")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=ROOT).decode().strip()
    if dirty:
        print(f"[pack] mã chưa commit, tar sẽ không khớp HEAD:\n{dirty}", file=sys.stderr)
        return 1
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    code_tar = args.out / "train_code.tar"
    raw = subprocess.check_output(["git", "archive", "--format=tar", "HEAD", *CODE_PATHS], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as source, tarfile.open(code_tar, "w") as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        info = tarfile.TarInfo("CODE_VERSION")
        info.size = len(commit) + 1
        target.addfile(info, io.BytesIO((commit + "\n").encode()))
        for rel in CONTRACT_FILES:
            blob = subprocess.check_output(["git", "show", f"{CONTRACT_BRANCH}:{rel}"], cwd=ROOT)
            entry = tarfile.TarInfo(f"train_contract/{Path(rel).name}")
            entry.size = len(blob)
            target.addfile(entry, io.BytesIO(blob))

    root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    release = root / RELEASE_REL
    release_sha = sha256_file(release / "release_manifest.json")
    if args.code_only:
        previous = json.loads((args.out / "bundle.json").read_text(encoding="utf-8"))
        if previous["release_manifest_sha256"] != release_sha:
            print(f"[pack] release đã đổi ({previous['release_manifest_sha256'][:12]} -> {release_sha[:12]}): phải đóng gói lại cả dữ liệu", file=sys.stderr)
            return 1
        data_tar = args.out / "train_data_v4.tar"
        if sha256_file(data_tar) != previous["train_data_v4.tar"]["sha256"]:
            print("[pack] train_data_v4.tar khác SHA trong bundle.json: phải đóng gói lại cả dữ liệu", file=sys.stderr)
            return 1
        bundle = {**previous, "git_commit": commit, "data_from_commit": previous.get("data_from_commit", previous["git_commit"]),
                  "train_code.tar": {"sha256": sha256_file(code_tar), "bytes": code_tar.stat().st_size}}
        (args.out / "bundle.json").write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
        print(f"[pack] chỉ mã: train_code.tar {bundle['train_code.tar']['sha256'][:12]}; dữ liệu giữ nguyên; commit {commit[:12]}")
        return 0
    images: set[str] = set()
    for split in SPLITS:
        with (release / "manifests" / f"{split}.jsonl").open(encoding="utf-8") as stream:
            images.update(json.loads(line)["relative_path"] for line in stream if line.strip())
    data_tar = args.out / "train_data_v4.tar"
    with tarfile.open(data_tar, "w") as tar:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=f"dataset_sources/{path.relative_to(root).as_posix()}", recursive=False)
        for i, rel in enumerate(sorted(images), start=1):
            tar.add(root / rel, arcname=f"dataset_sources/{rel}", recursive=False)
            if i % 10000 == 0:
                print(f"[pack] {i}/{len(images)} ảnh", flush=True)
    bundle = {"git_commit": commit, "release_manifest_sha256": release_sha, "images": len(images),
              "train_data_v4.tar": {"sha256": sha256_file(data_tar), "bytes": data_tar.stat().st_size},
              "train_code.tar": {"sha256": sha256_file(code_tar), "bytes": code_tar.stat().st_size}}
    (args.out / "bundle.json").write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    print(f"[pack] {len(images)} ảnh, {bundle['train_data_v4.tar']['bytes'] / 1e9:.2f} GB -> {args.out}; SHA release {release_sha}; commit {commit[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

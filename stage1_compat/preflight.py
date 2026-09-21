"""Preflight check và đóng băng manifest tương thích cho profile stage1_compat."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from stage1_compat.integrity import atomic_json, digest, file_hash, read_json, validate_indices
from stage1_compat.constants import DATASET_IDENTITY_PATH, VAL_SAMPLES, TEST_SAMPLES
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torchvision import datasets

from stage1_compat.baselines import BaselineRegistry
from stage1_compat.constants import (
    DEFAULT_ALPHAS,
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_OUTPUT_DIR,
    NUM_CLASSES,
    SEED,
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    TOTAL_DATASET_SAMPLES,
    TRAIN_SAMPLES,
    UPSTREAM_COMMIT,
    UPSTREAM_DIR,
    UPSTREAM_MANIFEST_PATH,
    UPSTREAM_REPO,
)


@dataclass
class PreflightResult:
    status: str  # "PASSED", "WARNING", "FAILED"
    upstream_manifest_ok: bool
    upstream_files_checked: int
    upstream_files_matched: int
    class_order_ok: bool
    total_classes: int
    split_integrity_ok: bool
    partitions_integrity_ok: bool
    baseline_registry_ok: bool
    baselines_available: list[str]
    missing_baselines: list[str]
    frozen_manifest_path: str
    report_path: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_upstream_manifest(manifest_path: Path = UPSTREAM_MANIFEST_PATH) -> tuple[bool, int, int, list[str]]:
    if not manifest_path.exists():
        return False, 0, 0, [f"Không tìm thấy manifest: {manifest_path}"]

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_dict = payload.get("files", {})
    if payload.get("commit") != UPSTREAM_COMMIT or len(files_dict) != 30:
        return False, len(files_dict), 0, ["Invalid upstream manifest identity"]
    total = len(files_dict)
    matched = 0
    errors = []

    for rel_path, expected_hash in files_dict.items():
        target = (UPSTREAM_DIR / rel_path).resolve()
        if not target.is_relative_to(UPSTREAM_DIR.resolve()):
            raise ValueError("Upstream path escapes snapshot")
        if not target.exists():
            errors.append(f"File thiếu: {rel_path}")
            continue
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            matched += 1
        else:
            errors.append(f"Sai hash: {rel_path} (kỳ vọng {expected_hash[:8]}..., thực tế {actual_hash[:8]}...)")

    return (matched == total), total, matched, errors


def check_class_order(data_dir: Path) -> tuple[bool, int, list[str], list[str]]:
    inspection_file = UPSTREAM_DIR / "experiments" / "results" / "dataset_inspection.json"
    if not inspection_file.exists():
        return False, 0, [], [f"Không tìm thấy {inspection_file}"]

    inspection_data = json.loads(inspection_file.read_text(encoding="utf-8"))
    expected_classes = sorted(inspection_data.get("classes", []))

    if not data_dir.exists():
        return False, len(expected_classes), expected_classes, [f"Không tìm thấy thư mục dataset: {data_dir}"]

    dataset = datasets.ImageFolder(str(data_dir))
    actual_classes = dataset.classes

    if actual_classes == expected_classes:
        return True, len(actual_classes), actual_classes, []
    else:
        diffs = [
            f"Lớp {i}: kỳ vọng '{exp}', thực tế '{act}'"
            for i, (exp, act) in enumerate(zip(expected_classes, actual_classes))
            if exp != act
        ]
        return False, len(actual_classes), actual_classes, diffs


def check_split_and_partitions(seed: int = SEED) -> tuple[bool, bool, dict[str, Any], list[str]]:
    errors = []
    # 1. Split
    split_file = UPSTREAM_DIR / "experiments" / "splits" / f"plantvillage_train0.72_val0.08_seed{seed}.json"
    split_ok = False
    split_info = {}
    if not split_file.exists():
        errors.append(f"Thiếu file split: {split_file}")
    else:
        sp = json.loads(split_file.read_text(encoding="utf-8"))
        train_idx = sp.get("train_indices", [])
        val_idx = sp.get("validation_indices", [])
        test_idx = sp.get("test_indices", [])
        total = sp.get("total_samples", 0)

        validate_indices([train_idx, val_idx, test_idx], TOTAL_DATASET_SAMPLES)
        if [len(train_idx), len(val_idx), len(test_idx)] != [TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES]:
            raise ValueError("Split size mismatch")
        all_idx = train_idx + val_idx + test_idx
        if total == TOTAL_DATASET_SAMPLES and len(all_idx) == TOTAL_DATASET_SAMPLES and len(set(all_idx)) == TOTAL_DATASET_SAMPLES:
            split_ok = True
            split_info = {
                "total": total,
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
                "ratios": f"{sp.get('train_ratio')}/{sp.get('validation_ratio')}/{sp.get('test_ratio')}",
            }
        else:
            errors.append(f"File split không đạt tính toàn vẹn: total={total}, all_idx={len(all_idx)}, unique={len(set(all_idx))}")

    # 2. Partitions
    partitions_ok = True
    part_info = {}
    for alpha in DEFAULT_ALPHAS:
        alpha_label = f"{int(alpha)}" if float(alpha).is_integer() else f"{alpha:g}".replace(".", "_")
        part_file = UPSTREAM_DIR / "experiments" / "partitions" / f"plantvillage_alpha{alpha_label}_clients5_seed{seed}.json"
        if not part_file.exists():
            errors.append(f"Thiếu file partition: {part_file}")
            partitions_ok = False
            continue
        p = json.loads(part_file.read_text(encoding="utf-8"))
        partitions = p.get("partitions", [])
        total_p = p.get("total_samples", 0)
        validate_indices(partitions, TRAIN_SAMPLES)
        flattened = [idx for client_indices in partitions for idx in client_indices]

        if total_p == TRAIN_SAMPLES and len(flattened) == TRAIN_SAMPLES and len(set(flattened)) == TRAIN_SAMPLES and len(partitions) == DEFAULT_NUM_CLIENTS:
            part_info[f"alpha_{alpha}"] = {
                "clients": len(partitions),
                "samples_per_client": [len(c) for c in partitions],
                "total": total_p,
            }
        else:
            partitions_ok = False
            errors.append(f"Partition alpha={alpha} không hợp lệ: total={total_p}, flat={len(flattened)}, unique={len(set(flattened))}")

    return split_ok, partitions_ok, {"split": split_info, "partitions": part_info}, errors


def create_frozen_manifest(
    data_dir: Path,
    output_dir: Path,
    class_names: list[str],
    split_info: dict[str, Any],
) -> Path:
    """Tạo manifest đóng băng dataset và split mới tại output/stage1_compat, không sửa manifest cũ."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.ImageFolder(str(data_dir))
    expected = read_json(DATASET_IDENTITY_PATH)["images"]
    def checked_sample(item):
        name, label = item
        relative = Path(name).relative_to(data_dir).as_posix()
        sha = file_hash(name)
        if expected.get(relative) != sha:
            raise ValueError(f"Dataset content mismatch: {relative}")
        return [relative, label, sha]
    samples = []
    # Bound concurrency; map preserves ImageFolder order while overlapping small-file reads.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(checked_sample, dataset.samples):
            samples.append(row)
            if len(samples) % 5000 == 0:
                print(f"Dataset hash verification: {len(samples)}/{len(dataset.samples)}", flush=True)
    if len(samples) != len(expected):
        raise ValueError("Dataset inventory mismatch")
    from stage1_compat.data import load_stage1_datasets, load_stage1_partition, get_split_file, get_partition_file
    train, _, _, _ = load_stage1_datasets(data_dir)
    diagnostics = {}
    for alpha in DEFAULT_ALPHAS:
        partitions = load_stage1_partition(alpha, train.targets)
        diagnostics[str(alpha)] = {"samples": [len(ids) for ids in partitions],
                                  "partition_sha256": file_hash(get_partition_file(alpha))}
    payload = {"schema": 2, "classes": class_names, "samples": samples,
               "split_sha256": file_hash(get_split_file()), "partitions": diagnostics,
               "upstream_commit": UPSTREAM_COMMIT,
               "historical_dataset_equivalence_verified": False}
    payload["identity_hash"] = digest(payload)
    manifest_target = output_dir / "frozen_dataset_manifest.v2.json"
    if manifest_target.exists():
        if read_json(manifest_target) != payload:
            raise ValueError("Frozen dataset manifest mismatch; preserve old output")
    else:
        atomic_json(manifest_target, payload)
    return manifest_target


def run_preflight(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> PreflightResult:
    resolved_data = Path(data_dir).resolve() if data_dir else DEFAULT_DATA_DIR.resolve()
    resolved_output = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    # 1. Upstream manifest
    manifest_ok, total_f, matched_f, manifest_errs = check_upstream_manifest()

    # 2. Class order
    class_ok, total_classes, classes, class_errs = check_class_order(resolved_data)

    # 3. Split and Partitions
    split_ok, part_ok, split_details, split_errs = check_split_and_partitions()

    # 4. Baseline registry
    registry = BaselineRegistry()
    available_baselines = list(registry.list_all().keys())
    missing_baselines = []
    # FedAvg alpha 100 historical is missing by design
    if "fedavg_historical_alpha100_seed42" not in available_baselines:
        missing_baselines.append("fedavg_historical_alpha100_seed42 (chưa từng chạy trong GĐ1, đánh dấu MISSING chuẩn)")
    # Other seeds
    for s in [123, 2026]:
        missing_baselines.append(f"baselines_seed{s} (không có trong lịch sử GĐ1, đánh dấu MISSING)")

    # 5. Create frozen manifest
    frozen_manifest = resolved_output / "frozen_dataset_manifest.v2.json"
    if manifest_ok and class_ok and split_ok and part_ok:
        frozen_manifest = create_frozen_manifest(resolved_data, resolved_output, classes, split_details)

    # 6. Compatibility report
    all_ok = manifest_ok and class_ok and split_ok and part_ok
    status = "PASSED" if all_ok else "FAILED"

    report_payload = {
        "status": status,
        "profile": "stage1_compat",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_manifest": {
            "valid": manifest_ok,
            "checked": total_f,
            "matched": matched_f,
            "errors": manifest_errs,
        },
        "class_mapping": {
            "valid": class_ok,
            "total_classes": total_classes,
            "errors": class_errs,
        },
        "splits_and_partitions": {
            "split_valid": split_ok,
            "partitions_valid": part_ok,
            "details": split_details,
            "errors": split_errs,
        },
        "baseline_registry": {
            "available": available_baselines,
            "missing": missing_baselines,
            "reference_only": True,
            "strict_comparison_eligible": False,
            "selection_policy_note": "Centralized/FedAvg chọn best val accuracy; Local-only dùng final epoch.",
            "data_provenance_note": "GĐ1 thiếu fingerprint SHA256 ảnh lịch sử và checkpoint client Local-only.",
        },
        "protocol_locks": {
            "model": "MobileNetV3-Small (38 classes)",
            "optimizer": "AdamW (lr=0.001, wd=0.0001)",
            "clients": 5,
            "rounds": 10,
            "batch_size": 32,
            "backend": "sequential",
        },
        "scientific_stage2_complete": False,
    }

    report_path = resolved_output / "stage1_compatibility_report.json"
    atomic_json(report_path, report_payload)

    return PreflightResult(
        status=status,
        upstream_manifest_ok=manifest_ok,
        upstream_files_checked=total_f,
        upstream_files_matched=matched_f,
        class_order_ok=class_ok,
        total_classes=total_classes,
        split_integrity_ok=split_ok,
        partitions_integrity_ok=part_ok,
        baseline_registry_ok=len(available_baselines) > 0,
        baselines_available=available_baselines,
        missing_baselines=missing_baselines,
        frozen_manifest_path=str(frozen_manifest),
        report_path=str(report_path),
        details=report_payload,
    )

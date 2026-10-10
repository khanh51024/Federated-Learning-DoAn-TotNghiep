"""Unit tests cho Gói 1: Preflight, protocol, baseline registry và dry-run."""

import json
from pathlib import Path
import pytest

from stage1_compat.baselines import BaselineRegistry
from stage1_compat.constants import (
    DEFAULT_ALPHAS,
    NUM_CLASSES,
    SEED,
    TOTAL_DATASET_SAMPLES,
    TRAIN_SAMPLES,
    UPSTREAM_DIR,
    UPSTREAM_MANIFEST_PATH,
)
from stage1_compat.dry_run import run_dry_run
from stage1_compat.identity import generate_job_identity, verify_job_identity
from stage1_compat.config import JobConfig
from stage1_compat.preflight import (
    check_class_order,
    check_split_and_partitions,
    check_upstream_manifest,
    run_preflight,
)


def test_upstream_manifest_matches_30_files():
    ok, total, matched, errors = check_upstream_manifest(UPSTREAM_MANIFEST_PATH)
    assert ok is True
    assert total == 30
    assert matched == 30
    assert len(errors) == 0


def test_class_mapping_matches_dataset_inspection():
    inspection_file = UPSTREAM_DIR / "experiments" / "results" / "dataset_inspection.json"
    assert inspection_file.exists()
    payload = json.loads(inspection_file.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    assert len(classes) == NUM_CLASSES
    # Test on actual dataset folder
    data_dir = Path("../PlantVillage-Dataset/raw/color")
    if data_dir.exists():
        ok, count, actual_classes, errors = check_class_order(data_dir)
        assert ok is True
        assert count == NUM_CLASSES
        assert actual_classes == classes


def test_split_and_partitions_integrity():
    split_ok, part_ok, details, errors = check_split_and_partitions(seed=SEED)
    assert split_ok is True
    assert part_ok is True
    assert len(errors) == 0
    assert details["split"]["total"] == TOTAL_DATASET_SAMPLES
    assert details["split"]["train"] == TRAIN_SAMPLES
    for alpha in DEFAULT_ALPHAS:
        assert f"alpha_{alpha}" in details["partitions"]
        assert details["partitions"][f"alpha_{alpha}"]["clients"] == 5
        assert details["partitions"][f"alpha_{alpha}"]["total"] == TRAIN_SAMPLES


def test_baseline_registry_integrity_and_missing_handling():
    registry = BaselineRegistry()

    # Centralized
    cent = registry.get_centralized(SEED)
    assert cent is not None
    assert cent.method == "centralized"
    assert cent.selection_policy == "best_validation_accuracy"
    assert cent.reference_only is True
    assert cent.strict_comparison_eligible is False
    assert 0.0 < cent.metrics["test_accuracy"] <= 1.0

    # Local-only
    loc1 = registry.get_local_only(1.0, SEED)
    assert loc1 is not None
    assert loc1.method == "local_only"
    assert loc1.selection_policy == "final_epoch"
    assert loc1.client_count == 5
    assert loc1.reference_only is True
    assert loc1.strict_comparison_eligible is False

    loc100 = registry.get_local_only(100.0, SEED)
    assert loc100 is not None
    assert loc100.selection_policy == "final_epoch"

    # Missing baselines must return None, NEVER fabricate
    assert registry.get_historical_fedavg(100.0, SEED) is None  # Never run in GĐ1
    assert registry.get_centralized(123) is None  # Nonexistent seed
    assert registry.get_local_only(1.0, 123) is None


def test_identity_generation_and_verification():
    job = JobConfig(job_id="test_fedavg", alpha=1.0, seed=42)
    identity = generate_job_identity(job)
    assert identity.identity_hash is not None
    assert len(identity.identity_hash) == 64
    assert verify_job_identity(identity, identity.to_dict()) is True

    # Mutated candidate must fail verification
    mutated = identity.to_dict()
    mutated["alpha"] = 999.0
    assert verify_job_identity(identity, mutated) is False


def test_dry_run_executes_without_training_loop(tmp_path):
    data_dir = Path("../PlantVillage-Dataset/raw/color")
    res = run_dry_run(data_dir=data_dir, output_dir=tmp_path, quota_hours=10.0)
    assert res.status == "SUCCESS"
    assert res.preflight_passed is True
    assert len(res.jobs_planned) == 2
    assert (tmp_path / "stage1_dry_run_report.json").exists()

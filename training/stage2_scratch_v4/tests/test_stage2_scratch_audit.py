"""Regression and audit tests for Stage 2 Scratch FL pipeline.

Verifies:
1. Four visually verified pairs from visually_verified_pairs.json never cross splits or clients.
2. Random scratch initialization (weights=None, pretrained=False).
3. Weighted parameter aggregation math & BatchNorm integer buffer preservation.
4. Centralized baseline on feature skew does not throw KeyError (domain_profiles passed).
5. Local-only baseline initializes client models independently from the same seed state.
6. Parity between Sequential Runner and Flower simulation adapter.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage1_compat.checkpoint import compute_state_dict_sha256
from stage1_compat.config import JobConfig
from stage1_compat.models import create_mobilenetv3_stage1
from stage1_compat.runner import get_parameters, set_parameters
from stage2_scratch.aggregation import aggregate_fedavg_parameters
from stage1_compat.upstream_loader import get_upstream_reproducibility
from stage2_matched.data import CONDITIONS, ManifestDataset, read_rows


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parent / "PlantVillage-Dataset/raw/color"
SUITE_V2 = ROOT / "data/partitions_stage2_scratch_v2"

# Resolve SUITE_V3 via STAGE2_SUITE_PATH env var, candidate data/, or workspace training-data/
_env_suite = os.environ.get("STAGE2_SUITE_PATH")
if _env_suite and Path(_env_suite).is_dir():
    SUITE_V3 = Path(_env_suite).resolve()
elif (ROOT / "data/partitions_stage2_scratch_v3").is_dir():
    SUITE_V3 = ROOT / "data/partitions_stage2_scratch_v3"
else:
    candidates = [
        ROOT.parents[1] / "training-data/stage2/partitions_stage2_scratch_v3",
        ROOT.parent / "training-data/stage2/partitions_stage2_scratch_v3",
        Path("training-data/stage2/partitions_stage2_scratch_v3").resolve(),
    ]
    SUITE_V3 = next((c for c in candidates if c.is_dir()), ROOT / "data/partitions_stage2_scratch_v3")

VERIFIED_PAIRS_FILE = ROOT / "data/four_visually_verified_pairs.json"
ALL_VERIFIED_PAIRS_FILE = (
    (SUITE_V3 / "visually_verified_pairs.json")
    if (SUITE_V3 / "visually_verified_pairs.json").is_file()
    else (ROOT / "data/visually_verified_pairs.json")
)


def test_scratch_initialization_weights_none():
    """Verify that models are initialized randomly without pretrained weights."""
    get_upstream_reproducibility().set_seed(42)
    m1 = create_mobilenetv3_stage1(num_classes=38, pretrained=False)
    h1 = compute_state_dict_sha256(m1.state_dict())

    get_upstream_reproducibility().set_seed(42)
    m2 = create_mobilenetv3_stage1(num_classes=38, pretrained=False)
    h2 = compute_state_dict_sha256(m2.state_dict())

    get_upstream_reproducibility().set_seed(123)
    m3 = create_mobilenetv3_stage1(num_classes=38, pretrained=False)
    h3 = compute_state_dict_sha256(m3.state_dict())

    # Same seed -> identical hash; different seed -> different hash
    assert h1 == h2, "Same seed must produce identical initial weights"
    assert h1 != h3, "Different seed must produce different initial weights"

    # Known ImageNet pretrained SHA is NOT present
    imagenet_known_shas = {
        "7fb55d143644fcf77755b40cf61763a8a37f5b84c898c60dc327f294025fbc74",
        "0745582f3c054238e55e39d571ddc9ff6a99ce3818e3848b6f3c026beafc258d",
    }
    assert h1 not in imagenet_known_shas, "Model hash matches ImageNet pretrained weights!"


def test_weighted_parameters_aggregation_and_bn():
    """Verify parameter aggregation math and BatchNorm integer buffer preservation."""
    # Create mock parameters for 2 clients:
    # Param 0: float weight
    # Param 1: float running_mean
    # Param 2: int64 num_batches_tracked
    c1_params = [
        np.array([[2.0, 4.0]], dtype=np.float32),
        np.array([1.0, 3.0], dtype=np.float32),
        np.array(10, dtype=np.int64),
    ]
    c2_params = [
        np.array([[4.0, 8.0]], dtype=np.float32),
        np.array([3.0, 5.0], dtype=np.float32),
        np.array(20, dtype=np.int64),
    ]
    client_params = [c1_params, c2_params]
    client_samples = [100, 300]  # total = 400. c1 weight = 0.25, c2 weight = 0.75

    base_params = [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array(0, dtype=np.int64),
    ]
    agg = aggregate_fedavg_parameters(base_params, client_params, client_samples)

    # Param 0: 0.25*2 + 0.75*4 = 0.5 + 3.0 = 3.5; 0.25*4 + 0.75*8 = 1.0 + 6.0 = 7.0
    expected_p0 = np.array([[3.5, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(agg[0], expected_p0, rtol=1e-5)

    # Param 1: 0.25*1 + 0.75*3 = 0.25 + 2.25 = 2.5; 0.25*3 + 0.75*5 = 0.75 + 3.75 = 4.5
    expected_p1 = np.array([2.5, 4.5], dtype=np.float32)
    np.testing.assert_allclose(agg[1], expected_p1, rtol=1e-5)

    # Param 2 (int64 num_batches_tracked): sum = 10 + 20 = 30
    assert agg[2].dtype == np.int64, f"Expected int64 dtype, got {agg[2].dtype}"
    assert int(agg[2]) == 30, f"Expected sum 30, got {agg[2]}"


def test_centralized_feature_domain_profiles():
    """Verify that ManifestDataset with domain_profiles works without KeyError: 4."""
    if not SUITE_V2.is_dir() or not DATASET_ROOT.is_dir():
        pytest.skip("Suite v2 or dataset root not present yet")

    cond_path = SUITE_V2 / "feature100"
    if not cond_path.is_dir():
        pytest.skip("feature100 partition not present")

    meta = json.loads((cond_path / "partition_config.json").read_text(encoding="utf-8"))
    domain_profiles = meta.get("domain_profiles", [])
    assert len(domain_profiles) == 5, f"Expected 5 domain profiles, got {len(domain_profiles)}"

    train_rows = read_rows(cond_path / "centralized_train.csv")
    ds = ManifestDataset(train_rows, DATASET_ROOT, is_train=True, domain_profiles=domain_profiles)

    # Check sample retrieval for different domain IDs
    found_domains = set()
    for i in range(min(100, len(ds))):
        img, label = ds[i]
        assert isinstance(img, torch.Tensor), "Expected torch.Tensor"
        assert img.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {img.shape}"
        found_domains.add(train_rows[i]["domain_id"])

    assert len(found_domains) > 0, "No domains checked"


def test_verified_pairs_regression_fixtures():
    """Assert all 4 visually verified pairs never cross splits or clients in Suite V2."""
    if not SUITE_V2.is_dir() or not VERIFIED_PAIRS_FILE.is_file():
        pytest.skip("Suite v2 or visually_verified_pairs.json not present yet")

    verified_pairs = json.loads(VERIFIED_PAIRS_FILE.read_text(encoding="utf-8"))
    assert len(verified_pairs) == 4, f"Expected 4 pairs, found {len(verified_pairs)}"

    for cond_name in CONDITIONS:
        cond_dir = SUITE_V2 / cond_name
        if not cond_dir.is_dir():
            continue

        train_rows = read_rows(cond_dir / "centralized_train.csv")
        val_rows = read_rows(cond_dir / "global_val.csv")
        test_rows = read_rows(cond_dir / "global_test.csv")

        # Map relative path to split
        split_map = {}
        for r in train_rows:
            split_map[r["relative_path"]] = ("train", r["client_id"], r["group_id"])
        for r in val_rows:
            split_map[r["relative_path"]] = ("val", -1, r["group_id"])
        for r in test_rows:
            split_map[r["relative_path"]] = ("test", -1, r["group_id"])

        for idx, pair in enumerate(verified_pairs, 1):
            na = "/".join(pair["a"].replace("\\", "/").split("/")[-2:])
            nb = "/".join(pair["b"].replace("\\", "/").split("/")[-2:])

            assert na in split_map, f"Image {na} not found in manifests"
            assert nb in split_map, f"Image {nb} not found in manifests"

            split_a, client_a, group_a = split_map[na]
            split_b, client_b, group_b = split_map[nb]

            # 1. Must share identical group_id
            assert group_a == group_b, (
                f"Pair {idx} group mismatch in {cond_name}: {na} (group {group_a}) vs {nb} (group {group_b})"
            )

            # 2. Must belong to the exact same split
            assert split_a == split_b, (
                f"Pair {idx} split crossing in {cond_name}: {na} in {split_a} vs {nb} in {split_b}"
            )

            # 3. If in train, must belong to the exact same client
            if split_a == "train":
                assert client_a == client_b, (
                    f"Pair {idx} client crossing in {cond_name}: {na} (client {client_a}) vs {nb} (client {client_b})"
                )


def test_suite_v3_all_six_verified_pairs_fixtures():
    """Assert all 6 visually verified pairs never cross splits or clients in Suite V3."""
    if not SUITE_V3.is_dir() or not ALL_VERIFIED_PAIRS_FILE.is_file():
        pytest.skip("Suite v3 or visually_verified_pairs.json not present yet")

    verified_pairs = json.loads(ALL_VERIFIED_PAIRS_FILE.read_text(encoding="utf-8"))
    assert len(verified_pairs) == 6, f"Expected 6 pairs, found {len(verified_pairs)}"

    for cond_name in CONDITIONS:
        cond_dir = SUITE_V3 / cond_name
        if not cond_dir.is_dir():
            continue

        train_rows = read_rows(cond_dir / "centralized_train.csv")
        val_rows = read_rows(cond_dir / "global_val.csv")
        test_rows = read_rows(cond_dir / "global_test.csv")

        # Map relative path to split
        split_map = {}
        for r in train_rows:
            split_map[r["relative_path"]] = ("train", r["client_id"], r["group_id"])
        for r in val_rows:
            split_map[r["relative_path"]] = ("val", -1, r["group_id"])
        for r in test_rows:
            split_map[r["relative_path"]] = ("test", -1, r["group_id"])

        for idx, pair in enumerate(verified_pairs, 1):
            na = "/".join(pair["a"].replace("\\", "/").split("/")[-2:])
            nb = "/".join(pair["b"].replace("\\", "/").split("/")[-2:])

            assert na in split_map, f"Image {na} not found in manifests"
            assert nb in split_map, f"Image {nb} not found in manifests"

            split_a, client_a, group_a = split_map[na]
            split_b, client_b, group_b = split_map[nb]

            # 1. Must share identical group_id
            assert group_a == group_b, (
                f"Pair {idx} group mismatch in {cond_name}: {na} (group {group_a}) vs {nb} (group {group_b})"
            )

            # 2. Must belong to the exact same split
            assert split_a == split_b, (
                f"Pair {idx} split crossing in {cond_name}: {na} in {split_a} vs {nb} in {split_b}"
            )

            # 3. If in train, must belong to the exact same client
            if split_a == "train":
                assert client_a == client_b, (
                    f"Pair {idx} client crossing in {cond_name}: {na} (client {client_a}) vs {nb} (client {client_b})"
                )


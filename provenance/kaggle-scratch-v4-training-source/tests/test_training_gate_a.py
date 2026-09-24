"""
Tests for Gate A: Environment, Data Helpers, Validation Split, and Partition Audits.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gate_a_dependencies_import():
    import torch
    import torchvision
    import flwr
    import ray

    assert torch.__version__ is not None
    assert torchvision.__version__ is not None
    assert flwr.__version__ is not None
    assert ray.__version__ is not None


def test_gate_a_data_helpers_regression(tmp_path):
    from src.data.dataset import load_manifest, quick_stats

    csv_file = tmp_path / "temp_manifest.csv"
    csv_file.write_text(
        "relative_path,label,class_name,client_id,split,group_id\n"
        "Apple___Apple_scab/img1.JPG,0,Apple___Apple_scab,0,train,leaf_1\n"
        "Apple___Black_rot/img2.JPG,1,Apple___Black_rot,0,train,leaf_2\n",
        encoding="utf-8",
    )

    records = load_manifest(csv_file)
    assert len(records) == 2
    assert records[0]["label"] == 0
    assert records[1]["label"] == 1

    stats = quick_stats(csv_file, num_classes=2)
    assert stats["images"] == 2
    assert stats["classes_present"] == 2
    assert stats["leaf_groups"] == 2


def test_gate_a_partitions_train_v1_index_and_audits():
    index_file = ROOT / "data" / "partitions_train_v1" / "index.json"
    assert index_file.exists(), "index.json must exist in data/partitions_train_v1"

    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reference test hash check
    assert data["test_paths_sha256"] == "340198e0b2f8945a1f2730b211aacd9ddcbd5b9c46404249771c69a77637e79e"
    assert data["test_samples"] == 10917
    assert data["val_samples"] == 5501
    assert data["train_pool_samples"] == 37887
    assert data["total_source_images"] == 54305

    # Val hash must be present and shared
    val_hash = data["val_paths_sha256"]
    assert len(val_hash) == 64

    # Both iid and label_skew must be present
    partitions = data["partitions"]
    assert len(partitions) >= 2

    for cfg_hash, part in partitions.items():
        assert part["audit"]["audited_ok"] is True
        assert part["audit"]["val_classes_present"] == 38
        assert part["train_samples"] == 37887
        assert part["val_samples"] == 5501
        assert part["test_samples"] == 10917
        assert sum(part["n_k"]) == 37887
        assert all(n >= 10 for n in part["n_k"])

        part_dir = ROOT / part["relative_dir"]
        assert (part_dir / "global_test.csv").exists()
        assert (part_dir / "global_val.csv").exists()
        assert (part_dir / "centralized_train.csv").exists()
        assert (part_dir / "fedavg_meta.json").exists()
        assert (part_dir / "partition_config.json").exists()


def test_gate_a_partitions_v3_not_overwritten():
    v3_dir = ROOT / "data" / "partitions_v3"
    assert v3_dir.exists()
    label_skew_v3 = v3_dir / "label_skew" / "label_skew__seed_42__alpha_0_1__feature_none"
    assert label_skew_v3.exists()

    with open(label_skew_v3 / "partition_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["val_samples"] == 0, "partitions_v3 must remain val_ratio=0 unchanged"
    assert cfg["test_samples"] == 10917

"""Strict preflight regression tests verifying real disk checks, tamper detection and group/split/client isolation."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from stage2_matched.data import CONDITIONS, preflight
from fl_training.content_audit import audit_image_content
from stage1_compat.integrity import file_hash

ROOT = Path(__file__).resolve().parents[1]
SUITE_V3 = ROOT / "data/partitions_stage2_scratch_v3"


@pytest.fixture
def minimal_suite_env(tmp_path):
    """Create a minimal valid mock suite and dataset directory for negative testing."""
    suite_dir = tmp_path / "mock_suite"
    suite_dir.mkdir()
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir()

    class_names = [f"Class_{i:02d}" for i in range(38)]
    # Create 38 dummy classes and 1 dummy image per class
    images = {}
    for c in class_names:
        c_dir = dataset_dir / c
        c_dir.mkdir(parents=True, exist_ok=True)
        img_file = c_dir / "img_01.jpg"
        content = f"content_of_{c}".encode("utf-8")
        img_file.write_bytes(content)
        rel_path = f"{c}/img_01.jpg"
        images[rel_path] = hashlib.sha256(content).hexdigest()

    # Visual verified pairs (at least 6 pairs)
    # We will define 6 pairs within the 38 images
    visual_pairs = [
        {"a": f"{class_names[0]}/img_01.jpg", "b": f"{class_names[0]}/img_01.jpg"},
        {"a": f"{class_names[1]}/img_01.jpg", "b": f"{class_names[1]}/img_01.jpg"},
        {"a": f"{class_names[2]}/img_01.jpg", "b": f"{class_names[2]}/img_01.jpg"},
        {"a": f"{class_names[3]}/img_01.jpg", "b": f"{class_names[3]}/img_01.jpg"},
        {"a": f"{class_names[4]}/img_01.jpg", "b": f"{class_names[4]}/img_01.jpg"},
        {"a": f"{class_names[5]}/img_01.jpg", "b": f"{class_names[5]}/img_01.jpg"},
    ]
    (suite_dir / "visually_verified_pairs.json").write_text(json.dumps(visual_pairs), encoding="utf-8")

    review = [
        {"a": f"{class_names[0]}/img_01.jpg", "b": f"{class_names[0]}/img_01.jpg", "decision": "accepted"},
    ]
    (suite_dir / "duplicate_review.json").write_text(json.dumps(review), encoding="utf-8")

    return suite_dir, dataset_dir


def test_preflight_fails_when_dataset_root_does_not_exist():
    """Verify that preflight fails immediately if dataset_root does not exist."""
    non_existent = ROOT / "non_existent_dataset_directory_12345"
    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        preflight(SUITE_V3, dataset_root=non_existent)


def test_preflight_fails_when_dataset_root_is_a_file(tmp_path):
    """Verify that preflight fails if dataset_root is a file instead of a directory."""
    file_root = tmp_path / "not_a_dir.txt"
    file_root.write_text("hello")
    with pytest.raises(FileNotFoundError, match="Dataset root does not exist or is not a directory"):
        preflight(SUITE_V3, dataset_root=file_root)


def test_preflight_fails_when_image_missing_from_disk(tmp_path):
    """Verify that preflight fails if an expected image file is missing on disk."""
    mock_dataset = tmp_path / "mock_dataset"
    mock_dataset.mkdir()
    # Create empty mock dataset dir - image files are missing
    with pytest.raises(FileNotFoundError, match="Image file not found on disk"):
        preflight(SUITE_V3, dataset_root=mock_dataset)


def test_audit_image_content_fails_when_bytes_modified_same_size(tmp_path):
    """Verify that audit_image_content detects byte modification even if size is identical."""
    partition_dir = tmp_path / "partition"
    partition_dir.mkdir()
    (partition_dir / "clients").mkdir()

    class_names = [f"Class_{i:02d}" for i in range(38)]
    meta = {
        "class_names": class_names,
        "content_aware": True,
        "group_aware": True,
        "condition": "label100",
    }
    (partition_dir / "partition_config.json").write_text(json.dumps(meta), encoding="utf-8")

    c0_rows = [
        {"relative_path": f"{class_names[0]}/img_01.jpg", "label": "0", "class_name": class_names[0], "group_id": "0", "client_id": "0"},
    ]
    val_rows = [
        {"relative_path": f"{class_names[1]}/img_01.jpg", "label": "1", "class_name": class_names[1], "group_id": "1", "client_id": "-1"},
    ]
    fieldnames = ["relative_path", "label", "class_name", "group_id", "client_id"]
    with (partition_dir / "clients/client_00.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(c0_rows)

    with (partition_dir / "global_val.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(val_rows)

    with (partition_dir / "global_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    dataset_root = tmp_path / "dataset"
    (dataset_root / class_names[0]).mkdir(parents=True)
    (dataset_root / class_names[1]).mkdir(parents=True)

    # Write expected content
    expected_bytes_0 = b"original_content_00"
    expected_bytes_1 = b"original_content_01"
    digest_0 = hashlib.sha256(expected_bytes_0).hexdigest()
    digest_1 = hashlib.sha256(expected_bytes_1).hexdigest()

    # Image 0 has tampered bytes on disk (same length 19 bytes)
    tampered_bytes_0 = b"tampered_content_00"
    assert len(expected_bytes_0) == len(tampered_bytes_0)
    (dataset_root / class_names[0] / "img_01.jpg").write_bytes(tampered_bytes_0)
    (dataset_root / class_names[1] / "img_01.jpg").write_bytes(expected_bytes_1)

    image_content = {
        f"{class_names[0]}/img_01.jpg": digest_0,
        f"{class_names[1]}/img_01.jpg": digest_1,
    }
    (partition_dir / "image_content.json").write_text(json.dumps(image_content), encoding="utf-8")

    # Calling audit_image_content must fail with ValueError because bytes differ from pinned inventory
    with pytest.raises(ValueError, match="Image bytes differ from pinned inventory"):
        audit_image_content(partition_dir, dataset_root)


def test_audit_image_content_fails_on_cross_split_byte_duplicate(tmp_path):
    """Verify that identical image bytes across splits/clients are caught."""
    partition_dir = tmp_path / "partition"
    partition_dir.mkdir()
    (partition_dir / "clients").mkdir()

    class_names = [f"Class_{i:02d}" for i in range(38)]
    meta = {
        "class_names": class_names,
        "content_aware": True,
        "group_aware": True,
        "condition": "label100",
    }
    (partition_dir / "partition_config.json").write_text(json.dumps(meta), encoding="utf-8")

    # Client 0 manifest
    c0_rows = [
        {"relative_path": f"{class_names[0]}/img_01.jpg", "label": "0", "class_name": class_names[0], "group_id": "0", "client_id": "0"},
    ]
    # Val manifest contains image with same bytes
    val_rows = [
        {"relative_path": f"{class_names[0]}/img_02.jpg", "label": "0", "class_name": class_names[0], "group_id": "1", "client_id": "-1"},
    ]

    fieldnames = ["relative_path", "label", "class_name", "group_id", "client_id"]
    with (partition_dir / "clients/client_00.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(c0_rows)

    with (partition_dir / "global_val.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(val_rows)

    with (partition_dir / "global_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    dataset_root = tmp_path / "dataset"
    (dataset_root / class_names[0]).mkdir(parents=True)
    
    # Both images have identical bytes!
    same_bytes = b"identical_image_data_12345"
    (dataset_root / class_names[0] / "img_01.jpg").write_bytes(same_bytes)
    (dataset_root / class_names[0] / "img_02.jpg").write_bytes(same_bytes)

    digest = hashlib.sha256(same_bytes).hexdigest()
    image_content = {
        f"{class_names[0]}/img_01.jpg": digest,
        f"{class_names[0]}/img_02.jpg": digest,
    }
    (partition_dir / "image_content.json").write_text(json.dumps(image_content), encoding="utf-8")

    with pytest.raises(ValueError, match="Identical image bytes across splits/clients"):
        audit_image_content(partition_dir, dataset_root)


def test_preflight_fails_when_duplicate_review_tampered(tmp_path):
    """Verify that preflight fails if duplicate_review.json hash does not match suite.json."""
    mock_suite = tmp_path / "suite"
    shutil.copytree(SUITE_V3, mock_suite)

    # Tamper with duplicate_review.json
    review_path = mock_suite / "duplicate_review.json"
    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    review_data[0]["decision"] = "rejected"
    review_path.write_text(json.dumps(review_data), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate_review.json hash mismatch"):
        preflight(mock_suite)


def test_preflight_fails_when_visually_verified_pairs_missing(tmp_path):
    """Verify that preflight fails if visually_verified_pairs.json is missing."""
    mock_suite = tmp_path / "suite"
    shutil.copytree(SUITE_V3, mock_suite)

    visual_path = mock_suite / "visually_verified_pairs.json"
    visual_path.unlink()

    with pytest.raises(FileNotFoundError, match="Missing visually_verified_pairs.json"):
        preflight(mock_suite)


def test_preflight_fails_when_verified_pair_crosses_splits(tmp_path):
    """Verify that preflight fails if a verified pair crosses splits (e.g. train vs test)."""
    mock_suite = tmp_path / "suite"
    shutil.copytree(SUITE_V3, mock_suite)

    # In label100, let's swap an image in centralized_train.csv with one in global_test.csv
    cond_dir = mock_suite / "label100"
    visual_pairs = json.loads((mock_suite / "visually_verified_pairs.json").read_text(encoding="utf-8"))
    pair = visual_pairs[0]
    target_img = "/".join(pair["a"].replace("\\", "/").split("/")[-2:])

    # Move target_img from train to test
    train_rows = list(csv.DictReader((cond_dir / "centralized_train.csv").open(encoding="utf-8")))
    test_rows = list(csv.DictReader((cond_dir / "global_test.csv").open(encoding="utf-8")))

    # Find row
    found_row = None
    for r in train_rows:
        if "/".join(r["relative_path"].replace("\\", "/").split("/")[-2:]) == target_img:
            found_row = r
            break

    if found_row:
        train_rows.remove(found_row)
        test_rows.append(found_row)

        with (cond_dir / "centralized_train.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
            w.writeheader()
            w.writerows(train_rows)

        with (cond_dir / "global_test.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
            w.writeheader()
            w.writerows(test_rows)

        with pytest.raises(ValueError):
            preflight(mock_suite)

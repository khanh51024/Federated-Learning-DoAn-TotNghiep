from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from src.data.plantvillage import PlantVillageDataset


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_real_mode_true_only_when_all_manifest_images_resolve(tmp_path: Path):
    root = tmp_path / "images"
    (root / "ClassA").mkdir(parents=True)
    (root / "ClassB").mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(root / "ClassA" / "a.jpg")
    Image.new("RGB", (16, 16), "green").save(root / "ClassB" / "b.jpg")

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"relative_path": "ClassA/a.jpg", "label": 0, "class_name": "ClassA"},
            {"relative_path": "ClassB/b.jpg", "label": 1, "class_name": "ClassB"},
        ],
    )

    ds = PlantVillageDataset(manifest, dataset_root=root, is_train=False)

    assert ds.real_mode is True
    assert ds.resolved_image_count == 2
    assert ds.missing_image_count == 0


def test_dataset_real_mode_false_when_even_one_manifest_image_is_missing(tmp_path: Path):
    root = tmp_path / "images"
    (root / "ClassA").mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(root / "ClassA" / "a.jpg")

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"relative_path": "ClassA/a.jpg", "label": 0, "class_name": "ClassA"},
            {"relative_path": "ClassB/missing.jpg", "label": 1, "class_name": "ClassB"},
        ],
    )

    ds = PlantVillageDataset(manifest, dataset_root=root, is_train=False)
    assert ds.real_mode is False
    assert ds.resolved_image_count == 1
    assert ds.missing_image_count == 1

    with pytest.raises(FileNotFoundError, match="requires real images"):
        PlantVillageDataset(
            manifest,
            dataset_root=root,
            is_train=False,
            require_real_images=True,
        )

"""
fl_training.data: Dataset adapters, evaluation transforms, and DataLoader builders.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.dataset import PlantVillageDataset, load_manifest
from src.data.fedavg import FedAvgPartition
from src.data.transforms import (
    BaseTransform,
    ClientDomainTransform,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILENETV3_IMAGE_SIZE,
    get_client_transform,
    get_default_transform,
    profiles_from_config,
)


def get_eval_transform(image_size: Tuple[int, int] = MOBILENETV3_IMAGE_SIZE) -> Callable[[Image.Image], torch.Tensor]:
    """
    Standard evaluation transform for MobileNetV3:
    Resize short side to 256 (bilinear), center crop 224x224,
    convert to float tensor [0, 1], normalize with ImageNet mean and std.
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])


class ManifestDataset(Dataset):
    """
    Manifest-backed dataset for training, validation, or test.
    Reads records directly, resolves paths against dataset_root or base_dir.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self.transform = transform or get_eval_transform()
        self.seed = int(seed)
        self.epoch = 0

        self.records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row["relative_path"].replace("\\", "/")
                # Determine disk path
                if self.dataset_root:
                    # Look for class_name / filename in relative_path
                    parts = rel.split("/")
                    # Usually: relative_path is ../PlantVillage-Dataset/raw/color/Class/file.jpg
                    # or PlantVillage-Dataset/raw/color/Class/file.jpg or Class/file.jpg
                    if len(parts) >= 2:
                        sub = Path(parts[-2]) / parts[-1]
                        candidate = self.dataset_root / sub
                        if candidate.exists():
                            full_path = candidate
                        else:
                            full_path = self.dataset_root / rel
                    else:
                        full_path = self.dataset_root / rel
                else:
                    full_path = Path(rel).resolve()

                self.records.append({
                    "relative_path": rel,
                    "full_path": full_path,
                    "label": int(row["label"]),
                    "class_name": row.get("class_name", ""),
                    "client_id": int(row.get("client_id", -1)),
                    "group_id": row.get("group_id", f"img::{rel}"),
                    "feature_skew": row.get("feature_skew", "none"),
                    "partition_seed": int(row.get("seed", self.seed) or self.seed),
                })

        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img_path = rec["full_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if isinstance(self.transform, dict):
                t = self.transform.get(rec["client_id"])
                if t is None:
                    raise KeyError(f"No transform configured for client_id={rec['client_id']}")
            else:
                t = self.transform

            # Python's hash() is process-randomized.  A digest gives the same
            # augmentation/noise seed under Ray worker reuse and multi-worker
            # DataLoaders for the same sample/client/epoch.
            seed_material = (
                f"{self.seed}|{rec['partition_seed']}|{rec['client_id']}|"
                f"{self.epoch}|{rec['relative_path']}"
            ).encode("utf-8")
            sample_seed = int.from_bytes(
                hashlib.blake2b(seed_material, digest_size=8).digest(), "little"
            ) & 0x7FFF_FFFF_FFFF_FFFF

            if isinstance(t, (BaseTransform, ClientDomainTransform)):
                data = t(img, rng_seed=sample_seed)
            else:
                data = t(img)

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))

        return data, rec["label"]


def build_training_loader(
    manifest_path: str | Path,
    dataset_root: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    transform: Optional[Callable] = None,
    client_id: Optional[int] = None,
    feature_skew: str = "none",
    partition_dir: Optional[str | Path] = None,
) -> DataLoader:
    """
    Build a deterministically seeded training DataLoader with set_epoch support.
    """
    train_transform = transform
    if train_transform is None:
        if client_id is None:
            train_transform = get_default_transform(augment=True)
        else:
            custom_profile = None
            config_path = Path(partition_dir) / "partition_config.json" if partition_dir else None
            if config_path and config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    partition_config = json.load(f)
                profiles = profiles_from_config(
                    partition_config.get("client_profiles"),
                    num_clients=int(partition_config.get("num_clients", client_id + 1)),
                    seed=int(partition_config.get("seed", seed)),
                    level=feature_skew,
                )
                if client_id < len(profiles):
                    custom_profile = profiles[client_id]
            train_transform = get_client_transform(
                client_id=client_id,
                seed=seed,
                level=feature_skew,
                augment=True,
                custom_profile=custom_profile,
            )
    dataset = ManifestDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        transform=train_transform,
        seed=seed,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,  # Keep final singleton batches
        pin_memory=pin_memory,
        generator=gen,
        persistent_workers=False,
    )

    def set_epoch(epoch: int) -> None:
        dataset.set_epoch(epoch)
        gen.manual_seed(seed + int(epoch) * 100_003)

    loader.set_epoch = set_epoch  # type: ignore
    return loader


def build_centralized_training_loader(
    partition_dir: str | Path,
    dataset_root: str | Path,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    feature_skew: str = "none",
) -> DataLoader:
    """Build the pooled baseline loader while replaying each row's client profile."""
    part_dir = Path(partition_dir)
    config: Dict[str, Any] = {}
    config_path = part_dir / "partition_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    with open(part_dir / "centralized_train.csv", "r", encoding="utf-8") as file:
        client_ids = sorted({int(row["client_id"]) for row in csv.DictReader(file)})
    profiles = profiles_from_config(
        config.get("client_profiles"),
        num_clients=max(client_ids, default=-1) + 1,
        seed=int(config.get("seed", seed)),
        level=feature_skew,
    )
    transform_map = {
        client_id: get_client_transform(
            client_id=client_id,
            seed=seed,
            level=feature_skew,
            augment=True,
            custom_profile=profiles[client_id] if client_id < len(profiles) else None,
        )
        for client_id in client_ids
    }
    return build_training_loader(
        manifest_path=part_dir / "centralized_train.csv",
        dataset_root=dataset_root,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform=transform_map,  # type: ignore[arg-type]
    )


def build_evaluation_loader(
    manifest_path: str | Path,
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build evaluation DataLoader without shuffle or random augmentation.
    """
    dataset = ManifestDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        transform=get_eval_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def get_client_manifest_path(partition_dir: str | Path, client_id: int) -> Path:
    return Path(partition_dir) / "clients" / f"client_{client_id:02d}.csv"


def get_client_sample_count(partition_dir: str | Path, client_id: int) -> int:
    """Return n_k, verified from manifest line count."""
    manifest_path = get_client_manifest_path(partition_dir, client_id)
    with open(manifest_path, "r", encoding="utf-8") as f:
        # Subtract header line
        return sum(1 for line in f if line.strip()) - 1

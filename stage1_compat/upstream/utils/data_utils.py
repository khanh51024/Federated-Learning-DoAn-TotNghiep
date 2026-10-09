"""Nạp PlantVillage với split train/validation/test dùng chung cho mọi baseline."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from utils.config import (
    DATA_DIR, DEFAULT_BATCH_SIZE, IMAGE_SIZE, RESULTS_DIR, SEED, SPLITS_DIR,
)
from utils.reproducibility import set_seed

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class TransformSubset(Dataset):
    """Một tập con ImageFolder với transform riêng, tránh augmentation ở test."""

    def __init__(self, base: datasets.ImageFolder, indices: Sequence[int], transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.targets = [base.targets[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base[self.indices[index]]
        return self.transform(image), label


def split_path(seed: int = SEED) -> Path:
    """Versioned path prevents silently reusing the obsolete 80/20 split."""
    return SPLITS_DIR / f"plantvillage_train0.72_val0.08_seed{seed}.json"


def get_base_dataset(data_dir: Path | str = DATA_DIR) -> datasets.ImageFolder:
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy PlantVillage color tại: {path}")
    return datasets.ImageFolder(path)


def get_or_create_split(
    total: int, labels: Sequence[int], seed: int = SEED,
    train_ratio: float = 0.72, validation_ratio: float = 0.08,
) -> tuple[list[int], list[int], list[int]]:
    """Create a deterministic stratified 72/8/20 split.

    Validation selects checkpoints; test remains untouched until final reporting.
    """
    if len(labels) != total:
        raise ValueError("Số nhãn không khớp số ảnh.")
    path = split_path(seed)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["total_samples"] != total:
            raise ValueError("Split đã lưu không khớp số lượng ảnh hiện tại. Xóa file split để tạo lại.")
        return payload["train_indices"], payload["validation_indices"], payload["test_indices"]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_indices, validation_indices, test_indices = [], [], []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        train_end = int(len(indices) * train_ratio)
        validation_end = train_end + int(len(indices) * validation_ratio)
        train_indices.extend(indices[:train_end].tolist())
        validation_indices.extend(indices[train_end:validation_end].tolist())
        test_indices.extend(indices[validation_end:].tolist())
    rng.shuffle(train_indices); rng.shuffle(validation_indices); rng.shuffle(test_indices)
    all_indices = train_indices + validation_indices + test_indices
    if len(all_indices) != total or len(set(all_indices)) != total:
        raise RuntimeError("Split không bao phủ chính xác toàn bộ dataset.")
    path.write_text(json.dumps({
        "seed": seed, "train_ratio": train_ratio, "validation_ratio": validation_ratio,
        "test_ratio": 1 - train_ratio - validation_ratio, "total_samples": total,
        "train_indices": train_indices, "validation_indices": validation_indices, "test_indices": test_indices,
    }), encoding="utf-8")
    return train_indices, validation_indices, test_indices


def load_datasets(seed: int = SEED, data_dir: Path | str = DATA_DIR):
    set_seed(seed)
    base = get_base_dataset(data_dir)
    train_indices, validation_indices, test_indices = get_or_create_split(len(base), base.targets, seed)
    return (
        TransformSubset(base, train_indices, TRAIN_TRANSFORM),
        TransformSubset(base, validation_indices, EVAL_TRANSFORM),
        TransformSubset(base, test_indices, EVAL_TRANSFORM),
        base.classes,
    )


def make_loader(dataset: Dataset, batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def save_json(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / name
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output

"""Dataset and DataLoader loading adapter tuân thủ chính xác protocol upstream GĐ1."""

import json
from stage1_compat.integrity import validate_indices
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from stage1_compat.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLIENTS,
    IMAGE_SIZE,
    NUM_CLASSES,
    SEED,
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    TOTAL_DATASET_SAMPLES,
    TRAIN_SAMPLES,
    UPSTREAM_DIR,
)

# Transforms chuẩn từ upstream GĐ1 (Resize trực tiếp 224x224, không crop)
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
    """Tập con ImageFolder với transform riêng biệt (độc lập augmentation train/eval)."""

    def __init__(self, base: datasets.ImageFolder, indices: Sequence[int], transform):
        self.identity_context: dict = {}
        self.round_seconds: float = 60.0
        self.initialization_sha256: str | None = None
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.targets = [base.targets[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base[self.indices[index]]
        return self.transform(image), label


def get_split_file(seed: int = SEED) -> Path:
    return UPSTREAM_DIR / "experiments" / "splits" / f"plantvillage_train0.72_val0.08_seed{seed}.json"


def get_partition_file(alpha: float, seed: int = SEED, clients: int = DEFAULT_NUM_CLIENTS) -> Path:
    alpha_label = f"{int(alpha)}" if float(alpha).is_integer() else f"{alpha:g}".replace(".", "_")
    return UPSTREAM_DIR / "experiments" / "partitions" / f"plantvillage_alpha{alpha_label}_clients{clients}_seed{seed}.json"


def load_stage1_datasets(
    data_dir: Path | str,
    seed: int = SEED,
) -> tuple[TransformSubset, TransformSubset, TransformSubset, list[str]]:
    """Nạp PlantVillage dataset với split 72/8/20 đã được pin từ GĐ1."""
    path = Path(data_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục PlantVillage tại: {path}")

    base_dataset = datasets.ImageFolder(str(path))
    total_samples = len(base_dataset)
    if total_samples != TOTAL_DATASET_SAMPLES:
        raise ValueError(
            f"Tổng số ảnh trong dataset ({total_samples}) không khớp với protocol GĐ1 ({TOTAL_DATASET_SAMPLES})."
        )
    if len(base_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Số lớp dataset ({len(base_dataset.classes)}) không khớp với protocol ({NUM_CLASSES})."
        )

    split_file = get_split_file(seed)
    if not split_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file split đã pin tại: {split_file}")

    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    if split_payload["total_samples"] != total_samples:
        raise ValueError("File split không khớp tổng số mẫu của dataset hiện tại.")

    train_indices = split_payload["train_indices"]
    validation_indices = split_payload["validation_indices"]
    test_indices = split_payload["test_indices"]

    # Kiểm tra tính toàn vẹn của split
    validate_indices([train_indices, validation_indices, test_indices], total_samples)
    all_indices = train_indices + validation_indices + test_indices
    if len(all_indices) != total_samples or len(set(all_indices)) != total_samples:
        raise RuntimeError("Split không bao phủ chính xác tập dữ liệu (có trùng lặp hoặc thiếu index).")

    train_set = TransformSubset(base_dataset, train_indices, TRAIN_TRANSFORM)
    val_set = TransformSubset(base_dataset, validation_indices, EVAL_TRANSFORM)
    test_set = TransformSubset(base_dataset, test_indices, EVAL_TRANSFORM)

    return train_set, val_set, test_set, base_dataset.classes


def load_stage1_partition(
    alpha: float,
    train_labels: Sequence[int],
    seed: int = SEED,
    num_clients: int = DEFAULT_NUM_CLIENTS,
) -> list[list[int]]:
    """Nạp Dirichlet partition đã được pin từ GĐ1 cho 5 clients."""
    partition_file = get_partition_file(alpha, seed, num_clients)
    if not partition_file.exists():
        raise FileNotFoundError(f"Không tìm thấy partition đã pin: {partition_file}")

    payload = json.loads(partition_file.read_text(encoding="utf-8"))
    partitions = payload["partitions"]

    if len(partitions) != num_clients:
        raise ValueError(f"Số client trong partition ({len(partitions)}) != {num_clients}")

    total_train = len(train_labels)
    if payload["total_samples"] != total_train:
        raise ValueError(f"Tổng số mẫu trong partition ({payload['total_samples']}) != {total_train}")

    validate_indices(partitions, total_train)
    if payload["seed"] != seed or payload["alpha"] != alpha or payload["num_clients"] != num_clients:
        raise ValueError("Partition metadata mismatch")
    actual_hist = [np.bincount([train_labels[i] for i in ids], minlength=NUM_CLASSES).tolist() for ids in partitions]
    if actual_hist != payload["class_distribution"]:
        raise ValueError("Partition class histogram mismatch")
    flattened = [idx for client_indices in partitions for idx in client_indices]
    if len(flattened) != total_train or len(set(flattened)) != total_train:
        raise ValueError("Partition không bao phủ đúng tập huấn luyện một cách rời rạc.")

    return partitions


def make_stage1_loader(
    dataset: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

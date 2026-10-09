"""
src.data.plantvillage: Quản lý nạp dữ liệu PlantVillage từ manifest CSV.
Hỗ trợ fallback sinh synthetic dummy tensor phục vụ CI/CD & Unit Tests khi không có dataset thô trên máy.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transform(is_train: bool = True) -> transforms.Compose:
    """Tạo transform chuẩn 224x224 cho MobileNetV3-Small."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


import os
import logging

logger = logging.getLogger(__name__)

class PlantVillageDataset(Dataset):
    """
    Dataset nạp ảnh từ CSV manifest.
    Hỗ trợ cơ chế tự động tạo synthetic dummy tensor nếu không tìm thấy file ảnh gốc
    trên máy, giúp toàn bộ pipeline và unit test chạy thông suốt mọi lúc.
    Khi require_real_images=True, sẽ báo lỗi nếu thiếu ảnh thực tế.
    """

    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: Optional[str | Path] = None,
        transform: Optional[Any] = None,
        is_train: bool = True,
        require_real_images: bool = False,
    ):
        self.csv_path = Path(csv_path)
        if dataset_root:
            self.dataset_root = Path(dataset_root)
        else:
            env_root = os.environ.get("DATASET_ROOT") or os.environ.get("PLANTVILLAGE_DATASET_ROOT")
            self.dataset_root = Path(env_root) if env_root else None

        self.transform = transform or get_default_transform(is_train=is_train)
        self.require_real_images = require_real_images
        self.samples: List[Dict[str, Any]] = []
        self._warned_synthetic = False

        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        "relative_path": row["relative_path"],
                        "label": int(row["label"]),
                        "class_name": row.get("class_name", ""),
                    })

        # Đảm bảo dataset không bao giờ rỗng để DataLoader không bị lỗi num_samples=0
        if not self.samples:
            for i in range(38):
                self.samples.append({
                    "relative_path": f"synthetic_{i:02d}.jpg",
                    "label": i % 38,
                    "class_name": f"Class_{i:02d}",
                })

        # Xác định data provenance ngay lúc khởi tạo thay vì chờ tới __getitem__.
        # Điều này cho phép Flower server/client phân biệt chắc chắn real vs synthetic.
        self.resolved_image_count = 0
        self.missing_image_count = len(self.samples)
        self.missing_image_examples: List[str] = []
        if self.dataset_root is not None:
            missing = []
            resolved = 0
            for sample in self.samples:
                candidate = self.dataset_root / sample["relative_path"]
                if candidate.is_file():
                    resolved += 1
                else:
                    if len(missing) < 20:
                        missing.append(sample["relative_path"])
            self.resolved_image_count = resolved
            self.missing_image_count = len(self.samples) - resolved
            self.missing_image_examples = missing

        self.real_mode = (
            self.dataset_root is not None
            and len(self.samples) > 0
            and self.missing_image_count == 0
        )

        if self.require_real_images and not self.real_mode:
            raise FileNotFoundError(
                f"PlantVillage manifest {self.csv_path} requires real images, but only "
                f"{self.resolved_image_count}/{len(self.samples)} paths resolve under {self.dataset_root}. "
                f"Missing examples: {self.missing_image_examples[:5]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        label = item["label"]
        rel_path = item["relative_path"]

        image_file: Optional[Path] = None
        if self.dataset_root:
            candidate = self.dataset_root / rel_path
            if candidate.exists():
                image_file = candidate

        if image_file and image_file.exists():
            image = Image.open(image_file).convert("RGB")
            tensor = self.transform(image)
        else:
            if self.require_real_images:
                raise FileNotFoundError(
                    f"Không tìm thấy file ảnh gốc PlantVillage: {rel_path} tại {self.dataset_root}. "
                    f"Vui lòng kiểm tra lại cấu hình DATASET_ROOT hoặc PLANTVILLAGE_DATASET_ROOT."
                )
            if not self._warned_synthetic:
                logger.warning(
                    "[Cảnh báo dữ liệu] Thiếu ảnh gốc tại %s (%s). Đang dùng dummy tensor phục vụ kiểm thử pipeline (không đại diện cho độ chính xác thực tế).",
                    self.dataset_root, rel_path
                )
                self._warned_synthetic = True

            # Fallback tạo dummy tensor ổn định theo tên file
            seed_val = int(hashlib.md5(rel_path.encode()).hexdigest()[:8], 16) % (2**31 - 1)
            rng = np.random.RandomState(seed_val)
            arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(arr)
            tensor = self.transform(image)

        return tensor, label


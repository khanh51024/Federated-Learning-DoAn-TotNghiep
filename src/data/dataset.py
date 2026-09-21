"""
Manifest-backed datasets and loaders for the three GĐ2 regimes.

One partition directory supports all three baselines from the same files, which
is what makes the Centralized / Federated / Local-only comparison fair:

  * Centralized  -> `centralized_train.csv` (union of all clients)
  * Federated    -> `clients/client_XX.csv` + that client's feature-skew profile
  * Local-only   -> `clients/client_XX.csv`, trained in isolation
  * Evaluation   -> `global_test.csv`, clean transform, shared by all three

PyTorch is optional: with torch installed you get real `torch.utils.data`
objects (num_workers, pin_memory); without it a dependency-free numpy loader
with the same iteration contract, so the data layer can be validated on a
machine that only has numpy/Pillow.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import DataLoader as TorchDataLoader

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TorchDataset = object
    TORCH_AVAILABLE = False

from .transforms import (
    MOBILENETV3_IMAGE_SIZE,
    BaseTransform,
    get_default_transform,
)

MANIFEST_FIELDS = [
    "relative_path", "label", "class_name", "client_id", "split",
    "group_id", "scenario", "alpha", "quantity_alpha", "feature_skew", "seed",
]


def sample_rng_seed(base_seed: int, sample_id: str, epoch: int = 0) -> int:
    """
    Stable per-sample RNG seed. Keeps feature-skew noise and augmentation
    reproducible run-to-run and identical across DataLoader workers.
    """
    payload = f"{base_seed}|{epoch}|{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2 ** 31 - 1)


class PlantVillageDataset(TorchDataset):
    """Dataset over one manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        base_dir: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        image_size: tuple = MOBILENETV3_IMAGE_SIZE,
        seed: int = 42,
        return_dict: bool = False,
        dataset_root: Optional[str | Path] = None,
        dataset_prefix: Optional[str] = None,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # Default base_dir = package root (…/gd2_federated_learning), which is
        # where relative dataset paths in the manifests are anchored.
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parents[2]
        else:
            self.base_dir = Path(base_dir)

        self.transform = transform or BaseTransform(image_size=image_size)
        self.seed = int(seed)
        self.return_dict = return_dict
        self.dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        self.dataset_prefix = (dataset_prefix or "").replace("\\", "/").rstrip("/")
        self.epoch = 0
        self.records: List[Dict[str, Any]] = []
        self._load_manifest()

    # -- loading ----------------------------------------------------------
    def _load_manifest(self) -> None:
        with open(self.manifest_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rel = (row.get("relative_path") or "").strip()
                if not rel:
                    continue
                if self.dataset_root is not None:
                    prefix = self.dataset_prefix + "/"
                    if not rel.startswith(prefix):
                        raise ValueError(f"Manifest path does not start with dataset prefix {prefix}: {rel}")
                    full_path = self.dataset_root / rel[len(prefix):].replace("/", os.sep)
                else:
                    full_path = self.base_dir / rel.replace("/", os.sep)
                self.records.append({
                    "relative_path": rel,
                    "full_path": full_path,
                    "label": int(row["label"]),
                    "class_name": row.get("class_name", ""),
                    "client_id": _to_int(row.get("client_id"), -1),
                    "split": row.get("split") or "train",
                    "group_id": row.get("group_id") or f"img::{rel}",
                    "scenario": row.get("scenario") or "unknown",
                    "alpha": _to_float(row.get("alpha"), -1.0),
                    "quantity_alpha": _to_float(row.get("quantity_alpha"), -1.0),
                    "feature_skew": row.get("feature_skew") or "none",
                    "seed": _to_int(row.get("seed"), -1),
                })
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    # -- dataset protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = rec["full_path"]
        if not path.exists():
            raise FileNotFoundError(f"Image not found on disk: {path}")

        with Image.open(path) as img:
            img = img.convert("RGB")
            rng_seed = sample_rng_seed(self.seed, rec["relative_path"], self.epoch)
            try:
                transform = self.transform.get(rec["client_id"]) if isinstance(self.transform, dict) else self.transform
                if transform is None:
                    raise KeyError(f"No transform configured for client_id={rec['client_id']}")
                data = transform(img, rng_seed=rng_seed)
            except TypeError:
                # Third-party callables that only accept the image.
                data = transform(img)

        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data, dtype=np.float32)
            if TORCH_AVAILABLE:
                data = torch.from_numpy(data)

        label = rec["label"]
        if TORCH_AVAILABLE:
            label = torch.tensor(label, dtype=torch.long)

        if self.return_dict:
            return {
                "image": data,
                "label": label,
                "class_name": rec["class_name"],
                "relative_path": rec["relative_path"],
                "client_id": rec["client_id"],
                "group_id": rec["group_id"],
                "index": idx,
            }
        return data, label

    # -- convenience ------------------------------------------------------
    @property
    def labels(self) -> List[int]:
        return [r["label"] for r in self.records]

    @property
    def group_ids(self) -> List[str]:
        return [r["group_id"] for r in self.records]

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
        return counts

    def label_histogram(self, num_classes: int) -> np.ndarray:
        hist = np.zeros(num_classes, dtype=np.int64)
        for lab in self.labels:
            hist[lab] += 1
        return hist


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


class DataLoaderSimple:
    """
    Dependency-free batch loader with the same iteration contract as
    `torch.utils.data.DataLoader`. Shuffle order is a pure function of
    (seed, epoch) via `set_epoch`, so runs are reproducible.
    """

    def __init__(
        self,
        dataset: PlantVillageDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shuffling differs per epoch but stays reproducible."""
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(len(self.dataset))
        rng = np.random.default_rng(self.seed * 10_007 + self.epoch)
        return rng.permutation(len(self.dataset))

    def __iter__(self):
        order = self._order()
        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            samples = [self.dataset[int(i)] for i in batch_idx]
            images = [s[0] for s in samples]
            labels = [s[1] for s in samples]

            if TORCH_AVAILABLE and isinstance(images[0], torch.Tensor):
                yield torch.stack(images), torch.stack(labels)
            else:
                yield np.stack(images, axis=0), np.asarray(labels)


def build_loader(
    dataset: PlantVillageDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
):
    """
    Return a torch DataLoader when torch is installed, else DataLoaderSimple.
    Both expose `set_epoch` and yield `(images, labels)`.
    """
    if TORCH_AVAILABLE:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory if pin_memory is not None else TORCH_AVAILABLE,
            generator=gen,
            persistent_workers=False,
        )
        def set_epoch(epoch: int) -> None:
            dataset.set_epoch(epoch)
            gen.manual_seed(seed + int(epoch) * 100_003)
        loader.set_epoch = set_epoch
        return loader
    return DataLoaderSimple(
        dataset, batch_size=batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last
    )


def load_manifest(
    manifest_path: str | Path,
    base_dir: Optional[str | Path] = None,
    dataset_root: Optional[str | Path] = None,
    dataset_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read a manifest CSV into plain dicts (used by audits and reports)."""
    ds = PlantVillageDataset.__new__(PlantVillageDataset)
    ds.manifest_path = Path(manifest_path)
    ds.base_dir = Path(base_dir) if base_dir is not None else Path(".")
    ds.dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
    ds.dataset_prefix = (dataset_prefix or "").replace("\\", "/").rstrip("/")
    ds.records = []
    ds._load_manifest()
    return ds.records


def quick_stats(manifest_path: str | Path, num_classes: int = 38) -> Dict[str, Any]:
    """Cheap manifest summary without touching any image file."""
    records = load_manifest(manifest_path)
    hist = np.zeros(num_classes, dtype=np.int64)
    groups = set()
    for r in records:
        hist[r["label"]] += 1
        groups.add(r["group_id"])
    return {
        "manifest": str(manifest_path),
        "images": len(records),
        "leaf_groups": len(groups),
        "classes_present": int(np.count_nonzero(hist)),
        "class_min": int(hist.min()),
        "class_max": int(hist.max()),
    }

"""
FedAvg-ready data layer.

This is the bridge between the partitioned manifests and a FedAvg training run
with a MobileNetV3 client model. It owns the three things a FedAvg server needs
and that are easy to get wrong:

1. **n_k and the aggregation weights.** FedAvg (McMahan et al., 2017) combines
   client parameters as a weighted mean with w_k = n_k / sum_j n_j. Those n_k
   must be the *training* sample counts of the clients actually participating
   in the round. They are read from `fedavg_meta.json`, so the server never
   recomputes them from a manifest it might have filtered.

2. **The three regimes from one partition.** Centralized (upper bound),
   Federated (the system under study) and Local-only (lower bound) all read
   the same directory and are scored on the same `global_test.csv`. That shared
   test set is what makes the reported gap meaningful.

3. **Feature skew applied consistently.** Client profiles are restored from
   `partition_config.json`, so replaying a partition reproduces the exact
   photometric domain each facility had during the original run.

Nothing here requires PyTorch; when torch is installed the loaders returned are
real `torch.utils.data.DataLoader`s, ready to hand to a Flower `NumPyClient` /
`Client` or a plain FedAvg loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset import (
    PlantVillageDataset,
    TORCH_AVAILABLE,
    build_loader,
)
from .transforms import (
    MOBILENETV3_IMAGE_SIZE,
    ClientFeatureProfile,
    get_client_transform,
    get_default_transform,
    profiles_from_config,
)

MANIFEST_NAME = "fedavg_meta.json"
CONFIG_NAME = "partition_config.json"


class FedAvgPartition:
    """Read-only view of one partition directory, shaped for FedAvg."""

    def __init__(
        self,
        partition_dir: str | Path,
        dataset_root: Optional[str | Path] = None,
        feature_skew: Optional[str] = None,
        image_size: tuple = MOBILENETV3_IMAGE_SIZE,
        augment_train: bool = True,
    ):
        self.dir = Path(partition_dir).resolve()
        meta_path = self.dir / MANIFEST_NAME
        cfg_path = self.dir / CONFIG_NAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Run scripts/partition_dataset.py first."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)
        self.config: Dict[str, Any] = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # Manifest `relative_path`s are anchored at the partition package root
        # (they start with the configured dataset path, e.g.
        # "../PlantVillage-Dataset/raw/color/..."). `manifest_anchor` records
        # how to get from this partition directory back to that root, so the
        # whole tree stays portable if the folder is moved.
        anchor_rel = self.meta["paths"].get("manifest_anchor", "../..")
        dataset_rel = self.meta["paths"]["dataset_root_relative_to_manifests"]
        self.base_dir = (self.dir / anchor_rel).resolve()
        self.dataset_root = (
            Path(dataset_root).resolve() if dataset_root is not None
            else (self.base_dir / dataset_rel).resolve()
        )
        self.dataset_rel = str(dataset_rel).replace("\\", "/").rstrip("/")
        self.dataset_root_overridden = dataset_root is not None

        self.num_clients = int(self.meta["num_clients"])
        self.num_classes = int(self.meta["num_classes"])
        self.class_names: List[str] = list(self.meta["class_names"])
        self.image_size = tuple(image_size)
        self.augment_train = bool(augment_train)

        # "none" disables feature skew for an ablation even if the partition
        # was generated with it; None means "use whatever was recorded".
        level = feature_skew if feature_skew is not None else self.meta["partition"]["feature_skew"]
        self.feature_skew = str(level).lower()
        self.seed = int(self.meta["partition"]["seed"])
        self.profiles: List[ClientFeatureProfile] = self._resolve_profiles()

        self._n_k = np.array(
            [self.meta["client_sample_counts"][f"client_{c:02d}"] for c in range(self.num_clients)],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # FedAvg essentials
    # ------------------------------------------------------------------
    def _resolve_profiles(self) -> List[ClientFeatureProfile]:
        recorded = self.config.get("client_profiles")
        if self.feature_skew == "none":
            return [
                ClientFeatureProfile(client_id=c, level="none",
                                     name=f"Facility_{c:02d} (feature skew disabled)")
                for c in range(self.num_clients)
            ]
        recorded_level = str(self.meta.get("partition", {}).get("feature_skew", "none")).lower()
        return profiles_from_config(
            recorded if recorded_level == self.feature_skew else None,
            num_clients=self.num_clients, seed=self.seed, level=self.feature_skew
        )

    def aggregation_weights_for(self, client_ids: List[int]) -> Dict[int, float]:
        """FedAvg weights normalized over the clients participating in one round."""
        ids = list(dict.fromkeys(int(c) for c in client_ids))
        if not ids or any(c < 0 or c >= self.num_clients for c in ids):
            raise ValueError("client_ids must contain valid, unique-in-effect client ids")
        counts = self._n_k[ids].astype(np.float64)
        if counts.sum() <= 0:
            raise ValueError("participating clients have no training samples")
        weights = counts / counts.sum()
        return {c: float(w) for c, w in zip(ids, weights)}

    def _dataset(self, manifest: Path, transform) -> PlantVillageDataset:
        return PlantVillageDataset(
            manifest_path=manifest,
            base_dir=self.base_dir,
            dataset_root=self.dataset_root if self.dataset_root_overridden else None,
            dataset_prefix=self.dataset_rel,
            transform=transform,
            image_size=self.image_size,
            seed=self.seed,
        )

    @property
    def n_samples(self) -> np.ndarray:
        """n_k -- local training set size of each client."""
        return self._n_k.copy()

    @property
    def aggregation_weights(self) -> np.ndarray:
        """w_k = n_k / sum_j n_j -- the FedAvg combination weights."""
        total = int(self._n_k.sum())
        if total == 0:
            raise ValueError("Partition has no training samples.")
        return self._n_k.astype(np.float64) / total

    def client_weight(self, client_id: int) -> float:
        return float(self.aggregation_weights[client_id])

    @property
    def total_train_samples(self) -> int:
        return int(self._n_k.sum())

    def participating_clients(self, min_samples: int = 1) -> List[int]:
        """Clients with enough data to take part in a round."""
        return [c for c in range(self.num_clients) if self._n_k[c] >= min_samples]

    # ------------------------------------------------------------------
    # datasets / loaders
    # ------------------------------------------------------------------
    def _manifest(self, name: str) -> Path:
        p = self.dir / name
        if not p.exists():
            raise FileNotFoundError(f"Manifest missing: {p}")
        return p

    def client_dataset(
        self,
        client_id: int,
        split: str = "train",
        feature_skew: Optional[bool] = None,
    ) -> PlantVillageDataset:
        """
        Local dataset of one facility. This single call serves BOTH the
        Federated client and the Local-only baseline -- the only difference is
        whether the resulting model is aggregated or not.
        """
        if not 0 <= client_id < self.num_clients:
            raise IndexError(f"client_id {client_id} out of range [0, {self.num_clients})")

        if split == "val":
            path = self.dir / "clients" / f"client_{client_id:02d}_val.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Regenerate with client_val_ratio > 0."
                )
        elif split == "train":
            path = self.dir / "clients" / f"client_{client_id:02d}.csv"
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        use_skew = self.feature_skew != "none" if feature_skew is None else bool(feature_skew)
        augment = self.augment_train and split == "train"
        if use_skew and split == "train":
            transform = get_client_transform(
                client_id=client_id,
                seed=self.seed,
                image_size=self.image_size,
                level=self.feature_skew,
                augment=augment,
                custom_profile=self.profiles[client_id],
            )
        else:
            transform = get_default_transform(image_size=self.image_size, augment=augment)

        return self._dataset(path, transform)

    def centralized_dataset(self) -> PlantVillageDataset:
        """Upper-bound baseline: every facility's data pooled in one place."""
        transforms = {}
        for c in range(self.num_clients):
            if self.feature_skew == "none":
                transforms[c] = get_default_transform(self.image_size, augment=self.augment_train)
            else:
                transforms[c] = get_client_transform(
                    c, self.seed, self.image_size, self.feature_skew,
                    self.augment_train, self.profiles[c],
                )
        return self._dataset(self._manifest("centralized_train.csv"), transforms)

    def global_test_dataset(self) -> PlantVillageDataset:
        """Shared evaluation set -- identical inputs for all three baselines."""
        return self._dataset(
            self._manifest("global_test.csv"),
            get_default_transform(image_size=self.image_size, augment=False),
        )

    def global_val_dataset(self) -> Optional[PlantVillageDataset]:
        path = self.dir / "global_val.csv"
        if not path.exists():
            return None
        return self._dataset(path, get_default_transform(image_size=self.image_size, augment=False))

    def client_loader(
        self,
        client_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        split: str = "train",
    ):
        return build_loader(
            self.client_dataset(client_id, split=split),
            batch_size=batch_size, shuffle=shuffle, seed=self.seed + client_id,
            num_workers=num_workers,
        )

    def centralized_loader(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0):
        return build_loader(
            self.centralized_dataset(), batch_size=batch_size, shuffle=shuffle,
            seed=self.seed, num_workers=num_workers,
        )

    def global_test_loader(self, batch_size: int = 128, shuffle: bool = False, num_workers: int = 0):
        return build_loader(
            self.global_test_dataset(), batch_size=batch_size, shuffle=shuffle,
            seed=self.seed, num_workers=num_workers,
        )

    def all_client_loaders(self, batch_size: int = 32, num_workers: int = 0) -> Dict[int, Any]:
        """One local loader per facility -- what a simulated FedAvg round iterates."""
        return {
            c: self.client_loader(c, batch_size=batch_size, num_workers=num_workers)
            for c in range(self.num_clients)
        }

    # ------------------------------------------------------------------
    # readiness check
    # ------------------------------------------------------------------
    def readiness_report(self, check_images: int = 0) -> Dict[str, Any]:
        """
        Verify the partition can actually drive a FedAvg + MobileNetV3 run.
        `check_images` opens that many image files per split as a disk-path
        smoke test (0 disables the check).
        """
        problems: List[str] = []
        warnings: List[str] = []

        w = self.aggregation_weights
        if abs(float(w.sum()) - 1.0) > 1e-6:
            problems.append(f"aggregation weights sum to {w.sum():.8f}, expected 1.0")

        empty = [c for c in range(self.num_clients) if self._n_k[c] == 0]
        if empty:
            problems.append(f"clients with zero training samples: {empty}")

        tiny = [c for c in range(self.num_clients) if 0 < self._n_k[c] < 50]
        if tiny:
            warnings.append(
                f"clients with <50 training samples (FedAvg updates will be noisy): {tiny}"
            )

        # Required manifests
        required = ["centralized_train.csv", "global_test.csv"] + [
            f"clients/client_{c:02d}.csv" for c in range(self.num_clients)
        ]
        for rel in required:
            if not (self.dir / rel).exists():
                problems.append(f"missing manifest: {rel}")

        # Model contract
        model = self.meta.get("model", {})
        if tuple(model.get("input_size", ())) != self.image_size:
            warnings.append(
                f"requested image_size {self.image_size} differs from recorded "
                f"{tuple(model.get('input_size', ()))}"
            )
        if model.get("num_classes") != self.num_classes:
            problems.append("model.num_classes disagrees with partition num_classes")

        # Integrity flags recorded at partition time
        integrity = self.meta.get("integrity", {})
        if not integrity.get("conserved", False):
            problems.append("partition failed the sample-conservation audit")
        if not integrity.get("leakage_free", False):
            problems.append("partition has leaf-group leakage across clients or train/test")
        if not integrity.get("min_size_satisfied", True):
            warnings.append("min_samples_per_client could not be satisfied for every client")

        # Global test must cover every class, otherwise macro-F1 is undefined
        hist = np.array(self.config.get("global_test_class_histogram", []), dtype=np.int64)
        if hist.size == self.num_classes:
            missing = [self.class_names[i] for i in np.flatnonzero(hist == 0)]
            if missing:
                warnings.append(f"classes absent from global test set: {missing}")

        image_probe = {}
        if check_images:
            for label, ds in (
                ("centralized", self.centralized_dataset()),
                ("client_00", self.client_dataset(0)),
                ("global_test", self.global_test_dataset()),
            ):
                n = min(check_images, len(ds))
                idxs = np.linspace(0, len(ds) - 1, n).astype(int) if n else []
                shapes = []
                for i in idxs:
                    img, lab = ds[int(i)]
                    arr = img.numpy() if TORCH_AVAILABLE and hasattr(img, "numpy") else np.asarray(img)
                    shapes.append(arr.shape)
                    if not (0 <= int(lab) < self.num_classes):
                        problems.append(f"{label}: label {int(lab)} out of range")
                uniq = set(shapes)
                image_probe[label] = {"probed": int(n), "shapes": sorted(uniq)}
                if len(uniq) != 1:
                    problems.append(f"{label}: inconsistent tensor shapes {sorted(uniq)}")
                elif shapes:
                    expected = (3,) + tuple(self.image_size)
                    if tuple(shapes[0]) != expected:
                        problems.append(
                            f"{label}: shape {shapes[0]} != MobileNetV3 contract {expected}"
                        )

        return {
            "partition_dir": str(self.dir),
            "dataset_root": str(self.dataset_root),
            "torch_available": TORCH_AVAILABLE,
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "total_train_samples": self.total_train_samples,
            "client_sample_counts": self._n_k.tolist(),
            "aggregation_weights": [round(float(x), 6) for x in w],
            "weights_sum": round(float(w.sum()), 8),
            "scenario": self.meta["partition"]["scenario"],
            "alpha": self.meta["partition"]["alpha"],
            "quantity_alpha": self.meta["partition"]["quantity_alpha"],
            "feature_skew": self.feature_skew,
            "group_aware": self.meta["partition"].get("group_aware", True),
            "model": model,
            "image_probe": image_probe,
            "problems": problems,
            "warnings": warnings,
            "ready": not problems,
        }

    def summary(self) -> Dict[str, Any]:
        """Compact one-glance description, for logs and reports."""
        return {
            "scenario": self.meta["partition"]["scenario"],
            "alpha": self.meta["partition"]["alpha"],
            "quantity_alpha": self.meta["partition"]["quantity_alpha"],
            "feature_skew": self.feature_skew,
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "n_k": self._n_k.tolist(),
            "weights": [round(float(x), 4) for x in self.aggregation_weights],
            "train": self.total_train_samples,
            "test": int(np.sum(self.config.get("global_test_class_histogram", [0]))),
        }

    def __repr__(self) -> str:
        return (
            f"FedAvgPartition(dir='{self.dir.name}', scenario='{self.meta['partition']['scenario']}', "
            f"alpha={self.meta['partition']['alpha']}, clients={self.num_clients}, "
            f"train={self.total_train_samples})"
        )

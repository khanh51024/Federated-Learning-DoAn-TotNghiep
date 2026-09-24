"""
DatasetPartitioner -- end-to-end GĐ2 pipeline.

    scan  ->  attach leaf groups  ->  group-aware global test split
          ->  non-IID client partition  ->  integrity audits
          ->  manifests + statistics + FedAvg metadata

Every split boundary in this pipeline is **group-aware**: the assignment unit
is a whole physical leaf, so near-duplicate frames never straddle train/test or
two clients (see `leaf_groups.py` for why that matters on PlantVillage).

Outputs written to `<output_dir>`:
    clients/client_XX.csv        per-facility training manifest (Federated / Local-only)
    clients/client_XX_val.csv    optional per-client validation manifest
    global_test.csv              shared evaluation set for all three baselines
    global_val.csv               optional global validation set
    centralized_train.csv        union of all clients (Centralized upper bound)
    client_class_matrix.csv      client x class counts
    statistics.csv               per-client metrics
    partition_config.json        full provenance + summary metrics + audits
    fedavg_meta.json             everything a FedAvg server/client needs
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dirichlet_split import partition as run_partition
from .leaf_groups import (
    IMAGE_EXTS,
    attach_group_ids,
    group_samples,
    load_leaf_map,
)
from .metrics import (
    audit_group_integrity,
    audit_sample_integrity,
    calculate_partition_metrics,
    compute_client_class_matrix,
)
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILENETV3_IMAGE_SIZE,
    create_deterministic_client_profile,
)

MANIFEST_FIELDS = [
    "relative_path", "label", "class_name", "client_id", "split",
    "group_id", "leaf_group_id", "scenario", "alpha", "quantity_alpha", "feature_skew", "seed",
]
# `split_id` is an internal partitioning key (leaf group, or the image itself
# when group_aware=False). It is never written to a manifest.


class DatasetPartitioner:
    """Orchestrates the whole partition + audit + export pipeline."""

    def __init__(
        self,
        dataset_path: str = "../PlantVillage-Dataset/raw/color",
        leaf_map_path: str = "../PlantVillage-Dataset/leaf-map.json",
        base_dir: Optional[str] = None,
        test_ratio: float = 0.20,
        val_ratio: float = 0.0,
        client_val_ratio: float = 0.0,
        num_clients: int = 10,
        alpha: float = 0.1,
        quantity_alpha: float = 0.1,
        seed: int = 42,
        scenario: str = "label_skew",
        feature_skew: str = "moderate",
        min_samples_per_client: int = 10,
        max_retries: int = 20,
        group_aware: bool = True,
        rare_class_threshold: int = 500,
        content_aware: bool = False,
        content_hashes: Optional[Dict[str, Any]] = None,
        verified_pairs: Optional[List[Any]] = None,
    ):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[2]
        self.dataset_rel_path = dataset_path.replace("\\", "/")
        self.dataset_full_path = (self.base_dir / self.dataset_rel_path).resolve()
        if not self.dataset_full_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_full_path}")

        self.leaf_map_rel_path = leaf_map_path.replace("\\", "/")
        self.leaf_map_full_path = (self.base_dir / self.leaf_map_rel_path).resolve()

        self.test_ratio = float(test_ratio)
        self.val_ratio = float(val_ratio)
        self.client_val_ratio = float(client_val_ratio)
        self.num_clients = num_clients
        self.alpha = float(alpha)
        self.quantity_alpha = float(quantity_alpha)
        self.seed = seed
        self.scenario = scenario.lower()
        self.feature_skew = feature_skew.lower()
        self.min_samples_per_client = min_samples_per_client
        self.max_retries = max_retries
        self.group_aware = group_aware
        self.rare_class_threshold = rare_class_threshold
        self.content_aware = content_aware
        self.verified_pairs = verified_pairs
        if self.content_aware and not self.group_aware:
            raise ValueError('content_aware requires group_aware=True')
        self.content_hashes: Dict[str, str] = {
            k.replace("\\", "/"): (str(v.get("sha256", "")) if isinstance(v, dict) else str(v))
            for k, v in (content_hashes or {}).items()
        }

        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be in (0, 1)")
        if not 0 <= self.val_ratio < 1 or not 0 <= self.client_val_ratio < 1:
            raise ValueError("validation ratios must be in [0, 1)")
        if self.test_ratio + self.val_ratio >= 1:
            raise ValueError("test_ratio + val_ratio must be < 1")
        if self.num_clients <= 0 or self.min_samples_per_client < 0 or self.max_retries <= 0:
            raise ValueError("num_clients/max_retries must be positive and min_samples non-negative")
        if self.alpha <= 0 or self.quantity_alpha <= 0:
            raise ValueError("alpha and quantity_alpha must be positive")
        if self.feature_skew not in {"none", "mild", "moderate", "strong"}:
            raise ValueError(f"Unknown feature_skew: {self.feature_skew}")

        self.class_names: List[str] = []
        self.class_to_id: Dict[str, int] = {}
        self.all_samples: List[Dict[str, Any]] = []
        self.leaf_audit: Dict[str, Any] = {}
        # Set when group_aware=False and the image-level split leaks leaf groups.
        self.leakage_warning: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # step 1: scan
    # ------------------------------------------------------------------
    def scan_dataset(self) -> None:
        """
        Collect every image, assign class ids by sorted directory name, and
        resolve leaf groups.

        Filenames are sorted explicitly: `os.scandir` returns filesystem order,
        which is not guaranteed stable across machines or filesystems, and an
        unstable input order silently changes the partition even with a fixed
        seed.
        """
        class_dirs = sorted(
            (p for p in self.dataset_full_path.iterdir() if p.is_dir()),
            key=lambda x: x.name,
        )
        self.class_names = [d.name for d in class_dirs]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        self.all_samples = []
        for class_name in self.class_names:
            class_dir = self.dataset_full_path / class_name
            label = self.class_to_id[class_name]
            names = sorted(
                e.name for e in os.scandir(class_dir)
                if e.is_file() and e.name.lower().endswith(IMAGE_EXTS)
            )
            for fname in names:
                self.all_samples.append({
                    "relative_path": f"{self.dataset_rel_path}/{class_name}/{fname}",
                    "class_name": class_name,
                    "label": label,
                })

        if not self.all_samples:
            raise RuntimeError(f"No images found under {self.dataset_full_path}")

        # The leaf map is ALWAYS resolved, even for the image-level ablation:
        # `group_id` keeps the true leaf so the audit can measure exactly how
        # much leakage a non-group-aware split introduces. `split_id` is what
        # the splitter actually treats as indivisible.
        leaf_map = load_leaf_map(self.leaf_map_full_path)
        if self.group_aware and not leaf_map:
            raise FileNotFoundError(
                f"A non-empty leaf map is required for group-aware splitting: {self.leaf_map_full_path}"
            )
        hash_map: Optional[Dict[str, str]] = self.content_hashes if self.content_aware else None
        if self.content_aware and not hash_map:
            # Compute hashes once when the standalone partition CLI is used
            # without prepare-data's source cache.
            computed_hashes: Dict[str, str] = {}
            for s in self.all_samples:
                rel = "/".join(s["relative_path"].replace("\\", "/").split("/")[-2:])
                computed_hashes[rel] = hashlib.sha256((self.dataset_full_path / rel).read_bytes()).hexdigest()
            hash_map = computed_hashes
        self.leaf_audit = attach_group_ids(self.all_samples, leaf_map, hash_map, verified_pairs=self.verified_pairs)
        if self.content_aware and hash_map is not None:
            self.content_hashes = hash_map
        for s in self.all_samples:
            s["split_id"] = s["group_id"] if self.group_aware else f"single::{s['relative_path']}"

    # ------------------------------------------------------------------
    # step 2: group-aware global test/val split
    # ------------------------------------------------------------------
    def split_global_test(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Stratified, group-aware extraction of the global test (and optional
        global val) set BEFORE any client partitioning. Whole leaf groups are
        moved, so the test set shares no leaf with the training pool.
        """
        rng = np.random.default_rng(self.seed)

        by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(self.class_names))}
        for item in self.all_samples:
            by_class[item["label"]].append(item)

        train_pool: List[Dict[str, Any]] = []
        test_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []

        for class_id in sorted(by_class):
            items = by_class[class_id]
            if not items:
                continue

            groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                groups.setdefault(it["split_id"], []).append(it)

    # ------------------------------------------------------------------
    # step 2: group-aware global test/val split
    # ------------------------------------------------------------------
    def split_global_test(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Stratified, group-aware extraction of the global test (and optional
        global val) set BEFORE any client partitioning. Whole leaf groups are
        moved, so the test set shares no leaf with the training pool.
        """
        rng = np.random.default_rng(self.seed)

        by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(self.class_names))}
        for item in self.all_samples:
            by_class[item["label"]].append(item)

        train_pool: List[Dict[str, Any]] = []
        test_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []

        for class_id in sorted(by_class):
            items = by_class[class_id]
            if not items:
                continue

            groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                groups.setdefault(it["split_id"], []).append(it)
            group_ids = sorted(groups)

            order = rng.permutation(len(group_ids))
            shuffled_ids = [group_ids[i] for i in order]

            n = len(items)
            n_test_target = round(n * self.test_ratio)
            n_val_target = round(n * self.val_ratio) if self.val_ratio > 0 else 0

            # A group is indivisible, so counts are reached by accumulation and
            # can land slightly off target. Guard: always keep >= 1 image for
            # train, and always keep >= 1 test image for the class.
            test_ids: List[str] = []
            val_ids: List[str] = []
            taken = 0
            for gid in shuffled_ids:
                if taken >= n_test_target:
                    break
                # Never consume everything: leave at least one group for train.
                if len(test_ids) + len(val_ids) >= len(shuffled_ids) - 1:
                    break
                test_ids.append(gid)
                taken += len(groups[gid])

            taken_val = 0
            for gid in shuffled_ids:
                if taken_val >= n_val_target:
                    break
                if gid in test_ids:
                    continue
                if len(test_ids) + len(val_ids) >= len(shuffled_ids) - 1:
                    break
                val_ids.append(gid)
                taken_val += len(groups[gid])

            test_ids_s, val_ids_s = set(test_ids), set(val_ids)
            for gid in group_ids:
                bucket = test_samples if gid in test_ids_s else (
                    val_samples if gid in val_ids_s else train_pool
                )
                bucket.extend(groups[gid])

            if not any(gid not in test_ids_s and gid not in val_ids_s for gid in group_ids):
                raise RuntimeError(
                    f"Class '{self.class_names[class_id]}' left with no training "
                    f"samples (only {len(group_ids)} leaf group(s)). Lower "
                    f"test_ratio or disable group_aware for this class."
                )

        return train_pool, test_samples, val_samples

    # ------------------------------------------------------------------
    # step 3: optional per-client validation split
    # ------------------------------------------------------------------
    def _split_client_val(
        self,
        client_samples: Dict[int, List[Dict[str, Any]]],
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
        """Group-aware per-client train/val split (val_ratio > 0 only)."""
        if self.client_val_ratio <= 0:
            return client_samples, {c: [] for c in client_samples}

        rng = np.random.default_rng(self.seed + 977)
        train_out: Dict[int, List[Dict[str, Any]]] = {}
        val_out: Dict[int, List[Dict[str, Any]]] = {}

        for c in sorted(client_samples):
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in client_samples[c]:
                groups.setdefault(it["split_id"], []).append(it)
            gids = sorted(groups)
            order = rng.permutation(len(gids))
            target = round(len(client_samples[c]) * self.client_val_ratio)

            val_ids, taken = set(), 0
            for i in order:
                if taken >= target:
                    break
                if len(val_ids) >= len(gids) - 1:
                    break  # keep at least one group for training
                gid = gids[i]
                val_ids.add(gid)
                taken += len(groups[gid])

            tr, va = [], []
            for gid in gids:
                (va if gid in val_ids else tr).extend(groups[gid])
            train_out[c], val_out[c] = tr, va

        return train_out, val_out

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def partition(self, output_dir: str | Path) -> Dict[str, Any]:
        if not self.all_samples:
            self.scan_dataset()

        train_pool, test_samples, val_samples = self.split_global_test()

        groups_by_class = group_samples(train_pool, key="split_id")

        client_samples, split_diag = run_partition(
            groups_by_class=groups_by_class,
            scenario=self.scenario,
            num_clients=self.num_clients,
            alpha=self.alpha,
            quantity_alpha=self.quantity_alpha,
            seed=self.seed,
            min_samples_per_client=self.min_samples_per_client,
            max_retries=self.max_retries,
        )

        client_samples, client_val = self._split_client_val(client_samples)

        # ---- stamp metadata -------------------------------------------
        for c in sorted(client_samples):
            for item in client_samples[c]:
                item.update({
                    "client_id": c, "split": "train", "scenario": self.scenario,
                    "alpha": self.alpha, "quantity_alpha": self.quantity_alpha,
                    "feature_skew": self.feature_skew, "seed": self.seed,
                })
            for item in client_val.get(c, []):
                item.update({
                    "client_id": c, "split": "val", "scenario": self.scenario,
                    "alpha": self.alpha, "quantity_alpha": self.quantity_alpha,
                    "feature_skew": self.feature_skew, "seed": self.seed,
                })
        for item in test_samples:
            item.update({
                "client_id": -1, "split": "test", "scenario": self.scenario,
                "alpha": self.alpha, "quantity_alpha": self.quantity_alpha,
                "feature_skew": "none", "seed": self.seed,
            })
        for item in val_samples:
            item.update({
                "client_id": -1, "split": "val", "scenario": self.scenario,
                "alpha": self.alpha, "quantity_alpha": self.quantity_alpha,
                "feature_skew": "none", "seed": self.seed,
            })

        # ---- audits (hard failures are real bugs, not warnings) ---------
        sample_audit = audit_sample_integrity(
            client_samples, train_pool, test_samples, val_samples, client_val=client_val
        )
        if not sample_audit["conserved"]:
            raise AssertionError(f"Sample integrity audit failed: {sample_audit}")

        group_audit = audit_group_integrity(
            client_samples, test_samples, val_samples, client_val=client_val
        )
        if not group_audit["leakage_free"]:
            if self.group_aware:
                # Group-aware splitting is supposed to make this impossible;
                # if it happens the splitter is broken, so fail loudly.
                raise AssertionError(f"Leaf-group leakage detected: {group_audit}")
            # group_aware=False is the deliberate image-level ablation: leakage
            # is expected and is reported as a measured quantity, not an error.
            self.leakage_warning = group_audit

        # ---- metrics ----------------------------------------------------
        metrics = calculate_partition_metrics(
            client_samples,
            total_classes=len(self.class_names),
            class_names=self.class_names,
            rare_class_threshold=self.rare_class_threshold,
        )
        matrix_df = compute_client_class_matrix(
            client_samples, num_classes=len(self.class_names), class_names=self.class_names
        )
        test_hist = np.zeros(len(self.class_names), dtype=np.int64)
        for it in test_samples:
            test_hist[it["label"]] += 1

        client_profiles = [
            create_deterministic_client_profile(c, seed=self.seed, level=self.feature_skew).to_dict()
            for c in range(self.num_clients)
        ]

        # ---- write ------------------------------------------------------
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        clients_dir = out_path / "clients"
        clients_dir.mkdir(exist_ok=True)
        # A repeated run into the same directory must not retain validation
        # manifests from an older configuration.
        for stale in clients_dir.glob("client_*.csv"):
            stale.unlink()
        for stale_name in ("global_val.csv",):
            stale = out_path / stale_name
            if stale.exists():
                stale.unlink()

        for c in range(self.num_clients):
            self._write_csv(clients_dir / f"client_{c:02d}.csv", client_samples[c])
            if client_val.get(c):
                self._write_csv(clients_dir / f"client_{c:02d}_val.csv", client_val[c])

        self._write_csv(out_path / "global_test.csv", test_samples)
        if val_samples:
            self._write_csv(out_path / "global_val.csv", val_samples)

        centralized = [row for c in range(self.num_clients) for row in client_samples[c]]
        self._write_csv(out_path / "centralized_train.csv", centralized)

        matrix_df.to_csv(out_path / "client_class_matrix.csv")
        pd.DataFrame(metrics["client_details"]).to_csv(out_path / "statistics.csv", index=False)

        n_k = [len(client_samples[c]) for c in range(self.num_clients)]
        total_train = int(sum(n_k))
        config_data = {
            "schema_version": 3,
            "dataset_path": self.dataset_rel_path,
            "leaf_map_path": self.leaf_map_rel_path,
            "group_aware": self.group_aware,
            "content_aware": self.content_aware,
            "total_images": len(self.all_samples),
            "total_classes": len(self.class_names),
            "class_names": self.class_names,
            "train_samples": total_train,
            "test_samples": len(test_samples),
            "val_samples": len(val_samples),
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "quantity_alpha": self.quantity_alpha,
            "seed": self.seed,
            "scenario": self.scenario,
            "feature_skew": self.feature_skew,
            "min_samples_per_client": self.min_samples_per_client,
            "test_ratio": self.test_ratio,
            "val_ratio": self.val_ratio,
            "client_val_ratio": self.client_val_ratio,
            "rare_class_threshold": self.rare_class_threshold,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split_diagnostics": split_diag,
            "leaf_group_audit": self.leaf_audit,
            "group_integrity": group_audit,
            "expected_leakage": self.leakage_warning,
            "sample_integrity": sample_audit,
            "global_test_class_histogram": test_hist.tolist(),
            "summary_metrics": {k: v for k, v in metrics.items() if k != "client_details"},
            "client_profiles": client_profiles,
        }
        fingerprint_fields = {
            key: config_data[key] for key in (
                "schema_version", "dataset_path", "leaf_map_path", "group_aware",
                "content_aware",
                "class_names", "num_clients", "alpha", "quantity_alpha", "seed",
                "scenario", "feature_skew", "test_ratio", "val_ratio", "client_val_ratio",
            )
        }
        provenance: Dict[str, Any] = self._provenance(test_samples)
        provenance["partition_config_sha256"] = hashlib.sha256(
            json.dumps(fingerprint_fields, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        config_data["provenance"] = provenance
        with open(out_path / "partition_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self._write_fedavg_meta(out_path, config_data, n_k, client_val, self.base_dir)
        if self.content_aware:
            inventory = {'/'.join(key.replace('\\', '/').split('/')[-2:]): digest
                         for key, digest in self.content_hashes.items()}
            (out_path / 'image_content.json').write_text(json.dumps(inventory, sort_keys=True), encoding='utf-8')

        return config_data

    # ------------------------------------------------------------------
    # writers
    # ------------------------------------------------------------------
    @staticmethod
    def _provenance(test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        def digest(values: List[str]) -> str:
            return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()

        versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
        try:
            import PIL
            versions["pillow"] = PIL.__version__
        except Exception:
            pass
        return {
            "global_test_paths_sha256": digest([s["relative_path"] for s in test_samples]),
            "class_mapping_sha256": digest([f"{s['label']}:{s['class_name']}" for s in test_samples]),
            "library_versions": versions,
        }

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        """
        Rows are sorted by (group_id, relative_path) so the CSV is byte-stable
        for a given seed -- diffable across runs and machines.
        """
        ordered = sorted(rows, key=lambda r: (r["group_id"], r["relative_path"]))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in ordered:
                writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})

    @staticmethod
    def _write_fedavg_meta(
        out_path: Path,
        config_data: Dict[str, Any],
        n_k: List[int],
        client_val: Dict[int, List[Dict[str, Any]]],
        base_dir: Path,
    ) -> None:
        """
        `fedavg_meta.json` is the single file a FedAvg server needs:
        sample counts n_k, the aggregation weights n_k / sum(n_k), the input
        contract for MobileNetV3, and the manifest layout.
        """
        total = int(sum(n_k))
        weights = [round(v / total, 8) for v in n_k] if total else [0.0] * len(n_k)
        # Relative path from this partition directory back to the anchor that
        # manifest `relative_path` values were written against. Storing it keeps
        # the tree portable: moving the whole package does not break loading.
        manifest_anchor = os.path.relpath(Path(base_dir).resolve(), Path(out_path).resolve())
        manifest_anchor = manifest_anchor.replace(os.sep, "/")
        meta = {
            "schema_version": 3,
            "algorithm": "FedAvg",
            "aggregation": "weighted mean of client parameters by n_k",
            "num_clients": config_data["num_clients"],
            "num_classes": config_data["total_classes"],
            "class_names": config_data["class_names"],
            "client_sample_counts": {f"client_{c:02d}": n_k[c] for c in range(len(n_k))},
            "client_aggregation_weights": {f"client_{c:02d}": weights[c] for c in range(len(weights))},
            "total_train_samples": total,
            "client_val_counts": {
                f"client_{c:02d}": len(client_val.get(c, [])) for c in range(len(n_k))
            },
            "model": {
                "backbone": "mobilenet_v3_small",
                "input_size": list(MOBILENETV3_IMAGE_SIZE),
                "channels": 3,
                "layout": "CHW",
                "dtype": "float32",
                "normalize_mean": IMAGENET_MEAN.tolist(),
                "normalize_std": IMAGENET_STD.tolist(),
                "num_classes": config_data["total_classes"],
            },
            "paths": {
                "dataset_root_relative_to_manifests": config_data["dataset_path"],
                "manifest_anchor": manifest_anchor,
                "client_manifests": "clients/client_{cid:02d}.csv",
                "client_val_manifests": "clients/client_{cid:02d}_val.csv",
                "global_test": "global_test.csv",
                "global_val": "global_val.csv",
                "centralized_train": "centralized_train.csv",
            },
            "partition": {
                "scenario": config_data["scenario"],
                "alpha": config_data["alpha"],
                "quantity_alpha": config_data["quantity_alpha"],
                "feature_skew": config_data["feature_skew"],
                "group_aware": config_data["group_aware"],
                "content_aware": config_data.get("content_aware", False),
                "seed": config_data["seed"],
                "test_ratio": config_data["test_ratio"],
            },
            "client_profiles": config_data["client_profiles"],
            "integrity": {
                "conserved": config_data["sample_integrity"]["conserved"],
                "leakage_free": config_data["group_integrity"]["leakage_free"],
                "min_size_satisfied": config_data["split_diagnostics"]["min_size_satisfied"],
            },
        }
        with open(out_path / "fedavg_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

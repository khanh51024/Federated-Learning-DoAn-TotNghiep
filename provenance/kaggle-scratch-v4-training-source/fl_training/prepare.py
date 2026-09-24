"""
fl_training.prepare: Prepare training partitions with train/val/test splits,
audits, and index for Federated Learning.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml  # type: ignore
from PIL import Image

from src.data.dirichlet_split import partition as run_partition
from src.data.leaf_groups import (
    IMAGE_EXTS,
    attach_group_ids,
    group_samples,
    load_leaf_map,
)
from src.data.metrics import (
    audit_group_integrity,
    audit_sample_integrity,
    calculate_partition_metrics,
    compute_client_class_matrix,
)
from src.data.partitioner import DatasetPartitioner, MANIFEST_FIELDS
from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILENETV3_IMAGE_SIZE,
    create_deterministic_client_profile,
)

REFERENCE_TEST_HASH_V3 = "340198e0b2f8945a1f2730b211aacd9ddcbd5b9c46404249771c69a77637e79e"


def compute_relative_paths_hash(relative_paths: List[str]) -> str:
    """Canonical SHA-256 hash of sorted relative paths."""
    normalized = [p.replace("\\", "/") for p in relative_paths]
    return hashlib.sha256("\n".join(sorted(normalized)).encode("utf-8")).hexdigest()


def compute_class_mapping_hash(samples: List[Dict[str, Any]]) -> str:
    """Canonical SHA-256 hash of label:class_name."""
    pairs = [f"{s['label']}:{s['class_name']}" for s in samples]
    return hashlib.sha256("\n".join(sorted(set(pairs))).encode("utf-8")).hexdigest()


def compute_semantic_config_hash(params: Dict[str, Any]) -> str:
    """
    Stable hash of parameters that affect data partitioning.
    Excludes machine paths, timestamps, etc.
    """
    canonical = {
        "alpha": float(params["alpha"]),
        "client_val_ratio": float(params.get("client_val_ratio", 0.0)),
        "content_aware": bool(params.get("content_aware", False)),
        "feature_skew": str(params.get("feature_skew", "none")),
        "group_aware": bool(params.get("group_aware", True)),
        "max_retries": int(params.get("max_retries", 20)),
        "min_samples_per_client": int(params.get("min_samples_per_client", 10)),
        "num_clients": int(params.get("num_clients", 10)),
        "quantity_alpha": float(params.get("quantity_alpha", 0.1)),
        "scenario": str(params["scenario"]),
        "schema_version": int(params.get("schema_version", 3)),
        "seed": int(params["seed"]),
        "test_ratio": float(params.get("test_ratio", 0.20)),
        "val_ratio": float(params.get("val_ratio", 0.10)),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_or_update_source_content_cache(
    dataset_full_path: Path,
    cache_path: Path,
    strict: bool = False,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Build or update source image content SHA-256 cache.
    Uses size/mtime_ns for fast verification; recalculates SHA-256 if changed or missing.
    """
    existing_cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                existing_cache = json.load(f).get("images", {})
        except Exception:
            existing_cache = {}

    all_files: List[Tuple[str, Path]] = []
    for root, _, files in os.walk(dataset_full_path):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                fp = Path(root) / fname
                rel = fp.relative_to(dataset_full_path).as_posix()
                all_files.append((rel, fp))

    all_files.sort(key=lambda x: x[0])

    updated_images: Dict[str, Any] = {}

    def process_image(item: Tuple[str, Path]) -> Tuple[str, Dict[str, Any]]:
        rel, fp = item
        st = fp.stat()
        cached = existing_cache.get(rel)
        if (
            not strict
            and cached
            and cached.get("size") == st.st_size
            and cached.get("mtime_ns") == st.st_mtime_ns
            and "sha256" in cached
        ):
            return rel, cached

        data = fp.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        return rel, {
            "sha256": sha,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rel, info in executor.map(process_image, all_files):
            updated_images[rel] = info

    aggregate_sha = hashlib.sha256(
        "\n".join(f"{k}:{updated_images[k]['sha256']}" for k in sorted(updated_images)).encode("utf-8")
    ).hexdigest()

    cache_data = {
        "dataset_root": str(dataset_full_path),
        "total_images": len(updated_images),
        "aggregate_content_sha256": aggregate_sha,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images": updated_images,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    return cache_data


def audit_manifest_directory(partition_dir: Path, total_source_images: int = 54305,
                             require_all_classes: bool = True,
                             dataset_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Comprehensive audit of a partition directory:
    - train/val/test disjoint by relative_path and by group_id
    - union of client train == centralized_train
    - client train + val + test == total source images
    - validation contains all 38 classes
    - each client has >= 10 samples
    - no duplicate samples across clients
    """
    clients_dir = partition_dir / "clients"
    client_files = sorted(clients_dir.glob("client_[0-9][0-9].csv"))
    if not client_files:
        raise AssertionError(f"No client manifests found in {clients_dir}")

    client_samples: Dict[int, List[Dict[str, Any]]] = {}
    client_paths: Dict[int, set] = {}
    client_groups: Dict[int, set] = {}
    all_train_paths: set = set()
    all_train_groups: set = set()

    def validate_rows(rows, name):
        if not rows:
            raise AssertionError(f"Empty manifest: {name}")
        if len({row["relative_path"] for row in rows}) != len(rows):
            raise AssertionError(f"Duplicate image inside {name}; n_k would be inflated")
        if any(not 0 <= int(row["label"]) < 38 for row in rows):
            raise AssertionError(f"Invalid class label in {name}")

    for idx, cf in enumerate(client_files):
        cid = int(cf.stem.split("_")[1])
        with open(cf, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        validate_rows(reader, cf.name)
        if any(int(row["client_id"]) != cid for row in reader):
            raise AssertionError(f"Client identity differs from filename: {cf}")
        if len(reader) < 10:
            raise AssertionError(f"Client {cid} has {len(reader)} < 10 samples in {cf}")
        client_samples[cid] = reader
        paths = set(r["relative_path"] for r in reader)
        groups = set(r["group_id"] for r in reader)

        # Check disjointness with other clients
        for other_cid, other_paths in client_paths.items():
            dup_paths = paths & other_paths
            if dup_paths:
                raise AssertionError(f"Duplicate images between client {cid} and {other_cid}: {len(dup_paths)}")
            dup_groups = groups & client_groups[other_cid]
            if dup_groups:
                raise AssertionError(f"Leaf group leakage between client {cid} and {other_cid}: {len(dup_groups)}")

        client_paths[cid] = paths
        client_groups[cid] = groups
        all_train_paths.update(paths)
        all_train_groups.update(groups)

    # Check centralized_train.csv
    central_file = partition_dir / "centralized_train.csv"
    with open(central_file, "r", encoding="utf-8") as f:
        central_rows = list(csv.DictReader(f))
    central_paths = set(r["relative_path"] for r in central_rows)
    validate_rows(central_rows, "centralized_train.csv")
    expected_records = {row["relative_path"]: row for rows in client_samples.values() for row in rows}
    for row in central_rows:
        original = expected_records.get(row["relative_path"])
        if original is not None and any(row.get(k) != original.get(k) for k in ("label", "class_name", "client_id", "group_id")):
            raise AssertionError("Centralized labels/client/group differ from client union")
    if central_paths != all_train_paths:
        raise AssertionError(
            f"centralized_train.csv mismatch with union of clients: "
            f"{len(central_paths)} vs {len(all_train_paths)}"
        )

    # Check global_test.csv
    test_file = partition_dir / "global_test.csv"
    with open(test_file, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    test_paths = set(r["relative_path"] for r in test_rows)
    validate_rows(test_rows, "global_test.csv")
    test_groups = set(r["group_id"] for r in test_rows)

    # Check global_val.csv
    val_file = partition_dir / "global_val.csv"
    if not val_file.exists():
        raise AssertionError(f"Missing required global_val.csv in {partition_dir}")
    with open(val_file, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    val_paths = set(r["relative_path"] for r in val_rows)
    validate_rows(val_rows, "global_val.csv")
    val_groups = set(r["group_id"] for r in val_rows)

    val_classes = set(int(r["label"]) for r in val_rows)
    if require_all_classes and len(val_classes) < 38:
        missing = sorted(set(range(38)) - val_classes)
        raise AssertionError(f"Validation set missing classes: {missing} (found {len(val_classes)}/38)")
    if require_all_classes and len({int(r["label"]) for r in test_rows}) < 38:
        raise AssertionError("Benchmark test must contain all 38 classes")

    # Disjointness checks
    tv_intersect = test_paths & val_paths
    if tv_intersect:
        raise AssertionError(f"Test and Val intersect by image: {len(tv_intersect)}")
    tr_te_intersect = all_train_paths & test_paths
    if tr_te_intersect:
        raise AssertionError(f"Train and Test intersect by image: {len(tr_te_intersect)}")
    tr_va_intersect = all_train_paths & val_paths
    if tr_va_intersect:
        raise AssertionError(f"Train and Val intersect by image: {len(tr_va_intersect)}")

    # Group disjointness checks
    tv_grp_intersect = test_groups & val_groups
    if tv_grp_intersect:
        raise AssertionError(f"Test and Val intersect by leaf group: {len(tv_grp_intersect)}")
    tr_te_grp_intersect = all_train_groups & test_groups
    if tr_te_grp_intersect:
        raise AssertionError(f"Train and Test intersect by leaf group: {len(tr_te_grp_intersect)}")
    tr_va_grp_intersect = all_train_groups & val_groups
    if tr_va_grp_intersect:
        raise AssertionError(f"Train and Val intersect by leaf group: {len(tr_va_grp_intersect)}")

    total_manifest_images = len(all_train_paths) + len(val_paths) + len(test_paths)
    if require_all_classes and total_manifest_images != total_source_images:
        raise AssertionError(
            f"Total partitioned images ({total_manifest_images}) != source images ({total_source_images})"
        )

    content_audit = None
    if dataset_root is not None:
        from .content_audit import audit_image_content
        expected_hashes = json.loads((partition_dir / "image_content.json").read_text(encoding="utf-8")) if (partition_dir / "image_content.json").exists() else None
        content_audit = audit_image_content(partition_dir, Path(dataset_root), hashes=expected_hashes)

    return {
        "audited_ok": True,
        "content_audit": content_audit,
        "num_clients": len(client_files),
        "train_samples": len(all_train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
        "total_samples": total_manifest_images,
        "val_classes_present": len(val_classes),
    }


def prepare_data(
    config_path: str | Path,
    base_dir: Optional[str | Path] = None,
    strict_cache: bool = False,
) -> Dict[str, Any]:
    """
    Main function to execute prepare-data:
    1. Parse config.
    2. Build or update source content cache.
    3. Partition datasets for specified scenarios into data/partitions_train_v1.
    4. Verify test hash matches partitions_v3 reference.
    5. Verify validation set has all 38 classes and is shared across all scenarios.
    6. Audit all manifests.
    7. Generate index.json.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pkg_root = Path(base_dir).resolve() if base_dir else config_path.parents[1]

    ds_cfg = cfg.get("dataset", {})
    pt_cfg = cfg.get("partition", {})
    out_cfg = cfg.get("output", {})
    target_scenarios = cfg.get("target_scenarios", [])

    dataset_rel_path = ds_cfg.get("path", "../PlantVillage-Dataset/raw/color")
    leaf_map_rel_path = ds_cfg.get("leaf_map_path", "../PlantVillage-Dataset/leaf-map.json")
    dataset_full_path = (pkg_root / dataset_rel_path).resolve()
    leaf_map_full_path = (pkg_root / leaf_map_rel_path).resolve()

    if not dataset_full_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_full_path}")

    # 1. Source content cache
    source_cache_rel = out_cfg.get("source_cache_path", "data/source_content_cache.json")
    source_cache_full = pkg_root / source_cache_rel
    source_cache = build_or_update_source_content_cache(
        dataset_full_path=dataset_full_path,
        cache_path=source_cache_full,
        strict=strict_cache or bool(ds_cfg.get("content_aware", False)),
    )
    total_source_images = source_cache["total_images"]

    # 2. Base partitioner setup
    test_ratio = float(ds_cfg.get("test_ratio", 0.20))
    val_ratio = float(ds_cfg.get("val_ratio", 0.10))
    client_val_ratio = float(cfg.get("client_val_ratio", 0.0))
    group_aware = bool(ds_cfg.get("group_aware", True))
    seed = int(pt_cfg.get("seed", 42))
    num_clients = int(pt_cfg.get("num_clients", 10))
    min_samples = int(pt_cfg.get("min_samples_per_client", 10))
    max_retries = int(pt_cfg.get("max_retries", 20))
    rare_thresh = int(pt_cfg.get("rare_class_threshold", 500))
    content_aware = bool(ds_cfg.get("content_aware", False))

    base_partitioner = DatasetPartitioner(
        dataset_path=dataset_rel_path,
        leaf_map_path=leaf_map_rel_path,
        base_dir=str(pkg_root),
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        client_val_ratio=client_val_ratio,
        num_clients=num_clients,
        seed=seed,
        group_aware=group_aware,
        min_samples_per_client=min_samples,
        max_retries=max_retries,
        rare_class_threshold=rare_thresh,
        content_aware=content_aware,
        content_hashes=source_cache.get("images", {}) if content_aware else None,
    )
    base_partitioner.scan_dataset()

    # Step: global test and validation split
    train_pool, test_samples, val_samples = base_partitioner.split_global_test()

    # Verify test hash
    test_paths = [s["relative_path"] for s in test_samples]
    test_hash = compute_relative_paths_hash(test_paths)
    ref_test_hash = out_cfg.get("reference_test_hash", REFERENCE_TEST_HASH_V3)
    if ref_test_hash and test_hash != ref_test_hash:
        raise AssertionError(
            f"Test set mismatch! Got {test_hash}, expected reference {ref_test_hash}. "
            f"Test set must remain strictly identical to partitions_v3."
        )

    # Verify val set
    val_paths = [s["relative_path"] for s in val_samples]
    val_hash = compute_relative_paths_hash(val_paths)
    val_classes = set(s["label"] for s in val_samples)
    if len(val_classes) < 38:
        missing = sorted(set(range(38)) - val_classes)
        raise AssertionError(f"Global validation set missing classes: {missing}")

    output_base_dir = pkg_root / out_cfg.get("base_dir", "data/partitions_train_v1")
    index_file = pkg_root / out_cfg.get("index_path", "data/partitions_train_v1/index.json")
    if index_file.exists():
        previous = json.loads(index_file.read_text(encoding="utf-8"))
        if previous.get("test_paths_sha256") != test_hash or previous.get("val_paths_sha256") != val_hash:
            raise ValueError("Existing index pins different validation/test sets; use a new versioned output directory")

    index_data: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": dataset_rel_path,
        "leaf_map_path": leaf_map_rel_path,
        "total_source_images": total_source_images,
        "test_samples": len(test_samples),
        "val_samples": len(val_samples),
        "train_pool_samples": len(train_pool),
        "test_paths_sha256": test_hash,
        "val_paths_sha256": val_hash,
        "class_mapping_sha256": compute_class_mapping_hash(test_samples),
        "partitions": {},
        "lookup": {},
    }

    results: List[Dict[str, Any]] = []

    for target in target_scenarios:
        scen = target["scenario"]
        scen_alpha = float(target.get("alpha", pt_cfg.get("alpha", 0.1)))
        scen_qalpha = float(target.get("quantity_alpha", pt_cfg.get("quantity_alpha", 0.1)))
        scen_fskew = str(target.get("feature_skew", "none"))
        scen_seed = int(target.get("seed", seed))

        # Check semantic config hash
        params = {
            "scenario": scen,
            "alpha": scen_alpha,
            "quantity_alpha": scen_qalpha,
            "feature_skew": scen_fskew,
            "seed": scen_seed,
            "num_clients": num_clients,
            "test_ratio": test_ratio,
            "val_ratio": val_ratio,
            "client_val_ratio": client_val_ratio,
            "content_aware": content_aware,
            "group_aware": group_aware,
            "min_samples_per_client": min_samples,
            "max_retries": max_retries,
            "schema_version": 3,
        }
        cfg_hash = compute_semantic_config_hash(params)
        short_hash = cfg_hash[:12]

        dir_name = f"{scen}__seed_{scen_seed}"
        if scen in ("label_skew", "label_quantity_skew"):
            dir_name += f"__alpha_{str(scen_alpha).replace('.', '_')}"
        if scen in ("quantity_skew", "label_quantity_skew"):
            dir_name += f"__qalpha_{str(scen_qalpha).replace('.', '_')}"
        dir_name += f"__feature_{scen_fskew}__{short_hash}"

        part_dir = output_base_dir / scen / dir_name
        rel_part_dir = part_dir.relative_to(pkg_root).as_posix()

        # Partition train_pool for this scenario
        groups_by_class = group_samples(train_pool, key="split_id")
        client_samples, split_diag = run_partition(
            groups_by_class=groups_by_class,
            scenario=scen,
            num_clients=num_clients,
            alpha=scen_alpha,
            quantity_alpha=scen_qalpha,
            seed=scen_seed,
            min_samples_per_client=min_samples,
            max_retries=max_retries,
        )

        # Build partition outputs
        part_dir.mkdir(parents=True, exist_ok=True)
        clients_dir = part_dir / "clients"
        clients_dir.mkdir(parents=True, exist_ok=True)

        for c in range(num_clients):
            for it in client_samples[c]:
                it.update({
                    "client_id": c, "split": "train", "scenario": scen,
                    "alpha": scen_alpha, "quantity_alpha": scen_qalpha,
                    "feature_skew": scen_fskew, "seed": scen_seed,
                })
            DatasetPartitioner._write_csv(clients_dir / f"client_{c:02d}.csv", client_samples[c])

        for it in test_samples:
            it.update({
                "client_id": -1, "split": "test", "scenario": scen,
                "alpha": scen_alpha, "quantity_alpha": scen_qalpha,
                "feature_skew": "none", "seed": scen_seed,
            })
        DatasetPartitioner._write_csv(part_dir / "global_test.csv", test_samples)

        for it in val_samples:
            it.update({
                "client_id": -1, "split": "val", "scenario": scen,
                "alpha": scen_alpha, "quantity_alpha": scen_qalpha,
                "feature_skew": "none", "seed": scen_seed,
            })
        DatasetPartitioner._write_csv(part_dir / "global_val.csv", val_samples)

        centralized = [row for c in range(num_clients) for row in client_samples[c]]
        DatasetPartitioner._write_csv(part_dir / "centralized_train.csv", centralized)

        # Metrics and statistics
        metrics = calculate_partition_metrics(
            client_samples,
            total_classes=len(base_partitioner.class_names),
            class_names=base_partitioner.class_names,
            rare_class_threshold=rare_thresh,
        )
        matrix_df = compute_client_class_matrix(
            client_samples,
            num_classes=len(base_partitioner.class_names),
            class_names=base_partitioner.class_names,
        )
        matrix_df.to_csv(part_dir / "client_class_matrix.csv")
        pd.DataFrame(metrics["client_details"]).to_csv(part_dir / "statistics.csv", index=False)

        n_k = [len(client_samples[c]) for c in range(num_clients)]
        total_train = int(sum(n_k))

        sample_audit = audit_sample_integrity(
            client_samples, train_pool, test_samples, val_samples, client_val={}
        )
        group_audit = audit_group_integrity(
            client_samples, test_samples, val_samples, client_val={}
        )

        client_profiles = [
            create_deterministic_client_profile(c, seed=scen_seed, level=scen_fskew).to_dict()
            for c in range(num_clients)
        ]

        test_hist = np.zeros(len(base_partitioner.class_names), dtype=np.int64)
        for it in test_samples:
            test_hist[it["label"]] += 1

        val_hist = np.zeros(len(base_partitioner.class_names), dtype=np.int64)
        for it in val_samples:
            val_hist[it["label"]] += 1

        config_data = {
            "schema_version": 3,
            "config_hash": cfg_hash,
            "dataset_path": dataset_rel_path,
            "leaf_map_path": leaf_map_rel_path,
            "group_aware": group_aware,
            "content_aware": content_aware,
            "total_images": total_source_images,
            "total_classes": len(base_partitioner.class_names),
            "class_names": base_partitioner.class_names,
            "train_samples": total_train,
            "test_samples": len(test_samples),
            "val_samples": len(val_samples),
            "num_clients": num_clients,
            "alpha": scen_alpha,
            "quantity_alpha": scen_qalpha,
            "seed": scen_seed,
            "scenario": scen,
            "feature_skew": scen_fskew,
            "min_samples_per_client": min_samples,
            "test_ratio": test_ratio,
            "val_ratio": val_ratio,
            "client_val_ratio": client_val_ratio,
            "rare_class_threshold": rare_thresh,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": {
                **DatasetPartitioner._provenance(test_samples),
                "global_val_paths_sha256": val_hash,
                "config_hash": cfg_hash,
            },
            "split_diagnostics": split_diag,
            "leaf_group_audit": base_partitioner.leaf_audit,
            "group_integrity": group_audit,
            "sample_integrity": sample_audit,
            "global_test_class_histogram": test_hist.tolist(),
            "global_val_class_histogram": val_hist.tolist(),
            "summary_metrics": {k: v for k, v in metrics.items() if k != "client_details"},
            "client_profiles": client_profiles,
        }

        with open(part_dir / "partition_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        DatasetPartitioner._write_fedavg_meta(
            part_dir, config_data, n_k, {}, pkg_root
        )
        if content_aware:
            inventory = {key: value["sha256"] for key, value in source_cache["images"].items()}
            (part_dir / "image_content.json").write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

        # Audit directory
        audit_res = audit_manifest_directory(part_dir, total_source_images=total_source_images)

        # Register in index
        lookup_key = f"{scen}__alpha_{scen_alpha}__qalpha_{scen_qalpha}__feature_{scen_fskew}__seed_{scen_seed}"
        part_info = {
            "scenario": scen,
            "alpha": scen_alpha,
            "quantity_alpha": scen_qalpha,
            "feature_skew": scen_fskew,
            "split_seed": scen_seed,
            "num_clients": num_clients,
            "config_hash": cfg_hash,
            "relative_dir": rel_part_dir,
            "n_k": n_k,
            "train_samples": total_train,
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "audit": audit_res,
        }
        index_data["partitions"][cfg_hash] = part_info
        index_data["lookup"][lookup_key] = cfg_hash
        results.append(part_info)

    # Save index.json
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "index_path": index_file.relative_to(pkg_root).as_posix(),
        "test_paths_sha256": test_hash,
        "val_paths_sha256": val_hash,
        "generated_partitions": results,
    }


def create_smoke_bundle(
    source_partition_dir: str | Path,
    bundle_dir: str | Path,
    target_client_images: int = 32,
    target_val_images: int = 64,
    seed: int = 42,
) -> Path:
    """
    Create an isolated 2-client smoke bundle for fast integration testing:
    - Picks client 0 and 1 from source partition
    - Selects whole leaf groups until ~target_client_images per client
    - Selects whole leaf groups from global_val until ~target_val_images
    - Never splits leaf groups
    - Writes self-contained manifest directory with fedavg_meta.json (num_clients=2)
    """
    src_dir = Path(source_partition_dir).resolve()
    dst_dir = Path(bundle_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    clients_dir = dst_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    def select_groups(manifest_csv: Path, target_count: int, stratified: bool = False) -> List[Dict[str, Any]]:
        with open(manifest_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r["group_id"], []).append(r)

        sorted_gids = sorted(groups.keys())
        # Holdout sampling must not depend on the preceding client distributions.
        selection_rng = np.random.default_rng(np.random.SeedSequence([
            seed, int.from_bytes(hashlib.sha256(manifest_csv.name.encode()).digest()[:4], 'little')
        ])) if stratified else rng
        shuffled_gids = [sorted_gids[i] for i in selection_rng.permutation(len(sorted_gids))]

        selected: List[Dict[str, Any]] = []
        chosen = set()
        if stratified:
            for label in sorted({r["label"] for r in rows}, key=int):
                gid = next(g for g in shuffled_gids if groups[g][0]["label"] == label)
                selected.extend(groups[gid])
                chosen.add(gid)
        for gid in shuffled_gids:
            if len(selected) >= target_count:
                break
            if gid not in chosen:
                selected.extend(groups[gid])
        return selected

    # 1. Client 0 & 1
    c0_rows = select_groups(src_dir / "clients" / "client_00.csv", target_client_images)
    c1_rows = select_groups(src_dir / "clients" / "client_01.csv", target_client_images)

    DatasetPartitioner._write_csv(clients_dir / "client_00.csv", c0_rows)
    DatasetPartitioner._write_csv(clients_dir / "client_01.csv", c1_rows)

    # 2. Centralized train
    centralized = c0_rows + c1_rows
    DatasetPartitioner._write_csv(dst_dir / "centralized_train.csv", centralized)

    # 3. Global Val
    val_rows = select_groups(src_dir / "global_val.csv", target_val_images, stratified=True)
    DatasetPartitioner._write_csv(dst_dir / "global_val.csv", val_rows)

    # Smoke evaluation covers every source class without breaking groups.
    test_rows = select_groups(src_dir / "global_test.csv", target_val_images, stratified=True)
    DatasetPartitioner._write_csv(dst_dir / "global_test.csv", test_rows)

    n_k = [len(c0_rows), len(c1_rows)]
    source_names = []
    source_meta = {}
    for filename in ("partition_config.json", "fedavg_meta.json"):
        if (src_dir / filename).exists():
            source_meta = json.loads((src_dir / filename).read_text(encoding="utf-8"))
            source_names = source_meta.get("class_names", [])
            if source_names:
                break
    meta = {
        "schema_version": 3,
        "smoke": True,
        "content_aware": source_meta.get("content_aware", False),
        "feature_skew": source_meta.get("feature_skew", "none"),
        "client_profiles": source_meta.get("client_profiles", [])[:2],
        "num_clients": 2,
        "class_names": source_names,
        "total_train_samples": sum(n_k),
        "client_sample_counts": {"client_00": n_k[0], "client_01": n_k[1]},
        "client_aggregation_weights": {
            "client_00": round(n_k[0] / sum(n_k), 8),
            "client_01": round(n_k[1] / sum(n_k), 8),
        },
    }
    with open(dst_dir / "fedavg_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # Training loaders read profiles from partition_config.json, not fedavg_meta.
    (dst_dir / "partition_config.json").write_text(json.dumps({
        **meta, "seed": source_meta.get("seed", seed),
        "scenario": source_meta.get("scenario", "label_skew"),
        "group_aware": source_meta.get("group_aware", True),
        "source_partition": str(src_dir),
    }, indent=2), encoding="utf-8")
    if (src_dir / "image_content.json").exists():
        inventory = json.loads((src_dir / "image_content.json").read_text(encoding="utf-8"))
        keys = ['/'.join(r['relative_path'].replace('\\', '/').split('/')[-2:])
                for r in centralized + val_rows + test_rows]
        (dst_dir / "image_content.json").write_text(json.dumps({k: inventory[k] for k in keys}, sort_keys=True), encoding="utf-8")

    return dst_dir

"""
fl_training.config: Parser, validator, and path resolver for training YAML contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

KNOWN_SECTIONS = {
    "schema_version",
    "data",
    "model",
    "training",
    "federation",
    "early_stopping",
    "lr_scheduler",
    "runtime",
    "output",
    "checkpoint",
    "diagnostics",
}

DATA_KEYS = {"dataset_root", "partition_index", "scenario", "alpha", "quantity_alpha", "feature_skew", "split_seed", "bundle_path"}
MODEL_KEYS = {"name", "num_classes", "weights"}
TRAINING_KEYS = {"seed", "local_epochs", "batch_size", "eval_batch_size", "optimizer", "lr", "momentum", "weight_decay", "amp", "num_workers"}
FEDERATION_KEYS = {"num_clients", "fraction_train", "min_train_nodes", "max_rounds", "timeout_seconds"}
EARLY_STOPPING_KEYS = {"enabled", "monitor", "min_delta", "patience_rounds", "warmup_rounds"}
LR_SCHEDULER_KEYS = {"enabled", "factor", "patience_rounds", "threshold", "min_lr"}
RUNTIME_KEYS = {"client_device", "server_device", "max_concurrent_clients", "cpus_per_client", "object_store_memory_mb", "startup_timeout_seconds"}
OUTPUT_KEYS = {"root", "progress", "evaluate_test_after_train"}
CHECKPOINT_KEYS = {"every_n_rounds", "keep_last_n"}
DIAGNOSTICS_KEYS = {
    "plateau_rounds", "divergence_window", "divergence_relative_increase",
    "low_recall_threshold", "small_client_samples", "update_norm_outlier_factor",
}


@dataclass
class ResolvedDataConfig:
    dataset_root: Path
    partition_index: Path
    partition_dir: Path
    scenario: str
    alpha: Optional[float]
    quantity_alpha: Optional[float]
    feature_skew: str
    split_seed: int
    partition_config_hash: str
    protocol_fingerprint: str
    class_names: List[str]
    bundle_path: Optional[Path] = None


@dataclass
class ResolvedModelConfig:
    name: str
    num_classes: int
    weights: Optional[str]


@dataclass
class ResolvedTrainingConfig:
    seed: int
    local_epochs: int
    batch_size: int
    eval_batch_size: int
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    amp: bool
    num_workers: int


@dataclass
class ResolvedFederationConfig:
    num_clients: int
    fraction_train: float
    min_train_nodes: int
    max_rounds: int
    timeout_seconds: int


@dataclass
class ResolvedEarlyStoppingConfig:
    enabled: bool
    monitor: str
    min_delta: float
    patience_rounds: int
    warmup_rounds: int


@dataclass
class ResolvedLRSchedulerConfig:
    enabled: bool
    factor: float
    patience_rounds: int
    threshold: float
    min_lr: float


@dataclass
class ResolvedRuntimeConfig:
    client_device: str  # "cuda" or "cpu"
    server_device: str  # "cpu"
    max_concurrent_clients: int
    cpus_per_client: int
    object_store_memory_mb: int = 256
    startup_timeout_seconds: int = 120


@dataclass
class ResolvedOutputConfig:
    root: Path
    run_dir: Path
    progress: bool
    evaluate_test_after_train: bool


@dataclass
class ResolvedCheckpointConfig:
    every_n_rounds: int
    keep_last_n: int


@dataclass
class ResolvedDiagnosticsConfig:
    plateau_rounds: int
    divergence_window: int
    divergence_relative_increase: float
    low_recall_threshold: float
    small_client_samples: int
    update_norm_outlier_factor: float


@dataclass
class TrainingConfig:
    schema_version: int
    data: ResolvedDataConfig
    model: ResolvedModelConfig
    training: ResolvedTrainingConfig
    federation: ResolvedFederationConfig
    early_stopping: ResolvedEarlyStoppingConfig
    lr_scheduler: ResolvedLRSchedulerConfig
    runtime: ResolvedRuntimeConfig
    output: ResolvedOutputConfig
    checkpoint: ResolvedCheckpointConfig
    diagnostics: ResolvedDiagnosticsConfig
    source_fingerprint: str
    semantic_config_hash: str
    run_id: str
    mode: str = "train"  # "train" or "smoke"
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert resolved configuration to serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "run_id": self.run_id,
            "source_fingerprint": self.source_fingerprint,
            "semantic_config_hash": self.semantic_config_hash,
            "data": {
                "dataset_root": str(self.data.dataset_root),
                "partition_index": str(self.data.partition_index),
                "partition_dir": str(self.data.partition_dir),
                "scenario": self.data.scenario,
                "alpha": self.data.alpha,
                "quantity_alpha": self.data.quantity_alpha,
                "feature_skew": self.data.feature_skew,
                "split_seed": self.data.split_seed,
                "partition_config_hash": self.data.partition_config_hash,
                "protocol_fingerprint": self.data.protocol_fingerprint,
                "class_names": self.data.class_names,
                "bundle_path": str(self.data.bundle_path) if self.data.bundle_path else None,
            },
            "model": asdict(self.model),
            "training": asdict(self.training),
            "federation": asdict(self.federation),
            "early_stopping": asdict(self.early_stopping),
            "lr_scheduler": asdict(self.lr_scheduler),
            "runtime": asdict(self.runtime),
            "output": {
                "root": str(self.output.root),
                "run_dir": str(self.output.run_dir),
                "progress": self.output.progress,
                "evaluate_test_after_train": self.output.evaluate_test_after_train,
            },
            "checkpoint": asdict(self.checkpoint),
            "diagnostics": asdict(self.diagnostics),
        }

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _check_keys(d: Dict[str, Any], allowed: set, section_name: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in section '{section_name}': {sorted(unknown)}")


def compute_source_fingerprint(package_root: Path | None = None) -> str:
    """Hash executable Python source used by training and evaluation."""
    root = (package_root or Path(__file__).resolve().parents[1]).resolve()
    digest = hashlib.sha256()
    for folder_name in ("fl_training", "src"):
        folder = root / folder_name
        for path in sorted(folder.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def compute_semantic_run_hash(
    raw_dict: Dict[str, Any], partition_hash: str, protocol_fingerprint: str = "",
    source_fingerprint: str = "",
) -> str:
    """Compute semantic hash of parameters affecting training outcome."""
    tr = raw_dict["training"]
    md = raw_dict["model"]
    fd = raw_dict["federation"]
    es = raw_dict["early_stopping"]
    ls = raw_dict["lr_scheduler"]
    rt = raw_dict["runtime"]

    canonical = {
        "partition_hash": partition_hash,
        "protocol_fingerprint": protocol_fingerprint,
        "source_fingerprint": source_fingerprint,
        "model_name": md["name"],
        "num_classes": md["num_classes"],
        "weights": md["weights"],
        "training_seed": tr["seed"],
        "local_epochs": tr["local_epochs"],
        "batch_size": tr["batch_size"],
        "eval_batch_size": tr["eval_batch_size"],
        "optimizer": tr["optimizer"],
        "lr": tr["lr"],
        "momentum": tr["momentum"],
        "weight_decay": tr["weight_decay"],
        "amp": tr["amp"],
        "num_clients": fd["num_clients"],
        "fraction_train": fd["fraction_train"],
        "early_stopping_enabled": es["enabled"],
        "early_stopping_patience": es["patience_rounds"] if es["enabled"] else None,
        "early_stopping_min_delta": es["min_delta"] if es["enabled"] else None,
        "early_stopping_warmup": es["warmup_rounds"] if es["enabled"] else None,
        "lr_scheduler_enabled": ls["enabled"],
        "lr_scheduler_factor": ls["factor"] if ls["enabled"] else None,
        "lr_scheduler_patience": ls["patience_rounds"] if ls["enabled"] else None,
        "lr_scheduler_threshold": ls["threshold"] if ls["enabled"] else None,
        "lr_scheduler_min_lr": ls["min_lr"] if ls["enabled"] else None,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_protocol_fingerprint(partition_dir: Path) -> str:
    """Hash manifests and metadata that define samples, classes, and preprocessing."""
    digest = hashlib.sha256()
    paths = [
        partition_dir / "partition_config.json",
        partition_dir / "fedavg_meta.json",
        partition_dir / "image_content.json",
        partition_dir / "centralized_train.csv",
        partition_dir / "global_val.csv",
        partition_dir / "global_test.csv",
        *sorted((partition_dir / "clients").glob("client_*.csv")),
    ]
    found = False

    def stable_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: stable_json(item)
                for key, item in value.items()
                if key not in {"created_at", "library_versions"}
            }
        if isinstance(value, list):
            return [stable_json(item) for item in value]
        return value

    for path in paths:
        if not path.exists():
            continue
        found = True
        digest.update(path.relative_to(partition_dir).as_posix().encode("utf-8"))
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as file:
                canonical = json.dumps(
                    stable_json(json.load(file)), sort_keys=True,
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
            digest.update(canonical)
        else:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest() if found else "unresolved"


def load_training_config(
    config_path: str | Path,
    base_dir: Optional[str | Path] = None,
    mode: str = "train",
    run_id_override: Optional[str] = None,
) -> TrainingConfig:
    """
    Parse, validate, and resolve paths and devices for a training YAML contract.
    """
    cfg_file = Path(config_path).resolve()
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    pkg_root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[1]

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a dictionary")

    _check_keys(raw, KNOWN_SECTIONS, "root")

    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {schema_version}, expected 1")

    # Validate sections
    d_raw = raw.get("data", {})
    m_raw = raw.get("model", {})
    t_raw = raw.get("training", {})
    f_raw = raw.get("federation", {})
    es_raw = raw.get("early_stopping", {})
    ls_raw = raw.get("lr_scheduler", {})
    rt_raw = raw.get("runtime", {})
    out_raw = raw.get("output", {})
    cp_raw = raw.get("checkpoint", {})
    diag_raw = raw.get("diagnostics", {})

    _check_keys(d_raw, DATA_KEYS, "data")
    _check_keys(m_raw, MODEL_KEYS, "model")
    _check_keys(t_raw, TRAINING_KEYS, "training")
    _check_keys(f_raw, FEDERATION_KEYS, "federation")
    _check_keys(es_raw, EARLY_STOPPING_KEYS, "early_stopping")
    _check_keys(ls_raw, LR_SCHEDULER_KEYS, "lr_scheduler")
    _check_keys(rt_raw, RUNTIME_KEYS, "runtime")
    _check_keys(out_raw, OUTPUT_KEYS, "output")
    _check_keys(cp_raw, CHECKPOINT_KEYS, "checkpoint")
    _check_keys(diag_raw, DIAGNOSTICS_KEYS, "diagnostics")
    # Reject silent int()/bool() coercion (e.g. 2.5 clients or "false" -> True).
    integer_fields = ((t_raw, ("seed", "local_epochs", "batch_size", "eval_batch_size", "num_workers")),
                      (f_raw, ("num_clients", "min_train_nodes", "max_rounds", "timeout_seconds")),
                      (rt_raw, ("max_concurrent_clients", "cpus_per_client", "object_store_memory_mb", "startup_timeout_seconds")),
                      (es_raw, ("patience_rounds", "warmup_rounds")),
                      (ls_raw, ("patience_rounds",)), (cp_raw, ("every_n_rounds", "keep_last_n")))
    for section, fields in integer_fields:
        for key in fields:
            if key in section and (isinstance(section[key], bool) or not isinstance(section[key], int)):
                raise ValueError(f"{key} must be an integer")
    for section, fields in ((t_raw, ("amp",)), (es_raw, ("enabled",)), (ls_raw, ("enabled",)),
                            (out_raw, ("progress", "evaluate_test_after_train"))):
        for key in fields:
            if key in section and not isinstance(section[key], bool):
                raise ValueError(f"{key} must be a YAML boolean")

    # Model validation
    if m_raw.get("name") != "mobilenet_v3_small":
        raise ValueError(f"Unsupported model name: {m_raw.get('name')}")
    num_classes = int(m_raw.get("num_classes", 38))
    if num_classes != 38:
        raise ValueError(f"num_classes must be 38, got {num_classes}")
    weights = m_raw.get("weights")
    if weights not in ("IMAGENET1K_V1", None):
        raise ValueError(f"weights must be 'IMAGENET1K_V1' or null, got {weights}")

    # Training validation
    seed = int(t_raw.get("seed", 42))
    local_epochs = int(t_raw.get("local_epochs", 1))
    batch_size = int(t_raw.get("batch_size", 16))
    eval_batch_size = int(t_raw.get("eval_batch_size", 32))
    opt_name = str(t_raw.get("optimizer", "sgd")).lower()
    if opt_name != "sgd":
        raise ValueError(f"Only 'sgd' optimizer supported in baseline, got {opt_name}")
    lr = float(t_raw.get("lr", 0.01))
    momentum = float(t_raw.get("momentum", 0.9))
    weight_decay = float(t_raw.get("weight_decay", 0.0001))
    req_amp = bool(t_raw.get("amp", True))
    num_workers = int(t_raw.get("num_workers", 0))

    if local_epochs < 1 or batch_size < 1 or eval_batch_size < 1:
        raise ValueError("local_epochs, batch_size, and eval_batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if not all(math.isfinite(v) for v in (lr, momentum, weight_decay)):
        raise ValueError("lr, momentum, and weight_decay must be finite")
    if lr <= 0 or not (0 <= momentum < 1) or weight_decay < 0:
        raise ValueError("lr must be > 0, momentum and weight_decay >= 0")

    # Federation validation
    num_clients = int(f_raw.get("num_clients", 10))
    fraction_train = float(f_raw.get("fraction_train", 1.0))
    min_train_nodes = int(f_raw.get("min_train_nodes", num_clients))
    max_rounds = int(f_raw.get("max_rounds", 100))
    timeout_seconds = int(f_raw.get("timeout_seconds", 3600))
    if num_clients < 1 or min_train_nodes < 1 or max_rounds < 1 or timeout_seconds < 1:
        raise ValueError("num_clients, min_train_nodes, max_rounds must be >= 1")
    if not math.isfinite(fraction_train) or not (0 < fraction_train <= 1.0):
        raise ValueError("fraction_train must be in (0, 1.0]")
    selected_clients = max(1, math.ceil(num_clients * fraction_train))
    if min_train_nodes > selected_clients:
        raise ValueError(
            f"min_train_nodes={min_train_nodes} exceeds clients selected per round={selected_clients}"
        )

    # Early stopping validation
    es_enabled = bool(es_raw.get("enabled", True))
    es_monitor = str(es_raw.get("monitor", "val_loss"))
    if es_monitor != "val_loss":
        raise ValueError(f"early_stopping monitor must be 'val_loss', got {es_monitor}")
    es_min_delta = float(es_raw.get("min_delta", 0.0001))
    es_patience = int(es_raw.get("patience_rounds", 10))
    es_warmup = int(es_raw.get("warmup_rounds", 10))
    if not math.isfinite(es_min_delta) or es_min_delta < 0 or es_patience < 1 or es_warmup < 0:
        raise ValueError("early_stopping requires finite min_delta >= 0, patience >= 1, warmup >= 0")

    # LR scheduler validation
    ls_enabled = bool(ls_raw.get("enabled", True))
    ls_factor = float(ls_raw.get("factor", 0.5))
    ls_patience = int(ls_raw.get("patience_rounds", 3))
    ls_threshold = float(ls_raw.get("threshold", 0.0001))
    ls_min_lr = float(ls_raw.get("min_lr", 0.000001))
    if not all(math.isfinite(v) for v in (ls_factor, ls_threshold, ls_min_lr)):
        raise ValueError("lr_scheduler numeric values must be finite")
    if not (0 < ls_factor < 1) or ls_patience < 0 or ls_threshold < 0 or not (0 <= ls_min_lr <= lr):
        raise ValueError("lr_scheduler requires 0 < factor < 1, patience/threshold >= 0, 0 <= min_lr <= lr")

    # Runtime and device validation
    req_client_dev = str(rt_raw.get("client_device", "auto")).lower()
    server_dev = str(rt_raw.get("server_device", "cpu")).lower()
    max_concurrent = int(rt_raw.get("max_concurrent_clients", 1))
    cpus_per_client = int(rt_raw.get("cpus_per_client", 2))
    object_store_memory_mb = int(rt_raw.get("object_store_memory_mb", 256))
    startup_timeout_seconds = int(rt_raw.get("startup_timeout_seconds", 120))
    if object_store_memory_mb < 80 or startup_timeout_seconds < 1:
        raise ValueError("Ray object_store_memory_mb must be >= 80 and startup_timeout_seconds >= 1")
    if max_concurrent < 1 or max_concurrent > num_clients or cpus_per_client < 1:
        raise ValueError("runtime concurrency must be in [1, num_clients] and cpus_per_client >= 1")
    if max_concurrent * cpus_per_client > (os.cpu_count() or 1):
        raise ValueError("runtime requests more client CPUs than available")

    if req_client_dev == "auto":
        resolved_client_dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif req_client_dev in ("cuda", "cpu"):
        if req_client_dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA client_device requested but torch.cuda.is_available() is False")
        resolved_client_dev = req_client_dev
    else:
        raise ValueError(f"Unknown client_device: {req_client_dev}")

    if resolved_client_dev == "cuda" and max_concurrent != 1:
        raise ValueError(
            "CUDA simulation currently supports max_concurrent_clients=1 on the single-GPU baseline"
        )

    if server_dev != "cpu":
        raise ValueError("server_device must be cpu in this implementation")

    resolved_amp = req_amp and (resolved_client_dev == "cuda")

    # Data path resolution
    d_root = Path(d_raw.get("dataset_root", "../PlantVillage-Dataset/raw/color"))
    dataset_root = (pkg_root / d_root).resolve() if not d_root.is_absolute() else d_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    bundle_path_raw = d_raw.get("bundle_path")
    bundle_path = None
    if bundle_path_raw:
        bp = Path(bundle_path_raw)
        bundle_path = (pkg_root / bp).resolve() if not bp.is_absolute() else bp

    part_idx_path = Path(d_raw.get("partition_index", "data/partitions_train_v1/index.json"))
    partition_index = (pkg_root / part_idx_path).resolve() if not part_idx_path.is_absolute() else part_idx_path

    scenario = str(d_raw.get("scenario", "label_skew"))
    alpha = float(d_raw["alpha"]) if d_raw.get("alpha") is not None else None
    qalpha = float(d_raw["quantity_alpha"]) if d_raw.get("quantity_alpha") is not None else None
    fskew = str(d_raw.get("feature_skew", "none"))
    split_seed = int(d_raw.get("split_seed", 42))
    for name, value in (("alpha", alpha), ("quantity_alpha", qalpha)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")

    partition_hash = ""
    partition_dir = Path("")

    if bundle_path and bundle_path.exists():
        partition_dir = bundle_path
        partition_hash = f"bundle::{bundle_path.name}"
    elif partition_index.exists():
        with open(partition_index, "r", encoding="utf-8") as f:
            idx_data = json.load(f)

        # Lookup by scenario and parameters
        found_hash = None
        for h, pinfo in idx_data.get("partitions", {}).items():
            if (
                pinfo.get("scenario") == scenario
                and pinfo.get("feature_skew") == fskew
                and pinfo.get("split_seed") == split_seed
            ):
                if alpha is not None and abs(pinfo.get("alpha", 0.0) - alpha) > 1e-6:
                    continue
                if qalpha is not None and abs(pinfo.get("quantity_alpha", 0.0) - qalpha) > 1e-6:
                    continue
                found_hash = h
                break

        if not found_hash:
            raise FileNotFoundError(
                f"Partition for scenario='{scenario}', alpha={alpha}, feature_skew='{fskew}', seed={split_seed} "
                f"not found in {partition_index}. Please run 'python -m fl_training.cli prepare-data' first."
            )

        partition_hash = found_hash
        rel_dir = idx_data["partitions"][found_hash]["relative_dir"]
        partition_dir = pkg_root / rel_dir
    else:
        # If partition index not yet created, allow placeholder for pre-preparation checks
        partition_hash = "unresolved_index"
        partition_dir = pkg_root / "data" / "partitions_train_v1"

    manifest_count = len(list((partition_dir / "clients").glob("client_[0-9][0-9].csv")))
    if partition_dir.exists() and manifest_count != num_clients:
        raise ValueError(
            f"Configured num_clients={num_clients}, but partition has {manifest_count} client manifests"
        )
    partition_metadata = partition_dir / "partition_config.json"
    if partition_metadata.exists():
        metadata = json.loads(partition_metadata.read_text(encoding="utf-8"))
        if metadata.get("feature_skew", "none") != fskew:
            raise ValueError("Requested feature_skew differs from stored partition profiles")

    protocol_fingerprint = compute_protocol_fingerprint(partition_dir)
    class_names: List[str] = []
    for metadata_name in ("partition_config.json", "fedavg_meta.json"):
        metadata_path = partition_dir / metadata_name
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                class_names = list(json.load(f).get("class_names", []))
            if class_names:
                break
    if class_names and len(class_names) != num_classes:
        raise ValueError(f"Partition class mapping has {len(class_names)} classes, expected {num_classes}")

    # Semantic run hash and run id
    source_fingerprint = compute_source_fingerprint(pkg_root)
    semantic_hash = compute_semantic_run_hash(
        raw, partition_hash, protocol_fingerprint, source_fingerprint,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if run_id_override:
        run_id = run_id_override
    else:
        run_id = f"{timestamp}__{scenario}__s{split_seed}_t{seed}__{semantic_hash[:8]}__{uuid.uuid4().hex[:8]}"

    out_root = Path(out_raw.get("root", "runs/fedavg"))
    output_root = (pkg_root / out_root).resolve() if not out_root.is_absolute() else out_root
    run_dir = output_root / run_id

    resolved_data = ResolvedDataConfig(
        dataset_root=dataset_root,
        partition_index=partition_index,
        partition_dir=partition_dir,
        scenario=scenario,
        alpha=alpha,
        quantity_alpha=qalpha,
        feature_skew=fskew,
        split_seed=split_seed,
        partition_config_hash=partition_hash,
        protocol_fingerprint=protocol_fingerprint,
        class_names=class_names or [str(i) for i in range(num_classes)],
        bundle_path=bundle_path,
    )

    resolved_model = ResolvedModelConfig(
        name="mobilenet_v3_small",
        num_classes=num_classes,
        weights=weights,
    )

    resolved_training = ResolvedTrainingConfig(
        seed=seed,
        local_epochs=local_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        optimizer=opt_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        amp=resolved_amp,
        num_workers=num_workers,
    )

    resolved_federation = ResolvedFederationConfig(
        num_clients=num_clients,
        fraction_train=fraction_train,
        min_train_nodes=min_train_nodes,
        max_rounds=max_rounds,
        timeout_seconds=timeout_seconds,
    )

    resolved_early_stopping = ResolvedEarlyStoppingConfig(
        enabled=es_enabled,
        monitor=es_monitor,
        min_delta=es_min_delta,
        patience_rounds=es_patience,
        warmup_rounds=es_warmup,
    )

    resolved_lr_scheduler = ResolvedLRSchedulerConfig(
        enabled=ls_enabled,
        factor=ls_factor,
        patience_rounds=ls_patience,
        threshold=ls_threshold,
        min_lr=ls_min_lr,
    )

    resolved_runtime = ResolvedRuntimeConfig(
        client_device=resolved_client_dev,
        server_device=server_dev,
        max_concurrent_clients=max_concurrent,
        cpus_per_client=cpus_per_client,
        object_store_memory_mb=object_store_memory_mb,
        startup_timeout_seconds=startup_timeout_seconds,
    )

    resolved_output = ResolvedOutputConfig(
        root=output_root,
        run_dir=run_dir,
        progress=bool(out_raw.get("progress", True)),
        evaluate_test_after_train=bool(out_raw.get("evaluate_test_after_train", True)),
    )
    every_n_rounds = int(cp_raw.get("every_n_rounds", 5))
    keep_last_n = int(cp_raw.get("keep_last_n", 3))
    if every_n_rounds < 1 or keep_last_n < 1:
        raise ValueError("checkpoint.every_n_rounds and keep_last_n must be >= 1")
    resolved_checkpoint = ResolvedCheckpointConfig(every_n_rounds, keep_last_n)
    diag_values = ResolvedDiagnosticsConfig(
        plateau_rounds=int(diag_raw.get("plateau_rounds", 3)),
        divergence_window=int(diag_raw.get("divergence_window", 3)),
        divergence_relative_increase=float(diag_raw.get("divergence_relative_increase", 0.02)),
        low_recall_threshold=float(diag_raw.get("low_recall_threshold", 0.5)),
        small_client_samples=int(diag_raw.get("small_client_samples", 50)),
        update_norm_outlier_factor=float(diag_raw.get("update_norm_outlier_factor", 5.0)),
    )
    if (
        diag_values.plateau_rounds < 1 or diag_values.divergence_window < 3
        or not math.isfinite(diag_values.divergence_relative_increase)
        or diag_values.divergence_relative_increase <= 0
        or not 0 <= diag_values.low_recall_threshold <= 1
        or diag_values.small_client_samples < 1
        or not math.isfinite(diag_values.update_norm_outlier_factor)
        or diag_values.update_norm_outlier_factor <= 1
    ):
        raise ValueError("diagnostics thresholds are outside their supported finite ranges")

    return TrainingConfig(
        schema_version=schema_version,
        data=resolved_data,
        model=resolved_model,
        training=resolved_training,
        federation=resolved_federation,
        early_stopping=resolved_early_stopping,
        lr_scheduler=resolved_lr_scheduler,
        runtime=resolved_runtime,
        output=resolved_output,
        checkpoint=resolved_checkpoint,
        diagnostics=diag_values,
        source_fingerprint=source_fingerprint,
        semantic_config_hash=semantic_hash,
        run_id=run_id,
        mode=mode,
        raw_dict=raw,
    )

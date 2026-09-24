"""
fl_training.checkpoint: Atomic saving and safe resumption of training states.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def model_state_sha256(state: Dict[str, torch.Tensor]) -> str:
    """Fingerprint every named tensor, including dtype, shape and BN buffers."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states in a safe, serializable format."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and state["torch_cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


def save_checkpoint_atomic(payload: Dict[str, Any], filepath: Path) -> None:
    """
    Atomic write using temporary file in the same directory, followed by os.replace.
    Ensures no partially-written checkpoint files exist upon interruption.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_file = filepath.parent / f"{filepath.name}.tmp_{os.getpid()}_{time.time_ns()}"
    try:
        torch.save(payload, temp_file)
        os.replace(temp_file, filepath)
    except Exception:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise


def save_round_checkpoint(
    run_dir: Path,
    round_num: int,
    model_state_dict: Dict[str, torch.Tensor],
    best_model_state_dict: Dict[str, torch.Tensor],
    best_round: int,
    best_loss: float,
    next_lr: float,
    early_stopping_state: Dict[str, Any],
    lr_scheduler_state: Dict[str, Any],
    history: List[Dict[str, Any]],
    resolved_config: Dict[str, Any],
    is_best: bool = False,
    extra_metrics: Optional[Dict[str, Any]] = None,
    checkpoint_every_n_rounds: int = 0,
    keep_last_n: int = 3,
    parent_run_id: Optional[str] = None,
    attempt_id: int = 1,
    optimizer_state: Optional[Dict[str, Any]] = None,
    amp_scaler_state: Optional[Dict[str, Any]] = None,
    amp_scaler_policy: str = "reset_each_train_local_call",
    best_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Save atomic last.pt and conditionally best.pt:
    - last.pt contains the committed state of round_num, including best_model_state_dict
      so best can be restored if needed.
    - best.pt is saved atomically when is_best is True.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all state dict tensors are CPU clones
    cpu_state = {k: v.detach().cpu().clone() for k, v in model_state_dict.items()}
    cpu_best_state = {k: v.detach().cpu().clone() for k, v in best_model_state_dict.items()}

    last_payload = {
        "format_version": 2,
        "round": int(round_num),
        "model_state_dict": cpu_state,
        "best_model_state_dict": cpu_best_state,
        "best_round": int(best_round),
        "best_loss": float(best_loss),
        "next_lr": float(next_lr),
        "early_stopping_state": early_stopping_state,
        "lr_scheduler_state": lr_scheduler_state,
        "history": history,
        "config": resolved_config,
        "semantic_config_hash": resolved_config.get("semantic_config_hash", ""),
        "rng_state": capture_rng_state(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra_metrics": extra_metrics or {},
        "best_metrics": copy.deepcopy(best_metrics if best_metrics is not None else (extra_metrics if is_best else {})),
        "parent_run_id": parent_run_id,
        "attempt_id": int(attempt_id),
    }

    if optimizer_state is not None:
        last_payload["optimizer_state"] = copy.deepcopy(optimizer_state)
    last_payload["amp_scaler_state"] = copy.deepcopy(amp_scaler_state)
    last_payload["amp_scaler_policy"] = str(amp_scaler_policy)
    last_path = run_dir / "last.pt"
    save_checkpoint_atomic(last_payload, last_path)

    best_path = None
    if is_best:
        best_payload = {
            "format_version": 2,
            "best_round": int(best_round),
            "best_loss": float(best_loss),
            "model_state_dict": cpu_best_state,
            "config": resolved_config,
            "semantic_config_hash": resolved_config.get("semantic_config_hash", ""),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": extra_metrics or {},
        }
        best_path = run_dir / "best.pt"
        save_checkpoint_atomic(best_payload, best_path)

    if checkpoint_every_n_rounds and round_num > 0 and round_num % checkpoint_every_n_rounds == 0:
        snapshot = run_dir / f"round_{round_num:04d}.pt"
        save_checkpoint_atomic(last_payload, snapshot)
        managed = sorted(
            (p for p in run_dir.glob("round_*.pt") if re.fullmatch(r"round_\d{4,}\.pt", p.name)),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for stale in managed[:-max(1, int(keep_last_n))]:
            stale.unlink()

    return last_path, best_path


def materialize_best_checkpoint(checkpoint_data: Dict[str, Any], run_dir: Path) -> Path:
    """Create best.pt for a continuation even if no later round improves."""
    best_path = Path(run_dir) / "best.pt"
    if best_path.exists():
        return best_path
    best_payload = {
        "format_version": 2,
        "best_round": int(checkpoint_data["best_round"]),
        "best_loss": float(checkpoint_data["best_loss"]),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in checkpoint_data["best_model_state_dict"].items()
        },
        "config": checkpoint_data.get("config", {}),
        "semantic_config_hash": checkpoint_data.get("semantic_config_hash", ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": copy.deepcopy(checkpoint_data.get("best_metrics", {})),
    }
    save_checkpoint_atomic(best_payload, best_path)
    return best_path


def save_inference_model(
    run_dir: Path,
    best_model_state_dict: Dict[str, torch.Tensor],
    resolved_config: Dict[str, Any],
    best_round: int,
    best_loss: float,
) -> Path:
    payload = {
        "format_version": 1,
        "artifact_type": "inference_model",
        "config": resolved_config,
        "semantic_config_hash": resolved_config.get("semantic_config_hash"),
        "model_state_sha256": model_state_sha256(best_model_state_dict),
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in best_model_state_dict.items()
        },
        "model": resolved_config.get("model", {}),
        "class_names": resolved_config.get("data", {}).get("class_names", []),
        "preprocessing": {
            "input_size": [224, 224],
            "normalization": "ImageNet",
            "evaluation": "resize-short-side-256-center-crop-224",
        },
        "training_seed": resolved_config.get("training", {}).get("seed"),
        "partition_config_hash": resolved_config.get("data", {}).get("partition_config_hash"),
        "protocol_fingerprint": resolved_config.get("data", {}).get("protocol_fingerprint"),
        "best_round": int(best_round),
        "best_val_loss": float(best_loss),
    }
    path = Path(run_dir) / "model_final.pt"
    save_checkpoint_atomic(payload, path)
    return path


def load_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint with torch.load."""
    cp_path = Path(checkpoint_path).resolve()
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {cp_path}")
    return torch.load(cp_path, map_location=map_location, weights_only=False)


def verify_checkpoint_compatibility(
    checkpoint_data: Dict[str, Any],
    current_config: Dict[str, Any],
) -> None:
    """
    Verify that checkpoint is compatible with current configuration to resume.
    Partition, model, training seed, and data fingerprints must match.
    """
    cp_hash = checkpoint_data.get("semantic_config_hash")
    curr_hash = current_config.get("semantic_config_hash")

    if not cp_hash or not curr_hash:
        raise ValueError("Cannot resume: checkpoint and current config must contain semantic config hashes")
    if cp_hash == curr_hash:
        cp_source = checkpoint_data.get("config", {}).get("source_fingerprint")
        current_source = current_config.get("source_fingerprint")
        if current_source and cp_source != current_source:
            raise ValueError("Cannot resume: source fingerprint mismatch or missing checkpoint provenance")
        cp_config = checkpoint_data.get("config", {})
        cp_mode, current_mode = cp_config.get("mode"), current_config.get("mode")
        if cp_mode and current_mode and cp_mode != current_mode:
            raise ValueError("Cannot resume: training mode mismatch")
        cp_rounds = cp_config.get("federation", {}).get("max_rounds")
        current_rounds = current_config.get("federation", {}).get("max_rounds")
        if cp_rounds is not None and current_rounds is not None and cp_rounds != current_rounds:
            raise ValueError("Cannot resume: fixed round budget mismatch")
        return

    # Version-1 checkpoints used a narrower hash.  Permit relocation only when
    # all explicitly recorded protocol fields still match; paths/output and an
    # increased max_rounds are operational overrides.
    cp_cfg = checkpoint_data.get("config", {})
    required = (
        ("data", "partition_config_hash"),
        ("model", "name"), ("model", "num_classes"), ("model", "weights"),
        ("training", "seed"), ("training", "local_epochs"),
        ("training", "batch_size"), ("training", "optimizer"),
        ("training", "lr"), ("training", "momentum"), ("training", "weight_decay"),
        ("federation", "num_clients"), ("federation", "fraction_train"),
    )
    is_legacy = int(checkpoint_data.get("format_version", 1)) < 2
    if is_legacy and cp_cfg and all(
        section in cp_cfg and section in current_config
        and key in cp_cfg[section] and key in current_config[section]
        and cp_cfg[section][key] == current_config[section][key]
        for section, key in required
    ):
        warnings.warn(
            "Resuming a legacy format-1 checkpoint using explicit protocol fields; "
            "that checkpoint predates the content fingerprint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    raise ValueError(
        f"Cannot resume: semantic config hash mismatch! Checkpoint hash: {cp_hash}, "
        f"Current hash: {curr_hash}. Partition, model, batch size, optimizer, and seed must match."
    )

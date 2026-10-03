"""Self-contained Kaggle Runner for Mixed FedAvg Campaign (pv_pd_v1).
Strictly adheres to campaign_spec.json protocol:
  - Local client optimizer: SGD (lr=0.01) per campaign_spec.hyperparameters.fedavg
  - FedAvg client local state: Freshly initialized per round with server weights
  - Aggregation: canonical sample count (n_k) weighting
  - Canonical W0 fingerprint: compute_model_w0_fingerprint from plant_data_contract
  - Sample-weighted loss accumulation in evaluation
  - Evaluates BEST checkpoint on post-training diagnostic test
  - Fail-closed guards: identity checks, RNG restoration, fresh-overwrite prevention
"""

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

def find_workspace_root(custom_workspace: Optional[str] = None) -> Path:
    """Find repository root dynamically across local laptop, Colab, or Kaggle."""
    if custom_workspace:
        p = Path(custom_workspace).resolve()
        if p.is_dir():
            return p
    env_root = os.environ.get("WORKSPACE_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()
    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir, script_dir.parent, script_dir.parent.parent]:
        if (parent / "dataset").is_dir() or (parent / "plant_data_contract").is_dir():
            return parent.resolve()
    cwd = Path.cwd().resolve()
    for parent in [cwd, cwd.parent]:
        if (parent / "dataset").is_dir():
            return parent.resolve()
    return cwd


WORKSPACE_ROOT = find_workspace_root()
if (WORKSPACE_ROOT / "plant_data_contract").is_dir() and str(WORKSPACE_ROOT / "plant_data_contract") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "plant_data_contract"))
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# Try to import plant_data_contract
try:
    from plant_data_contract.dataset import (
        CanonicalClassificationDataset,
        build_domain_balanced_sampler,
    )
    from plant_data_contract.transforms import (
        get_transforms,
        get_transform_metadata,
    )
    from plant_data_contract.models import (
        create_model,
        find_pretrained_weights_file,
        PINNED_PRETRAINED_FILE_SHA256,
        PINNED_PRETRAINED_BASE_STATE_SHA256,
    )
    from plant_data_contract.integrity import (
        canonical_json_dumps,
        compute_file_sha256,
        compute_model_w0_fingerprint,
        EXPECTED_RELEASE_MANIFEST_SHA256,
        verify_release_integrity,
        verify_manifest_records_contract,
        verify_image_dataset_integrity,
        compute_selected_partition_digest,
    )
    from plant_data_contract.partitions import verify_client_partitions, verify_partition_directory, verify_partition_audit
    from plant_data_contract.resume_guard import (
        validate_config_identity,
        validate_and_restore_rng,
        import_and_validate_best_checkpoint,
        check_fresh_overwrite_guard,
        save_atomic_checkpoint,
        save_crash_safe_checkpoint,
        load_verified_checkpoint,
        load_verified_best_checkpoint,
        compute_job_protocol_sha256,
        IMMUTABLE_PROTOCOL_KEYS,
    )
except ImportError:
    import subprocess
    whl_candidates = list(Path(".").glob("**/plant_data_contract*.whl")) + list(KAGGLE_INPUT_DIR.glob("**/plant_data_contract*.whl"))
    if whl_candidates:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", str(whl_candidates[0])])
    else:
        sys.path.insert(0, str(WORKSPACE_ROOT / "plant_data_contract"))
    from plant_data_contract.dataset import (
        CanonicalClassificationDataset,
        build_domain_balanced_sampler,
    )
    from plant_data_contract.transforms import (
        get_transforms,
        get_transform_metadata,
    )
    from plant_data_contract.models import (
        create_model,
        find_pretrained_weights_file,
        PINNED_PRETRAINED_FILE_SHA256,
        PINNED_PRETRAINED_BASE_STATE_SHA256,
    )
    from plant_data_contract.integrity import (
        canonical_json_dumps,
        compute_file_sha256,
        compute_model_w0_fingerprint,
        EXPECTED_RELEASE_MANIFEST_SHA256,
        verify_release_integrity,
        verify_manifest_records_contract,
        verify_image_dataset_integrity,
        compute_selected_partition_digest,
    )
    from plant_data_contract.partitions import verify_client_partitions, verify_partition_directory, verify_partition_audit
    from plant_data_contract.resume_guard import (
        validate_config_identity,
        validate_and_restore_rng,
        import_and_validate_best_checkpoint,
        check_fresh_overwrite_guard,
        save_atomic_checkpoint,
        save_crash_safe_checkpoint,
        load_verified_checkpoint,
        load_verified_best_checkpoint,
        compute_job_protocol_sha256,
        IMMUTABLE_PROTOCOL_KEYS,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fedavg_runner")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


from plant_data_contract.training_progress import TrainingProgress


def approved_w0_kwargs(spec, args):
    """Resolve the pilot W0 state beside the approved ImageNet weights."""
    if (spec.get("dataset_release") != "dataset/mixed/pv_pd_v3" or
            (spec.get("status") != "READY_FULL" and not str(spec.get("spec_version", "")).startswith("4."))
            or args.init != "imagenet_v1"):
        return {}
    name = spec.get("pretrained_w0_state_file")
    digest = spec.get("pretrained_w0_state_sha256")
    if not name or Path(name).name != name or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("READY_FULL pretrained v3 requires a pinned W0 state file and SHA-256")
    if not args.pretrained_weights:
        raise ValueError("READY_FULL pretrained v3 requires --pretrained-weights")
    return {"w0_state_path": Path(args.pretrained_weights).resolve().with_name(name),
            "expected_w0_state_sha256": digest}


def enforce_approved_v3_protocol(spec, args, w0_fingerprint, resolved_config):
    """Reject a full v3 run whose effective initialization differs from its pilot approval."""
    if args.mode != "full" or spec.get("dataset_release") != "dataset/mixed/pv_pd_v3":
        return
    if spec.get("status") != "READY_FULL":
        if args.preflight_only:
            return
        raise ValueError("Full v3 training requires a READY_FULL pilot-approved spec")
    variant = spec.get("selected_variant")
    expected_init = "scratch" if variant == "scratch" else "imagenet_v1"
    expected_ratio = 0.25 if variant == "pretrained_pd25" else 0.0
    if variant not in ("scratch", "pretrained", "pretrained_pd25"):
        raise ValueError("READY_FULL spec has an invalid selected_variant")
    raw_ratio = str(args.plantdoc_sampling_ratio).lower()
    ratio = 0.0 if raw_ratio in ("none", "false", "") else float(raw_ratio)
    expected_repeat_policy = spec.get("sampler_repeat_policy", "with_replacement")
    if (args.init != expected_init or abs(ratio - expected_ratio) > 1e-9 or
            args.transform_version != spec.get("preprocessing") or
            getattr(args, "sampler_repeat_policy", "with_replacement") != expected_repeat_policy):
        raise ValueError("Full v3 init/sampler/transform differs from approved pilot")
    if w0_fingerprint != spec.get("expected_w0_fingerprint"):
        raise ValueError("Full v3 W0 fingerprint differs from approved pilot")
    if expected_init == "imagenet_v1" and (resolved_config.get("pretrained_file_sha256")
                                           != spec.get("pretrained_weights_sha256")):
        raise ValueError("Full v3 pretrained weights SHA differs from approved spec")
    if resolved_config.get("release_manifest_sha256") != spec.get("release_manifest_sha256"):
        raise ValueError("Full v3 release SHA differs from approved spec")
from plant_data_contract.quality_metrics import summarize_quality
from plant_data_contract.full_gate import verify_full_data_gate
from plant_data_contract.quality_control import QualityControl, DEFAULT_CONVERGENCE


def create_runner_model(
    num_classes: int = 38,
    seed: Optional[int] = None,
    init: str = "scratch",
    pretrained_weights_path: Optional[str] = None,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    return create_model(
        num_classes=num_classes,
        seed=seed if seed is not None else 42,
        init=init,
        pretrained_weights_path=pretrained_weights_path,
    )

def create_model_compat(num_classes: int = 38, seed: Optional[int] = None) -> Tuple[nn.Module, str]:
    m, fp, _ = create_runner_model(num_classes=num_classes, seed=seed, init="scratch")
    return m, fp


def model_from_global_state(state: Dict[str, torch.Tensor]) -> nn.Module:
    """Load a round model without reseeding or consuming the training RNG stream."""
    with torch.random.fork_rng(devices=[]):
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict(state, strict=True)
    return model


def eval_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metas = [item[2] for item in batch]
    return images, targets, metas


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate classification metrics with sample-weighted loss accumulation."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_domains = []
    all_confidence = []

    with torch.no_grad():
        for b_idx, (images, targets, metas) in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite evaluation loss at batch {b_idx + 1}")

            b_size = len(targets)
            total_loss += float(loss.item()) * b_size
            total_samples += b_size

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_confidence.extend(torch.softmax(outputs, dim=1).amax(dim=1).cpu().numpy())
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_domains.extend([m.get("source_domain", "unknown") for m in metas])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_domains = np.array(all_domains)

    avg_loss = total_loss / max(1, total_samples)
    acc_overall = float((all_preds == all_targets).mean()) if len(all_targets) > 0 else 0.0

    # Macro-F1 across 38 classes (unrounded) and supported classes
    f1_scores = []
    f1_scores_supported = []
    present_classes = set(all_targets.tolist()) if len(all_targets) > 0 else set()
    for cls_id in range(38):
        tp = np.sum((all_preds == cls_id) & (all_targets == cls_id))
        fp = np.sum((all_preds == cls_id) & (all_targets != cls_id))
        fn = np.sum((all_preds != cls_id) & (all_targets == cls_id))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f)
        if cls_id in present_classes:
            f1_scores_supported.append(f)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    macro_f1_supported = float(np.mean(f1_scores_supported)) if f1_scores_supported else 0.0
    per_class_recall = {
        str(cls_id): {
            "support": int(np.sum(all_targets == cls_id)),
            "recall": (float(np.mean(all_preds[all_targets == cls_id] == cls_id))
                       if np.any(all_targets == cls_id) else None),
        }
        for cls_id in range(38)
    }

    # Per-source breakdown
    source_metrics = {}
    for dom in ["plantvillage", "plantdoc"]:
        mask = (all_domains == dom)
        if np.any(mask):
            sub_preds = all_preds[mask]
            sub_targets = all_targets[mask]
            sub_acc = float((sub_preds == sub_targets).mean())
            sub_present = set(sub_targets.tolist())
            sub_f1s = []
            sub_f1s_supported = []
            for cls_id in range(38):
                tp = np.sum((sub_preds == cls_id) & (sub_targets == cls_id))
                fp = np.sum((sub_preds == cls_id) & (sub_targets != cls_id))
                fn = np.sum((sub_preds != cls_id) & (sub_targets == cls_id))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                sub_f1s.append(f)
                if cls_id in sub_present:
                    sub_f1s_supported.append(f)
            source_metrics[dom] = {
                "count": int(np.sum(mask)),
                "per_class_recall": {
                    str(cls_id): {
                        "support": int(np.sum(sub_targets == cls_id)),
                        "recall": (float(np.mean(sub_preds[sub_targets == cls_id] == cls_id))
                                   if np.any(sub_targets == cls_id) else None),
                    }
                    for cls_id in range(38)
                },
                "accuracy": round(sub_acc, 4),
                "macro_f1": round(float(np.mean(sub_f1s)), 4),
                "raw_accuracy": sub_acc,
                "raw_macro_f1": float(np.mean(sub_f1s)),
                "macro_f1_supported": round(float(np.mean(sub_f1s_supported)), 4) if sub_f1s_supported else 0.0,
                "raw_macro_f1_supported": float(np.mean(sub_f1s_supported)) if sub_f1s_supported else 0.0,
            }

    return {
        "loss": round(avg_loss, 4),
        "raw_loss": avg_loss,
        "accuracy": round(acc_overall, 4),
        "raw_accuracy": acc_overall,
        "macro_f1": round(macro_f1, 4),
        "raw_macro_f1": macro_f1,
        "macro_f1_supported": round(macro_f1_supported, 4),
        "raw_macro_f1_supported": macro_f1_supported,
        "total_samples": total_samples,
        "per_class_recall": per_class_recall,
        "per_source": source_metrics,
        "quality": summarize_quality(all_targets, all_preds, all_domains, all_confidence),
    }


def find_dataset_root(
    custom_dataset_root: Optional[str] = None,
    custom_release_dir: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    release_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Locate canonical dataset root and release directory.
    If explicit custom paths are provided and invalid, raises immediately (fail-closed).
    """
    if custom_release_dir:
        rel = Path(custom_release_dir).resolve()
        if (rel / "release_manifest.json").is_file():
            ds_root = Path(custom_dataset_root).resolve() if custom_dataset_root else rel.parent.parent
            return ds_root, rel
        raise FileNotFoundError(f"Explicit release directory does not contain release_manifest.json: {rel}")

    rel_names = [release_name] if release_name else ["pv_pd_v2", "pv_pd_v1"]

    if custom_dataset_root:
        ds_root = Path(custom_dataset_root).resolve()
        for rname in rel_names:
            rel = ds_root / f"mixed/{rname}"
            if (rel / "release_manifest.json").is_file():
                return ds_root, rel
            rel_flat = ds_root / rname
            if (rel_flat / "release_manifest.json").is_file():
                return ds_root, rel_flat
        raise FileNotFoundError(f"Explicit dataset root does not contain release_manifest.json under {rel_names}: {ds_root}")

    ws = workspace_root or find_workspace_root()
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("/kaggle/working/dataset"),
        KAGGLE_INPUT_DIR / "plantvillage-plantdoc-canonical/dataset",
        KAGGLE_INPUT_DIR / "plantvillage-plantdoc-canonical",
        ws / "dataset",
        script_dir.parent.parent.parent / "dataset",
        Path("./dataset"),
        Path("../dataset"),
        Path("/content/dataset"),
    ]
    for c in candidates:
        for rname in rel_names:
            rel = c / f"mixed/{rname}"
            if rel.is_dir() and (rel / "release_manifest.json").is_file():
                return c.resolve(), rel.resolve()
            rel_flat = c / rname
            if rel_flat.is_dir() and (rel_flat / "release_manifest.json").is_file():
                return c.resolve(), rel_flat.resolve()

    raise FileNotFoundError(f"Could not locate canonical dataset containing {rel_names} in any candidate path.")



def aggregate_weights(
    client_weights: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
    base_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """FedAvg parameter aggregation weighted by client canonical sample count (n_k).
    - Float tensors & running stats: weighted average using normalized float64 accumulation.
    - Integer BatchNorm buffers (e.g. num_batches_tracked): base + sum(local - base).
    """
    if not client_weights or len(client_weights) != len(client_sample_counts):
        raise ValueError("Client states and sample counts must have the same nonzero length")
    total_samples = sum(client_sample_counts)
    if total_samples <= 0:
        raise ValueError(f"Total samples must be positive, got {total_samples}")

    for idx, s in enumerate(client_sample_counts):
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"Client {idx} sample count {s!r} must be an integer, got {type(s).__name__}")
        if s <= 0:
            raise ValueError(f"All client sample counts must be strictly positive, got {s} at index {idx}")

    p_k = [float(s) / float(total_samples) for s in client_sample_counts]
    first_weights = client_weights[0]
    base = base_weights or first_weights
    aggregated = {}

    for key, base_val in base.items():
        client_vals = [w[key] for w in client_weights]
        if any(value.shape != base_val.shape or value.dtype != base_val.dtype for value in client_vals):
            raise ValueError(f"Client tensor shape/dtype mismatch: {key}")

        if not base_val.is_floating_point():
            # Integer buffer (e.g. num_batches_tracked): base + sum(local - base)
            base_int = base_val.to(dtype=torch.int64)
            deltas = []
            for c_val in client_vals:
                diff = c_val.to(dtype=torch.int64) - base_int
                if (diff < 0).any():
                    raise ValueError(f"BatchNorm integer counter moved backwards: {key}")
                deltas.append(diff)
            new_counter = base_int + sum(deltas)
            aggregated[key] = new_counter.to(dtype=base_val.dtype)
        else:
            # Float weights & running stats: float64 weighted accumulation
            weighted_sum = torch.zeros_like(base_val, dtype=torch.float64)
            for p, c_val in zip(p_k, client_vals):
                if not torch.isfinite(c_val).all():
                    raise ValueError(f"Non-finite values detected in parameter {key}")
                weighted_sum += c_val.to(dtype=torch.float64) * p
            aggregated[key] = weighted_sum.to(dtype=base_val.dtype)

    return aggregated


def state_fingerprint(weights: Dict[str, torch.Tensor]) -> str:
    """Content fingerprint of an ordered, CPU model state."""
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_fedavg_round(before, clients, counts, after) -> None:
    """Verify weighted floats and additive integer counters for a committed round."""
    if len(clients) != len(counts) or not clients or any(n <= 0 for n in counts):
        raise ValueError("Invalid client weights or sample counts")
    total = sum(counts)
    probabilities = [n / total for n in counts]
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise ValueError("FedAvg client probabilities do not sum to one")
    for key, previous in before.items():
        actual = after[key]
        if previous.is_floating_point():
            expected = sum(cl[key].to(torch.float64) * n for cl, n in zip(clients, counts)) / total
            if not torch.allclose(actual.to(torch.float64), expected, rtol=1e-5, atol=1e-6):
                raise ValueError(f"FedAvg aggregation mismatch: {key}")
        else:
            base = previous.to(torch.int64)
            expected = base.clone()
            for client in clients:
                delta = client[key].to(torch.int64) - base
                if (delta < 0).any():
                    raise ValueError(f"BatchNorm integer counter moved backwards: {key}")
                expected += delta
            if not torch.equal(actual, expected.to(dtype=previous.dtype)):
                raise ValueError(f"FedAvg integer buffer mismatch: {key}")


def scaler_states(scalers, client_ids):
    if set(scalers) != set(client_ids):
        raise ValueError("AMP scaler client IDs do not match partition client IDs")
    return {cid: scalers[cid].state_dict() for cid in client_ids}


def restore_scaler_states(scalers, client_ids, states):
    if not isinstance(states, dict) or set(states) != set(client_ids) or set(scalers) != set(client_ids):
        raise ValueError("Resume AMP scaler client IDs do not match partition client IDs")
    for cid in client_ids:
        scalers[cid].load_state_dict(states[cid])


def write_committed_telemetry(output_dir: Path, history: List[Dict[str, Any]]) -> None:
    """Rebuild JSONL from checkpointed rounds, so resume cannot duplicate events."""
    target = output_dir / "fedavg_telemetry_events.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    seen = set()
    with temporary.open("w", encoding="utf-8") as stream:
        for record in history:
            for event in record.get("telemetry_events", []):
                key = (event["run_id"], event["round"], event["event_type"], event.get("client_id"))
                if key in seen:
                    raise ValueError(f"Duplicate telemetry event in checkpoint history: {key}")
                seen.add(key)
                stream.write(canonical_json_dumps(event) + "\n")
    os.replace(temporary, target)



def train_client(
    client_id: str,
    global_state: Dict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    lr: float = 0.01,
    epochs: int = 1,
    amp: bool = False,
    weight_decay: float = 1e-4,
    momentum: float = 0.0,
    init: str = "scratch",
    pretrained_weights_path: Optional[str] = None,
    max_batches: Optional[int] = None,
    progress_callback=None,
    scaler: Optional[GradScaler] = None,
    raw_sample_count: Optional[int] = None,
    clip_grad_norm: Optional[float] = None,
    domain_sampler_policy: Optional[str] = "standard",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Local client training simulation with SGD and optional AMP.
    Client models are stateless across rounds in standard FedAvg: a fresh client model is
    instantiated each round with the downloaded server weights; the server aggregates parameter
    tensors and does not maintain client optimizer states.
    """
    t_load_start = time.perf_counter_ns()
    model = model_from_global_state(global_state)
    model.to(device)
    client_load_ms = (time.perf_counter_ns() - t_load_start) / 1e6

    t_comp_start = time.perf_counter_ns()
    model.train()

    # Protocol optimizer: SGD with learning rate, momentum, and weight_decay from campaign_spec
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if scaler is None:
        scaler = GradScaler("cuda", enabled=bool(amp and torch.cuda.is_available()), init_scale=2048.0)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    steps = 0
    steps_applied = 0
    steps_skipped = 0
    samples_seen = 0
    unique_samples_seen = set()

    for ep in range(epochs):
        if domain_sampler_policy and "cycle_without_replacement" in domain_sampler_policy:
            if loader.sampler is not None and hasattr(loader.sampler, "set_epoch") and ep > 0:
                loader.sampler.set_epoch(loader.sampler.epoch + 1)
        for batch in loader:
            if max_batches and steps >= max_batches:
                break
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, targets, metas = batch
                for m in metas:
                    if isinstance(m, dict) and "sample_id" in m:
                        unique_samples_seen.add(m["sample_id"])
            else:
                images, targets = batch[0], batch[1]

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast("cuda", enabled=bool(amp and torch.cuda.is_available())):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite train loss for {client_id} at local epoch {ep + 1}")

            scaler.scale(loss).backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
            steps_skipped += int(step_was_skipped)
            steps_applied += int(not step_was_skipped)

            b_len = len(targets)
            total_loss += float(loss.item()) * b_len
            steps += 1
            samples_seen += b_len
            if progress_callback is not None:
                progress_callback(total_loss / samples_seen)

    if steps_applied == 0:
        raise RuntimeError(f"No optimizer step was applied for {client_id}")
    avg_loss = total_loss / max(1, samples_seen)
    client_compute_wall_ms = (time.perf_counter_ns() - t_comp_start) / 1e6

    t_ser_start = time.perf_counter_ns()
    cpu_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    client_serialize_ms = (time.perf_counter_ns() - t_ser_start) / 1e6

    # Compute delta norm and maximum absolute parameter change + per layer norms
    delta_sq = 0.0
    delta_max_abs = 0.0
    delta_sum_abs = 0.0
    delta_min_abs = float("inf")
    num_params = 0
    param_bytes = 0
    layer_delta_norms = {}
    for k, v in cpu_weights.items():
        num_params += v.numel()
        param_bytes += v.numel() * v.element_size()
        if v.is_floating_point() and k in global_state:
            diff = v - global_state[k]
            diff_f64 = diff.to(dtype=torch.float64)
            norm_layer = float(torch.linalg.norm(diff_f64).item())
            layer_delta_norms[k] = round(norm_layer, 6)
            delta_sq += float((diff_f64 ** 2).sum().item())
            diff_abs = diff.abs()
            m = float(diff_abs.max().item())
            if m > delta_max_abs:
                delta_max_abs = m
            mn = float(diff_abs.min().item())
            if mn < delta_min_abs:
                delta_min_abs = mn
            delta_sum_abs += float(diff_abs.sum().item())
    delta_l2 = float(np.sqrt(delta_sq))
    delta_mean_abs = float(delta_sum_abs / max(1, num_params))
    if delta_min_abs == float("inf"):
        delta_min_abs = 0.0
    model_fp = compute_model_w0_fingerprint(model)

    telemetry = {
        "client_id": client_id,
        "loss": round(avg_loss, 4),
        "raw_images_n_k": raw_sample_count or len(loader.dataset),
        "samples_seen": samples_seen,
        "unique_samples_seen": len(unique_samples_seen) if unique_samples_seen else samples_seen,
        "domain_sampler_policy": domain_sampler_policy,
        "steps": steps,
        "optimizer_steps_attempted": steps,
        "optimizer_steps_applied": steps_applied,
        "optimizer_steps_skipped": steps_skipped,
        "delta_l2": round(delta_l2, 6),
        "delta_max_abs": round(delta_max_abs, 6),
        "delta_mean_abs": round(delta_mean_abs, 6),
        "delta_min_abs": round(delta_min_abs, 6),
        "layer_delta_norms": layer_delta_norms,
        "w_client_after_fingerprint": model_fp,
        "num_params": num_params,
        "param_bytes": param_bytes,
        "payload_up_bytes": param_bytes,
        "load_ms": round(client_load_ms, 2),
        "compute_wall_ms": round(client_compute_wall_ms, 2),
        "serialize_ms": round(client_serialize_ms, 2),
        "transport_mode": "in_process_simulation",
        "network_rtt_ms": None,
        "network_one_way_ms": None,
        "simulation_note": "Sequential in-process simulation; zero simulated network transit time; not actual network latency",
    }
    return cpu_weights, telemetry


class ProtocolConfigDict(dict):
    """Configuration dictionary that recomputes canonical protocol_content_sha256 upon modification."""
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key != "protocol_content_sha256" and key in IMMUTABLE_PROTOCOL_KEYS:
            super().__setitem__("protocol_content_sha256", compute_job_protocol_sha256(self))


def resolve_fedavg_config(
    spec: Dict[str, Any],
    job_id: str,
    spec_path: Path,
    release_manifest_sha: str,
    w0_fingerprint: str,
    release_dir: Optional[Path] = None,
    rounds_override: Optional[int] = None,
    mode: str = "full",
    init: str = "scratch",
    pretrained_weights_path: Optional[str] = None,
    plantdoc_sampling_ratio: str = "none",
    sampler_repeat_policy: str = "with_replacement",
    transform_version: str = "canonical_v1",
    incumbent_summary_path: Optional[str] = None,
    incumbent_checkpoint_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build resolved config dictionary and locate job configuration in campaign_spec.
    Derives canonical protocol_content_sha256 from the effective immutable parameters.
    Location metadata (spec_file_path) and raw spec file hash (campaign_spec_sha256)
    are retained for audit provenance only.
    """
    fedavg_hparams = spec.get("hyperparameters", {}).get("fedavg", {})

    # 1. Enforce supported aggregation policy
    agg_name = fedavg_hparams.get("aggregation", "FedAvg")
    if agg_name not in ["FedAvg", "fedavg", "weighted_fedavg", "weighted_average"]:
        raise ValueError(f"Unsupported FedAvg aggregation '{agg_name}'. Supported aggregation: 'FedAvg'")

    # 2. Enforce SGD optimizer
    client_optimizer_name = fedavg_hparams.get("client_optimizer", "SGD")
    if client_optimizer_name.upper() != "SGD":
        raise ValueError(f"Unsupported client_optimizer '{client_optimizer_name}'. Protocol requires SGD.")

    client_lr = fedavg_hparams.get("client_lr", 0.01)
    if not isinstance(client_lr, (int, float)) or client_lr <= 0 or not np.isfinite(client_lr):
        raise ValueError(f"Invalid client_lr: {client_lr}")

    client_wd = fedavg_hparams.get("client_weight_decay", 1e-4)
    client_mom = fedavg_hparams.get("client_momentum", 0.0)
    amp = fedavg_hparams.get("amp", False)
    spec_rounds = fedavg_hparams.get("rounds", 15)
    if mode == "pilot":
        num_rounds = 1
    elif mode == "smoke":
        num_rounds = rounds_override or 2
    else:  # full
        if rounds_override is not None and rounds_override != spec_rounds:
            raise ValueError(
                f"Full mode protocol requires {spec_rounds} rounds (got rounds={rounds_override}). "
                f"Use --stop-after-round {rounds_override} for session budgeting or --mode pilot/smoke."
            )
        num_rounds = spec_rounds
    local_epochs = fedavg_hparams.get("client_epochs", 1)
    batch_size = fedavg_hparams.get("client_batch_size", 32)
    expected_clients = fedavg_hparams.get("clients_per_round", 5)
    if type(expected_clients) is not int or not 2 <= expected_clients <= 100:
        raise ValueError("FedAvg clients_per_round must be an integer from 2 to 100")
    if mode == "full" and (not 1 <= num_rounds <= 60 or local_epochs != 1):
        raise ValueError("Full FedAvg requires at most 60 rounds and one local epoch")

    job_cfg = None
    for j in spec.get("kaggle_fedavg_jobs", []):
        if j.get("job_id") == job_id:
            job_cfg = j
            break

    if not job_cfg:
        for j in spec.get("jobs", []):
            if j.get("job_id") == job_id:
                job_cfg = j
                break

    if not job_cfg:
        raise ValueError(f"Job ID '{job_id}' not found in campaign_spec.json")

    gate = spec.get("quality_gate", {})
    policy = QualityControl(0.0, policy=gate.get("convergence", {})).policy
    if mode == "full" and policy["min_step"] > num_rounds:
        raise ValueError(f"Convergence min_step={policy['min_step']} exceeds full rounds={num_rounds}")
    floor = gate.get("plantvillage_floor_by_seed", {}).get(str(job_cfg["seed"]))
    gate_status = "QUALITY_GATE_UNVERIFIED"
    if mode == "smoke":
        gate_status = "SMOKE_ONLY"
    elif mode == "full" and str(spec.get("spec_version", "")).startswith("4."):
        baseline = gate.get("full_incumbent_baseline")
        if not isinstance(baseline, dict):
            raise ValueError("QUALITY_GATE_UNVERIFIED: full incumbent baseline is required")
        required = ("release_manifest_sha256", "seed", "w0_fingerprint", "checkpoint_sha256", "plantvillage_macro_f1_supported")
        if any(k not in baseline for k in required):
            raise ValueError("QUALITY_GATE_UNVERIFIED: incomplete full incumbent baseline")
        metric = baseline["plantvillage_macro_f1_supported"]
        if (baseline["release_manifest_sha256"] != release_manifest_sha or baseline["seed"] != job_cfg["seed"]
                or baseline["w0_fingerprint"] != w0_fingerprint or not isinstance(baseline["checkpoint_sha256"], str)
                or len(baseline["checkpoint_sha256"]) != 64 or not isinstance(metric, (int, float))
                or not np.isfinite(metric) or not 0 < metric <= 1 or floor is None
                or floor < metric - 0.01):
            raise ValueError("QUALITY_GATE_UNVERIFIED: baseline provenance or PlantVillage floor mismatch")
        if not incumbent_summary_path or not incumbent_checkpoint_path:
            raise ValueError("QUALITY_GATE_UNVERIFIED: full incumbent summary and checkpoint paths are required")
        summary_path = Path(incumbent_summary_path)
        checkpoint_path = Path(incumbent_checkpoint_path)
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("Full incumbent summary/checkpoint is missing")
        if hashlib.sha256(summary_path.read_bytes()).hexdigest() != baseline.get("summary_sha256"):
            raise ValueError("QUALITY_GATE_UNVERIFIED: incumbent summary SHA mismatch")
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != baseline["checkpoint_sha256"]:
            raise ValueError("QUALITY_GATE_UNVERIFIED: incumbent checkpoint SHA mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_metric = summary.get("best_checkpoint_validation", {}).get("per_source", {}).get(
            "plantvillage", {}).get("raw_macro_f1_supported")
        if (summary.get("mode") != "full" or summary.get("seed") != baseline["seed"]
                or summary.get("release_manifest_sha256") != release_manifest_sha
                or summary.get("w0_fingerprint") != w0_fingerprint
                or summary.get("best_checkpoint_sha256") != baseline["checkpoint_sha256"]
                or summary_metric != metric):
            raise ValueError("QUALITY_GATE_UNVERIFIED: incumbent summary provenance mismatch")
        gate_status = "PASS"

    if release_dir:
        rel_dir = Path(release_dir)
    else:
        ws = find_workspace_root()
        rel_dir = None
        for rname in ["pv_pd_v2", "pv_pd_v1"]:
            cand = ws / f"dataset/mixed/{rname}"
            if (cand / "release_manifest.json").is_file():
                rel_dir = cand
                break
        if rel_dir is None:
            rel_dir = Path("dataset/mixed/pv_pd_v2") if Path("dataset/mixed/pv_pd_v2/release_manifest.json").is_file() else Path("dataset/mixed/pv_pd_v1")
    selected_part_digest = compute_selected_partition_digest(rel_dir, job_cfg["partition_scheme"])

    pretrained_file_sha = None
    if init == "imagenet_v1":
        wfile = find_pretrained_weights_file(pretrained_weights_path)
        pretrained_file_sha = hashlib.sha256(wfile.read_bytes()).hexdigest()

    effective_params = {
        "job_id": job_id,
        "mode": mode,
        "condition": "mixed",
        "partition_scheme": job_cfg["partition_scheme"],
        "seed": job_cfg["seed"],
        "rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "client_lr": client_lr,
        "client_optimizer": client_optimizer_name,
        "client_weight_decay": client_wd,
        "client_momentum": client_mom,
        "amp": bool(amp),
        "aggregation": agg_name,
        "expected_clients": expected_clients,
        "w0_fingerprint": w0_fingerprint,
        "release_manifest_sha256": release_manifest_sha,
        "init": init,
        "architecture_version": f"mobilenet_v3_small_{init}_v1",
        "preprocessing_version": transform_version,
        "plantdoc_sampling_ratio": str(plantdoc_sampling_ratio),
        "sampler_repeat_policy": sampler_repeat_policy,
        "pretrained_file_sha256": pretrained_file_sha,
        "scheduler_policy": "PlantDoc_plateau_factor0.5_patience4_abs0.002_min15",
        "selected_partition_digest": selected_part_digest,
        "quality_floor": floor,
        "quality_gate_status": gate_status,
        "quality_audit_sha256": spec.get("quality_gate", {}).get("audit_sha256"),
        "convergence_policy": policy,
    }
    protocol_content_sha = compute_job_protocol_sha256(effective_params)

    spec_bytes = spec_path.read_bytes() if (spec_path and Path(spec_path).is_file()) else b""
    raw_spec_sha256 = hashlib.sha256(spec_bytes).hexdigest() if spec_bytes else None

    resolved_config = ProtocolConfigDict(effective_params)
    resolved_config["protocol_content_sha256"] = protocol_content_sha
    resolved_config["campaign_spec_sha256"] = raw_spec_sha256  # Retained for audit provenance only
    resolved_config["spec_file_path"] = str(spec_path) if spec_path else None
    resolved_config["partition_seed"] = job_cfg.get("partition_seed", 42)
    resolved_config["training_seed"] = job_cfg["seed"]
    resolved_config["aggregation"] = {
        "float_policy": "float64_weighted_average",
        "int_buffer_policy": "base_plus_sum_deltas",
    }
    return resolved_config, job_cfg


def main():
    parser = argparse.ArgumentParser(description="FedAvg Mixed pv_pd_v1 Runner")
    parser.add_argument("--job-id", type=str, required=True, help="Job identifier from campaign_spec.json")
    parser.add_argument("--mode", choices=["full", "pilot", "smoke"], default="full", help="Execution mode (default: full)")
    parser.add_argument("--spec-file", type=str, default=None, help="Path to campaign_spec.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints and metrics")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint_last.pt to resume")
    parser.add_argument("--prior-best", type=str, default=None, help="Explicit path to prior checkpoint_best.pt when resuming")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output directory")
    parser.add_argument("--allow-legacy", action="store_true", default=False, help="Allow resuming legacy checkpoints")
    parser.add_argument("--incumbent-summary", type=str, default=None, help="Byte-verified full incumbent summary for v4 full quality gate")
    parser.add_argument("--incumbent-checkpoint", type=str, default=None, help="Byte-verified full incumbent best checkpoint for v4 full quality gate")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds (only valid in smoke mode)")
    parser.add_argument("--stop-after-round", type=int, default=None, help="Session budget: stop after completing this round and mark PAUSED_CHECKPOINTED")
    parser.add_argument("--workspace-root", type=str, default=None, help="Root path of workspace")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to canonical dataset root")
    parser.add_argument("--release-dir", type=str, default=None, help="Path to canonical release directory")
    parser.add_argument("--release-name", type=str, default=None, help="Release name under dataset root (e.g. pv_pd_v2, pv_pd_v1)")
    parser.add_argument("--expected-release-sha", type=str, default=None, help="Approved release manifest SHA-256")
    parser.add_argument("--preflight-only", action="store_true", default=False, help="Run integrity and configuration preflight checks without training or file mutation")
    parser.add_argument("--client-lr", type=float, default=None, help="Client learning rate")
    parser.add_argument("--client-optimizer", type=str, default=None, help="Client optimizer")
    parser.add_argument("--client-weight-decay", type=float, default=None, help="Client weight decay")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument("--batch-size", type=int, default=None, help="Client batch size")
    parser.add_argument("--local-epochs", type=int, default=None, help="Client local epochs")
    parser.add_argument("--init", choices=["scratch", "imagenet_v1"], default="scratch", help="Backbone initialization")
    parser.add_argument("--pretrained-weights", type=str, default=None, help="Explicit path to pretrained weights")
    parser.add_argument("--plantdoc-sampling-ratio", type=str, default="none", help="Domain balanced sampling ratio for PlantDoc")
    parser.add_argument("--sampler-repeat-policy", choices=["with_replacement", "cycle_without_replacement"], default="with_replacement", help="Domain sampler repeat policy; lower-repeat cycle mode is experimental")
    parser.add_argument("--transform-version", choices=["canonical_v1", "aspect_pad_v1", "aspect_pad_mean_v2", "moderate_augment_v1", "light_augment_v2"], default="canonical_v1", help="Preprocessing transform version")
    parser.add_argument("--full-image-verify", action="store_true", default=False, help="Perform 100%% byte SHA-256 and decode verification of all 50k+ images (slow)")
    args = parser.parse_args()

    explicit_flags = set()
    for arg_token in sys.argv[1:]:
        if arg_token.startswith("--"):
            explicit_flags.add(arg_token.split("=")[0])
        elif arg_token.startswith("-"):
            explicit_flags.add(arg_token.split("=")[0])
    args._explicit_flags = explicit_flags

    print("=" * 70)
    print(f"FEDAVG MIXED RUNNER: Job ID = {args.job_id}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load campaign specification & validate job configuration
    ws_root = find_workspace_root(args.workspace_root)
    script_dir = Path(__file__).resolve().parent
    spec_path = None
    if args.spec_file:
        p = Path(args.spec_file).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Explicitly specified --spec-file does not exist: {args.spec_file}")
        spec_path = p
    else:
        spec_candidates = [
            script_dir / "campaign_spec.json",
            Path("campaign_spec.json").resolve() if Path("campaign_spec.json").is_file() else None,
            (script_dir / "staging_bundle/campaign_spec.json").resolve() if (script_dir / "staging_bundle/campaign_spec.json").is_file() else None,
            ws_root / "plans/antigravity-kaggle-group-local-20260927/campaign_spec.json",
            KAGGLE_INPUT_DIR / "campaign_spec.json",
        ]
        for c in spec_candidates:
            if c and c.is_file():
                spec_path = c
                break

        if not spec_path:
            raise FileNotFoundError(
                f"Could not find campaign_spec.json. Checked: {[str(c) for c in spec_candidates if c]}"
            )

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    # Initial check that job exists in spec
    temp_job_cfg = None
    for j in spec.get("kaggle_fedavg_jobs", []):
        if j.get("job_id") == args.job_id:
            temp_job_cfg = j
            break
    if not temp_job_cfg:
        for j in spec.get("jobs", []):
            if j.get("job_id") == args.job_id:
                temp_job_cfg = j
                break
    if not temp_job_cfg:
        raise ValueError(f"Job ID '{args.job_id}' not found in campaign_spec.json")

    # Reject explicit CLI argument conflicts with spec (F6)
    fedavg_hparams = spec.get("hyperparameters", {}).get("fedavg", {})
    expected_wd = fedavg_hparams.get("client_weight_decay", 1e-4)
    if "--client-weight-decay" in explicit_flags and abs(float(args.client_weight_decay) - float(expected_wd)) > 1e-9:
        raise ValueError(f"CLI argument --client-weight-decay={args.client_weight_decay} conflicts with campaign_spec client_weight_decay={expected_wd}. Production spec is authoritative.")
    expected_amp = bool(fedavg_hparams.get("amp", False))
    if ("--amp" in explicit_flags or "--no-amp" in explicit_flags) and bool(args.amp) != expected_amp:
        raise ValueError(f"CLI argument --amp/--no-amp ({args.amp}) conflicts with campaign_spec amp={expected_amp}.")
    if "--client-lr" in explicit_flags and "client_lr" in fedavg_hparams and abs(args.client_lr - fedavg_hparams["client_lr"]) > 1e-9:
        raise ValueError(f"CLI argument --client-lr={args.client_lr} conflicts with campaign_spec client_lr={fedavg_hparams['client_lr']}. Production spec is authoritative.")
    if "--batch-size" in explicit_flags and "client_batch_size" in fedavg_hparams and args.batch_size != fedavg_hparams["client_batch_size"]:
        raise ValueError(f"CLI argument --batch-size={args.batch_size} conflicts with campaign_spec client_batch_size={fedavg_hparams['client_batch_size']}.")
    if "--local-epochs" in explicit_flags and "client_epochs" in fedavg_hparams and args.local_epochs != fedavg_hparams["client_epochs"]:
        raise ValueError(f"CLI argument --local-epochs={args.local_epochs} conflicts with campaign_spec client_epochs={fedavg_hparams['client_epochs']}.")
    if "--client-optimizer" in explicit_flags and "client_optimizer" in fedavg_hparams and args.client_optimizer.upper() != fedavg_hparams["client_optimizer"].upper():
        raise ValueError(f"CLI argument --client-optimizer={args.client_optimizer} conflicts with campaign_spec client_optimizer={fedavg_hparams['client_optimizer']}.")

    condition = "mixed"
    partition_scheme = temp_job_cfg["partition_scheme"]
    seed = temp_job_cfg["seed"]

    # 2. Locate dataset and release manifest & verify integrity
    dataset_root, release_dir = find_dataset_root(
        custom_dataset_root=args.dataset_root,
        custom_release_dir=args.release_dir,
        workspace_root=ws_root,
        release_name=args.release_name,
    )
    integrity_report = verify_release_integrity(
        release_dir,
        expected_release_sha=getattr(args, "expected_release_sha", None),
    )
    release_manifest_sha = integrity_report["release_manifest_sha256"]
    print(f"Dataset Root: {dataset_root}")
    print(f"Release Manifest SHA-256 Verified: {release_manifest_sha}")

    # Verify partition shards and records contract
    expected_clients = fedavg_hparams.get("clients_per_round", 5)
    client_ids = [f"client_{i:02d}" for i in range(expected_clients)]
    part_report = verify_partition_directory(release_dir, partition_scheme, client_ids)
    audit_report = verify_partition_audit(release_dir, partition_scheme, client_ids, release_manifest_sha)
    print(f"Client Partitions Verified: {partition_scheme}, {expected_clients} shards, "
          f"union = {part_report['total_samples']}, audit = {audit_report['status']}")

    manifest_contract = verify_manifest_records_contract(release_dir)
    print(f"Manifest Contract Verified: {len(manifest_contract['manifest_counts'])} splits")

    # Check paths and a small sample first, so an empty dataset reports a
    # concrete data error before the versioned quality gate is consulted.
    verify_all_images = getattr(args, "full_image_verify", False)
    sample_hash_count = None if verify_all_images else 100
    sample_decode_count = None if verify_all_images else 50
    image_report = verify_image_dataset_integrity(
        dataset_root=dataset_root,
        release_dir=release_dir,
        splits=["train", "val", "legacy_diagnostic_test"],
        sample_hash_count=sample_hash_count,
        sample_decode_count=sample_decode_count,
    )
    print(f"Image Dataset Checked: {image_report.get('records_checked', image_report.get('total_images_checked', 0))} paths, "
          f"{image_report.get('unique_files_hashed', 0)} byte hashes, "
          f"{image_report.get('unique_files_decoded', 0)} decodes ({image_report['status']})")
    if args.mode == "full" and (not args.preflight_only or spec.get("quality_gate")):
        verify_full_data_gate(release_dir, release_manifest_sha, spec)
    if (args.mode == "full" and not args.preflight_only and
            spec.get("dataset_release") == "dataset/mixed/pv_pd_v3" and
            spec.get("status") != "READY_FULL"):
        raise ValueError("Full v3 training requires a READY_FULL pilot-approved spec")
    if args.mode == "full" and not args.preflight_only:
        # The audit certifies its snapshot; verify current bytes again before
        # starting the full run, including calibration images.
        full_image_report = verify_image_dataset_integrity(
            dataset_root=dataset_root,
            release_dir=release_dir,
            splits=["train", "val", "calibration", "legacy_diagnostic_test"],
            sample_hash_count=None,
            sample_decode_count=None,
        )
        print(f"Full Image Verification: {full_image_report['unique_files_hashed']} byte hashes, "
              f"{full_image_report['unique_files_decoded']} decodes ({full_image_report['status']})")
    # 3. Model Initialization (Canonical W0)
    set_seed(seed)
    global_model, w0_fingerprint, model_metadata = create_model(
        num_classes=38,
        seed=seed,
        init=args.init,
        pretrained_weights_path=args.pretrained_weights,
        **approved_w0_kwargs(spec, args),
    )
    global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    print(f"Initialized Model: num_classes=38, init={args.init}, W0={w0_fingerprint}")

    # Build resolved configuration via canonical resolver
    resolved_config, job_cfg = resolve_fedavg_config(
        spec=spec,
        job_id=args.job_id,
        spec_path=spec_path,
        release_manifest_sha=release_manifest_sha,
        w0_fingerprint=w0_fingerprint,
        release_dir=release_dir,
        rounds_override=args.rounds,
        mode=args.mode,
        init=args.init,
        pretrained_weights_path=args.pretrained_weights,
        plantdoc_sampling_ratio=args.plantdoc_sampling_ratio,
        sampler_repeat_policy=args.sampler_repeat_policy,
        transform_version=args.transform_version,
        incumbent_summary_path=args.incumbent_summary,
        incumbent_checkpoint_path=args.incumbent_checkpoint,
    )
    enforce_approved_v3_protocol(spec, args, w0_fingerprint, resolved_config)

    if args.preflight_only:
        print("=" * 70)
        print("PREFLIGHT CHECK OK:")
        print(f"  Job ID: {args.job_id}")
        print(f"  Release Manifest SHA-256: {release_manifest_sha}")
        print(f"  Partition Scheme: {partition_scheme} ({expected_clients} shards verified, union={part_report['total_samples']})")
        print(f"  Manifests Verified: {len(manifest_contract['manifest_counts'])} splits")
        print(f"  Images: {image_report.get('records_checked', image_report.get('total_images_checked', 0))} paths, "
              f"{image_report.get('unique_files_hashed', 0)} hashes, "
              f"{image_report.get('unique_files_decoded', 0)} decodes ({image_report['status']})")
        print(f"  Init Mode: {args.init}")
        print(f"  Transform Version: {args.transform_version}")
        print(f"  Sampling Ratio: {args.plantdoc_sampling_ratio}")
        print(f"  Model W0 Fingerprint: {w0_fingerprint}")
        print(f"  Protocol Content SHA-256: {resolved_config['protocol_content_sha256']}")
        print("PREFLIGHT PASSED: Exiting cleanly with zero file mutation.")
        print("=" * 70)
        return

    if args.mode == "full" and not torch.cuda.is_available() and os.environ.get("ANTIGRAVITY_ALLOW_CPU_FULL") != "1":
        raise RuntimeError("CUDA is required for full mode training, but no CUDA GPU accelerator was detected.")

    output_dir = Path(args.output_dir or f"/kaggle/working/{args.job_id}/{args.mode}").resolve()

    # Fresh-overwrite guard: do not overwrite without explicit flag
    check_fresh_overwrite_guard(output_dir, is_resume=bool(args.resume), overwrite=args.overwrite)

    # Transforms from plant_data_contract
    train_transform = get_transforms(args.transform_version, is_train=True)
    eval_transform = get_transforms(args.transform_version, is_train=False)

    # Load client shards
    partition_dir = release_dir / f"partitions/{partition_scheme}"
    client_loaders = []
    client_sample_counts = []
    client_policies = {}
    batch_size = fedavg_hparams.get("client_batch_size", 32)
    client_ids = [f"client_{i:02d}" for i in range(expected_clients)]

    pd_sampling_str = str(args.plantdoc_sampling_ratio).lower()
    use_sampling = pd_sampling_str not in ("none", "0", "0.0", "", "false")

    def client_collate_fn(batch):
        return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), [b[2] for b in batch]

    for cid in client_ids:
        shard_path = partition_dir / f"{cid}.jsonl"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing partition shard: {shard_path}")
        cds = CanonicalClassificationDataset.from_manifest(
            shard_path, dataset_root=dataset_root, transform=train_transform, return_metadata=True
        )
        has_plantdoc = any(sample.get("source_domain") == "plantdoc" for sample in cds.samples)
        sampler_repeat_policy = getattr(args, "sampler_repeat_policy", "with_replacement")
        if use_sampling:
            if has_plantdoc:
                sampler = build_domain_balanced_sampler(
                    cds,
                    plantdoc_weight=float(pd_sampling_str),
                    seed=int(job_cfg["seed"]),
                    repeat_policy=sampler_repeat_policy,
                )
                policy = f"balanced_pd{pd_sampling_str}_{sampler_repeat_policy}"
            else:
                sampler = None
                policy = "uniform_fallback_no_plantdoc"
        else:
            sampler = None
            policy = "standard_uniform"
        client_policies[cid] = policy

        if sampler is not None:
            loader = DataLoader(cds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=client_collate_fn)
        else:
            loader = DataLoader(cds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=client_collate_fn)
        client_loaders.append(loader)
        original_ids = [sample["sample_id"] for sample in cds.samples]
        if len(set(original_ids)) != len(original_ids):
            raise ValueError(f"Duplicate original sample_id in {cid}; n_k cannot count sampler draws or repeated rows")
        client_sample_counts.append(len(original_ids))
        print(f"  [{cid}] Samples: {len(cds)} (Batches: {len(loader)}, Policy: {policy})")

    # Global Validation and Diagnostic Test loaders
    val_ds = CanonicalClassificationDataset.from_manifest(
        release_dir / "manifests/val.jsonl", dataset_root=dataset_root, transform=eval_transform, return_metadata=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)

    test_ds = CanonicalClassificationDataset.from_manifest(
        release_dir / "manifests/legacy_diagnostic_test.jsonl", dataset_root=dataset_root, transform=eval_transform, return_metadata=True
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)

    num_rounds = resolved_config["rounds"]
    local_epochs = resolved_config["local_epochs"]
    client_lr = resolved_config["client_lr"]
    quality_floor = resolved_config.get("quality_floor")
    if args.mode == "full" and quality_floor is None:
        raise ValueError("Full training requires a PlantVillage macro-F1 floor from the same-seed pilot")
    quality_control = QualityControl(quality_floor if quality_floor is not None else 0.0,
                                     policy=resolved_config["convergence_policy"])
    stop_requested = False

    start_round = 1
    best_val_f1_raw = -1.0
    history = []

    # 4. Strict Resume Handling: validate BEFORE creating or modifying output directory
    if args.resume:
        resume_target = Path(args.resume).resolve()
        if not resume_target.exists():
            raise FileNotFoundError(f"Resume checkpoint file not found: {resume_target}")

        resume_dir = resume_target.parent if resume_target.is_file() else resume_target
        ckpt, last_path, best_path = load_verified_checkpoint(
            output_dir=resume_dir,
            expected_config=resolved_config,
            is_fedavg=True,
            allow_legacy=args.allow_legacy,
        )

        validate_and_restore_rng(ckpt, allow_legacy=args.allow_legacy)

        prior_best_f1 = float(ckpt.get("best_val_f1_raw", ckpt.get("best_val_f1", -1.0)))
        best_round = ckpt.get("best_generation", ckpt.get("round", 0))

        if output_dir != resume_dir:
            best_path, prior_best_f1, best_round = import_and_validate_best_checkpoint(
                last_path,
                output_dir,
                resolved_config,
                is_fedavg=True,
                allow_legacy=args.allow_legacy,
                source_best_path_override=best_path,
            )

        global_weights = ckpt["model_state"]
        start_round = ckpt.get("round", 0) + 1
        best_val_f1_raw = prior_best_f1
        history = ckpt.get("history", [])
        if "quality_control_state" in ckpt:
            if ("effective_convergence_policy" in ckpt and
                    ckpt["effective_convergence_policy"] != resolved_config["convergence_policy"]):
                raise ValueError("Resume checkpoint convergence policy differs from current spec")
            quality_control = QualityControl(quality_floor if quality_floor is not None else 0.0,
                                             policy=resolved_config["convergence_policy"],
                                             state=ckpt["quality_control_state"])
            client_lr = float(ckpt["current_client_lr"])
        elif not args.allow_legacy:
            raise ValueError("Resume checkpoint lacks quality-control state")
        stop_requested = bool(ckpt.get("stop_requested", False))
        print(f"Resumed from {last_path} at round {start_round} (Prior Best Val F1: {best_val_f1_raw:.4f}, Best Round: {best_round})")

    # Only after validation passes, create output directory and save resolved configuration
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(canonical_json_dumps(resolved_config), encoding="utf-8")
    if args.resume:
        write_committed_telemetry(output_dir, history)
    print(f"Resolved Job Config: {resolved_config}")

    # 5. FedAvg Training Loop
    t_start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    session_paused = False
    last_completed_round = start_round - 1

    total_samples = sum(client_sample_counts)
    p_k = [float(s) / float(total_samples) for s in client_sample_counts]
    if abs(sum(p_k) - 1.0) > 1e-12:
        raise ValueError("FedAvg client probabilities do not sum to one")
    run_id = f"{args.job_id}:{resolved_config['protocol_content_sha256']}"

    max_client_batches = 5 if args.mode == "smoke" else None
    max_eval_batches = 5 if args.mode == "smoke" else None
    round_batches = sum(min(len(loader) * local_epochs, max_client_batches) if max_client_batches else len(loader) * local_epochs for loader in client_loaders)
    progress = TrainingProgress(num_rounds * round_batches)
    progress.completed = min(progress.total_steps, (start_round - 1) * round_batches)

    # Persistent client scalers across rounds to eliminate redundant AMP skipped steps
    client_scalers = {
        cid: GradScaler("cuda", enabled=bool(resolved_config["amp"] and torch.cuda.is_available()), init_scale=2048.0)
        for cid in client_ids
    }
    if args.resume:
        if "client_scaler_states" not in ckpt:
            if not args.allow_legacy:
                raise ValueError("Resume checkpoint lacks per-client AMP scaler states")
        else:
            restore_scaler_states(client_scalers, client_ids, ckpt["client_scaler_states"])

    for r in range(start_round, num_rounds + 1):
        if args.mode == "full" and stop_requested:
            logging.info("Convergence stop committed at round %s", r - 1)
            break
        last_completed_round = r
        t_r_start = time.perf_counter_ns()

        # Phase 1: Server prepare
        t_prep_start = time.perf_counter_ns()
        server_param_bytes = sum(v.numel() * v.element_size() for v in global_weights.values())
        server_prepare_ms = (time.perf_counter_ns() - t_prep_start) / 1e6
        server_before_weights = copy.deepcopy(global_weights)
        server_before_fingerprint = state_fingerprint(global_weights)

        client_weights_list = []
        client_telemetry_list = []
        telemetry_events = []

        for idx, (cid, loader, n_k) in enumerate(zip(client_ids, client_loaders, client_sample_counts)):
            if sampler_repeat_policy == "cycle_without_replacement":
                if loader.sampler is not None and hasattr(loader.sampler, "set_epoch"):
                    client_epoch = (r - 1) * len(client_ids) * local_epochs + idx * local_epochs
                    loader.sampler.set_epoch(client_epoch)
            c_weights, c_telemetry = train_client(
                client_id=cid,
                global_state=global_weights,
                loader=loader,
                device=device,
                epochs=local_epochs,
                lr=client_lr,
                amp=resolved_config["amp"],
                weight_decay=resolved_config["client_weight_decay"],
                momentum=resolved_config["client_momentum"],
                init=args.init,
                pretrained_weights_path=args.pretrained_weights,
                max_batches=max_client_batches,
                scaler=client_scalers[cid],
                raw_sample_count=n_k,
                domain_sampler_policy=client_policies.get(cid, "standard"),
                progress_callback=lambda loss, cid=cid: progress.update(
                    increment=1, phase="train", position=f"round {r}/{num_rounds} {cid}",
                    loss=loss, lr=client_lr,
                    vram_mb=torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
                ),
            )
            c_telemetry["weight_p_k"] = p_k[idx]
            c_telemetry["weight_p_k_display"] = round(p_k[idx], 6)
            c_telemetry["sample_count_n_k"] = n_k
            c_telemetry["payload_down_bytes"] = server_param_bytes
            client_weights_list.append(c_weights)
            client_telemetry_list.append(c_telemetry)

            # Record client telemetry event with unrounded float p_k
            client_event = {
                "run_id": run_id,
                "event_type": "client_round_execution",
                "round": r,
                "client_id": cid,
                "client_index": idx,
                "sample_count_n_k": n_k,
                "weight_p_k": p_k[idx],
                "transport_mode": "in_process_simulation",
                "network_rtt_ms": None,
                "network_one_way_ms": None,
                "simulation_note": "Sequential in-process simulation; zero simulated network transit time; not actual network latency",
                "payload_down_bytes": server_param_bytes,
                "payload_up_bytes": c_telemetry["payload_up_bytes"],
                "client_load_ms": c_telemetry["load_ms"],
                "client_compute_wall_ms": c_telemetry["compute_wall_ms"],
                "client_serialize_ms": c_telemetry["serialize_ms"],
                "train_loss": c_telemetry["loss"],
                "optimizer_steps_attempted": c_telemetry["optimizer_steps_attempted"],
                "optimizer_steps_applied": c_telemetry["optimizer_steps_applied"],
                "optimizer_steps_skipped": c_telemetry["optimizer_steps_skipped"],
                "delta_l2": c_telemetry["delta_l2"],
                "delta_max_abs": c_telemetry["delta_max_abs"],
                "delta_mean_abs": c_telemetry["delta_mean_abs"],
                "delta_min_abs": c_telemetry["delta_min_abs"],
                "layer_delta_norms": c_telemetry["layer_delta_norms"],
                "unique_samples_seen": c_telemetry["unique_samples_seen"],
                "raw_images_n_k": c_telemetry["raw_images_n_k"],
                "domain_sampler_policy": c_telemetry["domain_sampler_policy"],
                "w_client_fingerprint": c_telemetry["w_client_after_fingerprint"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            telemetry_events.append(client_event)

        # Server Aggregation with base_weights for integer BatchNorm buffer deltas
        t_agg_start = time.perf_counter_ns()
        next_weights = aggregate_weights(client_weights_list, client_sample_counts, base_weights=global_weights)
        verify_fedavg_round(global_weights, client_weights_list, client_sample_counts, next_weights)
        global_weights = next_weights
        server_after_fingerprint = state_fingerprint(global_weights)
        aggregate_ms = (time.perf_counter_ns() - t_agg_start) / 1e6

        # Global Validation
        eval_model = model_from_global_state(global_weights)
        eval_model.to(device)

        progress.update(phase="validate", position=f"round {r}/{num_rounds}", lr=client_lr)
        val_res = evaluate_metrics(eval_model, val_loader, device, max_batches=max_eval_batches)
        val_f1_raw = val_res["raw_macro_f1"]
        val_acc = val_res["accuracy"]
        pd_source = val_res.get("per_source", {}).get("plantdoc")
        pv_source = val_res.get("per_source", {}).get("plantvillage")
        if args.mode == "full" and (pd_source is None or pv_source is None):
            raise ValueError("Full validation must include both PlantDoc and PlantVillage")
        pd_f1 = pd_source["raw_macro_f1_supported"] if pd_source else 0.0
        pv_f1 = pv_source["raw_macro_f1_supported"] if pv_source else 0.0
        pd_crop = val_res["quality"]["by_source"]["plantdoc"]["crop_accuracy"] or 0.0
        decision = quality_control.observe(r, pd_f1, pv_f1, pd_crop)
        is_best = decision["is_best"]
        if is_best:
            best_val_f1_raw = pd_f1
        if decision["reduce_lr"] and args.mode == "full":
            client_lr *= quality_control.policy["lr_factor"]
        stop_requested = decision["stop"] if args.mode == "full" else False

        r_duration = (time.perf_counter_ns() - t_r_start) / 1e9
        round_wall_ms = r_duration * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Server round aggregation event with unrounded float p_k
        server_event = {
            "run_id": run_id,
            "event_type": "server_round_aggregation",
            "round": r,
            "total_clients": len(client_ids),
            "total_samples": total_samples,
            "client_weights_p_k": p_k,
            "client_weights_p_k_display": [round(p, 6) for p in p_k],
            "sum_p_k": float(sum(p_k)),
            "transport_mode": "in_process_simulation",
            "network_rtt_ms": None,
            "network_one_way_ms": None,
            "simulation_note": "Sequential in-process simulation; zero simulated network transit time; not actual network latency",
            "server_prepare_ms": round(server_prepare_ms, 2),
            "serialize_ms": None,
            "dispatch_ms": None,
            "receive_ms": None,
            "aggregate_ms": round(aggregate_ms, 2),
            "server_before_fingerprint": server_before_fingerprint,
            "server_after_fingerprint": server_after_fingerprint,
            "round_wall_ms": round(round_wall_ms, 2),
            "val_loss": val_res["loss"],
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1_raw,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        telemetry_events.append(server_event)

        round_record = {
            "round": r,
            "val_loss": val_res["loss"],
            "val_accuracy": val_acc,
            "val_macro_f1": round(val_f1_raw, 4),
            "val_macro_f1_raw": val_f1_raw,
            "plantdoc_macro_f1_supported": pd_f1,
            "plantvillage_macro_f1_supported": pv_f1,
            "plantdoc_crop_accuracy": pd_crop,
            "quality_eligible": decision["eligible"] if args.mode == "full" and resolved_config["quality_gate_status"] == "PASS" else False,
            "quality_gate_status": resolved_config["quality_gate_status"],
            "quality": val_res["quality"],
            "lr_next_round": client_lr,
            "duration_sec": round(r_duration, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "is_best": is_best,
            "client_telemetry": client_telemetry_list,
            "telemetry_events": telemetry_events,
        }
        history.append(round_record)

        # Gate D: Full snapshot of 5 clients + server at pilot (round 1), best, and final round
        if r == 1 or is_best or r == num_rounds:
            snapshot_file = output_dir / f"client_server_snapshots_round_{r:04d}.pt"
            snapshot_data = {
                "round": r,
                "is_best": is_best,
                "server_state_before": server_before_weights,
                "server_state_after": global_weights,
                "client_states": {cid: w for cid, w in zip(client_ids, client_weights_list)},
                "p_k": p_k,
                "n_k": client_sample_counts,
                "per_layer_delta_norms": {cid: t["layer_delta_norms"] for cid, t in zip(client_ids, client_telemetry_list)},
            }
            torch.save(snapshot_data, snapshot_file)

        progress.summary(
            f"Round {r:02d} Summary | Val F1: {val_f1_raw:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"VRAM: {peak_vram_mb:.1f}MB | Time: {r_duration:.1f}s | {'BEST' if is_best else ''}"
        )

        # Save crash-safe generation checkpoints
        progress.update(phase="checkpoint", position=f"round {r}/{num_rounds}", validation=val_res.get("per_source", {}).get("plantdoc", {}).get("raw_macro_f1_supported"), lr=client_lr, vram_mb=peak_vram_mb)
        checkpoint_state = {
            "round": r,
            "job_id": args.job_id,
            "condition": condition,
            "partition_scheme": partition_scheme,
            "seed": seed,
            "w0_fingerprint": w0_fingerprint,
            "release_manifest_sha256": release_manifest_sha,
            "resolved_config": resolved_config,
            "model_state": global_weights,
            "best_val_f1": round(best_val_f1_raw, 4),
            "best_val_f1_raw": best_val_f1_raw,
            "quality_control_state": quality_control.state_dict(),
            "effective_convergence_policy": dict(quality_control.policy),
            "client_scaler_states": scaler_states(client_scalers, client_ids),
            "current_client_lr": client_lr,
            "stop_requested": stop_requested,
            "history": history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        save_crash_safe_checkpoint(
            output_dir=output_dir,
            state=checkpoint_state,
            step=r,
            is_best=is_best,
            is_fedavg=True,
        )
        write_committed_telemetry(output_dir, history)

        # Check session budget
        if args.stop_after_round is not None and r >= args.stop_after_round and r < num_rounds:
            print(f"[Session Budget] Reached round {r}; safe checkpoint committed.")
            session_paused = True
            progress_status = "PAUSED_CHECKPOINTED"
        else:
            session_paused = False
            progress_status = "TRAINING"

        # Progress tracking JSON
        progress_info = {
            "status": progress_status,
            "job_id": args.job_id,
            "round": r,
            "max_rounds": num_rounds,
            "val_macro_f1": round(val_f1_raw, 4),
            "val_accuracy": val_acc,
            "best_val_f1": round(best_val_f1_raw, 4),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (output_dir / "progress.json").write_text(canonical_json_dumps(progress_info), encoding="utf-8")

        if session_paused:
            break
        if stop_requested:
            break

    progress.close()
    total_time = time.monotonic() - t_start

    # Final post-training diagnostic evaluation: LOAD VERIFIED BEST CHECKPOINT
    print(f"\n[Post-Training] Loading verified BEST checkpoint for final evaluation...")
    best_ckpt, best_ckpt_path = load_verified_best_checkpoint(
        output_dir=output_dir,
        expected_config=resolved_config,
        is_fedavg=True,
        allow_legacy=args.allow_legacy,
    )
    best_weights = best_ckpt["model_state"]
    best_round = best_ckpt.get("round", -1)
    best_ckpt_sha = compute_file_sha256(best_ckpt_path)

    eval_best_model = model_from_global_state(best_weights)
    eval_best_model.to(device)

    print("[Post-Training] Evaluating verified BEST checkpoint on Validation set...")
    final_best_val_res = evaluate_metrics(eval_best_model, val_loader, device, max_batches=max_eval_batches)
    train_eval_res = None
    train_validation_gap = None
    if not session_paused and resolved_config["mode"] == "full":
        print("[Post-Training] Measuring train-validation gap on all original train images...")
        train_eval_ds = CanonicalClassificationDataset.from_manifest(
            release_dir / "manifests/train.jsonl", dataset_root=dataset_root,
            transform=eval_transform, return_metadata=True)
        train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=eval_collate_fn)
        train_eval_res = evaluate_metrics(eval_best_model, train_eval_loader, device)
        train_validation_gap = {
            domain: train_eval_res["per_source"][domain]["raw_macro_f1_supported"]
                    - final_best_val_res["per_source"][domain]["raw_macro_f1_supported"]
            for domain in ("plantdoc", "plantvillage")
        }
        train_validation_gap["macro_f1_38"] = (train_eval_res["raw_macro_f1"]
                                                - final_best_val_res["raw_macro_f1"])

    # Diagnostic test evaluation only in full mode with >= 15 rounds
    if args.mode == "full" and not session_paused:
        if quality_control.best_eligible:
            print("[Post-Training] Evaluating verified BEST checkpoint on Legacy Diagnostic Test set...")
            test_res = evaluate_metrics(eval_best_model, test_loader, device, max_batches=max_eval_batches)
            status_str = "CONVERGED_EARLY" if stop_requested else "MAX_BUDGET_REACHED"
        else:
            test_res = "SKIPPED_QUALITY_GATE_FAILED"
            status_str = "QUALITY_GATE_FAILED"
    elif session_paused:
        print(f"[Post-Training] Session paused at round {last_completed_round}. Skipping legacy diagnostic test evaluation.")
        test_res = "SKIPPED_SESSION_PAUSED"
        status_str = "PAUSED_CHECKPOINTED"
    elif args.mode == "pilot":
        print("[Post-Training] Pilot execution budget (1 round): Skipping legacy diagnostic test evaluation.")
        test_res = "SKIPPED_IN_PILOT"
        status_str = "PILOT_COMPLETED"
    else:
        print(f"[Post-Training] Mode {args.mode}: Skipping legacy diagnostic test evaluation.")
        test_res = f"SKIPPED_IN_{args.mode.upper()}"
        status_str = f"{args.mode.upper()}_COMPLETED"

    summary = {
        "status": status_str,
        "final_holdout_status": "FINAL_HOLDOUT_PENDING",
        "mode": args.mode,
        "job_id": args.job_id,
        "condition": condition,
        "partition_scheme": partition_scheme,
        "seed": seed,
        "w0_fingerprint": w0_fingerprint,
        "release_manifest_sha256": release_manifest_sha,
        "total_runtime_seconds": round(total_time, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0, 1),
        "best_round": best_round,
        "best_checkpoint_sha256": best_ckpt_sha,
        "best_val_f1": round(best_val_f1_raw, 4),
        "history": history,
        "best_checkpoint_validation": final_best_val_res,
        "best_checkpoint_train": train_eval_res,
        "train_validation_gap": train_validation_gap,
        "legacy_diagnostic_test": test_res,
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(canonical_json_dumps(summary), encoding="utf-8")
    if not summary_file.is_file() or summary_file.stat().st_size == 0:
        raise RuntimeError(f"Failed to verify written summary file at {summary_file}")
    print(f"\nFedAvg execution completed successfully! Summary written to {summary_file}")


if __name__ == "__main__":
    main()

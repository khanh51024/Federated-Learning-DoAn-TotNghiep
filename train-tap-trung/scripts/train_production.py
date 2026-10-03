"""Production Trainer for Centralized MobileNetV3-Small on Canonical Mixed Release.
Supports explicit modes:
  - smoke: 2 epochs with 10 batches to verify weight updates and resume.
  - pilot: 1 FULL epoch on all train samples (no batch limits) to profile exact timing and VRAM.
  - full: Complete production training with EarlyStopping and ReduceLROnPlateau.
Guarantees:
  - Atomic checkpointing (checkpoint_best.pt, checkpoint_last.pt).
  - Complete state saving for resume: model, optimizer, scheduler, scaler, epoch, RNG, patience.
  - Fail-closed guards: identity mismatch, missing resume checkpoint, fresh-overwrite prevention.
  - Accurate sample-weighted loss accumulation.
  - Post-training evaluation on BEST checkpoint (not last); test evaluation skipped in smoke/pilot.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

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


def find_dataset_and_release(
    custom_dataset_root: Optional[str] = None,
    custom_release_dir: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    release_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Locate canonical dataset root and release directory dynamically across local, Colab, or Kaggle."""
    if custom_release_dir:
        rel = Path(custom_release_dir).resolve()
        if (rel / "release_manifest.json").is_file():
            ds_root = Path(custom_dataset_root).resolve() if custom_dataset_root else rel.parent.parent
            return ds_root, rel
        raise FileNotFoundError(f"Specified release directory does not contain release_manifest.json: {rel}")

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
        raise FileNotFoundError(f"Specified dataset root does not contain release_manifest.json under {rel_names}: {ds_root}")

    ws = workspace_root or find_workspace_root()
    script_dir = Path(__file__).resolve().parent
    candidates = [
        ws / "dataset",
        script_dir.parent.parent / "dataset",
        Path("./dataset"),
        Path("../dataset"),
        Path("/content/dataset"),
        Path("/content/drive/MyDrive/dataset"),
        Path("/kaggle/input/plantvillage-plantdoc-canonical/dataset"),
        Path("/kaggle/input/plantvillage-plantdoc-canonical"),
    ]

    for c in candidates:
        for rname in rel_names:
            rel = c / f"mixed/{rname}"
            if rel.is_dir() and (rel / "release_manifest.json").is_file():
                return c.resolve(), rel.resolve()
            rel_flat = c / rname
            if rel_flat.is_dir() and (rel_flat / "release_manifest.json").is_file():
                return c.resolve(), rel_flat.resolve()

    raise FileNotFoundError(f"Could not locate canonical dataset containing {rel_names} in candidate locations: {candidates}")



WORKSPACE_ROOT = find_workspace_root()
CENTRALIZED_ROOT = WORKSPACE_ROOT / "train-tap-trung"
for p in [WORKSPACE_ROOT / "plant_data_contract", CENTRALIZED_ROOT, WORKSPACE_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.constants import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

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
    )
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
    from plant_data_contract.colab_sync import (
        ColabPersistenceSync,
        ColabSyncError,
        REQUIRED_DRIVE_FOLDER_ID,
    )
    from plant_data_contract.colab_drive_destination import DriveDestinationManager
except ImportError:
    # Try local wheel install or relative import
    import subprocess
    whl_candidates = (
        list((WORKSPACE_ROOT / "plant_data_contract/dist").glob("*.whl"))
        + list((WORKSPACE_ROOT / "train-gd-2/kaggle-workspace/campaign-mixed-pv-pd-v1").glob("*.whl"))
        + list(Path(".").glob("*.whl"))
    )
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
    )
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
    from plant_data_contract.colab_sync import (
        ColabPersistenceSync,
        ColabSyncError,
        REQUIRED_DRIVE_FOLDER_ID,
    )

from plant_data_contract.training_progress import TrainingProgress


def approved_w0_kwargs(spec, args):
    """Resolve the pilot W0 state beside the approved ImageNet weights."""
    if ((spec or {}).get("dataset_release") != "dataset/mixed/pv_pd_v3" or
            spec.get("status") != "READY_FULL" or getattr(args, "init", "scratch") != "imagenet_v1"):
        return {}
    name = spec.get("pretrained_w0_state_file")
    digest = spec.get("pretrained_w0_state_sha256")
    if not name or Path(name).name != name or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("READY_FULL pretrained v3 requires a pinned W0 state file and SHA-256")
    weights = getattr(args, "pretrained_weights", None)
    if not weights:
        raise ValueError("READY_FULL pretrained v3 requires --pretrained-weights")
    return {"w0_state_path": Path(weights).resolve().with_name(name),
            "expected_w0_state_sha256": digest}


def enforce_approved_v3_protocol(spec, args, w0_fingerprint, resolved_config):
    """Reject a full v3 run whose effective initialization differs from its pilot approval."""
    if args.mode != "full" or (spec or {}).get("dataset_release") != "dataset/mixed/pv_pd_v3":
        return
    if spec.get("status") != "READY_FULL":
        if args.preflight_only:
            return
        raise ValueError("Full v3 training requires a READY_FULL pilot-approved spec")
    variant = spec.get("selected_variant")
    expected_init = "scratch" if variant == "scratch" else "imagenet_v1"
    expected_ratio = 0.25 if variant == "pretrained_pd25" else 0.0
    expected_repeat_policy = spec.get("sampler_repeat_policy", "with_replacement")
    if variant not in ("scratch", "pretrained", "pretrained_pd25"):
        raise ValueError("READY_FULL spec has an invalid selected_variant")
    raw_ratio = str(args.plantdoc_sampling_ratio).lower()
    ratio = 0.0 if raw_ratio in ("none", "false", "") else float(raw_ratio)
    if (args.init != expected_init or abs(ratio - expected_ratio) > 1e-9 or
            args.transform_version != spec.get("preprocessing") or
            args.sampler_repeat_policy != expected_repeat_policy):
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
from plant_data_contract.taxonomy import PLANTVILLAGE_38_CLASSES


def create_mobilenetv3_small_scratch(
    num_classes: int = 38,
    seed: Optional[int] = None,
) -> Tuple[nn.Module, str]:
    m, fp, _ = create_model(num_classes=num_classes, seed=seed if seed is not None else 42, init="scratch")
    return m, fp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_production")



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def eval_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metas = [item[2] for item in batch]
    return images, targets, metas


def prediction_error_type(target_class: str, predicted_class: str) -> str:
    if target_class == predicted_class:
        return "correct"
    target_plant = target_class.split("___", 1)[0]
    predicted_plant = predicted_class.split("___", 1)[0]
    return "wrong_plant" if target_plant != predicted_plant else "right_plant_wrong_disease"


def summarize_scene_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenes: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("source_domain") != "plantdoc":
            continue
        scene_id = row.get("group_id") or row.get("sample_id")
        scenes.setdefault(str(scene_id), []).append(row)
    if not scenes:
        return {"scene_count": 0, "status": "NO_PLANTDOC_SCENES"}

    crop_counts = sorted(len(scene_rows) for scene_rows in scenes.values())
    scene_accuracies = [
        sum(row["error_type"] == "correct" for row in scene_rows) / len(scene_rows)
        for scene_rows in scenes.values()
    ]
    return {
        "scene_count": len(scenes),
        "scenes_with_error": sum(any(row["error_type"] != "correct" for row in scene_rows)
                                 for scene_rows in scenes.values()),
        "scenes_with_high_confidence_error_0_9": sum(
            any(row["error_type"] != "correct" and float(row["confidence"]) >= 0.9
                for row in scene_rows)
            for scene_rows in scenes.values()
        ),
        "mean_crop_accuracy_per_scene": float(np.mean(scene_accuracies)),
        "crop_count_per_scene": {
            "min": crop_counts[0],
            "median": float(np.median(crop_counts)),
            "p90": float(np.percentile(crop_counts, 90)),
            "max": crop_counts[-1],
        },
    }


def write_evaluation_cache(cache: Dict[str, Any], output_dir: Path, prefix: str) -> Dict[str, Any]:
    """Persist per-image CSV and logits from an evaluation already performed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(cache["logits"], dtype=np.float32)
    targets = np.asarray(cache["targets"], dtype=np.int64)
    predictions = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metadata = cache["metadata"]
    rows: List[Dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        target_id = int(targets[index])
        predicted_id = int(predictions[index])
        target_class = PLANTVILLAGE_38_CLASSES[target_id]
        predicted_class = PLANTVILLAGE_38_CLASSES[predicted_id]
        rows.append({
            "split": cache["split_name"],
            "sample_id": meta.get("sample_id", ""),
            "group_id": meta.get("group_id", ""),
            "source_domain": meta.get("source_domain", "unknown"),
            "relative_path": meta.get("relative_path", ""),
            "decoded_width": meta.get("decoded_width"),
            "decoded_height": meta.get("decoded_height"),
            "target_id": target_id,
            "target_class": target_class,
            "predicted_id": predicted_id,
            "predicted_class": predicted_class,
            "confidence": float(probabilities[index, predicted_id]),
            "error_type": prediction_error_type(target_class, predicted_class),
        })

    csv_path = output_dir / f"{prefix}_predictions.csv"
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, csv_path)

    logits_path = output_dir / f"{prefix}_logits.npz"
    logits_tmp = logits_path.with_suffix(logits_path.suffix + ".tmp")
    with logits_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            logits=logits,
            targets=targets,
            sample_ids=np.asarray([row["sample_id"] for row in rows]),
            group_ids=np.asarray([row["group_id"] for row in rows]),
            split=np.asarray([cache["split_name"]]),
        )
    os.replace(logits_tmp, logits_path)

    return {
        "predictions_csv": str(csv_path),
        "logits_npz": str(logits_path),
        "scene_metrics": summarize_scene_predictions(rows),
    }


def _multiclass_ece(logits: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(targets)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    ece = torch.zeros((), dtype=logits.dtype, device=logits.device)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correctness[mask].float().mean()).abs()
    return float(ece.item())


def fit_temperature_from_cache(cache: Dict[str, Any], max_iter: int = 50) -> Dict[str, Any]:
    """Fit one temperature only on the explicitly named calibration split."""
    if cache.get("split_name") != "calibration":
        raise ValueError("Temperature fitting is allowed only on the calibration split")
    logits = torch.as_tensor(cache["logits"], dtype=torch.float64)
    targets = torch.as_tensor(cache["targets"], dtype=torch.long)
    if logits.ndim != 2 or not len(logits) or len(logits) != len(targets):
        raise ValueError("Calibration logits and targets must be non-empty and aligned")

    criterion = nn.CrossEntropyLoss()
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = criterion(logits / temperature, targets)
        loss.backward()
        return loss

    predictions_before = logits.argmax(dim=1)
    nll_before = float(criterion(logits, targets).item())
    ece_before = _multiclass_ece(logits, targets)
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-5.0, 5.0).exp().item())
    scaled_logits = logits / temperature
    predictions_after = scaled_logits.argmax(dim=1)
    argmax_unchanged = bool(torch.equal(predictions_before, predictions_after))
    if not argmax_unchanged:
        raise AssertionError("Positive temperature scaling changed argmax predictions")
    return {
        "schema_version": 1,
        "calibration_split": "calibration",
        "num_samples": len(targets),
        "temperature": temperature,
        "nll_before": nll_before,
        "nll_after": float(criterion(scaled_logits, targets).item()),
        "ece_10_before": ece_before,
        "ece_10_after": _multiclass_ece(scaled_logits, targets),
        "argmax_unchanged": argmax_unchanged,
    }


def resolve_incumbent_quality_gate(args: Any, release_manifest_sha: str) -> Dict[str, Any]:
    """Resolve a PV floor from a byte-verified full incumbent, never a scratch pilot."""
    summary_arg = getattr(args, "incumbent_summary", None)
    checkpoint_arg = getattr(args, "incumbent_checkpoint", None)
    if not summary_arg or not checkpoint_arg:
        return {
            "status": "QUALITY_GATE_UNVERIFIED",
            "reason": "Both --incumbent-summary and --incumbent-checkpoint are required",
            "floor": None,
        }

    summary_path = Path(summary_arg).resolve()
    checkpoint_path = Path(checkpoint_arg).resolve()
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Incumbent summary/checkpoint path does not exist")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("seed", -1)) != int(args.seed):
        raise ValueError("Incumbent seed does not match candidate seed")
    if summary.get("release_manifest_sha256") != release_manifest_sha:
        raise ValueError("Incumbent release SHA does not match candidate release SHA")

    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if summary.get("best_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Incumbent checkpoint bytes do not match its summary SHA")
    baseline = summary.get("best_checkpoint_validation", {}).get("per_source", {}).get(
        "plantvillage", {}
    ).get("raw_macro_f1_supported")
    if baseline is None or not 0.0 <= float(baseline) <= 1.0:
        raise ValueError("Incumbent summary lacks a valid full PlantVillage macro-F1")
    max_regression = float(getattr(args, "pv_max_regression", 0.01))
    if not 0.0 <= max_regression < 1.0:
        raise ValueError("--pv-max-regression must be in [0, 1)")
    return {
        "status": "VERIFIED_INCUMBENT_FULL",
        "reason": None,
        "floor": max(0.0, float(baseline) - max_regression),
        "baseline": float(baseline),
        "max_regression": max_regression,
        "checkpoint_sha256": checkpoint_sha,
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "seed": int(args.seed),
        "release_manifest_sha256": release_manifest_sha,
    }


def replay_quality_control_history(
    floor: float,
    policy: Dict[str, Any],
    history: List[Dict[str, Any]],
    initial_lr: float,
) -> QualityControl:
    """Migrate legacy inferred LR-drop state by replaying recorded real LRs."""
    quality_control = QualityControl(floor, policy=policy, require_observed_lr=True)
    previous_lr = float(initial_lr)
    for record in history:
        step = int(record["epoch"])
        quality_control.observe(
            step,
            float(record["plantdoc_macro_f1_supported"]),
            float(record["plantvillage_macro_f1_supported"]),
            float(record["plantdoc_crop_accuracy"]),
        )
        current_lr = float(record["lr"])
        quality_control.record_scheduler_step(step, previous_lr, current_lr)
        previous_lr = current_lr
    return quality_control


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    cache_out: Optional[Dict[str, Any]] = None,
    split_name: str = "validation",
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
    all_logits = []
    all_metadata = []

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
            if cache_out is not None:
                all_logits.append(outputs.detach().cpu().to(torch.float32).numpy())
                all_metadata.extend(dict(meta) for meta in metas)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_domains = np.array(all_domains)
    if cache_out is not None:
        cache_out.update({
            "split_name": split_name,
            "logits": np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 38), dtype=np.float32),
            "targets": all_targets.copy(),
            "metadata": all_metadata,
        })

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


class ProtocolConfigDict(dict):
    """Configuration dictionary that recomputes canonical protocol_content_sha256 upon modification."""
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key != "protocol_content_sha256" and key in IMMUTABLE_PROTOCOL_KEYS:
            super().__setitem__("protocol_content_sha256", compute_job_protocol_sha256(self))


def resolve_centralized_config(
    args: Any,
    w0_fingerprint: Optional[str],
    release_manifest_sha: str,
    spec: Optional[Dict[str, Any]] = None,
    spec_path: Optional[Path] = None,
    job_cfg: Optional[Dict[str, Any]] = None,
    release_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build resolved configuration dictionary for Centralized trainer."""
    opt_name = getattr(args, "optimizer", "AdamW") or "AdamW"
    if opt_name != "AdamW":
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Centralized runner requires AdamW.")

    weight_decay = getattr(args, "weight_decay", 1e-4)
    if weight_decay is None:
        weight_decay = 1e-4

    cent_hp = spec.get("hyperparameters", {}).get("centralized", {}) if spec else {}
    if args.mode == "full" and not (1 <= args.epochs <= 100):
        raise ValueError("Full centralized max_epochs must be within 1..100")
    explicit_flags = getattr(args, "_explicit_flags", set())

    # Reject any explicit CLI argument that conflicts with campaign spec or job configuration (F6)
    if ("--weight-decay" in explicit_flags or "-w" in explicit_flags) and "weight_decay" in cent_hp and abs(float(args.weight_decay) - float(cent_hp["weight_decay"])) > 1e-9:
        raise ValueError(f"CLI argument --weight-decay={args.weight_decay} conflicts with campaign_spec weight_decay={cent_hp['weight_decay']}. Production spec is authoritative.")
    if ("--amp" in explicit_flags or "--no-amp" in explicit_flags) and "amp" in cent_hp and bool(args.amp) != bool(cent_hp["amp"]):
        raise ValueError(f"CLI argument --amp/--no-amp ({args.amp}) conflicts with campaign_spec amp={cent_hp['amp']}.")
    if "--condition" in explicit_flags and job_cfg and job_cfg.get("condition") and args.condition != job_cfg["condition"]:
        raise ValueError(f"CLI argument --condition={args.condition} conflicts with job condition={job_cfg['condition']}.")
    if "--seed" in explicit_flags and job_cfg and job_cfg.get("seed") and args.seed != job_cfg["seed"]:
        raise ValueError(f"CLI argument --seed={args.seed} conflicts with job seed={job_cfg['seed']}.")
    if "--lr" in explicit_flags and "lr" in cent_hp and abs(args.lr - cent_hp["lr"]) > 1e-9:
        raise ValueError(f"CLI argument --lr={args.lr} conflicts with campaign_spec lr={cent_hp['lr']}. Production spec is authoritative.")
    if "--batch-size" in explicit_flags and "batch_size" in cent_hp and args.batch_size != cent_hp["batch_size"]:
        raise ValueError(f"CLI argument --batch-size={args.batch_size} conflicts with campaign_spec batch_size={cent_hp['batch_size']}.")
    if "--epochs" in explicit_flags and "max_epochs" in cent_hp and args.epochs != cent_hp["max_epochs"]:
        raise ValueError(f"CLI argument --epochs={args.epochs} conflicts with campaign_spec max_epochs={cent_hp['max_epochs']}.")
    if "--patience" in explicit_flags and "early_stopping_patience" in cent_hp and args.patience != cent_hp["early_stopping_patience"]:
        raise ValueError(f"CLI argument --patience={args.patience} conflicts with campaign_spec early_stopping_patience={cent_hp['early_stopping_patience']}.")
    if "--optimizer" in explicit_flags and "optimizer" in cent_hp and args.optimizer != cent_hp["optimizer"]:
        raise ValueError(f"CLI argument --optimizer={args.optimizer} conflicts with campaign_spec optimizer={cent_hp['optimizer']}.")

    if args.mode == "smoke":
        max_train_batches = 10
        max_val_batches = 5
    elif args.mode == "pilot":
        max_train_batches = None
        max_val_batches = None
    else:  # full
        max_train_batches = None
        max_val_batches = None

    from plant_data_contract.integrity import compute_selected_partition_digest

    rel_dir = None
    if release_dir:
        rel_dir = Path(release_dir)
    elif getattr(args, "release_dir", None):
        rel_dir = Path(args.release_dir)
    elif getattr(args, "dataset_root", None):
        rel_names = [getattr(args, "release_name", None), "pv_pd_v2", "pv_pd_v1"]
        for rname in rel_names:
            if rname:
                p = Path(args.dataset_root) / f"mixed/{rname}"
                if (p / "manifests/train.jsonl").is_file():
                    rel_dir = p
                    break
                p_flat = Path(args.dataset_root) / rname
                if (p_flat / "manifests/train.jsonl").is_file():
                    rel_dir = p_flat
                    break

    if rel_dir is None:
        ws = find_workspace_root(getattr(args, "workspace_root", None))
        for rname in ["pv_pd_v2", "pv_pd_v1"]:
            cand = ws / f"dataset/mixed/{rname}"
            if (cand / "manifests/train.jsonl").is_file():
                rel_dir = cand
                break

    if rel_dir is None:
        rel_dir = Path("dataset/mixed/pv_pd_v2") if Path("dataset/mixed/pv_pd_v2/manifests/train.jsonl").is_file() else Path("dataset/mixed/pv_pd_v1")

    selected_part_digest = compute_selected_partition_digest(rel_dir, args.condition)

    init_mode = getattr(args, "init", "scratch")
    tf_version = getattr(args, "transform_version", "canonical_v1")
    pd_sampling = getattr(args, "plantdoc_sampling_ratio", "none")
    sampler_repeat_policy = getattr(args, "sampler_repeat_policy", "with_replacement")
    pretrained_file_sha = None
    if init_mode == "imagenet_v1":
        wfile = find_pretrained_weights_file(getattr(args, "pretrained_weights", None))
        pretrained_file_sha = hashlib.sha256(wfile.read_bytes()).hexdigest()

    quality_gate = resolve_incumbent_quality_gate(args, release_manifest_sha)
    effective_params: Dict[str, Any] = {
        "condition": args.condition,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": opt_name,
        "weight_decay": weight_decay,
        "patience": args.patience,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "amp": bool(args.amp),
        "release_manifest_sha256": release_manifest_sha,
        "mode": args.mode,
        "epochs": args.epochs,
        "scheduler_policy": "ReduceLROnPlateau_PlantDoc_factor0.5_patience3_abs0.002_min15",
        "init": init_mode,
        "architecture_version": f"mobilenet_v3_small_{init_mode}_v1",
        "preprocessing_version": tf_version,
        "plantdoc_sampling_ratio": str(pd_sampling),
        "sampler_repeat_policy": sampler_repeat_policy,
        "sampler_seed": int(args.seed),
        "pretrained_file_sha256": pretrained_file_sha,
        "selected_partition_digest": selected_part_digest,
        "quality_gate_status": quality_gate["status"],
        "quality_gate_reason": quality_gate.get("reason"),
        "quality_floor": quality_gate.get("floor"),
        "incumbent_full": {key: value for key, value in quality_gate.items()
                           if key not in ("status", "reason", "floor")},
        "quality_audit_sha256": (spec or {}).get("quality_gate", {}).get("audit_sha256"),
        "convergence_policy": {**DEFAULT_CONVERGENCE, **(spec or {}).get("quality_gate", {}).get("convergence", {})},
    }
    pilot_manifest = getattr(args, "pilot_train_manifest", None)
    if pilot_manifest:
        if args.mode != "pilot":
            raise ValueError("--pilot-train-manifest is only valid in pilot mode")
        effective_params["pilot_train_manifest_sha256"] = hashlib.sha256(Path(pilot_manifest).read_bytes()).hexdigest()
    if w0_fingerprint:
        effective_params["w0_fingerprint"] = w0_fingerprint
    if getattr(args, "job_id", None):
        effective_params["job_id"] = args.job_id

    resolved = ProtocolConfigDict(effective_params)
    resolved["protocol_content_sha256"] = compute_job_protocol_sha256(effective_params)
    if getattr(args, "job_id", None):
        resolved["job_id"] = args.job_id
        resolved["partition_seed"] = job_cfg.get("partition_seed", 42) if job_cfg else 42
        resolved["training_seed"] = args.seed
    if spec_path:
        spec_bytes = spec_path.read_bytes() if (spec_path and Path(spec_path).is_file()) else b""
        resolved["campaign_spec_sha256"] = hashlib.sha256(spec_bytes).hexdigest() if spec_bytes else None
        resolved["spec_file_path"] = str(spec_path)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production Centralized Trainer for pv_pd_v1/v2")
    parser.add_argument("--job-id", type=str, default=None, help="Job identifier from campaign_spec.json")
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="full", help="Execution mode")
    parser.add_argument("--condition", choices=["mixed", "pv_only"], default="mixed", help="Data condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer type")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs for full mode")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="Session budget: stop after completing this epoch and mark PAUSED_CHECKPOINTED")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint_last.pt to resume")
    parser.add_argument("--prior-best", type=str, default=None, help="Explicit path to prior checkpoint_best.pt when resuming")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output directory")
    parser.add_argument("--allow-legacy", action="store_true", default=False, help="Allow resuming legacy checkpoints lacking modern guards")
    parser.add_argument("--amp", action="store_true", default=None, help="Use Automatic Mixed Precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--workspace-root", type=str, default=None, help="Root path of workspace")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to canonical dataset root")
    parser.add_argument("--release-dir", type=str, default=None, help="Path to canonical release directory")
    parser.add_argument("--release-name", type=str, default=None, help="Release name under dataset root (e.g. pv_pd_v2, pv_pd_v1)")
    parser.add_argument("--expected-release-sha", type=str, default=None, help="Approved release manifest SHA-256")
    parser.add_argument("--spec-file", type=str, default=None, help="Optional path to campaign_spec.json")
    parser.add_argument("--colab-sync-dir", type=str, default=None, help="Remote Google Drive directory to persist checkpoints per epoch")
    parser.add_argument("--drive-folder-id", type=str, default=None, help="Verified Drive folder ID (defaults to 1GpmEi_wFgszbrP9wr6_Xe6VA0xRagBCi)")
    parser.add_argument("--preflight-only", action="store_true", default=False, help="Run integrity and configuration preflight checks without training or file mutation")
    parser.add_argument("--init", choices=["scratch", "imagenet_v1"], default="scratch", help="Model initialization mode ('scratch' or 'imagenet_v1')")
    parser.add_argument("--pretrained-weights", type=str, default=None, help="Path to pretrained weights file")
    parser.add_argument("--plantdoc-sampling-ratio", type=str, default="none", help="Minibatch sampling target ratio for PlantDoc ('none', '0.15', '0.25')")
    parser.add_argument("--sampler-repeat-policy", choices=["with_replacement", "cycle_without_replacement"], default="with_replacement", help="Domain sampler repeat policy; lower-repeat cycle mode is experimental")
    parser.add_argument("--transform-version", choices=["canonical_v1", "aspect_pad_v1", "aspect_pad_mean_v2", "moderate_augment_v1", "light_augment_v2"], default="canonical_v1", help="Preprocessing transform pipeline version")
    parser.add_argument("--incumbent-summary", type=str, default=None, help="Full incumbent summary.json for the same seed/release")
    parser.add_argument("--incumbent-checkpoint", type=str, default=None, help="Full incumbent checkpoint whose SHA is recorded in the summary")
    parser.add_argument("--pv-max-regression", type=float, default=0.01, help="Maximum allowed PlantVillage macro-F1 regression from the verified full incumbent")
    parser.add_argument("--prepare-calibration", action="store_true", default=False, help="Fit temperature on the calibration split after selecting the best full checkpoint")
    parser.add_argument("--full-image-verify", action="store_true", default=False, help="Perform 100% byte SHA-256 and decode verification of all 50k+ images (slow)")
    parser.add_argument("--pilot-train-manifest", type=str, default=None, help="Fixed, release-subset JSONL for pilot mode only")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    explicit_flags = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}
    args._explicit_flags = explicit_flags

    print("=" * 70)
    print(f"PRODUCTION TRAINER: Mode={args.mode.upper()}, Condition={args.condition}, Seed={args.seed}")
    if args.job_id:
        print(f"Job ID: {args.job_id}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Spec parsing & job resolution if provided
    ws_root = find_workspace_root(args.workspace_root)
    script_dir = Path(__file__).resolve().parent
    spec_path = None
    spec = None
    job_cfg = None

    if args.spec_file or args.job_id:
        if args.spec_file:
            p = Path(args.spec_file).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Explicitly specified --spec-file does not exist: {args.spec_file}")
            spec_path = p
        else:
            candidates = [
                script_dir / "campaign_spec.json",
                Path("campaign_spec.json").resolve() if Path("campaign_spec.json").is_file() else None,
                ws_root / "plans/antigravity-kaggle-group-local-20260927/campaign_spec.json",
                Path("/content/campaign_spec.json") if Path("/content/campaign_spec.json").is_file() else None,
            ]
            for c in candidates:
                if c and c.is_file():
                    spec_path = c
                    break

        if spec_path and spec_path.is_file():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if args.job_id:
                for k in ["colab_centralized_jobs", "laptop_centralized_jobs", "centralized_jobs", "jobs"]:
                    for j in spec.get(k, []):
                        if j.get("job_id") == args.job_id:
                            job_cfg = j
                            break
                    if job_cfg:
                        break
                if not job_cfg:
                    raise ValueError(f"Job ID '{args.job_id}' not found in campaign_spec.json")
                if args.job_id == "cent_mixed_s42" and not args.preflight_only:
                    raise ValueError("Job cent_mixed_s42 is ALREADY COMPLETED on laptop RTX 2050 (F1=0.9615). Re-running is forbidden.")

                # Read parameters from spec
                cent_hp = spec.get("hyperparameters", {}).get("centralized", {})
                # Reject any explicit CLI argument that conflicts with campaign spec or job configuration (F6)
                if ("--weight-decay" in explicit_flags or "-w" in explicit_flags) and "weight_decay" in cent_hp and abs(float(args.weight_decay) - float(cent_hp["weight_decay"])) > 1e-9:
                    raise ValueError(f"CLI argument --weight-decay={args.weight_decay} conflicts with campaign_spec weight_decay={cent_hp['weight_decay']}. Production spec is authoritative.")
                if ("--amp" in explicit_flags or "--no-amp" in explicit_flags) and "amp" in cent_hp and bool(args.amp) != bool(cent_hp["amp"]):
                    raise ValueError(f"CLI argument --amp/--no-amp ({args.amp}) conflicts with campaign_spec amp={cent_hp['amp']}.")
                if "--condition" in explicit_flags and job_cfg and job_cfg.get("condition") and args.condition != job_cfg["condition"]:
                    raise ValueError(f"CLI argument --condition={args.condition} conflicts with job condition={job_cfg['condition']}.")
                if "--seed" in explicit_flags and job_cfg and job_cfg.get("seed") and args.seed != job_cfg["seed"]:
                    raise ValueError(f"CLI argument --seed={args.seed} conflicts with job seed={job_cfg['seed']}.")
                if "--lr" in explicit_flags and "lr" in cent_hp and abs(args.lr - cent_hp["lr"]) > 1e-9:
                    raise ValueError(f"CLI argument --lr={args.lr} conflicts with campaign_spec lr={cent_hp['lr']}. Production spec is authoritative.")
                if "--batch-size" in explicit_flags and "batch_size" in cent_hp and args.batch_size != cent_hp["batch_size"]:
                    raise ValueError(f"CLI argument --batch-size={args.batch_size} conflicts with campaign_spec batch_size={cent_hp['batch_size']}.")
                if "--epochs" in explicit_flags and "max_epochs" in cent_hp and args.epochs != cent_hp["max_epochs"]:
                    raise ValueError(f"CLI argument --epochs={args.epochs} conflicts with campaign_spec max_epochs={cent_hp['max_epochs']}.")
                if "--patience" in explicit_flags and "early_stopping_patience" in cent_hp and args.patience != cent_hp["early_stopping_patience"]:
                    raise ValueError(f"CLI argument --patience={args.patience} conflicts with campaign_spec early_stopping_patience={cent_hp['early_stopping_patience']}.")
                if "--optimizer" in explicit_flags and "optimizer" in cent_hp and args.optimizer != cent_hp["optimizer"]:
                    raise ValueError(f"CLI argument --optimizer={args.optimizer} conflicts with campaign_spec optimizer={cent_hp['optimizer']}.")
                if "condition" in job_cfg:
                    args.condition = job_cfg["condition"]
                if "seed" in job_cfg:
                    args.seed = job_cfg["seed"]
                if "lr" in cent_hp and "--lr" not in explicit_flags:
                    args.lr = cent_hp["lr"]
                if "batch_size" in cent_hp and "--batch-size" not in explicit_flags:
                    args.batch_size = cent_hp["batch_size"]
                if "max_epochs" in cent_hp and "--epochs" not in explicit_flags:
                    args.epochs = cent_hp["max_epochs"]
                if "early_stopping_patience" in cent_hp and "--patience" not in explicit_flags:
                    args.patience = cent_hp["early_stopping_patience"]
                if "optimizer" in cent_hp and "--optimizer" not in explicit_flags:
                    args.optimizer = cent_hp["optimizer"]
                elif args.optimizer is None:
                    args.optimizer = "AdamW"
                if "weight_decay" in cent_hp and "--weight-decay" not in explicit_flags:
                    args.weight_decay = cent_hp["weight_decay"]
                elif args.weight_decay is None:
                    args.weight_decay = 1e-4
                if "amp" in cent_hp and "--amp" not in explicit_flags and "--no-amp" not in explicit_flags:
                    args.amp = cent_hp["amp"]
                elif args.amp is None:
                    args.amp = True

    # A minimal pilot spec has no job block; AMP must still resolve to a bool.
    if args.amp is None:
        args.amp = True

    # 2. Dataset setup and Release Manifest Hash check
    dataset_root, release_dir = find_dataset_and_release(
        custom_dataset_root=args.dataset_root,
        custom_release_dir=args.release_dir,
        workspace_root=ws_root,
        release_name=args.release_name,
    )
    integrity_report = verify_release_integrity(release_dir, expected_release_sha=args.expected_release_sha)
    release_manifest_sha = integrity_report["release_manifest_sha256"]
    print(f"Dataset Root: {dataset_root}")
    print(f"Release Manifest SHA-256 Verified: {release_manifest_sha}")

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
    if args.mode == "full" and (not args.preflight_only or (spec or {}).get("quality_gate")):
        verify_full_data_gate(release_dir, release_manifest_sha, spec or {})
    if (args.mode == "full" and not args.preflight_only and
            (spec or {}).get("dataset_release") == "dataset/mixed/pv_pd_v3" and
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
    preflight_w0_fingerprint = None
    if args.preflight_only:
        # Preflight must report the same protocol identity as a real run.
        set_seed(args.seed)
        _, preflight_w0_fingerprint, _ = create_model(
            num_classes=38,
            seed=args.seed,
            init=getattr(args, "init", "scratch"),
            pretrained_weights_path=getattr(args, "pretrained_weights", None),
            **approved_w0_kwargs(spec, args),
        )

    resolved_config = resolve_centralized_config(
        args=args,
        w0_fingerprint=preflight_w0_fingerprint,
        release_manifest_sha=release_manifest_sha,
        spec=spec,
        spec_path=spec_path,
        job_cfg=job_cfg,
        release_dir=release_dir,
    )
    if args.prepare_calibration and args.mode != "full":
        raise ValueError("--prepare-calibration is only valid in full mode")
    if args.preflight_only:
        enforce_approved_v3_protocol(spec, args, preflight_w0_fingerprint, resolved_config)

    if args.preflight_only:
        print("=" * 70)
        print("PREFLIGHT CHECK OK:")
        if args.job_id:
            print(f"  Job ID: {args.job_id}")
        print(f"  Mode: {args.mode}")
        print(f"  Condition: {args.condition}")
        print(f"  Seed: {args.seed}")
        print(f"  Init Mode: {getattr(args, 'init', 'scratch')}")
        print(f"  Transform Version: {getattr(args, 'transform_version', 'canonical_v1')}")
        print(f"  Sampling Ratio: {getattr(args, 'plantdoc_sampling_ratio', 'none')}")
        print(f"  Sampler Repeat Policy: {getattr(args, 'sampler_repeat_policy', 'with_replacement')}")
        print(f"  PV Quality Gate: {resolved_config['quality_gate_status']}")
        if resolved_config.get("quality_floor") is not None:
            print(f"  PV Incumbent/Floor: {resolved_config['incumbent_full']['baseline']:.6f} / "
                  f"{resolved_config['quality_floor']:.6f}")
        else:
            print(f"  PV Quality Gate Reason: {resolved_config['quality_gate_reason']}")
        print(f"  Release Manifest SHA-256: {release_manifest_sha}")
        print(f"  Manifests Verified: {len(manifest_contract['manifest_counts'])} splits")
        print(f"  Images: {image_report.get('records_checked', image_report.get('total_images_checked', 0))} paths, "
              f"{image_report.get('unique_files_hashed', 0)} hashes, "
              f"{image_report.get('unique_files_decoded', 0)} decodes ({image_report['status']})")
        print(f"  Model W0 Fingerprint: {preflight_w0_fingerprint}")
        print(f"  Protocol Content SHA-256: {resolved_config['protocol_content_sha256']}")
        print("PREFLIGHT PASSED: Exiting cleanly with zero file mutation.")
        print("=" * 70)
        return

    if args.mode == "full" and resolved_config["quality_gate_status"] != "VERIFIED_INCUMBENT_FULL":
        raise ValueError("QUALITY_GATE_UNVERIFIED: full training requires a byte-verified full incumbent "
                         "from the same seed and release")

    if args.mode == "full" and not torch.cuda.is_available() and os.environ.get("ANTIGRAVITY_ALLOW_CPU_FULL") != "1":
        raise RuntimeError("Full mode training requires a CUDA GPU accelerator, but none was detected.")

    output_dir_str = args.output_dir or (f"colab_results/{args.job_id}/{args.mode}" if args.job_id else None)
    if not output_dir_str:
        raise ValueError("--output-dir is required when --job-id is not specified")
    output_dir = Path(output_dir_str).resolve()

    # 3. Fresh-overwrite guard
    check_fresh_overwrite_guard(output_dir, is_resume=bool(args.resume), overwrite=args.overwrite)

    # NOW initialize model and compute W0
    set_seed(args.seed)
    model, w0_fingerprint, model_metadata = create_model(
        num_classes=38,
        seed=args.seed,
        init=getattr(args, "init", "scratch"),
        pretrained_weights_path=getattr(args, "pretrained_weights", None),
        **approved_w0_kwargs(spec, args),
    )
    resolved_config["w0_fingerprint"] = w0_fingerprint
    enforce_approved_v3_protocol(spec, args, w0_fingerprint, resolved_config)

    train_manifest = release_dir / "manifests/train.jsonl"
    val_manifest = release_dir / "manifests/val.jsonl"
    calibration_manifest = release_dir / "manifests/calibration.jsonl"
    test_manifest = release_dir / "manifests/legacy_diagnostic_test.jsonl"

    tf_ver = getattr(args, "transform_version", "canonical_v1")
    train_transform = get_transforms(tf_ver, is_train=True)
    eval_transform = get_transforms(tf_ver, is_train=False)

    train_ds = CanonicalClassificationDataset.from_manifest(
        train_manifest, dataset_root=dataset_root, transform=train_transform, return_metadata=False
    )
    if args.pilot_train_manifest:
        if args.mode != "pilot":
            raise ValueError("--pilot-train-manifest is only valid in pilot mode")
        pilot_path = Path(args.pilot_train_manifest).resolve()
        pilot_rows = [json.loads(line) for line in pilot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        full_by_id = {row["sample_id"]: row for row in train_ds.samples}
        pilot_ids = [row["sample_id"] for row in pilot_rows]
        if not pilot_rows or len(pilot_ids) != len(set(pilot_ids)):
            raise ValueError("Pilot manifest is empty or has repeated sample IDs")
        if any(full_by_id.get(row["sample_id"]) != row for row in pilot_rows):
            raise ValueError("Pilot manifest contains a changed row or a non-train sample")
        full_group_ids = {row["group_id"] for row in train_ds.samples}
        pilot_group_ids = {row["group_id"] for row in pilot_rows}
        if not pilot_group_ids <= full_group_ids or {row["sample_id"] for row in train_ds.samples
                if row["group_id"] in pilot_group_ids} != set(pilot_ids):
            raise ValueError("Pilot manifest splits a scene group")
        if {row["sample_id"] for row in train_ds.samples if row["source_domain"] == "plantdoc"} - set(pilot_ids):
            raise ValueError("Pilot manifest must contain all eligible PlantDoc train samples")
        train_ds = CanonicalClassificationDataset(
            samples=pilot_rows, dataset_root=dataset_root, transform=train_transform, return_metadata=False
        )
    if args.condition == "pv_only":
        pv_samples = [s for s in train_ds.samples if s.get("source_domain") == "plantvillage"]
        train_ds = CanonicalClassificationDataset(
            samples=pv_samples, dataset_root=dataset_root, transform=train_transform, return_metadata=False
        )

    val_ds = CanonicalClassificationDataset.from_manifest(
        val_manifest, dataset_root=dataset_root, transform=eval_transform, return_metadata=True
    )
    test_ds = CanonicalClassificationDataset.from_manifest(
        test_manifest, dataset_root=dataset_root, transform=eval_transform, return_metadata=True
    )
    calibration_ds = None
    if args.prepare_calibration:
        calibration_ds = CanonicalClassificationDataset.from_manifest(
            calibration_manifest, dataset_root=dataset_root, transform=eval_transform, return_metadata=True
        )

    pd_sampling_str = getattr(args, "plantdoc_sampling_ratio", "none")
    use_sampling = pd_sampling_str not in ("none", "0", "0.0", "", None)
    sampler = None
    if use_sampling:
        pd_ratio = float(pd_sampling_str)
        sampler = build_domain_balanced_sampler(
            train_ds,
            plantdoc_weight=pd_ratio,
            seed=args.seed,
            repeat_policy=args.sampler_repeat_policy,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
        print(f"Applied domain-balanced sampler (plantdoc_weight={pd_ratio:.2f}, "
              f"repeat_policy={args.sampler_repeat_policy})")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)
    calibration_loader = None
    if calibration_ds is not None:
        calibration_loader = DataLoader(calibration_ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=0, collate_fn=eval_collate_fn)

    print(f"Loaded train samples: {len(train_ds)} (Batches: {len(train_loader)})")
    print(f"Loaded val samples: {len(val_ds)} (Batches: {len(val_loader)})")

    # 4. Model & Optimizer
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=resolved_config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3,
                                  threshold=0.002, threshold_mode="abs")
    scaler = GradScaler("cuda", enabled=bool(args.amp and torch.cuda.is_available()))
    criterion = nn.CrossEntropyLoss()

    max_train_batches = resolved_config["max_train_batches"]
    max_val_batches = resolved_config["max_val_batches"]

    start_epoch = 1
    best_val_f1_raw = -1.0
    history = []
    patience_counter = 0
    quality_floor = resolved_config.get("quality_floor")
    if args.mode == "full" and quality_floor is None:
        raise ValueError("QUALITY_GATE_UNVERIFIED: missing same-seed, same-release full incumbent PV floor")
    quality_control = QualityControl(quality_floor if quality_floor is not None else 0.0,
                                     policy=resolved_config["convergence_policy"],
                                     require_observed_lr=True)
    stop_requested = False

    # 5. Strict Resume handling via load_verified_checkpoint
    colab_sync = None
    if args.colab_sync_dir:
        drive_folder_id = args.drive_folder_id or REQUIRED_DRIVE_FOLDER_ID
        drive_api = getattr(args, "drive_api", None)
        drive_mgr = DriveDestinationManager(
            drive_api=drive_api,
            required_folder_id=drive_folder_id,
        )
        colab_sync = ColabPersistenceSync(
            local_staging_dir=output_dir,
            remote_dest_dir=Path(args.colab_sync_dir),
            drive_folder_id=drive_folder_id,
            drive_api=drive_api,
        )
        if not args.resume and not (output_dir / "checkpoint_last.pt").is_file():
            print("[Colab Sync] Checking for remote checkpoints on Drive to stage into local SSD...")
            if colab_sync.stage_remote_checkpoints_for_resume():
                args.resume = str(output_dir / "checkpoint_last.pt")
                print(f"[Colab Sync] Successfully staged remote checkpoints into {output_dir}")

    if args.resume:
        resume_target = Path(args.resume).resolve()
        resume_dir = resume_target.parent if resume_target.is_file() else resume_target
        ckpt, last_path, best_path = load_verified_checkpoint(
            output_dir=resume_dir,
            expected_config=resolved_config,
            is_fedavg=False,
            allow_legacy=args.allow_legacy,
        )
        validate_and_restore_rng(ckpt, allow_legacy=args.allow_legacy)

        prior_best_f1 = float(ckpt.get("best_val_f1_raw", ckpt.get("best_val_f1", -1.0)))
        best_epoch = ckpt.get("best_generation", ckpt.get("epoch", 0))

        if output_dir != resume_dir:
            best_path, prior_best_f1, best_epoch = import_and_validate_best_checkpoint(
                last_path,
                output_dir,
                resolved_config,
                is_fedavg=False,
                allow_legacy=args.allow_legacy,
                source_best_path_override=best_path,
            )

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and ckpt["scheduler_state"]:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and ckpt["scaler_state"] and args.amp:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1_raw = prior_best_f1
        patience_counter = ckpt.get("patience_counter", 0) if args.allow_legacy else ckpt["patience_counter"]
        history = ckpt.get("history", [])
        if "quality_control_state" in ckpt:
            quality_state = ckpt["quality_control_state"]
            if int(quality_state.get("lr_tracking_version", 1)) >= 2:
                quality_control = QualityControl(quality_floor if quality_floor is not None else 0.0,
                                                 policy=resolved_config["convergence_policy"],
                                                 state=quality_state,
                                                 require_observed_lr=True)
            elif history:
                quality_control = replay_quality_control_history(
                    quality_floor if quality_floor is not None else 0.0,
                    resolved_config["convergence_policy"],
                    history,
                    args.lr,
                )
                logging.warning("Migrated legacy inferred LR-drop state from recorded epoch LRs")
            elif not args.allow_legacy:
                raise ValueError("Legacy quality-control state cannot be migrated without history")
        elif not args.allow_legacy:
            raise ValueError("Resume checkpoint lacks quality-control state")
        stop_requested = quality_control.should_stop(ckpt.get("epoch", 0)) if history else False
        if sampler is not None and ckpt.get("sampler_state"):
            sampler.load_state_dict(ckpt["sampler_state"])
        print(f"Resumed from {last_path} at epoch {start_epoch} (Prior Best Val F1: {best_val_f1_raw:.4f}, Best Epoch: {best_epoch}, Patience: {patience_counter})")

    # Only after validation passes, create output directory and save resolved configuration
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(canonical_json_dumps(resolved_config), encoding="utf-8")

    if args.mode == "smoke":
        max_epochs = max(2, start_epoch)
    elif args.mode == "pilot":
        max_epochs = min(args.epochs, 5)
    else:  # full
        max_epochs = args.epochs

    print(f"\nBeginning execution: Epochs={max_epochs}, MaxTrainBatches={'ALL' if max_train_batches is None else max_train_batches}")

    t_run_start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    session_paused = False
    last_completed_epoch = start_epoch - 1
    progress_batches = min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader)
    progress = TrainingProgress(max_epochs * progress_batches)
    progress.completed = min(progress.total_steps, (start_epoch - 1) * progress_batches)
    if args.mode == "full" and stop_requested:
        print("Convergence stop was already committed in the resume checkpoint.")

    for epoch in range(start_epoch, max_epochs + 1):
        last_completed_epoch = epoch
        if args.mode == "full" and stop_requested:
            logging.info("Convergence stop committed at epoch %s", epoch - 1)
            break

        t_ep_start = time.monotonic()
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        ep_loss_sum = 0.0
        steps = 0
        steps_applied = 0
        steps_skipped = 0
        samples_processed = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            if max_train_batches and b_idx >= max_train_batches:
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast("cuda", enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite train loss at epoch {epoch}, batch {b_idx + 1}")

            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
            steps_skipped += int(step_was_skipped)
            steps_applied += int(not step_was_skipped)

            ep_loss_sum += float(loss.item()) * len(targets)
            steps += 1
            samples_processed += len(targets)
            progress.update(
                increment=1, phase="train", position=f"epoch {epoch}/{max_epochs} batch {b_idx + 1}/{progress_batches}",
                loss=ep_loss_sum / samples_processed, lr=optimizer.param_groups[0]["lr"],
                vram_mb=torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            )

            if (b_idx + 1) % 100 == 0 or (b_idx + 1) == len(train_loader):
                cur_loss = ep_loss_sum / samples_processed
                vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                step_info = {
                    "status": "TRAINING",
                    "condition": args.condition,
                    "seed": args.seed,
                    "epoch": epoch,
                    "max_epochs": max_epochs,
                    "batch": b_idx + 1,
                    "total_batches": len(train_loader),
                    "running_loss": round(cur_loss, 4),
                    "peak_vram_mb": round(vram, 1),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                (output_dir / "progress.json").write_text(canonical_json_dumps(step_info), encoding="utf-8")

        ep_duration = time.monotonic() - t_ep_start
        if steps_applied == 0:
            raise RuntimeError(f"No optimizer step was applied during epoch {epoch}")
        avg_train_loss = ep_loss_sum / max(1, samples_processed)

        sampler_statistics = (sampler.statistics_for_draw_count(samples_processed)
                              if sampler is not None else None)
        if sampler_statistics is not None:
            sampler_stats_path = output_dir / "sampler_epoch_stats.json"
            sampler_stats_tmp = sampler_stats_path.with_suffix(".json.tmp")
            sampler_stats_tmp.write_text(canonical_json_dumps(sampler_statistics), encoding="utf-8")
            os.replace(sampler_stats_tmp, sampler_stats_path)

        # Validation with sample-weighted loss
        progress.update(phase="validate", position=f"epoch {epoch}/{max_epochs}", loss=avg_train_loss,
                        lr=optimizer.param_groups[0]["lr"])
        val_cache: Dict[str, Any] = {}
        val_res = evaluate_metrics(model, val_loader, device, max_batches=max_val_batches,
                                   cache_out=val_cache, split_name="val")
        val_f1_raw = val_res["raw_macro_f1"]
        val_acc = val_res["accuracy"]
        pd_source = val_res.get("per_source", {}).get("plantdoc")
        pv_source = val_res.get("per_source", {}).get("plantvillage")
        if args.mode == "full" and (pd_source is None or pv_source is None):
            raise ValueError("Full validation must include both PlantDoc and PlantVillage")
        pd_f1 = pd_source["raw_macro_f1_supported"] if pd_source else 0.0
        pv_f1 = pv_source["raw_macro_f1_supported"] if pv_source else 0.0
        pd_crop = val_res["quality"]["by_source"]["plantdoc"]["crop_accuracy"] or 0.0
        decision = quality_control.observe(epoch, pd_f1, pv_f1, pd_crop)
        lr_before = optimizer.param_groups[0]["lr"]
        if epoch == quality_control.policy["min_step"]:
            scheduler.step(quality_control.significant_best)
        elif epoch > quality_control.policy["min_step"]:
            scheduler.step(pd_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_dropped = quality_control.record_scheduler_step(epoch, lr_before, current_lr)
        decision["lr_dropped"] = lr_dropped
        decision["reduce_lr"] = lr_dropped
        decision["stop"] = quality_control.should_stop(epoch)

        # PlantDoc supported-class F1, guarded by the same-seed PlantVillage floor.
        is_best = decision["is_best"]
        if is_best:
            best_val_f1_raw = pd_f1
            best_cache_info = write_evaluation_cache(val_cache, output_dir, "best_validation")
        else:
            best_cache_info = None
        patience_counter = quality_control.bad_evaluations
        stop_requested = decision["stop"] if args.mode == "full" else False

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_res["loss"],
            "val_accuracy": val_acc,
            "val_macro_f1": round(val_f1_raw, 4),
            "val_macro_f1_raw": val_f1_raw,
            "plantdoc_macro_f1_supported": pd_f1,
            "plantvillage_macro_f1_supported": pv_f1,
            "plantdoc_crop_accuracy": pd_crop,
            "quality_eligible": decision["eligible"],
            "quality": val_res["quality"],
            "scene_metrics": summarize_scene_predictions(
                [
                    {
                        "sample_id": meta.get("sample_id", ""),
                        "group_id": meta.get("group_id", ""),
                        "source_domain": meta.get("source_domain", "unknown"),
                        "confidence": float(torch.softmax(torch.from_numpy(val_cache["logits"][index]), dim=0).max().item()),
                        "error_type": prediction_error_type(
                            PLANTVILLAGE_38_CLASSES[int(val_cache["targets"][index])],
                            PLANTVILLAGE_38_CLASSES[int(val_cache["logits"][index].argmax())],
                        ),
                    }
                    for index, meta in enumerate(val_cache["metadata"])
                ]
            ),
            "lr": current_lr,
            "lr_before_scheduler": lr_before,
            "lr_dropped": lr_dropped,
            "actual_lr_drop_count": quality_control.lr_drops,
            "duration_sec": round(ep_duration, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "samples_processed": samples_processed,
            "optimizer_steps_attempted": steps,
            "optimizer_steps_applied": steps_applied,
            "optimizer_steps_skipped": steps_skipped,
            "sampler_statistics": ({key: value for key, value in sampler_statistics.items()
                                     if key != "draws_by_scene"}
                                    if sampler_statistics is not None else None),
            "best_validation_cache": best_cache_info,
            "is_best": is_best,
        }
        history.append(epoch_record)

        progress.summary(
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val F1: {val_f1_raw:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"VRAM: {peak_vram_mb:.1f} MB | Time: {ep_duration:.1f}s | {'BEST' if is_best else ''}"
        )

        # Save atomic checkpoints
        progress.update(phase="checkpoint", position=f"epoch {epoch}/{max_epochs}", loss=avg_train_loss,
                        validation=val_res.get("per_source", {}).get("plantdoc", {}).get("raw_macro_f1_supported"),
                        lr=current_lr, vram_mb=peak_vram_mb)
        state = {
            "epoch": epoch,
            "mode": args.mode,
            "condition": args.condition,
            "seed": args.seed,
            "w0_fingerprint": w0_fingerprint,
            "release_manifest_sha256": release_manifest_sha,
            "resolved_config": resolved_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if args.amp else None,
            "best_val_f1": round(best_val_f1_raw, 4),
            "best_val_f1_raw": best_val_f1_raw,
            "patience_counter": patience_counter,
            "quality_control_state": quality_control.state_dict(),
            "sampler_state": sampler.state_dict() if sampler is not None else None,
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
            state=state,
            step=epoch,
            is_best=is_best,
            is_fedavg=False,
        )

        # Check session budget
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch and epoch < max_epochs:
            logging.info("Session budget reached at epoch %s", epoch)
            session_paused = True
            progress_status = "PAUSED_CHECKPOINTED"
        else:
            session_paused = False
            progress_status = "TRAINING"

        # Save progress JSON for real-time monitoring BEFORE syncing to remote
        progress_info = {
            "status": progress_status,
            "condition": args.condition,
            "seed": args.seed,
            "epoch": epoch,
            "max_epochs": max_epochs,
            "val_macro_f1": round(val_f1_raw, 4),
            "val_accuracy": val_acc,
            "best_val_f1": round(best_val_f1_raw, 4),
            "patience_counter": patience_counter,
            "peak_vram_mb": round(peak_vram_mb, 1),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (output_dir / "progress.json").write_text(canonical_json_dumps(progress_info), encoding="utf-8")

        if colab_sync is not None:
            logging.info("Synchronizing epoch %s to remote Drive destination", epoch)
            colab_sync.sync_epoch(epoch=epoch, is_best=is_best)

        if session_paused:
            break

        if args.mode == "full" and stop_requested:
            logging.info("Convergence early stop at epoch %s", epoch)
            break

    progress.close()
    total_time = time.monotonic() - t_run_start

    # Post-training evaluation: LOAD VERIFIED BEST CHECKPOINT FOR FINAL EVALUATION
    print("\n[Post-Training] Loading verified BEST checkpoint for final evaluation...")
    best_ckpt, best_ckpt_path = load_verified_best_checkpoint(
        output_dir=output_dir,
        expected_config=resolved_config,
        is_fedavg=False,
        allow_legacy=args.allow_legacy,
    )
    model.load_state_dict(best_ckpt["model_state"])
    best_epoch = best_ckpt.get("epoch", -1)
    best_ckpt_sha = compute_file_sha256(best_ckpt_path)

    print("[Post-Training] Evaluating verified BEST checkpoint on Validation set...")
    final_val_cache: Dict[str, Any] = {}
    final_best_val_res = evaluate_metrics(
        model,
        val_loader,
        device,
        max_batches=max_val_batches,
        cache_out=final_val_cache,
        split_name="val",
    )
    final_best_cache_info = write_evaluation_cache(final_val_cache, output_dir, "best_validation")
    train_eval_res = None
    train_validation_gap = None
    if args.mode == "full" and not session_paused:
        print("[Post-Training] Measuring train-validation gap on all original train images...")
        train_eval_ds = CanonicalClassificationDataset.from_manifest(
            train_manifest, dataset_root=dataset_root, transform=eval_transform, return_metadata=True)
        train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=0, collate_fn=eval_collate_fn)
        train_eval_res = evaluate_metrics(model, train_eval_loader, device)
        train_validation_gap = {
            domain: train_eval_res["per_source"][domain]["raw_macro_f1_supported"]
                    - final_best_val_res["per_source"][domain]["raw_macro_f1_supported"]
            for domain in ("plantdoc", "plantvillage")
        }
        train_validation_gap["macro_f1_38"] = (train_eval_res["raw_macro_f1"]
                                                - final_best_val_res["raw_macro_f1"])

    if args.mode == "full" and not session_paused and quality_control.best_eligible:
        print("[Post-Training] Evaluating verified BEST checkpoint on Legacy Diagnostic Test set...")
        test_res = evaluate_metrics(model, test_loader, device)
        status_str = "CONVERGED_EARLY" if stop_requested else "MAX_BUDGET_REACHED"
    elif args.mode == "full" and not session_paused:
        test_res = "SKIPPED_QUALITY_GATE_FAILED"
        status_str = "QUALITY_GATE_FAILED"
    elif session_paused:
        print(f"\n[Post-Training] Session paused at epoch {last_completed_epoch}. Skipping legacy diagnostic test evaluation.")
        test_res = "SKIPPED_SESSION_PAUSED"
        status_str = "PAUSED_CHECKPOINTED"
    else:
        # Strictly NO test evaluation in smoke or pilot modes!
        print(f"\n[Post-Training] Mode={args.mode.upper()}: Skipping legacy diagnostic test evaluation to prevent benchmark leakage.")
        test_res = "SKIPPED_IN_PILOT_OR_SMOKE"
        status_str = f"{args.mode.upper()}_COMPLETED"

    calibration_result: Union[str, Dict[str, Any]] = "NOT_REQUESTED"
    if args.prepare_calibration:
        if not quality_control.best_eligible or session_paused:
            calibration_result = "SKIPPED_NO_ELIGIBLE_FINAL_BEST"
        else:
            print("[Post-Training] Preparing temperature calibration from calibration split only...")
            calibration_cache: Dict[str, Any] = {}
            evaluate_metrics(
                model,
                calibration_loader,
                device,
                cache_out=calibration_cache,
                split_name="calibration",
            )
            calibration_cache_info = write_evaluation_cache(
                calibration_cache, output_dir, "calibration"
            )
            calibration_result = fit_temperature_from_cache(calibration_cache)
            calibration_result.update({
                "checkpoint_sha256": best_ckpt_sha,
                "release_manifest_sha256": release_manifest_sha,
                "cache": calibration_cache_info,
            })
            calibration_path = output_dir / "calibration_artifact.json"
            calibration_tmp = calibration_path.with_suffix(".json.tmp")
            calibration_tmp.write_text(canonical_json_dumps(calibration_result), encoding="utf-8")
            os.replace(calibration_tmp, calibration_path)

    summary = {
        "status": status_str,
        "final_holdout_status": "FINAL_HOLDOUT_PENDING",
        "mode": args.mode,
        "condition": args.condition,
        "seed": args.seed,
        "w0_fingerprint": w0_fingerprint,
        "release_manifest_sha256": release_manifest_sha,
        "total_runtime_seconds": round(total_time, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0, 1),
        "total_samples": len(train_ds),
        "best_epoch": best_epoch,
        "best_checkpoint_sha256": best_ckpt_sha,
        "best_val_f1": round(best_val_f1_raw, 4),
        "quality_gate_status": resolved_config["quality_gate_status"],
        "quality_floor": resolved_config["quality_floor"],
        "incumbent_full": resolved_config["incumbent_full"],
        "history": history,
        "best_checkpoint_validation": final_best_val_res,
        "best_checkpoint_validation_cache": final_best_cache_info,
        "best_checkpoint_train": train_eval_res,
        "train_validation_gap": train_validation_gap,
        "calibration": calibration_result,
        "legacy_diagnostic_test": test_res,
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(canonical_json_dumps(summary), encoding="utf-8")
    if not summary_file.is_file() or summary_file.stat().st_size == 0:
        raise RuntimeError(f"Failed to verify written summary file at {summary_file}")
    print(f"\nExecution finished! Summary written to {summary_file}")

    if colab_sync is not None:
        print(f"[Colab Sync] Synchronizing final summary to remote Drive destination...")
        colab_sync.sync_final(summary_file)


if __name__ == "__main__":
    main()

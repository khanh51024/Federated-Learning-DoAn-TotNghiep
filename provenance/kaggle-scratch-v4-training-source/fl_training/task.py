"""
fl_training.task: Local client training loop and server evaluation routines.
Does NOT own a progress bar.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import check_parameters_finite, get_model_state_dict_cpu
from .budget import check_deadline


def train_local(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    amp: bool = False,
    round_num: int = 1,
    on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Perform local client training.
    AMP wraps forward + loss; GradScaler used on CUDA.
    Returns:
      state_dict_cpu: OrderedDict of CPU tensors
      metrics: dictionary conforming to Section 6 message contract
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(reduction="mean")

    use_cuda_amp = amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp) if use_cuda_amp else None

    loss_sum = 0.0
    correct = 0
    processed_examples = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    epoch_metrics = []
    t0 = time.time()

    for ep in range(epochs):
        if hasattr(loader, "set_epoch"):
            loader.set_epoch((round_num - 1) * epochs + ep)

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0
        epoch_steps = 0
        epoch_skipped_steps = 0
        epoch_t0 = time.time()
        for batch_idx, (images, labels) in enumerate(loader):
            check_deadline()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            optimizer.zero_grad()
            step_applied = True

            if use_cuda_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss ({loss.item()}) at round {round_num}, epoch {ep}, batch {batch_idx}")

                scaler.scale(loss).backward()  # type: ignore
                scale_before = scaler.get_scale()  # type: ignore
                scaler.step(optimizer)  # type: ignore
                scaler.update()  # type: ignore
                # GradScaler lowers its scale when non-finite gradients skip SGD.
                step_applied = scaler.get_scale() >= scale_before  # type: ignore
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss ({loss.item()}) at round {round_num}, epoch {ep}, batch {batch_idx}")

                loss.backward()
                optimizer.step()

            loss_sum += float(loss.item()) * batch_size
            preds = outputs.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            processed_examples += batch_size
            optimizer_steps += int(step_applied)
            skipped_optimizer_steps += int(not step_applied)
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_correct += int((preds == labels).sum().item())
            epoch_examples += batch_size
            epoch_steps += int(step_applied)
            epoch_skipped_steps += int(not step_applied)

        epoch_record = {
            "round": int(round_num),
            "local_epoch": int(ep + 1),
            "processed_examples": int(epoch_examples),
            "optimizer_steps": int(epoch_steps),
            "skipped_optimizer_steps": int(epoch_skipped_steps),
            "loss": float(epoch_loss_sum / epoch_examples) if epoch_examples else 0.0,
            "accuracy": float(epoch_correct / epoch_examples) if epoch_examples else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": float(time.time() - epoch_t0),
            "started_at_unix": float(epoch_t0),
            "ended_at_unix": float(time.time()),
        }
        epoch_metrics.append(epoch_record)
        if on_epoch_end is not None:
            on_epoch_end(epoch_record)

    duration = time.time() - t0

    state_dict_cpu = get_model_state_dict_cpu(model)
    check_parameters_finite(state_dict_cpu)

    metrics = {
        "loss-sum": float(loss_sum),
        "correct": int(correct),
        "processed-examples": int(processed_examples),
        "optimizer-steps": int(optimizer_steps),
        "skipped-optimizer-steps": int(skipped_optimizer_steps),
        "local-epochs": int(epochs),
        "duration-seconds": float(duration),
        "epoch-metrics": epoch_metrics,
    }

    return state_dict_cpu, metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    num_classes: int = 38,
) -> Dict[str, Any]:
    """
    Evaluate model on validation or test set:
    - model.eval() and torch.inference_mode()
    - verifies BatchNorm statistics are NOT modified during evaluation
    - accumulates 38x38 confusion matrix
    - calculates loss, top-1 accuracy, macro-F1, precision, recall
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    # Capture BatchNorm running stats before evaluation
    bn_stats_before: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.running_mean is not None:
                bn_stats_before[f"{name}.running_mean"] = module.running_mean.clone()
            if module.running_var is not None:
                bn_stats_before[f"{name}.running_var"] = module.running_var.clone()
            if module.num_batches_tracked is not None:
                bn_stats_before[f"{name}.num_batches_tracked"] = module.num_batches_tracked.clone()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.inference_mode():
        for images, labels in loader:
            check_deadline()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            outputs = model(images)
            if outputs.shape != (batch_size, num_classes):
                raise ValueError("Evaluation logits must have shape [batch, num_classes]")
            if not torch.isfinite(outputs).all():
                raise FloatingPointError("Non-finite evaluation logits")
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")

            total_loss += float(loss.item())
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

            # Accumulate confusion matrix (rows = true labels, cols = predicted)
            true_np = labels.cpu().numpy()
            pred_np = preds.cpu().numpy()
            for t, p in zip(true_np, pred_np):
                confusion_matrix[t, p] += 1

    # Verify BatchNorm stats did NOT change
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.running_mean is not None:
                if not torch.equal(module.running_mean, bn_stats_before[f"{name}.running_mean"]):
                    raise AssertionError(f"BatchNorm running_mean modified during evaluation for module {name}")
            if module.running_var is not None:
                if not torch.equal(module.running_var, bn_stats_before[f"{name}.running_var"]):
                    raise AssertionError(f"BatchNorm running_var modified during evaluation for module {name}")
            if module.num_batches_tracked is not None:
                if not torch.equal(module.num_batches_tracked, bn_stats_before[f"{name}.num_batches_tracked"]):
                    raise AssertionError(f"BatchNorm num_batches_tracked modified during evaluation for module {name}")

    if total_samples == 0:
        raise ValueError("Evaluation loader yielded 0 samples")

    val_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # Compute per-class precision, recall, F1
    per_class: List[Dict[str, Any]] = []
    f1_list: List[float] = []

    for c in range(num_classes):
        tp = int(confusion_matrix[c, c])
        fn = int(confusion_matrix[c, :].sum() - tp)
        fp = int(confusion_matrix[:, c].sum() - tp)
        support = int(confusion_matrix[c, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)

        per_class.append({
            "class_id": c,
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    macro_f1 = float(np.mean(f1_list))

    return {
        "loss": float(val_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "weighted_f1": float(sum(row["f1"] * row["support"] for row in per_class) / total_samples),
        "classes_with_support": sum(row["support"] > 0 for row in per_class),
        "averaging_policy": "macro over all configured classes; zero_division=0",
        "total_samples": int(total_samples),
        "total_correct": int(total_correct),
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class": per_class,
    }

"""
src.centralized.evaluate: Đánh giá mô hình trên tập kiểm thử chung (Global Test Set).
Tính toán Accuracy, Macro-F1, Loss, per-class metrics và Confusion Matrix 38x38.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 38,
) -> Dict[str, Any]:
    """Đánh giá toàn diện trên tập dữ liệu kiểm thử."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)

    # Tính macro-F1 & Confusion Matrix 38 lớp
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(all_preds, all_labels):
        if 0 <= l < num_classes and 0 <= p < num_classes:
            confusion_matrix[l, p] += 1

    f1_scores = []
    per_class_metrics = {}
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        per_class_metrics[str(c)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(confusion_matrix[c, :].sum()),
        }

    macro_f1 = float(np.mean(f1_scores))

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class_metrics": per_class_metrics,
    }

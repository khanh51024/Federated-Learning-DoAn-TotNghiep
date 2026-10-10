from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader: Iterable, optimizer, criterion, target_device: torch.device, description: str | None = None) -> float:
    model.train()
    loss_sum = 0.0
    total = 0
    batches = tqdm(loader, desc=description, leave=False, dynamic_ncols=True) if description else loader
    for images, labels in batches:
        images, labels = images.to(target_device), labels.to(target_device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
        if description:
            batches.set_postfix(loss=f"{loss_sum / total:.4f}")
    return loss_sum / max(total, 1)


@torch.inference_mode()
def evaluate(model, loader: Iterable, criterion, target_device: torch.device) -> dict:
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(target_device), labels.to(target_device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
    return {
        "loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples": total,
    }

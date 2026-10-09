"""
train.centralized: Huấn luyện mô hình tập trung (Centralized baseline)
và cung cấp hàm inference nhận diện ảnh bệnh lá cho Web demo.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PlantVillageDataset, get_default_transform
from .model import create_model, get_parameters, set_parameters


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 38,
) -> Dict[str, Any]:
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
        per_class_metrics[str(c)] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": int(confusion_matrix[c, :].sum())}

    macro_f1 = float(np.mean(f1_scores))

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class_metrics": per_class_metrics,
    }


def run_centralized_training(
    dataset_root: Optional[str | Path] = None,
    train_csv: Optional[str | Path] = None,
    val_csv: Optional[str | Path] = None,
    test_csv: Optional[str | Path] = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    pretrained: bool = False,
    smoke: bool = False,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir) if output_dir else Path("artifacts")
    checkpoints_dir = output_path / "checkpoints"
    metrics_dir = output_path / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Thiết lập đường dẫn mặc định từ partitions_stage2_scratch_v3
    base_part = Path("data/partitions_stage2_scratch_v3/label100")
    train_csv = Path(train_csv) if train_csv else base_part / "centralized_train.csv"
    val_csv = Path(val_csv) if val_csv else base_part / "global_val.csv"
    test_csv = Path(test_csv) if test_csv else base_part / "global_test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Centralized Training: device={device}, epochs={epochs}, batch_size={batch_size}, pretrained={pretrained}")

    train_ds = PlantVillageDataset(train_csv, dataset_root=dataset_root, is_train=True)
    val_ds = PlantVillageDataset(val_csv, dataset_root=dataset_root, is_train=False)
    test_ds = PlantVillageDataset(test_csv, dataset_root=dataset_root, is_train=False)

    if smoke:
        train_ds.samples = train_ds.samples[:64]
        val_ds.samples = val_ds.samples[:38]
        test_ds.samples = test_ds.samples[:38]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = create_model(num_classes=38, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "train_accuracy": round(t_acc, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        print(f"Epoch {epoch}/{epochs}: train_loss={t_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(get_parameters(model))

    total_duration = round(time.time() - start_time, 2)

    # Đánh giá 1 lần cuối trên fixed test set
    if best_state is not None:
        set_parameters(model, best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Lưu checkpoint. Smoke/không có output_dir riêng KHÔNG BAO GIỜ
    # được ghi vào slot canonical (bài học 25/09: smoke ghi đè centralized_best.pt).
    if smoke and not output_dir:
        smoke_ckpt_dir = checkpoints_dir / "smoke"
        smoke_ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = smoke_ckpt_dir / "centralized_best_smoke.pt"
    else:
        checkpoint_file = checkpoints_dir / "centralized_best.pt"
    torch.save({
        "model_state": best_state or get_parameters(model),
        "test_metrics": test_metrics,
        "pretrained": pretrained,
        "epochs": epochs,
    }, checkpoint_file)

    results_payload = {
        "experiment": "centralized",
        "pretrained": pretrained,
        "epochs": epochs,
        "total_train_time_s": total_duration,
        "test_metrics": test_metrics,
        "history": history,
    }

    out_file = metrics_dir / "centralized_results.json"
    if smoke and out_file.exists() and not output_dir:
        try:
            with open(out_file, "r", encoding="utf-8") as existing_f:
                existing_data = json.load(existing_f)
                if existing_data.get("epochs", 0) > 1:
                    out_file = metrics_dir / "smoke_centralized_results.json"
        except Exception:
            pass

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"[+] Centralized hoàn tất! Test Accuracy: {test_metrics['accuracy']*100:.2f}%, F1: {test_metrics['macro_f1']:.4f}")
    return results_payload


try:
    from src.centralized.inference import predict_image
except ImportError:
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_centralized_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
        smoke=args.smoke,
    )

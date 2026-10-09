"""
src.centralized.train: Huấn luyện mô hình tập trung (Centralized baseline) trên MobileNetV3-Small.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.plantvillage import PlantVillageDataset
from src.model.mobilenetv3 import create_model, get_parameters, set_parameters
from .evaluate import evaluate


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
    require_real_data: bool = False,
    seed: int = 42,
    output_dir: Optional[str | Path] = None,
    step_callback: Optional[Any] = None,
    is_cancelled: Optional[Any] = None,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_path = Path(output_dir) if output_dir else Path("artifacts")
    checkpoints_dir = output_path / "checkpoints"
    metrics_dir = output_path / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_part = Path("data/partitions_stage2_scratch_v3/label100")
    train_csv = Path(train_csv) if train_csv else base_part / "centralized_train.csv"
    val_csv = Path(val_csv) if val_csv else base_part / "global_val.csv"
    test_csv = Path(test_csv) if test_csv else base_part / "global_test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Centralized Training: device={device}, epochs={epochs}, batch_size={batch_size}, pretrained={pretrained}")

    train_ds = PlantVillageDataset(
        train_csv,
        dataset_root=dataset_root,
        is_train=True,
        require_real_images=require_real_data,
    )
    val_ds = PlantVillageDataset(
        val_csv,
        dataset_root=dataset_root,
        is_train=False,
        require_real_images=require_real_data,
    )
    test_ds = PlantVillageDataset(
        test_csv,
        dataset_root=dataset_root,
        is_train=False,
        require_real_images=require_real_data,
    )

    if smoke:
        # Chọn mẫu phân bổ đều theo 38 lớp thay vì lấy dồn ở đầu danh sách (tránh lệch nhãn 1 class)
        def _get_balanced_subset(samples_list: list, max_per_class: int = 1) -> list:
            buckets: Dict[int, list] = {}
            for s in samples_list:
                buckets.setdefault(s["label"], []).append(s)
            subset = []
            for lbl in sorted(buckets.keys()):
                subset.extend(buckets[lbl][:max_per_class])
            return subset

        train_ds.samples = _get_balanced_subset(train_ds.samples, max_per_class=2)  # 76 samples
        val_ds.samples = _get_balanced_subset(val_ds.samples, max_per_class=1)      # 38 samples
        test_ds.samples = _get_balanced_subset(test_ds.samples, max_per_class=1)    # 38 samples

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_generator)
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
        if is_cancelled and is_cancelled():
            print(f"[!] Nhận được yêu cầu dừng tại Epoch {epoch}. Dừng huấn luyện.")
            break

        epoch_start = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = round(time.time() - epoch_start, 2)

        epoch_record = {
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": round(t_loss, 4),
            "train_accuracy": round(t_acc, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_time_s": epoch_time,
            "status": "completed",
        }
        history.append(epoch_record)
        print(f"Epoch {epoch}/{epochs}: train_loss={t_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['macro_f1']:.4f}")

        if step_callback:
            try:
                step_callback(epoch_record)
            except Exception:
                pass

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(get_parameters(model))

        if is_cancelled and is_cancelled():
            print(f"[!] Nhận được yêu cầu dừng sau Epoch {epoch}. Dừng huấn luyện.")
            break

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
        "seed": seed,
    }, checkpoint_file)

    results_payload = {
        "experiment": "centralized",
        "pretrained": pretrained,
        "epochs": epochs,
        "seed": seed,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_centralized_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
        smoke=args.smoke,
        seed=args.seed,
    )

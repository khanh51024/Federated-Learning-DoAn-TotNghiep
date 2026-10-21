"""Vòng học một epoch và suy luận lấy logits; dùng chung cho Centralized, client FedAvg và Local-only."""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def make_loader(dataset: Any, batch_size: int, shuffle: bool, workers: int, seed: int, persistent: bool = False) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False, generator=generator, persistent_workers=persistent and workers > 0,
                      worker_init_fn=_seed_worker if workers > 0 else None)


def _seed_worker(worker_id: int) -> None:
    import random

    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_optimizer(model: Any, lr: float, momentum: float, weight_decay: float) -> Any:
    import torch

    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def cosine_lr(base_lr: float, step: int, total: int, warmup: int = 0) -> float:
    """Tốc độ học theo bước ngoài (epoch của Centralized/Local-only, vòng của FedAvg): khởi động tuyến tính rồi cosine về 0."""
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_epoch(model: Any, loader: Any, optimizer: Any, scaler: Any, device: Any, sample_weights: Any, label_smoothing: float) -> dict[str, float]:
    """Một lượt qua `loader`. Mất mát = cross-entropy (label smoothing) có trọng số từng mẫu, chuẩn hóa theo tổng trọng số của batch."""
    import torch

    model.train()
    criterion = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
    use_amp = device.type == "cuda"
    total_loss, correct, seen, steps = 0.0, 0, 0, 0
    started = time.time()
    for images, labels, index in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = sample_weights[index].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = (criterion(logits.float(), labels) * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("mất mát không hữu hạn: dừng để không lưu checkpoint hỏng")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        seen += len(labels)
        steps += 1
    return {"loss": total_loss / max(1, seen), "train_top1": correct / max(1, seen), "steps": steps, "images": seen,
            "seconds": round(time.time() - started, 1)}


def predict(model: Any, loader: Any, device: Any) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros((len(loader.dataset), 38), dtype=np.float32)
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, _, index in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images.to(device, non_blocking=True))
            out[index.numpy()] = logits.float().cpu().numpy()
    return out

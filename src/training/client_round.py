"""Phần việc của một client trong một vòng học liên kết; dùng chung cho mô phỏng một tiến trình (scripts/train_v4.py) và ClientApp của Flower (src/federated/flower_app.py).

Hai đường gọi đúng các hàm này nên khác biệt giữa chúng chỉ còn là cách truyền trọng số và cách lưu trạng thái, không phải công thức học."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.training.data import ManifestDataset, balanced_sampling_weights, class_prior, domain_balanced_weights
from src.training.loop import logit_adjustment, make_loader, make_optimizer, train_epoch

AUG_DESCRIPTION = {
    "robust_v1": "robust_v1 (src/data/robust_transforms.py)",
    "gpu_v2": "gpu_v2 (get_cpu_train_transform + src/training/gpu_augment.py)",
    "gpu_v2_ta": "gpu_v2_ta (get_cpu_train_transform với TrivialAugmentWide + src/training/gpu_augment.py, bỏ phần màu/xoay trên GPU)",
}


@dataclass(frozen=True)
class Recipe:
    """Những gì một client cần biết để học; là một phần của công thức chung trong config.json của run."""
    dataset_root: str
    seed: int
    batch_size: int
    momentum: float
    weight_decay: float
    label_smoothing: float
    local_epochs: int = 1
    workers: int = 8
    aug: str = "robust_v1"
    train_size: int = 224
    sampling: str = "weights"
    class_power: float = 1.0
    logit_adjust: float = 0.0
    channels_last: bool = False
    amp: str = "fp16"


def train_transform(recipe: Recipe) -> Any:
    from src.data.robust_transforms import get_cpu_train_transform, get_robust_train_transform

    if recipe.aug == "robust_v1":
        return get_robust_train_transform(recipe.train_size)
    return get_cpu_train_transform(recipe.train_size, trivial_augment=recipe.aug == "gpu_v2_ta")


def prepare_client(rows: list[dict[str, Any]], index: int, recipe: Recipe, transform: Any, device: Any, persistent: bool) -> dict[str, Any]:
    """Bộ nạp ảnh, trọng số mất mát, tăng cường trên GPU và logit adjustment của một tập học. `index` là số thứ tự client (Centralized dùng 0)."""
    import torch

    from src.training.gpu_augment import GpuAugment

    if recipe.sampling == "balanced":
        sampling = balanced_sampling_weights(rows, recipe.class_power)
        weights = torch.ones(len(rows), dtype=torch.float32)                 # cân bằng đã nằm ở bộ lấy mẫu
        seen_weights = sampling
    else:
        sampling = None
        weights = torch.tensor(domain_balanced_weights(rows), dtype=torch.float32)
        seen_weights = weights.tolist()
    # logit adjustment theo tần suất lớp mà chính tập học (với FedAvg/Local: shard của client) thực sự đưa vào mất mát
    adjust = logit_adjustment(class_prior(rows, seen_weights), recipe.logit_adjust)
    loader = make_loader(ManifestDataset(rows, Path(recipe.dataset_root), transform), recipe.batch_size, True, recipe.workers, recipe.seed * 100 + index,
                         persistent=persistent, sampling_weights=sampling)
    generator, augment = None, None
    if recipe.aug != "robust_v1":
        generator = torch.Generator(device=device)
        generator.manual_seed(recipe.seed * 1000 + index)
        augment = GpuAugment(generator, color=recipe.aug == "gpu_v2", rotate=recipe.aug == "gpu_v2")
    return {"loader": loader, "weights": weights, "adjust": adjust, "generator": generator, "augment": augment, "num_examples": len(rows)}


def train_one_client(local: Any, client: int, parent: dict[str, Any], algo: Any, data: dict[str, Any], lr: float, recipe: Recipe, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Một client học một vòng từ trạng thái `local` (đang giữ mô hình chung), bộ tối ưu làm mới mỗi vòng. Trả (trọng số sau khi học, số liệu vòng của client).

    Byte tính từ kích thước tensor gửi thô; đường Flower thay bằng kích thước thật của gói tin."""
    import torch

    from src.federated.algorithms import state_bytes, update_norms
    from src.model.mobilenetv3 import get_parameters, set_parameters

    start_state = algo.start_state(client, parent)
    if start_state is not parent:
        set_parameters(local, start_state)                                   # FedBN: BatchNorm riêng của client
    hooks = algo.make_hooks(client, local, parent, lr, device)
    optimizer = make_optimizer(local, lr, recipe.momentum, recipe.weight_decay)   # trạng thái bộ tối ưu của client làm mới mỗi vòng
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and recipe.amp == "fp16")
    stats = [train_epoch(local, data["loader"], optimizer, scaler, device, data["weights"], recipe.label_smoothing, data["augment"], data["adjust"], recipe.channels_last,
                         recipe.amp, hooks=hooks) for _ in range(recipe.local_epochs)]
    steps = sum(s["steps"] for s in stats)
    state = get_parameters(local)
    l2, l2_without_bn = update_norms(state, parent, algo.bn_keys)
    extra_up, extra_down = algo.extra_bytes(parent)
    algo.after_local(client, local, parent, lr, steps, data["num_examples"])
    if hooks is not None:
        hooks.close()
    return state, {"client": client, "loss": stats[-1]["loss"], "steps": steps, "delta_l2": round(l2, 4), "delta_l2_without_bn": round(l2_without_bn, 4),
                   "bytes_up": state_bytes(state) + extra_up, "bytes_down": state_bytes(parent) + extra_down}

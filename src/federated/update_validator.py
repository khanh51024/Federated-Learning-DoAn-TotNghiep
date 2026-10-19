"""Server-side validation utilities for federated model updates.

The server must never trust client-reported update statistics.  This module
recomputes integrity and anomaly signals from the actual received payload.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable

import numpy as np
import torch

from src.model.mobilenetv3 import compute_model_hash


TensorLike = torch.Tensor | np.ndarray


@dataclass
class UpdateValidationResult:
    accepted: bool
    suspicious: bool
    reason: str
    received_update_norm: float
    parent_hash_ok: bool
    finite_ok: bool
    structure_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_tensor(value: TensorLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.from_numpy(np.asarray(value))


def compute_update_norm(
    parent_weights: Dict[str, TensorLike],
    client_weights: Dict[str, TensorLike],
) -> float:
    """Compute ||W_client - W_parent||_2 over all floating-point tensors."""
    if set(parent_weights) != set(client_weights):
        raise ValueError("Parameter key mismatch between parent and client update")

    delta_sq = 0.0
    for key in parent_weights:
        parent = _to_tensor(parent_weights[key])
        child = _to_tensor(client_weights[key])
        if parent.shape != child.shape:
            raise ValueError(f"Shape mismatch for '{key}': {tuple(child.shape)} != {tuple(parent.shape)}")
        if parent.is_floating_point() or child.is_floating_point():
            diff = child.float() - parent.float()
            delta_sq += float(torch.sum(diff * diff).item())
    return float(np.sqrt(delta_sq))


def compute_subset_update_norm(
    parent_weights: Dict[str, TensorLike],
    client_weights: Dict[str, TensorLike],
    keys: Iterable[str],
) -> float:
    """||W_client - W_parent||_2 chỉ trên các khóa `keys` (bỏ qua tensor không phải số thực).

    `compute_update_norm` cộng cả bộ đệm BatchNorm (running_mean/running_var) nên khi ảnh vào mô hình chúng tự trôi
    dù trọng số học không đổi. Hàm này cho phép đo riêng phần THAM SỐ HỌC ĐƯỢC mà không đổi ngữ nghĩa của hàm cũ."""
    delta_sq = 0.0
    for key in keys:
        if key not in parent_weights or key not in client_weights:
            raise ValueError(f"Missing parameter key '{key}'")
        parent = _to_tensor(parent_weights[key])
        child = _to_tensor(client_weights[key])
        if parent.shape != child.shape:
            raise ValueError(f"Shape mismatch for '{key}': {tuple(child.shape)} != {tuple(parent.shape)}")
        if parent.is_floating_point() or child.is_floating_point():
            diff = child.float() - parent.float()
            delta_sq += float(torch.sum(diff * diff).item())
    return float(np.sqrt(delta_sq))


def learnable_update_stats(
    parent_weights: Dict[str, TensorLike],
    client_weights: Dict[str, TensorLike],
    learnable_keys: Iterable[str],
) -> Dict[str, float]:
    """Mức thay đổi của tham số học được: {"norm": ||ΔW||, "parent_norm": ||W||, "relative": ||ΔW|| / ||W||}."""
    keys = list(learnable_keys)
    norm = compute_subset_update_norm(parent_weights, client_weights, keys)
    zero = {k: np.zeros_like(np.asarray(_to_tensor(parent_weights[k]))) for k in keys}
    parent_norm = compute_subset_update_norm(zero, parent_weights, keys)
    return {"norm": norm, "parent_norm": parent_norm, "relative": norm / parent_norm if parent_norm > 0 else 0.0}


def validate_client_update(
    parent_weights: Dict[str, TensorLike],
    client_weights: Dict[str, TensorLike],
    *,
    reported_parent_hash: str | None = None,
    max_update_norm: float | None = None,
    zero_update_tolerance: float = 1e-12,
) -> UpdateValidationResult:
    """Validate an update using only facts independently verifiable by server.

    ``max_update_norm`` is deliberately optional because a safe threshold is
    experiment-specific.  When set, exceeding it marks the payload suspicious
    but keeps structure/integrity validation separate from policy.
    """
    expected_parent_hash = compute_model_hash(parent_weights)
    parent_hash_ok = reported_parent_hash == expected_parent_hash

    if set(parent_weights) != set(client_weights):
        return UpdateValidationResult(False, True, "parameter_keys_mismatch", 0.0, parent_hash_ok, False, False)

    for key in parent_weights:
        parent = _to_tensor(parent_weights[key])
        child = _to_tensor(client_weights[key])
        if parent.shape != child.shape:
            return UpdateValidationResult(False, True, f"shape_mismatch:{key}", 0.0, parent_hash_ok, False, False)
        if parent.dtype != child.dtype:
            # Dtype differences are rejected because Flower/model state buffers
            # can include integer tensors whose semantics must not be changed.
            return UpdateValidationResult(False, True, f"dtype_mismatch:{key}", 0.0, parent_hash_ok, False, False)
        if child.is_floating_point() and not torch.isfinite(child).all():
            return UpdateValidationResult(False, True, f"non_finite:{key}", 0.0, parent_hash_ok, False, True)

    update_norm = compute_update_norm(parent_weights, client_weights)
    reasons = []
    suspicious = False

    if not parent_hash_ok:
        suspicious = True
        reasons.append("parent_hash_mismatch")
    if update_norm <= zero_update_tolerance:
        suspicious = True
        reasons.append("near_zero_update")
    if max_update_norm is not None and update_norm > max_update_norm:
        suspicious = True
        reasons.append("update_norm_above_threshold")

    return UpdateValidationResult(
        accepted=parent_hash_ok,
        suspicious=suspicious,
        reason=",".join(reasons) if reasons else "ok",
        received_update_norm=update_norm,
        parent_hash_ok=parent_hash_ok,
        finite_ok=True,
        structure_ok=True,
    )

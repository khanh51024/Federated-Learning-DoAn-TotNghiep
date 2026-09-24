"""
fl_training.model: MobileNetV3-Small architecture, parameter extraction, and validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def create_mobilenet_v3_small(
    num_classes: int = 38,
    weights: Optional[str] = None,
) -> nn.Module:
    """
    Create MobileNetV3-Small model.
    When pretrained:
      Load torchvision mobilenet_v3_small with IMAGENET1K_V1 weights (1000 classes),
      then replace classifier[3] with nn.Linear(in_features, num_classes).
      Never pass both pretrained weights and num_classes=38 to constructor.
    When weights=None:
      Instantiate directly with weights=None and num_classes=num_classes.
    """
    if weights == "IMAGENET1K_V1":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress=False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif weights is None or weights == "null":
        model = mobilenet_v3_small(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported weights specifier: {weights}")

    return model


def get_model_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return an OrderedDict of CPU copies of model parameters and buffers."""
    state = model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def set_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor | np.ndarray],
    strict: bool = True,
) -> None:
    """
    Load state dict into model. Converts numpy arrays to torch tensors if needed.
    """
    converted: Dict[str, torch.Tensor] = {}
    current_state = model.state_dict()

    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            t = v
        else:
            raise TypeError(f"Unexpected tensor type for key '{k}': {type(v)}")

        # Restore original dtype if needed (especially integer buffers like num_batches_tracked)
        if k in current_state:
            target_dtype = current_state[k].dtype
            if t.dtype != target_dtype:
                t = t.to(dtype=target_dtype)

        converted[k] = t

    model.load_state_dict(converted, strict=strict)


def check_parameters_finite(state_dict: Dict[str, Any]) -> None:
    """Verify that all floating point tensors in state_dict contain no NaN or Inf."""
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                if not torch.isfinite(v).all():
                    raise FloatingPointError(f"Non-finite values detected in parameter '{k}'")
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                if not np.isfinite(v).all():
                    raise FloatingPointError(f"Non-finite values detected in numpy parameter '{k}'")

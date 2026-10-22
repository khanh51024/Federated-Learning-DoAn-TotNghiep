"""Unified model creation and initialization for both Centralized and FedAvg runners.

Guarantees:
  1. Identical W0 initialization across Centralized and FedAvg for matching seed/track.
  2. Pinned ImageNet pretrained weights loading from local artifact or torch cache with SHA-256 validation.
  3. Fail-closed verification: raises error if pretrained file hash does not match approved hash.
  4. Returns model, w0_fingerprint, and metadata dictionary for resume/protocol verification.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from plant_data_contract.integrity import compute_model_w0_fingerprint

PINNED_PRETRAINED_FILE_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
PINNED_PRETRAINED_BASE_STATE_SHA256 = "3adfbce13df024027cd59bfe2370aed39563091eec20fc14cdf58f4c48bc08ca"
PINNED_SCRATCH_SEED42_W0 = "447c66b2dd0ddeaf33437091a31a2f646ac0f29eb958027ff6be3c4e525afa19"
PINNED_PRETRAINED_SEED42_W0 = "eaf9197b42fb39163186a7285f7f1ed4f1e728bd83522260fdc5f0f51c110850"


def find_pretrained_weights_file(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Locate mobilenet_v3_small pretrained weights across known directories."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p

    candidates = [
        Path("training-artifacts/pretrained_weights/mobilenet_v3_small-047dcff4.pth"),
        Path("../training-artifacts/pretrained_weights/mobilenet_v3_small-047dcff4.pth"),
        Path(os.path.expanduser("~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth")),
        Path("/kaggle/input/mobilenet-v3-small-weights/mobilenet_v3_small-047dcff4.pth"),
        Path("/kaggle/working/mobilenet_v3_small-047dcff4.pth"),
        Path("/content/mobilenet_v3_small-047dcff4.pth"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Pretrained weights file 'mobilenet_v3_small-047dcff4.pth' not found in: {[str(c) for c in candidates]}"
    )


def create_model(
    num_classes: int = 38,
    seed: int = 42,
    init: str = "scratch",
    pretrained_weights_path: Optional[Union[str, Path]] = None,
    w0_state_path: Optional[Union[str, Path]] = None,
    expected_w0_state_sha256: Optional[str] = None,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    """Create MobileNetV3-Small classifier with deterministic initialization.

    Args:
        num_classes: Number of classification targets (default: 38).
        seed: Random seed for head initialization and scratch weights.
        init: "scratch" or "imagenet_v1".
        pretrained_weights_path: Optional explicit path to pretrained weights file.

    Returns:
        (model, w0_fingerprint, model_metadata)
    """
    if init not in ("scratch", "imagenet_v1"):
        raise ValueError(f"Unknown init mode: {init}. Must be 'scratch' or 'imagenet_v1'.")

    torch.manual_seed(seed)

    if init == "scratch":
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        for param in model.parameters():
            param.requires_grad = True
        w0_fingerprint = compute_model_w0_fingerprint(model)
        metadata = {
            "init": "scratch",
            "seed": seed,
            "architecture": "mobilenet_v3_small",
            "num_classes": num_classes,
            "w0_fingerprint": w0_fingerprint,
        }
        return model, w0_fingerprint, metadata

    elif init == "imagenet_v1":
        weights_file = find_pretrained_weights_file(pretrained_weights_path)
        file_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()
        if file_sha != PINNED_PRETRAINED_FILE_SHA256:
            raise ValueError(
                f"Pretrained weights file SHA-256 mismatch! Got: {file_sha}, expected: {PINNED_PRETRAINED_FILE_SHA256}"
            )

        model = mobilenet_v3_small(weights=None)
        weights_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(weights_dict, strict=False)

        # Compute and verify base backbone fingerprint
        base_hasher = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            base_hasher.update(k.encode("utf-8"))
            base_hasher.update(v.cpu().numpy().tobytes())
        base_sha = base_hasher.hexdigest()
        assert base_sha == PINNED_PRETRAINED_BASE_STATE_SHA256, (
            f"Pretrained base state SHA mismatch: {base_sha} vs {PINNED_PRETRAINED_BASE_STATE_SHA256}"
        )

        # Replace classification head with deterministic Xavier initialization.
        # A pinned W0 state can override this version-dependent RNG result when
        # the cloud runtime has a different PyTorch release from the pilot.
        in_features = model.classifier[-1].in_features
        torch.manual_seed(seed)
        new_head = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(new_head.weight)
        nn.init.zeros_(new_head.bias)
        model.classifier[-1] = new_head

        if w0_state_path is not None:
            state_path = Path(w0_state_path).resolve()
            if not state_path.is_file() or not expected_w0_state_sha256:
                raise ValueError("Pinned W0 state file and SHA-256 are required together")
            state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
            if state_sha != expected_w0_state_sha256:
                raise ValueError("Pinned W0 state file SHA-256 mismatch")
            pinned_state = torch.load(state_path, map_location="cpu", weights_only=True)
            model.load_state_dict(pinned_state, strict=True)

        for param in model.parameters():
            param.requires_grad = True

        w0_fingerprint = compute_model_w0_fingerprint(model)
        metadata = {
            "init": "imagenet_v1",
            "seed": seed,
            "architecture": "mobilenet_v3_small",
            "num_classes": num_classes,
            "pretrained_weights_file": str(weights_file),
            "pretrained_file_sha256": file_sha,
            "pretrained_base_state_sha256": base_sha,
            "w0_fingerprint": w0_fingerprint,
        }
        if w0_state_path is not None:
            metadata["w0_state_sha256"] = state_sha
        return model, w0_fingerprint, metadata

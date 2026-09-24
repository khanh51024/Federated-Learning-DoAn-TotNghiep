"""Deterministic seed helpers shared by server, clients, and baselines."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def derive_seed(base_seed: int, *parts: object) -> int:
    material = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "little"
    ) % (2**31 - 1)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        # Set before any CUDA context/BLAS handle is initialized, including baselines.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

"""Q18 (PLAN r8 D4 screening, D4_SCREEN_SPEC r1) client-side method helpers: FedRS scale and the FedNTD loss.

Imported by the Q18 sandbox copy of fedavg_mixed_runner.py. SCAFFOLD (q8r2_methods) and the FedLC margin (q8_methods.lc_margin) are
reused unchanged; P1 = SCAFFOLD + FedLC margin, P2 = SCAFFOLD + FedRS scale, P3 = FedNTD on top of plain FedAvg aggregation.
Sources (advisor notes, STATE r8_prep.d4_screen_inputs): FedRS, Li & Zhan, KDD 2021 (restricted softmax; implemented from the
description, no author code); FedNTD, Lee et al., NeurIPS 2022, arXiv:2106.03097 (not-true distillation).
"""
from typing import Sequence

import torch
import torch.nn.functional as F


def fedrs_scale(targets: Sequence[int], num_classes: int, alpha: float, device) -> torch.Tensor:
    """FedRS scale c_k = (n_k / max_j n_j) * (1 - alpha) + alpha per class, shape (1, C), float32.

    n_k = raw label count of the client's partition shard (not sampler draws). Absent class -> c = alpha; the most frequent class
    -> c = 1. alpha = 1 gives c = 1 everywhere (neutral). The runner multiplies the whole logit vector (W h + b) by c (registered).
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"FedRS alpha must be in [0, 1], got {alpha}")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for t in targets:
        t = int(t)
        if t < 0 or t >= num_classes:
            raise ValueError(f"label {t} outside [0, {num_classes})")
        counts[t] += 1.0
    if float(counts.max()) <= 0:
        raise ValueError("FedRS: empty client")
    c = (counts / counts.max()) * (1.0 - float(alpha)) + float(alpha)
    return c.to(dtype=torch.float32).unsqueeze(0).to(device)


def not_true_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Drop the true-class column of every row: (B, C) -> (B, C - 1), order of the other classes kept."""
    b, c = logits.shape
    keep = torch.ones_like(logits, dtype=torch.bool)
    keep[torch.arange(b, device=logits.device), targets] = False
    return logits[keep].view(b, c - 1)


def ntd_loss(local_logits: torch.Tensor, global_logits: torch.Tensor, targets: torch.Tensor, tau: float) -> torch.Tensor:
    """FedNTD not-true distillation: tau^2 * KL(q_g || q_l), q = softmax(not-true logits / tau), KL summed over classes and averaged
    over the batch (KLDivLoss 'batchmean'). Computed in float32; the global logits carry no gradient."""
    zl = not_true_logits(local_logits.float(), targets) / float(tau)
    zg = not_true_logits(global_logits.detach().float(), targets) / float(tau)
    log_ql = F.log_softmax(zl, dim=1)
    log_qg = F.log_softmax(zg, dim=1)
    kl = (log_qg.exp() * (log_qg - log_ql)).sum(dim=1).mean()
    return (float(tau) ** 2) * kl

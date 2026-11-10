"""Q8-LAB server/client method helpers (FedProx, FedAvgM, FedNova, FedLC).

Imported by the Q8 sandbox copy of fedavg_mixed_runner.py. FedAvg itself is NOT implemented
here: the runner keeps calling its own aggregate_weights + verify_fedavg_round for fedavg,
fedprox and fedlc (client-side methods). Design: plans/cai-thien-train-20261015/Q8_DESIGN.md.

Conventions (Q8_DESIGN F3): server-side momentum / normalization apply only to trainable
parameters (named_parameters keys). BatchNorm running stats and integer buffers come from the
runner's FedAvg aggregate (n_k-weighted float64 average; int buffers base + sum(delta)).
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch

METHODS = ("fedavg", "fedprox", "fedavgm", "fednova", "fedlc")
LC_ZERO_COUNT = 1e-8  # FL-bench src/client/fedlc.py (commit a3acdf27) for absent classes


def lc_margin(targets: Sequence[int], num_classes: int, tau: float, device) -> torch.Tensor:
    """FedLC margin tau * n_c^(-1/4) per class, shape (1, C), float32.

    n_c = raw label count of the client's partition shard (not sampler draws); absent class
    -> 1e-8 (FL-bench). Loss used by the runner: cross_entropy(logits - margin, y), i.e. the
    FL-bench form (true class kept in the denominator), computed with log-softmax.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for t in targets:
        t = int(t)
        if t < 0 or t >= num_classes:
            raise ValueError(f"label {t} outside [0, {num_classes})")
        counts[t] += 1.0
    counts = torch.where(counts > 0, counts, torch.full_like(counts, LC_ZERO_COUNT))
    margin = float(tau) * torch.pow(counts, -0.25)
    return margin.to(dtype=torch.float32).unsqueeze(0).to(device)


def prox_term(model: torch.nn.Module, global_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """sum over trainable parameters of ||w - w_global||^2 (FedProx, Li 2020)."""
    total = None
    for name, p in model.named_parameters():
        term = ((p - global_params[name]) ** 2).sum()
        total = term if total is None else total + term
    return total


def _p_k(counts: List[int]) -> List[float]:
    total = sum(counts)
    return [float(n) / float(total) for n in counts]


def fedavgm_update(
    global_w: Dict[str, torch.Tensor],
    fedavg_w: Dict[str, torch.Tensor],
    param_keys: Sequence[str],
    momentum: Optional[Dict[str, torch.Tensor]],
    beta: float,
    server_lr: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Hsu 2019 sec 4.2 heavy-ball form: dw = w - w_avg; v <- beta*v + dw; w <- w - lr_s*v.

    Q8 design (b): beta 0.9, lr_s 0.1, no Nesterov. Momentum kept in float64.
    Non-parameter keys are copied from the FedAvg aggregate.
    """
    new_w = dict(fedavg_w)
    new_m = {}
    for k in param_keys:
        g64 = global_w[k].to(torch.float64)
        dw = g64 - fedavg_w[k].to(torch.float64)
        v = dw if momentum is None else beta * momentum[k] + dw
        new_m[k] = v
        new_w[k] = (g64 - server_lr * v).to(dtype=global_w[k].dtype)
        if not torch.isfinite(new_w[k]).all():
            raise ValueError(f"Non-finite parameter after FedAvgM update: {k}")
    return new_w, new_m


def fednova_update(
    global_w: Dict[str, torch.Tensor],
    fedavg_w: Dict[str, torch.Tensor],
    client_ws: List[Dict[str, torch.Tensor]],
    counts: List[int],
    taus: List[int],
    param_keys: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Wang 2020 eq. 12, vanilla SGD (client_momentum 0): a_k = ones(tau_k), ||a_k||_1 = tau_k.

    w_new = w + tau_eff * sum_k p_k * (w_k - w) / tau_k, tau_eff = sum_k p_k * tau_k,
    p_k = n_k / n, tau_k = optimizer steps applied by client k this round.
    Non-parameter keys are copied from the FedAvg aggregate.
    """
    if any(int(t) <= 0 for t in taus):
        raise ValueError(f"FedNova needs tau_k > 0, got {taus}")
    p = _p_k(counts)
    tau_eff = sum(pk * float(t) for pk, t in zip(p, taus))
    new_w = dict(fedavg_w)
    for k in param_keys:
        g64 = global_w[k].to(torch.float64)
        d = torch.zeros_like(g64)
        for pk, t, cw in zip(p, taus, client_ws):
            d += pk * (cw[k].to(torch.float64) - g64) / float(t)
        new_w[k] = (g64 + tau_eff * d).to(dtype=global_w[k].dtype)
        if not torch.isfinite(new_w[k]).all():
            raise ValueError(f"Non-finite parameter after FedNova update: {k}")
    return new_w, tau_eff

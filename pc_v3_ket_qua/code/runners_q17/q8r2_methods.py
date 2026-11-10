"""P2 (Q8-LAB round 2) method helpers: SCAFFOLD (option II), MOON (z = representation before the output layer),
FedBN (BatchNorm kept local). Imported by the round-2 sandbox copy of fedavg_mixed_runner.py.

Decisions: advisor do-an-tot-nghiep-cd 2026-10-25 (STATE.p2_prep.advisor_decisions_25_10, moon_decision option D).
Sources: SCAFFOLD Karimireddy et al. ICML 2020 arXiv:1910.06378 eq. (3)-(5), option II; MOON Li, He, Song CVPR 2021
arXiv:2103.16257 eq. (3)-(4), code github.com/QinbinLi/MOON; FedBN Li et al. ICLR 2021 arXiv:2102.07623.

Conventions:
- SCAFFOLD control variates exist only for trainable parameters (named_parameters keys); BN running stats and
  integer buffers come from the runner's FedAvg aggregate. x is aggregated n_k-weighted (= runner FedAvg, eta_g = 1;
  the paper averages uniformly - documented difference). c <- c + (1/N) sum_i (c_i+ - c_i) as in the paper (S = N).
- MOON z = output of classifier[1] (Hardswish, 1024-d, before Dropout) of torchvision mobilenet_v3_small; global and
  previous-local models run in eval() under no_grad; loss_con = CE([cos(z,z_glob), cos(z,z_prev)] / tau, 0).
- FedBN keys = every key of every BatchNorm module (weight, bias, running_mean, running_var, num_batches_tracked).
"""
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

METHODS = ("fedavg", "scaffold", "moon", "fedbn")


# ---------------- SCAFFOLD ----------------
def scaffold_init(global_state: Dict[str, torch.Tensor], param_keys: Sequence[str], client_ids: Sequence[str]):
    zeros = {k: torch.zeros_like(global_state[k], dtype=torch.float32) for k in param_keys}
    return zeros, {cid: {k: v.clone() for k, v in zeros.items()} for cid in client_ids}


def scaffold_correction(c: Dict[str, torch.Tensor], c_i: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """(c - c_i) per trainable parameter, on the training device (added to the unscaled gradient)."""
    return {k: (c[k] - c_i[k]).to(device) for k in c}


def scaffold_apply(model: nn.Module, corr: Dict[str, torch.Tensor]) -> None:
    for n, p in model.named_parameters():
        if p.grad is not None:
            p.grad.add_(corr[n].to(dtype=p.grad.dtype))


def scaffold_client_update(c: Dict[str, torch.Tensor], c_i: Dict[str, torch.Tensor], x: Dict[str, torch.Tensor],
                           y_i: Dict[str, torch.Tensor], k_i: int, lr: float) -> Dict[str, torch.Tensor]:
    """Option II: c_i+ = c_i - c + (x - y_i) / (K_i * eta_l)."""
    assert k_i > 0 and lr > 0
    return {k: c_i[k] - c[k] + (x[k].float() - y_i[k].float()) / (float(k_i) * float(lr)) for k in c}


def scaffold_server_update(c: Dict[str, torch.Tensor], c_old: List[Dict[str, torch.Tensor]],
                           c_new: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """c <- c + (1/N) sum_i (c_i+ - c_i), all N clients participate (S = N)."""
    n = len(c_new)
    out = {}
    for k in c:
        acc = torch.zeros_like(c[k], dtype=torch.float64)
        for o, nw in zip(c_old, c_new):
            acc += (nw[k].double() - o[k].double())
        out[k] = (c[k].double() + acc / n).to(torch.float32)
    return out


# ---------------- MOON ----------------
def mbv3_forward_with_rep(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same ops, same order as torchvision MobileNetV3._forward_impl; returns (logits, z) with z = classifier[1] out."""
    f = model.features(x)
    f = model.avgpool(f)
    f = torch.flatten(f, 1)
    h = model.classifier[1](model.classifier[0](f))
    z = h.clone()  # classifier[1] is an in-place-capable Hardswish; keep z independent of Dropout
    logits = model.classifier[3](model.classifier[2](h))
    return logits, z


def moon_rep(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, z = mbv3_forward_with_rep(model, x)
    return z


def moon_loss(z: torch.Tensor, z_glob: torch.Tensor, z_prev: torch.Tensor, tau: float) -> torch.Tensor:
    cos = nn.CosineSimilarity(dim=-1)
    z, z_glob, z_prev = z.float(), z_glob.float(), z_prev.float()
    logits = torch.stack([cos(z, z_glob), cos(z, z_prev)], dim=1) / tau
    labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    return nn.functional.cross_entropy(logits, labels)


# ---------------- FedBN ----------------
def bn_keys(model: nn.Module) -> List[str]:
    keys = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            for k in mod.state_dict().keys():
                keys.append(f"{name}.{k}" if name else k)
    return sorted(keys)


def fedbn_client_start(global_state: Dict[str, torch.Tensor], local_bn: Dict[str, torch.Tensor]):
    """Shared weights from the server + the client's own BN keys."""
    out = dict(global_state)
    for k, v in local_bn.items():
        out[k] = v.clone()
    return out


def class_share(targets: Sequence[int], num_classes: int) -> List[float]:
    counts = [0] * num_classes
    for t in targets:
        counts[int(t)] += 1
    total = float(sum(counts))
    return [c / total for c in counts]


def fedbn_round_eval(global_state, local_bn, client_ids, client_loaders, build, evaluate, val_loader, device,
                     max_batches):
    """FedBN descriptive endpoint (advisor 25-10): for every client k the personalized model = shared weights + BN_k,
    evaluated on the common val. (a) uniform mean of raw_macro_f1 over the 5 models; (b) per k:
    sum_c p_k(c) * recall_k(c), p_k = label share of k's train shard (classes without val support are excluded and
    their p_k mass is reported)."""
    rows = []
    for cid, loader in zip(client_ids, client_loaders):
        model = build(fedbn_client_start(global_state, local_bn[cid])).to(device).eval()
        res = evaluate(model, val_loader, device, max_batches=max_batches)
        recall = res["per_class_recall"]
        num_classes = len(recall)
        p = class_share(loader.dataset.targets, num_classes)
        defined = [c for c in range(num_classes) if recall[str(c)]["recall"] is not None]
        weighted = sum(p[c] * recall[str(c)]["recall"] for c in defined)
        pd = (res.get("per_source", {}).get("plantdoc") or {})
        rows.append({"client": cid, "raw_macro_f1": res["raw_macro_f1"],
                     "raw_macro_f1_supported": res["raw_macro_f1_supported"], "raw_accuracy": res["raw_accuracy"],
                     "plantdoc_accuracy": pd.get("raw_accuracy"),
                     "plantdoc_raw_macro_f1_supported": pd.get("raw_macro_f1_supported"),
                     "client_dist_recall": weighted,
                     "p_mass_without_val_support": 1.0 - sum(p[c] for c in defined)})
        del model
    n = float(len(rows))
    return {"fedbn_eval": rows,
            "fedbn_endpoint_a_uniform_mean_raw_macro_f1": sum(r["raw_macro_f1"] for r in rows) / n,
            "fedbn_endpoint_a_uniform_mean_plantdoc_accuracy": (
                sum(r["plantdoc_accuracy"] for r in rows) / n if all(r["plantdoc_accuracy"] is not None
                                                                     for r in rows) else None),
            "fedbn_endpoint_b_client_dist_recall": {r["client"]: r["client_dist_recall"] for r in rows}}

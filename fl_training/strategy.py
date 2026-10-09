"""
fl_training.strategy: FedAvg aggregation weighted by n_k with BatchNorm integer handling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import check_parameters_finite


def aggregate_fedavg_state_dicts(
    client_replies: List[Dict[str, Any]],
    expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    expected_dtypes: Optional[Dict[str, torch.dtype]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Aggregate client state_dicts using FedAvg weighted by n_k (num-examples):
      w_(t+1) = sum(n_k * w_k) / sum(n_k)

    BatchNorm policy:
      - Floating point tensors (weights, biases, running_mean, running_var)
        are aggregated as weighted sum in float32 (or original float dtype).
      - Integer buffers ('num_batches_tracked') take the maximum across replies
        and restore integer dtype (torch.int64).

    Input reply contract:
      Each reply is a dict containing:
        - "state_dict": Dict[str, torch.Tensor | np.ndarray]
        - "num-examples": int (n_k)
        - "client_id": int
        - (optional training metrics: loss-sum, correct, processed-examples, optimizer-steps)

    Returns:
      (aggregated_state_dict_cpu, aggregated_metrics)
    """
    if not client_replies:
        raise ValueError("Cannot aggregate empty client replies list")

    for reply in client_replies:
        cid = reply.get("client_id")
        if isinstance(cid, bool) or not isinstance(cid, (int, np.integer)) or cid < 0:
            raise ValueError("client_id must be a nonnegative integer")

    # Sort replies by client_id for deterministic order
    sorted_replies = sorted(client_replies, key=lambda r: r.get("client_id", 0))

    client_ids = [int(r.get("client_id", -1)) for r in sorted_replies]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError(f"Duplicate client replies are not allowed: {client_ids}")
    for reply in sorted_replies:
        client_id = reply.get("client_id")
        n_k = reply.get("num-examples")
        if isinstance(n_k, bool) or not isinstance(n_k, (int, np.integer)) or int(n_k) <= 0:
            raise ValueError(f"Client {client_id} num-examples must be a positive integer, got {n_k!r}")
        if "state_dict" not in reply or not isinstance(reply["state_dict"], dict):
            raise ValueError(f"Client {client_id} reply is missing a state_dict")

    total_num_examples = sum(r["num-examples"] for r in sorted_replies)
    if total_num_examples <= 0:
        raise ValueError(f"Total num-examples across clients must be positive, got {total_num_examples}")

    first_sd = sorted_replies[0]["state_dict"]
    param_keys = list(first_sd.keys())
    if not param_keys:
        raise ValueError("Cannot aggregate an empty model")
    reference = {
        key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for key, value in first_sd.items()
    }
    if not all(isinstance(value, torch.Tensor) for value in reference.values()):
        raise TypeError("Model state must contain tensors")
    if expected_shapes is None:
        expected_shapes = {key: tuple(value.shape) for key, value in reference.items()}
    if expected_dtypes is None:
        expected_dtypes = {key: value.dtype for key, value in reference.items()}
    if set(expected_dtypes) != set(expected_shapes):
        raise ValueError("Expected shapes and dtypes must describe the same complete model")
    expected_key_set = set(expected_shapes) if expected_shapes is not None else set(param_keys)
    if set(param_keys) != expected_key_set:
        raise ValueError("First client state_dict keys do not match the expected model")
    for reply in sorted_replies:
        state = reply["state_dict"]
        if set(state) != expected_key_set:
            raise ValueError(f"Client {reply['client_id']} state_dict keys do not match the expected model")
        for key, value in state.items():
            tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Client {reply['client_id']} key '{key}' is not a tensor")
            if tensor.is_complex() or tensor.dtype == torch.bool:
                raise TypeError(f"Unsupported model dtype for '{key}': {tensor.dtype}")
            if expected_shapes is not None and tuple(tensor.shape) != tuple(expected_shapes[key]):
                raise ValueError(
                    f"Client {reply['client_id']} key '{key}' shape {tuple(tensor.shape)} "
                    f"!= expected {tuple(expected_shapes[key])}"
                )
            if expected_dtypes is not None and tensor.dtype != expected_dtypes[key]:
                raise ValueError(
                    f"Client {reply['client_id']} key '{key}' dtype {tensor.dtype} "
                    f"!= expected {expected_dtypes[key]}"
                )

    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in param_keys:
        # Determine if this key is an integer buffer (like num_batches_tracked)
        sample_tensor = reference[key]
        is_torch = isinstance(sample_tensor, torch.Tensor)
        is_integer = (
            (is_torch and not sample_tensor.is_floating_point())
            or (isinstance(sample_tensor, np.ndarray) and np.issubdtype(sample_tensor.dtype, np.integer))
            or key.endswith("num_batches_tracked")
        )

        if is_integer:
            # Integer buffer: policy is max across replies
            vals = []
            for r in sorted_replies:
                t = r["state_dict"][key]
                val = torch.as_tensor(t).detach().cpu()
                if key.endswith("num_batches_tracked") and (val < 0).any():
                    raise ValueError("BatchNorm batch counters must be nonnegative")
                vals.append(val)
            aggregated_state[key] = torch.stack(vals).amax(dim=0).to(sample_tensor.dtype)
        else:
            # Floating tensor: weighted average in FP32
            accum = None
            for r in sorted_replies:
                n_k = r["num-examples"]
                weight = n_k / total_num_examples
                t = r["state_dict"][key]
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                accumulation_dtype = torch.float64 if sample_tensor.dtype == torch.float64 else torch.float32
                t_float = t.detach().to(device="cpu", dtype=accumulation_dtype)

                # Check finiteness
                if not torch.isfinite(t_float).all():
                    raise FloatingPointError(f"Non-finite values in client {r.get('client_id')} for key '{key}'")

                if accum is None:
                    accum = t_float * weight
                else:
                    accum.add_(t_float * weight)

            target_dtype = expected_dtypes[key] if expected_dtypes and key in expected_dtypes else sample_tensor.dtype
            if accum.dtype != target_dtype:
                accum = accum.to(dtype=target_dtype)

            aggregated_state[key] = accum

    check_parameters_finite(aggregated_state)

    # Aggregate metrics
    total_loss_sum = sum(r.get("loss-sum", 0.0) for r in sorted_replies)
    total_correct = sum(r.get("correct", 0) for r in sorted_replies)
    total_processed = sum(r.get("processed-examples", 0) for r in sorted_replies)
    total_steps = sum(r.get("optimizer-steps", 0) for r in sorted_replies)

    train_loss = total_loss_sum / total_processed if total_processed > 0 else 0.0
    train_acc = total_correct / total_processed if total_processed > 0 else 0.0

    metrics = {
        "num-examples": int(total_num_examples),
        "loss-sum": float(total_loss_sum),
        "correct": int(total_correct),
        "processed-examples": int(total_processed),
        "optimizer-steps": int(total_steps),
        "skipped-optimizer-steps": sum(int(r.get("skipped-optimizer-steps", 0)) for r in sorted_replies),
        "train-loss": float(train_loss),
        "train-accuracy": float(train_acc),
        "participating-clients": len(sorted_replies),
    }

    return aggregated_state, metrics


def summarize_client_updates(
    global_state: Dict[str, torch.Tensor],
    client_replies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return compact per-client weight/update diagnostics without logging tensors."""
    summaries: List[Dict[str, Any]] = []
    eps = 1e-12
    for reply in sorted(client_replies, key=lambda r: int(r["client_id"])):
        global_sq = 0.0
        update_sq = 0.0
        value_min = float("inf")
        value_max = float("-inf")
        nonfinite = 0
        tensor_bytes = 0
        for key, raw_value in reply["state_dict"].items():
            value = torch.from_numpy(raw_value) if isinstance(raw_value, np.ndarray) else raw_value
            tensor_bytes += value.numel() * value.element_size()
            if not value.is_floating_point():
                continue
            value_f = value.detach().to(torch.float64)
            base_f = global_state[key].detach().to(torch.float64)
            finite = torch.isfinite(value_f)
            nonfinite += int((~finite).sum().item())
            if finite.any():
                finite_values = value_f[finite]
                value_min = min(value_min, float(finite_values.min().item()))
                value_max = max(value_max, float(finite_values.max().item()))
            global_sq += float(torch.sum(base_f * base_f).item())
            delta = value_f - base_f
            update_sq += float(torch.sum(delta * delta).item())
        global_norm = global_sq**0.5
        update_norm = update_sq**0.5
        summaries.append({
            "client_id": int(reply["client_id"]),
            "n_k": int(reply["num-examples"]),
            "global_l2": global_norm,
            "update_l2": update_norm,
            "relative_update_l2": update_norm / max(global_norm, eps),
            "min": value_min if value_min != float("inf") else None,
            "max": value_max if value_max != float("-inf") else None,
            "nonfinite_count": nonfinite,
            "tensor_bytes": int(tensor_bytes),
        })
    return summaries

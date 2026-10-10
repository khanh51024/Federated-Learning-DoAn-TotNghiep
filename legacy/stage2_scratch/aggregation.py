"""FedAvg aggregation module for Stage 2 Scratch.

Implements mathematically sound parameter and buffer aggregation:
1. Float weights and running statistics: weighted average by client sample count (n_k / sum(n_k)).
2. Integer BatchNorm buffers (e.g. num_batches_tracked):
   Policy: base + sum(local_k - base)
   Accurately tracks total minibatches processed across clients without multiplying the global base by K each round.
3. Strict validation: tensor count, shape, dtype, finite values, unique client IDs, and positive sample counts.
"""

from typing import List, Optional, Sequence
import numpy as np


def aggregate_fedavg_parameters(
    base_parameters: Sequence[np.ndarray],
    client_parameters: Sequence[Sequence[np.ndarray]],
    client_samples: Sequence[int],
    client_ids: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    """Tổng hợp tham số FedAvg cho Stage 2:
    - Float tensors & running stats: weighted average theo số mẫu client n_k.
    - Integer buffers (num_batches_tracked): base + sum(local - base).
    """
    if not client_parameters or len(client_parameters) != len(client_samples):
        raise ValueError("Aggregation requires matching client parameters and sample counts")
    if any(s <= 0 for s in client_samples):
        raise ValueError("All client sample counts must be strictly positive")
    if client_ids is not None:
        if len(set(client_ids)) != len(client_ids):
            raise ValueError(f"Duplicate client IDs in aggregation: {client_ids}")
        if len(client_ids) != len(client_parameters):
            raise ValueError("Client IDs count must match client parameters count")

    num_params = len(base_parameters)
    for c_idx, c_params in enumerate(client_parameters):
        if len(c_params) != num_params:
            raise ValueError(
                f"Client {c_idx} has {len(c_params)} parameters, expected {num_params}"
            )

    total_samples = sum(client_samples)
    aggregated = []

    for index in range(num_params):
        base_val = base_parameters[index]
        client_vals = [c_params[index] for c_params in client_parameters]

        # Validate shape and dtype consistency
        for c_idx, c_val in enumerate(client_vals):
            if c_val.shape != base_val.shape:
                raise ValueError(
                    f"Parameter {index} shape mismatch: client {c_idx} {c_val.shape} vs base {base_val.shape}"
                )
            if c_val.dtype != base_val.dtype:
                raise ValueError(
                    f"Parameter {index} dtype mismatch: client {c_idx} {c_val.dtype} vs base {base_val.dtype}"
                )

        if np.issubdtype(base_val.dtype, np.integer):
            # Integer BN counter: base + sum(local - base)
            deltas = [c_val - base_val for c_val in client_vals]
            new_counter = base_val + sum(deltas)
            aggregated.append(np.asarray(new_counter, dtype=base_val.dtype))
        else:
            # Float weights & running stats: weighted average
            for c_idx, c_val in enumerate(client_vals):
                if not np.all(np.isfinite(c_val)):
                    raise ValueError(f"Non-finite values detected in parameter {index} from client {c_idx}")

            weighted_sum = sum(
                samples * c_val for samples, c_val in zip(client_samples, client_vals)
            )
            agg_val = weighted_sum / total_samples
            aggregated.append(np.asarray(agg_val, dtype=base_val.dtype))

    return aggregated

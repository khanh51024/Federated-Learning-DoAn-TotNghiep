"""Reference arithmetic for the shared simulation/network aggregate rule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.federated.strategy import aggregate_fedavg


def test_weighted_float_and_batchnorm_counter_increments():
    parent = {
        "weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }
    updates = [
        {"client_id": 0, "num_examples": 1, "weights": {
            "weight": torch.tensor([2.0, 4.0]), "bn.num_batches_tracked": torch.tensor(9),
        }},
        {"client_id": 1, "num_examples": 3, "weights": {
            "weight": torch.tensor([6.0, 8.0]), "bn.num_batches_tracked": torch.tensor(10),
        }},
    ]
    result = aggregate_fedavg(updates, parent_weights=parent)
    np.testing.assert_allclose(result["weight"].numpy(), [5.0, 7.0])
    assert result["bn.num_batches_tracked"].item() == 12


def test_batchnorm_parent_is_required_and_counter_cannot_regress():
    update = {"client_id": 0, "num_examples": 1, "weights": {
        "bn.num_batches_tracked": torch.tensor(5, dtype=torch.int64),
    }}
    with pytest.raises(ValueError, match="Missing parent"):
        aggregate_fedavg([update])
    with pytest.raises(ValueError, match="regressed"):
        aggregate_fedavg([update], parent_weights={"bn.num_batches_tracked": torch.tensor(6)})

from __future__ import annotations

import torch

from src.federated.faulty_client import apply_fault_mode
from src.federated.update_validator import compute_update_norm, validate_client_update
from src.model.mobilenetv3 import compute_model_hash


def _weights():
    return {
        "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "counter": torch.tensor(2, dtype=torch.int64),
    }


def test_validator_accepts_normal_update_and_recomputes_norm():
    parent = _weights()
    child = {k: v.clone() for k, v in parent.items()}
    child["layer.weight"] += 0.5

    result = validate_client_update(
        parent,
        child,
        reported_parent_hash=compute_model_hash(parent),
    )

    assert result.accepted
    assert not result.suspicious
    assert abs(result.received_update_norm - 1.0) < 1e-6


def test_validator_flags_no_train_as_near_zero():
    parent = _weights()
    result = validate_client_update(
        parent,
        {k: v.clone() for k, v in parent.items()},
        reported_parent_hash=compute_model_hash(parent),
    )

    assert result.accepted
    assert result.suspicious
    assert result.received_update_norm == 0.0
    assert "near_zero_update" in result.reason


def test_validator_rejects_wrong_parent_hash():
    parent = _weights()
    child = {k: v.clone() for k, v in parent.items()}
    child["layer.weight"] += 0.1

    result = validate_client_update(parent, child, reported_parent_hash="bad-hash")

    assert not result.accepted
    assert result.suspicious
    assert "parent_hash_mismatch" in result.reason


def test_validator_rejects_non_finite_payload():
    parent = _weights()
    child = {k: v.clone() for k, v in parent.items()}
    child["layer.weight"][0, 0] = float("nan")

    result = validate_client_update(parent, child)

    assert not result.accepted
    assert result.suspicious
    assert "non_finite" in result.reason


def test_scaled_fault_changes_server_observed_norm():
    parent = _weights()
    trained = {k: v.clone() for k, v in parent.items()}
    trained["layer.weight"] += 0.25

    trained_norm = compute_update_norm(parent, trained)
    sent = apply_fault_mode("scaled_update", trained, parent, scale_factor=10.0)
    sent_norm = compute_update_norm(parent, sent)

    assert abs(sent_norm - (trained_norm * 10.0)) < 1e-5


def test_random_fault_is_observable_from_actual_payload():
    torch.manual_seed(123)
    parent = _weights()
    trained = {k: v.clone() for k, v in parent.items()}
    trained["layer.weight"] += 0.01

    sent = apply_fault_mode("random_weights", trained, parent)
    trained_norm = compute_update_norm(parent, trained)
    sent_norm = compute_update_norm(parent, sent)

    assert sent_norm != trained_norm
    assert sent_norm > trained_norm


def test_learnable_norm_ignores_batchnorm_buffers_but_full_norm_counts_them():
    """Ảnh nhiễu làm running_var trôi: chuẩn đầy đủ > 0 trong khi tham số học được không đổi (chuẩn học được = 0)."""
    from src.federated.update_validator import compute_subset_update_norm, learnable_update_stats

    def close(a: float, b: float) -> bool:
        return abs(a - b) < 1e-6

    parent = {"conv.weight": torch.ones(4), "bn.running_var": torch.ones(4)}
    child = {"conv.weight": parent["conv.weight"].clone(), "bn.running_var": parent["bn.running_var"] + 0.5}

    assert close(compute_update_norm(parent, child), 1.0)  # sqrt(4 * 0.5^2)
    assert compute_subset_update_norm(parent, child, ["conv.weight"]) == 0.0
    stats = learnable_update_stats(parent, child, ["conv.weight"])
    assert stats["norm"] == 0.0 and stats["relative"] == 0.0 and close(stats["parent_norm"], 2.0)

    learned = {"conv.weight": parent["conv.weight"] + 0.1, "bn.running_var": parent["bn.running_var"].clone()}
    stats = learnable_update_stats(parent, learned, ["conv.weight"])
    assert close(stats["norm"], 0.2) and close(stats["relative"], 0.1)

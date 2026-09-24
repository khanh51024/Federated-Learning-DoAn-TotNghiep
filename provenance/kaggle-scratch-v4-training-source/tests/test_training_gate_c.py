"""
Tests for Gate C: FedAvg Strategy, Controllers (Early Stopping & LR), and Checkpoint/Resume.
"""

from pathlib import Path
import pytest
import torch
import numpy as np

from fl_training.control import EarlyStoppingController, LRSchedulerController
from fl_training.strategy import aggregate_fedavg_state_dicts
from fl_training.checkpoint import (
    load_checkpoint,
    save_round_checkpoint,
    verify_checkpoint_compatibility,
)

ROOT = Path(__file__).resolve().parents[1]


def test_gate_c_hand_calculated_fedavg_fixture():
    """
    Section 10 Step 4 exact hand-calculated test fixture:
    - 2 clients with n_k = 1 and 3.
    - Floating weight tensors 2.0 and 6.0 -> result: (1*2 + 3*6)/4 = 5.0.
    - Integer counter tensors 3 and 7 -> result: max(3, 7) = 7 (int64).
    """
    reply_0 = {
        "client_id": 0,
        "num-examples": 1,
        "state_dict": {
            "weight": torch.tensor([2.0], dtype=torch.float32),
            "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
        },
        "loss-sum": 1.5,
        "correct": 1,
        "processed-examples": 1,
        "optimizer-steps": 1,
    }
    reply_1 = {
        "client_id": 1,
        "num-examples": 3,
        "state_dict": {
            "weight": torch.tensor([6.0], dtype=torch.float32),
            "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        },
        "loss-sum": 4.5,
        "correct": 2,
        "processed-examples": 3,
        "optimizer-steps": 3,
    }

    aggregated, metrics = aggregate_fedavg_state_dicts([reply_0, reply_1])

    # Check floating weighted average
    assert abs(aggregated["weight"].item() - 5.0) < 1e-6
    # Check integer buffer max policy
    assert aggregated["num_batches_tracked"].item() == 7
    assert aggregated["num_batches_tracked"].dtype == torch.int64

    # Check metrics
    assert metrics["num-examples"] == 4
    assert metrics["processed-examples"] == 4
    assert metrics["optimizer-steps"] == 4
    assert abs(metrics["train-loss"] - 1.5) < 1e-6
    assert abs(metrics["train-accuracy"] - 0.75) < 1e-6


def test_gate_c_early_stopping_plateau_stops_at_round_20():
    """
    Section 10 Step 5:
    Plateau after warmup stops at exactly round 20 if no improvement from round 0.
    (Warmup rounds = 10, patience = 10).
    """
    es = EarlyStoppingController(min_delta=0.0001, patience_rounds=10, warmup_rounds=10)
    es.init_round_0(val_loss=1.0)

    stopped_round = None
    for r in range(1, 30):
        if es.step(r, val_loss=1.0):
            stopped_round = r
            break

    assert stopped_round == 20, f"Expected early stop at round 20, got {stopped_round}"


def test_gate_c_early_stopping_significant_improvement_resets():
    es = EarlyStoppingController(min_delta=0.01, patience_rounds=5, warmup_rounds=0)
    es.init_round_0(val_loss=1.0)

    # Rounds 1..4: loss drops slightly but not exceeding min_delta (0.01)
    for r in range(1, 5):
        assert not es.step(r, val_loss=0.995)
    assert es.bad_rounds == 4

    # Round 5: significant drop from 1.0 to 0.95 (> 0.01) -> bad_rounds resets to 0!
    assert not es.step(5, val_loss=0.95)
    assert es.bad_rounds == 0
    assert abs(es.reference_loss - 0.95) < 1e-6


def test_gate_c_early_stopping_nan_fails():
    es = EarlyStoppingController()
    es.init_round_0(val_loss=1.0)
    with pytest.raises(FloatingPointError, match="Non-finite validation loss"):
        es.step(1, val_loss=float("nan"))


def test_gate_c_lr_scheduler_reduces_lr():
    ls = LRSchedulerController(initial_lr=0.01, factor=0.5, patience_rounds=3, threshold=0.0001)
    ls.init_round_0(val_loss=1.0)

    # Rounds 1..3: plateau, bad_rounds accumulates to 3
    for r in range(1, 4):
        new_lr = ls.step(r, val_loss=1.0)
        assert abs(new_lr - 0.01) < 1e-6

    # Round 4: bad_rounds becomes 4 > 3 -> LR reduced to 0.005!
    new_lr = ls.step(4, val_loss=1.0)
    assert abs(new_lr - 0.005) < 1e-6


def test_gate_c_checkpoint_atomic_and_resume(tmp_path):
    run_dir = tmp_path / "run_test"
    model_sd = {"weight": torch.tensor([1.0, 2.0])}
    best_sd = {"weight": torch.tensor([1.0, 2.0])}

    last_pt, best_pt = save_round_checkpoint(
        run_dir=run_dir,
        round_num=1,
        model_state_dict=model_sd,
        best_model_state_dict=best_sd,
        best_round=1,
        best_loss=0.5,
        next_lr=0.01,
        early_stopping_state={"bad_rounds": 0},
        lr_scheduler_state={"lr": 0.01},
        history=[{"round": 1, "loss": 0.5}],
        resolved_config={"semantic_config_hash": "hash_123"},
        is_best=True,
    )

    assert last_pt.exists()
    assert best_pt is not None and best_pt.exists()

    loaded_last = load_checkpoint(last_pt)
    assert loaded_last["round"] == 1
    assert loaded_last["best_loss"] == 0.5
    assert torch.equal(loaded_last["model_state_dict"]["weight"], model_sd["weight"])

    # Test compatibility check
    verify_checkpoint_compatibility(loaded_last, {"semantic_config_hash": "hash_123"})

    with pytest.raises(ValueError, match="Cannot resume: semantic config hash mismatch"):
        verify_checkpoint_compatibility(loaded_last, {"semantic_config_hash": "different_hash"})

"""
fl_training.control: Early stopping and learning rate scheduling controllers for server coordination.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class EarlyStoppingController:
    """
    Early stopping measured after each round on mean global validation loss:
    - Round 0 initializes reference_loss and best_loss.
    - Rounds 1..warmup_rounds: updates reference/best if improved, but keeps bad_rounds=0.
    - Rounds > warmup_rounds: increments bad_rounds if val_loss >= reference_loss - min_delta.
    - Resets bad_rounds to 0 on significant improvement (val_loss < reference_loss - min_delta).
    - Stops when bad_rounds >= patience_rounds.
    - NaN/Inf immediately raises FloatingPointError.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_delta: float = 0.0001,
        patience_rounds: int = 10,
        warmup_rounds: int = 10,
    ):
        self.enabled = bool(enabled)
        self.min_delta = float(min_delta)
        self.patience_rounds = int(patience_rounds)
        self.warmup_rounds = int(warmup_rounds)

        self.best_loss = float("inf")
        self.best_round = 0
        self.reference_loss = float("inf")
        self.bad_rounds = 0
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self.is_last_step_best = False

    def init_round_0(self, val_loss: float) -> None:
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Initial round 0 validation loss is non-finite: {val_loss}")
        self.best_loss = float(val_loss)
        self.best_round = 0
        self.reference_loss = float(val_loss)
        self.bad_rounds = 0
        self.stopped = False
        self.stop_reason = None
        self.is_last_step_best = True

    def step(self, round_num: int, val_loss: float) -> bool:
        """
        Evaluate early stopping at round_num.
        Returns True if early stopping triggers, False otherwise.
        """
        if not np.isfinite(val_loss):
            self.stopped = True
            self.stop_reason = f"Non-finite validation loss ({val_loss}) at round {round_num}"
            raise FloatingPointError(self.stop_reason)

        val_loss = float(val_loss)
        self.is_last_step_best = False

        # Update absolute best observed
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_round = round_num
            self.is_last_step_best = True

        if not self.enabled:
            return False

        # Check significant improvement over reference
        if val_loss < self.reference_loss - self.min_delta:
            self.reference_loss = val_loss
            self.bad_rounds = 0
        else:
            if round_num > self.warmup_rounds:
                self.bad_rounds += 1
                if self.bad_rounds >= self.patience_rounds:
                    self.stopped = True
                    self.stop_reason = (
                        f"Early stopping triggered at round {round_num}: "
                        f"no significant improvement over {self.reference_loss:.6f} "
                        f"(min_delta={self.min_delta}) for {self.bad_rounds} rounds"
                    )
                    return True
            else:
                self.bad_rounds = 0

        return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_delta": self.min_delta,
            "patience_rounds": self.patience_rounds,
            "warmup_rounds": self.warmup_rounds,
            "best_loss": self.best_loss,
            "best_round": self.best_round,
            "reference_loss": self.reference_loss,
            "bad_rounds": self.bad_rounds,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.enabled = bool(state["enabled"])
        self.min_delta = float(state["min_delta"])
        self.patience_rounds = int(state["patience_rounds"])
        self.warmup_rounds = int(state["warmup_rounds"])
        self.best_loss = float(state["best_loss"])
        self.best_round = int(state["best_round"])
        self.reference_loss = float(state["reference_loss"])
        self.bad_rounds = int(state["bad_rounds"])
        self.stopped = bool(state["stopped"])
        self.stop_reason = state.get("stop_reason")


class LRSchedulerController:
    """
    Learning rate controller with ReduceLROnPlateau semantics:
      mode='min', threshold_mode='abs', threshold=1e-4, patience=3, factor=0.5, min_lr=1e-6.
    Evaluates val_loss at each round.
    New LR applies to the subsequent round.
    """

    def __init__(
        self,
        enabled: bool = True,
        initial_lr: float = 0.01,
        factor: float = 0.5,
        patience_rounds: int = 3,
        threshold: float = 0.0001,
        min_lr: float = 0.000001,
    ):
        self.enabled = bool(enabled)
        self.lr = float(initial_lr)
        self.initial_lr = float(initial_lr)
        self.factor = float(factor)
        self.patience_rounds = int(patience_rounds)
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)

        self.best_loss = float("inf")
        self.bad_rounds = 0

    def init_round_0(self, val_loss: float) -> None:
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Initial round 0 validation loss is non-finite: {val_loss}")
        self.best_loss = float(val_loss)
        self.bad_rounds = 0

    def step(self, round_num: int, val_loss: float) -> float:
        """
        Update learning rate based on val_loss.
        Returns the learning rate to be used in round_num + 1.
        """
        if not self.enabled:
            return self.lr

        val_loss = float(val_loss)
        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
            if self.bad_rounds > self.patience_rounds:
                new_lr = max(self.lr * self.factor, self.min_lr)
                self.lr = new_lr
                self.bad_rounds = 0

        return self.lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lr": self.lr,
            "initial_lr": self.initial_lr,
            "factor": self.factor,
            "patience_rounds": self.patience_rounds,
            "threshold": self.threshold,
            "min_lr": self.min_lr,
            "best_loss": self.best_loss,
            "bad_rounds": self.bad_rounds,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.enabled = bool(state["enabled"])
        self.lr = float(state["lr"])
        self.initial_lr = float(state["initial_lr"])
        self.factor = float(state["factor"])
        self.patience_rounds = int(state["patience_rounds"])
        self.threshold = float(state["threshold"])
        self.min_lr = float(state["min_lr"])
        self.best_loss = float(state["best_loss"])
        self.bad_rounds = int(state["bad_rounds"])

"""Cooperative soft/hard deadlines shared by launcher, server, and workers."""

from __future__ import annotations

import math
import os
import time


class BudgetExhausted(RuntimeError):
    """Raised at a safe boundary so the last atomic checkpoint can be resumed."""


def _deadline(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite Unix timestamp") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive Unix timestamp")
    return value


def check_deadline(now: float | None = None) -> None:
    """Stop compute at the soft deadline, before the launcher hard-kills its tree."""
    current = time.time() if now is None else float(now)
    soft = _deadline("FL_TRAINING_SOFT_DEADLINE_UNIX")
    if soft is None:
        soft = _deadline("FL_TRAINING_DEADLINE_UNIX")
    hard = _deadline("FL_TRAINING_HARD_DEADLINE_UNIX")
    if soft is not None and hard is not None and soft > hard:
        raise ValueError("Soft deadline must not be later than hard deadline")
    if hard is not None and current >= hard:
        raise BudgetExhausted("Hard session deadline reached; resume the last committed checkpoint")
    if soft is not None and current >= soft:
        raise BudgetExhausted("Soft session deadline reached; saving/stopping before hard cleanup")

# Copied from fl_training/budget_state.py; frozen Stage2 source remains unchanged.
"""Budget accounting/planning independent of PyTorch and Kaggle credentials."""
from __future__ import annotations
import json
import math
import os
import time
import uuid
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def _finite_number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def choose_rounds(timings, job_modes, available_seconds, minimum, maximum, safety_factor):
    """Same fixed round budget for every mode, chosen from timings, never accuracy."""
    available_seconds = _finite_number(available_seconds, "available_seconds", minimum=0)
    safety_factor = _finite_number(safety_factor, "safety_factor", minimum=1)
    if (not job_modes or isinstance(minimum, bool) or isinstance(maximum, bool)
            or not isinstance(minimum, int) or not isinstance(maximum, int)
            or not 10 <= minimum <= maximum <= 20):
        raise ValueError("Invalid round range, modes or safety factor")
    slope = fixed = 0.0
    for mode in job_modes:
        if mode not in timings:
            raise ValueError(f"Missing calibration timing for {mode}")
        item = timings[mode]
        rate = _finite_number(item["seconds_per_round"], f"{mode}.seconds_per_round", minimum=0)
        overhead = _finite_number(item["fixed_seconds"], f"{mode}.fixed_seconds", minimum=0)
        if rate == 0:
            raise ValueError("Invalid calibration timing")
        slope += rate
        fixed += overhead
    rounds = min(maximum, math.floor((available_seconds / safety_factor - fixed) / slope))
    if rounds < minimum:
        raise RuntimeError("Measured matrix cannot fit the remaining budget at minimum rounds; no main training started")
    return rounds, safety_factor * (fixed + rounds * slope)


def apply_usage_observations(state, *, already_used_hours=None, quota_remaining_hours=None,
                             charge_hours=0.0, charge_id=None, now=time.time):
    """Merge user observations without ever decreasing or double-charging usage."""
    state["used_seconds"] = _finite_number(state.get("used_seconds", 0), "used_seconds", minimum=0)
    state.setdefault("external_charges", {})
    if already_used_hours is not None:
        already = _finite_number(already_used_hours, "already_used_hours", minimum=0, maximum=30)
        state["used_seconds"] = max(state["used_seconds"], already * 3600)
    charge = _finite_number(charge_hours, "charge_hours", minimum=0, maximum=30)
    if charge:
        if not charge_id:
            raise ValueError("charge_id is required for a nonzero external charge")
        previous = state["external_charges"].get(charge_id)
        if previous is not None and float(previous) != charge:
            raise ValueError("charge_id was already recorded with a different duration")
        if previous is None:
            state["used_seconds"] += charge * 3600
            state["external_charges"][charge_id] = charge
    if quota_remaining_hours is not None:
        quota = _finite_number(quota_remaining_hours, "quota_remaining_hours", minimum=0, maximum=30)
        state["quota_observation"] = {
            "remaining_hours": quota,
            "used_seconds_at_observation": state["used_seconds"],
            "recorded_at_unix": float(now()),
        }
    return state


def quota_remaining_from_state(state):
    observation = state.get("quota_observation")
    if not observation:
        return None
    remaining = _finite_number(observation["remaining_hours"], "quota remaining", minimum=0)
    observed_used = _finite_number(
        observation["used_seconds_at_observation"], "quota observed usage", minimum=0,
    )
    consumed = max(0.0, state["used_seconds"] - observed_used) / 3600
    return max(0.0, remaining - consumed)


class BudgetSession:
    def __init__(self, state, save, total_hours, reserve_hours, session_hours,
                 session_reserve_minutes, clock=time.monotonic, wall_clock=time.time,
                 quota_remaining_hours=None):
        total_hours = _finite_number(total_hours, "total_hours", minimum=0, maximum=30)
        reserve_hours = _finite_number(reserve_hours, "reserve_hours", minimum=0)
        session_hours = _finite_number(session_hours, "session_hours", minimum=0, maximum=8)
        session_reserve_minutes = _finite_number(
            session_reserve_minutes, "session_reserve_minutes", minimum=0,
        )
        if not (0 <= reserve_hours < total_hours and 0 < session_hours <= 8
                and 0 < session_reserve_minutes < session_hours * 60):
            raise ValueError("Invalid total/session/reserve budget")
        used = _finite_number(state.get("used_seconds", 0), "used_seconds", minimum=0)
        state["used_seconds"] = used
        quota_limit = float("inf")
        if quota_remaining_hours is not None:
            quota_remaining_hours = _finite_number(
                quota_remaining_hours, "quota_remaining_hours", minimum=0,
            )
            quota_limit = used + max(0.0, quota_remaining_hours - reserve_hours) * 3600
        self.state, self.save, self.clock, self.wall_clock = state, save, clock, wall_clock
        self.total_limit = min((total_hours - reserve_hours) * 3600, quota_limit)
        self.session_hard_limit = session_hours * 3600
        self.session_soft_limit = self.session_hard_limit - session_reserve_minutes * 60
        self.start = self.last = clock()
        self.wall_start = wall_clock()
        # Charge unrecorded time since the durable heartbeat, capped by the lease.
        lease = state.pop("active_lease", None)
        if lease:
            reserved = _finite_number(lease.get("reserved_seconds", 0), "lease.reserved_seconds", minimum=0)
            charged = _finite_number(lease.get("charged_seconds", 0), "lease.charged_seconds", minimum=0)
            heartbeat = _finite_number(lease.get("heartbeat_unix", self.wall_start), "lease.heartbeat_unix", minimum=0)
            recovered = min(max(0.0, reserved - charged), max(0.0, self.wall_start - heartbeat))
            state["used_seconds"] += recovered
            state["recovered_uncertain_lease"] = {
                "lease_id": lease.get("lease_id"), "charged_seconds": recovered,
            }
        self.save()

    def tick(self):
        now = self.clock()
        delta = max(0.0, now - self.last)
        self.state["used_seconds"] += delta
        if self.state.get("active_lease"):
            self.state["active_lease"]["charged_seconds"] += delta
            self.state["active_lease"]["heartbeat_unix"] = self.wall_clock()
        self.last = now
        self.save()

    def remaining(self):
        """Seconds until the soft no-new-work deadline or total budget."""
        return max(0.0, min(self.total_limit - self.state["used_seconds"],
                            self.session_soft_limit - (self.clock() - self.start)))

    def hard_remaining(self):
        return max(0.0, min(self.total_limit - self.state["used_seconds"],
                            self.session_hard_limit - (self.clock() - self.start)))

    def deadlines_unix(self):
        total_wall = self.wall_clock() + self.total_remaining()
        hard = min(self.wall_start + self.session_hard_limit, total_wall)
        return {
            "soft": min(self.wall_start + self.session_soft_limit, max(self.wall_clock(), hard - 60)),
            "hard": hard,
        }

    def total_remaining(self):
        return max(0.0, self.total_limit - self.state["used_seconds"])

    def begin_lease(self, seconds):
        seconds = _finite_number(seconds, "lease seconds", minimum=0)
        if seconds == 0 or self.state.get("active_lease"):
            raise ValueError("Lease must be positive and no other lease may be active")
        self.tick()
        self.state["active_lease"] = {
            "lease_id": uuid.uuid4().hex, "reserved_seconds": seconds,
            "charged_seconds": 0.0, "heartbeat_unix": self.wall_clock(),
        }
        self.save()

    def end_lease(self):
        self.tick()
        self.state.pop("active_lease", None)
        self.save()

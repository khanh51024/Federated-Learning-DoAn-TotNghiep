"""Checkpoint selection and convergence state shared by central and FedAvg runs."""

import math


DEFAULT_CONVERGENCE = {
    "min_step": 15,
    "min_delta": 0.002,
    "lr_patience": 4,
    "stop_patience": 12,
    "lr_factor": 0.5,
}


class QualityControl:
    def __init__(self, plantvillage_floor, policy=None, state=None, require_observed_lr=False):
        self.floor = float(plantvillage_floor)
        unknown = set(policy or {}) - set(DEFAULT_CONVERGENCE)
        if unknown:
            raise ValueError(f"Unknown convergence policy keys: {sorted(unknown)}")
        self.policy = {**DEFAULT_CONVERGENCE, **(policy or {})}
        self.require_observed_lr = bool(require_observed_lr)
        if not math.isfinite(self.floor):
            raise ValueError("PlantVillage macro-F1 floor must be finite")
        if not 0 <= self.floor <= 1:
            raise ValueError("PlantVillage macro-F1 floor must be a probability")
        for name in ("min_step", "lr_patience", "stop_patience"):
            value = self.policy[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_delta", "lr_factor"):
            value = self.policy[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.policy["min_delta"] < 0 or not 0 < self.policy["lr_factor"] < 1:
            raise ValueError("Invalid convergence delta or LR factor")
        self.best_score = None
        self.best_eligible = False
        self.significant_best = -1.0
        self.bad_evaluations = 0
        self.bad_since_lr_drop = 0
        self.lr_drops = 0
        self.last_lr_drop_step = 0
        self.lr_tracking_version = 2
        self.last_observed_lr = None
        if state:
            for field in ("best_score", "best_eligible", "significant_best", "bad_evaluations",
                          "bad_since_lr_drop", "lr_drops", "last_lr_drop_step"):
                if field not in state:
                    raise ValueError(f"Missing resume quality-control field: {field}")
                setattr(self, field, state[field])
            self.lr_tracking_version = int(state.get("lr_tracking_version", 1))
            self.last_observed_lr = state.get("last_observed_lr")
            state_mode = bool(state.get("require_observed_lr", False))
            if state_mode != self.require_observed_lr:
                raise ValueError("Resume quality-control LR tracking mode mismatch")

    def observe(self, step, plantdoc_f1, plantvillage_f1, crop_accuracy):
        metrics = (plantdoc_f1, plantvillage_f1, crop_accuracy)
        if any(not math.isfinite(float(value)) or value < 0 or value > 1 for value in metrics):
            raise ValueError("Validation quality metrics must be finite probabilities")
        eligible = plantvillage_f1 >= self.floor
        score = (float(plantdoc_f1), float(crop_accuracy))
        is_best = (self.best_score is None or (eligible and not self.best_eligible)
                   or (eligible == self.best_eligible and score > tuple(self.best_score)))
        if is_best:
            self.best_score = score
            self.best_eligible = eligible

        significant = plantdoc_f1 >= self.significant_best + self.policy["min_delta"]
        if significant:
            self.significant_best = float(plantdoc_f1)
            self.bad_evaluations = 0
            self.bad_since_lr_drop = 0
        else:
            self.bad_evaluations += 1
            self.bad_since_lr_drop += 1

        if step == self.policy["min_step"]:
            self.bad_since_lr_drop = 0

        if self.require_observed_lr:
            return {"is_best": is_best, "eligible": eligible, "significant": significant,
                    "reduce_lr": False, "lr_dropped": False, "stop": False}

        stop = (step >= self.policy["min_step"] and self.bad_evaluations >= self.policy["stop_patience"]
                and self.lr_drops > 0 and self.bad_since_lr_drop >= self.policy["lr_patience"])
        reduce_lr = (not stop and step >= self.policy["min_step"]
                     and self.bad_since_lr_drop >= self.policy["lr_patience"])
        if reduce_lr:
            self.lr_drops += 1
            self.last_lr_drop_step = step
            self.bad_since_lr_drop = 0
        return {"is_best": is_best, "eligible": eligible, "significant": significant,
                "reduce_lr": reduce_lr, "lr_dropped": reduce_lr, "stop": stop}

    def record_scheduler_step(self, step, lr_before, lr_after):
        if not self.require_observed_lr:
            raise RuntimeError("record_scheduler_step requires require_observed_lr=True")
        lr_before = float(lr_before)
        lr_after = float(lr_after)
        if not math.isfinite(lr_before) or not math.isfinite(lr_after) or lr_before <= 0 or lr_after <= 0:
            raise ValueError("Scheduler learning rates must be finite and positive")
        tolerance = max(1e-15, abs(lr_before) * 1e-12)
        if lr_after > lr_before + tolerance:
            raise ValueError("ReduceLROnPlateau unexpectedly increased the learning rate")
        dropped = lr_after < lr_before - tolerance
        if dropped:
            self.lr_drops += 1
            self.last_lr_drop_step = int(step)
            self.bad_since_lr_drop = 0
        self.lr_tracking_version = 2
        self.last_observed_lr = lr_after
        return dropped

    def should_stop(self, step):
        return (step >= self.policy["min_step"]
                and self.bad_evaluations >= self.policy["stop_patience"]
                and self.lr_drops > 0
                and self.bad_since_lr_drop >= self.policy["lr_patience"])

    def state_dict(self):
        return {name: getattr(self, name) for name in (
            "best_score", "best_eligible", "significant_best", "bad_evaluations",
            "bad_since_lr_drop", "lr_drops", "last_lr_drop_step", "lr_tracking_version",
            "last_observed_lr", "require_observed_lr")}

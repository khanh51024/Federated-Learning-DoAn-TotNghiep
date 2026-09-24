"""Cấu hình cho profile stage1_compat và 2 FedAvg jobs."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stage1_compat.constants import (
    DEFAULT_ALPHAS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROUNDS,
    DEFAULT_WEIGHT_DECAY,
    MAX_SESSION_HOURS,
    NUM_CLASSES,
    OPTIMIZER_NAME,
    RESERVE_HOURS,
    SAFETY_MARGIN,
    SEED,
    STOP_BEFORE_MINUTES,
    UPSTREAM_COMMIT,
)


@dataclass
class JobConfig:
    job_id: str
    method: str = "fedavg"
    alpha: float = 1.0
    seed: int = SEED
    num_clients: int = DEFAULT_NUM_CLIENTS
    rounds: int = DEFAULT_ROUNDS
    local_epochs: int = DEFAULT_LOCAL_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = OPTIMIZER_NAME
    momentum: float = 0.0
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    selection_metric: str = "validation_accuracy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1CompatProfileConfig:
    profile_name: str = "stage1_compat"
    upstream_commit: str = UPSTREAM_COMMIT
    backend: str = "sequential"
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    reference_only: bool = True
    strict_comparison_eligible: bool = False
    scientific_stage2_complete: bool = False

    # Budget & Quota
    user_quota_hours: float = 0.0
    reserve_hours: float = RESERVE_HOURS
    max_session_hours: float = MAX_SESSION_HOURS
    stop_before_minutes: float = STOP_BEFORE_MINUTES
    safety_margin: float = SAFETY_MARGIN

    # Jobs
    jobs: list[JobConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.jobs:
            self.jobs = [
                JobConfig(
                    job_id=f"fedavg_alpha{int(alpha) if alpha.is_integer() else str(alpha).replace('.', '_')}_seed{SEED}",
                    method="fedavg",
                    alpha=alpha,
                    seed=SEED,
                    num_clients=DEFAULT_NUM_CLIENTS,
                    rounds=DEFAULT_ROUNDS,
                    local_epochs=DEFAULT_LOCAL_EPOCHS,
                    batch_size=DEFAULT_BATCH_SIZE,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WEIGHT_DECAY,
                    optimizer=OPTIMIZER_NAME,
                )
                for alpha in DEFAULT_ALPHAS
            ]

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["data_dir"] = str(self.data_dir)
        res["output_dir"] = str(self.output_dir)
        return res


def get_default_profile_config(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    quota_hours: float = 0.0,
) -> Stage1CompatProfileConfig:
    resolved_data = Path(data_dir).resolve() if data_dir else DEFAULT_DATA_DIR.resolve()
    resolved_output = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    return Stage1CompatProfileConfig(
        data_dir=resolved_data,
        output_dir=resolved_output,
        user_quota_hours=quota_hours,
    )

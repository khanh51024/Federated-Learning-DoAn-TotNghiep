"""Tạo và xác minh identity định danh cho từng FedAvg job trong profile stage1_compat."""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from stage1_compat.integrity import digest, file_hash
from stage1_compat.constants import PACKAGE_ROOT
from typing import Any

from stage1_compat.config import JobConfig
from stage1_compat.constants import (
    IMAGE_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    OPTIMIZER_NAME,
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    UPSTREAM_COMMIT,
)


@dataclass
class JobIdentity:
    profile: str
    method: str
    seed: int
    alpha: float
    num_clients: int
    rounds: int
    local_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str
    model: str
    num_classes: int
    image_size: int
    split_protocol: str
    upstream_commit: str
    context: dict[str, Any]
    config_sha256: str
    source_sha256: str
    runtime: dict[str, str]
    identity_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_job_identity(job: JobConfig, profile_name: str = "stage1_compat", context: dict | None = None) -> JobIdentity:
    payload: dict[str, Any] = {
        "profile": profile_name,
        "method": job.method,
        "seed": job.seed,
        "alpha": float(job.alpha),
        "num_clients": job.num_clients,
        "rounds": job.rounds,
        "local_epochs": job.local_epochs,
        "batch_size": job.batch_size,
        "lr": float(job.lr),
        "weight_decay": float(job.weight_decay),
        "optimizer": job.optimizer,
        "model": MODEL_NAME,
        "num_classes": job.num_classes,
        "image_size": IMAGE_SIZE,
        "split_protocol": f"{SPLIT_TRAIN_RATIO}/{SPLIT_VAL_RATIO}/{SPLIT_TEST_RATIO}",
        "upstream_commit": UPSTREAM_COMMIT,
    }
    payload.update(context=context or {"scope": "synthetic_cpu_test"},
                   config_sha256=digest(job.to_dict()), source_sha256=source_fingerprint(),
                   runtime={n: version(n) for n in ("torch", "torchvision", "numpy", "Pillow", "scikit-learn")})
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    identity_hash = hashlib.sha256(encoded).hexdigest()

    return JobIdentity(
        profile=payload["profile"],
        method=payload["method"],
        seed=payload["seed"],
        alpha=payload["alpha"],
        num_clients=payload["num_clients"],
        rounds=payload["rounds"],
        local_epochs=payload["local_epochs"],
        batch_size=payload["batch_size"],
        lr=payload["lr"],
        weight_decay=payload["weight_decay"],
        optimizer=payload["optimizer"],
        model=payload["model"],
        num_classes=payload["num_classes"],
        image_size=payload["image_size"],
        split_protocol=payload["split_protocol"],
        upstream_commit=payload["upstream_commit"],
        context=payload["context"], config_sha256=payload["config_sha256"],
        source_sha256=payload["source_sha256"], runtime=payload["runtime"],
        identity_hash=identity_hash,
    )


def verify_job_identity(expected: JobIdentity, candidate: dict[str, Any]) -> bool:
    """Xác minh candidate identity khớp với expected identity."""
    return candidate == expected.to_dict()


def source_fingerprint():
    return digest({p.name: file_hash(p) for p in sorted(PACKAGE_ROOT.glob("*.py"))})


def verify_stored_identity(candidate):
    body = {k: v for k, v in candidate.items() if k != "identity_hash"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if actual != candidate.get("identity_hash") or candidate.get("source_sha256") != source_fingerprint():
        raise ValueError("Identity/source mismatch")
    return candidate

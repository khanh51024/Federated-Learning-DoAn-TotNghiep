"""Bind evaluation artifacts to the exact model and held-out protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .checkpoint import model_state_sha256
from .config import compute_protocol_fingerprint


def validate_evaluation_partition(payload: dict[str, Any], partition_dir: Path) -> str:
    fingerprint = compute_protocol_fingerprint(partition_dir)
    expected = payload.get("config", {}).get("data", {}).get("protocol_fingerprint")
    if expected and fingerprint != expected:
        raise ValueError("Evaluation partition content differs from checkpoint protocol fingerprint")
    return fingerprint


def bind_evaluation(result: dict[str, Any], payload: dict[str, Any], checkpoint: Path,
                    test_manifest: Path, protocol_fingerprint: str) -> dict[str, Any]:
    state_hash = model_state_sha256(payload["model_state_dict"])
    recorded_hash = payload.get("model_state_sha256")
    if recorded_hash and recorded_hash != state_hash:
        raise ValueError("Checkpoint model fingerprint does not match its tensors")
    result.update({
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_round": payload.get("round", payload.get("best_round")),
        "model_state_sha256": state_hash,
        "protocol_fingerprint": protocol_fingerprint,
        "test_manifest_sha256": hashlib.sha256(test_manifest.read_bytes()).hexdigest(),
        "run_id": payload.get("config", {}).get("run_id"),
    })
    # Preserve provenance even when best.pt and model_final.pt have equal tensors,
    # and preserve distinct numerical results (e.g. CPU versus CUDA evaluation).
    identity = json.dumps({k: v for k, v in result.items() if k != "evaluation_id"},
                          sort_keys=True, allow_nan=False)
    result["evaluation_id"] = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return result


def write_evaluation(result: dict[str, Any], output_dir: Path) -> None:
    """Keep every evaluation immutable; the top-level files select the latest."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "evaluations" / result["evaluation_id"]
    destination.mkdir(parents=True, exist_ok=True)
    for folder in (destination, output_dir):
        temporary = folder / "test_metrics.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(folder / "test_metrics.json")
        pd.DataFrame(result["confusion_matrix"]).to_csv(folder / "confusion_matrix.csv", index=False)

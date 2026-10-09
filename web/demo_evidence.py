"""Validated October 2026 mixed PlantVillage/PlantDoc demo evidence.

The checks establish byte provenance for published artifacts. Without the image
release they cannot establish that validation was independently reproduced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.import_branch_demo import (
    CONTRACT_FILES,
    RELEASE_SHA,
    SOURCES,
    VALIDATION_SHA,
    W0,
    classes_from_taxonomy,
    sha256,
    verify_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "artifacts/branch_demo/20261003"
PROTOCOL_ID = "mixed_pv_pd_v3_pretrained_seed42_validation"
MODEL_FILES = {
    "centralized": ("centralized", "checkpoint_best.pt", "Centralized v6", "GĐ1"),
    "fedavg": ("fedavg", "checkpoint_best.pt", "FedAvg v7", "GĐ2"),
    **{f"local_{i:02d}": ("local", f"client_{i:02d}/checkpoint_best.pt", f"Local {i:02d}", "GĐ1")
       for i in range(5)},
}


def _verified_manifest(root: Path, name: str) -> dict[str, Any]:
    branch = SOURCES[name][0]
    result_dir = root / name
    manifest = json.loads((result_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("branch") != branch or manifest.get("release_manifest_sha256") != RELEASE_SHA
            or manifest.get("validation_manifest_sha256") != VALIDATION_SHA):
        raise ValueError(f"{branch}: mismatched artifact manifest")
    for relative, expected in manifest["files"].items():
        candidate = (result_dir / relative).resolve()
        if not candidate.is_relative_to(result_dir.resolve()):
            raise ValueError(f"{branch}: artifact path escapes result directory")
        if (not candidate.is_file() or candidate.stat().st_size != expected["bytes"]
                or sha256(candidate) != expected["sha256"]):
            raise ValueError(f"{branch}: missing or hash-mismatched {relative}")
    return manifest


def load_evidence(root: Path = STORE) -> dict[str, Any]:
    """Fail closed: return no metrics or selectable models on any mismatch."""
    try:
        metadata = json.loads((root / "import_manifest.json").read_text(encoding="utf-8"))
        if (metadata.get("protocol_id") != PROTOCOL_ID
                or metadata.get("release_manifest_sha256") != RELEASE_SHA
                or metadata.get("validation_manifest_sha256") != VALIDATION_SHA
                or metadata.get("w0_fingerprint") != W0
                or metadata.get("seed") != 42
                or metadata.get("transform") != "canonical_v1"
                or metadata.get("num_classes") != 38
                or metadata.get("validation_counts") != {"plantdoc": 269, "plantvillage": 4344}):
            raise ValueError("import manifest protocol identity mismatch")
        for name, (branch, commit) in SOURCES.items():
            if metadata.get("sources", {}).get(name) != {"branch": branch, "commit": commit}:
                raise ValueError(f"{name}: source ref mismatch")
        for relative in CONTRACT_FILES:
            name = Path(relative).name
            if sha256(root / "contract" / name) != metadata["contract_hashes"].get(relative):
                raise ValueError(f"contract hash mismatch: {name}")
        classes = classes_from_taxonomy(root / "contract/taxonomy.py")
        if metadata.get("classes") != classes:
            raise ValueError("class order differs from pinned taxonomy")

        manifests = {name: _verified_manifest(root, name) for name in SOURCES}
        comparison_hashes = {
            item["files"]["validation_comparison.csv"]["sha256"] for item in manifests.values()
        }
        if len(comparison_hashes) != 1:
            raise ValueError("comparison bytes differ across branches")
        comparison = root / "fedavg/validation_comparison.csv"
        verify_comparison(comparison, manifests)
        local_summary = json.loads((root / "local/combined_summary.json").read_text(encoding="utf-8"))
        if (local_summary.get("seed") != 42
                or local_summary.get("w0_fingerprint") != W0
                or local_summary.get("release_manifest_sha256") != RELEASE_SHA
                or local_summary.get("partition_scheme") != "label_alpha_0_1"):
            raise ValueError("Local-only summary protocol mismatch")
        shard_counts = {
            name: int(local_summary["clients"][name]["n_k"])
            for name in (f"client_{i:02d}" for i in range(5))
        }
        if sum(shard_counts.values()) != 37238:
            raise ValueError("Client shard counts do not cover the published train split")
        central_summary = json.loads((root / "centralized/summary.json").read_text(encoding="utf-8"))
        if (central_summary.get("seed") != 42
                or central_summary.get("w0_fingerprint") != W0
                or central_summary.get("release_manifest_sha256") != RELEASE_SHA):
            raise ValueError("Centralized summary protocol mismatch")
        central_history = [
            {"epoch": int(item["epoch"]), "val_accuracy": float(item["val_accuracy"]),
             "train_loss": float(item["train_loss"]), "val_loss": float(item["val_loss"])}
            for item in central_summary.get("history", [])
        ]
        with comparison.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_name: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_name.setdefault(row["model"], {})[row["source"]] = {
                "count": int(row["count"]),
                "top1_accuracy": float(row["top1_accuracy"]),
                "macro_f1_supported": float(row["macro_f1_supported"]),
                "crop_accuracy": float(row["crop_accuracy"]),
            }
        models = []
        for model_id, (source_name, relative, display, phase) in MODEL_FILES.items():
            source = metadata["sources"][source_name]
            checkpoint_sha = manifests[source_name]["files"][relative]["sha256"]
            models.append({
                "model_id": model_id,
                "name": display,
                "phase": phase,
                "method": "Local-only" if source_name == "local" else display,
                "branch": source["branch"],
                "commit": source["commit"],
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_relative_path": f"{source_name}/{relative}",
                "metrics": by_name[display],
            })
        return {
            "available": True,
            "protocol_id": PROTOCOL_ID,
            "evidence_level": "published_artifact_hash_verified_not_locally_reevaluated",
            "metrics_source": "published validation_comparison.csv; no local image release",
            "release_manifest_sha256": RELEASE_SHA,
            "validation_manifest_sha256": VALIDATION_SHA,
            "w0_fingerprint": W0,
            "seed": 42,
            "transform": "canonical_v1",
            "classes": classes,
            "validation_counts": {"plantdoc": 269, "plantvillage": 4344},
            "shard_counts": shard_counts,
            "centralized_history": central_history,
            "non_iid_scenarios": ["label_alpha_0_1"],
            "scientific_stage2_complete": False,
            "models": models,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "protocol_id": PROTOCOL_ID,
            "evidence_level": "unavailable",
            "reason": str(exc),
            "scientific_stage2_complete": False,
            "models": [],
        }


def predict_published(image: Any, model_id: str, root: Path = STORE) -> dict[str, Any]:
    """Load only a pinned checkpoint, using the published canonical_v1 transform."""
    evidence = load_evidence(root)
    if not evidence["available"]:
        raise ValueError(evidence["reason"])
    model_info = next((m for m in evidence["models"] if m["model_id"] == model_id), None)
    if model_info is None:
        raise ValueError(f"Unknown published model: {model_id}")

    import numpy as np
    import torch
    from torch import nn
    from torch.serialization import safe_globals
    from torchvision import transforms as T
    from torchvision.models import mobilenet_v3_small
    from numpy._core.multiarray import _reconstruct

    path = root / model_info["checkpoint_relative_path"]
    # NumPy RNG state in the trusted training checkpoint requires these narrow
    # safe globals. The checkpoint is hashed before any deserialization.
    numpy_types = [type(np.dtype(kind)) for kind in (
        "float64", "int64", "float32", "uint32", "uint64", "int32", "uint8"
    )]
    with safe_globals([_reconstruct, np.ndarray, np.dtype, *numpy_types]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("release_manifest_sha256") != RELEASE_SHA
            or checkpoint.get("w0_fingerprint") != W0):
        raise ValueError("Checkpoint protocol metadata mismatch")
    if model_id in {"centralized", "fedavg"} and checkpoint.get("seed") != 42:
        raise ValueError("Checkpoint seed mismatch")
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if not isinstance(state, dict) or len(state) != 244:
        raise ValueError("Checkpoint model state is missing or malformed")

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict(state, strict=True)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    top_values, top_indices = probabilities.topk(3)
    classes = evidence["classes"]
    top3 = [
        {"class": classes[int(index)], "confidence": round(float(value) * 100, 2)}
        for value, index in zip(top_values, top_indices)
    ]
    margin = float(top_values[0] - top_values[1])
    return {
        "ready": True,
        "model_id": model_id,
        "method": model_info["method"],
        "branch": model_info["branch"],
        "commit": model_info["commit"],
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "seed": evidence["seed"],
        "metric_source": evidence["metrics_source"],
        "transform": evidence["transform"],
        "predicted_class": top3[0]["class"],
        "confidence_pct": top3[0]["confidence"],
        "top3": top3,
        "is_uncertain": bool(float(top_values[0]) < 0.5 or margin < 0.2),
        "confidence_calibrated": False,
        "model_validation": model_info["metrics"],
        "checkpoint_sha256": model_info["checkpoint_sha256"],
        "evidence_level": evidence["evidence_level"],
        "scope": "single cropped leaf image; validation is not an external-photo accuracy estimate",
    }

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

MANIFEST = reg.abs_path(reg.CANONICAL_MANIFEST_JSON)
OUT = reg.abs_path(reg.PROVENANCE_VALIDATION_JSON)


def _state_dict_from_checkpoint(payload: Any) -> Dict[str, torch.Tensor] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("model_state", "model_state_dict", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    return None


def _stable_state_hash(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _inspect_checkpoint(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "exists": path.exists(),
        "file_sha256": None,
        "state_sha256": None,
        "metadata": {},
    }
    if not path.exists():
        return result

    result["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        state = _state_dict_from_checkpoint(payload)
        if state is not None:
            result["state_sha256"] = _stable_state_hash(state)
        for key, value in payload.items():
            if key in {"model_state", "model_state_dict", "state_dict"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                result["metadata"][key] = value
            elif isinstance(value, dict):
                compact = {}
                for k2, v2 in value.items():
                    if isinstance(v2, (str, int, float, bool)) or v2 is None:
                        compact[k2] = v2
                if compact:
                    result["metadata"][key] = compact
    return result


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    benchmarks = manifest.get("benchmarks", {})

    central_path = ROOT / benchmarks.get("centralized", {}).get("checkpoint", reg.CENTRALIZED_BEST_PT)
    fed_path = ROOT / benchmarks.get("fedavg", {}).get("checkpoint", reg.FEDAVG_BEST_PT)

    central = _inspect_checkpoint(central_path)
    fed = _inspect_checkpoint(fed_path)

    issues = []

    # Tách "không có file" khỏi "có file nhưng sai hash": hai tình huống cần hành động khác nhau.
    if not fed["exists"]:
        issues.append(
            {
                "severity": "critical",
                "code": "fedavg_checkpoint_missing",
                "message": f"Không có {fed['path']} trên máy này; chỉ còn metrics của run canonical. "
                           "Cần tải lại checkpoint thật từ Kaggle (results/fedavg/<run>/best.pt hiện là con trỏ Git-LFS).",
            }
        )
    if not central["exists"]:
        issues.append(
            {
                "severity": "critical",
                "code": "centralized_checkpoint_missing",
                "message": f"Không có {central['path']} trên máy này; số liệu Centralized chỉ là metrics lịch sử.",
            }
        )

    manifest_fed_hash = benchmarks.get("fedavg", {}).get("state_dict_sha256")
    if fed["exists"] and manifest_fed_hash and fed.get("state_sha256") != manifest_fed_hash:
        issues.append(
            {
                "severity": "critical",
                "code": "fedavg_state_hash_mismatch",
                "message": "Current fedavg_best.pt does not match canonical_manifest state_dict_sha256.",
                "manifest": manifest_fed_hash,
                "actual": fed.get("state_sha256"),
            }
        )

    manifest_fed_acc = benchmarks.get("fedavg", {}).get("test_accuracy")
    ckpt_fed_acc = fed.get("metadata", {}).get("test_accuracy")
    if isinstance(manifest_fed_acc, (int, float)) and isinstance(ckpt_fed_acc, (int, float)):
        if abs(float(manifest_fed_acc) - float(ckpt_fed_acc)) > 1e-9:
            issues.append(
                {
                    "severity": "critical",
                    "code": "fedavg_accuracy_metadata_mismatch",
                    "message": "Checkpoint metadata test_accuracy differs from canonical manifest.",
                    "manifest": manifest_fed_acc,
                    "actual": ckpt_fed_acc,
                }
            )

    manifest_c_epochs_text = benchmarks.get("centralized", {}).get("training_scope", "")
    ckpt_c_epochs = central.get("metadata", {}).get("epochs")
    if central["exists"] and "10 epochs" in str(manifest_c_epochs_text) and ckpt_c_epochs != 10:
        issues.append(
            {
                "severity": "critical",
                "code": "centralized_epoch_mismatch",
                "message": "Current centralized_best.pt epochs metadata differs from canonical manifest training scope.",
                "manifest": 10,
                "actual": ckpt_c_epochs,
            }
        )

    scientific_stage2_complete = bool(benchmarks.get("fedavg", {}).get("scientific_stage2_complete", False))
    if not scientific_stage2_complete:
        issues.append(
            {
                "severity": "warning",
                "code": "scientific_stage2_incomplete",
                "message": "canonical_manifest explicitly marks scientific_stage2_complete=false.",
            }
        )

    status = "verified" if not any(i["severity"] == "critical" for i in issues) else "unverified"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "canonical_run_id": manifest.get("canonical_run_id"),
        "centralized_checkpoint": central,
        "fedavg_checkpoint": fed,
        "issues": issues,
        "safe_reporting_policy": (
            "Only report scientific accuracy/loss/F1 from a real-data run whose dataset preflight passes and whose metrics/checkpoint provenance are linked."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))  # stdout ASCII: console Windows không in được tiếng Việt
    return 0 if status == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())

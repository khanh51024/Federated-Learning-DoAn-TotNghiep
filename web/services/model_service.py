"""Service inference/registry duy nhất của web.

Mọi chỗ cần biết "model nào có thể dùng, checkpoint nào tồn tại, weights ra sao"
đi qua đây — nguồn là artifacts/registry.json (xem docs/MODEL_REGISTRY.md).
Không module web nào được tự hardcode đường dẫn checkpoint nữa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src import registry as reg

try:
    from src.federated.weight_inspector import inspect_layer_weights
except ImportError:
    from train.weight_inspector import inspect_layer_weights


def registry_models() -> Dict[str, Dict[str, Any]]:
    """Trả về {model_id: entry} từ registry; dict rỗng nếu registry chưa sinh."""
    try:
        return {m["model_id"]: m for m in reg.iter_models()}
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {}


def model_exists(model_id: str) -> bool:
    return bool(registry_models().get(model_id, {}).get("checkpoint_exists"))


def status_checkpoints_info() -> Dict[str, bool]:
    """Thay cho việc tự stat từng file: lấy sự thật tồn tại từ registry."""
    return {
        "centralized_canonical": model_exists("centralized_canonical"),
        "fedavg_canonical": model_exists("fedavg_canonical"),
        "centralized_demo": model_exists("demo_centralized_v6"),
        "fedavg_demo": model_exists("demo_fedavg_v7"),
    }


def status_models_available() -> Dict[str, bool]:
    """Model inference có thực sự dùng được (web_visible + checkpoint tồn tại)."""
    return {
        "centralized": model_exists("demo_centralized_v6"),
        "fedavg": model_exists("demo_fedavg_v7"),
        "traced_reference": model_exists("plantvillage_traced_reference"),
    }


def inspect_weights(layer: Optional[str] = None) -> Dict[str, Any]:
    """
    Thống kê trọng số layer của checkpoint FedAvg.
    Ưu tiên slot canonical (chỉ run khoa học thật mới ghi); nếu chưa có thì dùng
    checkpoint demo hash-verified. Không fallback âm thầm sang model ngẫu nhiên.
    """
    fedavg_ckpt = reg.abs_path(reg.FEDAVG_BEST_PT)
    if not fedavg_ckpt.exists():
        fallback = reg.abs_path(reg.DEMO_BUNDLE_DIR) / "fedavg" / "checkpoint_best.pt"
        if fallback.exists() and model_exists("demo_fedavg_v7"):
            fedavg_ckpt = fallback

    if not fedavg_ckpt.exists():
        return {
            "available": False,
            "message": "Chưa có checkpoint FedAvg (canonical slot trống và demo chưa import) để kiểm tra trọng số.",
            "available_layers": [],
            "selected_layer_stats": None,
        }

    try:
        ckpt = torch.load(fedavg_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        weights = {k: v.cpu().detach() if hasattr(v, "cpu") else torch.tensor(v) for k, v in state.items()}
    except Exception as e:
        return {
            "available": False,
            "message": f"Không thể nạp checkpoint: {str(e)}",
            "available_layers": [],
            "selected_layer_stats": None,
        }

    # Reference để tính update norm: slot canonical nếu có
    ref_weights = None
    ref_path = reg.abs_path(reg.FEDAVG_LAST_PT)
    if not ref_path.exists():
        ref_path = reg.abs_path(reg.CENTRALIZED_BEST_PT)
    if ref_path.exists():
        try:
            r_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
            r_state = r_ckpt.get("model_state", r_ckpt) if isinstance(r_ckpt, dict) else r_ckpt
            ref_weights = {k: v.cpu().detach() if hasattr(v, "cpu") else torch.tensor(v) for k, v in r_state.items()}
        except Exception:
            ref_weights = None

    all_layers = sorted(weights.keys())
    selected_layer = layer if layer and layer in weights else all_layers[0]

    ref_tensor = ref_weights.get(selected_layer) if ref_weights else None
    stats = inspect_layer_weights(weights[selected_layer], reference_tensor=ref_tensor, layer_name=selected_layer)

    return {
        "available": True,
        "available_layers": all_layers,
        "selected_layer_stats": stats,
        "has_reference": ref_tensor is not None,
        "checkpoint_source": fedavg_ckpt.name,
    }


def artifact_scientific_status(federated_source: str) -> Dict[str, Any]:
    """Checkpoint có đủ điều kiện báo cáo khoa học không — theo provenance thật."""
    provenance_path = reg.abs_path(reg.PROVENANCE_VALIDATION_JSON)
    provenance: Dict[str, Any] = {}
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except Exception:
            provenance = {}

    centralized_valid = provenance.get("status") == "verified"
    centralized_reason = (
        "canonical provenance verified"
        if centralized_valid
        else "canonical checkpoint/metrics provenance is unverified; see artifacts/artifact_provenance_validation.json"
    )

    if federated_source == "canonical":
        fed_valid = provenance.get("status") == "verified"
        fed_reason = centralized_reason
    elif federated_source == "network":
        history_path = reg.abs_path(reg.NETWORK_FLOWER_HISTORY_JSON)
        history: Dict[str, Any] = {}
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = {}
        require_real = bool(history.get("require_real_clients", False))
        require_real_evaluation = bool(history.get("require_real_evaluation", False))
        final_test_metrics = history.get("final_test_metrics")
        audits = history.get("audit_rounds", []) or []
        client_rows = [c for rnd in audits for c in (rnd.get("clients", []) or [])]
        all_real = bool(client_rows) and all(c.get("data_mode") == "real" for c in client_rows)
        has_real_test = (
            require_real_evaluation
            and isinstance(final_test_metrics, dict)
            and all(k in final_test_metrics for k in ("accuracy", "loss", "macro_f1"))
        )
        fed_valid = require_real and all_real and has_real_test
        if not history_path.exists():
            fed_reason = "no Flower network history found"
        elif not require_real:
            fed_reason = "network run did not enforce require_real_clients"
        elif not client_rows:
            fed_reason = "network run has no per-client data-mode audit"
        elif not all_real:
            fed_reason = "one or more Flower clients used synthetic/unknown data"
        elif not require_real_evaluation:
            fed_reason = "network run did not enforce a real held-out evaluator"
        elif not isinstance(final_test_metrics, dict):
            fed_reason = "network run has no final held-out test metrics"
        else:
            fed_reason = "network run used real clients and a real held-out evaluator with final test metrics"
    else:
        fed_valid = False
        fed_reason = "unknown federated source"

    return {
        "centralized": {"scientific_valid": centralized_valid, "reason": centralized_reason},
        "federated": {"scientific_valid": fed_valid, "reason": fed_reason},
        "overall_scientific_valid": centralized_valid and fed_valid,
    }

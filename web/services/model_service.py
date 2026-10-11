"""Service inference/registry duy nhất của web.

Mọi chỗ cần biết "model nào có thể dùng, checkpoint nào tồn tại, weights ra sao"
đi qua đây — nguồn là artifacts/registry.json (xem docs/MODEL_REGISTRY.md).
Không module web nào được tự hardcode đường dẫn checkpoint nữa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src import registry as reg

from src.federated.weight_inspector import inspect_layer_weights


def registry_models() -> Dict[str, Dict[str, Any]]:
    """Trả về {model_id: entry} từ registry; dict rỗng nếu registry chưa sinh hoặc đang bị ghi dở."""
    try:
        return {m["model_id"]: m for m in reg.iter_models()}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
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


_HASH_CACHE: Dict[Any, str] = {}


def _file_sha256(path: Path) -> str:
    """SHA-256 của file, cache theo (đường dẫn, kích thước, mtime) để không băm lại mỗi request."""
    import hashlib

    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def is_web_visible(model_id: str) -> bool:
    return registry_models().get(model_id, {}).get("web_visible") is True


def status_models_available() -> Dict[str, bool]:
    """Model inference có thực sự dùng được: đã qua đủ 5 điều kiện của cổng registry (web_visible)."""
    return {
        "centralized": is_web_visible("demo_centralized_v6"),
        "fedavg": is_web_visible("demo_fedavg_v7"),
        "traced_reference": is_web_visible("plantvillage_traced_reference"),
    }


# ---------------------------------------------------------------------------
# Một cửa inference: mọi dự đoán trên web đi qua predict()
# ---------------------------------------------------------------------------
class RegistryUnavailable(RuntimeError):
    """artifacts/registry.json chưa được sinh hoặc không đọc được."""


class UnknownModel(KeyError):
    """model_id không có trong registry."""


class ModelNotAllowed(RuntimeError):
    """Model có trong registry nhưng chưa đạt cổng web_visible."""

    def __init__(self, model_id: str, reasons: List[str]):
        self.model_id = model_id
        self.reasons = reasons
        super().__init__(f"{model_id} chưa đạt cổng hiển thị: {'; '.join(reasons) or 'không rõ lý do'}")


def run_backend(entry: Dict[str, Any], image: Any, temperature: Optional[float] = None) -> Dict[str, Any]:
    """Gọi đúng backend mà entry registry khai báo. Không kiểm web_visible (dành cho bài kiểm tra của registry)."""
    inference = entry.get("inference") or {}
    backend = inference.get("backend")
    if backend == reg.BACKEND_PUBLISHED_BUNDLE:
        from web.demo_evidence import predict_published

        return predict_published(image, inference["key"], temperature=temperature)
    if backend == reg.BACKEND_TORCHSCRIPT_REFERENCE:
        from web import plantvillage_reference as reference

        # Kiểm lại hash MỖI lần dùng (bundle published tự kiểm hash trong load_evidence; TorchScript thì chưa)
        if _file_sha256(reference.MODEL_FILE) != reg.PLANTVILLAGE_TRACED_SHA256:
            raise ValueError("PlantVillage TorchScript đã đổi so với hash đã ghim — chạy lại scripts/update_status.py")
        return reference.predict_plantvillage(image)
    raise ValueError(f"{entry.get('model_id')}: backend inference {backend!r} không được hỗ trợ")


def _gated_entry(model_id: str) -> Dict[str, Any]:
    models = registry_models()
    if not models:
        raise RegistryUnavailable("Chưa có artifacts/registry.json — chạy scripts/update_status.py")
    entry = models.get(model_id)
    if entry is None:
        raise UnknownModel(model_id)
    if entry.get("web_visible") is not True:
        raise ModelNotAllowed(model_id, list(entry.get("web_gate_failures") or []))
    return entry


def explain(model_id: str, image: Any, class_name: str) -> Dict[str, Any]:
    """Grad-CAM qua cùng cổng registry; chỉ hỗ trợ bundle published (TorchScript tham chiếu không có)."""
    entry = _gated_entry(model_id)
    inference = entry.get("inference") or {}
    if inference.get("backend") != reg.BACKEND_PUBLISHED_BUNDLE:
        raise ValueError(f"{model_id}: không hỗ trợ Grad-CAM")
    from web.demo_evidence import explain_published

    return explain_published(image, inference["key"], class_name)


def predict(model_id: str, image: Any, temperature: Optional[float] = None) -> Dict[str, Any]:
    """Dự đoán qua registry: từ chối model không tồn tại hoặc chưa đạt cổng web_visible."""
    entry = _gated_entry(model_id)
    result = dict(run_backend(entry, image, temperature))
    result["registry"] = {
        "model_id": entry["model_id"],
        "role": entry.get("role"),
        "display_label": entry.get("display_label"),
        "pipeline": entry.get("pipeline"),
        "dataset_id": entry.get("dataset_id"),
    }
    return result


_SHARD_CACHE: Dict[str, Any] = {}


def _shard_report() -> Optional[Dict[str, Any]]:
    """artifacts/client_shard_class_counts.json (cache theo kích thước+mtime); None nếu thiếu, hỏng hoặc không nhất quán."""
    path = reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (stat.st_size, stat.st_mtime_ns)
    if _SHARD_CACHE.get("key") != key:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict) or data.get("consistent") is not True:
            data = None
        _SHARD_CACHE.update(key=key, data=data)
    return _SHARD_CACHE.get("data")


def shard_report() -> Optional[Dict[str, Any]]:
    """Báo cáo shard đã kiểm nhất quán (cho evidence_service); None nếu thiếu/hỏng/không nhất quán."""
    return _shard_report()


def known_classes() -> List[str]:
    """38 lớp theo báo cáo shard (nguồn kiểm hash); rỗng nếu chưa có báo cáo."""
    report = _shard_report()
    return list(report.get("classes") or []) if report else []


def shard_context(
    federated_source: str, top3: List[Dict[str, Any]], true_class: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Giải thích kết quả: với mỗi lớp trong top-3, shard huấn luyện của client đang chọn có bao nhiêu ảnh.

    Chỉ trả khi báo cáo shard tồn tại và nhất quán; không có thì None (web vẫn chạy bình thường)."""
    report = _shard_report()
    if report is None:
        return None
    client = federated_source.replace("local_", "client_") if federated_source.startswith("local_") else None
    shard = (report.get("clients") or {}).get(client) if client else None
    if client and shard is None:
        return None
    totals = report.get("totals_all_clients") or {}
    classes = []
    # Nhãn thật (nếu người dùng biết) đứng đầu: đây là thông điệp giải thích mạnh nhất ("client này có 0 ảnh lớp đó")
    wanted = ([true_class] if true_class in totals else []) + [
        e.get("class") for e in (top3 or []) if e.get("class") in totals and e.get("class") != true_class
    ]
    for name in wanted:
        in_client = None if shard is None else int(shard["per_class"].get(name, 0))
        total = int(totals[name])
        classes.append({
            "class": name,
            "is_true_label": name == true_class,
            "in_top3": any(e.get("class") == name for e in (top3 or [])),
            "client_count": in_client,
            "all_clients_count": total,
            "client_share_pct": None if in_client is None or total == 0 else round(100.0 * in_client / total, 1),
        })
    return {
        "partition_scheme": report.get("partition_scheme"),
        "client": client,
        "client_n_k": None if shard is None else int(shard["n_k"]),
        "client_classes_with_images": None if shard is None else int(shard["classes_with_images"]),
        "classes": classes,
        "evidence": report.get("evidence"),
    }


def bundle_gate() -> tuple[bool, List[str]]:
    """Bộ đối chứng published là MỘT khối (cùng protocol/split): chỉ hiển thị khi MỌI model của nó đã qua cổng web_visible."""
    entries = [
        e for e in registry_models().values()
        if (e.get("inference") or {}).get("backend") == reg.BACKEND_PUBLISHED_BUNDLE
    ]
    if not entries:
        return False, ["registry không có model nào thuộc bundle đã công bố — chạy scripts/update_status.py"]
    blocked = [
        f"{e['model_id']}: {'; '.join(e.get('web_gate_failures') or ['không rõ lý do'])}"
        for e in entries if e.get("web_visible") is not True
    ]
    return (not blocked), blocked


def list_models() -> List[Dict[str, Any]]:
    """Danh sách model được phép hiển thị trên web (đã qua cổng), kèm nhãn ngữ cảnh."""
    out: List[Dict[str, Any]] = []
    for entry in registry_models().values():
        if entry.get("web_visible") is not True:
            continue
        metrics = entry.get("metrics") or {}
        has_metrics = bool(metrics.get("plantdoc") or metrics.get("plantvillage"))
        out.append({
            "model_id": entry["model_id"],
            "display_name": entry.get("display_name"),
            "display_label": entry.get("display_label"),
            "role": entry.get("role"),
            "family": entry.get("family"),
            "inference_key": (entry.get("inference") or {}).get("key"),
            "pipeline": entry.get("pipeline"),
            "dataset_id": entry.get("dataset_id"),
            "protocol_id": entry.get("protocol_id"),
            "checkpoint_status": entry.get("checkpoint_status"),
            "inference_test": (entry.get("inference_test") or {}).get("status"),
            "metrics_summary": {
                "plantdoc_top1": (metrics.get("plantdoc") or {}).get("top1_accuracy"),
                "plantvillage_top1": (metrics.get("plantvillage") or {}).get("top1_accuracy"),
            } if has_metrics else None,
            "metrics_source": entry.get("metrics_source"),
        })
    return sorted(out, key=lambda m: m["model_id"])


def _trusted_checkpoint(model_id: str, expected_status: str) -> Optional[Path]:
    """Đường dẫn checkpoint CHỈ khi registry xác nhận trạng thái tin cậy và hash trên đĩa khớp entry."""
    entry = registry_models().get(model_id) or {}
    if entry.get("checkpoint_status") != expected_status or not entry.get("checkpoint_path"):
        return None
    path = reg.abs_path(entry["checkpoint_path"])
    if not path.is_file() or _file_sha256(path) != entry.get("file_sha256"):
        return None
    return path


def inspect_weights(layer: Optional[str] = None) -> Dict[str, Any]:
    """
    Thống kê trọng số layer của checkpoint FedAvg — cùng cổng tin cậy với inference:
    ưu tiên slot canonical nếu registry xác nhận `matches_manifest`; nếu không dùng checkpoint demo
    chỉ khi model đó `web_visible` và hash trên đĩa khớp registry. Không fallback âm thầm.
    """
    fedavg_ckpt = _trusted_checkpoint("fedavg_canonical", "matches_manifest")
    if fedavg_ckpt is None and is_web_visible("demo_fedavg_v7"):
        fedavg_ckpt = _trusted_checkpoint("demo_fedavg_v7", "hash_verified")

    if fedavg_ckpt is None:
        return {
            "available": False,
            "message": "Chưa có checkpoint FedAvg đủ tin cậy (canonical chưa khớp manifest, demo chưa qua cổng registry) để kiểm tra trọng số.",
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

    all_layers = sorted(weights.keys())
    selected_layer = layer if layer and layer in weights else all_layers[0]
    stats = inspect_layer_weights(weights[selected_layer], reference_tensor=None, layer_name=selected_layer)

    return {
        "available": True,
        "available_layers": all_layers,
        "selected_layer_stats": stats,
        "has_reference": False,
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

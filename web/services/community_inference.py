"""Suy luận bằng mô hình cộng đồng: backbone + classifier[0..2] của FedAvg v7 (đóng băng) + head cộng đồng hiện hành.

Cùng một vector đặc trưng được đưa qua head cộng đồng và head gốc để so sánh. Đây là mô hình THỬ NGHIỆM: head thay đổi theo
từng vòng nên không nằm trong registry tĩnh và không bao giờ được trộn vào bảng bảy mô hình đã kiểm hash.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src import registry as reg
from src.federated import community as core
from web.services import model_service


def _top3(logits: np.ndarray, classes: List[str]) -> List[Dict[str, Any]]:
    shifted = logits.astype(np.float64) - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    order = np.argsort(-probabilities)[:3]
    return [{"class": classes[int(i)], "confidence": round(float(probabilities[int(i)]) * 100, 2)} for i in order]


def features_of(image: Any) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Vector 1024 chiều của một ảnh PIL qua phần đóng băng của FedAvg v7 (đường PyTorch đã kiểm SHA của bundle)."""
    import torch

    from web.demo_evidence import STORE, _input_tensor, _published_model, load_evidence

    if not model_service.is_web_visible(reg.COMMUNITY_MODEL_ID):
        raise ValueError(f"{reg.COMMUNITY_MODEL_ID} chưa đạt cổng hiển thị của registry")
    evidence = load_evidence()
    if not evidence.get("available"):
        raise ValueError(evidence.get("reason") or "bundle không hợp lệ")
    info, model = _published_model(evidence, "fedavg", STORE)
    tensor = _input_tensor(image)
    with torch.inference_mode():
        pooled = torch.flatten(model.avgpool(model.features(tensor)), 1)
        features = model.classifier[:3](pooled)[0].numpy().astype(np.float32)
    return features, list(evidence["classes"]), info


def predict(image: Any, version: int, head: core.Head, head_sha: str, base_head: core.Head,
            metrics: Dict[str, Any], base_metrics: Dict[str, Any]) -> Dict[str, Any]:
    features, classes, info = features_of(image)

    def logits(h: core.Head) -> np.ndarray:
        return h[core.WEIGHT_KEY] @ features + h[core.BIAS_KEY]

    community = _top3(logits(head), classes)
    baseline = _top3(logits(base_head), classes)
    return {
        "ready": True,
        "kind": "community_experimental",
        "version": version,
        "head_sha": head_sha,
        "predicted_class": community[0]["class"],
        "confidence_pct": community[0]["confidence"],
        "top3": community,
        "baseline": {
            "model": "FedAvg v7", "checkpoint_sha256": info["checkpoint_sha256"],
            "predicted_class": baseline[0]["class"], "confidence_pct": baseline[0]["confidence"], "top3": baseline,
        },
        "same_as_baseline": community[0]["class"] == baseline[0]["class"],
        "metrics": {k: metrics.get(k) for k in ("accuracy", "macro_recall", "plantvillage_accuracy", "plantdoc_accuracy")},
        "base_metrics": {k: base_metrics.get(k) for k in ("accuracy", "macro_recall", "plantvillage_accuracy", "plantdoc_accuracy")},
        "caveat": (
            "Mô hình cộng đồng là thử nghiệm: chỉ lớp cuối được học từ ảnh người dùng tự gán nhãn. Độ chính xác giám sát tính "
            "trên tập đã dùng để chọn mô hình gốc, không phải số liệu khoa học."
        ),
    }

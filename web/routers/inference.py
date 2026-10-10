"""Router inference: dual inference, PlantVillage reference, demo evidence, compare."""
from __future__ import annotations

import io
from typing import Any, Dict

from fastapi import File, HTTPException, Query, UploadFile
from fastapi.routing import APIRouter
from PIL import Image

from web.demo_evidence import PROTOCOL_ID, load_evidence, predict_published
from web.plantvillage_reference import predict_plantvillage

router = APIRouter()


@router.get("/api/demo/evidence")
def get_demo_evidence() -> Dict[str, Any]:
    return load_evidence()


@router.get("/api/compare")
def get_comparison_table(protocol_id: str = PROTOCOL_ID) -> Dict[str, Any]:
    """Return comparison data without inventing or silently trusting stale metrics."""
    if protocol_id != PROTOCOL_ID:
        raise HTTPException(status_code=400, detail="Protocol không có bộ đối chứng được kiểm hash.")
    evidence = load_evidence()
    if not evidence["available"]:
        return {"available": False, "reason": evidence["reason"], "rows": [], "models": []}
    rows = [
        {
            "model_id": model["model_id"],
            "model": model["name"],
            "phase": model["phase"],
            "method": model["method"],
            "source": source,
            "protocol_id": evidence["protocol_id"],
            "release_manifest_sha256": evidence["release_manifest_sha256"],
            "validation_manifest_sha256": evidence["validation_manifest_sha256"],
            "seed": evidence["seed"],
            "metric_source": evidence["metrics_source"],
            "evidence_level": evidence["evidence_level"],
            "branch": model["branch"],
            "commit": model["commit"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            **model["metrics"][source],
        }
        for model in evidence["models"]
        for source in ("plantdoc", "plantvillage")
    ]
    return {
        "available": True,
        "protocol_id": evidence["protocol_id"],
        "evidence_level": evidence["evidence_level"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "seed": evidence["seed"],
        "rows": rows,
        "models": evidence["models"],
        "scientific_stage2_complete": False,
    }


@router.post("/api/inference/plantvillage")
async def run_plantvillage_reference(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Standalone PlantVillage reference; independent from FedAvg comparison."""
    contents = await file.read()
    try:
        with Image.open(io.BytesIO(contents)) as image:
            return predict_plantvillage(image)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Ảnh hoặc model PlantVillage không hợp lệ: {exc}") from exc


@router.post("/api/inference")
async def run_dual_inference(
    file: UploadFile = File(...),
    federated_source: str = Query("fedavg"),
) -> Dict[str, Any]:
    """
    Inference song song độc lập (Dual Model) trên bộ checkpoint published demo
    (hash-verified qua web/demo_evidence.py):
    - Cột Centralized: model centralized của protocol published.
    - Cột Federated: fedavg hoặc local_00..04 cùng protocol.
    - Tuyệt đối không tráo đổi hay dùng chung một file TorchScript.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File tải lên không phải là ảnh hợp lệ")

    if federated_source not in {"fedavg", *(f"local_{i:02d}" for i in range(5))}:
        raise HTTPException(status_code=400, detail="Chỉ chọn FedAvg v7 hoặc một trong năm Local-only v7 cùng protocol.")
    try:
        central = predict_published(image, "centralized")
        selected = predict_published(image, federated_source)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Artifact suy luận chưa sẵn sàng: {exc}") from exc
    selected_name = "FedAvg v7" if federated_source == "fedavg" else f"Local {federated_source[-2:]}"
    return {
        "filename": file.filename,
        "federated_source": federated_source,
        "selected_model_name": selected_name,
        "protocol_id": PROTOCOL_ID,
        "scientific_status": {
            "evidence_level": "published_artifact_hash_verified_not_locally_reevaluated",
            "overall_scientific_valid": False,
            "reason": "Cùng validation nội bộ; chưa tái đánh giá tại máy demo hoặc trên ảnh ngoài mạng.",
        },
        "centralized": central,
        "federated": selected,
    }

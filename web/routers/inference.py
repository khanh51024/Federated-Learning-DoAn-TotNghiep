"""Router inference: dual inference, PlantVillage reference, demo evidence, compare."""
from __future__ import annotations

import io
from typing import Any, Dict

from fastapi import File, HTTPException, Query, UploadFile
from fastapi.routing import APIRouter
from PIL import Image

from web.demo_evidence import PROTOCOL_ID, load_evidence
from web.services import evidence_service, model_service

router = APIRouter()


def _predict(model_id: str, image: Image.Image) -> Dict[str, Any]:
    """Cửa duy nhất tới model: lỗi registry/cổng được đổi thành HTTP có lý do rõ ràng."""
    try:
        return model_service.predict(model_id, image)
    except model_service.UnknownModel as exc:
        # model_id do server tự suy ra từ lựa chọn đã kiểm; thiếu nghĩa là registry chưa được sinh đúng
        raise HTTPException(status_code=503, detail=f"Registry chưa có model: {exc} — chạy scripts/update_status.py") from exc
    except model_service.ModelNotAllowed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except model_service.RegistryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/models")
def get_models() -> Dict[str, Any]:
    """Các model được phép dùng trên web (đã qua 5 điều kiện của registry), kèm nhãn Demo/Tham chiếu."""
    models = model_service.list_models()
    return {"available": bool(models), "count": len(models), "models": models}


def _gated_evidence() -> Dict[str, Any]:
    """Số liệu đối chứng chỉ ra khi registry cho phép CẢ bộ; ngược lại giữ lại toàn bộ kèm lý do (không lọc nửa vời)."""
    ok, reasons = model_service.bundle_gate()
    if not ok:
        return {
            "available": False,
            "protocol_id": PROTOCOL_ID,
            "evidence_level": "withheld_by_registry_gate",
            "reason": "Cổng registry chặn bộ đối chứng — " + "; ".join(reasons),
            "scientific_stage2_complete": False,
            "models": [],
        }
    return load_evidence()


@router.get("/api/demo/evidence")
def get_demo_evidence() -> Dict[str, Any]:
    return _gated_evidence()


@router.get("/api/evidence/summary")
def get_evidence_summary() -> Dict[str, Any]:
    """Bằng chứng cho trang trình bày: tái đánh giá, PlantDoc theo lớp, hiệu chuẩn, shard, audit rò rỉ, checklist.

    Mỗi phần fail-closed riêng; toàn bộ phần có số liệu mô hình chỉ ra khi cổng registry cho phép CẢ bộ."""
    return evidence_service.summary(_gated_evidence())


@router.get("/api/compare")
def get_comparison_table(protocol_id: str = PROTOCOL_ID) -> Dict[str, Any]:
    """Return comparison data without inventing or silently trusting stale metrics."""
    if protocol_id != PROTOCOL_ID:
        raise HTTPException(status_code=400, detail="Protocol không có bộ đối chứng được kiểm hash.")
    evidence = _gated_evidence()
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
            return _predict("plantvillage_traced_reference", image)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Ảnh hoặc model PlantVillage không hợp lệ: {exc}") from exc


@router.post("/api/inference")
async def run_dual_inference(
    file: UploadFile = File(...),
    federated_source: str = Query("fedavg"),
    true_class: str | None = Query(None, description="Nhãn thật của ảnh nếu biết (chỉ để giải thích shard, không ảnh hưởng dự đoán)"),
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
    if true_class:
        classes = model_service.known_classes()
        if classes and true_class not in classes:
            raise HTTPException(status_code=400, detail="true_class không thuộc 38 lớp của taxonomy.")
    selected_id = "demo_fedavg_v7" if federated_source == "fedavg" else f"demo_{federated_source}"
    try:
        central = _predict("demo_centralized_v6", image)
        selected = _predict(selected_id, image)
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
        # Vì sao Local-only sai: shard huấn luyện của client đó có (hay không có) ảnh của lớp được dự đoán
        "shard_context": model_service.shard_context(federated_source, selected.get("top3") or [], true_class or None),
    }

"""Router inference: dual inference, 7 mô hình một lần, PlantVillage reference, demo evidence, compare."""
from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional

from fastapi import File, HTTPException, Query, UploadFile
from fastapi.routing import APIRouter
from PIL import Image
from starlette.concurrency import run_in_threadpool

from web.demo_evidence import PROTOCOL_ID, load_evidence
from web.services import evidence_service, model_service

router = APIRouter()


# Bảy mô hình của bundle đối chứng: (khóa hiển thị, model_id trong registry)
ALL_MODELS = (
    ("centralized", "demo_centralized_v6"),
    ("fedavg", "demo_fedavg_v7"),
    *((f"local_{i:02d}", f"demo_local_{i:02d}") for i in range(5)),
)
# Giới hạn ảnh tải lên. MAX_UPLOAD_BYTES phải khớp MAX_UPLOAD_BYTES trong web/static/js/pages/diagnosis.js
# (tests/test_web_static_frontend.py kiểm hai số này bằng nhau). Mô hình chỉ cần 224 px nên 40 triệu điểm ảnh là dư.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SCIENTIFIC_STATUS = {
    "evidence_level": "published_artifact_hash_verified",
    "overall_scientific_valid": False,
    "reason": ("Checkpoint đã kiểm SHA-256 trước khi nạp. Một ảnh tải lên không phải phép đo độ chính xác; "
               "xác suất thô chưa hiệu chuẩn. Tái đánh giá và test giữ riêng (nếu có báo cáo khớp) xem ở /api/evidence/summary."),
}


def _predict(model_id: str, image: Image.Image, temperature: Optional[float] = None) -> Dict[str, Any]:
    """Cửa duy nhất tới model: lỗi registry/cổng được đổi thành HTTP có lý do rõ ràng."""
    try:
        return model_service.predict(model_id, image, temperature)
    except model_service.UnknownModel as exc:
        # model_id do server tự suy ra từ lựa chọn đã kiểm; thiếu nghĩa là registry chưa được sinh đúng
        raise HTTPException(status_code=503, detail=f"Registry chưa có model: {exc} — chạy scripts/update_status.py") from exc
    except model_service.ModelNotAllowed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except model_service.RegistryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _explain(model_id: str, image: Image.Image, class_name: str) -> Dict[str, Any]:
    try:
        return model_service.explain(model_id, image, class_name)
    except (model_service.UnknownModel, model_service.ModelNotAllowed, model_service.RegistryUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _reject_oversized(data: bytes) -> None:
    """Gọi sau khi đọc tối đa MAX_UPLOAD_BYTES + 1 byte: dư một byte nghĩa là tệp vượt giới hạn (không đọc hết vào RAM)."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Ảnh vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Hãy chọn ảnh nhỏ hơn.")


def _reject_too_many_pixels(size: tuple[int, int]) -> None:
    """Kiểm trên kích thước đọc từ header (chưa giải mã) để ảnh nén cực lớn không làm cạn RAM khi giải mã."""
    pixels = size[0] * size[1]
    if pixels > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Ảnh có {pixels / 1_000_000:.0f} triệu điểm ảnh, vượt giới hạn {MAX_IMAGE_PIXELS // 1_000_000} triệu. Hãy thu nhỏ ảnh trước khi tải lên.",
        )


def _read_image(file: UploadFile) -> Image.Image:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    _reject_oversized(data)
    try:
        image = Image.open(io.BytesIO(data))  # mở lười: chỉ đọc header
        size = image.size
    except Exception:
        raise HTTPException(status_code=400, detail="File tải lên không phải là ảnh hợp lệ")
    _reject_too_many_pixels(size)
    try:
        return image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File tải lên không phải là ảnh hợp lệ")


def _check_true_class(true_class: Optional[str]) -> None:
    if true_class:
        classes = model_service.known_classes()
        if classes and true_class not in classes:
            raise HTTPException(status_code=400, detail="true_class không thuộc 38 lớp của taxonomy.")


def _temperatures() -> Dict[str, float]:
    """T đã kiểm danh tính (báo cáo tái đánh giá khớp bundle); rỗng nếu không có — khi đó không có khối `calibrated`."""
    return evidence_service.temperatures(_gated_evidence())


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
    """Standalone PlantVillage reference; independent from FedAvg comparison.

    Hàm `async` nên thân nó chạy thẳng trên event loop: forward của torch phải đẩy sang threadpool,
    nếu không một ảnh đang suy luận sẽ chặn cả `/api/status` mà giao diện đang poll.
    (`/api/inference/all` và `/api/inference` là hàm đồng bộ nên FastAPI tự chạy trong threadpool.)"""
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    _reject_oversized(contents)
    try:
        with Image.open(io.BytesIO(contents)) as image:
            _reject_too_many_pixels(image.size)
            return await run_in_threadpool(_predict, "plantvillage_traced_reference", image)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Ảnh hoặc model PlantVillage không hợp lệ: {exc}") from exc


@router.get("/api/evidence/details")
def get_evidence_details() -> Dict[str, Any]:
    """Ma trận nhầm lẫn (val + test giữ riêng) và thư viện ảnh "sai mà tự tin"; fail-closed theo cùng cổng registry."""
    return evidence_service.details(_gated_evidence())


@router.get("/api/evidence/partitions")
def get_partition_schemes() -> Dict[str, Any]:
    """Phân bố lớp theo client của 12 cách chia trong release (chỉ dữ liệu; chỉ α=0,1 có mô hình đã huấn luyện)."""
    return evidence_service.partitions(_gated_evidence())


@router.post("/api/inference/all")
def run_all_inference(
    file: UploadFile = File(...),
    true_class: str | None = Query(None, description="Nhãn thật nếu biết (chỉ để giải thích shard, không ảnh hưởng dự đoán)"),
    explain: bool = Query(False, description="Kèm ảnh Grad-CAM cho lớp mà từng mô hình dự đoán"),
) -> Dict[str, Any]:
    """Một ảnh qua cả 7 mô hình (mỗi mô hình đúng một lần, qua cổng registry). Hàm đồng bộ: FastAPI chạy trong threadpool
    nên poll /api/status không bị chặn khi đang suy luận."""
    image = _read_image(file)
    _check_true_class(true_class)
    temperatures = _temperatures()
    started = time.perf_counter()
    try:
        results = {key: _predict(registry_id, image, temperatures.get(key)) for key, registry_id in ALL_MODELS}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Artifact suy luận chưa sẵn sàng: {exc}") from exc
    predict_ms = (time.perf_counter() - started) * 1000
    explanations: Dict[str, Any] = {}
    if explain:
        started_explain = time.perf_counter()
        # Grad-CAM là phần phụ: lỗi của một mô hình chỉ ghi vào mục của mô hình đó, không bỏ 7 kết quả dự đoán
        for key, registry_id in ALL_MODELS:
            try:
                explanations[key] = _explain(registry_id, image, results[key]["predicted_class"])
            except HTTPException as exc:
                explanations[key] = {"error": str(exc.detail)}
            except Exception as exc:  # noqa: BLE001 — lỗi tính toán bất kỳ không được làm hỏng phần dự đoán
                explanations[key] = {"error": f"Không tạo được Grad-CAM: {type(exc).__name__}"}
        explain_ms = (time.perf_counter() - started_explain) * 1000
    return {
        "filename": file.filename,
        "protocol_id": PROTOCOL_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "order": [key for key, _ in ALL_MODELS],
        "models": results,
        "shard_context": {
            key: model_service.shard_context(key, results[key].get("top3") or [], true_class or None)
            for key, _ in ALL_MODELS if key != "centralized"
        },
        "explain": explanations,
        "calibration_available": bool(temperatures),
        "timing_ms": {"predict": round(predict_ms, 1), **({"explain": round(explain_ms, 1)} if explain else {})},
    }


@router.post("/api/inference")
def run_dual_inference(
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
    image = _read_image(file)
    if federated_source not in {"fedavg", *(f"local_{i:02d}" for i in range(5))}:
        raise HTTPException(status_code=400, detail="Chỉ chọn FedAvg v7 hoặc một trong năm Local-only v7 cùng protocol.")
    _check_true_class(true_class)
    selected_id = "demo_fedavg_v7" if federated_source == "fedavg" else f"demo_{federated_source}"
    temperatures = _temperatures()
    try:
        central = _predict("demo_centralized_v6", image, temperatures.get("centralized"))
        selected = _predict(selected_id, image, temperatures.get(federated_source))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Artifact suy luận chưa sẵn sàng: {exc}") from exc
    selected_name = "FedAvg v7" if federated_source == "fedavg" else f"Local {federated_source[-2:]}"
    return {
        "filename": file.filename,
        "federated_source": federated_source,
        "selected_model_name": selected_name,
        "protocol_id": PROTOCOL_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "centralized": central,
        "federated": selected,
        # Vì sao Local-only sai: shard huấn luyện của client đó có (hay không có) ảnh của lớp được dự đoán
        "shard_context": model_service.shard_context(federated_source, selected.get("top3") or [], true_class or None),
    }

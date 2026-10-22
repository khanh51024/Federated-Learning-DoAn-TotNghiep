"""Evidence for the model bundle the web serves (artifacts/active_bundle.json -> <bundle>/import_manifest.json).

The checks establish byte provenance of the bundle files and consistency of the identity written in its manifest (release, validation set,
seed, model list). Identity is NOT pinned here as constants: it comes from the bundle manifest (src/bundle.py), so swapping the bundle does not
edit this file. Without the image release they cannot establish that validation was independently reproduced.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Any

from src import bundle, registry as reg
from scripts.import_branch_demo import CONTRACT_FILES, classes_from_taxonomy, sha256


ROOT = Path(__file__).resolve().parents[1]
STORE = reg.abs_path(reg.DEMO_BUNDLE_DIR)
PROTOCOL_ID = reg.DEMO_PROTOCOL_ID
_HEX64 = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


# Băm lại ~60 MB mỗi lần suy luận là phần tốn nhất của một request. File lớn (checkpoint) được nhớ hash theo
# (đường dẫn, kích thước, mtime_ns): đổi nội dung là đổi mtime => băm lại. File nhỏ (JSON, CSV, contract) luôn băm lại.
_BIG_FILE_BYTES = 1 << 20
_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_LOCK = threading.Lock()


def _sha256_cached(path: Path) -> str:
    stat = path.stat()
    if stat.st_size < _BIG_FILE_BYTES:
        return sha256(path)
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    value = sha256(path)
    with _LOCK:
        _HASH_CACHE[key] = value
    return value


def _verified_manifest(root: Path, name: str, branch: str, release: str, validation: str) -> dict[str, Any]:
    result_dir = root / name
    manifest = json.loads((result_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("branch") != branch or manifest.get("release_manifest_sha256") != release
            or manifest.get("validation_manifest_sha256") != validation):
        raise ValueError(f"{branch}: mismatched artifact manifest")
    for relative, expected in manifest["files"].items():
        candidate = (result_dir / relative).resolve()
        if not candidate.is_relative_to(result_dir.resolve()):
            raise ValueError(f"{branch}: artifact path escapes result directory")
        if (not candidate.is_file() or candidate.stat().st_size != expected["bytes"]
                or _sha256_cached(candidate) != expected["sha256"]):
            raise ValueError(f"{branch}: missing or hash-mismatched {relative}")
    return manifest


def load_evidence(root: Path = STORE) -> dict[str, Any]:
    """Fail closed: return no metrics or selectable models on any mismatch."""
    try:
        manifest_bytes = (root / "import_manifest.json").read_bytes()
        pinned = bundle.pinned_manifest_sha256(root)
        if pinned and hashlib.sha256(manifest_bytes).hexdigest() != pinned:
            raise ValueError("import manifest differs from the hash pinned by artifacts/active_bundle.json")
        metadata = json.loads(manifest_bytes.decode("utf-8"))
        protocol_id = bundle.protocol_id(metadata)
        release, validation = metadata.get("release_manifest_sha256"), metadata.get("validation_manifest_sha256")
        w0, seed = metadata.get("w0_fingerprint"), metadata.get("seed")
        counts = bundle.validation_counts(metadata)
        if (not _is_sha256(release) or not _is_sha256(validation) or not isinstance(seed, int)
                or (w0 is not None and not _is_sha256(w0))
                or metadata.get("transform") != "canonical_v1"
                or metadata.get("num_classes") != 38
                or not counts or any(n <= 0 for n in counts.values())):
            raise ValueError("import manifest protocol identity mismatch")
        specs = bundle.model_specs(metadata)
        if len({s["model_id"] for s in specs}) != len(specs) or len({s["display_name"] for s in specs}) != len(specs):
            raise ValueError("duplicate model ids or display names in bundle manifest")
        sources = metadata.get("sources") or {}
        for name in {s["dir"] for s in specs}:
            ref = sources.get(name) or {}
            if not ref.get("branch") or not ref.get("commit"):
                raise ValueError(f"{name}: source ref missing")
        contract_hashes = metadata.get("contract_hashes") or {}
        if set(contract_hashes) != set(CONTRACT_FILES):
            raise ValueError("contract hashes missing from bundle manifest")
        for relative in CONTRACT_FILES:
            name = Path(relative).name
            if _sha256_cached(root / "contract" / name) != contract_hashes.get(relative):
                raise ValueError(f"contract hash mismatch: {name}")
        classes = classes_from_taxonomy(root / "contract/taxonomy.py")
        if metadata.get("classes") != classes:
            raise ValueError("class order differs from pinned taxonomy")

        dirs = list(dict.fromkeys(s["dir"] for s in specs))
        manifests = {name: _verified_manifest(root, name, sources[name]["branch"], release, validation) for name in dirs}
        comparison_hashes = {
            item["files"]["validation_comparison.csv"]["sha256"] for item in manifests.values()
        }
        if len(comparison_hashes) != 1:
            raise ValueError("comparison bytes differ across branches")
        comparison = root / ("fedavg" if "fedavg" in dirs else dirs[0]) / "validation_comparison.csv"
        bundle.verify_comparison(comparison, manifests, metadata)
        partition = metadata.get("partition_scheme", "label_alpha_0_1")
        local_summary = json.loads((root / "local/combined_summary.json").read_text(encoding="utf-8"))
        if (local_summary.get("seed") != seed
                or (w0 is not None and local_summary.get("w0_fingerprint") != w0)
                or local_summary.get("release_manifest_sha256") != release
                or local_summary.get("partition_scheme") != partition):
            raise ValueError("Local-only summary protocol mismatch")
        shard_counts = {
            name: int(local_summary["clients"][name]["n_k"])
            for name in (f"client_{i:02d}" for i in range(5))
        }
        if sum(shard_counts.values()) != bundle.shard_total(metadata):
            raise ValueError("Client shard counts do not cover the published train split")
        central_summary = json.loads((root / "centralized/summary.json").read_text(encoding="utf-8"))
        if (central_summary.get("seed") != seed
                or (w0 is not None and central_summary.get("w0_fingerprint") != w0)
                or central_summary.get("release_manifest_sha256") != release):
            raise ValueError("Centralized summary protocol mismatch")
        central_history = [
            {"epoch": int(item["epoch"]), "val_accuracy": float(item["val_accuracy"]),
             "train_loss": float(item["train_loss"]),
             "val_loss": float(item["val_loss"]) if item.get("val_loss") is not None else None}
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
        for spec in specs:
            source = sources[spec["dir"]]
            checkpoint_sha = manifests[spec["dir"]]["files"][spec["file"]]["sha256"]
            models.append({
                "model_id": spec["model_id"],
                "name": spec["display_name"],
                "phase": spec["phase"],
                "method": spec["method"],
                "branch": source["branch"],
                "commit": source["commit"],
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_relative_path": f"{spec['dir']}/{spec['file']}",
                "metrics": by_name[spec["display_name"]],
            })
        return {
            "available": True,
            "bundle_id": metadata.get("bundle_id"),
            "protocol_id": protocol_id,
            "evidence_level": metadata.get("evidence_level") or "published_artifact_hash_verified_not_locally_reevaluated",
            "metrics_source": metadata.get("metrics_source") or "published validation_comparison.csv; no local image release",
            "release_manifest_sha256": release,
            "validation_manifest_sha256": validation,
            "w0_fingerprint": w0,
            "w0_in_checkpoints": bool(metadata.get("w0_in_checkpoints", True)),
            "seed": seed,
            "transform": "canonical_v1",
            "classes": classes,
            "validation_counts": counts,
            "shard_counts": shard_counts,
            "centralized_history": central_history,
            "non_iid_scenarios": [partition],
            "scientific_stage2_complete": False,
            "training": metadata.get("training"),
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


# Mô hình đã dựng được giữ trong RAM (~6 MB mỗi mô hình) theo (đường dẫn, sha256 đã kiểm). sha đổi => dựng lại.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()
_TRANSFORM: Any = None


def _build_model(source: Any, model_id: str, evidence: dict[str, Any]) -> Any:
    """`source`: đường dẫn hoặc file-like chứa ĐÚNG các byte đã băm (xem `_published_model`). Nhận dạng cần khớp lấy từ `evidence` (manifest của gói)."""
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    from src.common.checkpoint_io import load_checkpoint

    # Checkpoint đã được băm trước khi giải nén (xem `_published_model`); `load_checkpoint` giữ
    # allowlist numpy ở mức tối thiểu và nạp `weights_only=True` thay vì mở pickle tùy ý.
    checkpoint = load_checkpoint(source)
    # Checkpoint của bộ train v4 không ghi `w0_fingerprint` (chỉ release, seed, code_fingerprint): gói đó khai `w0_in_checkpoints: false` trong manifest.
    if (checkpoint.get("release_manifest_sha256") != evidence["release_manifest_sha256"]
            or (evidence["w0_in_checkpoints"] and checkpoint.get("w0_fingerprint") != evidence["w0_fingerprint"])):
        raise ValueError("Checkpoint protocol metadata mismatch")
    if model_id in {"centralized", "fedavg"} and checkpoint.get("seed") != evidence["seed"]:
        raise ValueError("Checkpoint seed mismatch")
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if not isinstance(state, dict) or len(state) != 244:
        raise ValueError("Checkpoint model state is missing or malformed")

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():  # chỉ suy luận/Grad-CAM: không cần gradient của trọng số
        parameter.requires_grad_(False)
    return model.eval()


def _published_model(evidence: dict[str, Any], model_id: str, root: Path) -> tuple[dict[str, Any], Any]:
    model_info = next((m for m in evidence["models"] if m["model_id"] == model_id), None)
    if model_info is None:
        raise ValueError(f"Unknown published model: {model_id}")
    path = root / model_info["checkpoint_relative_path"]
    key = (str(path.resolve()), model_info["checkpoint_sha256"])
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            # Đọc file MỘT lần, băm chính các byte đó rồi mới giải nén: không dựa vào cache hash theo mtime (thay file giữ
            # nguyên kích thước + mtime vẫn bị bắt) và không có khe hở giữa lúc kiểm và lúc đọc.
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != model_info["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after hash verification")
            model = _build_model(io.BytesIO(data), model_id, evidence)
            for stale in [k for k in _MODEL_CACHE if k[0] == key[0]]:
                del _MODEL_CACHE[stale]
            _MODEL_CACHE[key] = model
    return model_info, model


def _input_tensor(image: Any) -> Any:
    global _TRANSFORM
    from torchvision import transforms as T

    if _TRANSFORM is None:
        _TRANSFORM = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    from src.common.image_io import to_rgb

    return _TRANSFORM(to_rgb(image)).unsqueeze(0)


def _top3(probabilities: Any, classes: list[str], k: int = 3) -> list[dict[str, Any]]:
    values, indices = probabilities.topk(min(k, len(classes)))
    return [{"class": classes[int(i)], "confidence": round(float(v) * 100, 2)} for v, i in zip(values, indices)]


def crop_of_class(class_name: str) -> str:
    """Tên loại cây của một lớp: phần trước '___' (vd 'Tomato___Late_blight' -> 'Tomato')."""
    return class_name.split("___", 1)[0]


def crop_names(classes: list[str]) -> list[str]:
    return sorted({crop_of_class(c) for c in classes})


def predict_published(image: Any, model_id: str, root: Path = STORE, temperature: float | None = None,
                      crop: str | None = None) -> dict[str, Any]:
    """Load only a pinned checkpoint, using the published canonical_v1 transform.

    `temperature` (fit trên tập calibration riêng) chỉ thêm khối `calibrated`; dự đoán và `confidence_pct` giữ nguyên softmax thô.
    `crop` (người dùng chọn loại cây): chỉ xét các lớp của cây đó và chuẩn hóa lại xác suất trong nhóm; kết quả KHÔNG thô nữa nên
    được đánh dấu ở khối `crop_restricted`. Chọn sai cây thì kết quả sai, mô hình không kiểm lại lựa chọn đó."""
    evidence = load_evidence(root)
    if not evidence["available"]:
        raise ValueError(evidence["reason"])
    import torch

    model_info, model = _published_model(evidence, model_id, root)
    tensor = _input_tensor(image)
    with torch.inference_mode():
        logits = model(tensor)[0]
    probabilities = torch.softmax(logits, dim=0)
    classes = evidence["classes"]
    keep = None
    if crop is not None:
        if crop not in crop_names(classes):
            raise ValueError("Loại cây không thuộc taxonomy")
        keep = torch.tensor([crop_of_class(c) == crop for c in classes])
    unrestricted_top3 = _top3(probabilities, classes)

    # Phần xác suất gốc (38 lớp) mà mô hình đặt vào cây được chọn: nếu thấp, mô hình không thấy cây đó trong ảnh và xác suất chuẩn hóa
    # lại bên dưới là giả tạo (một nhóm chỉ có một lớp luôn ra 100%).
    crop_mass = float(probabilities[keep].sum()) if keep is not None else 1.0
    if keep is not None:
        # softmax chỉ trên các lớp của cây được chọn (ổn định số học kể cả khi khối xác suất gốc của cây đó gần bằng 0)
        probabilities = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=0)
    in_scope = int(keep.sum()) if keep is not None else len(classes)
    top3 = _top3(probabilities, classes, min(3, in_scope))
    top_values = probabilities.topk(min(2, in_scope)).values
    margin = float(top_values[0] - top_values[1]) if len(top_values) > 1 else float(top_values[0])
    result = {
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
        "is_uncertain": bool(float(top_values[0]) < 0.5 or margin < 0.2 or crop_mass < 0.5),
        "confidence_calibrated": False,
        "model_validation": model_info["metrics"],
        "checkpoint_sha256": model_info["checkpoint_sha256"],
        "evidence_level": evidence["evidence_level"],
        "scope": "single cropped leaf image; validation is not an external-photo accuracy estimate",
    }
    if keep is not None:
        result["crop_restricted"] = {
            "crop": crop,
            "source": "người dùng chọn",
            "unrestricted_top1": unrestricted_top3[0],
            "crop_mass_pct": round(crop_mass * 100, 2),
            "classes_in_crop": in_scope,
            "low_crop_mass": bool(crop_mass < 0.5),
            "note": "Chỉ xét bệnh của cây đã chọn; xác suất chuẩn hóa lại trong nhóm cây đó, không còn là softmax thô của 38 lớp.",
        }
    if temperature and temperature > 0 and keep is None:
        # Nhiệt độ T học trên phân bố 38 lớp: không áp cho xác suất đã giới hạn theo cây (sẽ là số không được kiểm chứng)
        scaled = torch.softmax(logits / float(temperature), dim=0)
        top3_scaled = _top3(scaled, classes)  # T > 0 không đổi thứ tự lớp
        result["calibrated"] = {
            "method": "temperature_scaling",
            "temperature": round(float(temperature), 4),
            "fitted_on": "calibration split (tách khỏi train và validation)",
            "confidence_pct": top3_scaled[0]["confidence"],
            "top3": top3_scaled,
        }
    return result


def explain_published(image: Any, model_id: str, class_name: str, root: Path = STORE) -> dict[str, Any]:
    """Grad-CAM của mô hình cho lớp `class_name` trên đúng ảnh đầu vào (224×224 canonical_v1)."""
    evidence = load_evidence(root)
    if not evidence["available"]:
        raise ValueError(evidence["reason"])
    if class_name not in evidence["classes"]:
        raise ValueError("Lớp không thuộc taxonomy 38 lớp")
    from web.services import explain_service

    _, model = _published_model(evidence, model_id, root)
    tensor = _input_tensor(image)
    cam = explain_service.gradcam(model, tensor, evidence["classes"].index(class_name))
    return {
        "class": class_name,
        "image": explain_service.overlay_data_url(image, cam),
        "method": "Grad-CAM (features[-1], 7x7, nội suy lên 224x224)",
    }

"""Sinh artifacts/registry.json (nguồn sự thật máy đọc) + tái sinh canonical_manifest.json.

Quy tắc:
- Registry ĐỌC từ đĩa và các metrics gốc; không bao giờ ghi theo tài liệu narrativE.
- Mọi checkpoint được hash (file SHA-256 + state-dict SHA-256 theo cùng scheme
  với scripts/validate_artifact_provenance.py để hai bên so sánh được).
- canonical_manifest.json được tái sinh từ nguồn thật (metrics JSON + test_metrics.json
  của run Kaggle + resolved_config.yaml), kèm trạng thái checkpoint trung thực.

Chạy: python scripts/build_registry.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402
from scripts.import_branch_demo import classes_from_taxonomy  # noqa: E402
from src.common.checkpoint_io import load_checkpoint  # noqa: E402

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(rel_path: str) -> dict[str, Any] | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_dict_from_checkpoint(payload: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("model_state", "model_state_dict", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict) and state:
            return state
    return None


def stable_state_hash(state: dict[str, torch.Tensor]) -> str:
    """Cùng scheme với scripts/validate_artifact_provenance.py."""
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def inspect_checkpoint(rel_path: str, expected_sha256: str | None = None) -> dict[str, Any]:
    """Băm file; chỉ nạp khi không khai báo hash mong đợi hoặc hash khớp.

    Checkpoint của bundle công bố mà hash lệch manifest KHÔNG được nạp. Kể cả khi hash khớp vẫn
    nạp qua `load_checkpoint` (`weights_only=True`) để một file bị thay trên đĩa không thực thi mã."""
    path = ROOT / rel_path
    out: dict[str, Any] = {
        "checkpoint_path": rel_path,
        "checkpoint_exists": path.exists(),
        "file_sha256": None,
        "state_sha256": None,
        "bytes": None,
        "metadata": {},
    }
    if not path.exists():
        return out
    out["file_sha256"] = file_sha256(path)
    out["bytes"] = path.stat().st_size
    if expected_sha256 is not None and out["file_sha256"] != expected_sha256:
        out["metadata"] = {"load_skipped": "hash khác manifest — không nạp checkpoint chưa tin cậy"}
        return out
    try:
        payload = load_checkpoint(path)
    except Exception as exc:  # TorchScript / format lạ
        out["metadata"] = {"load_error": f"{type(exc).__name__}: {exc}"}
        return out
    if isinstance(payload, dict):
        state = _state_dict_from_checkpoint(payload)
        if state is not None:
            try:
                out["state_sha256"] = stable_state_hash(state)
            except Exception:
                pass
        for key, value in payload.items():
            if key in {"model_state", "model_state_dict", "state_dict"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                out["metadata"][key] = value
    return out


def classes_sha256(classes: list[str]) -> str:
    """Hash thứ tự 38 lớp — hai model chỉ so sánh được khi cùng hash này."""
    return hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()


def base_model(model_id: str, family: str, display_name: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "family": family,
        "display_name": display_name,
        "pipeline": None,
        "role": None,
        "architecture": "MobileNetV3-Small",
        "num_classes": 38,
        "checkpoint_path": None,
        "checkpoint_exists": False,
        "file_sha256": None,
        "state_sha256": None,
        "expected_state_sha256": None,
        "checkpoint_status": None,
        "preprocessing_id": None,
        "dataset_id": None,
        "protocol_id": None,
        "validation_manifest_sha256": None,
        "taxonomy_sha256": None,
        "leakage_audit": "unknown",
        "leakage_audit_scope": None,
        "seed": None,
        "metrics": {},
        "metrics_source": None,
        "metrics_waiver": None,
        "status": None,
        "inference": None,
        "inference_test": None,
        "web_visible": False,
        "web_gate_failures": [],
        "display_label": None,
        "notes": None,
    }


# ---------------------------------------------------------------------------
# dataset_truth
# ---------------------------------------------------------------------------
def build_dataset_truth() -> dict[str, Any]:
    suite = load_json(reg.SUITE_JSON) or {}
    conditions = {}
    for name, info in (suite.get("conditions") or {}).items():
        conditions[name] = {
            "scenario": info.get("scenario"),
            "alpha": info.get("alpha"),
            "quantity_alpha": info.get("quantity_alpha"),
            "n_k": info.get("n_k"),
        }
    counts = suite.get("counts") or {}
    return {
        "source": reg.SUITE_JSON,
        "protocol": suite.get("protocol"),
        "total_images": suite.get("source_images"),
        "split": {"train": counts.get("train"), "val": counts.get("val"), "test": counts.get("test")},
        "num_clients": suite.get("num_clients"),
        "split_seed": suite.get("split_seed"),
        "smoke": suite.get("smoke"),
        "conditions": conditions,
        "note": (
            "Đây là split reproduction cục bộ (5 clients). Run canonical Kaggle dùng partition v1 "
            "(10 clients, label_skew alpha=0.1, test 10,917 ảnh) — hai split KHÔNG trộn lẫn."
        ),
    }


# ---------------------------------------------------------------------------
# Họ scientific_canonical (metrics của run Kaggle; checkpoint không còn trên máy)
# ---------------------------------------------------------------------------
def _canonical_eval_note() -> str:
    test_metrics = load_json(reg.CANONICAL_TEST_METRICS_JSON) or {}
    return (
        f"Đánh giá trên Kaggle partition v1 label_skew alpha=0.1, global_test "
        f"{test_metrics.get('total_samples', '?')} ảnh (test_metrics.json của run). "
        "Checkpoint thật không còn trên máy này (best.pt là LFS pointer chưa pull; "
        f"{reg.FEDAVG_BEST_PT} từng bị smoke ghi đè acc=1.0)."
    )


def build_scientific_canonical() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    test_metrics = load_json(reg.CANONICAL_TEST_METRICS_JSON) or {}
    fed_metrics = load_json(reg.FEDAVG_RESULTS_JSON) or {}
    central_metrics = load_json(reg.CENTRALIZED_RESULTS_JSON) or {}
    local_metrics = load_json(reg.LOCAL_ONLY_RESULTS_JSON) or {}

    expected_fed_state = test_metrics.get("model_state_sha256")

    # --- fedavg_canonical ---
    m = base_model("fedavg_canonical", reg.FAMILY_SCIENTIFIC_CANONICAL, "FedAvg — scientific canonical")
    insp = inspect_checkpoint(reg.FEDAVG_BEST_PT)
    m.update(
        checkpoint_path=reg.FEDAVG_BEST_PT,
        checkpoint_exists=insp["checkpoint_exists"],
        file_sha256=insp["file_sha256"],
        state_sha256=insp["state_sha256"],
        expected_state_sha256=expected_fed_state,
        preprocessing_id="imagenet_eval_v1",
        protocol_id=f"kaggle_pv_v1_label_skew_a0.1/run:{reg.CANONICAL_RUN_ID}",
        seed=fed_metrics.get("seed", 42),
        metrics={
            "test_accuracy": (fed_metrics.get("test_metrics") or {}).get("accuracy"),
            "macro_f1": (fed_metrics.get("test_metrics") or {}).get("macro_f1"),
            "test_loss": (fed_metrics.get("test_metrics") or {}).get("loss"),
            "total_samples": (fed_metrics.get("test_metrics") or {}).get("total_samples"),
            "total_rounds": fed_metrics.get("total_rounds"),
            "best_round": fed_metrics.get("best_round"),
        },
        metrics_source=f"{reg.FEDAVG_RESULTS_JSON} + {reg.CANONICAL_TEST_METRICS_JSON}",
        status="unverified_metrics_only",
        display_label="Scientific — metrics-only",
        notes=_canonical_eval_note(),
    )
    if insp["checkpoint_exists"]:
        meta_acc = insp["metadata"].get("test_accuracy")
        if expected_fed_state and insp["state_sha256"] == expected_fed_state:
            m["checkpoint_status"] = "matches_manifest"
            m["status"] = "verified_checkpoint"
        elif meta_acc == 1.0 or meta_acc == 0.0263:
            m["checkpoint_status"] = "smoke_overwrite_detected"
            m["notes"] = (m["notes"] or "") + f" HIỆN TẠI slot này đang chứa checkpoint smoke — cần cách ly vào {reg.SMOKE_CHECKPOINTS_DIR}/."
        else:
            m["checkpoint_status"] = "unknown_mismatch"
    else:
        m["checkpoint_status"] = "missing_on_this_machine"
    models.append(m)

    # --- centralized_canonical ---
    m = base_model("centralized_canonical", reg.FAMILY_SCIENTIFIC_CANONICAL, "Centralized — scientific canonical")
    insp = inspect_checkpoint(reg.CENTRALIZED_BEST_PT)
    tm = central_metrics.get("test_metrics") or central_metrics.get("test") or {}
    m.update(
        checkpoint_path=reg.CENTRALIZED_BEST_PT,
        checkpoint_exists=insp["checkpoint_exists"],
        file_sha256=insp["file_sha256"],
        state_sha256=insp["state_sha256"],
        preprocessing_id="imagenet_eval_v1",
        protocol_id=f"kaggle_pv_v1_label_skew_a0.1/centralized",
        metrics={
            "test_accuracy": tm.get("accuracy") or tm.get("test_accuracy"),
            "macro_f1": tm.get("macro_f1"),
            "test_loss": tm.get("loss"),
            "epochs": tm.get("epochs") or central_metrics.get("epochs"),
        },
        metrics_source=reg.CENTRALIZED_RESULTS_JSON,
        status="historical_unverified",
        display_label="Scientific — historical (chưa tái kiểm chứng)",
        notes=(
            "Số liệu 98.25% ghi nhận từ metrics JSON; checkpoint thật không có trên máy "
            "(slot từng chứa bản smoke 2 epochs). Chỉ trích dẫn kèm nhãn 'historical'."
        ),
    )
    if insp["checkpoint_exists"]:
        meta_acc = insp["metadata"].get("test_accuracy")
        tm_acc = (insp["metadata"].get("test_metrics") or {}).get("accuracy")
        if meta_acc == 1.0 or tm_acc == 1.0 or insp["metadata"].get("epochs") == 2:
            m["checkpoint_status"] = "smoke_overwrite_detected"
        else:
            m["checkpoint_status"] = "unknown_mismatch"
    else:
        m["checkpoint_status"] = "missing_on_this_machine"
    models.append(m)

    # --- local_only_canonical ---
    m = base_model("local_only_canonical", reg.FAMILY_SCIENTIFIC_CANONICAL, "Local-only — scientific canonical")
    per_client = local_metrics.get("per_client") or local_metrics.get("clients") or []
    m.update(
        protocol_id="kaggle_pv_v1_label_skew_a0.1/local-only",
        metrics={
            "mean_accuracy": local_metrics.get("mean_accuracy"),
            "std_accuracy": local_metrics.get("std_accuracy", local_metrics.get("client_accuracy_std")),
            "mean_macro_f1": local_metrics.get("mean_macro_f1"),
            "num_clients": local_metrics.get("num_clients"),
            "per_client_accuracy": (
                [c.get("accuracy") for c in per_client] if isinstance(per_client, list) else per_client
            ),
        },
        metrics_source=reg.LOCAL_ONLY_RESULTS_JSON,
        status="historical_unverified",
        display_label="Scientific — historical (chưa tái kiểm chứng)",
        notes="Baseline 5 client cô lập; không có checkpoint trên máy.",
    )
    m["checkpoint_status"] = "not_applicable"
    models.append(m)
    return models


# ---------------------------------------------------------------------------
# Họ published_demo (branch_demo hash-verified)
# ---------------------------------------------------------------------------
def _find_in_artifact_manifest(manifest: dict[str, Any] | None, suffix: str) -> dict[str, Any] | None:
    if not manifest:
        return None
    files = manifest.get("files") or {}
    for name, info in files.items():
        if name.endswith(suffix) or name == suffix:
            return info
    return None


def _demo_metrics(file_data: dict[str, Any] | None) -> dict[str, Any]:
    if not file_data:
        return {}
    validation = file_data.get("validation") if isinstance(file_data.get("validation"), dict) else file_data
    overall = validation.get("overall") or {}
    by_source = validation.get("by_source") or {}

    def src(name: str) -> dict[str, Any]:
        s = by_source.get(name) or {}
        return {
            "count": s.get("count"),
            "top1_accuracy": s.get("top1_accuracy"),
            "macro_f1_supported": s.get("macro_f1_supported"),
            "crop_accuracy": s.get("crop_accuracy"),
        }

    quality_by_source = (validation.get("quality") or {}).get("by_source") or {}
    return {
        "overall": {
            "count": overall.get("count"),
            "top1_accuracy": overall.get("top1_accuracy"),
            "macro_f1_supported": overall.get("macro_f1_supported"),
            "crop_accuracy": overall.get("crop_accuracy"),
            "high_confidence_wrong_count_0_9": overall.get("high_confidence_wrong_count_0_9"),
        },
        "plantdoc": src("plantdoc"),
        "plantvillage": src("plantvillage"),
        "quality": {
            "high_confidence_threshold": (validation.get("quality") or {}).get("high_confidence_threshold"),
            "by_source": quality_by_source,
            "top_confusions": (validation.get("top_confusions") or [])[:5],
        },
    }


def build_published_demo() -> list[dict[str, Any]]:
    import_manifest = load_json(reg.DEMO_IMPORT_MANIFEST_JSON)
    if import_manifest is None:
        return []
    protocol_id = import_manifest.get("protocol_id", reg.DEMO_PROTOCOL_ID)
    seed = import_manifest.get("seed")
    bundle_taxonomy = classes_sha256(import_manifest.get("classes") or [])
    validation_manifest = import_manifest.get("validation_manifest_sha256")

    specs = [
        ("demo_centralized_v6", "Centralized v6 (published demo)", "centralized", "centralized"),
        ("demo_fedavg_v7", "FedAvg v7 (published demo)", "fedavg", "fedavg"),
    ] + [
        (f"demo_local_{i:02d}", f"Local-only client {i:02d} (published demo)", f"local/client_{i:02d}", "local")
        for i in range(5)
    ]

    models: list[dict[str, Any]] = []
    for model_id, display_name, rel_dir, branch_dir in specs:
        m = base_model(model_id, reg.FAMILY_PUBLISHED_DEMO, display_name)
        ckpt_rel = f"{reg.DEMO_BUNDLE_DIR}/{rel_dir}/checkpoint_best.pt"
        art_manifest = load_json(f"{reg.DEMO_BUNDLE_DIR}/{branch_dir}/artifact_manifest.json")
        if branch_dir == "local":
            art_manifest = load_json(f"{reg.DEMO_BUNDLE_DIR}/local/artifact_manifest.json")
        expected = _find_in_artifact_manifest(art_manifest, f"{rel_dir.split('/')[-1]}/checkpoint_best.pt") or \
            _find_in_artifact_manifest(art_manifest, "checkpoint_best.pt")
        insp = inspect_checkpoint(ckpt_rel, expected_sha256=(expected or {}).get("sha256"))
        validation = load_json(f"{reg.DEMO_BUNDLE_DIR}/{rel_dir}/validation_metrics.json")
        m.update(
            checkpoint_path=ckpt_rel,
            checkpoint_exists=insp["checkpoint_exists"],
            file_sha256=insp["file_sha256"],
            state_sha256=insp["state_sha256"],
            expected_state_sha256=None,
            checkpoint_status=(
                "hash_verified"
                if insp["file_sha256"] and expected and insp["file_sha256"] == expected.get("sha256")
                else ("missing" if not insp["checkpoint_exists"] else "hash_mismatch")
            ),
            preprocessing_id=import_manifest.get("transform", "canonical_v1"),
            inference={
                "backend": reg.BACKEND_PUBLISHED_BUNDLE,
                "key": rel_dir.replace("local/client_", "local_"),
            },
            protocol_id=protocol_id,
            validation_manifest_sha256=validation_manifest,
            taxonomy_sha256=bundle_taxonomy,
            seed=seed,
            metrics=_demo_metrics(validation),
            metrics_source=f"{reg.DEMO_BUNDLE_DIR}/{rel_dir}/validation_metrics.json",
            status="verified_published_artifact",
            web_visible=insp["checkpoint_exists"],
            display_label="Demo (hash-verified, pretrained mixed PV+PlantDoc)",
            notes=(
                "Artifact publish từ branch train, hash khớp artifact_manifest khi import. "
                "Chưa đánh giá lại cục bộ (evidence_level="
                f"{import_manifest.get('evidence_level')})."
            ),
        )
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Họ network_demo / reference / smoke
# ---------------------------------------------------------------------------
def build_network_and_reference() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []

    net_specs = [
        ("network_federated_global", "Flower network global (smoke)", reg.FEDERATED_NETWORK_GLOBAL_PT,
         "Checkpoint global từ smoke network localhost (5 clients, 2 rounds, synthetic)."),
        ("network_attack_matrix_observe", "Attack matrix — observe", reg.ATTACK_MATRIX_OBSERVE_PT,
         "Thí nghiệm fault modes, policy observe-only."),
        ("network_attack_matrix_reject", "Attack matrix — reject", reg.ATTACK_MATRIX_REJECT_PT,
         "Thí nghiệm fault modes, policy reject-suspicious."),
        ("network_rtt_checkpoint_test", "RTT + checkpoint test", reg.NETWORK_RTT_CHECKPOINT_PT,
         "Bài test RTT per-client + lưu/nạp checkpoint qua mạng."),
    ]
    for model_id, display_name, rel_path, note in net_specs:
        m = base_model(model_id, reg.FAMILY_NETWORK_DEMO, display_name)
        insp = inspect_checkpoint(rel_path)
        m.update(
            checkpoint_path=rel_path,
            checkpoint_exists=insp["checkpoint_exists"],
            file_sha256=insp["file_sha256"],
            state_sha256=insp["state_sha256"],
            checkpoint_status="present" if insp["checkpoint_exists"] else "missing",
            protocol_id="network_smoke_localhost",
            metrics={k: v for k, v in insp["metadata"].items() if k in ("round", "rounds", "num_clients", "source")},
            status="smoke_demo_only",
            display_label="Network demo (không khoa học)",
            notes=note + " CẤM dùng accuracy từ các checkpoint này trong luận văn.",
        )
        models.append(m)

    m = base_model("plantvillage_traced_reference", reg.FAMILY_REFERENCE, "PlantVillage TorchScript (tham chiếu)")
    insp = inspect_checkpoint(reg.PLANTVILLAGE_TRACED_PT)
    m.update(
        architecture="MobileNetV2 (Hugging Face transformers)",
        checkpoint_path=reg.PLANTVILLAGE_TRACED_PT,
        checkpoint_exists=insp["checkpoint_exists"],
        file_sha256=insp["file_sha256"],
        expected_state_sha256=None,
        checkpoint_status=(
            "missing" if not insp["checkpoint_exists"]
            else "hash_verified" if insp["file_sha256"] == reg.PLANTVILLAGE_TRACED_SHA256
            else "hash_mismatch"
        ),
        preprocessing_id="hf_half_256_224",
        protocol_id="plantvillage_traced_reference",
        inference={"backend": reg.BACKEND_TORCHSCRIPT_REFERENCE},
        metrics_waiver=(
            "Model ngoài (Hugging Face), không do đồ án huấn luyện; số đo nằm ở artifacts/plantvillage_reference_eval.json "
            "(báo cáo đánh giá lại riêng, không thuộc bundle bảy mô hình đã kiểm hash); web gắn nhãn 'Tham chiếu'."
        ),
        status="reference_only",
        display_label="Tham chiếu (không phải FedAvg)",
        notes=("TorchScript của mô hình Hugging Face linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (MobileNetV2, "
               "học trên bộ Kaggle New Plant Diseases, bản tăng cường của PlantVillage); đồ án KHÔNG huấn luyện mà đánh giá lại: "
               "xem artifacts/plantvillage_reference_eval.json. Softmax thô."),
    )
    models.append(m)
    return models


def build_smoke() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    smoke_dir = ROOT / reg.SMOKE_CHECKPOINTS_DIR
    if not smoke_dir.exists():
        return models
    for pt in sorted(smoke_dir.glob("*.pt")):
        rel = pt.relative_to(ROOT).as_posix()
        m = base_model(pt.stem, reg.FAMILY_SMOKE, f"Smoke — {pt.stem}")
        insp = inspect_checkpoint(rel)
        m.update(
            checkpoint_path=rel,
            checkpoint_exists=insp["checkpoint_exists"],
            file_sha256=insp["file_sha256"],
            state_sha256=insp["state_sha256"],
            checkpoint_status="quarantined",
            metrics={k: insp["metadata"].get(k) for k in ("test_accuracy", "last_accuracy")},
            status="smoke_quarantined",
            display_label="Smoke (cấm luận văn)",
            notes="Được cách ly khỏi canonical slot sau đợt đồng bộ 2026-10-09.",
        )
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def build_gates() -> dict[str, Any]:
    gates: dict[str, Any] = {}
    provenance = load_json(reg.PROVENANCE_VALIDATION_JSON)
    if provenance is not None:
        gates["provenance"] = {
            "generated_at": provenance.get("generated_at"),
            "status": provenance.get("status"),
            "critical_issues": [i.get("code") for i in provenance.get("issues", []) if i.get("severity") == "critical"],
            "warning_issues": [i.get("code") for i in provenance.get("issues", []) if i.get("severity") == "warning"],
            "source": reg.PROVENANCE_VALIDATION_JSON,
        }
    pre_demo = load_json(reg.PRE_DEMO_CHECK_JSON)
    if pre_demo is not None:
        gates["pre_demo_check"] = {
            "generated_at": pre_demo.get("generated_at"),
            "overall": pre_demo.get("overall"),
            "source": reg.PRE_DEMO_CHECK_JSON,
        }
    final = load_json(reg.FINAL_ACCEPTANCE_JSON)
    if final is not None:
        gates["final_acceptance"] = {
            "generated_at": final.get("generated_at"),
            "code_fingerprint": final.get("code_fingerprint"),
            "technical": final.get("technical_acceptance"),
            "scientific": final.get("scientific_acceptance"),
            "readiness_overall": final.get("readiness_overall"),
            "scientific_blockers": final.get("scientific_blockers"),
            "source": reg.FINAL_ACCEPTANCE_JSON,
        }
    network = load_json(reg.NETWORK_ARTIFACT_VALIDATION_JSON)
    if network is not None:
        gates["network_artifact"] = {
            "generated_at": network.get("generated_at"),
            "ok": network.get("ok"),
            "require_real_clients": (network.get("checks") or {}).get("require_real_clients"),
            "source": reg.NETWORK_ARTIFACT_VALIDATION_JSON,
        }
    return gates


# ---------------------------------------------------------------------------
# canonical_manifest.json (tái sinh trung thực)
# ---------------------------------------------------------------------------
def build_canonical_manifest(models: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {m["model_id"]: m for m in models}
    fed = by_id["fedavg_canonical"]
    central = by_id["centralized_canonical"]
    local = by_id["local_only_canonical"]
    fed_m = fed["metrics"]
    c_m = central["metrics"]
    l_m = local["metrics"]
    return {
        "schema_version": 2,
        "canonical_run_id": reg.CANONICAL_RUN_ID,
        "dataset": {
            "name": "PlantVillage",
            "classes": 38,
            "total_images": 54305,
            "partition_protocol": (
                "Kaggle partitions_train_v1 — label_skew Dirichlet alpha=0.1, 10 clients "
                "(theo resolved_config.yaml của run)"
            ),
            "evaluation_dataset": f"global_test ({fed_m.get('total_samples')} ảnh, Kaggle)",
            "local_reproduction_suite": {
                "path": reg.PARTITIONS_DIR,
                "split": "38,984/4,400/10,921 (5 clients, seed 42)",
                "note": "Split cục bộ để tái chạy; CHƯA có run nào trên máy hiện tại.",
            },
        },
        "benchmarks": {
            "centralized": {
                "name": "Centralized Baseline (Upper Bound)",
                "architecture": "MobileNetV3-Small",
                "training_scope": "Full centralized training",
                "evaluation_dataset": f"global_test ({fed_m.get('total_samples')} ảnh)",
                "test_accuracy": c_m.get("test_accuracy"),
                "test_accuracy_pct": f"{(c_m.get('test_accuracy') or 0) * 100:.2f}%",
                "macro_f1": c_m.get("macro_f1"),
                "test_loss": c_m.get("test_loss"),
                "checkpoint": reg.CENTRALIZED_BEST_PT,
                "checkpoint_status": central["checkpoint_status"],
                "evidence_status": "historical_unverified",
            },
            "fedavg": {
                "name": "Federated Learning FedAvg (Global Model)",
                "run_id": reg.CANONICAL_RUN_ID,
                "architecture": "MobileNetV3-Small",
                "total_rounds": fed_m.get("total_rounds"),
                "best_round": fed_m.get("best_round"),
                "evaluation_dataset": f"global_test ({fed_m.get('total_samples')} ảnh)",
                "test_accuracy": fed_m.get("test_accuracy"),
                "test_accuracy_pct": f"{(fed_m.get('test_accuracy') or 0) * 100:.4f}%",
                "macro_f1": fed_m.get("macro_f1"),
                "test_loss": fed_m.get("test_loss"),
                "checkpoint": reg.FEDAVG_BEST_PT,
                "state_dict_sha256": fed["expected_state_sha256"],
                "checkpoint_status": fed["checkpoint_status"],
                "evidence_status": "unverified_metrics_only",
                "scientific_stage2_complete": False,
            },
            "local_only": {
                "name": "Local-only Baseline (Isolated Nodes)",
                "architecture": "MobileNetV3-Small",
                "evaluation_dataset": f"global_test ({fed_m.get('total_samples')} ảnh)",
                "mean_accuracy": l_m.get("mean_accuracy"),
                "mean_accuracy_pct": f"{(l_m.get('mean_accuracy') or 0) * 100:.2f}%",
                "macro_f1": l_m.get("mean_macro_f1"),
                "client_accuracy_std": l_m.get("std_accuracy"),
                "evidence_status": "historical_unverified",
                "notes": "Baseline 5 client cô lập; không có checkpoint trên máy.",
            },
        },
        "safe_reporting_policy": (
            "Chỉ trích dẫn accuracy/loss/F1 kèm nhãn trạng thái (verified / historical / "
            "metrics-only) và nguồn metrics. Không so sánh với họ published_demo (pretrained "
            "mixed PlantVillage+PlantDoc)."
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
FAMILY_DEFAULTS: dict[str, dict[str, str]] = {
    reg.FAMILY_SCIENTIFIC_CANONICAL: {
        "pipeline": reg.PIPELINE_LEGACY,
        "role": reg.ROLE_HISTORICAL,
        "dataset_id": "PlantVillage 54.305 ảnh — Kaggle partitions_train_v1 (10 clients, label_skew alpha=0.1)",
    },
    reg.FAMILY_PUBLISHED_DEMO: {
        "pipeline": reg.PIPELINE_EXTERNAL,
        "role": reg.ROLE_CURRENT,
        "dataset_id": "Mixed PlantVillage + PlantDoc (validation: 4.344 PV + 269 PlantDoc)",
    },
    reg.FAMILY_NETWORK_DEMO: {
        "pipeline": reg.PIPELINE_SRC,
        "role": reg.ROLE_TECHNICAL_DEMO,
        "dataset_id": "Ảnh nhiễu ngẫu nhiên (smoke Flower localhost, kiểm chứng hệ thống)",
    },
    reg.FAMILY_REFERENCE: {
        "pipeline": reg.PIPELINE_EXTERNAL,
        "role": reg.ROLE_REFERENCE,
        "dataset_id": "Kaggle New Plant Diseases (bản tăng cường của PlantVillage, 38 lớp); mô hình ngoài, đồ án chỉ đánh giá lại",
    },
    reg.FAMILY_SMOKE: {
        "pipeline": reg.PIPELINE_SRC,
        "role": reg.ROLE_QUARANTINED,
        "dataset_id": "Dữ liệu dummy (smoke)",
    },
}

def apply_family_defaults(models: list[dict[str, Any]]) -> None:
    repo_classes = (load_json(reg.DATASET_MANIFEST_JSON) or {}).get("classes") or []
    repo_taxonomy = classes_sha256(repo_classes) if repo_classes else None
    for m in models:
        for key, value in FAMILY_DEFAULTS[m["family"]].items():
            if m.get(key) is None:
                m[key] = value
        if m.get("taxonomy_sha256") is None:
            m["taxonomy_sha256"] = repo_taxonomy


# ---------------------------------------------------------------------------
# Bài kiểm tra forward bắt buộc + cổng web_visible (5 điều kiện)
# ---------------------------------------------------------------------------
def run_forward_test(model: dict[str, Any]) -> dict[str, Any]:
    """Chạy một ảnh mẫu qua đúng đường inference mà web dùng. Không bao giờ ném lỗi."""
    sample = ROOT / reg.SAMPLE_IMAGE
    result: dict[str, Any] = {"status": "not_run", "sample": reg.SAMPLE_IMAGE}
    if not sample.exists():
        result["detail"] = "thiếu ảnh mẫu"
        result["status"] = "fail"
        return result
    try:
        from PIL import Image

        with Image.open(sample) as img:
            image = img.convert("RGB")
        from web.services import model_service

        out = model_service.run_backend(model, image)
        confidences = [float(t["confidence"]) for t in out.get("top3", [])]
        classes = (load_json(reg.DATASET_MANIFEST_JSON) or {}).get("classes", [])
        ok = (
            out.get("ready") is True
            and len(confidences) == 3
            and all(0.0 <= c <= 100.0 for c in confidences)
            and sum(confidences) <= 100.01
            and out.get("predicted_class") in classes
        )
        result.update(
            status="pass" if ok else "fail",
            top1=out.get("predicted_class"),
            top1_confidence_pct=out.get("confidence_pct"),
        )
    except Exception as exc:  # noqa: BLE001 — gate phải báo lỗi, không được làm sập build
        result.update(status="fail", detail=f"{type(exc).__name__}: {exc}"[:300])
    return result


def apply_leakage_audit(models: list[dict[str, Any]]) -> None:
    """leakage_audit=passed CHỈ cho model của bundle khi: audit release đạt, val hash khớp, và bài forward xác nhận checkpoint
    ràng buộc đúng release (predict_published kiểm release_manifest_sha256/W0 trong metadata checkpoint)."""
    audit = load_json(reg.RELEASE_LEAKAGE_AUDIT_JSON)
    for model in models:
        model["leakage_audit_scope"] = None
        if not audit or audit.get("passed") is not True:
            continue
        bound = (
            model.get("pipeline") == reg.PIPELINE_EXTERNAL
            and model.get("family") == reg.FAMILY_PUBLISHED_DEMO
            and model.get("checkpoint_status") == "hash_verified"
            and model.get("validation_manifest_sha256") == audit.get("validation_manifest_sha256")
            and (model.get("inference_test") or {}).get("status") == "pass"
        )
        if bound:
            model["leakage_audit"] = "passed"
            model["leakage_audit_scope"] = (
                f"split của release {audit.get('release_name')} (train/val/calibration rời nhau theo group, scene, byte, sample); "
                "checkpoint ràng buộc release qua metadata, chưa kiểm lại từng bước train"
            )


def apply_web_gate(models: list[dict[str, Any]], *, forward: bool = True) -> None:
    """web_visible = True chỉ khi đạt đủ 5 điều kiện của review. Lý do rớt ghi vào web_gate_failures."""
    # Nguồn ĐỘC LẬP với cả dataset_manifest.json lẫn import_manifest.json: chính file hợp đồng taxonomy.py đi kèm bundle
    # (importer đã ghim hash file này). Không đọc được => fail-closed.
    try:
        bundle_classes = classes_from_taxonomy(ROOT / reg.DEMO_BUNDLE_DIR / "contract" / "taxonomy.py")
    except (OSError, ValueError, SyntaxError):
        bundle_classes = []
    expected_taxonomy = classes_sha256(bundle_classes) if bundle_classes else None

    for m in models:
        failures: list[str] = []
        if m["role"] not in (reg.ROLE_CURRENT, reg.ROLE_REFERENCE):
            m["web_visible"] = False
            m["web_gate_failures"] = [f"role={m['role']} không thuộc diện hiển thị"]
            continue

        # 1. checkpoint + hash hợp lệ
        if m.get("checkpoint_status") != "hash_verified":
            failures.append(f"1.hash: checkpoint_status={m.get('checkpoint_status')}")
        # 2. class mapping đúng thứ tự taxonomy
        if not expected_taxonomy or m.get("taxonomy_sha256") != expected_taxonomy:
            failures.append("2.taxonomy: thứ tự 38 lớp không khớp contract/taxonomy.py của bundle (nguồn độc lập)")
        # 3. preprocessing khớp phiên bản model
        if m.get("preprocessing_id") not in reg.KNOWN_PREPROCESSING:
            failures.append(f"3.preprocessing: {m.get('preprocessing_id')!r} chưa được khai báo")
        # 4. metrics gắn được với dataset + protocol
        has_metrics = bool(m.get("metrics_source")) and (ROOT / str(m["metrics_source"])).exists()
        waived = bool(m.get("metrics_waiver")) and m["role"] == reg.ROLE_REFERENCE
        if not (has_metrics or waived) or not m.get("protocol_id") or not m.get("dataset_id"):
            failures.append("4.metrics: thiếu metrics_source/protocol_id/dataset_id")
        # 5. bài kiểm tra forward bắt buộc (chỉ chạy khi 1-4 đã đạt)
        if not failures and forward:
            m["inference_test"] = run_forward_test(m)
            if m["inference_test"]["status"] != "pass":
                detail = m["inference_test"].get("detail") or m["inference_test"]["status"]
                failures.append(f"5.inference: {detail}")
        elif failures:
            m["inference_test"] = {"status": "not_run", "detail": "chưa đạt điều kiện 1-4"}
        else:
            m["inference_test"] = {"status": "not_run", "detail": "bỏ qua bằng --skip-forward"}
            failures.append("5.inference: chưa chạy bài forward (--skip-forward)")

        m["web_gate_failures"] = failures
        m["web_visible"] = not failures


# ---------------------------------------------------------------------------
# Danh mục .pt trên đĩa — không file nào được "vô danh"
# ---------------------------------------------------------------------------
def _excluded_dir(name: str, at_root: bool) -> bool:
    if at_root and name in reg.CHECKPOINT_SCAN_EXCLUDE_ROOT:
        return True
    return any(name == ex or (ex.endswith(("-", "_")) and name.startswith(ex)) for ex in reg.CHECKPOINT_SCAN_EXCLUDE_ANY)


def _walk_pt_files() -> list[str]:
    import os

    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        at_root = Path(dirpath) == ROOT
        dirnames[:] = sorted(d for d in dirnames if not _excluded_dir(d, at_root))
        for name in filenames:
            if name.endswith(".pt"):
                found.append((Path(dirpath) / name).relative_to(ROOT).as_posix())
    return sorted(found)


def _lfs_pointer_oid(path: Path) -> str | None:
    try:
        if path.stat().st_size > 1024:
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("version https://git-lfs"):
        return None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            return line.split(":", 1)[1].strip()
    return None


def build_artifact_catalog(models: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phân loại mọi *.pt trên đĩa vào model đã đăng ký hoặc một nhóm khai báo; còn lại = unclassified."""
    in_models = {m["checkpoint_path"] for m in models if m.get("checkpoint_path")}
    groups: dict[str, dict[str, Any]] = {}
    unclassified: list[str] = []
    all_pt = _walk_pt_files()

    def add(group_id: str, role: str, description: str, rel: str, oid: str | None = None) -> None:
        g = groups.setdefault(group_id, {"group_id": group_id, "role": role, "description": description, "paths": []})
        g["paths"].append(rel)
        if oid:
            g.setdefault("lfs_oids", {})[rel] = oid

    smoke_runs: dict[str, bool] = {}
    for rel in all_pt:
        if rel in in_models:
            continue
        path = ROOT / rel
        if rel.startswith(reg.REPORT_EXPERIMENTS_GLOB + "/"):
            run_dir = "/".join(rel.split("/")[:3])
            if run_dir not in smoke_runs:
                plan = load_json(f"{run_dir}/plan.json") or {}
                smoke_runs[run_dir] = plan.get("mode") == "smoke"
            if smoke_runs[run_dir]:
                add("report_experiments_smoke", reg.ROLE_QUARANTINED,
                    "Run report_experiments mode=smoke (ảnh nhiễu ngẫu nhiên, accuracy 0.0263) — cấm dùng trong luận văn", rel)
            else:
                add("report_experiments_real", reg.ROLE_TECHNICAL_DEMO,
                    "Run report_experiments mode=real CHƯA promote — chỉ là bằng chứng sau khi qua scripts/promote_verified_run.py", rel)
        elif rel.startswith(reg.VERIFIED_DIR + "/") or rel.startswith(reg.VERIFIED_NETWORK_DIR + "/"):
            add("verified_promoted", reg.ROLE_CURRENT,
                "Artifact đã promote bởi promote_verified_run.py / promote_verified_network_run.py sau khi qua gate", rel)
        elif rel.startswith(reg.WEB_RUNS_GLOB + "/"):
            add("web_runs", reg.ROLE_TECHNICAL_DEMO,
                "Checkpoint từ run do web khởi động (smoke Flower hoặc demo huấn luyện); chưa có bằng chứng khoa học", rel)
        elif rel.startswith(reg.KAGGLE_RESULTS_DIR + "/"):
            oid = _lfs_pointer_oid(path)
            if oid:
                add("kaggle_run_lfs_pointer", reg.ROLE_HISTORICAL,
                    "Checkpoint run Kaggle canonical: chỉ còn con trỏ Git-LFS, bytes thật KHÔNG có trên máy "
                    "(repo không có remote) — cần tải lại từ Kaggle", rel, oid)
            else:
                add("kaggle_run_checkpoint_real", reg.ROLE_HISTORICAL,
                    "Checkpoint run Kaggle đã có bytes thật; để provenance kiểm được phải đặt vào slot canonical (xem PROJECT_STATUS)", rel)
        else:
            unclassified.append(rel)

    group_list = [groups[k] for k in sorted(groups)]
    for g in group_list:
        g["paths"] = sorted(g["paths"])
        g["count"] = len(g["paths"])
    catalog_gate = {
        "status": "complete" if not unclassified else "incomplete",
        "total_pt_files": len(all_pt),
        "in_models": sum(1 for rel in all_pt if rel in in_models),
        "in_groups": sum(g["count"] for g in group_list),
        "unclassified": unclassified,
        "source": "quét đĩa (scripts/build_registry.py)",
    }
    return group_list, catalog_gate


# ---------------------------------------------------------------------------
# Báo cáo bằng chứng đã có (domain gap, audit ảnh ngoài)
# ---------------------------------------------------------------------------
def build_evidence_reports() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    gap = load_json(reg.DOMAIN_GAP_REPORT_JSON)
    if gap:
        reports["domain_gap"] = {
            "source": reg.DOMAIN_GAP_REPORT_JSON,
            "evidence": gap.get("evidence"),
            "models": gap.get("models"),
            "recommendation": gap.get("recommendation"),
        }
    per_class = load_json(reg.PER_CLASS_REPORT_JSON)
    if per_class:
        consistent = all(
            all(src["consistency"].values()) for model in per_class["models"].values() for src in model.values()
        )
        reports["per_class"] = {
            "source": reg.PER_CLASS_REPORT_JSON,
            "evidence": per_class.get("evidence"),
            "internally_consistent": consistent,
            "plantdoc_classes_covered": per_class["summary"]["plantdoc_classes_covered"],
            "weak_classes_fedavg": per_class["summary"]["weak_classes_fedavg"],
        }
    shards = load_json(reg.CLIENT_SHARD_COUNTS_JSON)
    if shards:
        reports["client_shards"] = {
            "source": reg.CLIENT_SHARD_COUNTS_JSON,
            "evidence": shards.get("evidence"),
            "partition_scheme": shards.get("partition_scheme"),
            "consistent": shards.get("consistent"),
            "client_n_k": {name: c["n_k"] for name, c in (shards.get("clients") or {}).items()},
            "total_train_samples": shards.get("total_train_samples"),
        }
    audit = load_json(reg.INFERENCE_AUDIT_JSON)
    if audit:
        images = audit.get("images") or []
        reports["inference_audit"] = {
            "source": reg.INFERENCE_AUDIT_JSON,
            "note": audit.get("note"),
            "image_count": len(images),
            "models_disagree_count": sum(1 for i in images if i.get("models_disagree")),
            "uncertain_flagged_count": sum(
                1 for i in images for k in ("centralized", "fedavg") if (i.get(k) or {}).get("is_uncertain")
            ),
        }
    return reports


def build_registry(*, forward: bool = True) -> dict[str, Any]:
    models = (
        build_scientific_canonical()
        + build_published_demo()
        + build_network_and_reference()
        + build_smoke()
    )
    apply_family_defaults(models)
    apply_web_gate(models, forward=forward)
    apply_leakage_audit(models)
    artifact_groups, catalog_gate = build_artifact_catalog(models)
    gates = build_gates()
    gates["checkpoint_catalog"] = catalog_gate
    return {
        "schema_version": reg.SCHEMA_VERSION,
        "generator": "scripts/build_registry.py",
        "canonical_run_id": reg.CANONICAL_RUN_ID,
        "official_direction": {
            "current_family": reg.FAMILY_PUBLISHED_DEMO,
            "protocol_id": reg.DEMO_PROTOCOL_ID,
            "flow": (
                "train ngoài repo (Kaggle/branch) -> bundle + manifest + SHA-256 -> "
                "import/kiểm hash -> registry -> status/web/báo cáo"
            ),
        },
        "dataset_truth": build_dataset_truth(),
        "families": {
            name: {"description": desc}
            for name, desc in reg.FAMILY_DESCRIPTIONS.items()
        },
        "models": models,
        "artifact_groups": artifact_groups,
        "evidence_reports": build_evidence_reports(),
        "gates": gates,
    }


def main_build(*, forward: bool = True) -> dict[str, Any]:
    """Sinh registry.json + canonical_manifest.json từ sự thật trên đĩa. Dùng bởi update_status.py."""
    registry = build_registry(forward=forward)
    (ROOT / reg.REGISTRY_JSON).write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[build_registry] wrote {reg.REGISTRY_JSON} ({len(registry['models'])} models)")

    canonical = build_canonical_manifest(registry["models"])
    (ROOT / reg.CANONICAL_MANIFEST_JSON).write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[build_registry] rewrote {reg.CANONICAL_MANIFEST_JSON} (honest regeneration)")
    return registry


def refresh_registry_gates() -> None:
    """Đọc lại các file gate vừa chạy và ghi vào registry.json (không build lại model)."""
    path = ROOT / reg.REGISTRY_JSON
    registry = json.loads(path.read_text(encoding="utf-8"))
    catalog = (registry.get("gates") or {}).get("checkpoint_catalog")
    registry["gates"] = build_gates()
    if catalog is not None:
        registry["gates"]["checkpoint_catalog"] = catalog
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-forward", action="store_true",
                        help="Bỏ qua bài forward (nhanh; web_visible sẽ không được cấp)")
    args = parser.parse_args()
    main_build(forward=not args.skip_forward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

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


def inspect_checkpoint(rel_path: str) -> dict[str, Any]:
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
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
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


def base_model(model_id: str, family: str, display_name: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "family": family,
        "display_name": display_name,
        "architecture": "MobileNetV3-Small",
        "num_classes": 38,
        "checkpoint_path": None,
        "checkpoint_exists": False,
        "file_sha256": None,
        "state_sha256": None,
        "expected_state_sha256": None,
        "checkpoint_status": None,
        "preprocessing_id": None,
        "protocol_id": None,
        "seed": None,
        "metrics": {},
        "metrics_source": None,
        "status": None,
        "web_visible": False,
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
        "artifacts/checkpoints/fedavg_best.pt từng bị smoke ghi đè acc=1.0)."
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
            m["notes"] = (m["notes"] or "") + " HIỆN TẠI slot này đang chứa checkpoint smoke — cần cách ly vào artifacts/checkpoints/smoke/."
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


def _demo_metrics(validation: dict[str, Any] | None) -> dict[str, Any]:
    if not validation:
        return {}
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

    return {
        "overall": {
            "count": overall.get("count"),
            "top1_accuracy": overall.get("top1_accuracy"),
            "macro_f1_supported": overall.get("macro_f1_supported"),
            "crop_accuracy": overall.get("crop_accuracy"),
        },
        "plantdoc": src("plantdoc"),
        "plantvillage": src("plantvillage"),
    }


def build_published_demo() -> list[dict[str, Any]]:
    import_manifest = load_json(reg.DEMO_IMPORT_MANIFEST_JSON)
    if import_manifest is None:
        return []
    protocol_id = import_manifest.get("protocol_id", reg.DEMO_PROTOCOL_ID)
    seed = import_manifest.get("seed")

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
        insp = inspect_checkpoint(ckpt_rel)
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
            protocol_id=protocol_id,
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
        ("network_attack_matrix_observe", "Attack matrix — observe", "artifacts/checkpoints/attack_matrix_observe.pt",
         "Thí nghiệm fault modes, policy observe-only."),
        ("network_attack_matrix_reject", "Attack matrix — reject", "artifacts/checkpoints/attack_matrix_reject.pt",
         "Thí nghiệm fault modes, policy reject-suspicious."),
        ("network_rtt_checkpoint_test", "RTT + checkpoint test", "artifacts/checkpoints/network_rtt_checkpoint_test.pt",
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
        checkpoint_path=reg.PLANTVILLAGE_TRACED_PT,
        checkpoint_exists=insp["checkpoint_exists"],
        file_sha256=insp["file_sha256"],
        checkpoint_status="present" if insp["checkpoint_exists"] else "missing",
        preprocessing_id="get_default_transform(eval)",
        status="reference_only",
        web_visible=insp["checkpoint_exists"],
        display_label="Tham chiếu (không phải FedAvg)",
        notes="TorchScript inference tham chiếu độc lập; softmax chưa calibration.",
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
            "status": provenance.get("status"),
            "critical_issues": [i.get("code") for i in provenance.get("issues", []) if i.get("severity") == "critical"],
            "warning_issues": [i.get("code") for i in provenance.get("issues", []) if i.get("severity") == "warning"],
            "source": reg.PROVENANCE_VALIDATION_JSON,
        }
    pre_demo = load_json(reg.PRE_DEMO_CHECK_JSON)
    if pre_demo is not None:
        gates["pre_demo_check"] = {
            "overall": pre_demo.get("overall"),
            "source": reg.PRE_DEMO_CHECK_JSON,
        }
    final = load_json(reg.FINAL_ACCEPTANCE_JSON)
    if final is not None:
        gates["final_acceptance"] = {
            "technical": final.get("technical_acceptance"),
            "scientific": final.get("scientific_acceptance"),
            "readiness_overall": final.get("readiness_overall"),
            "scientific_blockers": final.get("scientific_blockers"),
            "source": reg.FINAL_ACCEPTANCE_JSON,
        }
    network = load_json(reg.NETWORK_ARTIFACT_VALIDATION_JSON)
    if network is not None:
        gates["network_artifact"] = {
            "ok": network.get("ok"),
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
def build_registry() -> dict[str, Any]:
    models = (
        build_scientific_canonical()
        + build_published_demo()
        + build_network_and_reference()
        + build_smoke()
    )
    return {
        "schema_version": 1,
        "generator": "scripts/build_registry.py",
        "canonical_run_id": reg.CANONICAL_RUN_ID,
        "dataset_truth": build_dataset_truth(),
        "families": {
            name: {"description": desc}
            for name, desc in reg.FAMILY_DESCRIPTIONS.items()
        },
        "models": models,
        "gates": build_gates(),
    }


def main() -> int:
    registry = build_registry()
    out_path = ROOT / reg.REGISTRY_JSON
    out_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build_registry] wrote {reg.REGISTRY_JSON} ({len(registry['models'])} models)")

    canonical = build_canonical_manifest(registry["models"])
    (ROOT / reg.CANONICAL_MANIFEST_JSON).write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[build_registry] rewrote {reg.CANONICAL_MANIFEST_JSON} (honest regeneration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

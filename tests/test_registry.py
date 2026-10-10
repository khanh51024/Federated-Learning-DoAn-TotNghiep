"""Hợp đồng của artifacts/registry.json (docs/MODEL_REGISTRY.md)."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import build_registry as br
from src import registry as reg

REQUIRED_FIELDS = {
    "model_id", "family", "pipeline", "role", "dataset_id", "protocol_id", "taxonomy_sha256",
    "checkpoint_path", "checkpoint_exists", "checkpoint_status", "preprocessing_id",
    "leakage_audit", "web_visible", "web_gate_failures",
}
NEVER_VISIBLE_ROLES = {reg.ROLE_HISTORICAL, reg.ROLE_TECHNICAL_DEMO, reg.ROLE_QUARANTINED}


@pytest.fixture(scope="module")
def registry() -> dict:
    return reg.load_registry()


def test_schema_and_required_fields(registry):
    assert registry["schema_version"] == reg.SCHEMA_VERSION
    ids = [m["model_id"] for m in registry["models"]]
    assert len(ids) == len(set(ids)), "model_id phải duy nhất"
    for model in registry["models"]:
        assert REQUIRED_FIELDS <= set(model), (model["model_id"], REQUIRED_FIELDS - set(model))
        assert model["family"] in reg.FAMILY_DESCRIPTIONS
        assert model["role"] in {
            reg.ROLE_CURRENT, reg.ROLE_HISTORICAL, reg.ROLE_TECHNICAL_DEMO, reg.ROLE_REFERENCE, reg.ROLE_QUARANTINED,
        }


def test_only_gated_models_are_web_visible(registry):
    for model in registry["models"]:
        if model["role"] in NEVER_VISIBLE_ROLES:
            assert model["web_visible"] is False, f"{model['model_id']} (role={model['role']}) không được hiện trên web"
        if model["web_visible"]:
            assert model["web_gate_failures"] == []
            assert model["checkpoint_status"] == "hash_verified"
            assert model["preprocessing_id"] in reg.KNOWN_PREPROCESSING
            assert (model["inference_test"] or {}).get("status") == "pass"
            assert model["inference"]["backend"] in {reg.BACKEND_PUBLISHED_BUNDLE, reg.BACKEND_TORCHSCRIPT_REFERENCE}
        else:
            assert model["web_gate_failures"], f"{model['model_id']} bị chặn nhưng không ghi lý do"


def test_smoke_family_is_always_quarantined(registry):
    smoke = [m for m in registry["models"] if m["family"] == reg.FAMILY_SMOKE]
    assert smoke, "phải còn checkpoint smoke được cách ly để chứng minh cơ chế"
    for model in smoke:
        assert model["role"] == reg.ROLE_QUARANTINED
        assert model["checkpoint_path"].startswith(reg.SMOKE_CHECKPOINTS_DIR + "/")


def test_registry_hashes_match_disk(registry):
    checked = 0
    for model in registry["models"]:
        if not model["checkpoint_exists"] or not model["file_sha256"]:
            continue
        digest = hashlib.sha256((reg.ROOT / model["checkpoint_path"]).read_bytes()).hexdigest()
        assert digest == model["file_sha256"], f"{model['model_id']}: registry.json lệch đĩa — chạy scripts/update_status.py"
        checked += 1
    assert checked >= 8


def test_registry_reflects_current_disk_state(registry):
    """registry.json phải khớp bản sinh lại từ đĩa (không chạy bài forward cho nhanh)."""
    fresh = br.build_registry(forward=False)
    committed = {m["model_id"]: m for m in registry["models"]}
    assert set(committed) == {m["model_id"] for m in fresh["models"]}
    for model in fresh["models"]:
        old = committed[model["model_id"]]
        for key in ("role", "family", "file_sha256", "checkpoint_status", "checkpoint_exists"):
            assert old[key] == model[key], f"{model['model_id']}.{key}: registry.json cũ — chạy scripts/update_status.py"


def test_every_checkpoint_on_disk_is_catalogued(registry):
    _groups, gate = br.build_artifact_catalog(registry["models"])
    assert gate["unclassified"] == [], f".pt chưa có tên trong danh mục: {gate['unclassified']}"


def test_unlisted_checkpoint_is_reported(registry, monkeypatch):
    real_walk = br._walk_pt_files
    monkeypatch.setattr(br, "_walk_pt_files", lambda: real_walk() + ["artifacts/stray_test_checkpoint.pt"])
    _groups, gate = br.build_artifact_catalog(registry["models"])
    assert gate["status"] == "incomplete"
    assert "artifacts/stray_test_checkpoint.pt" in gate["unclassified"]


def test_outputs_of_the_legitimate_pipeline_are_catalogued(registry, monkeypatch):
    """Đầu ra của promote_verified_* và run thật phải có chỗ trong danh mục (không làm checklist kẹt CHƯA)."""
    real_walk = br._walk_pt_files
    extra = [
        "artifacts/verified/20261010_x/fedavg_label1_best.pt",
        "artifacts/verified_network/20261010_x/global_model.pt",
        "artifacts/report_experiments/20261010_000000/label1/checkpoints/fedavg_best.pt",
        "results/fedavg/run_that/best.pt",
    ]
    monkeypatch.setattr(br, "_walk_pt_files", lambda: real_walk() + extra)
    groups, gate = br.build_artifact_catalog(registry["models"])
    assert gate["status"] == "complete", gate["unclassified"]
    by_id = {g["group_id"]: g for g in groups}
    assert extra[0] in by_id["verified_promoted"]["paths"] and extra[1] in by_id["verified_promoted"]["paths"]
    assert extra[2] in by_id["report_experiments_real"]["paths"]
    assert extra[3] in by_id["kaggle_run_checkpoint_real"]["paths"]


def test_pt_scan_excludes_only_root_level_dirs(tmp_path, monkeypatch):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "a.pt").write_bytes(b"x")
    (tmp_path / "artifacts" / "tmp").mkdir(parents=True)
    (tmp_path / "artifacts" / "tmp" / "b.pt").write_bytes(b"x")
    (tmp_path / "artifacts" / "__pycache__").mkdir()
    (tmp_path / "artifacts" / "__pycache__" / "c.pt").write_bytes(b"x")
    monkeypatch.setattr(br, "ROOT", tmp_path)
    assert br._walk_pt_files() == ["artifacts/tmp/b.pt"]


def test_untrusted_bundle_checkpoint_is_never_unpickled(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "ROOT", tmp_path)
    (tmp_path / "x.pt").write_bytes(b"khong phai pickle that")
    info = br.inspect_checkpoint("x.pt", expected_sha256="0" * 64)
    assert info["state_sha256"] is None
    assert "load_skipped" in info["metadata"]


def test_web_gate_grants_visibility_only_when_all_five_pass(monkeypatch):
    classes = json.loads((reg.ROOT / reg.DATASET_MANIFEST_JSON).read_text(encoding="utf-8"))["classes"]
    model = {
        "model_id": "x", "role": reg.ROLE_CURRENT, "checkpoint_status": "hash_verified",
        "taxonomy_sha256": br.classes_sha256(classes), "preprocessing_id": "canonical_v1",
        "metrics_source": reg.DATASET_MANIFEST_JSON, "protocol_id": "p", "dataset_id": "d",
        "metrics_waiver": None, "inference_test": None,
    }
    monkeypatch.setattr(br, "run_forward_test", lambda m: {"status": "pass"})
    ok = dict(model)
    br.apply_web_gate([ok], forward=True)
    assert ok["web_visible"] is True and ok["web_gate_failures"] == []

    monkeypatch.setattr(br, "run_forward_test", lambda m: {"status": "fail", "detail": "ValueError: boom"})
    bad = dict(model)
    br.apply_web_gate([bad], forward=True)
    assert bad["web_visible"] is False
    assert bad["web_gate_failures"] == ["5.inference: ValueError: boom"]

    skipped = dict(model)
    br.apply_web_gate([skipped], forward=False)
    assert skipped["web_visible"] is False, "--skip-forward không được cấp quyền hiển thị"


def test_taxonomy_expectation_comes_from_the_bundle_contract_file(monkeypatch):
    classes = json.loads((reg.ROOT / reg.DATASET_MANIFEST_JSON).read_text(encoding="utf-8"))["classes"]
    base = {
        "model_id": "x", "role": reg.ROLE_REFERENCE, "checkpoint_status": "hash_verified",
        "taxonomy_sha256": br.classes_sha256(classes), "preprocessing_id": "get_default_transform(eval)",
        "metrics_source": None, "metrics_waiver": "w", "protocol_id": "p", "dataset_id": "d", "inference_test": None,
    }
    ok = dict(base)
    br.apply_web_gate([ok], forward=False)
    assert not any(f.startswith("2.taxonomy") for f in ok["web_gate_failures"])

    # contract/taxonomy.py của bundle đổi thứ tự lớp -> model dùng thứ tự khác bị chặn
    monkeypatch.setattr(br, "classes_from_taxonomy", lambda path: list(reversed(classes)))
    swapped = dict(base)
    br.apply_web_gate([swapped], forward=False)
    assert any(f.startswith("2.taxonomy") for f in swapped["web_gate_failures"])

    # sửa import_manifest.json KHÔNG thể làm điều kiện này dễ hơn: nguồn kỳ vọng không đọc file đó nữa
    real_load = br.load_json
    monkeypatch.setattr(br, "classes_from_taxonomy", lambda path: classes)
    monkeypatch.setattr(br, "load_json", lambda rel: {"classes": list(reversed(classes))} if rel == reg.DEMO_IMPORT_MANIFEST_JSON else real_load(rel))
    still_ok = dict(base)
    br.apply_web_gate([still_ok], forward=False)
    assert not any(f.startswith("2.taxonomy") for f in still_ok["web_gate_failures"])

    # không đọc được hợp đồng => fail-closed
    def unreadable(path):
        raise ValueError("PLANTVILLAGE_38_CLASSES was not found")

    monkeypatch.setattr(br, "classes_from_taxonomy", unreadable)
    closed = dict(base)
    br.apply_web_gate([closed], forward=False)
    assert any(f.startswith("2.taxonomy") for f in closed["web_gate_failures"])

"""Gói chỉ đánh giá trên tập kiểm định (`evidence_scope: "validation_only"` trong import_manifest.json).

Các mục cần tập kiểm thử niêm phong hoặc dữ liệu ngoài không chạy lại: chúng bị giữ lại với lý do phạm vi, không trang nào lỗi,
và việc thiếu khối hiệu chuẩn (không có tập calibration) không làm hỏng phần tái đánh giá.
"""
from __future__ import annotations

import copy
import json
import shutil
from typing import Any, Callable

import pytest

from src import registry as reg
from web.demo_evidence import STORE, load_evidence
from web.routers.inference import _gated_evidence
from web.services import evidence_service as es

# mục của /api/evidence/summary cần dữ liệu test/ngoài, và tệp báo cáo tương ứng
NEEDS_TEST_DATA = {
    "holdout": reg.HOLDOUT_EVAL_JSON,
    "statistics": reg.STATISTICS_JSON,
    "fullframe": reg.FULLFRAME_EVAL_JSON,
    "fairness": reg.CLIENT_FAIRNESS_JSON,
    "plantvillage_reference": reg.PLANTVILLAGE_EVAL_JSON,
    "community_simulation": reg.COMMUNITY_SIMULATION_JSON,
    "leakage": reg.RELEASE_LEAKAGE_AUDIT_JSON,
    "training": reg.TRAINING_PROTOCOL_REPORT_JSON,
    "gd4": reg.GD4_COMPRESSION_JSON,
}


@pytest.fixture(scope="module")
def evidence() -> dict:
    ev = _gated_evidence()
    if not ev.get("available"):
        pytest.skip(f"bundle đối chứng chưa khả dụng: {ev.get('reason')}")
    return ev


def _without(monkeypatch, *relatives: str) -> None:
    real = es._load_json
    monkeypatch.setattr(es, "_load_json", lambda rel: None if rel in relatives else real(rel))


def test_manifest_scope_is_exposed_in_the_evidence_payload(tmp_path):
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(STORE, bundle_copy)
    assert load_evidence(bundle_copy).get("evidence_scope") in (None, "validation_only")   # gói gốc: tùy manifest, không bao giờ thiếu khóa
    manifest_path = bundle_copy / "import_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_scope"] = "validation_only"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    evidence = load_evidence(bundle_copy)
    assert evidence["available"], evidence.get("reason")
    assert evidence["evidence_scope"] == "validation_only"
    manifest.pop("evidence_scope")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    assert load_evidence(bundle_copy)["evidence_scope"] is None


def test_withheld_sections_of_a_validation_only_bundle_say_why(evidence, monkeypatch):
    _without(monkeypatch, *NEEDS_TEST_DATA.values())
    scoped = {**evidence, "evidence_scope": "validation_only"}
    result = es.summary(scoped)
    assert result["available"] is True
    for name in NEEDS_TEST_DATA:
        assert result[name] == {"available": False, "reason": es.VALIDATION_ONLY_REASON}, name
    assert "tập kiểm định" in es.VALIDATION_ONLY_REASON and "niêm phong" in es.VALIDATION_ONLY_REASON
    # các mục dựa trên tập kiểm định vẫn dựng bình thường
    for name in ("reeval", "plantdoc", "shards", "identity"):
        assert result[name].get("available", True) is True, name
    # trang chi tiết và các cách chia không lỗi khi thiếu báo cáo test
    assert es.details(scoped)["available"] is True and "test" not in es.details(scoped)["confusion"]
    assert es.partitions(scoped)["available"] is True
    # cùng dữ liệu nhưng gói không khai phạm vi: giữ nguyên lý do kỹ thuật cũ ("chưa có …")
    plain = es.summary({**evidence, "evidence_scope": None})
    assert plain["holdout"]["available"] is False and plain["holdout"]["reason"] != es.VALIDATION_ONLY_REASON


def test_report_without_bundle_identity_is_withheld_even_when_a_file_exists(evidence):
    """Mô phỏng cộng đồng không kiểm release/checkpoint và dùng tập test: gói validation-only không được hiện số của gói khác."""
    if es._load_json(reg.COMMUNITY_SIMULATION_JSON) is None:
        pytest.skip("chưa có báo cáo mô phỏng cộng đồng trên đĩa")
    assert es.summary(evidence)["community_simulation"]["available"] is True
    scoped = es.summary({**evidence, "evidence_scope": "validation_only"})
    assert scoped["community_simulation"] == {"available": False, "reason": es.VALIDATION_ONLY_REASON}


def test_only_the_test_split_of_a_partly_available_section_is_rewritten():
    section = {"available": True, "splits": {"val": {"available": True, "count": 3}, "test": {"available": False, "reason": "chưa có"}}}
    result = es._apply_evidence_scope({"evidence_scope": "validation_only"}, {"statistics": copy.deepcopy(section), "reeval": {"available": False, "reason": "x"}})
    assert result["statistics"]["splits"]["test"] == {"available": False, "reason": es.VALIDATION_ONLY_REASON}
    assert result["statistics"]["splits"]["val"] == {"available": True, "count": 3}
    assert result["reeval"] == {"available": False, "reason": "x"}, "mục không cần dữ liệu test giữ nguyên lý do riêng"
    untouched = es._apply_evidence_scope({"evidence_scope": None}, {"holdout": {"available": False, "reason": "chưa có"}})
    assert untouched["holdout"]["reason"] == "chưa có"


def _strip_calibration(report: dict[str, Any]) -> None:
    for model in report["models"].values():
        model.pop("calibration", None)
        model.pop("temperature_fitted_on_calibration", None)
    report.pop("calibration_count", None)


def _tamper(monkeypatch, relative: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    real = es._load_json

    def fake(rel: str):
        data = real(rel)
        if rel == relative and data is not None:
            data = copy.deepcopy(data)
            mutate(data)
        return data

    monkeypatch.setattr(es, "_load_json", fake)


def test_missing_calibration_blocks_do_not_break_reeval_and_give_no_temperatures(evidence, monkeypatch):
    if es._load_json(reg.LOCAL_REEVAL_JSON) is None:
        pytest.skip("chưa có báo cáo tái đánh giá để bỏ khối hiệu chuẩn")
    _tamper(monkeypatch, reg.LOCAL_REEVAL_JSON, _strip_calibration)
    section = es.summary(evidence)["reeval"]
    assert section["available"] is True, section.get("reason")
    assert all(m["temperature"] is None for m in section["models"])
    assert all(row["ece_raw"] is None and row["ece_temperature_scaled"] is None for m in section["models"] for row in m["calibration"].values())
    assert es.temperatures(evidence) == {}, "không có nhiệt độ thì dự đoán không có khối `calibrated`"


def test_a_report_may_leave_out_extra_models_but_never_a_core_model(evidence):
    expected = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}
    assert es._checkpoint_problems({"checkpoint_sha256": dict(expected)}, evidence) == []
    extra = {**expected, "scaffold_int8": "f" * 64}
    assert es._checkpoint_problems({"checkpoint_sha256": extra}, evidence), "mô hình lạ trong báo cáo vẫn bị từ chối"
    for core in es.MODEL_ORDER:
        missing = {k: v for k, v in expected.items() if k != core}
        assert es._checkpoint_problems({"checkpoint_sha256": missing}, evidence), core
    wrong = {**expected, "fedavg": "0" * 64}
    assert es._checkpoint_problems({"checkpoint_sha256": wrong}, evidence)
    bundle_with_extras = {**evidence, "models": [*evidence["models"], {"model_id": "future_method", "checkpoint_sha256": "a" * 64},
                                                  {"model_id": "future_method_int8", "checkpoint_sha256": "b" * 64}]}
    assert es._checkpoint_problems({"checkpoint_sha256": dict(expected)}, bundle_with_extras) == [], "gói có thêm mô hình mà báo cáo chưa đo"

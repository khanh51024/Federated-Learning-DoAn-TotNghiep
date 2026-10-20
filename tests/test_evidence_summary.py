"""GET /api/evidence/summary: mỗi phần fail-closed riêng, danh tính đối chiếu với bundle đã kiểm hash."""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.routers.inference import _gated_evidence
from web.services import evidence_service

SECTIONS = ("reeval", "plantdoc", "domain_gap", "leakage", "shards")


@pytest.fixture(scope="module")
def evidence() -> dict:
    ev = _gated_evidence()
    if not ev.get("available"):
        pytest.skip(f"bundle đối chứng chưa khả dụng: {ev.get('reason')}")
    return ev


def test_summary_builds_every_section_from_real_artifacts(evidence):
    result = evidence_service.summary(evidence)
    assert result["available"] is True
    for name in (*SECTIONS, "checklist"):
        assert result[name]["available"] is True, (name, result[name].get("reason"))

    reeval = result["reeval"]
    assert [m["model_id"] for m in reeval["models"]] == list(evidence_service.MODEL_ORDER)
    assert reeval["comparisons_total"] == 14 and reeval["main_models_exact"] is True
    # số tái đánh giá khớp số đã kiểm hash trong dung sai của chính báo cáo (Local-only có thể lệch <= 1 ảnh)
    published = {m["model_id"]: m["metrics"] for m in evidence["models"]}
    for model in reeval["models"]:
        assert model["plantvillage"]["top1"] == pytest.approx(published[model["model_id"]]["plantvillage"]["top1_accuracy"], abs=2e-3)
        assert model["plantdoc"]["top1"] == pytest.approx(published[model["model_id"]]["plantdoc"]["top1_accuracy"], abs=2e-3)
    assert set(reeval["pv_recall"]) == set(evidence_service.MODEL_ORDER)

    shards = result["shards"]
    assert shards["classes_total"] == 38 and len(shards["clients"]) == 5
    assert sum(c["n_k"] for c in shards["clients"]) == shards["total_train"] == 37238
    for row in shards["classes"]:
        assert row["total"] == sum(row["per_client"].values())
    assert result["leakage"]["passed"] is True and result["leakage"]["overlap_total"] == 0
    assert result["checklist"]["done"] + result["checklist"]["blocked"] + result["checklist"]["todo"] == result["checklist"]["total"]


def test_everything_with_model_numbers_is_withheld_when_bundle_gate_fails():
    result = evidence_service.summary({"available": False, "reason": "cổng registry chặn"})
    assert result["available"] is False and result["identity"] is None
    for name in SECTIONS:
        assert result[name] == {"available": False, "reason": "cổng registry chặn"}
    assert result["checklist"]["available"] is True  # checklist là trạng thái dự án, không phải số liệu mô hình


def test_reeval_is_withheld_when_a_checkpoint_hash_differs(evidence):
    tampered = copy.deepcopy(evidence)
    tampered["models"][1]["checkpoint_sha256"] = "0" * 64
    result = evidence_service.summary(tampered)
    assert result["reeval"]["available"] is False
    assert "SHA-256" in result["reeval"]["reason"]


def test_reports_are_withheld_when_published_numbers_differ(evidence):
    tampered = copy.deepcopy(evidence)
    fedavg = next(m for m in tampered["models"] if m["model_id"] == "fedavg")
    fedavg["metrics"]["plantdoc"]["top1_accuracy"] += 0.05
    result = evidence_service.summary(tampered)
    assert result["plantdoc"]["available"] is False
    assert result["domain_gap"]["available"] is False


def test_leakage_and_shards_are_withheld_on_identity_mismatch(evidence):
    tampered = copy.deepcopy(evidence)
    tampered["validation_manifest_sha256"] = "f" * 64
    tampered["protocol_id"] = "another_protocol"
    tampered["shard_counts"] = {**tampered["shard_counts"], "client_00": 1}
    result = evidence_service.summary(tampered)
    assert result["leakage"]["available"] is False
    assert result["shards"]["available"] is False
    assert result["reeval"]["available"] is False


def test_missing_report_files_do_not_raise(evidence, monkeypatch):
    monkeypatch.setattr(evidence_service, "_load_json", lambda relative: None)
    monkeypatch.setattr(evidence_service.model_service, "shard_report", lambda: None)
    result = evidence_service.summary(evidence)
    for name in (*SECTIONS, "checklist"):
        assert result[name]["available"] is False and result[name]["reason"]


def test_inconsistent_shard_report_is_withheld(evidence, monkeypatch):
    broken = {"protocol_id": evidence["protocol_id"], "clients": {}, "classes": [], "totals_all_clients": {}}
    monkeypatch.setattr(evidence_service.model_service, "shard_report", lambda: broken)
    assert evidence_service.summary(evidence)["shards"]["available"] is False


def test_http_route_returns_the_summary(evidence):
    response = TestClient(app).get("/api/evidence/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["identity"]["protocol_id"] == evidence["protocol_id"]
    assert len(body["identity"]["models"]) == 7
    assert len(response.content) < 145_000  # payload nhỏ để mở trang nhanh trên server 2 GiB RAM


def test_checklist_is_redacted_when_the_bundle_is_withheld():
    result = evidence_service.summary({"available": False, "reason": "cổng registry chặn"})
    checklist = result["checklist"]
    assert checklist["available"] is True and checklist["redacted"] is True
    assert checklist["readiness"] == ""
    for item in checklist["items"]:
        assert set(item) == {"id", "title", "status"}, "detail/unlock chứa số của mô hình, không được lộ khi bị chặn"


def test_checklist_keeps_detail_when_the_bundle_is_available(evidence):
    checklist = evidence_service.summary(evidence)["checklist"]
    assert checklist["redacted"] is False
    assert all("detail" in item for item in checklist["items"])


def test_leakage_pass_flag_is_not_trusted_without_matching_numbers(evidence, monkeypatch):
    original = evidence_service._load_json(evidence_service.reg.RELEASE_LEAKAGE_AUDIT_JSON)
    assert original is not None
    forged = copy.deepcopy(original)
    forged["passed"] = True
    forged["pairwise_overlaps"]["train__val"]["byte_sha256"] = 3  # trùng lặp thật nhưng cờ vẫn "đạt"
    monkeypatch.setattr(evidence_service, "_load_json", lambda relative: forged if relative.endswith("release_leakage_audit.json") else original)
    assert evidence_service.summary(evidence)["leakage"]["available"] is False


def test_reeval_recomputes_totals_instead_of_trusting_the_report(evidence, monkeypatch):
    original = evidence_service._load_json(evidence_service.reg.LOCAL_REEVAL_JSON)
    forged = copy.deepcopy(original)
    forged["comparisons_exact"] = 14
    forged["main_models_exact"] = True
    forged["models"]["local_01"]["vs_published"]["plantvillage"]["reproduced"] = False
    forged["models"]["local_01"]["vs_published"]["plantvillage"]["correct_diff_images"] = 1
    monkeypatch.setattr(evidence_service, "_load_json", lambda relative: forged if relative.endswith("local_reeval_report.json") else evidence_service._JSON_CACHE.get(relative, (None, None))[1])
    result = evidence_service.summary(evidence)["reeval"]
    assert result["available"] is True
    assert result["comparisons_exact"] < 14, "tổng phải tính lại từ từng phép so sánh"
    assert result["comparisons_total"] == 14


def test_non_finite_numbers_become_null_not_a_500(evidence, monkeypatch):
    original = evidence_service._load_json(evidence_service.reg.LOCAL_REEVAL_JSON)
    forged = copy.deepcopy(original)
    forged["models"]["fedavg"]["calibration"]["plantdoc"]["ece_raw"] = float("nan")
    forged["models"]["fedavg"]["temperature_fitted_on_calibration"] = float("inf")
    monkeypatch.setattr(evidence_service, "_load_json", lambda relative: forged if relative.endswith("local_reeval_report.json") else evidence_service._JSON_CACHE.get(relative, (None, None))[1])
    response = TestClient(app).get("/api/evidence/summary")
    assert response.status_code == 200
    fedavg = next(m for m in response.json()["reeval"]["models"] if m["model_id"] == "fedavg")
    assert fedavg["calibration"]["plantdoc"]["ece_raw"] is None and fedavg["temperature"] is None


def test_summary_has_the_plantvillage_reference_and_community_simulation_sections(evidence):
    """Hai cách nhận còn lại của đồ án có báo cáo riêng, tách khỏi bảy mô hình đã kiểm hash (không làm các mục cũ bị giữ lại)."""
    result = evidence_service.summary(evidence)
    assert result["reeval"]["available"] is True
    pv = result["plantvillage_reference"]
    assert pv["available"] is True, pv.get("reason")
    assert pv["model"]["trained_by_project"] is False and pv["leakage"]["plantvillage"] == "likely" and pv["leakage"]["plantdoc"] == "none"
    assert pv["preprocessing_chosen"] == "hf_half_256_224"
    cmp = pv["preprocessing_comparison_on_val"]
    assert cmp["hf_half_256_224"]["val_top1"] > cmp["imagenet_256_224"]["val_top1"] + 0.3, "chuẩn hóa của tác giả phải tốt hơn hẳn ImageNet"
    assert pv["holdout"]["by_source"]["plantdoc"]["count"] == 172
    assert pv["vs_fedavg_on_holdout"]["plantdoc"]["fedavg_top1"] > pv["vs_fedavg_on_holdout"]["plantdoc"]["plantvillage_model_top1"]

    sim = result["community_simulation"]
    assert sim["available"] is True, sim.get("reason")
    assert {r["scenario"] for r in sim["runs"]} == {"iid", "three_groups", "one_group", "plantdoc_only"}
    gated = [r for r in sim["runs"] if r["gate"]]
    assert {r["k"] for r in gated} == {3, 10, 25} and all(len(r["mean_curve"]["accuracy"]) == sim["rounds"] + 1 for r in sim["runs"])
    assert sim["test"]["images"] == 11015 and sim["base_test"]["plantdoc_count"] == 172


def test_plantvillage_report_is_withheld_when_it_does_not_match_the_pinned_checkpoint(evidence, monkeypatch):
    report = evidence_service._load_json(evidence_service.reg.PLANTVILLAGE_EVAL_JSON)
    tampered = copy.deepcopy(report)
    tampered["model"]["checkpoint_sha256"] = "0" * 64
    monkeypatch.setattr(evidence_service, "_load_json", lambda path: tampered if path == evidence_service.reg.PLANTVILLAGE_EVAL_JSON else None)
    result = evidence_service._plantvillage_reference(evidence)
    assert result["available"] is False and "SHA-256" in result["reason"]

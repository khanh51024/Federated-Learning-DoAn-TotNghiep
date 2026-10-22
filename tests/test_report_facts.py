"""Dữ kiện (`summary.facts`) và nhóm nguồn của gói: câu phạm vi / hạn chế trên web đọc từ đây, không viết sẵn trong JS."""
from __future__ import annotations

import re

import pytest

from src import bundle
from web.routers.inference import _gated_evidence
from web.services import evidence_service as es
from tests.test_web_static_frontend import JS_BY_REL


@pytest.fixture(scope="module")
def evidence() -> dict:
    ev = _gated_evidence()
    if not ev.get("available"):
        pytest.skip(f"bundle đối chứng chưa khả dụng: {ev.get('reason')}")
    return ev


def _training(**overrides):
    base = {"optimizer": "SGD", "lr": 0.02, "batch_size": 128, "label_smoothing": 0.1, "precision": "bf16", "train_transform": "gpu_v2",
            "loss_weighting": "domain_balanced", "selection": "macro-F1", "steps_total": 40, "local_epochs": 1}
    return {**base, **overrides}


def _facts(training: dict | None, seeds=(42, 123), delta=False):
    evidence = {"seed": 42, "training": training}
    return es._facts(evidence, {"available": True, "seeds": list(seeds)}, {"available": True, "delta_l2_recorded": delta, "transport_mode": "in_process_simulation"},
                     {"available": False})


def test_same_recipe_is_derived_from_the_bundle_training_block():
    same = _facts({"centralized": _training(), "fedavg": _training(units="vòng"), "local_00": _training()})
    assert same["same_recipe"] is True and same["recipe_differences"] == []
    differ = _facts({"centralized": _training(optimizer="AdamW", lr=0.001), "fedavg": _training(), "local_00": _training()})
    assert differ["same_recipe"] is False and differ["recipe_differences"] == ["optimizer", "lr"]
    unknown = _facts(None)
    assert unknown["same_recipe"] is None and unknown["recipe"] is None


def test_facts_report_seeds_and_what_was_not_recorded():
    facts = _facts({"centralized": _training(), "fedavg": _training()}, seeds=(42, 123, 2026), delta=False)
    assert facts["seeds_run"] == [42, 123, 2026] and facts["seed_served"] == 42
    assert facts["delta_l2_recorded"] is False and facts["transport_mode"] == "in_process_simulation"
    assert facts["partition_scheme"] and facts["holdout_audited"] is None   # audit chưa có (available False): không khẳng định


def test_facts_of_the_active_bundle(evidence):
    summary = es.summary(evidence)
    facts = summary["facts"]
    assert facts["available"] is True
    assert facts["seed_served"] == evidence["seed"] and facts["seeds_run"] == summary["seeds"]["seeds"]
    match = re.search(r"alpha_(\d+)_(\d+)", facts["partition_scheme"])
    assert (facts["alpha"] is None) == (match is None)
    if facts["same_recipe"] is not None:
        trainings = evidence["training"].values()
        lr_values = {t.get("lr") for t in trainings}
        assert (len(lr_values) == 1) or facts["same_recipe"] is False


def test_facts_are_withheld_with_the_rest_when_the_bundle_gate_fails():
    result = es.summary({"available": False, "reason": "cổng registry chặn"})
    assert result["facts"] == {"available": False, "reason": "cổng registry chặn"}


def test_source_groups_follow_the_bundle_counts():
    groups = bundle.source_groups({"plantdoc": 269, "plantvillage": 4344, "field_other": 2133}, {"plantdoc": 172, "plantvillage": 10843, "field_other": 2086})
    assert [g["id"] for g in groups] == ["plantvillage", "field_other", "plantdoc"]
    assert groups[1]["val_count"] == 2133 and groups[1]["holdout_count"] == 2086 and groups[0]["kind"] == "lab" and groups[2]["kind"] == "field"
    legacy = bundle.source_groups({"plantdoc": 269, "plantvillage": 4344})
    assert [g["id"] for g in legacy] == ["plantvillage", "plantdoc"]
    unknown = bundle.source_groups({"plantvillage": 10, "new_source": 5})
    assert unknown[-1]["id"] == "new_source" and unknown[-1]["label"] == "new_source"


def test_every_source_of_the_bundle_reaches_the_api(evidence):
    summary = es.summary(evidence)
    sources = es.all_sources(evidence)
    assert set(evidence["validation_counts"]) <= set(sources)
    assert [g["id"] for g in summary["identity"]["source_groups"]] == [s for s in ("plantvillage", "field_other", "plantdoc") if s in sources]
    for model in summary["reeval"]["models"]:
        for source in sources:
            assert model[source]["count"] == evidence["validation_counts"][source]
            assert source in model["calibration"]
    for model in summary["holdout"]["models"]:
        assert all(source in model for source in sources)


def test_ui_text_does_not_claim_what_only_the_data_can_say():
    """Không quay lại các câu của bản v3: câu về số lần chạy, cùng điều kiện, hệ số lệch và số nguồn ảnh phải đọc từ facts/nguồn của gói."""
    banned = ("Một kịch bản, một lần chạy", "Kết quả từ một lần chạy với một cách chia", "chưa huấn luyện cùng điều kiện, nên", "Hai nguồn ảnh",
              "FedAvg báo trung bình", "(PlantVillage + PlantDoc)", "Một mức lệch dữ liệu (α = 0,1)", "ba hạt giống; các client")
    for rel, source in JS_BY_REL.items():
        flat = " ".join(source.split())
        for phrase in banned:
            assert phrase not in flat, f"{rel}: câu cố định của bản cũ quay lại: “{phrase}”"
    for rel in ("pages/overview.js", "pages/results.js", "pages/conclusion.js", "pages/methods.js", "pages/noniid.js"):
        assert "facts-text.js" in JS_BY_REL[rel], f"{rel}: câu phạm vi phải dựng từ facts-text.js"

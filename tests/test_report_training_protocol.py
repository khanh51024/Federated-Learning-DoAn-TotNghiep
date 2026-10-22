"""Training report must keep runtime, split and checkpoint provenance separate."""
import copy
import json

import pytest

from src import bundle
from web.routers.inference import _gated_evidence
from web.services import evidence_service
from web.services.training_evidence import training_report


@pytest.fixture(scope="module")
def source():
    evidence = _gated_evidence()
    assert evidence["available"], evidence.get("reason")
    result = evidence_service.summary(evidence)["training"]
    assert result["available"], result.get("reason")
    return evidence, evidence_service._load_json(evidence_service.reg.TRAINING_PROTOCOL_REPORT_JSON)


def _bundle_training():
    manifest = bundle.load_manifest()
    return manifest, manifest["training"]


def test_report_contains_verified_history_configs_and_release(source):
    evidence, report = source
    manifest, training = _bundle_training()
    models = {m["model_id"]: m for m in report["models"]}
    assert len(models) == 7
    central, fed = models["centralized"], models["fedavg"]
    for mid, model in models.items():
        assert model["selected_step"] == training[mid]["best_step"], mid
        assert model["completed_steps"] == training[mid]["budget"], mid
    assert central["config"]["optimizer"] == training["centralized"]["optimizer"]
    assert fed["config"]["client_optimizer"] == training["fedavg"]["optimizer"]
    assert report["transport_mode"] == "in_process_simulation"
    assert [len(central["history"]), len(fed["history"])] == [training["centralized"]["budget"], training["fedavg"]["budget"]]
    assert all(models[f"local_{i:02d}"]["history"] == [] for i in range(5))
    summary = json.loads((bundle.ROOT / bundle.active_bundle_dir() / "fedavg" / "summary.json").read_text(encoding="utf-8"))
    assert sum(r["duration_sec"] for r in fed["history"]) == pytest.approx(sum(h["seconds"] for h in summary["history"]))
    payloads = {c[key] for r in fed["history"] for c in r["clients"] for key in ("payload_down_bytes", "payload_up_bytes")}
    assert len(payloads) == 1 and payloads.pop() > 0
    assert report["delta_l2_recorded"] is False
    for row in fed["history"]:
        assert sum(c["weight_p_k"] for c in row["clients"]) == pytest.approx(1)
        assert all(c["delta_l2"] is None for c in row["clients"])
    splits = {s["name"]: s for s in report["release"]["splits"]}
    assert splits["train"]["count"] == bundle.shard_total(manifest)
    assert splits["val"]["by_source"] == bundle.validation_counts(manifest)
    assert splits["legacy_diagnostic_test"]["by_source"] == bundle.holdout(manifest)["counts"]
    assert all(s["count"] == sum(s["by_source"].values()) for s in splits.values())
    validation = json.loads((bundle.ROOT / bundle.active_bundle_dir() / "fedavg" / "validation_metrics.json").read_text(encoding="utf-8"))
    assert fed["validation_overall"]["top1_accuracy"] == pytest.approx(validation["validation"]["overall"]["top1_accuracy"])


@pytest.mark.parametrize("tamper", ["checkpoint", "runtime", "client", "weight", "split", "source", "drift_claimed", "drift_flag_flipped"])
def test_rejects_misleading_report(source, tamper):
    evidence, report = source
    bad = copy.deepcopy(report)
    fed = next(m for m in bad["models"] if m["model_id"] == "fedavg")
    if tamper == "checkpoint":
        bad["checkpoint_sha256"]["fedavg"] = "0" * 64
    elif tamper == "runtime":
        bad["transport_mode"] = "flower"
    elif tamper == "client":
        fed["history"][0]["clients"].pop()
    elif tamper == "weight":
        fed["history"][0]["clients"][0]["weight_p_k"] = 1
    elif tamper == "drift_claimed":
        # gói khai chưa ghi độ lệch nhưng báo cáo có số: web không được vẽ số không có nguồn
        fed["history"][0]["clients"][0]["delta_l2"] = 1.0
    elif tamper == "drift_flag_flipped":
        bad["delta_l2_recorded"] = True
    elif tamper == "split":
        next(s for s in bad["release"]["splits"] if s["name"] == "val")["by_source"]["plantdoc"] += 1
    else:
        bad["sources"][next(iter(bad["sources"]))] = "0" * 64
    with pytest.raises(ValueError):
        training_report(evidence, lambda _: bad)


def test_recorded_drift_is_validated_when_the_bundle_has_it(source):
    """Gói có độ lệch ghi lại (gói cũ trong git, hoặc lần train sau): giá trị phải hữu hạn và độ lệch phần học được không vượt tổng."""
    evidence, report = source
    good = copy.deepcopy(report)
    good["delta_l2_recorded"] = True
    fed = next(m for m in good["models"] if m["model_id"] == "fedavg")
    for row in fed["history"]:
        for client in row["clients"]:
            client["delta_l2"], client["delta_l2_learnable"] = 2.0, 1.0
    assert training_report(evidence, lambda _: good)["delta_l2_recorded"] is True
    fed["history"][0]["clients"][0]["delta_l2_learnable"] = 4.0
    with pytest.raises(ValueError):
        training_report(evidence, lambda _: good)
    fed["history"][0]["clients"][0]["delta_l2_learnable"] = None
    with pytest.raises(ValueError):
        training_report(evidence, lambda _: good)


def test_missing_and_blocked_report_displays_no_training_numbers():
    assert training_report({}, lambda _: None)["available"] is False
    result = evidence_service.summary({"available": False, "reason": "registry unavailable"})
    assert result["training"] == {"available": False, "reason": "registry unavailable"}


def test_projection_serves_centralized_history_and_holdout_per_class(source):
    evidence, _ = source
    manifest, training = _bundle_training()
    summary = evidence_service.summary(evidence)
    central = next(m for m in summary["training"]["models"] if m["model_id"] == "centralized")
    assert len(central["history"]) == training["centralized"]["budget"] and central["status"] == "completed"
    assert all("duration_sec" not in row for row in central["history"])
    assert all("validation_overall" in m for m in summary["training"]["models"])
    by = {m["model_id"]: m for m in summary["holdout"]["models"]}
    report = evidence_service._load_json(evidence_service.reg.HOLDOUT_EVAL_JSON)
    plantdoc_images = bundle.holdout(manifest)["counts"]["plantdoc"]
    for mid in ("centralized", "fedavg"):
        per_class = {r["class"]: r for r in by[mid]["plantdoc_per_class"]}
        assert sum(r["support"] for r in per_class.values()) == plantdoc_images
        assert sum(r["correct"] for r in per_class.values()) == report["models"][mid]["by_source"]["plantdoc"]["correct"]

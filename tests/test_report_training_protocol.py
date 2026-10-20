"""Training report must keep runtime, split and checkpoint provenance separate."""
import copy

import pytest

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


def test_report_contains_verified_history_configs_and_release(source):
    evidence, report = source
    models = {m["model_id"]: m for m in report["models"]}
    assert len(models) == 7
    central, fed = models["centralized"], models["fedavg"]
    assert (central["selected_step"], central["completed_steps"]) == (22, 83)
    assert (fed["selected_step"], fed["completed_steps"]) == (27, 32)
    assert central["config"]["optimizer"] == "AdamW"
    assert fed["config"]["client_optimizer"] == "SGD"
    assert report["transport_mode"] == "in_process_simulation"
    assert [len(central["history"]), len(fed["history"])] == [83, 27]
    assert [models[f"local_{i:02d}"]["selected_step"] for i in range(5)] == [32, 37, 43, 11, 46]
    assert all(models[f"local_{i:02d}"]["config"] == {} for i in range(5))
    assert sum(r["duration_sec"] for r in fed["history"]) == pytest.approx(5313.32)
    assert sum(c["payload_down_bytes"] + c["payload_up_bytes"] for r in fed["history"] for c in r["clients"]) == 1694504880
    for row in fed["history"]:
        assert sum(c["weight_p_k"] for c in row["clients"]) == pytest.approx(1)
    splits = {s["name"]: s for s in report["release"]["splits"]}
    assert [splits[s]["count"] for s in ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine")] == [37238, 4613, 4225, 11015, 6120]
    assert splits["train"]["by_source"]["plantdoc"] == 2115
    assert fed["validation_overall"]["top1_accuracy"] == pytest.approx(0.9044006069802731)


@pytest.mark.parametrize("tamper", ["checkpoint", "runtime", "client", "weight", "split", "source"])
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
    elif tamper == "split":
        next(s for s in bad["release"]["splits"] if s["name"] == "val")["by_source"]["plantdoc"] = 172
    else:
        bad["sources"][next(iter(bad["sources"]))] = "0" * 64
    with pytest.raises(ValueError):
        training_report(evidence, lambda _: bad)


def test_missing_and_blocked_report_displays_no_training_numbers():
    assert training_report({}, lambda _: None)["available"] is False
    result = evidence_service.summary({"available": False, "reason": "registry unavailable"})
    assert result["training"] == {"available": False, "reason": "registry unavailable"}


def test_projection_serves_centralized_history_and_holdout_per_class(source):
    evidence, _ = source
    summary = evidence_service.summary(evidence)
    central = next(m for m in summary["training"]["models"] if m["model_id"] == "centralized")
    assert len(central["history"]) == 83 and central["status"] == "CONVERGED_EARLY"
    assert central["train_accuracy_by_source"]["plantdoc"] == pytest.approx(0.9967)
    drops = [r["epoch"] for prev, r in zip(central["history"], central["history"][1:]) if r["lr"] < prev["lr"]]
    assert drops[0] == 19
    assert min(central["history"], key=lambda r: r["val_loss"])["epoch"] == 4
    assert all("validation_overall" in m for m in summary["training"]["models"])
    by = {m["model_id"]: m for m in summary["holdout"]["models"]}
    per_class = {m_id: {r["class"]: r for r in by[m_id]["plantdoc_per_class"]} for m_id in ("centralized", "fedavg")}
    assert sum(r["support"] for r in per_class["centralized"].values()) == 172
    assert sum(r["correct"] for r in per_class["centralized"].values()) == 109
    assert sum(r["correct"] for r in per_class["fedavg"].values()) == 86
    mosaic = "Tomato___Tomato_mosaic_virus"
    assert (per_class["centralized"][mosaic]["correct"], per_class["fedavg"][mosaic]["correct"]) == (32, 17)

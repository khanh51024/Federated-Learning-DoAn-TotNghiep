from __future__ import annotations

import json

import pytest

from src import registry as reg
from web.app import get_comparison_table, get_metrics, get_system_status


def test_comparison_table_does_not_expose_unverified_historical_scores_as_current_results():
    result = get_comparison_table()
    assert result["available"] is True
    assert result["scientific_stage2_complete"] is False
    assert len(result["rows"]) == 14
    assert {row["source"] for row in result["rows"]} == {"plantdoc", "plantvillage"}
    assert {row["seed"] for row in result["rows"]} == {42}
    assert all(row["protocol_id"] == result["protocol_id"] for row in result["rows"])
    assert all(row["checkpoint_sha256"] for row in result["rows"])


def test_system_status_marks_canonical_run_unverified():
    status = get_system_status()
    assert status["canonical_run"]["scientific_valid"] is False
    assert status["canonical_run"]["status_label"] == "UNVERIFIED"


def test_canonical_metrics_endpoint_carries_scientific_valid_false():
    result = get_metrics("centralized", source="canonical")
    assert isinstance(result, dict)
    assert result["source"] == "canonical"
    assert result["scientific_valid"] is False
    assert "unverified" in result["scientific_reason"].lower()


@pytest.mark.parametrize("mode, run_dir, stem", [
    ("centralized", "centralized_demo", "centralized_results"),
    ("federated", "fedavg_demo", "fedavg_results"),  # routers/training.py ghi FedAvg mô phỏng vào fedavg_demo
])
def test_web_run_metrics_are_found_where_the_training_router_writes_them(monkeypatch, tmp_path, mode, run_dir, stem):
    metrics = tmp_path / "web_runs" / run_dir / "metrics"
    metrics.mkdir(parents=True)
    (metrics / f"{stem}.json").write_text(json.dumps({"experiment": mode, "history": []}), encoding="utf-8")
    monkeypatch.setattr(reg, "WEB_RUNS_DIR", str(tmp_path / "web_runs"))
    monkeypatch.setattr(reg, "DEMO_RUNS_DIR", str(tmp_path / "no_demo_runs"))

    result = get_metrics(mode, source="demo")
    assert isinstance(result, dict), result
    assert result["experiment"] == mode
    assert result["source"] == "web_run" and result["scientific_valid"] is False

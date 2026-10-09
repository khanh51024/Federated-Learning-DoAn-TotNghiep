from __future__ import annotations

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

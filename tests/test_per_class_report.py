"""Báo cáo theo lớp suy ra từ số liệu bundle đã công bố (không cần ảnh)."""
from __future__ import annotations

import json

import pytest

from scripts import report_per_class as rpc
from src import registry as reg


@pytest.fixture(scope="module")
def report() -> dict:
    return rpc.build_report()


def test_published_numbers_are_internally_consistent(report):
    for model, sources in report["models"].items():
        for source, info in sources.items():
            assert all(info["consistency"].values()), (model, source, info["consistency"])


def test_error_breakdown_adds_up_to_the_sample_count(report):
    for sources in report["models"].values():
        for info in sources.values():
            breakdown = info["error_breakdown"]
            assert sum(breakdown.values()) == info["count"]


def test_matches_numbers_in_the_registry(report):
    registry = {m["model_id"]: m for m in reg.iter_models()}
    fed = registry["demo_fedavg_v7"]["metrics"]
    pd = report["models"]["FedAvg v7"]["plantdoc"]
    assert pd["top1_accuracy"] == fed["plantdoc"]["top1_accuracy"]
    assert pd["count"] == fed["plantdoc"]["count"] == 269
    assert pd["high_confidence_wrong"]["count"] == fed["overall"]["high_confidence_wrong_count_0_9"] - \
        report["models"]["FedAvg v7"]["plantvillage"]["high_confidence_wrong"]["count"]


def test_class_table_is_sorted_by_fedavg_plantdoc_recall(report):
    recalls = [row["fedavg_plantdoc_recall"] for row in report["fedavg_plantdoc_class_table"]]
    assert recalls == sorted(recalls)
    assert report["fedavg_plantdoc_class_table"][0]["fedavg_plantdoc_recall"] == min(recalls)
    weak = set(report["summary"]["weak_classes_fedavg"])
    assert weak == {r["class"] for r in report["fedavg_plantdoc_class_table"] if r["fedavg_plantdoc_recall"] < rpc.WEAK_RECALL}


def test_report_states_what_it_cannot_claim(report):
    assert "not locally re-evaluated" in report["evidence"]
    assert any("confusion matrix" in item for item in report["not_available_without_images"])
    assert any("calibration" in item for item in report["not_available_without_images"])


def test_committed_report_matches_regeneration(report):
    committed = json.loads((reg.ROOT / reg.PER_CLASS_REPORT_JSON).read_text(encoding="utf-8"))
    assert committed == report, "artifacts/plantdoc_per_class_report.json cũ — chạy scripts/update_status.py"

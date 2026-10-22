"""Báo cáo độ chính xác theo lớp mà client có / không có ảnh học (scripts/report_client_fairness.py) và phần `fairness` của /api/evidence/summary."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts import report_client_fairness as rf
from web.routers.inference import _gated_evidence
from web.services import evidence_service as es


def test_analyze_splits_images_by_whether_the_client_has_the_class():
    targets = np.array([0, 0, 1, 2, 2, 2])
    eye = np.eye(3, dtype=np.float32)
    logits = {
        "centralized": eye[targets],                                   # đúng hết
        "fedavg": eye[[0, 1, 1, 2, 0, 2]],                             # đúng ảnh 0, 2, 3, 5
        "local_00": eye[[0, 0, 0, 0, 0, 0]],                           # chỉ biết lớp 0
    }
    present = {"client_00": np.array([True, False, False])}
    out = rf.analyze(targets, logits, {"client_00": "local_00"}, present)["client_00"]
    assert out["classes_with_images"] == {"n": 2, "centralized": 2, "fedavg": 1, "local": 2}
    assert out["classes_without_images"] == {"n": 4, "centralized": 4, "fedavg": 3, "local": 0}


@pytest.fixture(scope="module")
def evidence() -> dict:
    ev = _gated_evidence()
    if not ev.get("available"):
        pytest.skip(f"bundle đối chứng chưa khả dụng: {ev.get('reason')}")
    return ev


def test_fairness_section_is_consistent_with_the_bundle(evidence):
    section = es.summary(evidence)["fairness"]
    if not section["available"]:
        pytest.skip(section["reason"])
    totals = {"val": sum(evidence["validation_counts"].values()), "test": sum(es.reg.HOLDOUT_COUNTS.values())}
    for name, block in section["splits"].items():
        assert block["available"] is True, (name, block)
        assert set(block["by_client"]) == set(evidence["shard_counts"])
        for client, row in block["by_client"].items():
            assert 0 < row["n_classes_with_images"] <= section["classes_total"]
            # ảnh của tập đánh giá chia đúng làm hai phần: lớp client có ảnh và lớp client thiếu ảnh
            assert row["classes_with_images"]["n"] + row["classes_without_images"]["n"] == totals[name], (name, client)
            for group in ("classes_with_images", "classes_without_images"):
                assert all(0.0 <= v <= 1.0 for v in (row[group]["centralized"], row[group]["fedavg"], row[group]["local"]) if v is not None)


def test_fairness_is_withheld_when_identity_or_counts_differ(evidence, monkeypatch):
    original = es._load_json(es.reg.CLIENT_FAIRNESS_JSON)
    if original is None:
        pytest.skip("chưa có báo cáo công bằng giữa client")
    forged = copy.deepcopy(original)
    forged["checkpoint_sha256"]["fedavg"] = "0" * 64
    monkeypatch.setattr(es, "_load_json", lambda rel: forged if rel == es.reg.CLIENT_FAIRNESS_JSON else es._JSON_CACHE.get(rel, (None, None))[1])
    assert es.summary(evidence)["fairness"]["available"] is False

    forged = copy.deepcopy(original)
    block = forged["splits"]["test"]["by_client"]["client_00"]["classes_with_images"]
    block["n"] += 1                                   # tổng ảnh không còn bằng số ảnh của tập đánh giá
    monkeypatch.setattr(es, "_load_json", lambda rel: forged if rel == es.reg.CLIENT_FAIRNESS_JSON else es._JSON_CACHE.get(rel, (None, None))[1])
    section = es.summary(evidence)["fairness"]
    assert section["available"] is True and section["splits"]["test"]["available"] is False
    assert section["splits"]["val"]["available"] is True


def test_fairness_is_withheld_with_the_rest_when_the_bundle_gate_fails():
    assert es.summary({"available": False, "reason": "x"})["fairness"] == {"available": False, "reason": "x"}

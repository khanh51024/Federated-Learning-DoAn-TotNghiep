"""Cổng pre_demo_check phải kiểm gói web đang dùng (v4b), không kiểm slot Kaggle cũ; kết quả ghi vào tmp_path, không đè artifact đang track."""
from __future__ import annotations

import json
import sys

import pytest

from scripts import pre_demo_check as pdc


@pytest.fixture()
def report(tmp_path, monkeypatch):
    if pdc.bundle_context() is None:
        pytest.skip("gói web đang dùng không dựng từ train_v4: cổng dùng cách kiểm cũ")
    out = tmp_path / "pre_demo_check.json"
    monkeypatch.setattr(pdc, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["pre_demo_check.py"])
    assert pdc.main() == 0
    return {c["name"]: c for c in json.loads(out.read_text(encoding="utf-8"))["checks"]}


def test_release_data_gate_replaces_scratch_dataset_gate(report):
    assert report["release_data"]["status"] == "pass"
    assert "plantvillage_real_data" not in report


def test_provenance_and_multiseed_follow_the_active_bundle(report):
    assert report["canonical_provenance"]["status"] == "pass"
    assert "pipeline Kaggle cũ" in report["canonical_provenance"]["detail"]
    assert report["multiseed_real_summary"]["status"] == "pass"
    # ngưỡng đặt trước chưa đạt phải còn hiện trong chi tiết, không bị che
    assert "ngưỡng đặt trước" in report["multiseed_real_summary"]["detail"]


def test_network_gate_is_not_loosened(report):
    # Chỉ được "pass" khi có run Flower dữ liệu thật + số đo giữ riêng thật (smoke kỹ thuật không đủ)
    network = report["network_global_artifact"]
    if network["status"] == "pass":
        assert "real" in network["detail"]

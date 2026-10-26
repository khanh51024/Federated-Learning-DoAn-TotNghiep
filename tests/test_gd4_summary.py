"""Phần `gd4` của /api/evidence/summary (nén trọng số khi truyền và lượng tử hóa int8, scripts/eval_gd4_compression.py) và tab 4.6 của chương Kết quả."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from web.routers.inference import _gated_evidence
from web.services import evidence_service as es

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JS = (ROOT / "web" / "static" / "js" / "pages" / "results.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def evidence() -> dict:
    ev = _gated_evidence()
    if not ev.get("available"):
        pytest.skip(f"bundle đối chứng chưa khả dụng: {ev.get('reason')}")
    return ev


def _with_report(monkeypatch, report):
    real = es._load_json
    monkeypatch.setattr(es, "_load_json", lambda rel: report if rel == es.reg.GD4_COMPRESSION_JSON else real(rel))


def test_gd4_section_is_consistent_with_the_bundle(evidence):
    section = es.summary(evidence)["gd4"]
    if not section["available"]:
        pytest.skip(section["reason"])
    total = sum(es.reg.HOLDOUT_COUNTS.values())
    assert section["count"] == total
    assert 0 < section["plantdoc_count"] < section["count"] - section["plantvillage_count"] < total
    assert set(section["models"]) == set(es.MAIN_MODELS)
    for model in section["models"].values():
        assert [r["scheme"] for r in model["transport"]] == list(es._GD4_TRANSPORT)
        assert [r["variant"] for r in model["onnx"]] == list(es._GD4_ONNX)
        base = model["transport"][0]
        assert base["payload_ratio"] == 1.0 and base["predictions_changed"] == 0
        assert model["onnx"][0]["predictions_changed"] == 0, "ONNX fp32 phải đoán y hệt PyTorch fp32"
        for row in model["transport"] + model["onnx"]:
            assert 0 <= row["predictions_changed"] <= total
            assert all(0.0 <= row[k] <= 1.0 for k in ("top1", "plantvillage_top1", "field_top1", "plantdoc_top1"))
        # nén phải nhỏ hơn bản gốc
        assert all(r["payload_bytes"] < base["payload_bytes"] for r in model["transport"][1:])
        assert all(r["bytes"] < model["onnx"][0]["bytes"] for r in model["onnx"][1:])


def test_gd4_is_withheld_when_identity_or_counts_differ(evidence, monkeypatch):
    original = es._load_json(es.reg.GD4_COMPRESSION_JSON)
    if original is None:
        pytest.skip("chưa có báo cáo GĐ4")

    forged = copy.deepcopy(original)
    forged["models"]["fedavg"]["checkpoint_sha256"] = "0" * 64     # số đo của một checkpoint khác
    _with_report(monkeypatch, forged)
    assert es.summary(evidence)["gd4"]["available"] is False

    forged = copy.deepcopy(original)
    forged["release_manifest_sha256"] = "0" * 64
    _with_report(monkeypatch, forged)
    assert es.summary(evidence)["gd4"]["available"] is False

    forged = copy.deepcopy(original)
    forged["models"]["centralized"]["weights_transport"]["fp16"]["n"] -= 1   # một dòng đo trên tập khác
    _with_report(monkeypatch, forged)
    section = es.summary(evidence)["gd4"]
    assert section["available"] is False and "GĐ4" in section["reason"]


def test_gd4_is_withheld_with_the_rest_when_the_bundle_gate_fails():
    assert es.summary({"available": False, "reason": "x"})["gd4"] == {"available": False, "reason": "x"}


def test_compression_tab_is_wired_and_reads_numbers_from_the_api():
    tabs = re.findall(r'role="tab" id="subtab-(\w+)"[^>]*aria-controls="result-(\w+)"', INDEX_HTML)
    assert ("compression", "compression") in tabs
    assert 'id="result-compression" class="subpanel" role="tabpanel" aria-labelledby="subtab-compression"' in INDEX_HTML
    assert "compression: buildCompression" in RESULTS_JS and 'sectionHead("4.6"' in RESULTS_JS
    block = RESULTS_JS[RESULTS_JS.index("// ───────── Tab Nén và lượng tử hóa"):RESULTS_JS.index("const BUILDERS")]
    assert "ctx.summary?.gd4" in block
    literal = re.findall(r"(?<![\w$.{])\d{1,3}[,.]\d+\s*(?:%|MB|ms)", block)
    assert not literal, f"số đo gõ cứng trong tab GĐ4: {literal}"

"""Phân bố lớp theo client (giải thích vì sao Local-only sai) và cách web dùng nó."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scripts import report_client_shards as rcs
from src import bundle, registry as reg
from web.app import app
from web.demo_evidence import load_evidence
from web.services import model_service

CLASS = "Tomato___Bacterial_spot"


def published_nk() -> dict:
    """n_k của từng client theo gói đang dùng (import_manifest + summary đã kiểm hash), không viết cứng theo một đợt huấn luyện."""
    evidence = load_evidence()
    assert evidence["available"], evidence.get("reason")
    return dict(evidence["shard_counts"])


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads((reg.ROOT / reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.open(reg.ROOT / reg.SAMPLE_IMAGE).convert("RGB").save(buffer, "JPEG")
    return buffer.getvalue()


def test_report_is_consistent_and_matches_published_bundle(report):
    nk = published_nk()
    assert report["consistent"] is True
    assert all(v for k, v in report["checks"].items() if k != "release_manifest_has_shard_hashes_for_scheme")
    assert {n: c["n_k"] for n, c in report["clients"].items()} == nk
    assert report["total_train_samples"] == sum(nk.values()) == bundle.shard_total(bundle.load_manifest())
    assert sum(report["totals_all_clients"].values()) == report["total_train_samples"]
    assert len(report["classes"]) == 38


def test_the_explanation_the_demo_relies_on(report):
    """Với phân vùng lệch nhãn (Dirichlet α = 0,1), nhiều lớp dồn gần hết vào MỘT client và vắng ở các client còn lại."""
    totals = report["totals_all_clients"]
    concentrated = [c for c in report["classes"] if totals[c] and max(cl["per_class"][c] for cl in report["clients"].values()) / totals[c] > 0.9]
    assert len(concentrated) >= 8
    counts = {n: c["per_class"][CLASS] for n, c in report["clients"].items()}
    assert min(counts.values()) < 0.1 * max(counts.values())       # lớp này vắng gần hết ở ít nhất một client


def test_regeneration_matches_committed_report(report):
    if not (reg.abs_path(reg.RELEASE_DIR) / "manifests/train.jsonl").is_file():
        pytest.skip("máy không có dataset (data/dataset_sources) — không sinh lại được")
    assert rcs.build_report() == report


def test_shard_context_for_local_and_fedavg(report):
    top3 = [{"class": CLASS, "confidence": 50.0}, {"class": "Tomato___healthy", "confidence": 10.0}]
    per = {n: c["per_class"][CLASS] for n, c in report["clients"].items()}
    total = report["totals_all_clients"][CLASS]
    empty = min(per, key=per.get)                      # client có ít ảnh nhất của lớp này
    full = max(per, key=per.get)                       # client có nhiều ảnh nhất
    local = model_service.shard_context(f"local_{empty[-2:]}", top3)
    assert local["client"] == empty and local["client_n_k"] == report["clients"][empty]["n_k"]
    first = local["classes"][0]
    assert first["client_count"] == per[empty] and first["all_clients_count"] == total
    assert first["client_share_pct"] == round(100 * per[empty] / total, 1)

    local2 = model_service.shard_context(f"local_{full[-2:]}", top3)["classes"][0]
    assert local2["client_count"] == per[full] and local2["client_share_pct"] == round(100 * per[full] / total, 1)

    fed = model_service.shard_context("fedavg", top3)
    assert fed["client"] is None and fed["client_n_k"] is None
    assert fed["classes"][0]["client_count"] is None and fed["classes"][0]["all_clients_count"] == total


def test_shard_context_ignores_unknown_classes_and_unknown_clients():
    ctx = model_service.shard_context("local_00", [{"class": "Khong_ton_tai", "confidence": 1.0}])
    assert ctx["classes"] == []
    assert model_service.shard_context("local_99", [{"class": "Tomato___healthy", "confidence": 1.0}]) is None


def test_endpoint_returns_context_and_stays_up_without_report(client, monkeypatch):
    response = client.post("/api/inference?federated_source=local_02", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    ctx = body["shard_context"]
    assert ctx["client"] == "client_02" and ctx["client_n_k"] == published_nk()["client_02"]
    assert [c["class"] for c in ctx["classes"]] == [t["class"] for t in body["federated"]["top3"]]

    monkeypatch.setattr(model_service, "_shard_report", lambda: None)
    degraded = client.post("/api/inference?federated_source=local_02", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert degraded.status_code == 200
    assert degraded.json()["shard_context"] is None


def test_inconsistent_report_is_not_served(tmp_path, monkeypatch):
    bad = tmp_path / "shards.json"
    bad.write_text(json.dumps({"consistent": False, "clients": {}, "totals_all_clients": {}}), encoding="utf-8")
    monkeypatch.setattr(reg, "CLIENT_SHARD_COUNTS_JSON", str(bad))
    monkeypatch.setattr(reg, "abs_path", lambda rel: bad)
    model_service._SHARD_CACHE.clear()
    try:
        assert model_service.shard_context("local_00", [{"class": "Tomato___healthy", "confidence": 1.0}]) is None
    finally:
        model_service._SHARD_CACHE.clear()


def test_true_label_is_listed_first_even_when_not_in_top3():
    report = json.loads((reg.ROOT / reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))
    top3 = [{"class": "Tomato___Target_Spot", "confidence": 95.0}]
    ctx = model_service.shard_context("local_01", top3, "Tomato___Bacterial_spot")
    first = ctx["classes"][0]
    assert first["class"] == "Tomato___Bacterial_spot"
    assert first["is_true_label"] is True and first["in_top3"] is False
    assert first["client_count"] == report["clients"]["client_01"]["per_class"][CLASS]
    assert first["all_clients_count"] == report["totals_all_clients"][CLASS]
    assert [c["class"] for c in ctx["classes"]] == ["Tomato___Bacterial_spot", "Tomato___Target_Spot"]
    assert ctx["classes"][1]["is_true_label"] is False and ctx["classes"][1]["in_top3"] is True


def test_true_label_equal_to_prediction_is_not_duplicated():
    top3 = [{"class": "Tomato___Bacterial_spot", "confidence": 90.0}]
    ctx = model_service.shard_context("local_02", top3, "Tomato___Bacterial_spot")
    assert len(ctx["classes"]) == 1
    assert ctx["classes"][0]["is_true_label"] is True and ctx["classes"][0]["in_top3"] is True


def test_endpoint_validates_true_class(client):
    files = {"file": ("leaf.jpg", _jpeg(), "image/jpeg")}
    ok = client.post("/api/inference?federated_source=local_01&true_class=Tomato___Bacterial_spot", files=files)
    assert ok.status_code == 200
    assert ok.json()["shard_context"]["classes"][0]["class"] == "Tomato___Bacterial_spot"
    bad = client.post("/api/inference?federated_source=local_01&true_class=Khong_co", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert bad.status_code == 400

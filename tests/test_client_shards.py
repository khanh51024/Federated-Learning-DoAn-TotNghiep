"""Phân bố lớp theo client (giải thích vì sao Local-only sai) và cách web dùng nó."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scripts import report_client_shards as rcs
from src import registry as reg
from web.app import app
from web.services import model_service

PUBLISHED_NK = {"client_00": 5434, "client_01": 7858, "client_02": 5376, "client_03": 8495, "client_04": 10075}


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
    assert report["consistent"] is True
    assert all(v for k, v in report["checks"].items() if k != "release_manifest_has_shard_hashes_for_scheme")
    assert {n: c["n_k"] for n, c in report["clients"].items()} == PUBLISHED_NK
    assert report["total_train_samples"] == sum(PUBLISHED_NK.values()) == 37238
    assert sum(report["totals_all_clients"].values()) == 37238
    assert len(report["classes"]) == 38


def test_the_explanation_the_demo_relies_on(report):
    bacterial = "Tomato___Bacterial_spot"
    counts = {n: c["per_class"][bacterial] for n, c in report["clients"].items()}
    assert counts["client_02"] == 1549
    assert counts["client_01"] == 0
    # mỗi lớp bệnh cà chua dồn chủ yếu vào MỘT client
    assert max(counts.values()) / report["totals_all_clients"][bacterial] > 0.9


def test_regeneration_matches_committed_report(report):
    if not (reg.abs_path(reg.RELEASE_V3_DIR) / "manifests/train.jsonl").is_file():
        pytest.skip("máy không có dataset (data/dataset_sources) — không sinh lại được")
    assert rcs.build_report() == report


def test_shard_context_for_local_and_fedavg(report):
    top3 = [{"class": "Tomato___Bacterial_spot", "confidence": 50.0}, {"class": "Tomato___healthy", "confidence": 10.0}]
    local = model_service.shard_context("local_01", top3)
    assert local["client"] == "client_01" and local["client_n_k"] == 7858
    first = local["classes"][0]
    assert first["client_count"] == 0 and first["all_clients_count"] == 1597 and first["client_share_pct"] == 0.0

    local2 = model_service.shard_context("local_02", top3)["classes"][0]
    assert local2["client_count"] == 1549 and local2["client_share_pct"] == round(100 * 1549 / 1597, 1)

    fed = model_service.shard_context("fedavg", top3)
    assert fed["client"] is None and fed["client_n_k"] is None
    assert fed["classes"][0]["client_count"] is None and fed["classes"][0]["all_clients_count"] == 1597


def test_shard_context_ignores_unknown_classes_and_unknown_clients():
    ctx = model_service.shard_context("local_00", [{"class": "Khong_ton_tai", "confidence": 1.0}])
    assert ctx["classes"] == []
    assert model_service.shard_context("local_99", [{"class": "Tomato___healthy", "confidence": 1.0}]) is None


def test_endpoint_returns_context_and_stays_up_without_report(client, monkeypatch):
    response = client.post("/api/inference?federated_source=local_02", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    ctx = body["shard_context"]
    assert ctx["client"] == "client_02" and ctx["client_n_k"] == 5376
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
    top3 = [{"class": "Tomato___Target_Spot", "confidence": 95.0}]
    ctx = model_service.shard_context("local_01", top3, "Tomato___Bacterial_spot")
    first = ctx["classes"][0]
    assert first["class"] == "Tomato___Bacterial_spot"
    assert first["is_true_label"] is True and first["in_top3"] is False
    assert first["client_count"] == 0 and first["all_clients_count"] == 1597
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

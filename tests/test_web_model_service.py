"""Cổng inference một cửa: web chỉ dùng được model đã qua cổng web_visible của registry."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import registry as reg
from web.app import app
from web.services import model_service


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.open(reg.ROOT / reg.SAMPLE_IMAGE).convert("RGB").save(buffer, "JPEG")
    return buffer.getvalue()


def test_models_endpoint_lists_only_gated_models(client):
    payload = client.get("/api/models").json()
    ids = {m["model_id"] for m in payload["models"]}
    assert payload["count"] == len(ids) >= 8
    assert {"demo_centralized_v6", "demo_fedavg_v7", "plantvillage_traced_reference"} <= ids
    for entry in payload["models"]:
        assert entry["role"] in (reg.ROLE_CURRENT, reg.ROLE_REFERENCE)
        assert entry["display_label"]
    registry_ids = {m["model_id"] for m in reg.iter_models()}
    hidden = registry_ids - ids
    assert {"fedavg_canonical", "fedavg_best_smoke", "network_federated_global"} <= hidden


def test_predict_rejects_unknown_and_gated_models():
    image = Image.open(reg.ROOT / reg.SAMPLE_IMAGE).convert("RGB")
    with pytest.raises(model_service.UnknownModel):
        model_service.predict("khong_ton_tai", image)
    with pytest.raises(model_service.ModelNotAllowed) as excinfo:
        model_service.predict("fedavg_best_smoke", image)
    assert excinfo.value.reasons, "phải nêu lý do bị chặn"


def test_predict_fails_closed_when_registry_missing(monkeypatch):
    monkeypatch.setattr(model_service, "registry_models", lambda: {})
    with pytest.raises(model_service.RegistryUnavailable):
        model_service.predict("demo_fedavg_v7", Image.new("RGB", (8, 8)))


def test_inference_endpoint_labels_results_with_registry_context(client):
    response = client.post("/api/inference?federated_source=fedavg", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["federated"]["registry"]["model_id"] == "demo_fedavg_v7"
    assert body["centralized"]["registry"]["model_id"] == "demo_centralized_v6"
    assert "Demo" in body["federated"]["registry"]["display_label"]
    assert len(body["federated"]["top3"]) == 3
    assert body["scientific_status"]["overall_scientific_valid"] is False


def test_reference_endpoint_goes_through_registry(client):
    response = client.post("/api/inference/plantvillage", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["registry"]["role"] == reg.ROLE_REFERENCE


def test_unknown_federated_source_is_rejected(client):
    response = client.post("/api/inference?federated_source=bogus", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 400


def test_gated_model_returns_503_with_reasons(client, monkeypatch):
    models = {m["model_id"]: dict(m) for m in reg.iter_models()}
    models["demo_fedavg_v7"].update(web_visible=False, web_gate_failures=["1.hash: checkpoint_status=hash_mismatch"])
    monkeypatch.setattr(model_service, "registry_models", lambda: models)
    response = client.post("/api/inference?federated_source=fedavg", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 503
    assert "hash_mismatch" in response.json()["detail"]
    assert model_service.status_models_available()["fedavg"] is False


def test_unknown_internal_model_is_a_server_state_problem_not_a_client_error(client, monkeypatch):
    models = {k: v for k, v in model_service.registry_models().items() if k != "demo_centralized_v6"}
    monkeypatch.setattr(model_service, "registry_models", lambda: models)
    response = client.post("/api/inference?federated_source=fedavg", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 503


def test_registry_read_errors_do_not_become_500(monkeypatch):
    def boom():
        raise PermissionError("registry.json đang bị ghi dở")

    monkeypatch.setattr(reg, "iter_models", boom)
    assert model_service.registry_models() == {}


def test_weights_inspect_obeys_the_same_gate_as_inference(client, monkeypatch):
    models = {m["model_id"]: dict(m) for m in reg.iter_models()}
    assert client.get("/api/weights/inspect").json()["available"] is True
    models["demo_fedavg_v7"].update(web_visible=False, checkpoint_status="hash_mismatch")
    monkeypatch.setattr(model_service, "registry_models", lambda: models)
    assert client.get("/api/weights/inspect").json()["available"] is False


def test_weights_inspect_rejects_checkpoint_whose_hash_changed(client, monkeypatch):
    models = {m["model_id"]: dict(m) for m in reg.iter_models()}
    models["demo_fedavg_v7"]["file_sha256"] = "0" * 64
    monkeypatch.setattr(model_service, "registry_models", lambda: models)
    assert client.get("/api/weights/inspect").json()["available"] is False


def test_reference_model_hash_is_rechecked_on_every_use(client, monkeypatch):
    monkeypatch.setattr(reg, "PLANTVILLAGE_TRACED_SHA256", "0" * 64)
    response = client.post("/api/inference/plantvillage", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code in (400, 503)
    assert "hash" in response.json()["detail"].lower()

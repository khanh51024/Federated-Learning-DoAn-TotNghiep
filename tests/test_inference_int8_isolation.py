"""POST /api/inference/all với mô hình int8: lỗi của bản int8 không làm hỏng cả yêu cầu, int8 không gọi Grad-CAM, Centralized* không có shard.

Dùng bản sao gói v4b có thêm mục int8 (tests/test_int8_torchscript.py); registry thật không biết mục int8 nên `_predict` được thay bằng
đường gọi thẳng `predict_published` trên bản sao, phần còn lại của router (cô lập lỗi, Grad-CAM, shard_context) chạy thật.
"""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from src import bundle
from web import demo_evidence
from web.app import app
from web.routers import inference
from web.services import model_service
from tests.test_int8_torchscript import _add_int8_to_bundle_copy

ERROR_TEXT = "Grad-CAM không hỗ trợ mô hình int8 (TorchScript lượng tử hoá không có gradient); hãy dùng mô hình fp32 gốc"


@pytest.fixture(scope="module")
def bundle_copy(tmp_path_factory):
    return _add_int8_to_bundle_copy(tmp_path_factory.mktemp("int8_router"))


@pytest.fixture()
def jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 120, 40)).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture()
def wired(bundle_copy, monkeypatch):
    """Router nối với bản sao gói: hai mô hình (fedavg, fedavg_int8), nhật ký các lần gọi dự đoán/Grad-CAM, và công tắc làm hỏng từng mô hình."""
    calls = {"explain": [], "fail": set()}
    monkeypatch.setattr(inference, "ALL_MODELS", (("fedavg", "web_fedavg"), ("fedavg_int8", "web_fedavg_int8")))
    monkeypatch.setattr(inference, "INT8_MODELS", frozenset({"fedavg_int8"}))
    monkeypatch.setattr(inference, "_temperatures", lambda: {})

    def predict(registry_id, image, temperature=None, crop=None):
        key = registry_id.removeprefix("web_")
        if key in calls["fail"]:
            raise ValueError(f"{key}: giả lập hỏng")
        return demo_evidence.predict_published(image, key, bundle_copy, temperature, crop)

    def explain(registry_id, image, class_name):
        calls["explain"].append(registry_id)
        return {"class": class_name, "image": "data:image/jpeg;base64,AAAA", "method": "giả lập"}

    monkeypatch.setattr(inference, "_predict", predict)
    monkeypatch.setattr(inference, "_explain", explain)
    return calls


def _post(jpeg: bytes, **params):
    return TestClient(app).post("/api/inference/all", files={"file": ("leaf.jpg", jpeg, "image/jpeg")}, params=params)


def test_int8_result_carries_its_format_and_gets_no_gradcam(wired, jpeg):
    body = _post(jpeg, explain="true").json()
    assert body["order"] == ["fedavg", "fedavg_int8"]
    fp32, small = body["models"]["fedavg"], body["models"]["fedavg_int8"]
    assert fp32["ready"] and "model_format" not in fp32
    assert small["ready"] and small["model_format"] == bundle.FORMAT_TORCHSCRIPT_INT8 and small["quantization"]["fp32_of"] == "fedavg"
    assert "int8 động, chỉ lớp Linear (classifier); các lớp conv vẫn fp32" in small["quantization"]["note"]
    assert wired["explain"] == ["web_fedavg"], "chỉ mô hình fp32 được tạo Grad-CAM"
    assert body["explain"]["fedavg"]["image"].startswith("data:image/")
    assert body["explain"]["fedavg_int8"] == {"error": inference.INT8_NO_GRADCAM}
    assert "calibrated" not in fp32 and body["calibration_available"] is False   # không có nhiệt độ: chỉ có softmax thô


def test_a_broken_int8_model_only_marks_its_own_entry(wired, jpeg):
    wired["fail"].add("fedavg_int8")
    response = _post(jpeg, explain="true")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["models"]["fedavg"]["ready"] is True
    broken = body["models"]["fedavg_int8"]
    assert broken["ready"] is False and "giả lập hỏng" in broken["error"]
    assert body["explain"]["fedavg"]["image"] and body["explain"]["fedavg_int8"] == {"error": inference.INT8_NO_GRADCAM}
    assert "fedavg_int8" in body["shard_context"] and body["timing_ms"]["predict"] >= 0


def test_a_broken_fp32_model_still_fails_the_whole_request(wired, jpeg):
    wired["fail"].add("fedavg")
    response = _post(jpeg)
    assert response.status_code == 503 and "giả lập hỏng" in response.json()["detail"]


def test_int8_failure_from_the_registry_gate_is_isolated_too(wired, jpeg, monkeypatch, bundle_copy):
    def predict(registry_id, image, temperature=None, crop=None):
        if registry_id == "web_fedavg_int8":
            raise HTTPException(status_code=503, detail="web_fedavg_int8 chưa đạt cổng hiển thị")
        return demo_evidence.predict_published(image, "fedavg", bundle_copy, temperature, crop)

    monkeypatch.setattr(inference, "_predict", predict)
    body = _post(jpeg).json()
    assert body["models"]["fedavg"]["ready"] and body["models"]["fedavg_int8"]["ready"] is False
    assert "chưa đạt cổng hiển thị" in body["models"]["fedavg_int8"]["error"]


def test_explain_turns_the_value_error_of_int8_into_a_readable_http_error(monkeypatch):
    def refuse(model_id, image, class_name):
        raise ValueError(ERROR_TEXT)

    monkeypatch.setattr(model_service, "explain", refuse)
    with pytest.raises(HTTPException) as caught:
        inference._explain("web_fedavg_int8", Image.new("RGB", (8, 8)), "X")
    assert caught.value.status_code == 400 and caught.value.detail == ERROR_TEXT


def test_every_centralized_variant_is_left_out_of_the_shard_context(monkeypatch, jpeg):
    keys = ("centralized", "centralized_sgd", "centralized_int8", "fedavg", "scaffold")
    monkeypatch.setattr(inference, "ALL_MODELS", tuple((k, f"web_{k}") for k in keys))
    monkeypatch.setattr(inference, "INT8_MODELS", frozenset({"centralized_int8"}))
    monkeypatch.setattr(inference, "_temperatures", lambda: {})
    monkeypatch.setattr(inference, "_predict", lambda registry_id, image, temperature=None, crop=None: {"ready": True, "predicted_class": "X", "top3": []})
    body = _post(jpeg).json()
    assert body["order"] == list(keys)
    assert set(body["shard_context"]) == {"fedavg", "scaffold"}


def test_int8_models_of_the_active_bundle_are_read_from_its_manifest():
    expected = {s["model_id"] for s in bundle.model_specs(bundle.load_manifest()) if s["format"] == bundle.FORMAT_TORCHSCRIPT_INT8}
    assert inference.INT8_MODELS == frozenset(expected)
    assert all(key in dict(inference.ALL_MODELS) for key in expected)

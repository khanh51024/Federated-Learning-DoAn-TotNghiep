"""Chọn loại cây (crop): chỉ xét bệnh của cây đó, đánh dấu rõ là xác suất đã chuẩn hóa lại, từ chối tên cây lạ."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src import registry as reg
from web.app import app
from web.demo_evidence import crop_names, crop_of_class
from web.routers.inference import _gated_evidence

GALLERY = reg.abs_path(reg.DEMO_GALLERY_DIR)


def test_crop_helpers():
    assert crop_of_class("Tomato___Late_blight") == "Tomato"
    assert crop_of_class("Corn_(maize)___Common_rust_") == "Corn_(maize)"
    names = crop_names(["Apple___scab", "Apple___healthy", "Tomato___blight"])
    assert names == ["Apple", "Tomato"]


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not _gated_evidence().get("available"):
        pytest.skip("bundle đối chứng chưa khả dụng")
    return TestClient(app)


def _tomato_image() -> tuple[str, bytes]:
    catalog = json.loads(reg.abs_path(reg.DEMO_GALLERY_CATALOG_JSON).read_text(encoding="utf-8"))
    item = next(i for i in catalog["items"] if i["label"].startswith("Tomato___"))
    return item["file"], (GALLERY / item["file"]).read_bytes()


def _post(client, params):
    name, data = _tomato_image()
    return client.post("/api/inference/all", files={"file": (name, data, "image/jpeg")}, params=params)


def test_crop_restricts_every_model_to_the_chosen_plant(client):
    body = _post(client, {"crop": "Apple"}).json()  # ảnh cà chua nhưng người dùng chọn Apple: chỉ lớp Apple được xét
    for key, result in body["models"].items():
        assert result["predicted_class"].startswith("Apple___"), key
        assert all(t["class"].startswith("Apple___") for t in result["top3"]), key
        assert sum(t["confidence"] for t in result["top3"]) <= 100.01
        marker = result["crop_restricted"]
        assert marker["crop"] == "Apple"
        assert marker["unrestricted_top1"]["class"].startswith("Tomato___") or key.startswith("local_")
        assert "calibrated" not in result  # T học trên 38 lớp, không áp cho xác suất đã giới hạn


def test_choosing_the_right_crop_keeps_the_prediction_inside_that_crop_and_marks_it(client):
    free = _post(client, {}).json()
    chosen = _post(client, {"crop": "Tomato"}).json()
    for key in free["order"]:
        assert "crop_restricted" not in free["models"][key]
        result = chosen["models"][key]
        assert result["predicted_class"].startswith("Tomato___")
        assert result["crop_restricted"]["unrestricted_top1"] == free["models"][key]["top3"][0]
        if free["models"][key]["predicted_class"].startswith("Tomato___"):
            assert result["predicted_class"] == free["models"][key]["predicted_class"]  # đã đúng cây thì nhãn không đổi
            assert result["confidence_pct"] >= free["models"][key]["confidence_pct"] - 0.01  # chuẩn hóa lại chỉ làm tăng


def test_unknown_crop_is_rejected_without_echoing_the_input(client):
    for bad in ("Banana", "tomato", "Banana<script>"):
        response = _post(client, {"crop": bad})
        assert response.status_code == 400, bad
        assert bad not in response.text  # không phản chiếu chuỗi người dùng gửi


def test_single_class_crop_is_flagged_uncertain_when_the_model_does_not_see_that_crop(client):
    # Blueberry chỉ có một lớp: nếu không báo mức "khối xác suất gốc", kết quả sẽ luôn là 100% và không bao giờ "không chắc"
    body = _post(client, {"crop": "Blueberry"}).json()
    for key, result in body["models"].items():
        info = result["crop_restricted"]
        assert info["classes_in_crop"] == 1 and result["predicted_class"] == "Blueberry___healthy", key
        assert len(result["top3"]) == 1 and result["top3"][0]["confidence"] == 100.0
        assert info["crop_mass_pct"] < 50 and info["low_crop_mass"] is True and result["is_uncertain"] is True, key


def test_special_character_crop_names_pass_through_the_query_string(client):
    for crop in ("Corn_(maize)", "Cherry_(including_sour)", "Pepper,_bell"):
        body = _post(client, {"crop": crop}).json()
        assert all(r["predicted_class"].startswith(crop + "___") for r in body["models"].values()), crop

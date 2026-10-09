"""Contract tests for the PlantVillage reference inference on the web."""
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from web.plantvillage_reference import ROOT, predict_plantvillage
from src.data.plantvillage import get_default_transform


def test_reference_output_contract():
    img = Image.new("RGB", (320, 270), color=(80, 160, 70))
    result = predict_plantvillage(img)
    classes = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))["classes"]
    assert len(classes) == 38
    assert result["predicted_class"] in classes
    assert len(result["top3"]) == 3
    assert all(x["class"] in classes and 0 <= x["confidence"] <= 100 for x in result["top3"])
    assert result["confidence_calibrated"] is False
    assert result["checkpoint"] == "plantvillage_traced.pt"


def test_reference_preprocessing_matches_evaluation():
    import web.plantvillage_reference as module
    assert module.get_default_transform is get_default_transform
    steps = get_default_transform(is_train=False).transforms
    assert [type(step).__name__ for step in steps] == ["Resize", "CenterCrop", "ToTensor", "Normalize"]


def test_reference_endpoint():
    from fastapi.testclient import TestClient
    from web.app import app
    buffer = io.BytesIO()
    Image.new("RGB", (250, 250), "green").save(buffer, "PNG")
    response = TestClient(app).post("/api/inference/plantvillage", files={"file": ("leaf.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["model_name"].startswith("PlantVillage")


def test_reject_non_image():
    from fastapi.testclient import TestClient
    from web.app import app
    response = TestClient(app).post("/api/inference/plantvillage", files={"file": ("invalid.txt", b"not an image", "text/plain")})
    assert response.status_code == 400

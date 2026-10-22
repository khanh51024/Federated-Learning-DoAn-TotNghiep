"""POST /api/inference/all: cùng kết quả với /api/inference, hiệu chuẩn không đổi dự đoán, Grad-CAM, cache hash an toàn."""
from __future__ import annotations

import base64
import io
import json
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import bundle, registry as reg
from web import demo_evidence
from web.app import app
from web.routers.inference import _gated_evidence
from web.services import explain_service

GALLERY = reg.abs_path(reg.DEMO_GALLERY_DIR)
SOURCES = ("fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not _gated_evidence().get("available"):
        pytest.skip("bundle đối chứng chưa khả dụng")
    return TestClient(app)


def _gallery_images() -> list[tuple[str, bytes, str]]:
    catalog = json.loads(reg.abs_path(reg.DEMO_GALLERY_CATALOG_JSON).read_text(encoding="utf-8"))
    return [(item["file"], (GALLERY / item["file"]).read_bytes(), item["label"]) for item in catalog["items"]]


def test_all_endpoint_matches_the_pairwise_endpoint_on_every_gallery_image(client):
    for name, data, label in _gallery_images():
        response = client.post("/api/inference/all", files={"file": (name, data, "image/jpeg")}, params={"true_class": label})
        assert response.status_code == 200, (name, response.text)
        body = response.json()
        assert body["order"] == ["centralized", *SOURCES]
        for source in SOURCES:
            pair = client.post("/api/inference", files={"file": (name, data, "image/jpeg")},
                               params={"federated_source": source}).json()
            for key, result in (("centralized", pair["centralized"]), (source, pair["federated"])):
                assert body["models"][key]["predicted_class"] == result["predicted_class"], (name, key)
                assert body["models"][key]["top3"] == result["top3"], (name, key)
        assert set(body["shard_context"]) == set(SOURCES)
        assert body["shard_context"]["local_02"]["client"] == "client_02"


def test_calibrated_probabilities_keep_the_prediction(client):
    name, data, label = _gallery_images()[11]
    body = client.post("/api/inference/all", files={"file": (name, data, "image/jpeg")}).json()
    assert body["calibration_available"] is True
    for key, result in body["models"].items():
        calibrated = result["calibrated"]
        assert result["confidence_calibrated"] is False  # confidence_pct vẫn là softmax thô
        assert [t["class"] for t in calibrated["top3"]] == [t["class"] for t in result["top3"]], key
        assert calibrated["confidence_pct"] == calibrated["top3"][0]["confidence"]
        assert 0 < calibrated["temperature"] < 20
        assert sum(t["confidence"] for t in calibrated["top3"]) <= 100.0 + 1e-6
        if calibrated["temperature"] > 1:  # T > 1 làm phân phối phẳng hơn: xác suất cao nhất không tăng
            assert calibrated["confidence_pct"] <= result["confidence_pct"] + 1e-6


def test_explain_returns_one_gradcam_overlay_per_model(client):
    name, data, _ = _gallery_images()[0]
    body = client.post("/api/inference/all", files={"file": (name, data, "image/jpeg")}, params={"explain": "true"}).json()
    assert set(body["explain"]) == set(body["order"])
    for key, item in body["explain"].items():
        # Router cố tình nuốt lỗi Grad-CAM thành {"error": ...} để một model hỏng không mất 7 kết quả
        # dự đoán; không chặn ở đây thì test chỉ hỏng vì thiếu khóa, không cho biết nguyên nhân thật.
        assert "error" not in item, f"{key}: {item['error']}"
        assert item["class"] == body["models"][key]["predicted_class"]
        assert item["image"].startswith("data:image/jpeg;base64,")
        with Image.open(io.BytesIO(base64.b64decode(item["image"].split(",", 1)[1]))) as image:
            assert image.size == (224, 224)
    assert "explain" in body["timing_ms"]


def test_bad_inputs_are_rejected(client):
    _, data, _ = _gallery_images()[0]
    assert client.post("/api/inference/all", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    bad_label = client.post("/api/inference/all", files={"file": ("a.jpg", data, "image/jpeg")}, params={"true_class": "Nope___x"})
    assert bad_label.status_code == 400


def test_gradcam_map_is_normalised_and_overlay_has_input_size():
    import numpy as np

    cam = np.zeros((7, 7), dtype=np.float32)
    cam[3, 3] = 1.0
    image = Image.new("RGB", (300, 200), (40, 120, 40))
    out = explain_service.overlay(image, cam)
    assert out.size == (224, 224)
    centre, corner = out.getpixel((112, 112)), out.getpixel((0, 0))
    assert centre != corner  # vùng có CAM cao được tô màu, góc giữ màu gốc


def test_hash_cache_recomputes_after_the_file_changes(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(b"a" * (demo_evidence._BIG_FILE_BYTES + 10))
    first = demo_evidence._sha256_cached(path)
    path.write_bytes(b"b" * (demo_evidence._BIG_FILE_BYTES + 10))  # cùng kích thước, khác nội dung
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert demo_evidence._sha256_cached(path) != first


def test_details_and_partitions_endpoints(client):
    details = client.get("/api/evidence/details").json()
    assert details["available"] is True and len(details["classes"]) == 38
    matrix = details["confusion"]["val"]["fedavg"]["plantdoc"]
    assert len(matrix) == 38 and sum(map(sum, matrix)) == bundle.validation_counts(bundle.load_manifest())["plantdoc"]
    assert details["errors"]["available"] is True and details["errors"]["count"] > 0
    first = details["errors"]["items"][0]
    assert (reg.abs_path(reg.ERROR_GALLERY_DIR) / first["thumb"]).is_file()
    partitions = client.get("/api/evidence/partitions").json()
    assert partitions["available"] is True and partitions["trained_scheme"] == "label_alpha_0_1"
    assert partitions["schemes"]["label_alpha_0_1"]["has_trained_models"] is True
    assert {"iid", partitions["trained_scheme"]} <= set(partitions["schemes"])

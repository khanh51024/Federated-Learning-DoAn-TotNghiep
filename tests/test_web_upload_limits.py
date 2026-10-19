"""Giới hạn ảnh tải lên: tệp quá lớn / quá nhiều điểm ảnh bị từ chối bằng 413 TRƯỚC khi giải mã hay suy luận."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from web.app import app
from web.routers import inference

UPLOAD_URLS = ("/api/inference/all", "/api/inference", "/api/inference/plantvillage")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("url", UPLOAD_URLS)
def test_oversized_file_is_rejected_with_413(client: TestClient, url: str):
    blob = b"\xff\xd8" + b"0" * (inference.MAX_UPLOAD_BYTES + 10)  # nội dung không quan trọng: kích thước bị kiểm trước
    response = client.post(url, files={"file": ("big.jpg", blob, "image/jpeg")})
    assert response.status_code == 413, (url, response.status_code)
    assert "MB" in response.json()["detail"]


@pytest.mark.parametrize("url", UPLOAD_URLS)
def test_image_with_too_many_pixels_is_rejected_before_decoding(client: TestClient, url: str):
    side = int(inference.MAX_IMAGE_PIXELS ** 0.5) + 200
    buffer = io.BytesIO()
    Image.new("L", (side, side)).save(buffer, "PNG")  # ảnh trắng nén rất nhỏ nhưng khi giải mã tốn ~side² byte
    assert len(buffer.getvalue()) < inference.MAX_UPLOAD_BYTES
    response = client.post(url, files={"file": ("wide.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 413, (url, response.status_code)
    assert "điểm ảnh" in response.json()["detail"]


def test_normal_sized_image_is_not_rejected_by_the_limits(client: TestClient):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "green").save(buffer, "JPEG")
    response = client.post("/api/inference/all", files={"file": ("leaf.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code != 413  # có thể 503 nếu bộ đối chứng chưa khả dụng; chỉ giới hạn kích thước là điều được kiểm

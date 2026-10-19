"""Tiêu đề bảo mật của trang chính: CSP chặt (không unsafe-inline/unsafe-eval), nosniff, công tắc lùi, không phá /docs và /quan-tri."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from web.app import ADMIN_PAGE_HEADERS, PAGE_CSP, app, security_headers_for


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _directives(csp: str) -> dict[str, list[str]]:
    return {part.split()[0]: part.split()[1:] for part in csp.split(";") if part.strip()}


def test_main_page_carries_a_strict_csp_and_frame_headers(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CROPFED_CSP_MODE", raising=False)
    response = client.get("/")
    csp = response.headers["content-security-policy"]
    assert csp == PAGE_CSP
    rules = _directives(csp)
    assert rules["default-src"] == ["'self'"] and rules["object-src"] == ["'none'"] and rules["frame-ancestors"] == ["'none'"]
    assert rules["script-src"] == ["'self'", "'wasm-unsafe-eval'"], "wasm cho onnxruntime-web, không unsafe-eval/unsafe-inline"
    assert rules["style-src"] == ["'self'"]
    assert {"data:", "blob:"} <= set(rules["img-src"]) and rules["worker-src"] == ["'self'"]
    assert "'unsafe-inline'" not in csp
    assert not re.search(r"(?<!wasm-)'unsafe-eval'", csp)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY" and response.headers["referrer-policy"] == "same-origin"


def test_api_and_scripts_get_nosniff_but_no_page_csp(client: TestClient):
    for path in ("/api/status", "/static/js/app.js", "/static/js/workers/community-worker.js"):
        response = client.get(path)
        assert response.headers["x-content-type-options"] == "nosniff", path
        assert "content-security-policy" not in response.headers, f"{path}: CSP của trang không áp cho script/API (worker lấy chính sách từ phản hồi của nó)"


def test_swagger_docs_are_exempt_and_admin_page_keeps_its_own_stricter_policy(client: TestClient):
    assert "content-security-policy" not in client.get("/docs").headers, "/docs nạp Swagger từ CDN, CSP của trang sẽ làm nó hỏng"
    admin = client.get("/quan-tri")
    assert admin.headers["content-security-policy"] == ADMIN_PAGE_HEADERS["Content-Security-Policy"]
    assert admin.headers["referrer-policy"] == "no-referrer", "setdefault: không ghi đè chính sách riêng của trang quản trị"


def test_csp_kill_switch_modes(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CROPFED_CSP_MODE", "report")
    report = client.get("/")
    assert report.headers["content-security-policy-report-only"] == PAGE_CSP and "content-security-policy" not in report.headers
    monkeypatch.setenv("CROPFED_CSP_MODE", "off")
    off = client.get("/")
    assert "content-security-policy" not in off.headers and "content-security-policy-report-only" not in off.headers
    assert off.headers["x-content-type-options"] == "nosniff"
    monkeypatch.setenv("CROPFED_CSP_MODE", "giá-trị-lạ")
    assert "content-security-policy" in client.get("/").headers, "giá trị lạ phải rơi về enforce, không âm thầm tắt CSP"


def test_security_headers_function_is_pure():
    assert security_headers_for("/api/x", "application/json") == {"X-Content-Type-Options": "nosniff"}
    headers = security_headers_for("/", "text/html; charset=utf-8", csp_mode="enforce")
    assert headers["Content-Security-Policy"] == PAGE_CSP and "Content-Security-Policy-Report-Only" not in headers

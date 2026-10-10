"""Hợp đồng giao diện web tĩnh: ID/handler khớp giữa HTML và JS, tài nguyên tự host, header cache, quy ước hiệu năng.

Không cần trình duyệt: kiểm bằng phân tích văn bản và TestClient. Chạy nhanh, không cần dataset.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import app

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
JS_FILES = {name: (STATIC / "js" / name).read_text(encoding="utf-8") for name in ("main.js", "published_demo.js", "toast.js")}
ALL_JS = "\n".join(JS_FILES.values())


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _html_ids() -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', HTML))


def test_every_literal_element_id_used_by_js_exists_in_html():
    used = set(re.findall(r'getElementById\("([^"$]+)"\)', ALL_JS))
    used |= set(re.findall(r'(?:renderTop3|setMetricById)\("([^"]+)"', ALL_JS))
    for prefix in ("cent", "fed"):  # updateProgressBlock dựng id từ tiền tố
        used |= {f"{prefix}-progress-{part}" for part in ("bar-box", "fill", "label", "detail")}
    missing = sorted(used - _html_ids())
    assert not missing, f"JS tham chiếu id không có trong index.html: {missing}"


def test_inline_event_handlers_are_defined_in_js():
    handlers = set(re.findall(r'\bon(?:click|change)="(\w+)\(', HTML))
    assert handlers, "index.html không còn handler inline nào — kiểm tra lại test này"
    undefined = sorted(h for h in handlers if not re.search(rf"\bfunction\s+{h}\b", ALL_JS))
    assert not undefined, f"handler inline chưa được định nghĩa: {undefined}"


def test_six_tabs_keep_their_data_tab_targets():
    tabs = re.findall(r'class="nav-item[^"]*"\s+data-tab="([^"]+)"', HTML)
    assert tabs == [
        "tab-dashboard", "tab-centralized", "tab-federated", "tab-clients", "tab-experiments", "tab-inference",
    ]
    for tab in tabs:
        assert f'id="{tab}"' in HTML


def test_css_defines_every_class_that_html_and_js_toggle():
    used: set[str] = set()
    for value in re.findall(r'\bclass="([^"]*)"', HTML):
        used.update(value.split())
    used |= set(re.findall(r'classList\.(?:add|remove|toggle|contains)\("([\w-]+)"', ALL_JS))
    undefined = sorted(t for t in used if not re.search(rf"\.{re.escape(t)}(?![\w-])", CSS))
    # .upload-placeholder chỉ là mục tiêu để ẩn/hiện bằng JS, không cần kiểu riêng
    undefined = [t for t in undefined if t not in {"upload-placeholder", "header-actions"}]
    assert not undefined, f"lớp CSS chưa định nghĩa: {undefined}"


def test_no_blocking_alert_and_no_external_runtime_assets():
    assert not re.search(r"(?<![\w.])alert\(", re.sub(r"//[^\n]*", "", ALL_JS)), "còn alert() — dùng window.toast"
    page = HTML + CSS + ALL_JS
    for needle in ("fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"):
        assert needle not in page, f"demo phải chạy offline, còn phụ thuộc {needle}"


def test_css_performance_conventions():
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)  # bỏ comment: chỉ kiểm quy tắc thật
    assert not re.search(r"transition\s*:\s*all\b", rules), "dùng thuộc tính cụ thể thay cho transition: all"
    assert not re.search(r"backdrop-filter\s*:", rules), "backdrop-filter làm giật khi cuộn"
    assert "prefers-reduced-motion" in CSS
    assert ":focus-visible" in CSS


def test_text_font_has_vietnamese_subset_and_all_files_exist():
    fonts_css = (STATIC / "fonts" / "fonts.css").read_text(encoding="utf-8")
    assert "Be Vietnam Pro" in fonts_css
    for family in ("Be Vietnam Pro", "JetBrains Mono"):
        faces = re.findall(rf"font-family: '{family}';.*?\}}", fonts_css, flags=re.S)
        assert any("U+1EA0-1EF9" in face for face in faces), f"{family} thiếu subset vietnamese"
    for url in re.findall(r"url\('(/static/fonts/[^']+)'\)", fonts_css):
        path = STATIC / url.removeprefix("/static/")
        assert path.is_file() and path.read_bytes()[:4] == b"wOF2", f"{url} thiếu hoặc không phải woff2"
    for href in re.findall(r'rel="preload"[^>]*href="(/static/fonts/[^"]+)"', HTML):
        assert (STATIC / href.removeprefix("/static/")).is_file(), f"preload trỏ tới tệp không có: {href}"


def test_chart_library_is_vendored_and_pinned():
    chart = STATIC / "vendor" / "chart.umd.js"
    assert chart.is_file()
    assert "Chart.js v4.4.1" in chart.read_text(encoding="utf-8", errors="replace")[:200]
    assert re.search(r'src="/static/vendor/chart\.umd\.js\?v=4\.4\.1"', HTML)


def test_fonts_are_cacheable_but_app_assets_stay_no_store(client: TestClient):
    font = client.get("/static/fonts/be-vietnam-pro-400-vietnamese.woff2")
    assert font.status_code == 200
    assert font.headers["content-type"] == "font/woff2"
    assert "max-age=86400" in font.headers["cache-control"]

    for path in ("/static/css/style.css", "/static/js/main.js", "/static/js/toast.js", "/"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "no-store" in response.headers["cache-control"], path

    missing = client.get("/static/fonts/does-not-exist.woff2")
    assert missing.status_code == 404
    assert "max-age" not in missing.headers.get("cache-control", ""), "404 không được bị cache một ngày"

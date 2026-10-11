"""Hợp đồng giao diện web tĩnh (6 bước): ID/route khớp, module import được, tài nguyên tự host, header cache, quy ước hiệu năng.

Không cần trình duyệt: kiểm bằng phân tích văn bản và TestClient. Chạy nhanh, không cần dataset.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import app

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
JS_DIR = STATIC / "js"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = "\n".join((STATIC / "css" / name).read_text(encoding="utf-8") for name in ("style.css", "pages.css"))
JS_FILES = sorted(p for p in JS_DIR.rglob("*.js"))
JS_BY_REL = {p.relative_to(JS_DIR).as_posix(): p.read_text(encoding="utf-8") for p in JS_FILES}
ALL_JS = "\n".join(JS_BY_REL.values())
ROUTE_IDS = ["tong-quan", "non-iid", "ket-qua", "chan-doan", "huan-luyen", "bang-chung"]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _known_ids() -> set[str]:
    """ID có trong index.html hoặc được JS dựng bằng template (id="...")."""
    return set(re.findall(r'\bid="([^"$]+)"', HTML)) | set(re.findall(r'\bid="([^"$]+)"', ALL_JS))


def test_every_literal_element_id_used_by_js_exists():
    used = set(re.findall(r'(?:byId|getElementById)\("([^"$]+)"\)', ALL_JS))
    used |= set(re.findall(r'\$\("#([\w-]+)"\)', ALL_JS))
    missing = sorted(used - _known_ids())
    assert not missing, f"JS tham chiếu id không có trong index.html hoặc template: {missing}"


def test_six_routes_match_nav_sections_and_page_modules():
    router = JS_BY_REL["core/router.js"]
    routes = re.findall(r'id:\s*"([\w-]+)",\s*step:\s*(\d+)', router)
    assert [r for r, _ in routes] == ROUTE_IDS
    assert [int(n) for _, n in routes] == [1, 2, 3, 4, 5, 6]
    nav = re.findall(r'class="nav-item"\s+href="#([\w-]+)"\s+data-route="([\w-]+)"', HTML)
    assert [a for a, _ in nav] == ROUTE_IDS and all(a == b for a, b in nav)
    for route in ROUTE_IDS:
        assert re.search(rf'<section id="page-{route}" class="tab-pane[^"]*" data-route="{route}"', HTML), route
    for module in re.findall(r'import\("\.\./pages/([\w-]+\.js)"\)', router):
        assert (JS_DIR / "pages" / module).is_file(), module
        assert "export async function mount" in JS_BY_REL[f"pages/{module}"], f"{module} thiếu mount()"


def test_es_module_imports_resolve_and_export_the_imported_names():
    problems = []
    for rel, source in JS_BY_REL.items():
        if rel == "toast.js":
            continue  # script cổ điển, không phải module
        for names, target in re.findall(r'import\s*\{([^}]*)\}\s*from\s*"([^"]+)"', source, flags=re.S):
            path = (JS_DIR / rel).parent.joinpath(target).resolve()
            if not path.is_file():
                problems.append(f"{rel}: không có {target}")
                continue
            exported = set(re.findall(r"export\s+(?:async\s+)?(?:function|const|let|class)\s+([\w$]+)", path.read_text(encoding="utf-8")))
            for name in (n.strip().split(" as ")[0].strip() for n in names.split(",") if n.strip()):
                if name not in exported:
                    problems.append(f"{rel}: {target} không export {name}")
    assert not problems, problems


def test_no_inline_event_handlers_and_scripts_load_as_modules():
    assert not re.search(r'\bon(?:click|change|input|submit)\s*=', HTML), "dùng addEventListener, không handler inline"
    assert re.search(r'<script type="module" src="/static/js/app\.js', HTML)
    assert re.search(r'<script src="/static/js/toast\.js', HTML)
    assert not (JS_DIR / "main.js").exists() and not (JS_DIR / "published_demo.js").exists(), "bản UI cũ phải được gỡ"


def test_css_defines_every_class_that_html_and_js_use():
    used: set[str] = set()
    for value in re.findall(r'\bclass="([^"]*)"', HTML):
        used.update(value.split())
    # class dựng trong template JS: bỏ phần nội suy ${...}; token cụt (kết thúc bằng '-') là tiền tố ghép động
    for value in re.findall(r'\bclass="([^"]*)"', ALL_JS):
        used.update(re.sub(r"\$\{[^}]*\}", " ", value).split())
    used |= set(re.findall(r'classList\.(?:add|remove|toggle|contains)\("([\w-]+)"', ALL_JS))
    used = {t for t in used if re.fullmatch(r"[a-z][\w-]*[a-z0-9]", t)}
    undefined = sorted(t for t in used if not re.search(rf"\.{re.escape(t)}(?![\w-])", CSS))
    exempt = {"upload-placeholder", "header-actions", "text-rose", "fed-fill", "icon-xs", "nav-label", "selected-meta"}
    undefined = [t for t in undefined if t not in exempt]
    assert not undefined, f"lớp CSS chưa định nghĩa: {undefined}"


def test_no_blocking_alert_and_no_external_runtime_assets():
    assert not re.search(r"(?<![\w.])alert\(", re.sub(r"//[^\n]*", "", ALL_JS)), "còn alert() — dùng window.toast"
    page = HTML + CSS + ALL_JS
    for needle in ("fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"):
        assert needle not in page, f"demo phải chạy offline, còn phụ thuộc {needle}"


def test_css_performance_and_accessibility_conventions():
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)  # bỏ comment: chỉ kiểm quy tắc thật
    assert not re.search(r"transition\s*:\s*all\b", rules), "dùng thuộc tính cụ thể thay cho transition: all"
    assert not re.search(r"backdrop-filter\s*:", rules), "backdrop-filter làm giật khi cuộn"
    assert "prefers-reduced-motion" in CSS
    assert ":focus-visible" in CSS
    assert ".skip-link" in CSS and 'class="skip-link"' in HTML


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

    for path in ("/static/css/style.css", "/static/css/pages.css", "/static/js/app.js", "/static/js/core/router.js",
                 "/static/js/pages/diagnosis.js", "/static/js/toast.js", "/"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "no-store" in response.headers["cache-control"], path

    missing = client.get("/static/fonts/does-not-exist.woff2")
    assert missing.status_code == 404
    assert "max-age" not in missing.headers.get("cache-control", ""), "404 không được bị cache một ngày"

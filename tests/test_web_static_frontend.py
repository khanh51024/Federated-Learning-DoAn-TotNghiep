"""Hợp đồng giao diện web tĩnh (6 bước): ID/route khớp, module import được, tài nguyên tự host, header cache, quy ước hiệu năng.

Không cần trình duyệt: kiểm bằng phân tích văn bản và TestClient. Chạy nhanh, không cần dataset.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
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
    for family in ("Be Vietnam Pro", "JetBrains Mono", "Source Serif 4"):
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


BANNED_VISIBLE_TERMS = {
    r"Val accuracy": "dùng 'độ chính xác trên tập kiểm tra'",
    r"Học liên hiệp": "thuật ngữ thống nhất là 'học liên kết'",
    r"Test giữ riêng": "dùng 'tập giữ riêng'",
    r"tập (?:ảnh )?độc lập": "'độc lập' nói quá: tập này chỉ là tập giữ riêng, không dùng để chọn mô hình",
    r"(?<!giả định )ảnh độc lập": "xem 'tập giữ riêng' (chỉ 'giả định ảnh độc lập' của thống kê là hợp lệ)",
    r"Kiểm tra độc lập": "xem 'tập giữ riêng'",
    r"\bbệnh này\b": "nhóm có thể là lá khỏe: dùng 'nhóm này'",
    r"Lớp bệnh": "dùng 'Nhóm'",
}


def _visible_source() -> str:
    """HTML + JS đã bỏ comment (chỉ kiểm chuỗi người xem nhìn thấy)."""
    text = HTML + "\n" + "\n".join(JS_BY_REL.values())
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def test_visible_copy_has_no_banned_terms():
    source = _visible_source()
    hits = [f"{pattern}: {why}" for pattern, why in BANNED_VISIBLE_TERMS.items() if re.search(pattern, source)]
    assert not hits, "Chuỗi hiển thị còn thuật ngữ đã bỏ: " + "; ".join(hits)


def test_nav_labels_match_route_labels():
    router = JS_BY_REL["core/router.js"]
    labels = re.findall(r'label:\s*"([^"]+)"', router)
    nav = re.findall(r'<span class="nav-label">([^<]+)</span>', HTML)
    assert [label.replace("&", "&amp;") for label in labels] == nav, (labels, nav)



# ───────── Đợt làm lại giao diện: thanh trên cùng, token, chuyển động ─────────
def test_top_navigation_replaces_sidebar():
    header = re.search(r'<header class="app-topbar" id="app-topbar">.*?</header>', HTML, flags=re.S)
    assert header, "thiếu header.app-topbar#app-topbar"
    assert 'class="nav-menu"' in header.group(0) and header.group(0).count('class="nav-item"') == 6
    for gone in ('class="sidebar"', "mobile-nav-toggle", "present-steps", "app-sidebar", "mobile-topbar"):
        assert gone not in HTML, f"UI cũ còn sót: {gone}"
    app = JS_BY_REL["app.js"]
    assert "initSidebarToggle" not in app and "mobile-nav-toggle" not in app


def _tokens(block_selector: str) -> dict[str, str]:
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    match = re.search(re.escape(block_selector) + r"\s*\{(.*?)\n\}", css, flags=re.S)
    assert match, block_selector
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", match.group(1)))


def _rgba(value: str, under: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(float(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    parts = [float(x) for x in re.match(r"rgba?\(([^)]+)\)", value).group(1).split(",")]
    r, g, b = parts[:3]
    alpha = parts[3] if len(parts) > 3 else 1.0
    if alpha < 1 and under is not None:
        return tuple(alpha * c + (1 - alpha) * u for c, u in zip((r, g, b), under))
    return (r, g, b)


def _luminance(rgb: tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _mix(color, base, share: float):
    return tuple(share * c + (1 - share) * b for c, b in zip(color, base))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_ui_tokens_meet_contrast(theme: str):
    tokens = _tokens(":root") if theme == "light" else {**_tokens(":root"), **_tokens(':root[data-theme="dark"]')}
    get = lambda name, under=None: _rgba(tokens[name], under)  # noqa: E731
    base, surface = get("--bg-base"), get("--bg-surface")
    grounds = {n: get(n) for n in ("--bg-base", "--bg-surface", "--bg-subtle", "--bg-sunken")}
    problems: list[str] = []

    def need(label: str, fg, bg, minimum: float, maximum: float | None = None) -> None:
        ratio = _contrast(fg, bg)
        if ratio < minimum or (maximum is not None and ratio > maximum):
            problems.append(f"{theme}: {label} = {ratio:.2f}")

    for text in ("--text-primary", "--text-secondary", "--text-muted"):
        for name, ground in grounds.items():
            need(f"{text} trên {name}", get(text), ground, 4.5)
    for accent in ("--accent-emerald", "--accent-emerald-strong", "--accent-sky", "--accent-amber", "--accent-rose"):
        for name in ("--bg-base", "--bg-surface"):
            need(f"{accent} trên {name}", get(accent), grounds[name], 4.5)
    for fg, tint in (("--accent-emerald-strong", "--tint-emerald"), ("--accent-sky", "--tint-sky"),
                     ("--accent-amber", "--tint-amber"), ("--accent-rose", "--tint-rose")):
        need(f"{fg} trên {tint}", get(fg), get(tint, surface), 4.5)
    for bg in ("--accent-emerald", "--accent-rose", "--accent-amber"):
        need(f"chữ trên nền {bg}", get("--text-on-accent"), get(bg), 4.5)
    for chart in ("--chart-1", "--chart-2", "--chart-4", "--chart-5"):
        need(f"{chart} (đồ họa) trên --bg-surface", get(chart), surface, 3.0)
    need("--border-input trên --bg-surface", get("--border-input"), surface, 3.0)
    need("--border-focus trên --bg-base", get("--border-focus"), base, 3.0)
    # ô heatmap / ma trận: chữ chính trên nền trộn đậm nhất
    for token, cap in (("--heat-count", 0.62), ("--heat-good", 0.70), ("--heat-bad", 0.70)):
        need(f"chữ trên ô {token}", get("--text-primary"), _mix(get(token), surface, cap), 4.5)
    # lưới chấm phải rất mờ (trang trí, không cạnh tranh với nội dung)
    need("--dot trên --bg-base", get("--dot", base), base, 1.0, 1.3)
    assert not problems, "; ".join(problems)


def _class_words() -> set[str]:
    """Từ có thể là tên lớp: bỏ chú thích HTML/JS, thuộc tính id, byId("…")/getElementById("…") và selector #id,
    để một id hay câu chú thích trùng tên không làm một lớp CSS chết trông như còn dùng."""
    html = re.sub(r"<!--.*?-->", "", HTML, flags=re.S)
    html = re.sub(r'\bid="[^"]*"', "", html)
    js = re.sub(r"/\*.*?\*/", "", ALL_JS, flags=re.S)
    js = re.sub(r"(?m)(^|[^:\"'`\\])//.*$", r"\1", js)
    js = re.sub(r"(?:byId|getElementById)\(\s*[\"'][^\"']*[\"']\s*\)", "", js)
    js = re.sub(r"#[A-Za-z_][\w-]*", "", js)
    return set(re.findall(r"[A-Za-z_][\w-]*", html + "\n" + js))


def _css_classes_used_somewhere() -> tuple[set[str], set[str]]:
    words = _class_words()
    prefixes = {w for w in words if w.endswith("-")} | {"level-", "row-", "toast-", "seg-", "repro-", "pillar-", "summary-", "conf-"}
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    classes = set(re.findall(r"\.([A-Za-z_][\w-]*)", css))
    # bỏ phần chữ số thập phân/đuôi tệp lọt vào regex (vd. `0.5s`, `.woff2`)
    classes = {c for c in classes if re.match(r"[A-Za-z_]", c) and not re.fullmatch(r"\d.*", c)}
    unused = {c for c in classes if c not in words and not any(c.startswith(p) for p in prefixes)}
    return classes, unused


def test_every_css_class_is_referenced():
    """Chiều ngược của test lớp-chưa-định-nghĩa: CSS chết (của giao diện cũ) không được quay lại."""
    _classes, unused = _css_classes_used_somewhere()
    # lớp chỉ do JS ghép tên từ nhiều mảnh (không có nguyên từ trong nguồn) — liệt kê tường minh
    allowed = {"is-cam"}
    leftover = sorted(unused - allowed)
    assert not leftover, f"CSS chết (không lớp nào dùng): {leftover}"


def _rule_bodies(selector: str) -> list[str]:
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    return [m.group(1) for m in re.finditer(re.escape(selector) + r"\s*\{([^}]*)\}", rules)]


_STATE_SELECTOR = re.compile(r"\.is-active|\.active\b|:hover|:focus|\.is-done|\.is-on|\.is-selected|\[aria-(?:selected|pressed|checked|current)")
_LAYOUT_DECL = re.compile(r"(?<![\w-])(?:font-weight|font-size|letter-spacing|padding(?:-[a-z]+)?|margin(?:-[a-z]+)?|width|height|border-width|line-height)\s*:")


def test_state_rules_do_not_change_box_or_text_metrics():
    """Mọi luật cho trạng thái (hover/active/selected/done/on) chỉ đổi màu, bóng, độ trong, transform — không đổi
    font-weight, cỡ chữ, đệm, lề, kích thước: những thứ đó làm layout lại và nhảy chữ khi chuyển trạng thái."""
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    offenders = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", rules):
        selector, body = match.group(1).strip(), match.group(2)
        if selector.startswith("@") or not _STATE_SELECTOR.search(selector):
            continue
        # `.hidden`/`display` và vị trí được phép; chỉ cấm số đo hộp/chữ
        for decl in _LAYOUT_DECL.findall(body):
            offenders.append(f"{selector} {{ {decl.strip()} … }}")
    assert not offenders, "trạng thái đổi số đo: " + "; ".join(sorted(set(offenders))[:12])


def test_hover_and_active_states_do_not_trigger_layout():
    """Trạng thái rê chuột/đang chọn chỉ được đổi màu hoặc transform: font-weight/border làm bảng, thanh điều hướng layout lại."""
    # Dải sáng hàng/cột là phần tử 1×1 bị scale(): box-shadow/border/outline sẽ bị phóng theo thành quầng lớn.
    bands = _rule_bodies(".xh-band")
    assert bands, "thiếu .xh-band"
    for body in bands:
        assert not re.search(r"box-shadow|border|outline", body), f".xh-band không được có viền/bóng: {body.strip()}"
    for selector in (".xh-head", ".nav-item.active", ".nav-item:hover"):
        bodies = _rule_bodies(selector)
        assert bodies, f"thiếu luật {selector}"
        for body in bodies:
            assert "font-weight" not in body and "padding" not in body, f"{selector} đổi bề rộng chữ → layout: {body.strip()}"


def test_motion_conventions():
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    layout_props = r"\b(?:width|height|top|left|right|bottom|margin(?:-\w+)?|padding(?:-\w+)?|inset|max-height|min-height)\b"
    bad = []
    for decl in re.findall(r"transition(?:-property)?\s*:\s*([^;}]+)[;}]", rules):
        # chỉ xét danh sách thuộc tính được animate (bỏ phần thời lượng/easing dạng var()/số)
        names = re.sub(r"var\([^)]*\)|[\d.]+m?s|cubic-bezier\([^)]*\)|ease(?:-in-out|-in|-out)?|linear", " ", decl)
        if re.search(layout_props, names):
            bad.append(decl.strip())
    assert not bad, f"transition đụng thuộc tính bố cục (gây layout): {bad}"

    keyframes = re.findall(r"@keyframes\s+([\w-]+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", rules)
    assert len(keyframes) >= 5, "regex @keyframes không còn khớp: test sẽ pass rỗng"
    for name, body in keyframes:
        props = set(re.findall(r"([a-z-]+)\s*:", re.sub(r"\{|\}", ";", body)))
        extra = props - {"transform", "opacity"}
        assert not extra, f"@keyframes {name} animate thuộc tính ngoài transform/opacity: {sorted(extra)}"

    # keyframe viết literal trong JS: chỉ opacity/transform (+ offset/easing)
    for rel, source in JS_BY_REL.items():
        for call in re.findall(r"(?:\bplay|\.animate)\((?:[^()]|\([^()]*\))*\)", source):
            for prop in re.findall(r"\{\s*([a-zA-Z]+)\s*:", call) + re.findall(r",\s*([a-zA-Z]+)\s*:", call):
                assert prop in {"opacity", "transform", "offset", "easing", "delay", "duration", "fill", "pseudoElement",
                                "scale", "x", "y"}, f"{rel}: keyframe/tuỳ chọn lạ trong {call[:80]!r}: {prop}"

    motion = JS_BY_REL["core/motion.js"]
    assert "(pointer: fine)" in motion and "prefersReducedMotion" in motion
    assert re.search(r"export function onShow", JS_BY_REL["pages/overview.js"]), "bước 1 phải chạy lại minh họa mỗi lần vào"


def _outside_details(source: str) -> str:
    """Nguồn template JS với mọi khối <details>…</details> bị cắt bỏ (nội dung gập mặc định không thấy được)."""
    return re.sub(r"<details\b.*?</details>", "", source, flags=re.S)


def test_honest_caveats_are_visible_outside_collapsed_sections():
    overview = _outside_details(JS_BY_REL["pages/overview.js"])
    results = _outside_details(JS_BY_REL["pages/results.js"])
    for label, text in (("bước 1", overview), ("bước 3", results)):
        flat = " ".join(text.split())
        for phrase in ("chọn mô hình", "một lần chạy", "chưa phải độ chính xác ngoài thực địa"):
            assert phrase in flat, f"{label}: thiếu lời cảnh báo hiển thị ngoài phần gập: “{phrase}”"
    full = " ".join(JS_BY_REL["pages/overview.js"].split())
    assert "bộ chẩn đoán có sẵn của bộ dữ liệu (legacy)" in full and "chưa nằm trong kiểm tra trùng ảnh" in full, "tập giữ riêng phải được gọi đúng là bộ chẩn đoán legacy"
    assert 'class="trust-chip ok" href="#ket-qua"' not in full, "chip tập giữ riêng không được xanh 'đạt' khi chưa qua kiểm tra trùng ảnh"


def test_hero_replay_survives_pane_switches_and_live_diagram_is_state_driven():
    overview = JS_BY_REL["pages/overview.js"]
    # ResizeObserver: pane ẩn (bề rộng 0) và thanh cuộn xuất hiện không được hủy hoạt ảnh vừa chạy khi vào lại bước 1
    observer = overview[overview.index("new ResizeObserver"):]
    observer = observer[: observer.index(".observe(diagram)")]
    assert "width <= 0" in observer and "> 24" in observer, "ResizeObserver của hero phải bỏ qua 0↔W và dung sai thanh cuộn"
    training = JS_BY_REL["pages/training.js"]
    assert "completed > topo.lastRound && visible && view.federated" in training, "chỉ nhịp khi một vòng học liên kết thật vừa xong"
    assert "live_progress.length" not in training.replace("live?.length", ""), "không đếm epoch Centralized như vòng liên kết"
    assert 'kind === "centralized" && task?.status === "running"' in training, "đang chạy Centralized thì không hiện client đang kết nối"


def test_segmented_and_cam_markup_are_accessible():
    assert re.search(r'<img id="preview-cam"[^>]*alt=""[^>]*aria-hidden="true"', HTML), "ảnh Grad-CAM chưa có src phải ẩn khỏi AT"
    input_rules = _rule_bodies(".input-field")
    assert input_rules and "var(--border-input)" in input_rules[0], "viền ô nhập phải dùng --border-input (≥ 3:1)"
    overview = JS_BY_REL["pages/overview.js"]
    hero = overview[overview.index("function heroHtml"):overview.index("function playHero")]
    assert 'class="fl-figure" role="img"' in hero, "role=img phải đặt ở khung sơ đồ"
    assert '<figure class="fl-diagram" id="fl-diagram" role="img"' not in hero, "role=img không được bọc nút Xem lại"


def test_fullframe_limitation_numbers_come_from_the_api_not_the_source():
    """Con số "ảnh cả khung làm mô hình kém đi" lấy từ summary.fullframe; không được gõ cứng trong JS."""
    source = JS_BY_REL["pages/evidence.js"]
    assert "summary?.fullframe" in source and "ff.splits" in source
    block = source[source.index("const ff = summary?.fullframe"):]
    block = block[: block.index('out.push(["So sánh Centralized')]
    literal_percent = re.findall(r"(?<![\w$.{])\d{1,3}[,.]\d\s*%", block)
    assert not literal_percent, f"số phần trăm gõ cứng trong giới hạn ảnh nguyên khung: {literal_percent}"



# ───────── Sửa lỗi nghiệm thu UI 12/10/2026 ─────────
def test_training_form_is_validated_and_stop_waits_for_the_task_to_be_accepted():
    source = JS_BY_REL["pages/training.js"]
    start = source[source.index("async function start()"):source.index("async function stopTraining()")]
    post = start.index("postJson(")
    assert start.index("validateTrainForm()") < post, "phải kiểm biểu mẫu trước khi gửi"
    assert start.index("stop.disabled = true") < post, "nút Dừng không được bật trước khi máy chủ nhận tác vụ"
    assert start.index("setDisabled(stop, false)") > post, "chỉ bật Dừng sau khi POST thành công"


def test_result_tabs_stay_wired_when_evidence_is_unavailable():
    source = JS_BY_REL["pages/results.js"]
    mount = source[source.index("export async function mount()"):]
    assert mount.index(".subtab") < mount.index("if (!evidence.available)"), "handler phải gắn trước nhánh unavailable"
    assert "if (!ctx) return;" in source[source.index("async function activate"):source.index("export async function mount()")]


def test_route_change_clears_form_errors_but_keeps_connection_status():
    assert "keepSticky" in JS_BY_REL["toast.js"]
    router = JS_BY_REL["core/router.js"]
    show = router[router.index("async function show("):]
    # dọn ngay đầu show(), trước mọi await: nếu đợi mount() của trang đích xong thì toast lỗi còn che nội dung
    assert show.index("dismissAll({ keepSticky: true })") < show.index("await ")
    assert re.search(r"Mất kết nối máy chủ.*?sticky: true", JS_BY_REL["core/status.js"], flags=re.S)


def test_wide_grid_minimums_never_exceed_their_container():
    """Cột tối thiểu cố định ≥ 280 px làm lưới tràn ngang ở viewport 320 px: phải bọc bằng min(100%, …)."""
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    fixed = [int(n) for n in re.findall(r"repeat\(auto-(?:fit|fill),\s*minmax\((\d+)px,\s*1fr\)\)", rules)]
    assert not [n for n in fixed if n >= 280], f"lưới có cột tối thiểu cố định: {fixed}"


def test_upload_limit_is_the_same_on_client_and_server():
    from web.routers import inference

    match = re.search(r"MAX_UPLOAD_BYTES\s*=\s*(\d+)\s*\*\s*1024\s*\*\s*1024", JS_BY_REL["pages/diagnosis.js"])
    assert match, "diagnosis.js thiếu MAX_UPLOAD_BYTES"
    megabytes = int(match.group(1))
    assert megabytes * 1024 * 1024 == inference.MAX_UPLOAD_BYTES
    assert f"tối đa {megabytes} MB" in HTML, "gợi ý trên vùng tải ảnh phải nêu đúng giới hạn"


@pytest.mark.skipif(shutil.which("node") is None, reason="cần Node.js để nạp mô-đun JS")
def test_validation_errors_from_the_api_read_in_vietnamese():
    url = (JS_DIR / "core" / "dom.js").resolve().as_uri()
    script = (
        f"import {{ formatApiError }} from {json.dumps(url)};"
        "const item = (type, field, ctx, msg) => ({ detail: [{ type, loc: ['body', field], ctx, msg }] });"
        "console.log(JSON.stringify(["
        "formatApiError(item('greater_than_equal', 'epochs', { ge: 1 }, 'Input should be greater than or equal to 1'), 'x'),"
        "formatApiError(item('less_than_equal', 'rounds', { le: 100 }, 'Input should be less than or equal to 100'), 'x'),"
        "formatApiError(item('int_parsing', 'seed', {}, 'Input should be a valid integer'), 'x'),"
        "formatApiError(item('some_new_type', 'epochs', {}, 'Original message'), 'x'),"
        "formatApiError({ detail: 'Một tác vụ khác đang chạy' }, 'x'),"
        "]))"
    )
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        "Số epoch: tối thiểu là 1.",
        "Số vòng: tối đa là 100.",
        "Seed: phải là một số hợp lệ.",
        "Original message",
        "Một tác vụ khác đang chạy",
    ]


def test_segmented_groups_wrap_instead_of_overflowing_and_required_numbers_are_marked():
    bodies = _rule_bodies(".segmented")
    assert bodies and any("max-width: 100%" in body and "flex-wrap: wrap" in body for body in bodies), \
        "nhóm nút phân đoạn có flex-shrink:0 nên cần max-width:100% để xuống dòng ở 320 px"
    for field in ("fl-rounds", "fl-epochs", "fl-seed", "cent-epochs", "cent-seed"):
        tag = re.search(rf'<input[^>]*id="{field}"[^>]*>', HTML)
        assert tag and " required" in tag.group(0), f"{field} phải có required để ô rỗng không bị gửi lên máy chủ"


@pytest.mark.skipif(shutil.which("node") is None, reason="cần Node.js để nạp mô-đun JS")
def test_network_failure_reads_in_vietnamese_not_the_browsers_english():
    url = (JS_DIR / "core" / "api.js").resolve().as_uri()
    script = (
        f"import {{ getJson }} from {json.dumps(url)};"
        "try { await getJson('http://127.0.0.1:9/x', { timeoutMs: 5000 }); }"
        "catch (e) { console.log(JSON.stringify([e.name, e.message, e.status])); }"
    )
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert done.returncode == 0, done.stderr
    name, message, status = json.loads(done.stdout)
    assert name == "ApiError" and status == 0
    assert "Không kết nối được máy chủ" in message and "fetch" not in message.lower()


def test_scrollable_regions_are_keyboard_focusable_and_rows_use_real_buttons():
    """axe: vùng cuộn phải focus được (2.1.1); aria-expanded không hợp lệ trên <tr> của table nên dùng nút thật."""
    sources = {"index.html": HTML, **{rel: text for rel, text in JS_BY_REL.items() if rel.startswith("pages/")}}
    for name, text in sources.items():
        for tag in re.findall(r"<div[^>]*class=\"[^\"]*table-responsive[^\"]*\"[^>]*>", text):
            assert 'tabindex="0"' in tag and re.search(r'role="(?:region|group)"', tag) and "aria-label=" in tag, f"{name}: {tag[:90]}"
    diagnosis = JS_BY_REL["pages/diagnosis.js"]
    assert 'class="row-toggle"' in diagnosis and "aria-controls=" in diagnosis
    assert not re.search(r'<tr class="diag-row[^>]*aria-expanded', diagnosis), "aria-expanded không được đặt trên <tr>"


def test_step_nav_is_hidden_until_the_page_is_mounted_to_avoid_layout_shift():
    router = JS_BY_REL["core/router.js"]
    show = router[router.index("async function show("):]
    assert show.index('classList.add("is-pending")') < show.index("await mod.mount")
    assert show.index('classList.remove("is-pending")') > show.index("await mod.mount")
    assert re.search(r"\.step-nav\.is-pending\s*\{\s*visibility:\s*hidden", CSS), "visibility giữ chỗ nên không gây layout"


def test_chart_library_does_not_block_the_app_and_modulepreload_targets_exist():
    assert re.search(r'<script src="/static/vendor/chart\.umd\.js\?v=4\.4\.1" async></script>', HTML), \
        "Chart.js phải async: defer khiến app.js phải đợi 200 KB (LCP mobile)"
    preloaded = re.findall(r'<link rel="modulepreload" href="/static/js/([\w/.-]+)">', HTML)
    assert preloaded, "thiếu modulepreload cho chuỗi import của bước mặc định"
    for rel in preloaded:
        assert (JS_DIR / rel).is_file(), rel
    charts = JS_BY_REL["core/charts.js"]
    assert "waiting.set(canvas, factory)" in charts and "settleWaiting" in charts, "makeChart phải xếp hàng khi Chart.js chưa nạp xong"


def test_wide_screen_content_is_capped_where_readability_needs_it():
    """Khung chính rộng tới 1760 px; hình 16:9 theo dõi và thanh phân bố phải có trần riêng để không cao/dài quá mức."""
    assert re.search(r"--page-max:\s*1[67]\d\dpx", CSS), "khung chính phải rộng hơn 1280 px ở màn hình rộng"
    stage = [b for b in _rule_bodies(".fl-live .fl-stage")]
    assert stage and "max-width:" in stage[0], "sơ đồ theo dõi cần trần bề rộng"
    bars = _rule_bodies(".viewer-bars")
    assert bars and "max-width:" in bars[0], "thanh phân bố cần trần bề rộng để nhãn và số không cách nhau quá xa"


def test_hero_sends_the_model_back_and_training_ring_is_compositor_only():
    flnet = JS_BY_REL["core/flnet.js"]
    overview = JS_BY_REL["pages/overview.js"]
    assert "reverse = false" in flnet and "points.reverse()" in flnet
    assert "reverse: true" in overview, "hero phải có đợt gửi mô hình chung quay lại client"
    assert "reverse: true" not in JS_BY_REL["pages/training.js"], "sơ đồ thật chỉ chạy theo sự kiện thật, không dựng đợt quay lại"
    ring = re.search(r"\.fl-client\.is-training \.fl-node-card::after\s*\{([^}]*)\}", CSS)
    assert ring and "animation: topo-ring" in ring.group(1)
    keyframes = re.search(r"@keyframes topo-ring\s*\{(.*?)\n\}", CSS, flags=re.S)
    assert keyframes and set(re.findall(r"([a-z-]+)\s*:", keyframes.group(1))) <= {"transform", "opacity"}


def test_hover_effects_are_pointer_gated_compositor_only_and_charts_build_when_visible():
    rules = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    block = rules[rules.index("@media (hover: hover) and (pointer: fine)"):]
    block = block[: block.index("@media (prefers-reduced-motion: reduce)")]
    for decl in re.findall(r"transition\s*:\s*([^;}]+)[;}]", block):
        names = re.sub(r"var\([^)]*\)|[\d.]+m?s|cubic-bezier\([^)]*\)|ease(?:-in-out|-in|-out)?|linear", " ", decl)
        assert not re.search(r"box-shadow|filter|width|height", names), f"hover animate thuộc tính đắt: {decl}"
    assert "overflow: hidden" in block and "box-shadow: var(--shadow-2)" in block and "::after" in block
    results = JS_BY_REL["pages/results.js"]
    assert "makeChartWhenVisible" in results and "makeChart(" not in results.replace("makeChartWhenVisible(", "")
    assert "createImageBitmap" in JS_BY_REL["pages/diagnosis.js"], "ảnh lớn phải được thu nhỏ ngoài luồng chính"

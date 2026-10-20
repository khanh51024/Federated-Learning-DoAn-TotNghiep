"""Trang quản trị học phân tán cộng đồng (/quan-tri): tệp tĩnh riêng, tiêu đề bảo mật, tham chiếu đủ tài nguyên, không script/style
inline (CSP của trang cấm), mọi id mà JS dùng đều có trong HTML, và dữ liệu mà trang cần có trong /admin/state."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.routers.community_fl import StoreState, store_state

# dùng lại bộ dựng tài sản tổng hợp và tiện ích của test giao thức
from tests.test_community_fl import ADMIN, SECRET, Clock, _admin, _synthetic_assets  # noqa: F401
from web.services.community_store import CommunityStore

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
ADMIN_DIR = STATIC / "admin"
HTML = (ADMIN_DIR / "admin.html").read_text(encoding="utf-8")
JS = (ADMIN_DIR / "admin.js").read_text(encoding="utf-8")


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_page_is_served_with_hardened_headers(client: TestClient):
    response = client.get("/quan-tri")
    assert response.status_code == 200 and "Quản trị" in response.text
    headers = response.headers
    assert "no-store" in headers["cache-control"]
    assert "noindex" in headers["x-robots-tag"]
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"
    csp = headers["content-security-policy"]
    for needle in ("default-src 'self'", "script-src 'self'", "frame-ancestors 'none'", "base-uri 'none'"):
        assert needle in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp


def test_admin_page_is_not_one_of_the_public_pages_and_is_linked_only_from_the_training_gate():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    gate = re.search(r'<div[^>]*id="train-gate".*?</div>', index, flags=re.S)
    assert gate and "/quan-tri" in gate.group(0), "khung cổng huấn luyện phải dẫn tới trang quản trị"
    outside = index.replace(gate.group(0), "")
    assert "quan-tri" not in outside, "ngoài khung cổng huấn luyện, trang trình bày không được liên kết tới trang quản trị"
    header = re.search(r'<header class="app-topbar".*?</header>', index, flags=re.S)
    assert header and "quan-tri" not in header.group(0), "menu công khai không có liên kết quản trị"
    router = (STATIC / "js" / "core" / "router.js").read_text(encoding="utf-8")
    assert "quan-tri" not in router


def test_page_references_only_existing_local_assets():
    refs = re.findall(r'(?:href|src)="(/static/[^"?]+)', HTML)
    assert refs and all((STATIC / ref.removeprefix("/static/")).is_file() for ref in refs), refs
    assert not re.search(r'https?://(?!www\.w3\.org)', HTML + JS), "trang quản trị không được gọi tài nguyên ngoài"
    for local_import in re.findall(r'from "(/static/[^"]+)"', JS):
        assert (STATIC / local_import.removeprefix("/static/")).is_file(), local_import


def test_page_is_csp_clean_no_inline_script_style_or_handlers():
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", HTML), "script inline bị CSP chặn"
    assert not re.search(r"<style\b", HTML)
    assert not re.search(r'\sstyle\s*=', HTML + JS), "thuộc tính style inline bị CSP chặn (style-src 'self')"
    assert not re.search(r'\son[a-z]+\s*=', HTML), "dùng addEventListener, không handler inline"
    assert "innerHTML" in JS and "esc(" in JS
    assert "eval(" not in JS and "new Function" not in JS and "document.write" not in JS


def test_every_element_id_used_by_the_script_exists_in_the_page():
    used = set(re.findall(r'\$\("([\w-]+)"\)', JS)) | set(re.findall(r"\$\(`([\w-]+)`\)", JS))
    used |= set(re.findall(r'text\("([\w-]+)"', JS)) | set(re.findall(r'show\("([\w-]+)"', JS))
    declared = set(re.findall(r'\bid="([\w-]+)"', HTML))
    dynamic = {f"pf-{m}" for m in re.findall(r'key: "(\w+)"', JS)}  # ô cấu hình do JS dựng
    missing = sorted(used - declared - dynamic)
    assert not missing, f"admin.js dùng id không có trong admin.html: {missing}"


def test_script_parses_as_an_es_module():
    if shutil.which("node") is None:
        pytest.skip("cần Node.js")
    for name in ("admin.js", "theme.js"):
        source = (ADMIN_DIR / name).read_text(encoding="utf-8")
        mode = ["--input-type=module"] if name == "admin.js" else []
        done = subprocess.run(["node", *mode, "--check"], input=source, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert done.returncode == 0, f"{name}: {done.stderr}"


def test_policy_fields_in_the_form_match_the_server_policy_and_bounds(tmp_path: Path):
    from src.federated import community as core

    keys = set(re.findall(r'key: "(\w+)"', JS))
    assert keys == set(core.Policy().to_dict()), f"form lệch policy: {sorted(keys ^ set(core.Policy().to_dict()))}"
    assert keys <= set(core.POLICY_BOUNDS) | {"dp_enabled"}


def test_admin_state_carries_what_the_page_needs(tmp_path: Path):
    clock = Clock()
    store = CommunityStore(tmp_path / "state", _synthetic_assets(tmp_path), SECRET, ADMIN, clock=clock)
    app.dependency_overrides[store_state] = lambda: StoreState(store, None)
    try:
        api = TestClient(app)
        assert _admin(api, "settings", {"open": True, "changes": {"k_target": 4, "k_min": 2}}).status_code == 200
        _admin(api, "codes", {"count": 1, "max_uses": 2, "ttl_hours": 1, "label": "buổi thử"})
        state = _admin(api, "state", method="get").json()
        assert state["open"] is True and state["round"]["k_target"] == 4 and state["round"]["received"] == 0
        assert state["policy_defaults"]["k_target"] == 3 and state["policy_bounds"]["k_target"][0] == 2
        assert state["codes"][0]["label"] == "buổi thử" and state["codes"][0]["created_at"] > 0
        assert "server_time" in state and "participants" in state
    finally:
        app.dependency_overrides.pop(store_state, None)

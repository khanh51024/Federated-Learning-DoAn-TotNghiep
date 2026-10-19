"""Kiểm thử đầu-cuối học liên kết cộng đồng bằng trình duyệt thật (Playwright + Chromium) và một máy chủ thật.

Kịch bản: máy chủ uvicorn thật (artifact thật, trạng thái ở thư mục tạm) → quản trị mở phiên (K = 3) và phát mã → BA context trình
duyệt độc lập, mỗi người nhập mã, thêm ảnh gallery có nhãn, chọn nhóm, huấn luyện trong Worker (onnxruntime-web) và gửi → máy chủ
gộp vòng 1 → cả ba trang cùng thấy phiên bản mới. Vòng 2: người thứ tư gửi head bị đầu độc (đảo dấu + nhân lớn) qua API → bị cắt
chuẩn / cổng chặn. Đo thêm: số "long task" trên luồng chính khi huấn luyện, và không có byte ảnh nào đi qua mạng.

Chạy:  python scripts/e2e_community.py            (cần đã chạy export_community_feature_model.py + build_community_eval_features.py)
Thoát mã 0 nếu mọi kiểm tra đạt, 1 nếu có kiểm tra hỏng, 2 nếu thiếu điều kiện chạy (artifact/Playwright/Chromium).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402
from src.federated import community as core  # noqa: E402

ADMIN = "e2e-admin-token-0123456789-abc"
SECRET = "e2e-secret-value"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'OK ' if ok else 'LỖI'}] {name}{' — ' + detail if detail else ''}", flush=True)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]



def wait_js(page: Any, expression: str, timeout: float = 15000) -> None:
    """Chờ một biểu thức JS đúng bằng cách thăm dò từ Python.

    `wait_for_function` chạy eval TRONG trang nên bị CSP (không có 'unsafe-eval') chặn; `page.evaluate` đi qua CDP nên không bị."""
    deadline = time.time() + timeout / 1000
    while time.time() < deadline:
        if page.evaluate(f"() => Boolean({expression})"):
            return
        time.sleep(0.15)
    raise TimeoutError(f"wait_js hết {timeout:.0f} ms: {expression[:120]}")


CSP_VIOLATIONS: list[str] = []
CSP_PROBE = """
  document.addEventListener('securitypolicyviolation', e => {
    if (window.__cspReport) window.__cspReport(
      `${e.violatedDirective} ${e.blockedURI} ${e.sourceFile || ''}:${e.lineNumber || 0} ${(e.sample || '').slice(0, 40)}`);
  });
"""


def new_ctx(browser: Any, **kwargs: Any) -> Any:
    """Mọi context của e2e đi qua đây để ghi lại vi phạm CSP (nếu có) vào CSP_VIOLATIONS."""
    context = browser.new_context(**kwargs)
    context.expose_function("__cspReport", lambda message: CSP_VIOLATIONS.append(message))  # Playwright cần hàm Python thật, không nhận phương thức dựng sẵn
    context.add_init_script(CSP_PROBE)
    return context

def http(method: str, url: str, payload: Any = None, token: str | None = None, timeout: float = 60) -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:  # type: ignore[attr-defined]
        body = error.read()
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, body.decode("utf-8", "replace")


def wait_for(predicate, timeout: float, interval: float = 0.5, what: str = "điều kiện") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return
        except OSError:  # máy chủ chưa nhận kết nối
            pass
        time.sleep(interval)
    raise TimeoutError(f"Quá {timeout:.0f} giây khi đợi {what}")


def admin_ui_checks(browser: Any, base: str, admin_token: str) -> None:
    """Trang /quan-tri bằng trình duyệt thật: đăng nhập, mở/đóng phiên, cấu hình, tạo mã (kể cả ghi chú chứa mã độc), thu hồi."""
    context = new_ctx(browser, viewport={"width": 1280, "height": 900})
    page = context.new_page()
    problems: list[str] = []
    # Chromium ghi log cho mọi phản hồi 4xx; lần nhập sai token ở dưới là CÓ CHỦ ĐÍCH nên chỉ bỏ qua đúng dòng 401 đó.
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" and "status of 401" not in m.text else None)
    page.on("pageerror", lambda e: problems.append(str(e)))
    response = page.goto(f"{base}/quan-tri", wait_until="load", timeout=60000)
    csp = response.headers.get("content-security-policy", "") if response else ""
    check("trang quản trị: có CSP chặn script/style inline", "script-src 'self'" in csp and "unsafe-inline" not in csp)

    page.fill("#token-input", "mat-khau-sai-hoan-toan")
    page.click("#btn-login")
    page.wait_for_selector("#login-error:not(.hidden)", timeout=15000)
    check("trang quản trị: mật khẩu sai bị từ chối bằng thông báo tiếng Việt", "không đúng" in page.locator("#login-error").inner_text())
    check("trang quản trị: ô tài khoản có sẵn 'admin'", page.locator("#user-input").input_value() == "admin")
    page.fill("#user-input", "nguoi-la")
    page.fill("#token-input", admin_token)
    page.click("#btn-login")
    page.wait_for_timeout(2000)  # (wait_js thay wait_for_function vì cái sau dùng eval nên bị CSP chặn)
    check("trang quản trị: đúng mật khẩu nhưng sai tài khoản vẫn bị từ chối",
          page.locator("#admin-app").is_hidden() and "không đúng" in page.locator("#login-error").inner_text())
    page.fill("#user-input", "admin")

    # "Ghi nhớ trên máy này" KHÔNG còn tick sẵn — phải bật tường minh thì bài kiểm ở dưới mới có nghĩa.
    page.check("#remember-token")
    page.fill("#token-input", admin_token)
    page.click("#btn-login")
    page.wait_for_selector("#admin-app:not(.hidden)", timeout=30000)
    wait_js(page, "document.getElementById('kpi-version').textContent.trim().startsWith('v')", timeout=15000)
    check("trang quản trị: đăng nhập đúng token, thấy phiên bản và vòng",
          page.locator("#kpi-version").inner_text().startswith("v") and page.locator("#session-badge").inner_text() in ("Đang mở", "Đang đóng"),
          f"{page.locator('#kpi-version').inner_text()} / {page.locator('#session-badge').inner_text()}")

    def state() -> dict:
        return http("GET", f"{base}/api/community/admin/state", token=admin_token)[1]

    # đóng phiên (vòng đang có người gửi thì phải bấm lần hai)
    was_open = state()["open"]
    toggle = page.locator("#btn-toggle-session")
    toggle.click()
    if "lần nữa" in toggle.inner_text():
        toggle.click()
    wait_for(lambda: state()["open"] != was_open, 30, 0.5, "phiên đổi trạng thái")
    check("trang quản trị: nút mở/đóng phiên đổi đúng trạng thái trên máy chủ", state()["open"] != was_open)
    if not state()["open"]:
        page.locator("#btn-toggle-session").click()
        wait_for(lambda: state()["open"], 30, 0.5, "phiên mở lại")

    # cấu hình
    page.fill("#pf-k_target", "4")
    page.click("#btn-save-policy")
    wait_for(lambda: state()["policy"]["k_target"] == 4, 30, 0.5, "K = 4")
    check("trang quản trị: lưu cấu hình K", state()["policy"]["k_target"] == 4)
    page.fill("#pf-round_ttl_seconds", "20")
    page.click("#btn-save-policy")
    wait_for(lambda: state()["policy"]["round_ttl_seconds"] == 1200, 30, 0.5, "hạn vòng 20 phút")
    check("trang quản trị: đổi đơn vị phút → giây đúng", state()["policy"]["round_ttl_seconds"] == 1200)
    page.fill("#pf-k_target", "1")
    page.click("#btn-save-policy")
    page.wait_for_selector("#policy-error:not(.hidden)", timeout=15000)
    check("trang quản trị: giá trị ngoài giới hạn bị chặn ở biểu mẫu, máy chủ không đổi",
          state()["policy"]["k_target"] == 4, page.locator("#policy-error").inner_text())
    page.click("#btn-reset-policy")
    page.click("#btn-save-policy")
    wait_for(lambda: state()["policy"]["k_target"] == 3, 30, 0.5, "về mặc định")

    # tạo mã, ghi chú chứa mã độc phải hiện như chữ thường
    page.evaluate("window.__xss = 0")
    page.fill("#codes-count", "2")
    page.fill("#codes-uses", "2")
    page.fill("#codes-label", '<img src=x onerror="window.__xss=1"><b>x</b>')
    page.click("#btn-create-codes")
    page.wait_for_selector("#codes-result:not(.hidden)", timeout=30000)
    shown = [li.inner_text().split("\n")[0].strip() for li in page.locator("#codes-list li").all()]
    check("trang quản trị: tạo mã và hiển thị đúng 2 mã", len(shown) == 2 and all(len(c.replace("-", "")) == 10 for c in shown), str(shown))
    first_join = http("POST", f"{base}/api/community/join", {"code": shown[0]})
    check("trang quản trị: mã vừa tạo dùng được để vào phiên", first_join[0] == 200)
    wait_js(page, "document.getElementById('codes-body').textContent.includes('onerror')", timeout=30000)
    check("trang quản trị: ghi chú chứa HTML hiện như văn bản, không chạy mã",
          page.evaluate("window.__xss") == 0 and page.locator("#codes-body img").count() == 0)
    link = page.locator("#codes-list li .code-link").first.inner_text()
    check("trang quản trị: link tham gia có mã và trỏ trang Học liên kết", "?code=" in link and link.endswith("#hoc-lien-ket"), link)

    # thu hồi một mã (nút hai bước)
    revoke = page.locator("#codes-body [data-revoke]").first
    revoke.click()
    revoke.click()
    wait_for(lambda: any(c["revoked"] for c in state()["codes"]), 30, 0.5, "mã bị thu hồi")
    check("trang quản trị: thu hồi mã", any(c["revoked"] for c in state()["codes"]))

    # trang công khai điền sẵn mã từ link rồi xóa khỏi thanh địa chỉ
    public = new_ctx(browser, viewport={"width": 1280, "height": 900}).new_page()
    public.goto(f"{base}/?code={shown[1]}#hoc-lien-ket", wait_until="load", timeout=60000)
    public.wait_for_selector("#comm-code", timeout=30000)
    wait_js(public, "document.getElementById('comm-code').value.length > 0", timeout=15000)
    check("link tham gia: điền sẵn mã và xóa mã khỏi URL",
          public.locator("#comm-code").input_value().replace("-", "") == shown[1].replace("-", "") and "code=" not in public.url,
          public.url)

    # tải lại giữ phiên đăng nhập trong tab; đăng xuất thì token bị xóa
    page.reload(wait_until="load")
    page.wait_for_selector("#admin-app:not(.hidden)", timeout=30000)
    check("trang quản trị: tải lại vẫn giữ đăng nhập trong tab", True)

    # "Ghi nhớ trên máy này" (đã bật tường minh lúc đăng nhập): tab mới của cùng trình duyệt vào thẳng, không hỏi token
    second = context.new_page()
    second.goto(f"{base}/quan-tri", wait_until="load", timeout=60000)
    second.wait_for_selector("#admin-app:not(.hidden)", timeout=30000)
    check("trang quản trị: ghi nhớ trên máy, tab mới vào thẳng không cần nhập token", True)
    second.close()

    page.click("#btn-logout")
    page.wait_for_selector("#login-panel:not(.hidden)", timeout=15000)
    check("trang quản trị: đăng xuất xóa token khỏi tab và khỏi bộ nhớ máy",
          page.evaluate("sessionStorage.getItem('cropfed.admin.token')") is None
          and page.evaluate("localStorage.getItem('cropfed.admin.token')") is None)

    # bỏ chọn ghi nhớ: token không được lưu lâu dài
    page.uncheck("#remember-token")
    page.fill("#token-input", admin_token)
    page.click("#btn-login")
    page.wait_for_selector("#admin-app:not(.hidden)", timeout=30000)
    check("trang quản trị: bỏ chọn ghi nhớ thì không lưu vào bộ nhớ máy",
          page.evaluate("localStorage.getItem('cropfed.admin.token')") is None)
    page.click("#btn-logout")
    page.wait_for_selector("#login-panel:not(.hidden)", timeout=15000)

    # #token= đã bị GỠ: link kiểu đó phải bị bỏ qua (không tự đăng nhập, không ghi mật khẩu vào bộ nhớ máy)
    # nhưng fragment vẫn bị xóa để mật khẩu trong link cũ không nằm lại trên thanh địa chỉ.
    stale = new_ctx(browser, viewport={"width": 1280, "height": 900}).new_page()
    stale.goto(f"{base}/quan-tri#token={admin_token}", wait_until="load", timeout=60000)
    stale.wait_for_selector("#login-panel:not(.hidden)", timeout=30000)
    check("trang quản trị: link #token= không còn được nhận và không lưu mật khẩu",
          stale.locator("#admin-app").is_hidden()
          and "token=" not in stale.url and admin_token not in stale.url
          and stale.evaluate("localStorage.getItem('cropfed.admin.token')") is None
          and stale.evaluate("sessionStorage.getItem('cropfed.admin.token')") is None,
          stale.url)
    stale.close()

    narrow = new_ctx(browser, viewport={"width": 320, "height": 700}).new_page()
    narrow.goto(f"{base}/quan-tri", wait_until="load", timeout=60000)
    narrow.fill("#token-input", admin_token)
    narrow.click("#btn-login")
    narrow.wait_for_selector("#admin-app:not(.hidden)", timeout=30000)
    overflow = narrow.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check("trang quản trị: 320 px không tràn ngang", overflow <= 1, f"lệch {overflow}px")
    check("trang quản trị: không có lỗi console", not problems, "; ".join(problems)[:200])
    context.close()


def training_gate_checks(browser: Any, base: str, admin_token: str) -> None:
    """Cổng huấn luyện: API trả 401/422 đúng, trang Phụ lục khóa nút khi chưa đăng nhập và mở khi đã đăng nhập ở trang quản trị."""
    train = {"epochs": 2, "smoke": True, "seed": 1}
    check("huấn luyện: không đăng nhập bị 401 (Centralized)", http("POST", f"{base}/api/centralized/train", train)[0] == 401)
    check("huấn luyện: không đăng nhập bị 401 (Flower)", http("POST", f"{base}/api/federated/train", {"engine": "network"})[0] == 401)
    check("huấn luyện: không đăng nhập bị 401 (Dừng)", http("POST", f"{base}/api/train/stop", {})[0] == 401)
    over = http("POST", f"{base}/api/centralized/train", {**train, "epochs": 11}, admin_token)
    check("huấn luyện: đã đăng nhập nhưng vượt trần epoch bị 422, không có job nào chạy", over[0] == 422, str(over[1])[:90])
    status = http("GET", f"{base}/api/status")[1]
    check("huấn luyện: /api/status nêu cần quyền quản trị và các trần", status["training"]["requires_admin"] is True
          and status["training"]["limits"]["max_rounds"] >= 1, json.dumps(status["training"]["limits"]))

    posts: list[str] = []
    context = new_ctx(browser, viewport={"width": 1280, "height": 900})
    page = context.new_page()
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST" and "/train" in r.url else None)
    page.goto(f"{base}/#phu-luc", wait_until="load", timeout=60000)
    page.wait_for_selector("#train-gate:not(.hidden)", timeout=30000)
    check("phụ lục D: chưa đăng nhập thì hiện khung 'Cần quyền quản trị' kèm liên kết", page.locator("#train-gate a[href='/quan-tri']").count() == 1)
    page.click("#btn-train-start")
    page.wait_for_timeout(600)
    check("phụ lục D: bấm Bắt đầu khi chưa đăng nhập không gửi yêu cầu nào lên máy chủ", not posts, str(posts))
    check("phụ lục D: trần số vòng/epoch từ máy chủ được áp lên ô nhập",
          int(page.locator("#cent-epochs").get_attribute("max")) <= status["training"]["limits"]["max_centralized_epochs"])

    login = context.new_page()  # cùng trình duyệt: đăng nhập ở tab khác với "Ghi nhớ trên máy này"
    login.goto(f"{base}/quan-tri", wait_until="load", timeout=60000)
    login.check("#remember-token")  # không còn tick sẵn: tab phụ lục chỉ tự mở khóa khi bản ghi nhớ nằm trong localStorage
    login.fill("#token-input", admin_token)
    login.click("#btn-login")
    login.wait_for_selector("#admin-app:not(.hidden)", timeout=30000)
    page.wait_for_selector("#train-gate", state="hidden", timeout=15000)
    check("phụ lục D: đăng nhập ở tab quản trị thì khung khóa tự biến mất (không cần tải lại)", True)
    login.close()
    context.close()


def main() -> int:
    manifest_path = reg.abs_path(reg.COMMUNITY_MANIFEST_JSON)
    if not manifest_path.is_file() or "eval_features" not in json.loads(manifest_path.read_text(encoding="utf-8")):
        print("Thiếu artifact cộng đồng: chạy scripts/export_community_feature_model.py và build_community_eval_features.py", file=sys.stderr)
        return 2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Cần Playwright (pip install playwright && playwright install chromium)", file=sys.stderr)
        return 2

    gallery_dir = reg.abs_path(reg.DEMO_GALLERY_DIR)
    catalog = json.loads(reg.abs_path(reg.DEMO_GALLERY_CATALOG_JSON).read_text(encoding="utf-8"))
    samples = [(gallery_dir / item["file"], item["label"]) for item in catalog["items"]]
    groups = [samples[0::3], samples[1::3], samples[2::3]]

    state_dir = tempfile.mkdtemp(prefix="community-e2e-")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "COMMUNITY_SECRET": SECRET, "COMMUNITY_ADMIN_TOKEN": ADMIN, "COMMUNITY_STATE_DIR": state_dir,
           "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    log_path = Path(state_dir) / "server.log"  # ghi ra tệp: ống (PIPE) đầy sẽ làm máy chủ treo khi không ai đọc
    log_file = log_path.open("wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
    try:
        wait_for(lambda: http("GET", f"{base}/api/community/status")[0] == 200, 90, 1, "máy chủ khởi động")
        status = http("GET", f"{base}/api/community/status")[1]
        check("trạng thái công khai: bật, mặc định ĐÓNG", status["enabled"] is True and status["open"] is False, json.dumps({k: status[k] for k in ("enabled", "open", "version")}))
        check("tách khỏi dữ liệu admin: không có token thì 401", http("GET", f"{base}/api/community/admin/state")[0] == 401)

        code, _ = http("POST", f"{base}/api/community/admin/settings",
                       {"open": True, "changes": {"k_target": 3, "k_min": 3, "clip_norm": 1.0, "round_ttl_seconds": 1800}}, ADMIN)
        check("quản trị mở phiên", code == 200)
        _, made = http("POST", f"{base}/api/community/admin/codes", {"count": 4, "max_uses": 1, "ttl_hours": 2, "label": "e2e"}, ADMIN)
        codes: list[str] = made["codes"]

        seen_requests: list[dict[str, Any]] = []
        long_tasks: list[float] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            pages = []
            for index in range(3):
                context = new_ctx(browser, viewport={"width": 1280, "height": 900})
                page = context.new_page()
                console_errors: list[str] = []
                page.on("console", lambda m, bag=console_errors: bag.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e, bag=console_errors: bag.append(str(e)))
                page.on("request", lambda r, bag=seen_requests: bag.append({
                    "url": r.url, "method": r.method, "bytes": len(r.post_data_buffer or b""),
                    "type": (r.headers.get("content-type") or "")}))
                page.add_init_script("""
                  window.__longTasks = [];
                  try { new PerformanceObserver(l => l.getEntries().forEach(e => window.__longTasks.push(e.duration)))
                        .observe({ entryTypes: ['longtask'] }); } catch (e) {}
                """)
                page.goto(f"{base}/#hoc-lien-ket")
                page.wait_for_selector("#comm-main:not(.hidden)", timeout=30000)
                pages.append((context, page, console_errors))

            check("giao diện nhận ra phiên đang mở", all(p.locator("#comm-badge").inner_text().strip() == "Đang mở" for _, p, _ in pages))

            for index, (context, page, _errors) in enumerate(pages):
                page.fill("#comm-code", codes[index])
                page.click("#btn-comm-join")
                page.wait_for_selector("#comm-work-box:not(.hidden)", timeout=30000)
            check("ba client tham gia bằng ba mã riêng", True)

            for index, (context, page, _errors) in enumerate(pages):
                files = [str(path) for path, _label in groups[index]]
                if index == 0:
                    # Chọn nhóm TRƯỚC khi thêm ảnh: mọi ảnh thêm vào phải được gán sẵn nhóm đó (rồi vẫn sửa được từng ảnh bên dưới)
                    page.select_option("#comm-bulk-label", value=groups[0][0][1])
                page.set_input_files("#comm-files", files)
                page.wait_for_selector("select[data-label]", timeout=15000)
                if index == 0:
                    values = page.eval_on_selector_all("select[data-label]", "els => els.map(e => e.value)")
                    check("chọn nhóm trước: mọi ảnh thêm vào được gán sẵn nhóm đó",
                          len(values) == len(files) and set(values) == {groups[0][0][1]}, str(values[:3]))
                    page.select_option("#comm-bulk-label", value="")
                zone = page.locator("#comm-dropzone")
                if index == 0:
                    check("ô thêm ảnh bằng tiếng Việt (không còn 'Choose Files')",
                          "Thêm ảnh lá" in zone.inner_text() and "Choose" not in zone.inner_text())
                selects = page.locator("select[data-label]")
                for position, (_path, label) in enumerate(groups[index]):
                    selects.nth(position).select_option(value=label)
                wait_js(page, "!document.getElementById('btn-comm-train').disabled", timeout=15000)

            for _context, page, _errors in pages:
                page.evaluate("window.__longTasks.length = 0")  # chỉ đo từ lúc bấm huấn luyện (bỏ phần nạp trang)
            started = time.time()
            for _context, page, _errors in pages:
                page.click("#btn-comm-train")
            for _context, page, _errors in pages:
                page.wait_for_selector("#comm-local-result:not(.hidden)", timeout=240000)
            train_seconds = time.time() - started
            texts = [p.locator("#comm-local-result").inner_text() for _, p, _ in pages]
            check("cả ba client huấn luyện xong trong trình duyệt", all("Huấn luyện xong" in t for t in texts), f"{train_seconds:.1f} s (gồm tải mô hình)")
            for _context, page, _errors in pages:
                long_tasks.extend(page.evaluate("window.__longTasks"))

            for _context, page, _errors in pages:
                page.click("#btn-comm-submit")
                page.wait_for_selector("#comm-submit-result:not(.hidden)", timeout=60000)
            results = [p.locator("#comm-submit-result").inner_text() for _, p, _ in pages]
            check("cả ba cập nhật được máy chủ nhận", all("Đã gửi" in t for t in results), results[-1].replace("\n", " ")[:160])

            _, history = http("GET", f"{base}/api/community/history")
            check("vòng 1 gộp thành phiên bản 1", history["current_version"] == 1 and history["totals"]["accepted"] == 1, json.dumps(history["totals"]))
            for _context, page, _errors in pages:
                wait_js(page, "document.getElementById('comm-kpi-version').textContent.trim() === 'v1'", timeout=60000)
            check("cả ba trình duyệt cùng thấy phiên bản mới v1", True)

            v0, v1 = history["versions"][0]["metrics"]["accuracy"], history["versions"][1]["metrics"]["accuracy"]
            check("cổng: độ chính xác giám sát v1 không thoái lui quá ngưỡng", v1 >= v0 - 0.005, f"v0 {v0:.4f} → v1 {v1:.4f}")

            # ── vòng 2: một client đầu độc qua API cùng hai client trung thực (qua API) ──
            _, join_a = http("POST", f"{base}/api/community/join", {"code": codes[3]})
            token_poison = join_a["token"]
            _, round_view = http("GET", f"{base}/api/community/round", token=token_poison)
            parent = core.decode_head(round_view["head"])
            poisoned = core.as_head(-parent[core.WEIGHT_KEY] * 40.0, parent[core.BIAS_KEY] + 5.0)
            counts = [0] * 38
            counts[0] = 30
            code_status, answer = http("POST", f"{base}/api/community/update", {
                "round_id": round_view["round"]["round_id"], "parent_sha": round_view["round"]["parent_sha"],
                "head": core.encode_head(poisoned), "num_examples": 30, "label_counts": counts, "local": {}, "dp_applied": False,
            }, token_poison)
            check("head đầu độc: bị cắt chuẩn ở phía máy chủ", code_status == 200 and answer["clipped"] and answer["norm_used"] <= 1.0001,
                  f"chuẩn gốc {answer.get('norm')}, sau cắt {answer.get('norm_used')}")
            _, after = http("GET", f"{base}/api/community/history")
            check("đầu độc không làm mô hình hiện hành đổi trước khi đủ K", after["current_version"] == 1)

            # ── không có byte ảnh nào rời máy ──
            posts = [r for r in seen_requests if r["method"] == "POST"]
            image_posts = [r for r in posts if r["type"].startswith(("multipart", "image/"))]
            check("không có yêu cầu nào tải ảnh lên", not image_posts and all(r["bytes"] < 400_000 for r in posts),
                  f"{len(posts)} POST, lớn nhất {max((r['bytes'] for r in posts), default=0)} byte")
            check("chỉ có đường /api/community nhận dữ liệu", all("/api/community/" in r["url"] for r in posts))

            slow = [d for d in long_tasks if d >= 100]
            check("luồng chính không bị chặn lâu khi huấn luyện (Worker)", len(slow) == 0,
                  f"{len(long_tasks)} long task, {len(slow)} ≥ 100 ms, dài nhất {max(long_tasks, default=0):.0f} ms")
            check("không có lỗi console trong ba trang", all(not e for _, _, e in pages), "; ".join(sum((e for _, _, e in pages), []))[:200])

            # ── bố cục 320 px không tràn ngang ──
            narrow = new_ctx(browser, viewport={"width": 320, "height": 700})
            small = narrow.new_page()
            small.goto(f"{base}/#hoc-lien-ket")
            small.wait_for_selector("#comm-main:not(.hidden)", timeout=30000)
            overflow = small.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            check("320 px: không tràn ngang", overflow <= 1, f"lệch {overflow}px")
            narrow.close()

            # ── chẩn đoán bằng mô hình cộng đồng (qua API, ảnh gallery) ──
            boundary = "----e2e"
            image_path = samples[0][0]
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() \
                + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            request = urllib.request.Request(f"{base}/api/community/predict", data=body, method="POST")
            request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            with urllib.request.urlopen(request, timeout=90) as response:
                prediction = json.loads(response.read())
            check("dự đoán bằng mô hình cộng đồng (phiên bản 1) có so sánh FedAvg v7",
                  prediction["ready"] and prediction["version"] == 1 and "baseline" in prediction, prediction["predicted_class"])

            admin_ui_checks(browser, base, ADMIN)
            training_gate_checks(browser, base, ADMIN)

            for context, _page, _errors in pages:
                context.close()
            browser.close()
    except Exception as error:  # noqa: BLE001
        check("kịch bản chạy hết không ngoại lệ", False, f"{type(error).__name__}: {error}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    check("không có vi phạm CSP trong mọi trang đã mở", not CSP_VIOLATIONS, "; ".join(CSP_VIOLATIONS[:3]))
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} kiểm tra đạt")
    if failed:
        log_file.flush()
        print("Nhật ký máy chủ (cuối):\n" + log_path.read_text(encoding="utf-8", errors="replace")[-1500:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

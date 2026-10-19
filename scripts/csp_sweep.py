"""Quét vi phạm CSP của trang chính: mở đủ 6 bước, mọi tab con, mọi mục gấp, chẩn đoán một ảnh mẫu, nền tối, bằng Chromium/Firefox/WebKit.

Máy chủ thử chạy với CSP ở chế độ `enforce` (mặc định). Một vi phạm nghĩa là có thứ thật sự bị chặn.
Chạy:  python scripts/csp_sweep.py [--browser chromium|firefox|webkit|all]
Thoát mã 0 nếu không có vi phạm/lỗi console, 1 nếu có, 2 nếu thiếu điều kiện chạy.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e2e_community import ADMIN, CSP_VIOLATIONS, SECRET, free_port, http, new_ctx, wait_for  # noqa: E402

ROUTES = ["tong-quan", "non-iid", "ket-qua", "chan-doan", "huan-luyen", "bang-chung"]


def sweep(browser_type: Any, base: str, label: str) -> list[str]:
    problems: list[str] = []
    browser = browser_type.launch()
    for width, mobile in ((1440, False), (390, True)):
        # Firefox không hỗ trợ is_mobile: chỉ đổi khổ màn hình
        mobile_opts = {"is_mobile": True, "has_touch": True} if mobile and browser_type.name != "firefox" else {}
        context = new_ctx(browser, viewport={"width": width, "height": 900}, **mobile_opts)
        page = context.new_page()
        page.on("console", lambda m, p=problems: p.append(f"console {m.type}: {m.text[:140]}") if m.type == "error" else None)
        page.on("pageerror", lambda e, p=problems: p.append(f"pageerror: {str(e)[:140]}"))
        for route in ROUTES:
            page.goto(f"{base}/#{route}", wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(1200)
            for tab in page.locator(f"#page-{route} [role='tab']").all()[:8]:
                try:
                    tab.click(timeout=3000)
                    page.wait_for_timeout(700)
                except Exception:  # noqa: BLE001 - tab ẩn ở khổ hẹp: bỏ qua
                    pass
            for summary in page.locator(f"#page-{route} details > summary").all()[:14]:
                try:
                    summary.click(timeout=3000)
                    page.wait_for_timeout(250)
                except Exception:  # noqa: BLE001
                    pass
            page.wait_for_timeout(500)
        # Chẩn đoán một ảnh mẫu (dựng thanh độ tin cậy, Grad-CAM, xem trước) rồi nền tối
        page.goto(f"{base}/#chan-doan", wait_until="networkidle", timeout=90000)
        page.locator("#gallery-grid .gallery-item").first.click()
        page.wait_for_timeout(3500)
        page.goto(f"{base}/#tong-quan", wait_until="networkidle")
        theme = page.locator("#btn-theme")
        if theme.count():
            theme.first.click()
            page.wait_for_timeout(400)
        context.close()
    browser.close()
    problems.extend(f"CSP: {v}" for v in CSP_VIOLATIONS)
    CSP_VIOLATIONS.clear()
    return [f"[{label}] {item}" for item in problems]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default="all", choices=["chromium", "firefox", "webkit", "all"])
    args = parser.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Cần Playwright (pip install playwright && playwright install)", file=sys.stderr)
        return 2

    state_dir = tempfile.mkdtemp(prefix="csp-sweep-")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {key: value for key, value in os.environ.items() if key != "CROPFED_CSP_MODE"}
    env.update({"COMMUNITY_SECRET": SECRET, "COMMUNITY_ADMIN_TOKEN": ADMIN, "COMMUNITY_STATE_DIR": state_dir, "PYTHONUNBUFFERED": "1"})
    log_file = (Path(state_dir) / "server.log").open("wb")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "web.app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                              cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    problems: list[str] = []
    try:
        wait_for(lambda: http("GET", f"{base}/api/status")[0] == 200, 120, 1, "máy chủ khởi động")
        csp = http("GET", f"{base}/api/status")  # chỉ để chắc máy chủ trả lời
        del csp
        with sync_playwright() as pw:
            names = ["chromium", "firefox", "webkit"] if args.browser == "all" else [args.browser]
            for name in names:
                try:
                    found = sweep(getattr(pw, name), base, name)
                except Exception as error:  # noqa: BLE001 - một trình duyệt hỏng không được che kết quả của các trình duyệt khác
                    found = [f"[{name}] không chạy được: {type(error).__name__}: {str(error)[:160]}"]
                print(f"{name}: {'sạch' if not found else f'{len(found)} vấn đề'}")
                problems.extend(found)
    finally:
        server.terminate()
        try:
            server.wait(15)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
    for item in dict.fromkeys(problems):
        print("  " + item)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

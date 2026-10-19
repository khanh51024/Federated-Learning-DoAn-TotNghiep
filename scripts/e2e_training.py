"""Kiểm thử đầu-cuối bước 5 (huấn luyện trên máy chủ) bằng trình duyệt thật: phiên Flower 5 client và Centralized.

Kiểm điều người xem phải thấy: khung "Bằng chứng máy đã tính thật" (6 tiến trình, chuỗi mã băm mô hình chung, mức đổi của tham số học
được), bảng số liệu từng vòng có dòng thật, KPI đúng theo chế độ (Centralized không hiện client/RTT của phiên Flower trước), nút
Bắt đầu đổi nhãn khi cần đăng nhập, và không có vi phạm CSP.

Chạy:  python scripts/e2e_training.py        (cần Playwright + Chromium; xem extra `e2e` trong pyproject.toml)
Thoát mã 0 nếu mọi kiểm tra đạt, 1 nếu có kiểm tra hỏng, 2 nếu thiếu điều kiện chạy.

Máy chủ thử chạy với WEB_RUNS_DIR/NETWORK_SMOKE_RUNS_DIR trỏ vào tmp/ (registry bỏ qua thư mục này), nên các tệp .pt do phiên huấn luyện
sinh ra KHÔNG làm lệch artifacts/registry.json và không để lại rác trong artifacts/web_runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e2e_community import ADMIN, CHECKS, CSP_VIOLATIONS, SECRET, check, free_port, http, new_ctx, wait_for, wait_js  # noqa: E402

BOOT = """
import sys
sys.path.insert(0, {root!r})
from src import registry as r
r.WEB_RUNS_DIR = "tmp/e2e-runs"
r.NETWORK_SMOKE_RUNS_DIR = "tmp/e2e-runs/network_smoke"
r.TRAINING_AUDIT_JSONL = "tmp/e2e-runs/training_audit.jsonl"
import uvicorn
uvicorn.run("web.app:app", host="127.0.0.1", port={port}, log_level="warning")
"""


def run_checks(base: str, outdir: Path | None) -> None:
    from playwright.sync_api import sync_playwright

    credential = json.dumps({"u": "admin", "p": ADMIN})
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        # 1) Chưa đăng nhập: nút đổi nhãn nhưng vẫn bấm được và không gửi yêu cầu nào.
        locked = new_ctx(browser, viewport={"width": 1440, "height": 1000})
        page = locked.new_page()
        posts: list[str] = []
        page.on("request", lambda r: posts.append(r.url) if r.method == "POST" and "/train" in r.url else None)
        page.goto(f"{base}/#huan-luyen", wait_until="load", timeout=60000)
        wait_js(page, "!document.getElementById('train-gate').classList.contains('hidden')", 30000)
        label = page.locator("#btn-train-start").inner_text()
        check("chưa đăng nhập: nút Bắt đầu đổi nhãn nhưng không bị vô hiệu hóa",
              "Đăng nhập để bắt đầu" in label and page.locator("#btn-train-start").is_enabled(), label)
        page.click("#btn-train-start")
        page.wait_for_timeout(500)
        check("chưa đăng nhập: bấm nút không gửi yêu cầu huấn luyện", not posts, str(posts))
        callout = page.locator("#page-huan-luyen .callout-warn", has_text="kiểm chứng hệ thống")
        check("khung giới thiệu nói rõ kiểm chứng hệ thống, ảnh nhiễu ngẫu nhiên",
              callout.count() == 1 and "ảnh nhiễu ngẫu nhiên" in callout.first.inner_text())
        locked.close()

        # 2) Đã đăng nhập: chạy Flower 5 client qua giao diện.
        context = new_ctx(browser, viewport={"width": 1440, "height": 1000})
        context.add_init_script(f"try {{ sessionStorage.setItem('cropfed.admin.token', {json.dumps(credential)}); }} catch (e) {{}}")
        page = context.new_page()
        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text[:160]) if m.type == "error" else None)
        page.goto(f"{base}/#huan-luyen", wait_until="load", timeout=60000)
        wait_js(page, "!document.getElementById('train-gate') || document.getElementById('train-gate').classList.contains('hidden')", 30000)
        check("đã đăng nhập: khung khóa ẩn, nút ghi 'Bắt đầu'", "Bắt đầu" in page.locator("#btn-train-start").inner_text()
              and "Đăng nhập" not in page.locator("#btn-train-start").inner_text())
        wait_js(page, "document.getElementById('topology').dataset.phase", 30000)
        check("chưa có phiên: sơ đồ ở trạng thái thu gọn", page.locator("#topology").get_attribute("data-phase") == "idle")
        page.click("#train-mode-fed")
        page.click("#btn-train-start")
        wait_js(page, "document.getElementById('proof-procs').textContent.startsWith('6/6')", 120000)
        procs_text = page.locator("#proof-procs").inner_text()
        check("Flower: khung bằng chứng hiện 6 tiến trình đang chạy", procs_text.startswith("6/6"), procs_text)
        check("Flower: KPI client/RTT hiện, KPI Centralized ẩn",
              page.locator("#kpi-fed").is_visible() and page.locator("#kpi-cent").is_hidden())
        if outdir:
            page.locator("#page-huan-luyen").screenshot(path=str(outdir / "flower-running.png"))
        wait_js(page, "document.getElementById('train-live-badge').textContent.includes('Hoàn tất')", 240000)
        page.locator("details:has(#train-progress-table)").evaluate("el => { el.open = true; }")
        rows = page.locator("#train-progress-body tr")
        check("Flower: bảng số liệu có đủ 2 dòng vòng (không còn 'chờ kết quả vòng đầu tiên')", rows.count() == 2, str(rows.count()))
        row_text = [rows.nth(i).inner_text() for i in range(rows.count())]
        check("Flower: mỗi vòng ghi 5/0/0 client và trạng thái 'Đã gộp'",
              all("5 / 0 / 0" in text and "Đã gộp" in text for text in row_text), " | ".join(row_text)[:200])
        hashes = [rows.nth(i).locator(".hash-chip").get_attribute("title") or "" for i in range(rows.count())]
        check("Flower: mã băm mô hình chung có 64 ký tự và đổi giữa hai vòng",
              len(hashes) == 2 and all(len(h) == 64 for h in hashes) and hashes[0] != hashes[1], str([h[:10] for h in hashes]))
        note = page.locator("#proof-note").inner_text()
        check("Flower: khung bằng chứng xác nhận chuỗi mã băm khớp", "Chuỗi mã băm khớp qua 2 vòng" in note, note[:140])
        check("Flower: khung bằng chứng nêu mức đổi của tham số học được (nhỏ)", "Tham số học được chỉ đổi" in note)
        procs_after = page.locator("#proof-procs").inner_text()
        check("Flower: sau phiên 6 tiến trình đã thoát với mã 0", "đã thoát (mã 0)" in procs_after, procs_after)
        check("Flower: nút Bắt đầu trở lại bình thường", page.locator("#btn-train-start").is_enabled()
              and "Bắt đầu" in page.locator("#btn-train-start").inner_text())
        if outdir:
            page.locator("#page-huan-luyen").screenshot(path=str(outdir / "flower-done.png"))

        # 3) Centralized sau Flower: không được hiện client/RTT của phiên Flower trước.
        page.click("#train-mode-cent")
        page.fill("#cent-epochs", "2")
        check("chọn Centralized: sơ đồ và KPI client/RTT ẩn, KPI epoch hiện, khung bằng chứng ẩn",
              page.locator("#topology").is_hidden() and page.locator("#kpi-fed").is_hidden()
              and page.locator("#kpi-cent").is_visible() and page.locator("#train-proof").is_hidden())
        page.click("#btn-train-start")
        badge = page.locator("#train-live-badge").inner_text()
        check("bấm Bắt đầu: huy hiệu của phiên trước được xóa ngay", "Hoàn tất" not in badge, badge)
        wait_js(page, "document.getElementById('dash-epochs').textContent.replace(/ /g, '') === '2/2'", 240000)
        wait_js(page, "document.getElementById('train-live-badge').textContent.includes('Hoàn tất')", 60000)
        epochs_text = page.locator("#dash-epochs").inner_text()
        check("Centralized: KPI epoch đã xong = 2/2", epochs_text.replace(" ", "") == "2/2", epochs_text)
        rows = page.locator("#train-progress-body tr")
        check("Centralized: bảng có 2 dòng epoch, không có cột độ chính xác",
              rows.count() == 2 and "Chính xác" not in page.locator("#train-progress-head").inner_text())
        message = page.locator("#train-progress-detail").inner_text()
        check("Centralized: thông báo không lấy độ chính xác làm tiêu điểm", "Acc" not in message and "%" not in message, message)
        if outdir:
            page.locator("#page-huan-luyen").screenshot(path=str(outdir / "centralized-done.png"))
        check("không có lỗi console ở bước 5", not console_errors, "; ".join(console_errors[:3]))
        context.close()
        browser.close()


def main() -> int:
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("Cần Playwright (pip install playwright && playwright install chromium)", file=sys.stderr)
        return 2
    outdir_env = os.environ.get("E2E_TRAINING_SHOTS")
    outdir = Path(outdir_env) if outdir_env else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    state_dir = tempfile.mkdtemp(prefix="training-e2e-")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {key: value for key, value in os.environ.items() if key not in {"DATASET_ROOT", "PLANTVILLAGE_DATASET_ROOT"}}
    env.update({"COMMUNITY_SECRET": SECRET, "COMMUNITY_ADMIN_TOKEN": ADMIN, "COMMUNITY_STATE_DIR": state_dir,
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2"})
    log_path = Path(state_dir) / "server.log"
    log_file = log_path.open("wb")
    server = subprocess.Popen([sys.executable, "-c", BOOT.format(root=str(ROOT), port=port)], cwd=ROOT, env=env,
                              stdout=log_file, stderr=subprocess.STDOUT)
    try:
        wait_for(lambda: http("GET", f"{base}/api/status")[0] == 200, 120, 1, "máy chủ khởi động")
        run_checks(base, outdir)
        check("không có vi phạm CSP trong mọi trang đã mở", not CSP_VIOLATIONS, "; ".join(CSP_VIOLATIONS[:3]))
    except Exception as error:  # noqa: BLE001 - báo thành một kiểm tra hỏng thay vì làm mất kết quả đã có
        check("kịch bản chạy hết không ngoại lệ", False, f"{type(error).__name__}: {error}")
    finally:
        server.terminate()
        try:
            server.wait(15)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} kiểm tra đạt")
    if failed:
        print("Nhật ký máy chủ (cuối):\n" + log_path.read_text(encoding="utf-8", errors="replace")[-1500:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

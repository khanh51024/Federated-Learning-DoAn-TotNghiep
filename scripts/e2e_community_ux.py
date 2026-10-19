"""Kiểm thử trải nghiệm người tham gia (Client FL) bằng trình duyệt thật: phần bổ sung cho scripts/e2e_community.py.

Phủ các tình huống mà báo cáo review UI/UX 14/10/2026 (docs/UI_UX_REVIEW_2026-10-14.md, mục C) chỉ ra:
  - phiên đóng: có hướng dẫn nhìn thấy, nút khóa có lý do, mã điền sẵn từ liên kết nói đúng tình trạng;
  - nút khóa luôn kèm lý do (chưa đủ ảnh, đã gửi cho vòng này) kể cả sau khi tải lại trang;
  - thanh bốn cột mốc và thông báo "đã gộp" cho người đã gửi (kể cả sau khi tải lại);
  - mất phản hồi khi gửi: hỏi lại máy chủ, không báo "không nhận" khi máy chủ đã nhận; 429 có câu riêng;
  - tiến trình xử lý ảnh bị đặt lại: huấn luyện lại thành công mà không phải xóa ảnh;
  - trình duyệt thiếu OffscreenCanvas: cảnh báo ngay, trước khi nhập mã.

Chạy:  python scripts/e2e_community_ux.py     (cùng điều kiện với e2e_community.py)
Thoát mã 0 nếu mọi kiểm tra đạt, 1 nếu có kiểm tra hỏng, 2 nếu thiếu điều kiện chạy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import e2e_community as base_e2e  # noqa: E402
from scripts.e2e_community import ADMIN, CHECKS, CSP_PROBE, SECRET, check, free_port, http, wait_for, wait_js  # noqa: E402
from src import registry as reg  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

WORKER_SPY = """
  (() => {
    const Original = window.Worker;
    window.__workers = [];
    window.Worker = function (...args) { const w = new Original(...args); window.__workers.push(w); return w; };
    window.Worker.prototype = Original.prototype;
  })();
"""


def open_page(browser: Any, base: str, route: str = "/#huan-luyen", width: int = 390, script: str | None = None):
    context = base_e2e.new_ctx(browser, viewport={"width": width, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    if script:
        page.add_init_script(script)
    page.goto(f"{base}{route}", wait_until="load", timeout=60000)
    return context, page, errors


def add_labeled_images(page: Any, samples: list[tuple[Path, str]]) -> None:
    page.set_input_files("#comm-files", [str(path) for path, _label in samples])
    wait_js(page, f"document.querySelectorAll('select[data-label]').length >= {len(samples)}", timeout=15000)
    selects = page.locator("select[data-label]")
    for position, (_path, label) in enumerate(samples):
        selects.nth(position).select_option(value=label)


def text(page: Any, selector: str) -> str:
    return " ".join(page.locator(selector).inner_text().split())


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

    state_dir = tempfile.mkdtemp(prefix="community-ux-")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "COMMUNITY_SECRET": SECRET, "COMMUNITY_ADMIN_TOKEN": ADMIN, "COMMUNITY_STATE_DIR": state_dir,
           "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    log_file = (Path(state_dir) / "server.log").open("wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
    try:
        wait_for(lambda: http("GET", f"{base}/api/community/status")[0] == 200, 90, 1, "máy chủ khởi động")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()

            # ── 1. Phiên đóng (mặc định) ──
            context, page, errors = open_page(browser, base, "/?code=ABCDE-FGHJK#huan-luyen")
            page.wait_for_selector("#comm-main:not(.hidden)", timeout=30000)
            page.wait_for_selector("#comm-closed-note:not(.hidden)", timeout=15000)
            check("phiên đóng: câu giải thích nằm cạnh ô nhập mã, nhìn thấy được",
                  "đang đóng" in text(page, "#comm-closed-note") and page.locator("#comm-closed-note").is_visible())
            check("phiên đóng: có khối 'Cách tham gia' 4 bước", page.locator("#comm-howto li").count() == 4 and page.locator("#comm-howto").is_visible())
            check("phiên đóng: nút Tham gia khóa, mã vẫn được điền sẵn",
                  page.locator("#btn-comm-join").is_disabled() and page.locator("#comm-code").input_value().replace("-", "") == "ABCDEFGHJK")
            wait_js(page, "document.body.innerText.includes('Phiên đang đóng; nút Tham gia sẽ bật')", timeout=15000)
            check("liên kết ?code= khi đóng: thông báo khớp với nút bị khóa", True)
            overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            check("phiên đóng: 390 px không tràn ngang", overflow <= 1, f"lệch {overflow}px")
            context.close()

            # ── 2. Trình duyệt thiếu OffscreenCanvas ──
            # mở phiên trước để thấy cảnh báo không do phiên đóng
            code, _ = http("POST", f"{base}/api/community/admin/settings",
                           {"open": True, "changes": {"k_target": 3, "k_min": 3, "clip_norm": 1.0, "round_ttl_seconds": 1800}}, ADMIN)
            check("quản trị mở phiên", code == 200)
            _, made = http("POST", f"{base}/api/community/admin/codes", {"count": 6, "max_uses": 1, "ttl_hours": 2, "label": "ux"}, ADMIN)
            codes: list[str] = made["codes"]
            context, page, errors = open_page(browser, base, script="delete window.OffscreenCanvas;")
            page.wait_for_selector("#comm-unsupported:not(.hidden)", timeout=30000)
            check("thiếu OffscreenCanvas: cảnh báo ngay trước khi nhập mã, nút Tham gia khóa",
                  "OffscreenCanvas" in text(page, "#comm-unsupported") and page.locator("#btn-comm-join").is_disabled())
            context.close()

            # ── 3. Người tham gia thứ nhất: lý do nút khóa, thanh cột mốc ──
            context, page, errors = open_page(browser, base, script=WORKER_SPY)
            page.wait_for_selector("#comm-main:not(.hidden)", timeout=30000)
            page.fill("#comm-code", codes[0])
            page.click("#btn-comm-join")
            page.wait_for_selector("#comm-work-box:not(.hidden)", timeout=30000)
            wait_js(page, "document.getElementById('comm-flow-note').innerText.trim().length > 0", timeout=15000)
            check("thanh cột mốc: 'Đã tham gia' xong, bước kế tiếp được nêu",
                  page.locator("#comm-steps li[data-step=joined].is-done").count() == 1
                  and "Bước tiếp theo" in text(page, "#comm-flow-note"), text(page, "#comm-flow-note"))
            wait_js(page, "document.getElementById('comm-train-reason').innerText.includes('ít nhất')", timeout=15000)
            check("nút Huấn luyện khóa có lý do (chưa đủ ảnh)",
                  page.locator("#btn-comm-train").is_disabled() and "ít nhất 5 ảnh" in text(page, "#comm-train-reason"),
                  text(page, "#comm-train-reason"))
            check("nút Gửi khóa có lý do (chưa huấn luyện)",
                  page.locator("#btn-comm-submit").is_disabled() and "huấn luyện" in text(page, "#comm-submit-reason"),
                  text(page, "#comm-submit-reason"))

            add_labeled_images(page, groups[0])
            wait_js(page, "!document.getElementById('btn-comm-train').disabled", timeout=15000)
            check("đủ ảnh có nhóm: dòng lý do biến mất", text(page, "#comm-train-reason") == "")
            page.click("#btn-comm-train")
            page.wait_for_selector("#comm-local-result:not(.hidden)", timeout=240000)
            check("huấn luyện xong: cột mốc 'Đã huấn luyện' bật",
                  page.locator("#comm-steps li[data-step=trained].is-done").count() == 1
                  and "Bấm “Gửi trọng số" in text(page, "#comm-flow-note"), text(page, "#comm-flow-note"))
            page.select_option("select[data-label] >> nth=0", index=2)
            check("đổi nhóm sau huấn luyện: khung 'Huấn luyện xong' cũ được ẩn, nút Gửi có lý do",
                  page.locator("#comm-local-result").is_hidden() and "huấn luyện" in text(page, "#comm-submit-reason"))
            add_labeled_images(page, groups[0])  # chọn lại đúng nhóm (cùng ảnh bị bỏ qua vì trùng; nhóm đặt lại ở dưới)
            wait_js(page, "!document.getElementById('btn-comm-train').disabled", timeout=15000)
            page.click("#btn-comm-train")
            page.wait_for_selector("#comm-local-result:not(.hidden)", timeout=240000)

            # ── 4. Mất phản hồi khi gửi: máy chủ đã nhận, trình duyệt không nhận được phản hồi ──
            def drop_response(route: Any) -> None:
                route.fetch()  # máy chủ nhận và xử lý thật
                route.abort("connectionreset")  # nhưng phản hồi không về tới trình duyệt

            page.route("**/api/community/update", drop_response)
            page.click("#btn-comm-submit")
            page.wait_for_selector("#comm-submit-result:not(.hidden)", timeout=60000)
            wait_js(page, "document.getElementById('comm-submit-result').innerText.length > 0", timeout=15000)
            lost = text(page, "#comm-submit-result")
            check("mất phản hồi: báo 'đã nhận', không báo 'không nhận'",
                  "đã nhận cập nhật của bạn" in lost and "không nhận cập nhật" not in lost, lost[:140])
            check("mất phản hồi: cột mốc 'Đã gửi' bật", page.locator("#comm-steps li[data-step=sent].is-done").count() == 1)
            page.unroute("**/api/community/update")

            # ── 5. Tải lại trang sau khi đã gửi: nút khóa phải nói lý do ──
            page.reload(wait_until="load")
            page.wait_for_selector("#comm-work-box:not(.hidden)", timeout=30000)
            wait_js(page, "document.getElementById('comm-kpi-round').textContent.trim().startsWith('#')", timeout=30000)
            add_labeled_images(page, groups[0])
            page.wait_for_timeout(500)
            reason = text(page, "#comm-train-reason")
            check("tải lại sau khi đã gửi: nút khóa kèm lý do 'đã gửi cho vòng này'",
                  page.locator("#btn-comm-train").is_disabled() and "đã gửi cập nhật cho vòng" in reason, reason)
            check("tải lại sau khi đã gửi: cột mốc 'Đã gửi' vẫn bật", page.locator("#comm-steps li[data-step=sent].is-done").count() == 1)

            # ── 6. Hai người còn lại gửi → vòng gộp → người thứ nhất được báo "đã gộp" ──
            others = []
            for index in (1, 2):
                other_context, other_page, _errs = open_page(browser, base, width=1280)
                other_page.wait_for_selector("#comm-main:not(.hidden)", timeout=30000)
                other_page.fill("#comm-code", codes[index])
                other_page.click("#btn-comm-join")
                other_page.wait_for_selector("#comm-work-box:not(.hidden)", timeout=30000)
                add_labeled_images(other_page, groups[index])
                wait_js(other_page, "!document.getElementById('btn-comm-train').disabled", timeout=15000)
                other_page.click("#btn-comm-train")
                other_page.wait_for_selector("#comm-local-result:not(.hidden)", timeout=240000)
                others.append((other_context, other_page))
            for _ctx, other_page in others:
                other_page.click("#btn-comm-submit")
                other_page.wait_for_selector("#comm-submit-result:not(.hidden)", timeout=60000)
            merged_text = text(others[-1][1], "#comm-submit-result")
            check("người gửi cuối: thấy kết quả gộp ngay trong phản hồi", "đã gộp" in merged_text, merged_text[:140])
            others[-1][1].wait_for_selector("#comm-steps li[data-step=merged].is-done", timeout=30000)
            check("người gửi cuối: cột mốc 'Đã gộp' bật", True)
            wait_js(page, 
                "document.getElementById('comm-flow-note').innerText.includes('đã góp vào phiên bản')", timeout=120000)
            check("người gửi từ trước (đã tải lại trang): được báo cập nhật của mình nằm trong phiên bản mới",
                  page.locator("#comm-steps li[data-step=merged].is-done").count() == 1, text(page, "#comm-flow-note"))

            # ── 7. Vòng 2: tiến trình xử lý ảnh bị đặt lại → huấn luyện lại thành công, không phải xóa ảnh ──
            wait_js(page, 
                "document.getElementById('comm-kpi-round').textContent.trim() === '#2'", timeout=60000)
            page.wait_for_timeout(500)
            add_labeled_images(page, groups[0])
            wait_js(page, "!document.getElementById('btn-comm-train').disabled", timeout=30000)
            page.click("#btn-comm-train")
            page.wait_for_selector("#comm-local-result:not(.hidden)", timeout=240000)
            check("vòng 2: huấn luyện lần đầu thành công", "Huấn luyện xong" in text(page, "#comm-local-result"))
            check("vòng 2: bắt đầu lần đóng góp mới, cột mốc 'Đã gộp' được đặt lại",
                  page.locator("#comm-steps li[data-step=merged].is-done").count() == 0)
            page.evaluate("window.__workers.at(-1).dispatchEvent(new ErrorEvent('error', { message: 'giả lập: worker bị đặt lại' }))")
            page.wait_for_timeout(300)
            # đánh dấu khung kết quả cũ: lần huấn luyện mới ghi đè nội dung nên dấu này biến mất khi có kết quả mới
            page.evaluate("document.getElementById('comm-local-result').innerHTML = '<b>DAU-CU</b>'")
            page.click("#btn-comm-train")
            wait_js(page, 
                "(() => { const el = document.getElementById('comm-local-result'); "
                "return !el.classList.contains('hidden') && !el.innerText.includes('DAU-CU'); })()", timeout=240000)
            check("worker bị đặt lại: huấn luyện lại thành công mà không phải xóa ảnh",
                  "Chưa huấn luyện được" not in text(page, "#comm-local-result"), text(page, "#comm-local-result")[:120])

            # ── 8. 429 do proxy: câu riêng, nói rõ chưa gửi ──
            page.route("**/api/community/update", lambda route: route.fulfill(
                status=429, content_type="text/html", body="<html><body>Too Many Requests</body></html>"))
            page.click("#btn-comm-submit")
            wait_js(page, "document.getElementById('comm-submit-result').innerText.includes('giới hạn tần suất')", timeout=60000)
            check("429: báo giới hạn tần suất và nói rõ cập nhật chưa được gửi",
                  "chưa được gửi" in text(page, "#comm-submit-result"), text(page, "#comm-submit-result")[:140])
            page.unroute("**/api/community/update")
            check("không có lỗi trang (pageerror) trong toàn bộ kịch bản", not errors, "; ".join(errors)[:200])
            context.close()
            for ctx, _pg in others:
                ctx.close()
            browser.close()
    except Exception as error:  # noqa: BLE001
        check("kịch bản chạy hết không ngoại lệ", False, f"{type(error).__name__}: {error}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} kiểm tra đạt")
    if failed:
        print("Hỏng:", *failed, sep="\n  - ")
    return 1 if failed else 0


if __name__ == "__main__":
    # tránh cảnh báo biến không dùng: base_e2e giữ để scripts.e2e_community được nạp một lần
    _ = base_e2e
    sys.exit(main())

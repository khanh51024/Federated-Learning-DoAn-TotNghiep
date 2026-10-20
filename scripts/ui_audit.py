"""Đo giao diện web bằng Chromium thật: tràn ngang, chiều cao, cỡ chữ, số ảnh, axe-core, ảnh chụp và băm các API số liệu.

Mục đích: có con số TRƯỚC/SAU cho mỗi đợt nâng cấp UI (docs/CropFed_FL_Ke_hoach_nang_cap_UI_UX_API_2026-10-14.docx) và chứng
minh số liệu khoa học không đổi (băm JSON của /api/compare, /api/evidence/*).

Chạy:
    python scripts/ui_audit.py --label baseline                   # tự dựng máy chủ cục bộ
    python scripts/ui_audit.py --label prod --base-url https://datn.tranngochung.id.vn   # CHỈ ĐỌC (không gọi POST)
    python scripts/ui_audit.py --label after --quick              # chỉ 390 và 1440 px
    python scripts/ui_audit.py --compare tmp/ui-audit/baseline.json tmp/ui-audit/after.json

Kết quả: tmp/ui-audit/<nhãn>.json và tmp/ui-audit/<nhãn>/*.png (gitignored). Thoát mã 1 nếu có tràn ngang, lỗi axe mức
serious/critical, hoặc lỗi trang; 0 nếu sạch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "ui-audit"
AXE_VERSION = "4.10.2"
AXE_FILE = ROOT / "tmp" / "axe-core" / "axe.min.js"

ROUTES = ["gioi-thieu", "du-lieu", "phuong-phap", "ket-qua", "ket-luan", "chan-doan", "hoc-lien-ket", "phu-luc"]
SUBTABS = ["summary", "holdout", "errors", "calibration"]
VIEWPORTS = [320, 390, 768, 1280, 1440, 1920]
QUICK_VIEWPORTS = [390, 1440]
SNAPSHOT_APIS = ["/api/compare", "/api/evidence/summary", "/api/evidence/details", "/api/evidence/partitions", "/api/demo/evidence"]
SERIOUS = {"serious", "critical"}

MEASURE_JS = """() => {
  const d = document.documentElement;
  const pane = document.querySelector('.tab-pane.active') || document.querySelector('main') || document.body;
  const scope = pane.querySelector('.subpanel.is-active') || pane;  // chỉ tìm trong bước ĐANG hiện
  return {
    overflow: d.scrollWidth - d.clientWidth,
    height: Math.round(d.scrollHeight),
    bodyFont: parseFloat(getComputedStyle(document.body).fontSize),
    imgs: scope.querySelectorAll('img').length,
    cards: scope.querySelectorAll('.error-card').length,
    galleryItems: document.querySelectorAll('#gallery-grid .gallery-item').length,
    skeletons: scope.querySelectorAll('.skeleton').length,
  };
}"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ensure_axe() -> str:
    if not AXE_FILE.is_file():
        AXE_FILE.parent.mkdir(parents=True, exist_ok=True)
        work = AXE_FILE.parent / "pkg"
        work.mkdir(exist_ok=True)
        npm = "npm.cmd" if os.name == "nt" else "npm"
        subprocess.run([npm, "pack", f"axe-core@{AXE_VERSION}", "--silent"], cwd=work, check=True, capture_output=True, timeout=180)
        archive = next(work.glob("axe-core-*.tgz"))
        import tarfile

        with tarfile.open(archive) as tar:
            member = tar.extractfile("package/axe.min.js")
            assert member is not None
            AXE_FILE.write_bytes(member.read())
    return AXE_FILE.read_text(encoding="utf-8")


def start_local_server() -> tuple[subprocess.Popen[bytes], str]:
    port = free_port()
    state = OUT_DIR / "server-state"
    state.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "COMMUNITY_SECRET": "ui-audit-secret-0123456789", "COMMUNITY_ADMIN_TOKEN": "ui-audit-admin-token-0123456789",
           "COMMUNITY_STATE_DIR": str(state), "PYTHONUNBUFFERED": "1"}
    log = (OUT_DIR / "server.log").open("wb")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "web.app:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{base}/api/status", timeout=2)
            return proc, base
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise SystemExit("máy chủ cục bộ không khởi động được; xem tmp/ui-audit/server.log")


def snapshot_apis(base: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in SNAPSHOT_APIS:
        try:
            with urllib.request.urlopen(base + path, timeout=60) as response:
                raw = response.read()
            canonical = json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            out[path] = {"sha256": hashlib.sha256(canonical).hexdigest(), "bytes": len(raw)}
        except Exception as exc:  # noqa: BLE001
            out[path] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def wait_ready(page: Any, route: str) -> None:
    if route == "quan-tri":
        page.wait_for_selector("#login-form", timeout=30000)
        return
    page.wait_for_selector(f"#page-{route}.active", timeout=30000)
    for _ in range(100):
        pending = page.evaluate("document.getElementById('step-nav')?.classList.contains('is-pending') ?? false")
        if not pending:
            break
        page.wait_for_timeout(150)
    page.wait_for_timeout(250)


def run_axe(page: Any, axe_source: str) -> list[dict[str, Any]]:
    page.evaluate("(src) => { if (!window.axe) { (0, eval)(src); } }", axe_source)
    return page.evaluate(
        "axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] }, resultTypes: ['violations'] })"
        ".then(r => r.violations.map(v => ({ id: v.id, impact: v.impact, nodes: v.nodes.length })))")


def keyboard_journey(browser: Any, base: str) -> dict[str, Any]:
    """Hành trình chính chỉ bằng bàn phím: Tab tới một ảnh mẫu ở trang Chẩn đoán, Enter, thấy kết quả; chuyển bước bằng phím; focus không mất."""
    context = browser.new_context(viewport={"width": 1280, "height": 900}, bypass_csp=True)
    page = context.new_page()
    out: dict[str, Any] = {"state": "ban-phim", "ok": False, "notes": []}
    try:
        page.goto(f"{base}/#chan-doan", wait_until="load", timeout=60000)
        wait_ready(page, "chan-doan")
        page.wait_for_selector("#gallery-grid .gallery-item", timeout=30000)
        reached = False
        for _ in range(60):
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement && document.activeElement.classList.contains('gallery-item')"):
                reached = True
                break
        out["notes"].append(f"Tab tới ảnh mẫu: {'được' if reached else 'KHÔNG'}")
        visible = page.evaluate("""() => { const el = document.activeElement; if (!el) return false;
            const s = getComputedStyle(el); return s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0; }""")
        out["notes"].append(f"có viền focus nhìn thấy: {'có' if visible else 'KHÔNG'}")
        if reached:
            page.keyboard.press("Enter")
            page.wait_for_selector("#local-view .diag-headline", timeout=180000)
            kept = page.evaluate("document.activeElement && document.activeElement !== document.body")
            out["notes"].append(f"sau Enter focus còn trên phần tử: {'có' if kept else 'KHÔNG (mất focus)'}")
            out["ok"] = bool(visible and kept)
        # đi sang bước kế bằng phím: Tab tới liên kết "Sau" rồi Enter
        moved = False
        for _ in range(80):
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement && document.activeElement.id === 'step-next'"):
                page.keyboard.press("Enter")
                page.wait_for_selector("#page-hoc-lien-ket.active", timeout=15000)
                moved = True
                break
        out["notes"].append(f"chuyển bước bằng phím: {'được' if moved else 'KHÔNG'}")
        out["ok"] = out["ok"] and moved
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"lỗi: {type(exc).__name__}: {str(exc)[:100]}")
    finally:
        context.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run")
    parser.add_argument("--base-url", default=None, help="Mặc định: dựng máy chủ cục bộ. Với URL ngoài, chỉ gọi GET.")
    parser.add_argument("--quick", action="store_true", help="Chỉ 390 và 1440 px")
    parser.add_argument("--no-axe", action="store_true")
    parser.add_argument("--no-shots", action="store_true")
    parser.add_argument("--interact", action="store_true",
                        help="Thêm trạng thái 'đã chẩn đoán một ảnh mẫu' (gửi POST /api/inference/all; mặc định bật khi dựng máy chủ cục bộ)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shots = OUT_DIR / args.label
    if not args.no_shots:
        shots.mkdir(parents=True, exist_ok=True)
    axe_source = "" if args.no_axe else ensure_axe()
    server = None
    base = args.base_url.rstrip("/") if args.base_url else None
    if base is None:
        server, base = start_local_server()
    viewports = QUICK_VIEWPORTS if args.quick else VIEWPORTS
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        api = snapshot_apis(base)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for theme in ("light", "dark"):
                for width in viewports:
                    context = browser.new_context(viewport={"width": width, "height": 900}, bypass_csp=True)
                    context.add_init_script(f"try {{ localStorage.setItem('cropfed.theme', '{theme}'); }} catch (e) {{}}")
                    page = context.new_page()
                    errors: list[str] = []
                    page.on("pageerror", lambda e, bag=errors: bag.append(str(e)))
                    page.on("console", lambda m, bag=errors: bag.append(m.text) if m.type == "error" else None)
                    states: list[tuple[str, str | None]] = [(r, None) for r in ROUTES] + [("quan-tri", None)]
                    for route in ROUTES:
                        if route == "ket-qua":
                            states += [(route, tab) for tab in SUBTABS[1:]]
                    for route, tab in sorted(states, key=lambda s: (s[0] == "quan-tri", ROUTES.index(s[0]) if s[0] in ROUTES else 99)):
                        errors.clear()
                        url = f"{base}/quan-tri" if route == "quan-tri" else f"{base}/#{route}"
                        if tab is None:  # các tab của Kết quả dùng lại đúng trang đang mở
                            page.goto(url, wait_until="load", timeout=60000)
                        wait_ready(page, route)
                        if tab:
                            page.click(f".subtab[data-subtab='{tab}']")
                            page.wait_for_selector(f"#result-{tab}.is-active", timeout=15000)
                            for _ in range(80):
                                if page.evaluate("document.querySelectorAll('.subpanel.is-active .skeleton').length") == 0:
                                    break
                                page.wait_for_timeout(100)
                            page.wait_for_timeout(300)
                        metrics = page.evaluate(MEASURE_JS)
                        axe = [] if args.no_axe else run_axe(page, axe_source)
                        name = f"{route}{'-' + tab if tab else ''}-{width}-{theme}"
                        if not args.no_shots and width in (390, 1440):
                            page.screenshot(path=str(shots / f"{name}.png"), full_page=True)
                        rows.append({"state": f"{route}{'/' + tab if tab else ''}", "width": width, "theme": theme, **metrics,
                                     "axe": axe, "errors": list(errors)})
                    if args.interact or args.base_url is None:
                        errors.clear()
                        page.goto(f"{base}/#chan-doan", wait_until="load", timeout=60000)
                        wait_ready(page, "chan-doan")
                        page.click("#gallery-grid .gallery-item")
                        page.wait_for_selector("#local-view .diag-headline", timeout=180000)
                        for _ in range(80):
                            if page.evaluate("document.querySelectorAll('#local-view .skeleton').length") == 0:
                                break
                            page.wait_for_timeout(100)
                        page.wait_for_timeout(500)
                        metrics = page.evaluate(MEASURE_JS)
                        metrics["headline"] = page.evaluate("(document.querySelector('#local-view .diag-headline') || {}).innerText || ''")[:160]
                        axe = [] if args.no_axe else run_axe(page, axe_source)
                        if not args.no_shots and width in (390, 1440):
                            page.screenshot(path=str(shots / f"chan-doan-ket-qua-{width}-{theme}.png"), full_page=True)
                        rows.append({"state": "chan-doan/da-chan-doan", "width": width, "theme": theme, **metrics,
                                     "axe": axe, "errors": list(errors)})
                    context.close()
            keyboard = keyboard_journey(browser, base) if (args.interact or args.base_url is None) else None
            browser.close()
    finally:
        if server is not None:
            server.terminate()
    result = {"label": args.label, "base": base, "seconds": round(time.time() - started, 1), "api": api, "rows": rows,
              "keyboard": keyboard,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT_DIR / f"{args.label}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return summarize(result)


def summarize(result: dict[str, Any]) -> int:
    rows = result["rows"]
    overflow = [r for r in rows if r["overflow"] > 1]
    serious = [(r, v) for r in rows for v in r["axe"] if v["impact"] in SERIOUS]
    errors = [r for r in rows if r["errors"]]
    print(f"[{result['label']}] {len(rows)} trạng thái trong {result['seconds']} s ({result['base']})")
    print(f"  tràn ngang: {len(overflow)} | axe serious/critical: {len(serious)} | trạng thái có lỗi trang: {len(errors)}")
    minor: dict[str, int] = {}
    for r in rows:
        for v in r["axe"]:
            if v["impact"] not in SERIOUS:
                minor[v["id"]] = minor.get(v["id"], 0) + 1
    if minor:
        print("  axe mức vừa/nhẹ (số trạng thái): " + ", ".join(f"{k}×{n}" for k, n in sorted(minor.items())))
    keyboard = result.get("keyboard")
    if keyboard:
        print(f"  bàn phím: {'ĐẠT' if keyboard['ok'] else 'CHƯA ĐẠT'} — " + "; ".join(keyboard["notes"]))
    for r in overflow[:8]:
        print(f"    TRÀN {r['state']} @ {r['width']} {r['theme']}: +{r['overflow']} px")
    for r, v in serious[:8]:
        print(f"    AXE {v['id']} ({v['impact']}, {v['nodes']} nút) ở {r['state']} @ {r['width']} {r['theme']}")
    for r in errors[:5]:
        print(f"    LỖI {r['state']} @ {r['width']}: {r['errors'][0][:120]}")
    light = [r for r in rows if r["theme"] == "light" and r["width"] == 1440]
    if light:
        print("  @1440 sáng: trạng thái · chiều cao · ảnh · thẻ lỗi · cỡ chữ")
        for r in light:
            print(f"    {r['state']:22s} {r['height']:6d} px  img={r['imgs']:4d}  cards={r['cards']:4d}  font={r['bodyFont']:.0f}")
    return 1 if (overflow or serious or errors or (keyboard and not keyboard["ok"])) else 0


def compare(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    print(f"So sánh {before['label']} → {after['label']}")
    changed_api = [p for p in before["api"] if before["api"][p].get("sha256") != after["api"].get(p, {}).get("sha256")]
    print("  API số liệu:", "KHÔNG ĐỔI" if not changed_api else f"ĐỔI: {changed_api}")
    key = lambda r: (r["state"], r["width"], r["theme"])  # noqa: E731
    old = {key(r): r for r in before["rows"]}
    print("  trạng thái @1440 sáng: chiều cao, ảnh, thẻ, cỡ chữ (trước → sau)")
    for r in after["rows"]:
        if r["width"] != 1440 or r["theme"] != "light":
            continue
        o = old.get(key(r))
        if o:
            print(f"    {r['state']:22s} h {o['height']:6d} → {r['height']:6d} | img {o['imgs']:4d} → {r['imgs']:4d} | "
                  f"cards {o['cards']:4d} → {r['cards']:4d} | font {o['bodyFont']:.0f} → {r['bodyFont']:.0f}")
    worse = sum(1 for r in after["rows"] if r["overflow"] > 1) - sum(1 for r in before["rows"] if r["overflow"] > 1)
    print(f"  tràn ngang: {sum(1 for r in before['rows'] if r['overflow'] > 1)} → {sum(1 for r in after['rows'] if r['overflow'] > 1)}")
    return 1 if (changed_api or worse > 0) else 0


if __name__ == "__main__":
    sys.exit(main())

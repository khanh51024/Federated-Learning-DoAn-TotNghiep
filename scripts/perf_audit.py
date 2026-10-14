"""Đo hiệu năng tải trang bằng Chromium: FCP, LCP, CLS, TBT (xấp xỉ), số byte và số yêu cầu, cho từng bước, ở hai hồ sơ.

Hồ sơ `mobile`: 390 px, CPU chậm ×4, mạng "4G chậm" (1,6 Mbps xuống, trễ 150 ms). Hồ sơ `desktop`: 1440 px, không giả lập.
Mỗi lần đo dùng context MỚI (bộ nhớ đệm trống) để thấy lần đầu vào trang; lấy trung vị của --runs lần.

INP cần tương tác thật và không đo ở đây. Đây là số đo CÓ NHIỄU (máy, mạng): so sánh trước/sau cùng điều kiện, không dùng như
điểm Lighthouse chính thức.

Chạy:
    python scripts/perf_audit.py --label baseline --base-url https://datn.tranngochung.id.vn     # chỉ GET
    python scripts/perf_audit.py --label local                                                   # tự dựng máy chủ cục bộ
    python scripts/perf_audit.py --compare tmp/perf-audit/baseline.json tmp/perf-audit/after.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "perf-audit"
ROUTES = ["tong-quan", "non-iid", "ket-qua", "chan-doan", "huan-luyen", "bang-chung"]
PROFILES = {
    "mobile": {"width": 390, "height": 844, "cpu": 4,
               "net": {"offline": False, "latency": 150, "downloadThroughput": 1.6 * 1024 * 1024 / 8, "uploadThroughput": 750 * 1024 / 8}},
    "desktop": {"width": 1440, "height": 900, "cpu": 1, "net": None},
}
# mục tiêu của kế hoạch (docs/CropFed_FL_Ke_hoach_nang_cap_UI_UX_API_2026-10-14.docx, UI-09)
TARGETS = {"lcp_ms": 2500, "cls": 0.1}

INIT_JS = """
window.__perf = { lcp: 0, cls: 0, fcp: 0, tbt: 0 };
try { new PerformanceObserver(l => { for (const e of l.getEntries()) window.__perf.lcp = e.startTime; }).observe({ type: 'largest-contentful-paint', buffered: true }); } catch (e) {}
try { new PerformanceObserver(l => { for (const e of l.getEntries()) if (!e.hadRecentInput) window.__perf.cls += e.value; }).observe({ type: 'layout-shift', buffered: true }); } catch (e) {}
try { new PerformanceObserver(l => { for (const e of l.getEntries()) if (e.name === 'first-contentful-paint') window.__perf.fcp = e.startTime; }).observe({ type: 'paint', buffered: true }); } catch (e) {}
try { new PerformanceObserver(l => { for (const e of l.getEntries()) window.__perf.tbt += Math.max(0, e.duration - 50); }).observe({ type: 'longtask', buffered: true }); } catch (e) {}
"""

COLLECT_JS = """() => {
  const res = performance.getEntriesByType('resource');
  const nav = performance.getEntriesByType('navigation')[0];
  const bytes = res.reduce((s, r) => s + (r.transferSize || 0), 0) + (nav ? nav.transferSize || 0 : 0);
  const fonts = res.filter(r => /\\.woff2?($|\\?)/.test(r.name)).length;
  const imgs = res.filter(r => r.initiatorType === 'img').length;
  return { ...window.__perf, bytes, requests: res.length + 1, fonts, imgs };
}"""


def start_local() -> tuple[subprocess.Popen[bytes], str]:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "web.app:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, stdout=(OUT / "server.log").open("wb"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{base}/api/status", timeout=2)
            return proc, base
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise SystemExit("máy chủ cục bộ không lên")


def measure(browser: Any, base: str, route: str, profile: dict[str, Any]) -> dict[str, float]:
    context = browser.new_context(viewport={"width": profile["width"], "height": profile["height"]})
    context.add_init_script(INIT_JS)
    page = context.new_page()
    cdp = context.new_cdp_session(page)
    if profile["cpu"] > 1:
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": profile["cpu"]})
    if profile["net"]:
        cdp.send("Network.enable")
        cdp.send("Network.emulateNetworkConditions", profile["net"])
    try:
        page.goto(f"{base}/#{route}", wait_until="load", timeout=120000)
        page.wait_for_selector(f"#page-{route}.active", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=30000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(800)
        data = page.evaluate(COLLECT_JS)
    finally:
        context.close()
    return {k: float(v) for k, v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--routes", nargs="*", default=ROUTES)
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]))

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    server = None
    base = args.base_url.rstrip("/") if args.base_url else None
    if base is None:
        server, base = start_local()
    results: dict[str, Any] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for name, profile in PROFILES.items():
                for route in args.routes:
                    runs = [measure(browser, base, route, profile) for _ in range(args.runs)]
                    results[f"{name}/{route}"] = {key: round(statistics.median(r[key] for r in runs), 3) for key in runs[0]}
                    results[f"{name}/{route}"]["runs"] = len(runs)
            browser.close()
    finally:
        if server is not None:
            server.terminate()
    payload = {"label": args.label, "base": base, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": args.runs, "results": results}
    (OUT / f"{args.label}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return report(payload)


def report(payload: dict[str, Any]) -> int:
    print(f"[{payload['label']}] {payload['base']} · trung vị {payload['runs']} lần · bộ nhớ đệm trống")
    print(f"  {'hồ sơ/bước':22s} {'FCP':>7s} {'LCP':>7s} {'CLS':>6s} {'TBT':>6s} {'KB':>7s} {'req':>4s} {'font':>4s}")
    over = 0
    for key, r in payload["results"].items():
        flag = ""
        if r["lcp"] > TARGETS["lcp_ms"] or r["cls"] > TARGETS["cls"]:
            flag = "  ◄ vượt mục tiêu"
            over += 1
        print(f"  {key:22s} {r['fcp']:7.0f} {r['lcp']:7.0f} {r['cls']:6.3f} {r['tbt']:6.0f} {r['bytes'] / 1024:7.0f} {int(r['requests']):4d} {int(r['fonts']):4d}{flag}")
    print(f"  mục tiêu: LCP ≤ {TARGETS['lcp_ms']} ms, CLS ≤ {TARGETS['cls']} ({over} trạng thái vượt)")
    return 0


def compare(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    print(f"So sánh {before['label']} → {after['label']}   (LCP ms · CLS · KB)")
    for key, a in after["results"].items():
        b = before["results"].get(key)
        if b:
            print(f"  {key:22s} LCP {b['lcp']:6.0f} → {a['lcp']:6.0f} | CLS {b['cls']:5.3f} → {a['cls']:5.3f} | KB {b['bytes'] / 1024:6.0f} → {a['bytes'] / 1024:6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

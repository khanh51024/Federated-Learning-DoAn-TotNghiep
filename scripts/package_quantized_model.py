"""Đóng gói mô hình int8 thành sản phẩm và đo độ trễ suy luận trên trình duyệt hoặc điện thoại (N4.1 của yêu cầu đề tài).

`scripts/eval_gd4_compression.py` dựng tệp ONNX trong thư mục làm việc (tmp/gd4, ngoài git). Script này:
  package    chép biến thể int8 đã chọn vào artifacts/quantized/<gói>/ kèm manifest: SHA-256, kích thước, checkpoint nguồn và số đo độ chính xác từ artifacts/gd4_compression.json
  latency    đo độ trễ một ảnh, một luồng, bằng onnxruntime-web (wasm, bản web/static/vendor/ort) trên trình duyệt Chromium qua Playwright; ghi artifacts/quantized/<gói>/latency_browser.json
  serve      mở trang đo trong mạng nội bộ để chạy trên điện thoại thật: mở địa chỉ được in ra, trang hiện kết quả và tự gửi về máy này, ghi artifacts/quantized/<gói>/latency_devices.jsonl

Biến thể mặc định `int8_se_gemm_pw_late` là biến thể giữ được độ chính xác nhỏ nhất đã đo (docs/GD4_NEN_LUONG_TU_HOA_2026-10-26.md mục 3); lượng tử hóa mọi lớp làm mô hình sụp đổ nên không đóng gói.

    python scripts/package_quantized_model.py package
    /c/Python313/python.exe -X utf8 scripts/package_quantized_model.py latency         # Playwright chỉ có ở Python hệ thống
    python scripts/package_quantized_model.py serve --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GD4_JSON = ROOT / "artifacts" / "gd4_compression.json"
ORT_DIR = ROOT / "web" / "static" / "vendor" / "ort"
VARIANT = "int8_se_gemm_pw_late"
DEFAULT_WORK = ROOT / "tmp" / "gd4"
REPEATS = 100

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Đo độ trễ</title>
<style>body{font:16px system-ui;margin:16px;background:#fcfcfb;color:#0b0b0b}td,th{padding:4px 12px;text-align:right}th:first-child,td:first-child{text-align:left}</style>
<h1>Đo độ trễ suy luận</h1><p id="status">Đang đo, giữ trang mở…</p><table id="table"></table>
<script type="module">
import * as ort from '/ort/ort.wasm.min.mjs';
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = '/ort/';
const models = MODELS;
const repeats = REPEATS_VALUE;
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const data = new Float32Array(3 * 224 * 224);
let state = 12345;
for (let i = 0; i < data.length; i++) { state = (state * 1103515245 + 12345) & 0x7fffffff; data[i] = state / 0x7fffffff * 4 - 2; }
const rows = [];
for (const model of models) {
  const session = await ort.InferenceSession.create(model.url, { executionProviders: ['wasm'] });
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, 224, 224]) };
  for (let i = 0; i < 5; i++) await session.run(feeds);
  const times = [];
  for (let i = 0; i < repeats; i++) { const t = performance.now(); await session.run(feeds); times.push(performance.now() - t); }
  rows.push({ name: model.name, median_ms: median(times), min_ms: Math.min(...times), max_ms: Math.max(...times), repeats });
}
const result = { user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency, threads: 1, backend: 'onnxruntime-web wasm', rows };
document.getElementById('table').innerHTML = '<tr><th>Mô hình</th><th>Trung vị (ms)</th></tr>' + rows.map(r => `<tr><td>${r.name}</td><td>${r.median_ms.toFixed(1)}</td></tr>`).join('');
await fetch('/result', { method: 'POST', body: JSON.stringify(result) });
document.getElementById('status').textContent = 'Xong, đã gửi kết quả về máy chủ đo.';
window.RESULT = result;
</script>"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(work: Path, out: Path, variant: str, models: tuple[str, ...]) -> dict[str, Any]:
    """Chép tệp int8 và ghi manifest; mọi số đo lấy từ artifacts/gd4_compression.json, không viết tay."""
    gd4 = json.loads(GD4_JSON.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"bundle_id": gd4["bundle_id"], "release_manifest_sha256": gd4["release_manifest_sha256"], "variant": variant, "evidence": "số đo chép từ artifacts/gd4_compression.json",
                                "measured_on": {"split": gd4["split"], "images": gd4["count"], "environment": gd4["environment"]}, "models": {}}
    for model in models:
        source = work / f"{model}_{variant}.onnx"
        if not source.is_file():
            raise SystemExit(f"thiếu {source}; chạy scripts/eval_gd4_compression.py trước")
        block = gd4["models"][model]
        expected = block["onnx"]["variants"][variant]["bytes"]
        if source.stat().st_size != expected:
            raise SystemExit(f"{source} có {source.stat().st_size} byte, khác {expected} byte đã đo trong gd4_compression.json: tệp không phải bản đã đo")
        target = out / source.name
        shutil.copy2(source, target)
        accuracy = {key: block["onnx"][f"onnx_{variant}"][key] for key in ("n", "top1", "macro_f1", "plantvillage_top1", "field_top1", "predictions_changed_vs_pytorch_fp32") if key in block["onnx"][f"onnx_{variant}"]}
        manifest["models"][model] = {"file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size, "fp32_bytes": block["onnx"]["fp32_bytes"], "size_ratio": round(target.stat().st_size / block["onnx"]["fp32_bytes"], 4),
                                     "source_checkpoint_sha256": block["checkpoint_sha256"], "accuracy": accuracy, "accuracy_fp32": {k: block["onnx"]["onnx_fp32"][k] for k in accuracy if k in block["onnx"]["onnx_fp32"]},
                                     "latency_laptop_cpu_one_thread": block.get("latency")}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


class LatencyServer(http.server.ThreadingHTTPServer):
    """Phục vụ trang đo, thư viện ort và các tệp mô hình; nhận kết quả qua POST /result. Chỉ phục vụ đúng các tệp đã khai báo."""
    daemon_threads = True

    def __init__(self, address: tuple[str, int], files: dict[str, Path], page: str):
        self.files, self.page, self.results = files, page, []
        super().__init__(address, LatencyHandler)


class LatencyHandler(http.server.BaseHTTPRequestHandler):
    server: LatencyServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - chữ ký của thư viện chuẩn
        return None

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - tên do thư viện chuẩn quy định
        if self.path == "/":
            self._send(self.server.page.encode("utf-8"), "text/html; charset=utf-8")
            return
        path = self.server.files.get(self.path)
        if path is None:
            self.send_error(404)
            return
        kind = {".mjs": "text/javascript", ".wasm": "application/wasm"}.get(path.suffix, "application/octet-stream")
        self._send(path.read_bytes(), kind)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/result":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        self.server.results.append(json.loads(self.rfile.read(length).decode("utf-8")))
        self._send(b"ok", "text/plain")


def build_server(model_files: dict[str, Path], host: str, port: int) -> LatencyServer:
    files = {f"/ort/{p.name}": p for p in ORT_DIR.iterdir() if p.is_file()}
    entries = []
    for name, path in model_files.items():
        if not path.is_file():
            raise SystemExit(f"thiếu tệp mô hình {path}")
        files[f"/model/{path.name}"] = path
        entries.append({"name": name, "url": f"/model/{path.name}"})
    page = PAGE.replace("MODELS", json.dumps(entries)).replace("REPEATS_VALUE", str(REPEATS))
    return LatencyServer((host, port), files, page)


def model_files(out: Path, work: Path, model: str, variant: str) -> dict[str, Path]:
    return {f"{model}_fp32": work / f"{model}_fp32.onnx", f"{model}_{variant}": out / f"{model}_{variant}.onnx"}


def measure_in_chromium(server: LatencyServer) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("window.RESULT !== undefined", timeout=600_000)
        result = page.evaluate("window.RESULT")
        browser.close()
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("package", "latency", "serve"))
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--variant", default=VARIANT)
    parser.add_argument("--model", default="fedavg", help="mô hình để đo độ trễ")
    parser.add_argument("--bundle", default=None, help="mặc định lấy từ artifacts/gd4_compression.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--label", default="", help="tên thiết bị ghi kèm kết quả")
    args = parser.parse_args()
    bundle = args.bundle or json.loads(GD4_JSON.read_text(encoding="utf-8"))["bundle_id"]
    out = ROOT / "artifacts" / "quantized" / bundle
    if args.action == "package":
        manifest = package(args.work_dir, out, args.variant, ("centralized", "fedavg"))
        for name, block in manifest["models"].items():
            print(f"[int8] {name}: {block['bytes'] / 1e6:.2f} MB ({100 * block['size_ratio']:.0f}% của float32), top-1 {block['accuracy'].get('top1')} so với {block['accuracy_fp32'].get('top1')}")
        print(f"[int8] ghi {out}")
        return 0
    files = model_files(out, args.work_dir, args.model, args.variant)
    if args.action == "latency":
        server = build_server(files, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        result = measure_in_chromium(server)
        server.shutdown()
        report = {**result, "label": args.label or "laptop Chromium headless", "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (out / "latency_browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for row in result["rows"]:
            print(f"[int8] {row['name']}: trung vị {row['median_ms']:.1f} ms")
        return 0
    server = build_server(files, args.host, args.port)
    lan = socket.gethostbyname(socket.gethostname())
    print(f"[int8] mở trên điện thoại cùng mạng: http://{lan}:{args.port}/   (Ctrl+C để dừng; kết quả ghi vào {out / 'latency_devices.jsonl'})", flush=True)
    seen = 0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(1)
            for result in server.results[seen:]:
                with (out / "latency_devices.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({**result, "label": args.label, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")
                print("[int8] nhận kết quả:", [(r["name"], round(r["median_ms"], 1)) for r in result["rows"]], result["user_agent"][:80], flush=True)
                seen += 1
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

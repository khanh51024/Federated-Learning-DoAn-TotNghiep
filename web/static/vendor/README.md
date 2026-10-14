# Thư viện bên thứ ba (đã ghim phiên bản)

| Thư viện | Phiên bản | Tệp | Nguồn | Giấy phép |
|---|---|---|---|---|
| Chart.js | 4.4.1 | `chart.umd.js` (đã minify) | `https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js` | MIT |
| onnxruntime-web | 1.20.1 | `ort/ort.wasm.min.mjs`, `ort/ort-wasm-simd-threaded.mjs`, `ort/ort-wasm-simd-threaded.wasm` (bản CPU/wasm, không WebGPU) | `npm pack onnxruntime-web@1.20.1` → `package/dist/` (byte y nguyên) | MIT (Microsoft) |

Tự host để demo chạy được khi không có Internet và để phiên bản không đổi ngầm (trước đây `index.html` nạp `chart.js` không ghim từ CDN).
Nâng cấp: tải bản mới vào đây, đổi `?v=` trong `index.html`, rồi chạy lại `python scripts/update_status.py`.

## onnxruntime-web (học liên kết cộng đồng)

Chạy phần đóng băng của mô hình (backbone + `classifier[0..2]`) trong Web Worker để trích đặc trưng 1024 chiều từ ảnh của người dùng.
Chỉ dùng bản wasm, `numThreads = 1` nên KHÔNG cần header COOP/COEP. Ba tệp phải đi cùng nhau: `.mjs` nạp `-threaded.mjs`, rồi
`-threaded.wasm` cùng thư mục. Nâng cấp: `npm pack onnxruntime-web@<phiên bản>`, chép đúng ba tệp trên từ `package/dist/`, rồi chạy
`tests/test_community_js.py` (khớp ONNX) và `scripts/e2e_community.py`. Phải xuất lại ONNX nếu opset mới không nhận file cũ.


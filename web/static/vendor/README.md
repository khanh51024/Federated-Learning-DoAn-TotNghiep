# Thư viện bên thứ ba (đã ghim phiên bản)

| Thư viện | Phiên bản | Tệp | Nguồn | Giấy phép |
|---|---|---|---|---|
| Chart.js | 4.4.1 | `chart.umd.js` (đã minify) | `https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js` | MIT |
| onnxruntime-web | 1.20.1 | `ort/ort.wasm.min.mjs`, `ort/ort-wasm-simd-threaded.mjs`, `ort/ort-wasm-simd-threaded.wasm` (bản CPU/wasm, không WebGPU) | `npm pack onnxruntime-web@1.20.1` → `package/dist/` (byte y nguyên) | MIT (Microsoft) |

Tự host để demo chạy được khi không có Internet và để phiên bản không đổi ngầm (trước đây `index.html` nạp `chart.js` không ghim từ CDN).
Nâng cấp: tải bản mới vào đây, đổi `?v=` trong `index.html`, rồi chạy lại `python scripts/update_status.py`.

## onnxruntime-web (học phân tán cộng đồng)

Chạy phần đóng băng của mô hình (backbone + `classifier[0..2]`) trong Web Worker để trích đặc trưng 1024 chiều từ ảnh của người dùng.
Chỉ dùng bản wasm, `numThreads = 1` nên KHÔNG cần header COOP/COEP. Ba tệp phải đi cùng nhau: `.mjs` nạp `-threaded.mjs`, rồi
`-threaded.wasm` cùng thư mục. Nâng cấp: `npm pack onnxruntime-web@<phiên bản>`, chép đúng ba tệp trên từ `package/dist/`, rồi chạy
`tests/test_community_js.py` (khớp ONNX) và `scripts/e2e_community.py`. Phải xuất lại ONNX nếu opset mới không nhận file cũ.

## Biểu tượng thương hiệu (Lucide `boxes`)

Không có tệp nào trong thư mục này: hình học được chép **y nguyên** vào HTML ở bốn chỗ —
`<link rel="icon">` (data URI) và `<svg class="brand-svg">` trong `web/static/index.html` và `web/static/admin/admin.html`.

| Mục | Giá trị |
|---|---|
| Bộ icon | Lucide (`lucide-static`) |
| Phiên bản đã lấy | 1.59.0 |
| Tên biểu tượng | `boxes` — ba hộp xếp chồng, ẩn dụ nhiều gói trọng số gom về một mô hình |
| Tệp gốc | `lucide-static@1.59.0/icons/boxes.svg` (`https://cdn.jsdelivr.net/npm/lucide-static@1.59.0/icons/boxes.svg`) |
| Giấy phép | ISC |

Giữ nguyên `viewBox="0 0 24 24"`, `stroke-width="2"`, `stroke-linecap/linejoin="round"` và `stroke="currentColor"`
để CSS (`.brand-svg`) đổi màu theo nền sáng/tối. Data URI của favicon thì phải tô cứng `stroke='%23047857'`
vì favicon không thừa hưởng `currentColor`.

Nâng cấp/đổi biểu tượng: tải `icons/<tên>.svg` của phiên bản muốn dùng, chép 12 phần tử `<path>` vào cả bốn chỗ trên
(HTML nội tuyến dùng `"` — dấu `<`/`>` và `"` trong favicon phải đổi thành `%3C`/`%3E` và `'`), rồi cập nhật số phiên bản ở đây
và trong chú thích `favicon:` của hai file HTML. Kiểm lại bằng cách mở `#chuan-b` hoặc xem favicon ở 16 px.


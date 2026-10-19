# Review web 19/10/2026 — kiến trúc, bước 5 "huấn luyện thật", UI/UX

Phạm vi: toàn bộ phần web (`web/`), tập trung vào (1) có đúng là máy chủ đang huấn luyện không và làm sao tự kiểm,
(2) kiến trúc và stack, (3) UI/UX sáu bước và client cộng đồng. Phương pháp: chạy web thật trên máy phát triển, mở bằng
Chromium/Firefox/WebKit ở 1440, 390 và 320 px, đo CPU/tiến trình bằng `psutil` (không tin giao diện), so trọng số checkpoint với
trọng số khởi tạo tái tạo từ cùng seed. Số liệu nêu ở đây là số đo trên máy phát triển, không phải kết quả khoa học.

## 1. Câu trả lời ngắn cho "nó có thật sự đang huấn luyện không?"

Phần chạy trên máy chủ **tính toán thật nhưng không học được gì có nghĩa**; học thật chỉ có ở client cộng đồng trong trình duyệt.

| Chế độ | Tính toán thật? | Có học? |
|---|---|---|
| Centralized (bước 5) | Có: khoảng 125 giây CPU trong 12 giây (gần 10 lõi) | Gần như không |
| FedAvg mô phỏng (chỉ gọi qua API) | Có: khoảng 123 giây CPU | Gần như không |
| Flower 5 client (bước 5) | Có: 6 tiến trình Python thật (máy chủ khoảng 5 giây CPU, mỗi client khoảng 11,5 giây CPU và 410 MB), cổng gRPC mở, thoát sạch sau phiên | Gần như không |
| Client cộng đồng (trình duyệt) | Có, trong Web Worker | Có: độ chính xác giám sát tăng sau một vòng gộp |

Nguyên nhân phía máy chủ: web chỉ cho chạy chế độ `smoke` (76 ảnh train, 2 ảnh mỗi lớp) và máy chủ demo không có thư mục ảnh,
nên `PlantVillageDataset` sinh **ảnh nhiễu ngẫu nhiên** (`src/data/plantvillage.py`). Đo được:

- tham số học được chỉ đổi 0,002–0,007% so với trọng số khởi tạo cùng seed;
- độ chính xác kiểm tra đứng ở 1/38 (2,63%) và loss kiểm tra đứng ở ln 38 (3,6376) suốt các epoch/vòng;
- chuẩn cập nhật khoảng 0,59 trong nhật ký Flower gần như toàn là bộ đệm BatchNorm `running_var` trôi (khoảng 98% bình phương chuẩn),
  không phải học.

Kết luận thiết kế: mục tiêu của phần này là **kiểm chứng hệ thống** (điều phối, gRPC, kiểm duyệt update, phép gộp FedAvg), không phải
đo độ chính xác. Giao diện đã được sửa theo đó (mục 3).

### Cách tự kiểm là nó đang chạy

1. Task Manager / `top`: khi bấm Bắt đầu, Centralized chiếm gần hết CPU; Flower sinh 6 tiến trình `python`.
2. `Get-Process python` và `netstat -ano | findstr 8089` (VM: `ps aux | grep client_app`, `ss -ltnp | grep 8089`).
3. Khung **"Bằng chứng máy đã tính thật"** ở bước 5 (mục 3): số tiến trình, vòng đã gộp, chuỗi mã băm mô hình chung.
4. Thư mục phiên `artifacts/web_runs/network_smoke/<phiên>/`: `client_00..04.log`, `history.json` (thời gian học và chuẩn tham số học
   được của từng client), `global.pt`.
5. Client cộng đồng: phiên bản mô hình tăng (v0 → v1), cột "Độ chính xác giám sát" đổi, trang `/quan-tri` thấy vòng đã gộp. Đây là
   bằng chứng học thật rõ nhất khi trình bày.

## 2. Kiến trúc và stack

| Tầng | Đang dùng | Đánh giá |
|---|---|---|
| Backend | FastAPI + uvicorn một worker sau nginx; `routers/` + `services/`; số liệu đọc từ `artifacts/registry.json` | Hợp lý. Một worker là bắt buộc vì vòng học và khóa nằm trong bộ nhớ |
| Lưu trữ cộng đồng | SQLite | Đủ cho quy mô lớp học |
| Huấn luyện thử | Flower chạy ngoài tiến trình; Centralized và FedAvg mô phỏng chạy **trong** tiến trình web (`BackgroundTasks`) dưới khóa một tác vụ, trần tham số, hẹn giờ, kiểm RAM | Chấp nhận được cho demo; docstring `web/app.py` đã sửa cho đúng thực tế |
| Frontend | HTML + ES module thuần, không bước build; router theo hash tải lười từng trang | Lựa chọn đúng: sáu trang chủ yếu để trình bày, framework không thêm gì mà thêm toolchain |
| Thư viện | Chart.js 4.4.1, onnxruntime-web 1.20.1 (wasm, trong Web Worker), tự host, ghim phiên bản | Tốt, chạy được khi không có Internet |
| Bảo mật trang | CSP cho mọi trang HTML (mục 4), `/quan-tri` có chính sách riêng chặt hơn | Mới thêm trong đợt này |

Điểm mạnh nên đưa vào slide: tiền xử lý ảnh bằng JavaScript khớp từng byte với PyTorch và có test đối chiếu Python ↔ JS.

## 3. Thay đổi đã làm

### Bước 5 — kiểm chứng hệ thống, không đo độ chính xác
- Khung giới thiệu nói rõ mục tiêu, dữ liệu là ảnh nhiễu ngẫu nhiên và 1/38 là đúng kỳ vọng; chữ "ảnh tổng hợp" bị thay trên toàn giao diện.
- Khung **"Bằng chứng máy đã tính thật"** (Flower): số tiến trình đang chạy rồi "đã thoát (mã 0)", số vòng đã gộp, mã băm mô hình chung và
  dòng xác nhận chuỗi mã băm khớp (mô hình chung của vòng trước là mô hình cha của vòng sau).
- Máy chủ ghi thêm vào nhật ký mỗi vòng: `parent_model_hash`, `global_model_hash`, `received_learnable_update_norm`,
  `learnable_update_relative`; `final_model_hash` trong `history.json`. **Không đổi** `received_update_norm`, ngưỡng hay phép gộp.
- Bảng số liệu từng vòng không còn cột độ chính xác; Flower hiện dòng thật mỗi vòng (trước đây luôn trống vì `live_progress` rỗng).
- Centralized không còn hiện client/RTT của phiên Flower trước; nút Bắt đầu đổi nhãn "Đăng nhập để bắt đầu" khi cần quyền (vẫn bấm được).
- Sửa lỗi: `/api/metrics/federated?source=demo` luôn 404 (sai tên thư mục); giao diện báo "0/5 update hợp lệ" suốt vòng 2.

### Hạ tầng
- Cache: JS/CSS/HTML dùng `no-cache` (luôn hỏi lại nhưng nhận 304) thay cho `no-store`; trang chủ có ETag/304; phông và thư viện giữ
  `max-age=86400` kể cả khi nhận 304 (trước đây 304 rơi vào nhánh `no-store`).
- Router: lượt dựng cũ không còn chạy `onShow`, bật lại thanh Trước/Sau hay báo cho người nghe sau khi người dùng đã sang bước khác.
- Phông: đo thấy trang tải 15–16 tệp (~283 KB) và mọi preload đều được dùng; tiêu đề trang (weight 700) chưa được preload nên đã thêm.
  Muốn giảm thật phải gộp trọng số (phông biến thiên), việc này để sau.
- Nạp checkpoint qua một cửa `weights_only=True` (`src/common/checkpoint_io.py`), thay cho `weights_only=False`.

### UI/UX
- Bảng bảy mô hình (bước 3) và bảng dự đoán từng mô hình (bước 4) mở sẵn trên màn rộng (từ 1101 px), vẫn gấp trên điện thoại.
- Một dòng "Phạm vi" thống nhất ở bước 1 và 3 (một lần chạy, chưa phải độ chính xác ngoài thực địa) kèm liên kết sang bước 6.
- Quy tắc mở/gấp: mục cấp khối là thẻ "+/−" (`.overview-detail`, `.panel-fold`), mục phụ là thanh xám "▸" (`.fold`).
- Bước 4 trên điện thoại: ô tải ảnh lên đầu, chữ theo loại con trỏ ("Chạm để chọn hoặc chụp ảnh"), nhãn mức nhận đúng ngắn, hiệu ứng
  nâng thẻ chỉ cho chuột.
- Vùng chạm tối thiểu khoảng 40 px trên màn cảm ứng (nút nhỏ, chip, nút chọn, tóm tắt mục gấp, ô chọn).
- Client cộng đồng: ô thêm ảnh bằng tiếng Việt (có kéo thả), chọn nhóm **trước** rồi thêm ảnh (ảnh mới tự mang nhóm đó, vẫn sửa từng ảnh,
  không bao giờ lấy nhãn từ dự đoán của mô hình), danh sách ảnh cập nhật từng phần (giữ focus), bốn cột mốc luôn một hàng, nút "Rời phiên"
  gọn kèm gợi ý riêng, bảng "Các vòng gần đây" thành thẻ ở màn hẹp, chữ quyền riêng tư rút gọn kèm mục "Chi tiết gửi gì".

## 4. Chính sách bảo mật của trang

CSP áp cho mọi phản hồi HTML (trừ `/docs`, `/redoc` của FastAPI và `/quan-tri` đã có chính sách riêng):

```
default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data: blob:;
font-src 'self'; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'
```

- Không có `'unsafe-inline'` hay `'unsafe-eval'`. `'wasm-unsafe-eval'` dành cho onnxruntime-web biên dịch WebAssembly trong Web Worker.
- Để bỏ `style-src 'unsafe-inline'`: 13 chỗ `style="…"` trong template JS được đổi sang `data-style` rồi áp bằng CSSOM
  (`applyStyleData` trong `core/dom.js`, chỉ nhận biến CSS và `transform` với giá trị an toàn); script nền sáng/tối nội tuyến chuyển ra
  `js/theme-boot.js`.
- Công tắc lùi nhanh không cần sửa mã: biến môi trường `CROPFED_CSP_MODE` = `enforce` (mặc định, giá trị lạ cũng rơi về enforce) |
  `report` (chỉ báo, không chặn) | `off`.
- Kiểm bằng `scripts/csp_sweep.py`: mở đủ sáu bước, mọi tab con, mọi mục gấp, chẩn đoán một ảnh mẫu, nền tối, trên Chromium, Firefox và
  WebKit, không có vi phạm. Bộ dò đã được thử bằng cách chèn style/script nội tuyến và bắt được cả hai.

## 5. Kiểm chứng

Kết quả kiểm thử đầy đủ nằm ở `PROJECT_STATUS.md` (máy chấm sinh), không chép số vào đây. Các kịch bản trình duyệt thật (chạy tay, cần
Playwright; xem extra `e2e` trong `pyproject.toml`):

- `scripts/e2e_community.py`: học liên kết cộng đồng, trang quản trị, cổng huấn luyện; ghi nhận mọi vi phạm CSP;
- `scripts/e2e_community_ux.py`: các tình huống UX của client; **đã biết chập chờn** (khoảng một nửa số lần chạy hỏng đúng một kiểm tra ở
  bước "worker bị đặt lại"; chạy lại trên bản trước đợt này cho kết quả tương tự, nên không phải do thay đổi mới);
- `scripts/e2e_training.py`: bước 5 (Flower 5 client và Centralized), ghi kết quả vào `tmp/` để không làm lệch registry;
- `scripts/csp_sweep.py`: quét CSP trên ba trình duyệt.

## 6. Rủi ro còn lại và việc để sau

- Phần huấn luyện phía máy chủ vẫn chạy trên ảnh nhiễu: muốn thấy độ chính xác tăng trên máy chủ cần bộ ảnh thật và huấn luyện đầu phân
  loại trên backbone đóng băng (như client cộng đồng), chưa làm vì không cần cho mục tiêu "kiểm chứng hệ thống".
- Multi-seed vẫn cần GPU (khoảng 10 giờ GPU mỗi seed); máy phát triển chỉ có CPU. Giới hạn đã được nêu ở `EXPERIMENT_PROTOCOL.md` và bước 6.
- Gộp bớt trọng số phông để giảm số tệp tải đầu; thêm `// @ts-check` + JSDoc cho khoảng 6.500 dòng JS.
- Safari cũ (trước 16) có thể không có `'wasm-unsafe-eval'` nếu Worker kế thừa chính sách của trang: huấn luyện cộng đồng sẽ lỗi ở đó.
  Chưa thử trên iPhone thật. Lùi bằng `CROPFED_CSP_MODE=report`.

## 7. Tái cấu trúc thành web báo cáo (cùng ngày, sau đợt review)

Vấn đề: sáu bước trộn nội dung báo cáo, demo và kiểm chứng vận hành. Cùng một ba số độ chính xác xuất hiện khoảng 7 lần trên 3 trang, lời lưu ý
về tập giữ riêng khoảng 9 lần, và các khái niệm "tập kiểm tra", "tập giữ riêng" không được định nghĩa ở đâu.

Cấu trúc mới (chi tiết ở [WEB_DEMO_GD1_GD2.md](WEB_DEMO_GD1_GD2.md)):

- **Báo cáo, năm chương**: Giới thiệu, Dữ liệu, Phương pháp (trang mới), Kết quả, Kết luận (trang mới, nhận phần hạn chế và 3 thẻ kết luận).
- **Demo, hai trang**: Chẩn đoán ảnh, Học liên kết (khối client cộng đồng tách khỏi trang huấn luyện máy chủ).
- **Phụ lục kỹ thuật** (chỉ vào từ chân trang): kiểm chứng số liệu, nguồn gốc mô hình (SHA-256, cột "Chấm lại"), hạng mục kỹ thuật, kiểm thử Flower/Centralized trên máy chủ, Weight Inspector, thiết bị.
- Quy tắc "mỗi số một nhà": số tổng hợp chỉ ở Giới thiệu (tóm tắt) và Kết quả (đầy đủ); tập kiểm tra/tập giữ riêng chỉ định nghĩa ở Dữ liệu; mã băm, chấm lại, trùng ảnh, danh mục kỹ thuật chỉ ở Phụ lục.
- Nội dung chính hiện sẵn: biểu đồ và bảng bảy mô hình, năm thẻ client, danh sách hạn chế đầy đủ; chỉ bảng dài và mục thống kê phụ còn gấp. Chương Kết quả còn bốn tab (trước đây sáu).
- Địa chỉ cũ (`#tong-quan`, `#non-iid`, `#huan-luyen`, `#bang-chung`) được router đổi sang địa chỉ mới; mã tham gia đã in vẫn dùng được.
- Sửa kèm: bảng nhật ký vòng của phụ lục không còn cột độ chính xác/loss/Macro-F1 (chạy trên ảnh nhiễu); câu "Tổng N ảnh:" bị gãy ở thẻ tập giữ riêng; tiêu đề thẻ trùng ảnh nhất quán; bảng bảy mô hình luôn mở (trước đây chỉ tự mở sau khi vào tab PlantDoc).

## 8. Review UI lần 2 (desktop và laptop)

Đo lại 8 trang ở 1280, 1366, 1440, 1536, 1920 px, nền sáng và tối (điện thoại để sau theo yêu cầu). Đã sửa:

- Tiêu đề mục đánh số như báo cáo ("1.1 Bài toán", "4.3.1 …", "A. Kiểm chứng số liệu") qua một helper `sectionHead` trong `core/dom.js`;
  câu tiêu đề dài cũ thành dòng dẫn. Cấp tiêu đề thống nhất: mục h3, khung con h4, khung trong mục con h5.
- Độ dài dòng: danh sách hạn chế, hướng phát triển, hạng mục đã xong và chữ trong khung lưu ý giữ khoảng 70–80 ký tự mỗi dòng
  ở mọi bề rộng (trước đây tới 140 ký tự ở 1920 px).
- Chương 2: định nghĩa ba tập ảnh không còn in đậm cả đoạn; ví dụ Tomato — Bacterial spot có bảng nhỏ "Mô hình | Ảnh nhóm này đã học | Tỉ lệ nhận đúng".
- Chương 5: thẻ kết luận hiện hiệu số ngắn (−6,1 / +37,7 / −45,9 điểm) kèm một dòng nói rõ phép so sánh.
- Kết quả: mục gấp của tab Tập giữ riêng cùng kiểu với các mục gấp khác, bỏ tiêu đề lặp; tab dùng roving tabindex.
- Demo: trang Chẩn đoán chỉ còn một câu hướng dẫn khi chưa có ảnh; trang Học liên kết chia hai cột từ 1280 px (các bước bên trái,
  "Các vòng gần đây" bên phải), tiêu đề khung không lặp tiêu đề trang.
- Phụ lục: "Chạy thử trên máy chủ" thay cho "Mô phỏng trên máy chủ"; mục lục dính có viền dưới.

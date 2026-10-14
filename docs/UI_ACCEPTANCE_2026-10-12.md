# Báo cáo nghiệm thu UI — 12/10/2026 (cập nhật sau sửa lỗi)

Phạm vi: web demo 6 bước, đối chiếu đủ 74 mục trong hai checklist người dùng cung cấp. Đây là báo cáo kiểm thử, không phải chứng nhận WCAG/ISO.

**Kết luận: chưa thể nghiệm thu toàn bộ.** Cả 7 lỗi của vòng kiểm thử trước đã được sửa và đo lại; vòng này phát hiện và sửa thêm 8 vấn đề. Còn 1 mục Chưa đạt (P1: LCP trên hồ sơ mobile giả lập vượt 2,5 s), chưa thử screen reader thật, và 44 mục ở trạng thái "Một phần".

Tổng hợp 74 mục: **23** Đạt · **1** Chưa đạt · **44** Một phần · **2** Chưa xác minh · **4** Không áp dụng. Trước vòng sửa: 12 Đạt · 11 Chưa đạt · 37 Một phần · 10 Chưa xác minh · 4 Không áp dụng. Các mục "Một phần" và "Chưa xác minh" không được tính là đạt.

## Cách đọc trạng thái

- **Đạt**: đã quan sát trực tiếp trên UI hoặc đo bằng công cụ trong phạm vi nêu ở cột bằng chứng.
- **Chưa đạt**: có bằng chứng lỗi cụ thể.
- **Một phần**: đã thử một số trường hợp đại diện, chưa đủ để kết luận toàn bộ điều kiện của mục.
- **Chưa xác minh**: thiếu phép thử hoặc phép đo cần thiết; không tính là đạt.
- **Không áp dụng**: tính năng không có trong web demo hiện tại.

## Môi trường và phạm vi kiểm thử

- Server chạy mã đã sửa bằng `.venv` Python 3.11 + uvicorn; trình duyệt Chromium điều khiển bằng Playwright (Python 3.13), không phải trình duyệt tích hợp của vòng trước.
- **Bố cục**: 6 bước × 4 viewport (320×700, 768×900, 700×320, 1920×1080), 6 bước ở dark 320, 6 tab con của Kết quả ở 320; mô phỏng zoom 200%/400% (viewport 640/320 CSS px, DPR 2/4).
- **Chức năng**: Bắt đầu→Dừng ngay ở 3 chế độ huấn luyện; 7 ca sai biểu mẫu; upload 11 MB / 0 byte / văn bản / ảnh 9,5 MB hợp lệ; chặn API để ép bằng chứng `available:false`, abort, 500, 502 HTML, trễ 5 s, `/api/status` treo; bấm lặp; đổi kích thước và đổi bước giữa tác vụ.
- **Trợ năng**: axe-core 4.10.2 trên 48 trạng thái (6 bước × light/dark × 320/1280; 6 tab Kết quả; Chẩn đoán có/không kết quả), chạy với `prefers-reduced-motion`; duyệt phím Tab từng bước.
- **Hiệu năng (lab)**: LCP/CLS/FCP/long task/INP-xấp-xỉ bằng PerformanceObserver, 3 lần mỗi ô; hồ sơ desktop (không throttle) và mobile giả lập (CPU 4×, 1,6 Mbps, RTT 150 ms); cuộn có đo khung hình; soak 300 lần điều hướng.
- **Tự động**: pytest toàn bộ trên mã cuối: 274 passed (gồm 18 hàm test mới: 13 hợp đồng frontend, 3 giới hạn upload ≈ 7 ca, 2 race Dừng); `node --check` 19 tệp JS.
- **Giới hạn**: số đo là dữ liệu lab một máy, không phải p75 thực địa; INP chỉ là thời lượng sự kiện lớn nhất trong một kịch bản ngắn; zoom là mô phỏng; chưa có screen reader thật; chỉ Chromium; không có thiết bị cảm ứng thật.

## Lỗi đã xử lý

| Mức | Vấn đề (vòng trước) | Cách sửa | Bằng chứng sau sửa |
| --- | --- | --- | --- |
| P2 | Bước 6 tràn ngang ở 320 px (334/320) | `.todo-grid`, `.verify-grid`, `.pair-grid` dùng `minmax(min(100%, N), 1fr)`; giảm padding/margin ở ≤560 px | 36/36 ca `scrollWidth = innerWidth`; test `test_wide_grid_minimums_never_exceed_their_container` |
| P2 | Bắt đầu→Dừng ngay: báo "không có tác vụ" rồi tác vụ vẫn chạy. Nguyên nhân gốc nằm cả ở máy chủ: `status="running"` chỉ được đặt trong BackgroundTask sau khi trả lời và `cancel_requested` bị đặt lại | `_mark_task_running()` đặt trạng thái đồng bộ trong endpoint; BackgroundTask không đặt lại; nút Dừng chờ "Đang khởi tạo…" tới khi POST thành công | 3/3 chế độ: Dừng bị khóa khi vừa bấm Bắt đầu, sau đó dừng thành công, trạng thái cuối `cancelled`, không có toast "Không có tác vụ"; 2 test engine |
| P2 | Epoch=0 được gửi lên và nhận 422 tiếng Anh, không có lỗi tại trường | `validateTrainForm()` đọc `validity`; lỗi tiếng Việt dưới trường, `aria-invalid`, `aria-describedby`, focus; `required` cho ô số bắt buộc; `formatApiError` Việt hóa 422 | 7/7 ca chặn trước khi gửi; test chạy mô-đun bằng Node |
| P2 | Upload không giới hạn dung lượng; máy chủ đọc hết vào RAM (cả `/api/inference/plantvillage`) | Giới hạn 10 MB và 40 triệu điểm ảnh ở client và máy chủ (413, kiểm trước khi giải mã) | 11 MB / 0 byte bị chặn ở client; POST thẳng → 413; 7 ca test mới; test khớp hằng số client/server |
| P2 | Evidence `available:false`: 5 subtab không có handler | Gắn handler trước nhánh unavailable; mọi panel nêu lý do | 6/6 tab đổi panel và có lý do |
| P3 | Subtab có thanh cuộn dọc thừa (44/45 px) | `overflow-y:hidden`, bù 1 px đáy, `outline-offset:-3px` cho focus | `clientHeight = scrollHeight = 45`; ảnh chụp vạch chọn nguyên vẹn |
| P3 | Toast lỗi không tự tắt, che nội dung khi đổi bước ở 320 px | Dọn toast lỗi ngay đầu `show()` của router; toast mất kết nối `sticky` được giữ | Lỗi còn nguyên 8 s ở cùng bước, biến mất khi đổi bước; toast offline còn |
| P3 | Thông báo trạng thái dùng thuật ngữ "Flower smoke" | Đổi sang "Chạy thử Flower" (chỉ chuỗi hiển thị; mã máy giữ nguyên) | grep không còn chuỗi cũ; 0 test phụ thuộc |
| **Mới** P2 | Tab "Sai mà rất tự tin" tràn ngang 339/320 px (vòng trước chỉ đo tab mặc định) | `.segmented { max-width:100% }` để nhóm nút xuống dòng | 320/320; test `test_segmented_groups_wrap…` |
| **Mới** P2 | Ô số để trống vẫn được gửi (không có `required`) | `required` trên 5 ô số | ca `epoch=''`: 0 POST |
| **Mới** P2 | `aria-expanded` đặt trên `<tr>` của table (axe `aria-conditional-attr`, 28 nút) | Nút `.row-toggle` thật trong ô tên mô hình | axe 0; Enter/Space/chuột đều mở top-3 |
| **Mới** P2 | Vùng bảng cuộn được không focus bằng phím (axe `scrollable-region-focusable`, 16 nút) | `tabindex="0"` + `role` + `aria-label` cho 14 vùng cuộn | axe 0; test mới |
| **Mới** P3 | `aria-label` trên `div` không có role (axe `aria-prohibited-attr`) | `role="group"` | axe 0 |
| **Mới** P2 | CLS 0,07–0,22 ở hồ sơ mobile giả lập: thanh Trước/Sau nhảy từ y≈217 xuống y≈761 khi trang mount xong | Ẩn thanh (`visibility`) tới khi mount xong | CLS ≤ 0,005 ở 5/6 bước, 0,086 ở Kết quả (≤ 0,1) |
| **Mới** P2 | LCP mobile giả lập 4,2–5,3 s: `app.js` phải chờ Chart.js 201 KB (script `defer`) và chuỗi import nhiều tầng | Chart.js `async` + hàng đợi trong `makeChart` (có ghi chú sau 8 s nếu lỗi); `modulepreload` cho chuỗi import bước mặc định | Bước mặc định 4,39 → 2,74 s; Chẩn đoán 4,30 → 2,76 s; biểu đồ vẫn dựng ở desktop và mobile giả lập |
| **Mới** P3 | Lỗi mạng hiện "Failed to fetch"; HTTP 5xx không JSON hiện "HTTP 502" | Câu tiếng Việt cho `TypeError` của `fetch` và cho 4xx/5xx không có thông điệp | C3 đạt; test chạy mô-đun bằng Node |

## Vấn đề còn mở

| Mức | Vấn đề và bằng chứng | Hướng xử lý |
| --- | --- | --- |
| P2 | LCP hồ sơ mobile giả lập vẫn > 2,5 s: 2,74 s (mặc định), 3,19 s (Huấn luyện), 4,36–4,74 s (Non-IID, Kết quả, Bằng chứng). FCP ~2,1 s do 2 CSS chặn render và 8 font preload cạnh tranh băng thông | `modulepreload` module của bước theo hash trong URL; giảm font preload; tách `pages.css` theo bước. Cần đo lại sau mỗi thay đổi |
| P3 | Chuyển tab con ở Kết quả trên CPU 4×: sự kiện lớn nhất 192 ms (median), 224 ms (max) — sát/vượt 200 ms | Dựng ma trận nhầm lẫn theo từng phần hoặc hoãn bằng `requestIdleCallback` |
| P3 | Ảnh hợp lệ 9,5 MB gây long task 1,08 s khi xem trước | Giảm kích thước ảnh xem trước bằng canvas trước khi gắn vào `<img>` |
| P3 | Tab "Sai mà rất tự tin" cao ~56.500 px ở 320 px, không phân trang | Phân trang hoặc "Xem thêm" |
| P3 | axe `heading-order` (moderate): `h1` rồi `h3` ở mọi bước, 44 trạng thái | Đổi cấp tiêu đề panel (`h3`→`h2`, `h4`→`h3`) kèm rà CSS selector theo thẻ |
| P3 | axe `skip-link` (moderate) trên liên kết điều hướng `#tong-quan`…: đích hash không phải id phần tử | Theo thiết kế của router hash; có thể bỏ qua hoặc thêm `aria-current` thay vì href fragment |
| P3 | Nhãn tiếng Anh trong Tùy chọn nâng cao: Engine, Validator, Held-out evaluation root | Việt hóa nhãn |
| Thông tin | axe không quyết được 28 phần tử về tương phản (3 nền gradient, 24 bị che một phần, 1 nền ảnh) và 4 bảng nhiệt `th-has-data-cells` | Rà thủ công |

### Kiểm tra riêng lỗi "Đã dừng toàn bộ tiến trình chạy thử Flower."

Chạy Flower, bấm Dừng khi task thật sự chạy: tiến trình thành "đã dừng theo yêu cầu", chi tiết ở bước Huấn luyện giữ "Đã dừng toàn bộ tiến trình chạy thử Flower." để ghi lịch sử phiên, còn badge chung ở đầu trang về **"Hệ thống sẵn sàng"** (`status.js` phân biệt `cancelled`). Đã lặp lại sau khi sửa race: 3/3 chế độ cho cùng kết quả. Thông báo này nay chỉ xuất hiện đúng lúc có tác vụ để dừng.

Ảnh chụp badge sau khi dừng (vòng trước): [ui-acceptance-proof-2026-10-12.png](ui-acceptance-proof-2026-10-12.png).


## Checklist 74 mục

### 1. Visual Design (6)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| V1 | Phân cấp thị giác, nội dung và hành động chính | Đạt | Hero bước Tổng quan, tiêu đề từng bước và CTA chính rõ trong desktop/mobile đã xem. |
| V2 | Căn lề, khoảng cách nhất quán | Một phần | Container chung; 0/36 ca tràn ngang sau sửa. Chưa đo khoảng cách từng component. |
| V3 | Màu sắc hài hòa và biểu thị trạng thái | Một phần | Light/dark quan sát được; axe-core 0 vi phạm `color-contrast`, nhưng 28 phần tử trên 6 bước "cần xem lại" (nền gradient/bị che) chưa kết luận. |
| V4 | Shadow, border, radius, icon có hệ thống | Một phần | Dùng token chung; chưa so từng biến thể và icon ở mọi trạng thái. |
| V5 | Không chồng đè, cắt mất, nhòe, sai tỉ lệ | Một phần | Hai lỗi đã biết (bước 6 và tab "Sai mà rất tự tin" tràn 320 px; toast che nội dung) đã sửa và đo lại; chưa soát bằng mắt từng trạng thái nên chưa nâng lên Đạt. |
| V6 | Trang trí không lấn át nội dung | Một phần | Nền chấm và hình minh họa không che CTA ở các màn đã xem; chưa có thử người dùng. |

### 2. Layout & Responsive (7)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| L1 | Không tràn ngang/mất nội dung khi resize | Đạt | 36/36 ca `scrollWidth ≤ innerWidth`: 6 bước × (320×700, 768×900, 700×320, 1920×1080) + 6 bước dark 320 + 6 tab con của Kết quả ở 320; thêm 12 ca zoom mô phỏng. Đã sửa 2 nguồn tràn. |
| L2 | Grid/card/table/dialog thích ứng | Đạt | Lưới `.todo-grid/.verify-grid/.pair-grid` dùng `minmax(min(100%,…),1fr)`; nhóm nút phân đoạn xuống dòng; bảng cuộn trong khung. Demo không có dialog. |
| L3 | Breakpoint không tạo khoảng trống/lệch/chữ đè | Một phần | Đã đo 320, 390, 640, 700, 768, 1280, 1920; chưa quét các điểm gãy trung gian (560/760/1100/1180) và chưa đo chữ đè tự động. |
| L4 | Menu/popover/tooltip trong màn hình | Một phần | Điều hướng và tab con dùng được ở 320; không có ca popover/tooltip đầy đủ. |
| L5 | Màn nhỏ/lớn/xoay ngang dọc | Đạt | Đã đo 320×700, 390×844, 700×320 (ngang), 768×900, 1280×800, 1920×1080; đổi 320↔1280 khi đang huấn luyện và khi đang hiện kết quả không tràn. |
| L6 | Scroll/sticky không che nội dung | Một phần | Subtab hết thanh cuộn dọc thừa (45/45 px); toast lỗi tắt khi đổi bước. Chưa kiểm sticky/anchor ở mọi trang. |
| L7 | Zoom chữ/mật độ hiển thị vẫn dùng được | Một phần | Mô phỏng 200% và 400% (viewport 640/320 CSS px, DPR 2/4): 0 px tràn ở 6 bước. Chưa thử Ctrl+plus/zoom chữ thật của trình duyệt. |

### 3. Typography & Content (6)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| T1 | Font dễ đọc, đủ ký tự tiếng Việt | Đạt | Đã xem tiêu đề, nhãn, bảng và thông báo có dấu trong light/dark. |
| T2 | Heading/body/caption/label/link phân biệt | Đạt | Phân cấp thể hiện rõ ở Tổng quan, Kết quả và Huấn luyện. |
| T3 | Kích thước, đậm, line-height nhất quán | Một phần | Token CSS chung; chưa đo từng component ở mọi viewport. |
| T4 | Nội dung dài wrap/scroll/truncate hợp lý | Một phần | Nội dung bước Bằng chứng và tab Sai mà rất tự tin không còn tràn ở 320 px; chưa thử chuỗi dài do người dùng nhập (tên tệp). |
| T5 | Câu chữ rõ, tránh thuật ngữ khó hiểu | Một phần | Thông báo trạng thái đã đổi "Flower smoke" → "Chạy thử Flower". Còn thuật ngữ kỹ thuật ở mục nâng cao; chưa có thử người dùng. |
| T6 | Số/ngày/đơn vị/tiền tệ đồng nhất | Một phần | Số dùng `vi-VN` (`core/dom.js`); không có tiền tệ; chưa rà toàn bộ ngày/múi giờ. |

### 4. UX & Navigation (6)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| U1 | Biết đang ở đâu và bước tiếp | Đạt | Nav 1–6, tiêu đề "Bước x/6", trạng thái tab hiện tại và CTA chuyển bước. |
| U2 | Menu/luồng dễ hiểu | Đạt | Đã đi qua cả 6 bước theo thứ tự, dùng hash route và nút bước kế. |
| U3 | Có quay lại/hủy/thoát/hoàn tác khi phù hợp | Một phần | Có điều hướng lại và Dừng huấn luyện; chưa kiểm hoàn tác cho mọi thao tác. |
| U4 | Chức năng chính dễ thấy | Đạt | CTA chẩn đoán, chuyển dữ liệu, bắt đầu/dừng huấn luyện hiện rõ trong màn tương ứng. |
| U5 | Tránh thao tác thừa/nhập lại | Một phần | Giá trị form và ảnh đã chọn được giữ sau lỗi và sau khi đổi bước; chưa kiểm mọi chuyển trang. |
| U6 | Hành động nguy hiểm có xác nhận/phục hồi | Không áp dụng | Luồng demo không xóa dữ liệu không thể phục hồi; Dừng có thể khởi chạy lại. |

### 5. Interaction & Feedback (7)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| I1 | Click/tap/hover/focus/keyboard dự đoán được | Một phần | Race Bắt đầu→Dừng đã sửa (3 chế độ); duyệt Tab 6 bước không bẫy phím, vòng focus luôn thấy. Chưa rà hover/cảm ứng thiết bị thật. |
| I2 | Hành động có feedback nhanh/dễ nhận | Một phần | Nút Dừng hiện "Đang khởi tạo…" tới khi máy chủ nhận tác vụ; chẩn đoán, tab, dừng đều có feedback; chưa rà mọi hành động. |
| I3 | Xử lý lâu có progress, ngăn gửi lặp | Đạt | Chẩn đoán đặt busy rồi bật lại (cả khi API trễ 5 s); bấm lặp 6 ảnh liên tiếp chỉ còn 1 kết quả; Flower hiển thị tiến độ. |
| I4 | Dropdown/modal/tab/tooltip nhất quán | Một phần | Sáu subtab Kết quả và radio chạy; không có modal/tooltip để kết luận toàn mục. |
| I5 | Không có nút bấm vô tác dụng | Một phần | 6 subtab đổi panel và nêu lý do khi bằng chứng `available:false` (đã ép bằng chặn API). Chưa liệt kê hết mọi nút. |
| I6 | Không phụ thuộc riêng hover cho touch | Một phần | Luồng chính thao tác bằng click ở 320; chưa rà toàn bộ hover CSS/thiết bị cảm ứng thật. |
| I7 | Không mất dữ liệu khi đổi màn/đóng dialog | Một phần | Huấn luyện: đổi bước đi-về giữa lúc đang chạy giữ seed/vòng, nút Dừng, tiến độ. Chẩn đoán: ảnh và 7 kết quả còn nguyên. Chưa kiểm các màn còn lại. |

### 6. Component States (6)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| C1 | Default/hover/focus/active/selected/disabled | Một phần | Selected/disabled/focus quan sát được; chưa rà toàn bộ hover. |
| C2 | Loading/success/error/empty/no-results | Một phần | Đã thấy loading, completed, cancelled, 422, 413, 502, bằng chứng thiếu; chưa ép mọi empty/no-results. |
| C3 | Lỗi mạng/timeout/server dễ hiểu | Đạt | 422 → "Số epoch: tối thiểu là 1."; mất mạng → "Không kết nối được máy chủ…"; HTTP 502 HTML → "Máy chủ gặp lỗi (HTTP 502)…"; `/api/status` treo >8 s → "Mất kết nối API". |
| C4 | Disabled có lý do/hướng dẫn | Một phần | Nút busy chuyển spinner có chữ; chưa kiểm mọi điều kiện disabled. |
| C5 | Success chỉ sau thành công thật | Một phần | Hoàn tất Flower 2 vòng và upload suy luận mới hiện success; chưa kiểm mọi API race. |
| C6 | Không skeleton/spinner vô thời hạn | Một phần | Mọi yêu cầu có timeout trong `core/api.js` (status 8 s, suy luận 120 s); đã thử trễ 5 s. Chưa chờ hết 120 s. |

### 7. Forms & Validation (7)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| F1 | Mỗi input có label rõ | Đạt | AX snapshot, mã nguồn và axe-core (0 vi phạm `label`) cho thấy nhãn Số epoch, Seed, file và radio. |
| F2 | Bắt buộc/tùy chọn/định dạng giải thích | Một phần | Ô số có `required`/min/max và lỗi nêu giới hạn; vùng tải ảnh nêu "JPG, PNG, WEBP, tối đa 10 MB". Chưa có dấu bắt buộc nhìn thấy. |
| F3 | Validate đúng lúc, lỗi sát trường | Đạt | 7/7 ca (epoch 0, 11, rỗng; vòng 51; seed −1; Centralized epoch 0, 51): 0 POST, lỗi dưới trường, `aria-invalid`, `aria-describedby`, focus vào trường lỗi, lỗi tự xóa khi sửa. |
| F4 | Không xóa dữ liệu hợp lệ sau gửi lỗi | Đạt | Sau lỗi validate/422, giá trị nhập và ảnh đã chọn vẫn giữ. |
| F5 | Tab order/bàn phím nhập phù hợp | Một phần | Duyệt Tab 6 bước: 18–43 điểm dừng, không bẫy, không focus vào phần tử ẩn. Chưa đối chiếu thứ tự Tab với thứ tự nhìn. |
| F6 | Cảnh báo nguy hiểm/dữ liệu chưa lưu | Không áp dụng | Không có biểu mẫu lưu bản nháp hoặc hành động xóa dữ liệu. |
| F7 | Số/file/dữ liệu bất thường xử lý | Đạt | Số ngoài khoảng bị chặn tại form; tệp 11 MB, 0 byte, văn bản bị chặn ở client; POST thẳng 11 MB và ảnh > 40 triệu điểm ảnh → 413 (cả 3 endpoint, có test). |

### 8. Accessibility (7)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| A1 | Tương phản phù hợp | Một phần | Token light/dark vượt 4.5:1 (test tự động) và axe-core 0 vi phạm; còn 28 phần tử trên 6 bước axe chưa quyết được (3 nền gradient, 24 bị che một phần, 1 nền ảnh). |
| A2 | Thông tin không chỉ qua màu | Đạt | Badge và kết quả có chữ/trạng thái, không chỉ màu. |
| A3 | Vùng tương tác đủ kích thước | Một phần | Một số link/copy/checkbox có hộp nhỏ trong đo DOM; chưa kiểm ngoại lệ 24 px từng cái. |
| A4 | Bàn phím/focus/order hợp lý | Một phần | Skip link, tab/radio bằng phím; vùng cuộn và nút mở top-3 truy cập được bằng phím; focus chuyển tới trường lỗi. Chưa đối chiếu trật tự. |
| A5 | Screen reader/hệ trợ năng | Chưa xác minh | axe-core 4.10.2 + AX tree + ARIA trong mã; chưa dùng NVDA/JAWS/VoiceOver. |
| A6 | Ảnh có mô tả, icon có tên | Đạt | axe-core: 0 vi phạm `image-alt`, `button-name`, `link-name`, `svg-img-alt` trên 48 trạng thái (6 bước × 2 theme × 2 độ rộng, 6 tab Kết quả, Chẩn đoán có/không kết quả). |
| A7 | Zoom chữ/giảm chuyển động | Một phần | Có `prefers-reduced-motion`; quét axe và đo trong chế độ giảm chuyển động chạy được; zoom chỉ mô phỏng. |

### 9. Performance & Stability (6)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| P1 | Load/phản hồi theo chỉ tiêu | Chưa đạt | Số đo lab (Playwright, Chromium), không phải p75 thực địa. Desktop localhost: LCP 0,33–0,52 s. Hồ sơ mobile giả lập (CPU 4×, 1,6 Mbps, RTT 150 ms): LCP 2,74 s (bước mặc định), 2,76 s (Chẩn đoán), 3,19 s (Huấn luyện), 4,36–4,74 s (3 bước còn lại) — **vượt 2,5 s**. Suy luận 7 mô hình ~0,8–1,1 s phía máy chủ (số đo vòng trước). |
| P2 | Cuộn/animation không giật | Một phần | Cuộn 3 trang dài (Ma trận nhầm lẫn, Non-IID, Bằng chứng): median 16,7 ms, p95 16,8 ms, 0% khung > 33 ms, 0 long task, cả desktop và CPU 4×. Chưa profile animation (hero, thanh độ tin cậy). |
| P3 | Không layout shift gây bấm nhầm | Đạt | CLS lab: desktop 0 ở cả 6 bước; mobile giả lập ≤ 0,086 (trước sửa: 0,07–0,22 do thanh Trước/Sau bị đẩy xuống khi nội dung mount xong). 3 lần/ô. |
| P4 | Tài nguyên tải hợp lý | Một phần | Tải lạnh 0,74–1,04 MB, 36–74 yêu cầu. Chart.js (201 KB) không còn chặn thực thi app.js; chuỗi import của bước mặc định được `modulepreload`. Còn 8 file font preload + 2 CSS chặn render (FCP mobile giả lập ~2,1 s). |
| P5 | Dữ liệu lớn không làm treo UI | Một phần | Ảnh JPEG hợp lệ 9,5 MB: có kết quả sau 3,6 s nhưng **long task 1,08 s** trên luồng chính (giải mã ảnh xem trước). Ảnh > 10 MB hoặc > 40 triệu điểm ảnh bị từ chối trước khi giải mã. |
| P6 | Không rò rỉ trong phiên dài | Một phần | Soak ngắn: 50 vòng × 6 bước (300 lần điều hướng) + 10 lần chẩn đoán: heap 4,09 → 4,18 MB, nút DOM 5057 → 5078, listener 105 → 105 (phẳng sau khi 6 trang đã mount). Chưa có phiên nhiều giờ. |

### 10. Design Consistency (5)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| D1 | Component cùng loại cùng kiểu/hành vi | Một phần | CTA, tab, badge có style chung; subtab nhất quán cả khi bằng chứng không khả dụng. Chưa lập inventory. |
| D2 | Khoảng cách/màu/chữ/icon theo token | Đạt | CSS dùng biến `--space-*`, `--fs-*`, màu và component style chung; đã đối chiếu mã nguồn. |
| D3 | Ngôn ngữ/tên hành động nhất quán | Một phần | Lỗi 422 và "Flower smoke" đã Việt hóa. Còn nhãn tiếng Anh ở Tùy chọn nâng cao: "Engine", "Validator", "Held-out evaluation root". |
| D4 | Không nhiều biến thể UI dư thừa | Một phần | Các bước dùng khung chung; chưa lập inventory mọi biến thể. |
| D5 | Web đúng kỳ vọng nền tảng | Một phần | Link, button, form và URL hash hoạt động; chỉ thử Chromium. |

### 11. Localization & Adaptation (5)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| N1 | Bản dịch dài không phá layout | Chưa xác minh | Web chỉ có tiếng Việt; chưa có bản dịch/chuỗi dài giả lập. |
| N2 | Dấu tiếng Việt/ký tự đặc biệt đúng | Đạt | Nội dung nhiều bước và thông báo có dấu hiển thị đúng trong cả hai theme. |
| N3 | Ngày/tiền tệ/múi giờ/số hợp lý | Một phần | Số định dạng `vi-VN`; tiền tệ không áp dụng; chưa kiểm toàn bộ thời gian. |
| N4 | Light/dark và hệ thống đọc được | Một phần | Cả hai theme qua 6 bước và axe-core (48 trạng thái); chưa kiểm hết tương phản và cài đặt OS khác. |
| N5 | RTL khi cần | Không áp dụng | Demo tiếng Việt, không có yêu cầu RTL. |

### 12. Edge Cases & Robustness (6)

| # | Tiêu chí | Kết quả | Bằng chứng / giới hạn |
| --- | --- | --- | --- |
| E1 | Rỗng/cực dài/quá nhiều/ký tự lạ | Một phần | Tệp rỗng/văn bản/quá lớn được xử lý; chưa thử chuỗi cực dài. Tab "Sai mà rất tự tin" dài ~56.500 px ở 320 px, không phân trang. |
| E2 | API chậm/mất mạng/lỗi | Đạt | Đã thử: trễ 5 s, abort, 500 JSON, 502 HTML, `/api/status` treo; thông báo tiếng Việt và giao diện hồi phục. |
| E3 | Thao tác nhanh/lặp liên tục | Đạt | Bắt đầu→Dừng ngay ở Centralized, FedAvg simulation, Flower: 0 toast sai, trạng thái cuối `cancelled`; bấm 6 ảnh liên tiếp + đổi chế độ 25 lần: 1 ảnh được chọn, 7 kết quả, 0 lỗi. |
| E4 | Danh sách dài/cuộn/lọc/phân trang | Một phần | Lọc PlantDoc và subtab/table chạy; danh sách dài không có phân trang/ảo hóa. |
| E5 | Resize/xoay khi đang thao tác | Đạt | Đổi 1280↔320 khi đang huấn luyện và khi đang hiện kết quả chẩn đoán: giữ trạng thái, không tràn. |
| E6 | Quyền khác nhau/phiên đăng nhập hết hạn | Không áp dụng | Web demo không có đăng nhập/phân quyền. |

## Số đo hiệu năng (lab)

Median của 3 lần/ô, tải lạnh, sau đó chạy một chuỗi thao tác ngắn (đổi tab con/chế độ, bấm Sau). INP~ là thời lượng sự kiện tương tác lớn nhất, không phải INP chuẩn.

| Bước | Hồ sơ | LCP (s) | CLS | FCP (s) | INP~ (ms) | TBT (ms) | Yêu cầu / KB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tổng quan | desktop | 0,49 | 0 | 0,16 | 24 | 11 | 37 / 737 |
| Non-IID | desktop | 0,33 | 0 | 0,13 | 24 | 9 | 38 / 779 |
| Kết quả | desktop | 0,38 | 0 | 0,13 | 48 | 8 | 62 / 1042 |
| Chẩn đoán | desktop | 0,49 | 0 | 0,13 | 24 | 9 | 74 / 1043 |
| Huấn luyện | desktop | 0,52 | 0 | 0,14 | 16 | 9 | 39 / 768 |
| Bằng chứng | desktop | 0,33 | 0 | 0,12 | 32 | 12 | 36 / 737 |
| Tổng quan | mobile giả lập | **2,74** | 0 | 2,21 | 32 | 1116 | 37 / 737 |
| Non-IID | mobile giả lập | **4,50** | 0 | 2,14 | 40 | 1017 | 38 / 779 |
| Kết quả | mobile giả lập | **4,74** | 0,086 | 2,14 | 192 | 1150 | 43 / 906 |
| Chẩn đoán | mobile giả lập | **2,76** | 0,005 | 2,14 | 64 | 637 | 74 / 1043 |
| Huấn luyện | mobile giả lập | **3,19** | 0,003 | 2,22 | 72 | 1050 | 39 / 768 |
| Bằng chứng | mobile giả lập | **4,36** | 0 | 2,18 | 40 | 1194 | 36 / 737 |

Trước khi sửa (cùng hồ sơ mobile giả lập): LCP 4,20–5,29 s và CLS 0,066–0,224 ở cả 6 bước.

## Kiểm lại sau chỉnh bố cục màn hình rộng (13/10/2026)

Một phiên khác đã nới khung nội dung (1280 → 1760 px, gutter co theo viewport), thêm ảnh lá mẫu và nhãn "5 client" ở hero, giảm chấm nền và chiều cao hero. Lượt này kiểm lại thay đổi đó bằng cùng bộ đo:

| Phép đo | Kết quả |
| --- | --- |
| 6 bước × 320 / 768 / 1280 / 1536 / 1920 / 2560 px | 0 ca tràn ngang; khung chính 1536 px ở 1536, 1760 px ở ≥ 1920 |
| Mở mọi `<details>` (lồng nhau) ở 6 bước và 6 tab Kết quả, 4 viewport | 44 ca; tìm thấy và sửa 1 ca tràn 2 px ở bước Bằng chứng 320 px (`.limit-list` là lưới `auto`) |
| Chiều cao hero Tổng quan | 330 px ở ≥ 1536 px, 341 px ở 1280 px (trước: 395 px; giảm ~16%, đạt mục tiêu 15–20%). Thu ảnh lá 220 → 176 px, sơ đồ 440 → 400 px, padding 20 → 16 px |
| Độ dài dòng của đoạn văn | tối đa 747 px (~85ch) ở mọi bề rộng |
| axe-core, 48 trạng thái | vẫn chỉ `heading-order` và `skip-link` (moderate); không vi phạm mới |
| Console | không có lỗi |

Chuyển động: hero có thêm đợt "mô hình chung quay lại từng client" sau đợt gửi trọng số (minh họa, tôn trọng giảm chuyển động); ở bước Huấn luyện, client đang huấn luyện thật có vòng sáng lan ra từ thẻ (chỉ transform/opacity). Sơ đồ thật vẫn chỉ chạy theo sự kiện thật, không dựng đợt quay lại.

Hai điểm lệch chỉ lộ ra khi khung rộng, đã sửa: sơ đồ theo dõi ở bước Huấn luyện cao 659 px ở 1920 px (nay tối đa 820×461 px), và thanh phân bố ở bước Dữ liệu dài ~1.400 px làm nhãn client cách xa số ảnh (nay tối đa 960 px).

## Điều kiện để ký nghiệm thu toàn bộ

1. Đưa LCP hồ sơ mobile giả lập ≤ 2,5 s (hiện 2,74–4,74 s) hoặc ghi rõ môi trường triển khai thực tế là localhost/LAN để bỏ hồ sơ này; INP của Kết quả sát 200 ms cần theo dõi.
2. Thử screen reader thật (NVDA/JAWS/VoiceOver) và zoom chữ 200% thật bằng trình duyệt; mục A5 chưa thể đánh dấu đạt khi thiếu phép thử này.
3. Chạy các mục "Một phần" còn lại với phép thử bổ sung trước khi đánh dấu đạt; "Không áp dụng" không được cộng như mục đạt.
4. Đo lại trên thiết bị cảm ứng thật và trình duyệt khác Chromium (D5, I6).
5. Quyết định xử lý nhóm vấn đề còn mở ở trên (heading, danh sách dài, ảnh lớn).

Không có lỗi P0/P1 đã xác nhận trong phạm vi thử. Những mục chưa xác minh không thể được suy ra là đạt chỉ vì các luồng thông thường hoạt động.

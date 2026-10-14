# Review màn hình rộng và đối chiếu UI — 13/10/2026

## 1. Kết luận

Bản thử đã rút chữ và phân cấp thông tin tốt hơn, nhưng bố cục màn hình rộng vẫn bị giới hạn bởi một khung dùng chung 1280px. Các trang thao tác và xem bảng chưa tận dụng được phần màn hình còn lại. Màn hình đầu còn nhiều lớp khung, nhãn và thiếu hình ảnh lá cây thực tế. Đây là các nhận xét thiết kế, không phải kết luận nghiệm thu toàn bộ chức năng.

## 2. Vì sao hai bên trống?

Source of truth:

- `web/static/css/style.css`: `--page-max: 1280px`, `--page-gutter: 32px`.
- `.main-content`: `max-width: var(--page-max)`, `margin: 0 auto`, padding ngang theo gutter.
- Nền chấm vẫn phủ toàn bộ viewport, làm hai lề nhìn thấy rõ.

Trong ảnh người dùng cung cấp, phần nội dung từ x≈191 đến x≈1711, rộng khoảng 1520px; mỗi bên trống khoảng 190px. Đây là đo gần đúng từ ảnh, không phải giá trị CSS. Không suy ra browser zoom hay Windows scale chỉ từ ảnh.

Đo live riêng trên in-app browser với viewport yêu cầu 1920×1000:

| Thành phần | Giá trị CSS đọc từ DOM |
| --- | ---: |
| Vùng hiển thị trừ scrollbar | 1905px |
| Khung main | 1280px |
| Lề ngoài main mỗi bên | 312,5px |
| Padding main mỗi bên | 32px |
| Hero thực dùng | 1216px |
| Từ mép viewport tới nội dung hero | 344,5px |
| Chiều cao hero | 395,36px |

Cả Dữ liệu, Kết quả, Chẩn đoán, Huấn luyện và Bằng chứng cũng có cùng khung main 1280px tại viewport này. Evidence JSON: `ui-comparison-2026-10-13/width-measurements.json`.

Hai lề giúp giới hạn độ dài dòng, nhưng một giới hạn chung cho cả đoạn văn, bảng dữ liệu và khu thao tác tạo ra việc dùng diện tích chưa hợp lý. Màn hình nhỏ không bộc lộ hạn chế này vì khung chưa chạm max-width.

## 3. Nguồn đối chiếu

### Plantix — sản phẩm chẩn đoán cây

- Đã mở trực tiếp [Plantix](https://www.plantix.net/en/) trên browser, đóng hộp cookie và lưu screenshot.
- Trang đầu tập trung vào chẩn đoán cây và một nút Download Plantix.
- Visual lớn là ảnh cây/người dùng kết hợp màn hình ứng dụng; người xem nhận ra lĩnh vực ngay.
- Các nội dung chẩn đoán, tư vấn, thư viện và số liệu xuất hiện theo các section phía dưới.
- Khoảng trắng nhiều nhưng hướng chú ý rõ; không thể kết luận mọi web cùng chủ đề đều phải sát hai mép.

### Flower — nền tảng học liên kết/collaborative AI

- Đã mở trực tiếp [Flower](https://flower.ai/) trên browser và lưu screenshot.
- Hero tập trung vào một tiêu đề lớn, một câu mô tả và hai CTA: thử sản phẩm/học tutorial.
- Nội dung mạng cộng tác, công cụ và bằng chứng cộng đồng nằm trong các section sau.
- Cách trình bày có ít khung bao quanh hero hơn CropFed; khoảng trắng dùng để giữ tiêu đề và CTA làm điểm nhấn.

Screenshot ngoại bộ được chụp ở viewport 1280px (1265px vùng nội dung), còn CropFed được đo riêng ở 1920px. Đây là so sánh phân cấp, visual và cấu trúc; không phải phép so sánh kích thước ở cùng viewport.

Plant.id đã được tìm trong nguồn công khai nhưng không đưa vào đánh giá hình thức trực tiếp: browser gặp lỗi chứng chỉ, không xác minh được giao diện render. Không vượt cảnh báo chứng chỉ.

## 4. Các điểm UI cần chỉnh tiếp

| Ưu tiên | Điểm hiện tại | Tác động | Hướng chỉnh |
| --- | --- | --- | --- |
| Cao | Mọi mục bị giới hạn 1280px | Bảng, ảnh và khu thao tác chưa tận dụng màn hình rộng | Khung linh hoạt theo viewport, trần khoảng 1760px; đoạn văn giữ giới hạn dòng riêng |
| Cao | Header, tiêu đề Tổng quan và hero cùng dẫn vào một thông điệp | Tốn chiều cao trước khi thấy nội dung hữu ích | Giảm khối tiêu đề phía trên; hero gọn hơn khoảng 15–20% trên desktop |
| Cao | Hero chỉ có sơ đồ, chưa có ảnh lá/kết quả ví dụ | Người mới thấy học liên kết nhưng chưa thấy rõ sản phẩm chẩn đoán lá | Thêm một ảnh mẫu thật và một kết quả có nhãn nguồn; giữ sơ đồ làm minh họa quy trình |
| Vừa | Hero chứa thêm card sơ đồ, node card và hàng card số liệu | Nhiều lớp viền, hình thức còn giống slide/dashboard | Giảm khung lồng nhau; dùng nền, khoảng cách và đường phân cách cho phân nhóm |
| Vừa | Nhãn Client bị ẩn khi container sơ đồ ≤480px | Trong ảnh chỉ còn 00–04; người mới phải suy ra ý nghĩa | Một nhãn nhóm “5 client” luôn hiện; hoặc giữ Client ở kích thước còn đọc được |
| Vừa | Độ chính xác là số lớn, người xem dễ đọc như chất lượng sản phẩm chung | Phạm vi dữ liệu có thể bị bỏ qua | Đặt “Trên tập kiểm tra PlantVillage” gần tiêu đề số; giữ caveat hiện có |
| Vừa | Thanh sáu bước có nhiều tên dài | Phù hợp trình bày tuần tự nhưng nặng với người chỉ muốn chẩn đoán | Tăng ưu tiên đường vào Chẩn đoán/Kết quả; giữ sáu bước cho chế độ trình bày |
| Vừa | Serif đậm, nền ngà và nhiều viền | Nhận diện nghiên cứu/nông nghiệp rõ nhưng cảm giác công cụ AI còn nhẹ | Giữ palette xanh; thử sans cho tiêu đề tác vụ và giảm viền. Đây là lựa chọn thẩm mỹ, không phải lỗi chuẩn |
| Thấp | Nền chấm phủ cả hai lề | Khoảng trống càng dễ bị chú ý | Làm nền ngoài nội dung phẳng hoặc giảm độ rõ của chấm |
| Thấp | Chuyển động đã có nhưng rất nhẹ | Người xem ảnh tĩnh khó nhận ra feedback | Ưu tiên dòng gửi/gộp trọng số và trạng thái đang chạy; không cần thêm hiệu ứng chuột để lấp chỗ trống |

Khung xanh quanh thương hiệu trong ảnh có thể là trạng thái focus. Không coi đó là một border thừa cần xóa; cần giữ dấu hiệu focus để sử dụng bằng bàn phím.

## 5. Đề xuất riêng cho sáu mục

1. **Tổng quan:** bố cục rộng hơn; câu giới thiệu ngắn + ảnh lá/kết quả mẫu; sơ đồ gọn; scope gắn với số liệu.
2. **Dữ liệu:** tận dụng chiều ngang để đặt phân bố năm client cạnh ví dụ; bảng đầy đủ vẫn mở khi cần.
3. **Kết quả:** ba cách học trên cùng nguồn là hướng đúng của bản thử. Cho biểu đồ/bảng rộng hơn khi mở; luôn hiển thị nguồn và số ảnh. Giữ phương pháp trong phần mở rộng.
4. **Chẩn đoán:** không gian thao tác là nơi hưởng lợi nhiều nhất từ khung rộng. Gallery/ảnh đầu vào một bên, kết luận/kết quả bên kia; nhãn tải ảnh cần dễ thấy hơn danh sách mẫu.
5. **Huấn luyện:** khu thiết lập gọn, khu tiến trình và sơ đồ rộng; trạng thái đánh giá cuối vẫn hiện ngoài phần thu gọn. Nhật ký chi tiết mở theo nhu cầu.
6. **Bằng chứng:** tiếp tục giữ ba giới hạn chính ngắn. Có thể dùng khung vừa phải hơn các trang thao tác, để đoạn văn không kéo quá dài.

## 6. Hướng triển khai được đề xuất

- Màn hình rộng: khoảng 88–92% vùng hiển thị cho bố cục chính, có trần rộng hơn; khoảng cách hai mép theo viewport.
- Desktop tiêu chuẩn: gutter 24–40px; mobile 16–20px. Đây là giá trị thử nghiệm để kiểm tra, không phải quy chuẩn bắt buộc.
- Đoạn mô tả giữ khoảng 55–70 ký tự/dòng; bảng, biểu đồ và khu thao tác dùng toàn chiều rộng vùng làm việc.
- Chỉnh chiều rộng và chiều cao trước; tiếp đến giảm khung lồng nhau và thêm ảnh mẫu thật; cuối cùng tinh chỉnh chuyển động.
- Kiểm tra lại 320/768/1280/1536/1920px, việc mở details và keyboard focus sau khi thay đổi.

Lượt này là review: chưa thay đổi CSS/JS của sản phẩm. Chưa đo FPS/Core Web Vitals hoặc so sánh tốc độ với web tham chiếu. Kết luận về visual là nhận xét từ ảnh và render browser; kết luận về chiều rộng được xác minh bằng DOM và mã nguồn.

## 7. Ảnh đối chiếu

### CropFed ở viewport 1920px

![CropFed — khung nội dung giới hạn trên màn hình rộng](D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/ui-comparison-2026-10-13/cropfed-1920.png)

### Plantix ở viewport 1280px

![Plantix — tiêu đề, CTA và ảnh sản phẩm](D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/ui-comparison-2026-10-13/plantix-1280.png)

### Flower ở viewport 1280px

![Flower — tiêu đề lớn và hai CTA](D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/ui-comparison-2026-10-13/flower-1280.png)

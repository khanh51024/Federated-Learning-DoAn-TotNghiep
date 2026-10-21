# Hướng dẫn chụp ảnh lá cây để bổ sung dữ liệu

Ảnh do chính bạn chụp là nguồn quý nhất của release dữ liệu v4, vì đúng điều kiện sử dụng thật. Tài liệu này nói rõ cần chụp gì, chụp thế nào và ghi chép ra sao
để ảnh dùng được cho cả học lẫn đo. Công cụ kiểm tra: `scripts/check_own_photos.py`.

## 1. Chụp lớp nào trước

Ưu tiên theo khoảng trống dữ liệu (`artifacts/data_gap_report.json`) và theo cây bạn tiếp cận được ở Việt Nam. Mỗi lớp cần ít nhất 100 ảnh từ ít nhất 10 cây
(mục tiêu 300 ảnh), ở ít nhất 2 vườn hoặc ruộng khác nhau.

| Ưu tiên | Lớp | Ghi chú |
|---|---|---|
| Cao | Tomato: khỏe, Early blight, Late blight, Leaf Mold, Septoria, Bacterial spot, Target spot, Spider mites, TYLCV, mosaic | cà chua phổ biến; Target spot và Spider mites hiện gần như không có ảnh ngoài đồng |
| Cao | Corn: khỏe, Common rust, Northern Leaf Blight, Gray leaf spot | ngô |
| Cao | Pepper, bell: khỏe, Bacterial spot | ớt chuông |
| Cao | Potato: khỏe, Early blight, Late blight | khoai tây |
| Trung bình | Grape: khỏe, Black rot, Leaf blight (Isariopsis), Esca | nho |
| Trung bình | Strawberry: khỏe, Leaf scorch | dâu tây |
| Trung bình | Orange (cam, quýt): Haunglongbing (vàng lá gân xanh) | |
| Thấp | Apple, Cherry, Blueberry, Peach, Raspberry, Soybean, Squash | khó tìm ở Việt Nam; nếu có thì chụp |

Nếu một cây có nhiều lá bệnh, vẫn chụp nhiều lá nhưng dùng **cùng một mã cây** (xem mục 3).

## 2. Cách chụp

- Chụp ngoài trời, ánh sáng tự nhiên, **không dùng nền trắng hay nền xám đặt sẵn** (ảnh phòng thí nghiệm đã có quá nhiều).
- Với mỗi cây, chụp cả hai kiểu: (a) cận một lá, lá chiếm khoảng 30–70% khung hình; (b) lá còn gắn trên cây, thấy các lá xung quanh.
- Đổi góc, đổi khoảng cách, đổi thời điểm trong ngày, cả lá mặt trên và mặt dưới nếu triệu chứng ở mặt dưới.
- Chụp cả lá khỏe của cùng loại cây (lớp "khỏe" cần thật nhiều ảnh, vì hiện hầu như chưa có ảnh khỏe ngoài đồng).
- Điện thoại bất kỳ, tắt chế độ làm đẹp, không dùng bộ lọc màu. Không cần chỉnh ảnh.
- Không chụp lại cùng một lá nhiều lần chỉ khác vài điểm ảnh; hai ảnh giống nhau gần như hoàn toàn không giúp ích.

## 3. Ghi chép: tệp `labels.csv`

Đặt ảnh vào `data/own_photos/raw/` (có thể chia thư mục con theo ngày hoặc vườn) và tạo tệp `data/own_photos/labels.csv`
(chạy `python scripts/check_own_photos.py --template` để có tệp mẫu). Mỗi ảnh một dòng:

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `file` | đường dẫn ảnh trong `raw/` | `vuon_A/IMG_2041.jpg` |
| `class` | tên lớp đầy đủ hoặc tên ngắn không phân biệt hoa thường | `Tomato___Late_blight` hoặc `tomato late blight` |
| `plant_id` | mã cây; **mọi ảnh của cùng một cây phải cùng mã** | `ca_chua_A_03` |
| `location` | vườn hoặc ruộng | `vuon_A` |
| `date` | ngày chụp | `2026-10-22` |
| `device` | điện thoại | `Samsung A54` |
| `verified` | `expert` nếu có người chuyên môn xác nhận nhãn; `self` nếu bạn tự nhận biết; `unsure` nếu chưa chắc | `self` |

Quy tắc quan trọng nhất là **mã cây**. Công cụ chia ảnh thành phần học (khoảng 70% số cây) và phần thử (khoảng 30% số cây) theo cây, không theo ảnh.
Nếu hai ảnh của cùng một cây nằm ở hai phía, mô hình sẽ "thấy trước" cây trong lúc học và độ chính xác đo được sẽ cao giả tạo.
Lớp chỉ có một cây thì toàn bộ vào phần học và không đo được; vì vậy mỗi lớp cần ít nhất 3 cây.

Ảnh đánh dấu `unsure` không được dùng. Nếu không chắc ảnh bị bệnh gì, đánh dấu `unsure` còn hơn đoán: nhãn sai làm hại mô hình nhiều hơn là thiếu ảnh.

## 4. Chạy kiểm tra

```powershell
.\.venv\Scripts\python.exe -m scripts.check_own_photos
```

Công cụ xoay ảnh theo EXIF, thu nhỏ cạnh dài về 640 px, loại ảnh trùng và ảnh quá nhỏ, rồi ghi `data/own_photos/manifest.jsonl`, `report.json` và
`test_lock.json` (mã băm của phần thử). Phần thử được khóa trước khi bất kỳ ảnh học nào đưa vào release; **không đổi nhãn hay chuyển ảnh giữa hai phần sau khi đã khóa**.

## 5. Ảnh có bị gửi đi đâu không

Không. Ảnh nằm trong `data/` (không được đưa vào git) và chỉ dùng trên máy của bạn để dựng release và đo mô hình. Nếu sau này bạn muốn công khai ảnh, cần quyết định riêng.

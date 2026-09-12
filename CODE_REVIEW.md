# Báo cáo kiểm tra tầng dữ liệu GĐ2

Ngày kiểm tra: 2026-09-08

## Phạm vi

Đối chiếu code do Claude tạo với GĐ2 của đề cương: mô phỏng dữ liệu PlantVillage
phân tán, không đồng nhất theo nhãn, số lượng và đặc trưng; chuẩn bị dữ liệu cho
FedAvg với MobileNetV3. Không triển khai huấn luyện.

## Vấn đề đã sửa

1. `_repair_min_size` chuyển từng ảnh nên có thể xé nhóm ảnh cùng một lá. Bản mới
   chỉ chuyển nguyên nhóm và dừng trước khi ghi manifest nếu không đạt ngưỡng.
2. Lệch số lượng dùng LogNormal nhưng NIID-Bench mục IV-D quy định
   `q ~ Dirichlet(quantity_alpha)`. Schema 3 dùng tham số Dirichlet riêng và từ
   chối cấu hình cũ có `size_sigma`.
3. `dataset_root` override dựng sai thư mục neo. Loader mới tách prefix manifest
   khỏi root ảnh được cung cấp.
4. Tắt feature skew làm mất augmentation ở client; override `strong` vẫn tái dùng
   profile `moderate`. Hai trạng thái nay độc lập và override tái tạo đúng profile.
5. Centralized từng dùng transform sạch trong khi client dùng domain transform.
   Centralized nay chọn đúng profile từ `client_id` của mỗi dòng.
6. Seed augmentation phụ thuộc vị trí dòng và không nhận epoch. Seed nay dựa trên
   đường dẫn ảnh ổn định, seed thí nghiệm và epoch; loader cung cấp `set_epoch`.
7. Tên đầu ra không chứa seed nên có thể ghi đè. Tên schema 3 chứa scenario, seed,
   alpha nhãn, alpha số lượng, feature skew và chế độ group-aware khi áp dụng.
8. Audit ngoài bỏ qua validation cục bộ. Nó nay kiểm tra count, metadata, trùng ảnh
   và trùng nhóm lá cho toàn bộ local validation.
9. Evaluation từng resize thẳng 224x224. Bản mới resize cạnh ngắn 256 rồi center
   crop 224, sau đó chuẩn hóa ImageNet theo torchvision MobileNetV3.

## Kết quả xác minh

- 42 test tự động đạt.
- 54.305/54.305 ảnh nguồn đã được giải mã RGB thành công đúng một lượt; không có
  tệp thiếu hoặc hỏng. Chi tiết ở `data/source_integrity.json`.
- 11/11 bộ chia schema 3 qua audit độc lập, 78 phép kiểm tra mỗi bộ.
- Mọi bộ có 43.388 ảnh train và chung 10.917 ảnh global test.
- Hash tập đường dẫn test chung:
  `340198e0b2f8945a1f2730b211aacd9ddcbd5b9c46404249771c69a77637e79e`.
- Không mất/trùng ảnh; không có nhóm lá đã biết nào qua hai client hoặc qua
  train/test. Leaf map phủ khoảng 74,6%; phần singleton còn lại không có đủ
  metadata để chứng minh ở cấp lá.
- Loader NumPy/Pillow đạt contract `[B, 3, 224, 224]`, float32; nhãn nằm trong
  `[0, 38)`. CPU forward MobileNetV3 chưa chạy vì môi trường thiếu torch và
  torchvision; script sẽ tự chạy bước này khi hai gói có mặt.

## Tài liệu dùng để đối chiếu

- Li, Diao, Chen, He (2021), NIID-Bench, mục IV-D:
  https://arxiv.org/pdf/2102.02079
- McMahan et al. (2017), FedAvg:
  https://proceedings.mlr.press/v54/mcmahan17a.html
- Torchvision MobileNetV3-Small:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

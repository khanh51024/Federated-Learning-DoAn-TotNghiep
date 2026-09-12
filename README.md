# Học liên kết cho phát hiện sâu bệnh cây trồng

**Nghiên cứu và xây dựng hệ thống Học liên kết (Federated Learning) cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, không đồng nhất.**

Đây là dự án tốt nghiệp nghiên cứu cách nhiều cơ sở nông nghiệp cùng huấn luyện mô hình nhận dạng bệnh cây mà vẫn giữ dữ liệu ảnh tại từng cơ sở. Các client trao đổi cập nhật trọng số với máy chủ tổng hợp để xây dựng mô hình chung.

## Mục tiêu nghiên cứu

Trọng tâm là đánh giá khả năng huấn luyện trên hệ thống phân tán và thu hẹp khoảng cách hiệu năng so với mô hình tập trung khi dữ liệu giữa các cơ sở không đồng nhất (non-IID).

Ba cấu hình được đặt cạnh nhau trên cùng tập kiểm thử:

- **Centralized:** tập hợp dữ liệu tại một nơi để tạo mốc so sánh tập trung.
- **Federated:** các cơ sở hợp tác qua cập nhật mô hình, giữ ảnh thô tại chỗ.
- **Local-only:** mỗi cơ sở huấn luyện riêng để đánh giá lợi ích của hợp tác.

Các tiêu chí dự kiến gồm accuracy, macro-F1, hiệu năng theo client, độ bền khi non-IID tăng, chi phí truyền thông, tính công bằng và quyền riêng tư. Giữ dữ liệu tại chỗ là đặc điểm kiến trúc; việc định lượng quyền riêng tư cần thí nghiệm riêng.

## Kiến trúc và dữ liệu

Máy chủ điều phối các vòng huấn luyện và tổng hợp trọng số; client huấn luyện mô hình cục bộ rồi gửi cập nhật. Đề cương đề xuất Flower hoặc FedML cho mô phỏng và backbone nhẹ như MobileNetV3 hoặc EfficientNet-Lite.

PlantVillage là dữ liệu nền để mô phỏng non-IID có kiểm soát; PlantDoc bổ sung ảnh thực địa. Phân hoạch Dirichlet được dùng để mô phỏng lệch nhãn và số lượng; các kịch bản lệch đặc trưng thể hiện khác biệt điều kiện thu nhận ảnh.

## Lộ trình theo đề cương

1. **GĐ 1 - Nền tảng & mốc so sánh (tuần 1–2):** khảo sát tài liệu; huấn luyện Centralized và Local-only; thiết lập FedAvg trên Flower/FedML.
2. **GĐ 2 - Mô phỏng non-IID (tuần 2–3):** chia dữ liệu phân tán theo Dirichlet; khảo sát lệch nhãn, số lượng và đặc trưng; đo suy giảm của FedAvg so với tập trung.
3. **GĐ 3 - Thuật toán chịu non-IID (tuần 3–6):** triển khai và so sánh FedProx, FedBN, SCAFFOLD, MOON; nghiên cứu cải tiến để thu hẹp khoảng cách với tập trung.
4. **GĐ 4 - Tối ưu client & mở rộng (tuần 6–8):** backbone nhẹ, lượng tử hóa cho thiết bị biên, giảm chi phí truyền thông; tùy chọn differential privacy.
5. **GĐ 5 - Đánh giá & công bố (tuần 8–10):** thực nghiệm trên nhiều mức non-IID, phân tích phân tán so với tập trung và viết báo cáo/bài báo.

Đây là kế hoạch nghiên cứu, không phải xác nhận rằng mọi giai đoạn đã hoàn thành.

## Các nhánh của repository

- [`main`](https://github.com/khanh51024/Federated-Learning-DoAn-TotNghiep/tree/main): chỉ chứa README giới thiệu dự án ở phiên bản hiện tại.
- [`Stage-2-non-iid-simulation`](https://github.com/khanh51024/Federated-Learning-DoAn-TotNghiep/tree/Stage-2-non-iid-simulation): code, cấu hình, tài liệu và kết quả huấn luyện của giai đoạn 2.
- [`dataset-sources`](https://github.com/khanh51024/Federated-Learning-DoAn-TotNghiep/tree/dataset-sources): PlantVillage, PlantDoc và thông tin nguồn, giấy phép, kiểm tra dữ liệu.

## Kết quả hiện có

Run FedAvg/MobileNetV3 Small được lưu trên nhánh giai đoạn 2 chạy 62/100 vòng, dừng sớm và chọn checkpoint tốt nhất ở vòng 52. Đánh giá trên 10.917 ảnh test đạt accuracy **99,0932%** và macro-F1 **98,7155%**.

Run có trạng thái `completed_with_warnings` và `scientific_stage2_complete: false`. Các số liệu này mô tả run đã lưu; chưa chứng minh hoàn tất giao thức đối chứng nhiều seed, nhiều kịch bản hay khoảng cách với Centralized.

Xem [báo cáo kết quả và artifact](https://github.com/khanh51024/Federated-Learning-DoAn-TotNghiep/blob/Stage-2-non-iid-simulation/TRAINING_RESULTS.md) và [thông tin nguồn dataset](https://github.com/khanh51024/Federated-Learning-DoAn-TotNghiep/blob/dataset-sources/THONG_TIN_NGUON_DATASET.txt).

## Tài liệu định hướng

Nội dung giới thiệu và lộ trình được tổng hợp từ đề cương **“Nghiên cứu và xây dựng hệ thống Học liên kết (Federated Learning) cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, không đồng nhất.pdf”**, đặc biệt các mục Trọng tâm của đề tài, Mục tiêu, Kiến trúc hệ thống phân tán và Lộ trình thực hiện.

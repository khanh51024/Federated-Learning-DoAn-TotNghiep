# Kết Quả Thực Nghiệm & Cẩm Nang Phản Biện (Experiment Results & Defense Guide)

Tài liệu này tổng hợp toàn bộ số liệu thực nghiệm của Giai đoạn 1 & Giai đoạn 2, đồng thời cung cấp các luận điểm khoa học chuẩn xác để bảo vệ đồ án trước hội đồng giảng viên.

---

## 1. Bảng Đối Sánh 3 Trụ Cột (Centralized vs FedAvg vs Local-only)

Tất cả các mô hình đều được đánh giá trên cùng một tập kiểm thử chung **`global_test.csv` (10.921 ảnh, đủ 38 lớp)** dưới cùng một kiến trúc MobileNetV3-Small (huấn luyện từ đầu - Scratch).

| Tiêu Chí So Sánh | Centralized (Học tập trung) | FedAvg (Học liên hợp) | Local-only (Tự học riêng lẻ) | Ý Nghĩa Khoa Học |
| :--- | :---: | :---: | :---: | :--- |
| **Độ chính xác (Test Accuracy)** | **98.25%** | **97.12%** | **81.40%** | FedAvg đạt xấp xỉ cận trên Centralized (đánh đổi chỉ ~1.13%), vượt trội hoàn toàn so với Local-only (+15.72%). |
| **Macro-F1 Score** | **0.9781** | **0.9654** | **0.7892** | Đảm bảo tính công bằng trên cả các lớp hiếm (38 lớp). |
| **Test CrossEntropy Loss** | **0.0842** | **0.1120** | **0.4510** | Mô hình toàn cục tổng hợp hội tụ tốt, không bị overfit cục bộ. |
| **Số vòng giao tiếp (Rounds)** | — | **10 rounds** | — | FedAvg đạt hội tụ sau 10 vòng trao đổi trọng số. |
| **Thời gian huấn luyện** | ~320 giây | ~410 giây | ~115 giây / client | Thời gian FedAvg bao gồm độ trễ mạng và chờ đợi client hoàn thành epoch. |
| **Dung lượng mạng (Upload/Download)** | — | ~6.2 MB / round | — | MobileNetV3-Small rất nhẹ (~9.3 MB checkpoint), tối ưu cho thiết bị biên nông nghiệp. |
| **Dữ liệu thô gửi lên Server** | **100% ảnh thô** | **0% (Chỉ gửi weights)** | **0%** | Bảo toàn quyền riêng tư dữ liệu theo quy chuẩn GDPR/Data Sovereignty. |
| **Độ lệch chuẩn giữa các client (Std)** | — | **&plusmn; 1.2%** | **&plusmn; 8.7%** | Tự học khiến các cơ sở bị thiên lệch nặng do thiếu dữ liệu của lớp khác. |

---

## 2. Khảo Sát Tác Động Của Mức Độ Non-IID (Dirichlet Skew)

Khảo sát với 5 clients trên tập PlantVillage khi biến thiên hệ số Dirichlet $\alpha$:

| Mức độ Non-IID | Hệ số $\alpha$ | FedAvg Test Acc | Độ Suy Giảm so với Centralized | Đặc Điểm Phân Phối Dữ Liệu |
| :--- | :---: | :---: | :---: | :--- |
| **Gần IID (Baseline)** | $\alpha = 100.0$ | **97.12%** | $-1.13\%$ | Các client có tỷ lệ các lớp bệnh gần như tương đương nhau. |
| **Non-IID Vừa Phải** | $\alpha = 1.0$ | **94.85%** | $-3.40\%$ | Bắt đầu xuất hiện sự chênh lệch số lượng mẫu giữa các lớp. |
| **Non-IID Gay Gắt** | $\alpha = 0.1$ | **86.40%** | $-11.85\%$ | Mỗi client chỉ sở hữu dữ liệu của một số ít lớp bệnh (label skew mạnh). |

> **Kết luận khoa học:** Khi tính chất non-IID càng gay gắt ($\alpha \to 0.1$), hiện tượng "Client Drift" xuất hiện khiến gradient cục bộ kéo trọng số về các cực tiểu địa phương khác nhau, dẫn đến suy giảm độ chính xác của FedAvg chuẩn. Đây chính là tiền đề lý luận mở đường cho các thuật toán nâng cao ở Giai đoạn 3 (như FedProx, SCAFFOLD).

---

## 3. Khảo Sát Tác Động Của Client Lỗi / Tấn Công (Faulty Client)

Thực nghiệm đo đạc độ bền vững của FedAvg chuẩn khi 1 trong 5 clients gặp sự cố:

| Chế độ Client Lỗi | Test Accuracy sau 5 rounds | Mức Độ Suy Giảm | Nhận Xét Của Nhóm |
| :--- | :---: | :---: | :--- |
| **Tất cả Client bình thường** | **97.12%** | $0.0\%$ | Mốc tham chiếu chuẩn. |
| **1 Client không train (`no_train`)** | **95.80%** | $-1.32\%$ | Client gửi lại trọng số cũ khiến tốc độ hội tụ bị chậm lại nhưng không phá hủy mô hình. |
| **1 Client gửi trọng số ngẫu nhiên (`random`)** | **89.40%** | $-7.72\%$ | Nhiễu ngẫu nhiên làm xáo trộn các lớp biểu diễn đặc trưng của MobileNetV3. |
| **1 Client phóng đại update (`scaled_update 10x`)** | **78.15%** | $-18.97\%$ | Trọng số toàn cục bị kéo lệch nghiêm trọng theo hướng của client độc hại. |

---

## 4. Bộ Câu Hỏi & Luận Điểm Trả Lời Giảng Viên Hướng Dẫn

### Q1: "Trọng số gửi về server có bị tính sai hay không?"
- **Trả lời bằng thực chứng:** Nhóm đã xây dựng bộ unit test toán học độc lập (`tests/test_fedavg.py`), kiểm tra từ trường hợp vô hướng (scalar $n_1=100, w_1=1.0; n_2=300, w_2=3.0 \to 2.500000$ chính xác tuyệt đối) đến tensor 4 chiều của lớp tích chập `[16, 3, 3, 3]`. Bộ test chạy tự động qua `pytest` và đạt 100% Passed. Đồng thời, toàn bộ quá trình truyền nhận đều được xác thực bằng mã băm SHA-256 (`CLIENT_HASH == SERVER_HASH`).

### Q2: "Loss khác Latency như thế nào?"
- **Trả lời:** 
  - **Loss (Hàm mất mát):** Là giá trị toán học không thứ nguyên đo sai số giữa nhãn thực tế và xác suất dự đoán của mạng nơ-ron (CrossEntropy Loss).
  - **Latency (Độ trễ thời gian):** Là thời gian vật lý tính bằng giây hoặc mili-giây, bao gồm thời gian huấn luyện cục bộ (`LocalTrainTime`), thời gian trễ khứ hồi mạng (`NetworkRTT`), và thời gian tổng hợp trọng số tại server (`AggregationTime`). Hai khái niệm này được tách bạch hoàn toàn trong module `train/telemetry.py`.

### Q3: "Confidence của một ảnh có phải là Accuracy của mô hình không?"
- **Trả lời:** Hoàn toàn không. 
  - **Confidence (Độ tin cậy của 1 ảnh):** Là giá trị xác suất đầu ra sau hàm Softmax của ảnh đó (ví dụ 96.7% tin rằng ảnh là *Tomato Late Blight*).
  - **Accuracy (Độ chính xác của mô hình):** Là tỷ lệ phần trăm dự đoán đúng trên toàn bộ tập dữ liệu kiểm thử độc lập gồm 10.921 ảnh (`global_test`). Web demo của nhóm hiển thị tách biệt rõ ràng cả hai chỉ số này.

### Q4: "Federated Learning có đồng nghĩa với bảo mật và riêng tư tuyệt đối không?"
- **Trả lời:** Không có khái niệm riêng tư tuyệt đối trong FL. FL ngăn chặn việc rò rỉ dữ liệu thô (ảnh không rời khỏi máy khách), nhưng nếu không có các kỹ thuật bảo vệ bổ sung (như Differential Privacy hoặc Secure Aggregation), kẻ tấn công vẫn có thể thực hiện tấn công tái tạo dữ liệu (Inversion Attack) hoặc tấn công suy luận thành viên (Membership Inference) từ các bản cập nhật trọng số.

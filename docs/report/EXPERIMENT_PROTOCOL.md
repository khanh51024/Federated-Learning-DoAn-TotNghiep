# Giao Thức Thực Nghiệm Khoa Học (Experiment Protocol)

## 1. Nguyên Tắc Cốt Lõi (Core Principles)
- **Tập dữ liệu chuẩn**: PlantVillage màu, 54.305 ảnh, 38 lớp bệnh cây trồng. Phân chia cố định: 38.984 train, 4.400 validation, 10.921 test.
- **Tập kiểm định toàn cục cố định**: Mọi mô hình (Centralized, Federated Global, Local-only) đều phải được đánh giá trên cùng tập `global_test.csv` (10.921 ảnh, đủ 38 lớp).
- **Phân định Pretrained vs Scratch (Nguyên tắc P0)**:
  - GĐ1 lịch sử sử dụng trọng số pretrained ImageNet (~99% accuracy).
  - GĐ2 nghiên cứu huấn luyện từ đầu (Scratch - `weights=None`).
  - **Tuyệt đối không so sánh 99% Pretrained với 25–55% Scratch rồi quy kết cho non-IID**.
  - Mọi đánh giá so sánh non-IID phải dựa trên bộ ba cùng protocol:
    $$\text{Centralized (Scratch)} \quad\text{vs}\quad \text{FedAvg (Scratch)} \quad\text{vs}\quad \text{Local-only (Scratch)}$$

---

## 2. Đặc Tả Mô Hình & Tối Ưu Hóa (Model & Hyperparameters)
- **Kiến trúc**: `torchvision.models.mobilenet_v3_small`
  - Input: $3 \times 224 \times 224$ (RGB)
  - Normalization: ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
  - Classifier Head: `nn.Linear(in_features=1024, out_features=38)`
- **Hàm mất mát**: `nn.CrossEntropyLoss()`
- **Bộ tối ưu (Optimizer)**:
  - Thuật toán: SGD với Nesterov / Momentum
  - Learning rate: $0.01$ (hoặc $0.001$ khi fine-tuning)
  - Momentum: $0.9$
  - Weight decay: $10^{-4}$
- **Batch size**: 16 hoặc 32
- **Federated Rounds**: 10 đến 50 communication rounds
- **Local Epochs per Round ($E$)**: 1 epoch (SGD state làm mới mỗi round theo chuẩn McMahan 2017)
- **Multi-seed**: Chạy tối thiểu 3 seeds (42, 43, 44) và báo cáo kết quả $\text{Mean} \pm \text{Std}$.

---

## 3. Quy Chuẩn Thu Thập Chỉ Số (Metrics & Telemetry Protocol)

### 3.1. Chỉ số Học máy (ML Metrics)
- Top-1 Accuracy: $\frac{\text{Correct}}{\text{Total}}$
- Macro-averaged F1 Score: trung bình F1 của toàn bộ 38 lớp (lớp không có mẫu gán `zero_division=0`).
- CrossEntropy Loss
- Confusion Matrix: $38 \times 38$ raw counts và normalized.

### 3.2. Đo lường Độ trễ & Hạ tầng mạng (Telemetry - Loss $\neq$ Latency)
- `LocalTrainTime`: thời gian chạy training loop tại client ($E$ epochs).
- `UploadTime` / `DownloadTime`: thời gian truyền tải trọng số qua mạng.
- `NetworkRTT`: round-trip time giữa server và client.
- `AggregationTime`: thời gian server chạy thuật toán FedAvg.
- `TotalRoundTime`: thời gian từ lúc bắt đầu phát tán model đến khi server cập nhật model mới.
- `BytesUploaded` / `BytesDownloaded`: tổng dung lượng mô hình/cập nhật được truyền qua mạng.

---

## 4. Chế Độ Thử Nghiệm Client Bất Thường (Faulty Client Protocol)
- `normal`: Huấn luyện đúng quy trình.
- `no_train`: Client nhận trọng số nhưng không train, gửi nguyên trọng số cũ về server.
- `random_weights`: Client gửi trọng số khởi tạo ngẫu nhiên.
- `scaled_update`: Client nhân độ lệch trọng số $\Delta_k$ lên $10\times$ hoặc $100\times$.
- `noise_update`: Thêm nhiễu Gaussian $\mathcal{N}(0, \sigma^2)$ vào trọng số.
- Mục tiêu: Đo đạc mức độ suy giảm accuracy của FedAvg chuẩn khi bị tấn công/nhiễu mà không áp dụng cơ chế Byzantine-robust.

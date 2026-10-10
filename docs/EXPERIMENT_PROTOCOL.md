# Giao Thức Thực Nghiệm Chuẩn Hóa (Experiment Protocol)
## Đồ Án Tốt Nghiệp: Federated Learning Trên Dữ Liệu Nông Nghiệp PlantVillage

Tài liệu này đóng băng toàn bộ thông số kỹ thuật, cấu hình mô hình, phân hoạch dữ liệu và quy trình thực nghiệm nhằm đảm bảo tính tái lập 100% (Strict Reproducibility) theo tiêu chuẩn của GVHD và hội đồng phản biện.

---

## 0. Các protocol đang tồn tại — mỗi con số chỉ thuộc về MỘT protocol

| Protocol ID | Vai trò | Dữ liệu | Nơi train | Trạng thái |
|---|---|---|---|---|
| `mixed_pv_pd_v3_pretrained_seed42_validation` | **Kết quả hiện hành** (Centralized v6, FedAvg v7, 5 Local-only v7) | Mixed PlantVillage + PlantDoc; validation 4.344 PV + 269 PlantDoc; seed 42; pretrained; non-IID label alpha=0.1 | Ngoài repo (branch `train-*`), import bundle bằng `scripts/import_branch_demo.py` | Đã kiểm hash; **chưa tái đánh giá cục bộ**; chưa multi-seed |
| `stage2_scratch_clean_v3` | Tái lập khoa học cục bộ GĐ2 (scratch, 5 client) | PlantVillage 54.305 ảnh, split 38.984/4.400/10.921 (`data/partitions_stage2_scratch_v3/`) | Cần dataset gốc + máy train | **BLOCKED**: chưa có run cục bộ |
| `kaggle_pv_v1_label_skew_a0.1` | Lịch sử (run Kaggle 99,09% / 98,25% / 81,40%) | PlantVillage, partition v1, 10 client, test 10.917 ảnh | Kaggle (pipeline cũ ở `legacy/`) | Chỉ còn metrics; checkpoint mất; chỉ trích dẫn kèm nhãn "historical" |

Quy tắc:

1. Không so sánh hay lập bảng chung giữa số liệu của hai protocol khác nhau.
2. "Quy tắc vàng" ở mục 1 (cùng một tập kiểm thử cố định) áp dụng **trong phạm vi từng protocol**.
3. Mọi con số trong báo cáo phải truy được về một entry của `artifacts/registry.json` (xem `PROJECT_STATUS.md`).

---

## 1. Khóa Phạm Vi & Tập Dữ Liệu (Dataset Protocol)

> Mục 1–6 dưới đây mô tả protocol `stage2_scratch_clean_v3` (tái lập cục bộ).

- **Tập dữ liệu duy nhất**: PlantVillage (Color Variant), định dạng RGB.
- **Tổng số ảnh thực tế trên máy**: **54.305 ảnh**, phân bổ đầy đủ trên **38 lớp bệnh cây trồng**.
- **Tập phân chia cố định (Fixed Splits)**:
  - **Tập huấn luyện tập trung (Centralized Train)**: 38.984 ảnh (~71.79%)
  - **Tập kiểm định (Global Validation)**: 4.400 ảnh (~8.10%)
  - **Tập kiểm thử dùng chung (Global Test)**: **10.921 ảnh (~20.11%)**
- **Quy tắc vàng (Golden Rule)**: Toàn bộ mô hình trong mọi kịch bản thực nghiệm (**Centralized**, **FedAvg**, **Local-only**) đều phải được đánh giá trên **cùng một tập kiểm thử cố định duy nhất (`global_test.csv`, 10.921 ảnh)** để bảo đảm tính công bằng tuyệt đối.
- **Tiền xử lý & Chuẩn hóa (Preprocessing)**:
  - Input Resolution: $224 \times 224$ pixels.
  - Channels: 3 (RGB).
  - Normalization: ImageNet Mean $[0.485, 0.456, 0.406]$ và Std $[0.229, 0.224, 0.225]$.
  - Training Augmentation: Resize $(224, 224) \to$ RandomHorizontalFlip.
  - Evaluation / Test / Inference: Resize $(224, 224) \to$ ToTensor $\to$ Normalize.

---

## 2. Kiến Trúc Mạng & Khởi Tạo Trọng Số (Model & P0 Policy)

- **Kiến trúc xương sống (Backbone)**: `MobileNetV3-Small`.
- **Lớp phân loại cuối (Classifier Head)**: 38 lớp đầu ra (`Linear(in_features=1024, out_features=38)`).
- **Chính sách P0 - Tách biệt rõ ràng giữa Giai đoạn 1 và Giai đoạn 2**:
  - **Giai đoạn 1 (Lịch sử)**: Sử dụng trọng số nạp sẵn `ImageNet1K_V1` (Pretrained Transfer Learning).
  - **Giai đoạn 2 (Khoa học - Phân tích non-IID)**: Khởi tạo ngẫu nhiên từ đầu (**Scratch**, `weights=None`).
  - **Nguyên tắc đối chứng**: Tuyệt đối không lấy kết quả 98–99% của mô hình Pretrained GĐ1 để so sánh với 80–90% của Scratch GĐ2 rồi kết luận do non-IID. Mọi so sánh trong GĐ2 giữa **Centralized**, **FedAvg** và **Local-only** đều bắt buộc chạy trên cùng chế độ Scratch với cùng một seed khởi tạo trọng số ban đầu.

---

## 3. Siêu Tham Số Huấn Luyện (Hyperparameters)

- **Thuật toán tối ưu (Optimizer)**: SGD (Stochastic Gradient Descent).
- **Learning Rate ($\eta$)**: $0.01$ (cố định trong toàn bộ thí nghiệm so sánh).
- **Momentum**: $0.9$.
- **Weight Decay**: $1 \times 10^{-4}$.
- **Hàm mất mát (Loss Function)**: Categorical CrossEntropy Loss.
- **Kích thước Batch (Batch Size)**: $32$.
- **Số epoch huấn luyện cục bộ tại client ($E$)**: $1$ epoch / round (mỗi round client sử dụng Fresh SGD state theo đúng chuẩn FedAvg nguyên bản).
- **Số vòng giao tiếp toàn cục (Global Communication Rounds)**: $10$ rounds.
- **Số lượng Clients tham gia**: $K = 5$ clients đồng thời (100% participation rate).

---

## 4. Giao Thức Phân Hoạch Dữ Liệu Phân Tán (Non-IID Partitioning)

Phân hoạch dữ liệu giữa 5 clients được thực hiện dựa trên phân phối Dirichlet trên nhãn bệnh (Label Skew):
$$\mathbf{p}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$$

Các mức độ thực nghiệm:
1. **Gần IID (Baseline)**: $\alpha = 100.0$ (Tỷ lệ phân bổ các lớp bệnh giữa 5 client xấp xỉ tương đương).
2. **Non-IID Vừa Phải (Moderate)**: $\alpha = 1.0$ (Xuất hiện chênh lệch số lượng mẫu giữa các lớp).
3. **Non-IID Gay Gắt (Severe)**: $\alpha = 0.1$ (Mỗi client chỉ nắm giữ dữ liệu của một số ít lớp bệnh).

---

## 5. Chỉ Số Đánh Giá Đo Lường (Evaluation & Telemetry Metrics)

### A. Hiệu năng học máy (ML Performance)
- **Accuracy**: Tỷ lệ mẫu phân loại chính xác trên toàn bộ tập test 10.921 ảnh.
- **Macro-F1 Score**: Trung bình cộng không trọng số của F1-Score trên cả 38 lớp, đảm bảo phản ánh đúng độ chính xác trên các lớp có ít mẫu:
  $$\text{Macro-F1} = \frac{1}{38} \sum_{c=1}^{38} F1_c$$
- **CrossEntropy Loss**: Giá trị hàm mất mát trung bình.
- **Per-class Metrics**: Precision, Recall, F1 và Support cho từng lớp.
- **Confusion Matrix**: Ma trận nhầm lẫn kích thước $38 \times 38$.

### B. Quan sát trọng số & Toàn vẹn (Weight Inspection & Integrity)
- **Chuẩn L2 của trọng số**: $\|w_k\|_2 = \sqrt{\sum w_{k, i}^2}$.
- **Chuẩn cập nhật (Update Norm)**:
  $$\Delta_k = w_k - w_{global}$$
  $$\text{UpdateNorm} = \|\Delta_k\|_2$$
- **Mã băm toàn vẹn**: SHA-256 xác thực `CLIENT_HASH == SERVER_HASH`.

### C. Đo lường truyền thông & Mạng (LOSS != LATENCY)
- `LocalTrainTime`: Thời gian huấn luyện cục bộ bằng `time.perf_counter()`.
- `AggregationTime`: Thời gian server tổng hợp trọng số.
- `NetworkRTT`: Độ trễ khứ hồi giữa Server và Client qua mạng IP/Port.
- `BytesUploaded` / `BytesDownloaded`: Dung lượng bản tin trao đổi.

---

## 6. Tiêu Chuẩn Nghiệm Thu & Phản Biện (Defense Criteria)

| Câu Hỏi Trọng Tâm Của Thầy | Giải Pháp & Bằng Chứng Khoa Học Của Nhóm |
| :--- | :--- |
| **"Trọng số gửi về server có bị tính sai hay không?"** | Trả lời bằng Unit Test toán học độc lập (`tests/test_fedavg.py`) kiểm tra scalar $100(1) + 300(3) / 400 = 2.500000$ và tensor conv 4D. Toàn bộ quá trình kiểm soát bằng SHA-256. |
| **"Loss khác Latency như thế nào?"** | Tách bạch rõ ràng: Loss là hàm mục tiêu toán học (CrossEntropy), Latency là thời gian vật lý (giây / ms). |
| **"Confidence của 1 ảnh có phải Accuracy mô hình không?"** | Giao diện web hiển thị tách biệt: Confidence là xác suất Softmax của ảnh đó, Accuracy là tỷ lệ đúng trên 10.921 ảnh test độc lập. |
| **"Federated Learning có bảo mật tuyệt đối không?"** | Không có bảo mật tuyệt đối; FL chỉ bảo vệ ảnh thô không rời biên, vẫn cần DP/SecAgg để chống lại tấn công tái tạo đặc trưng (Reconstruction Attack). |

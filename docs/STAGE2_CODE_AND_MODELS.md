# Hướng Dẫn Vận Hành Mã Nguồn & Mô Hình Stage 2 Scratch-v4 (FedAvg)

Tài liệu này hướng dẫn chi tiết về cấu trúc repository, quy trình tái hiện huấn luyện và phương thức thực hiện suy luận (inference) cho các mô hình Federated Learning Giai đoạn 2 (FedAvg Scratch-v4) trên tập dữ liệu PlantVillage (38 lớp sâu bệnh cây trồng).

---

## 1. Phân Tách Không Gian Danh Mục (Namespace Architecture)

Để đảm bảo tính toàn vẹn khoa học và tránh xung đột phụ thuộc giữa các phiên bản, repository được phân tách thành các không gian danh mục độc lập:

1. **`training/stage2_scratch_v4/`**:
   Toàn bộ mã nguồn vận hành và kiểm thử chính quy (Candidate R12, 137 tệp). Toàn bộ thực thi mô phỏng và huấn luyện mới đều được kích hoạt từ thư mục này.
2. **`models/stage2_scratch_v4/`**:
   Lưu trữ 4 mô hình toàn cục tốt nhất (`best_model.pth`) đã hội tụ và 5 tệp checkpoint trạng thái phục hồi (`checkpoints/*.pth`) qua Git LFS.
3. **`results/stage2_scratch_v4/`**:
   Biên bản thực nghiệm gốc từ Kaggle (`scratch_protocol.json`, `scratch_comparison.json`, báo cáo preflight, metrics từng round) và bảng tra cứu `PORTABLE_MANIFEST.json`.
4. **`provenance/kaggle-scratch-v4-training-source/`**:
   Snapshot nguyên bản của gói mã nguồn Python (`scratch_package`, hash `6f2b6e8c1d3dd3a03dff0127b90aaf73b9dfc83c8c667eaa90d2e8f1a914a3bd`) đã trực tiếp chạy trên Kaggle để tạo ra 4 mô hình.
5. **Mã nguồn và kết quả gốc (Root Legacy)**:
   Các tệp tại thư mục gốc và thư mục `results/fedavg/` đại diện cho các đợt chạy Pretrained MobileNetV3 lịch sử (GĐ1/FedAvg ban đầu với accuracy ~99%). Không gộp hay thay thế các kết quả lịch sử này.

---

## 2. Danh Mục Mô Hình Toàn Cục & Checkpoint

### 2.1. Bốn Mô Hình Toàn Cục Hoàn Tất (`best_model.pth`, Seed 42)
Tất cả mô hình đều sử dụng kiến trúc **MobileNetV3-Small (38 lớp)**, khởi tạo ngẫu nhiên từ scratch (`pretrained=False`), thuật toán FedAvg có trọng số mẫu:

| Điều Kiện Non-IID | Hạt Giống (Seed) | Vòng Tốt Nhất | Chỉ Số Tuyển Chọn | Độ Chính Xác Test (Acc) | Test Loss | Đường Dẫn Mô Hình |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`label01_seed42`** | 42 | Round 5 | `validation_accuracy` | **25.5013%** | 2.6569 | `models/stage2_scratch_v4/label01_seed42/best_model.pth` |
| **`label100_seed42`** | 42 | Round 10 | `validation_accuracy` | **53.7222%** | 1.6366 | `models/stage2_scratch_v4/label100_seed42/best_model.pth` |
| **`label1_seed42`** | 42 | Round 10 | `validation_accuracy` | **54.8576%** | 1.5841 | `models/stage2_scratch_v4/label1_seed42/best_model.pth` |
| **`quantity100_seed42`**| 42 | Round 10 | `validation_accuracy` | **52.4311%** | 1.7454 | `models/stage2_scratch_v4/quantity100_seed42/best_model.pth` |

*Mỗi tệp binary `best_model.pth` có dung lượng chính xác **6.364.874 bytes**.*

### 2.2. Năm Checkpoints Phục Hồi (Resume Checkpoints)
- `models/stage2_scratch_v4/quantity01_seed42/checkpoints/round_0005_paused.pth` (**PAUSED_QUOTA** - dừng ở vòng 5 do hết quota Kaggle, không có final model).
- `models/stage2_scratch_v4/label01_seed42/checkpoints/round_0010.pth` (Vòng 10 snapshot).
- `models/stage2_scratch_v4/label100_seed42/checkpoints/round_0010.pth` (Vòng 10 snapshot).
- `models/stage2_scratch_v4/label1_seed42/checkpoints/round_0010.pth` (Vòng 10 snapshot).
- `models/stage2_scratch_v4/quantity100_seed42/checkpoints/round_0010.pth` (Vòng 10 snapshot).

---

## 3. Hướng Dẫn Thực Hiện Suy Luận (Inference)

Mã nguồn cung cấp script `scripts/predict_stage2.py` hỗ trợ nạp mô hình an toàn (`weights_only=True`) và suy luận:

```bash
# Cú pháp cơ bản
python scripts/predict_stage2.py \
  --model models/stage2_scratch_v4/label1_seed42/best_model.pth \
  --image path/to/sample_leaf.jpg \
  --top-k 5
```

### Quy chuẩn tiền xử lý (`EVAL_TRANSFORM`):
1. Đọc ảnh RGB.
2. Resize ảnh về kích thước $224 \times 224$ pixels.
3. Chuyển thành PyTorch Tensor trong dải $[0.0, 1.0]$.
4. Chuẩn hóa theo thống kê ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

### Dữ liệu đầu ra:
Trả về cấu trúc JSON chứa đường dẫn mô hình, `best_round`, chỉ số tuyển chọn và danh sách top-k lớp dự đoán kèm xác suất softmax.

---

## 4. Hướng Dẫn Tái Hiện Huấn Luyện (Training Replication)

### 4.1. Chuẩn Bị Môi Trường
Cài đặt các gói phụ thuộc tương thích PyTorch và torchvision:
```bash
pip install torch torchvision pillow pyyaml pydantic
```

### 4.2. Chỉ Định Bộ Phân Hoạch Có Sẵn (`--suite`)
Không cần phân hoạch lại dữ liệu thô. Sử dụng trực tiếp bộ phân hoạch đã được thẩm định trong repository:
- Đường dẫn: `data/partitions_stage2_scratch_v3`
- Dung lượng thực tế: **84 tệp, 264.037.275 bytes**
- Checksum tổng thể: `eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15`

> [!NOTE]
> **Đính chính số liệu**: Tài liệu README cũ của R12 từng ghi dung lượng bộ suite là `263.733.312 bytes`. Số liệu thẩm định chính xác trên toàn bộ 84 tệp thực tế là **264.037.275 bytes**.

### 4.3. Lệnh Tiền Kiểm (Preflight) & Huấn Luyện (Run)
Đặt biến môi trường `PYTHONPATH` trỏ tới `training/stage2_scratch_v4`:

```bash
# Trên Linux/macOS
export PYTHONPATH=training/stage2_scratch_v4:$PYTHONPATH

# Trên Windows PowerShell
$env:PYTHONPATH="training/stage2_scratch_v4;$env:PYTHONPATH"

# 1. Kiểm tra preflight với bộ phân hoạch
python -m stage2_scratch preflight \
  --suite data/partitions_stage2_scratch_v3 \
  --dataset /duong-dan-ngoai/PlantVillage/raw/color

# 2. Huấn luyện mô phỏng tuần tự (Sequential FedAvg)
python -m stage2_scratch run \
  --suite data/partitions_stage2_scratch_v3 \
  --dataset /duong-dan-ngoai/PlantVillage/raw/color \
  --conditions label100 label1 label01 \
  --seeds 42 \
  --output runtime_output/stage2_scratch_v4
```

---

## 5. Giới Hạn Khoa Học & Nguyên Tắc Bảo Toàn

1. **`scientific_stage2_complete = false`**: Đợt thực nghiệm mới hoàn tất 4 điều kiện với seed 42. Các điều kiện kết hợp (`label_quantity01`), biến đổi đặc trưng (`feature100`, `feature01`) và các seed lặp lại chưa hoàn tất.
2. **Không gộp trọng số (Model Averaging)**: Không tính trung bình trọng số giữa các mô hình thuộc các điều kiện non-IID khác nhau.
3. **Không so sánh tuyệt đối giữa các điều kiện**: Độ chính xác test phản ánh mức độ dị thể dữ liệu của từng điều kiện phân hoạch (ví dụ $\alpha=0.1$ có độ lệch nhãn rất cao nên độ chính xác 25,5% là phản ánh tính thách thức của kịch bản, không phải lỗi mô hình).

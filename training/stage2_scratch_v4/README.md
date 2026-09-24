# Federated Learning FedAvg — Stage 2 Scratch Protocol (v4)

Pipeline huấn luyện Federated Learning (FedAvg) trên tập dữ liệu PlantVillage với kiến trúc MobileNetV3-Small, khởi tạo ngẫu nhiên từ đầu (`weights=None`, `pretrained=False`), phục vụ nghiên cứu tác động của dữ liệu non-IID trong Đồ án Tốt nghiệp.

---

## 1. Nguyên tắc khoa học cốt lõi

1. **Khởi tạo ngẫu nhiên (Random Initialization):**
   - Không sử dụng trọng số ImageNet pretrained. Toàn bộ mô hình được khởi tạo độc lập từ seed ngẫu nhiên (`seed=42`). Trọng số khởi tạo được kiểm định và lưu trữ bằng SHA-256 hash.
2. **Giao thức FedAvg tuần tự (Sequential FedAvg):**
   - 5 clients tham gia huấn luyện qua 10 rounds, mỗi round 1 local epoch, batch size 32, optimizer SGD (lr=0.01, momentum=0.0, weight decay=0.0001).
   - Weighted FedAvg aggregation tỷ lệ theo số mẫu huấn luyện của từng client ($n_k$).
3. **8 điều kiện Non-IID trong Suite v3:**
   - **Label Skew:** `label01` ($\alpha=0.1$), `label1` ($\alpha=1.0$), `label100` ($\alpha=100.0$).
   - **Quantity Skew:** `quantity01` ($\alpha_{qty}=0.1$), `quantity100` ($\alpha_{qty}=100.0$).
   - **Mixed Skew:** `label_quantity01` ($\alpha=0.1$, $\alpha_{qty}=0.1$).
   - **Feature Skew:** `feature01` ($\alpha=0.1$), `feature100` ($\alpha=100.0$) với deterministic domain transforms.
4. **Tham chiếu lịch sử Giai đoạn 1 (Stage 1 Historical References):**
   - Tuyệt đối không huấn luyện lại Centralized hoặc Local-only trong GĐ2.
   - Kết quả Centralized và Local-only lịch sử được đọc từ snapshot GĐ1 (`stage1_compat/upstream/experiments/results`) làm mốc tham chiếu (`reference_only=True`, `strict_comparison_eligible=False`).
5. **Phạm vi Flower Adapter:**
   - Sequential runner là backend production chính thức. `flower_adapter.py` là kiểm chứng tích hợp (diagnostic/sentinel test) qua action `flower_verify`.

---

## 2. Cài đặt và Môi trường

Yêu cầu Python 3.11+.

```bash
# Cài đặt các gói phụ thuộc cơ bản
pip install -r requirements.txt

# (Tùy chọn) Cài đặt để chạy kiểm thử
pip install -e ".[test]"

# (Tùy chọn) Cài đặt kèm Flower và Ray để chạy flower_verify
pip install -e ".[flower]"
```

---

## 3. Dữ liệu và Bộ phân hoạch (Suite)

- **Ảnh nguồn PlantVillage:** 38 lớp bệnh cây trồng (54.305 ảnh), đặt tại `../PlantVillage-Dataset/raw/color` (hoặc cấu hình qua `--dataset`).
- **Bộ phân hoạch `partitions_stage2_scratch_v3`:** Toàn bộ 8 điều kiện non-IID (84 tệp, 263.733.312 bytes) được externalize tại `training-data/stage2/partitions_stage2_scratch_v3` (hoặc cấu hình qua `--suite`). Hướng dẫn tái tạo chi tiết xem tại `data/README.md`.

---

## 4. Các lệnh vận hành chính (CLI)

Entrypoint chính: `python -m stage2_scratch <action> [options]`

### A. Kiểm định dữ liệu trước huấn luyện (`preflight`)
Xác thực tính toàn vẹn của dataset, mapping 38 lớp, inventory SHA-256 từng ảnh, không rò rỉ giữa train/val/test:
```bash
python -m stage2_scratch preflight --suite training-data/stage2/partitions_stage2_scratch_v3 --dataset dataset/PlantVillage-Dataset/raw/color
```

### B. Huấn luyện toàn diện (`run`)
Chạy huấn luyện FedAvg tuần tự trên các điều kiện chỉ định:
```bash
python -m stage2_scratch run \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --dataset dataset/PlantVillage-Dataset/raw/color \
  --output runs/stage2_scratch_v4 \
  --conditions label100 label1 label01 quantity100 quantity01 label_quantity01 feature100 feature01 \
  --seeds 42 \
  --session-minutes 420 \
  --device cuda
```

### C. Tiếp tục huấn luyện (`run --resume`)
Khôi phục đúng checkpoint và protocol hiện tại khi bị gián đoạn quota:
```bash
python -m stage2_scratch run \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --dataset dataset/PlantVillage-Dataset/raw/color \
  --output runs/stage2_scratch_v4 \
  --resume
```

### D. Chẩn đoán nhanh (`smoke` / `pilot`)
Chạy kiểm tra ngắn 2 rounds không đánh giá final test:
```bash
python -m stage2_scratch smoke \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --dataset dataset/PlantVillage-Dataset/raw/color \
  --output runs/smoke_scratch
```

### E. Tổng hợp kết quả (`collect`)
Tổng hợp kết quả các jobs đã hoàn thành, sinh báo cáo paired deltas:
```bash
python -m stage2_scratch collect \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --output runs/stage2_scratch_v4
```

### F. Đối chiếu mốc tham chiếu GĐ1 (`compare-stage1`)
Nạp 4 entry baseline lịch sử từ snapshot GĐ1 đã xác minh SHA-256:
```bash
python -m stage2_scratch compare-stage1 \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --output runs/stage2_scratch_v4
```

### G. Kiểm chứng Flower Adapter (`flower_verify`)
Chạy kiểm tra tích hợp 1 round giữa Flower client/server app và sequential oracle:
```bash
python -m stage2_scratch flower_verify \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --dataset dataset/PlantVillage-Dataset/raw/color \
  --output runs/flower_check
```

---

## 5. Đóng gói triển khai Kaggle

Để xuất gói nén độc lập `fl_package_v4.zip` và manifest `release_manifest_v4.json`:
```bash
python scripts/package_release_v4.py \
  --suite training-data/stage2/partitions_stage2_scratch_v3 \
  --output-dir training-artifacts/stage2/candidate-r1/packages \
  --notebook notebooks/fedavg_stage2_scratch_v4.ipynb
```

Notebook canonical tại `notebooks/fedavg_stage2_scratch_v4.ipynb` đã được cấu hình action guard:
- GPU smoke gate chỉ kích hoạt khi `ACTION == 'run'`.
- Collect & compare-stage1 chỉ chạy khi có output tương ứng.
- Toàn bộ execution counts và outputs đã được xóa sạch trước khi phát hành.

# Giai đoạn 2: Federated Learning trên PlantVillage (Bản snapshot: FedAvg Scratch)

Nhánh này (`code-train-fedavg-old`, tiền thân là `Stage-2-non-iid-simulation`) lưu trữ snapshot mã nguồn mô phỏng Federated Learning Giai đoạn 2 (FedAvg non-IID trên PlantVillage), bao gồm toàn bộ closure phụ thuộc: `stage2_scratch`, `stage2_matched`, `stage1_compat`, `fl_training`, `src`, configs, scripts, tests, notebooks và suite manifest phân hoạch.

Dataset gốc và thông tin nguồn được lưu tách riêng trên nhánh `dataset-sources`.

---

## 1. Bản lưu code FedAvg Scratch (`stage2_scratch`)

Phiên bản hiện hành trong bản lưu này là giao thức **`stage2_scratch_fedavg_v4`**, huấn luyện FedAvg hoàn toàn từ trọng số khởi tạo ngẫu nhiên (**`weights=None` / `pretrained=False`**).

- **Entrypoint chính**:
  ```powershell
  python -m stage2_scratch --help
  python -m stage2_scratch preflight --suite data/partitions_stage2_scratch_v3
  python -m stage2_scratch run --suite data/partitions_stage2_scratch_v3 --conditions label100 label1 label01 --seeds 42
  python -m stage2_scratch collect --output runs/stage2_scratch_v3 --suite data/partitions_stage2_scratch_v3
  ```
- **Backend thực thi**:
  - Mặc định là **mô phỏng tuần tự** (`sequential simulation runner` trong `stage2_scratch.runner`), đảm bảo tính lặp lại tất định và quản lý bộ nhớ ổn định.
  - Có adapter Flower độc lập tại `stage2_scratch.flower_adapter` phục vụ kiểm định tương đương (`flower_verify`).
- **Cấu hình mặc định của Scratch v4**:
  - Số client: 5 clients.
  - Vòng huấn luyện: 10 rounds, 1 local epoch/round, batch size 32.
  - Optimizer: mặc định SGD (learning rate resolve thành 0.01) hoặc tùy chọn AdamW.
  - Bộ phân hoạch mặc định: `data/partitions_stage2_scratch_v3` bao gồm 8 điều kiện non-IID (label skew: `label100`, `label1`, `label01`; quantity skew: `quantity100`, `quantity01`; label+quantity: `label_quantity01`; feature Dirichlet: `feature100`, `feature01`).
- **Notebook hiện hành**: `kaggle_stage2_scratch_v4.ipynb`.
  *(Các notebook `kaggle_stage2_scratch_v1.ipynb` đến `v3` và `kaggle_stage2_matched_v5.ipynb` được lưu giữ phục vụ tra cứu lịch sử).*
- **Lưu ý tài liệu**: `docs/STAGE2_SCRATCH_V1.md` là tài liệu ghi nhận lịch sử của bản thiết kế ban đầu (dùng AdamW và split cũ), không phải đặc tả cấu hình v4 hiện hành.

---

## 2. Tách bạch kết quả thực nghiệm và Baseline lịch sử

- **Kết quả Pretrained lịch sử**: Các artifacts và báo cáo trong thư mục `results/fedavg/...` (chứa các điểm accuracy ~99%) thuộc về đợt chạy MobileNetV3 pretrained của pipeline trước đây. **Tuyệt đối không trộn lẫn hoặc nhận các kết quả pretrained này làm kết quả của luồng scratch**.
- **Baselines Centralized và Local-only**: Không được huấn luyện lại từ scratch trong GĐ2; hệ thống đọc từ kết quả nghiệm thu GĐ1 làm tham chiếu lịch sử thông qua lệnh:
  ```powershell
  python -m stage2_scratch compare-stage1 --output runs/stage2_scratch_v3
  ```

---

## 3. Thiết kế dữ liệu và Phân hoạch Non-IID

- **PlantVillage màu**: 54.305 ảnh, 38 lớp trong `../PlantVillage-Dataset/raw/color`.
- **Tách holdout trước khi chia client**: Tách global test (~20%) và validation trước khi phân bổ tập train cho các client.
- **Bảo toàn nhóm lá (Leaf integrity)**: Giữ nguyên các nhóm ảnh chụp cùng một lá (`leaf_group_id`) trên cùng một phân vùng để chống rò rỉ dữ liệu giữa train/val/test.
- **Trục non-IID**:
  - *Lệch nhãn*: Phân bổ Dirichlet theo nhãn lớp với hệ số $\alpha \in \{100.0, 1.0, 0.1\}$.
  - *Lệch số lượng*: Phân bổ Dirichlet theo dung lượng client với $\alpha_q \in \{100.0, 0.1\}$.
  - *Lệch đặc trưng*: Biến đổi miền ảnh (độ sáng, tương phản, bão hòa) xác định riêng cho từng client theo profile cố định.

---

## 4. Kiểm định và Kiểm thử

Chạy kiểm tra tính toàn vẹn và bộ test của snapshot:

```powershell
# Kiểm tra CLI
python -m stage2_scratch --help

# Kiểm tra test scratch & protocol parity
pytest tests/test_stage2_scratch.py tests/test_stage2_protocol.py tests/test_flower_parity.py -v
```

*Ghi chú về trạng thái test*: Bộ kiểm thử cốt lõi cho `stage2_scratch` và `stage2_protocol` đạt 18/18 tests passed. Test fixture trong `tests/test_stage2_matched.py` yêu cầu dữ liệu ảnh mock cục bộ khi kiểm tra tính toàn vẹn nội dung ảnh, phản ánh đúng hiện trạng mã nguồn được đóng gói snapshot.

---

## 5. Nguồn tham khảo

1. Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He (2021), *Federated Learning on Non-IID Data Silos: An Experimental Study* (NIID-Bench): https://arxiv.org/pdf/2102.02079
2. Brendan McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data*: https://proceedings.mlr.press/v54/mcmahan17a.html
3. Torchvision MobileNetV3 specifications: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

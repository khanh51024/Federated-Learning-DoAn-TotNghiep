# Giai đoạn 2: Federated Learning trên PlantVillage (Bản Tổng Hợp: FedAvg Scratch-v4 & Models)

Nhánh này (`code-train-fedavg-old`) lưu trữ toàn diện mã nguồn huấn luyện, dữ liệu phân hoạch và các mô hình toàn cục đã tổng hợp (FedAvg) thuộc Giai đoạn 2 (mô phỏng Federated Learning non-IID trên PlantVillage, 38 lớp sâu bệnh cây trồng).

Repository được tổ chức theo kiến trúc namespace độc lập nhằm phân tách rõ ràng giữa mã nguồn vận hành hiện hành, mô hình nhị phân, dữ liệu lịch sử và nguồn gốc huấn luyện (provenance).

---

## 1. Cấu Trúc Repository & Các Không Gian Tên (Namespaces)

```
.
├── training/stage2_scratch_v4/       # Bộ mã nguồn Candidate R12 (137 tệp), chạy và huấn luyện chính quy từ đây
├── models/stage2_scratch_v4/         # 4 mô hình toàn cục hoàn tất + 5 checkpoints phục hồi (Git LFS)
│   ├── MODEL_INDEX.json              # Chỉ mục kỹ thuật chi tiết của tất cả mô hình và checkpoint
│   ├── README.md                     # Model cards và hướng dẫn tra cứu
│   ├── label01_seed42/               # Điều kiện Dirichlet label skew alpha=0.1 (best_model.pth)
│   ├── label100_seed42/              # Điều kiện Dirichlet label skew alpha=100.0 (best_model.pth)
│   ├── label1_seed42/                # Điều kiện Dirichlet label skew alpha=1.0 (best_model.pth)
│   ├── quantity100_seed42/           # Điều kiện quantity skew alpha_q=100.0 (best_model.pth)
│   └── quantity01_seed42/            # Điều kiện quantity skew alpha_q=0.1 (PAUSED_QUOTA checkpoint)
├── results/stage2_scratch_v4/        # Metadata gốc của đợt chạy Kaggle & PORTABLE_MANIFEST.json
├── provenance/kaggle-scratch-v4-training-source/  # Snapshot mã nguồn gốc trực tiếp sinh ra 4 mô hình
├── scripts/
│   └── predict_stage2.py             # Script suy luận an toàn (weights_only=True, EVAL_TRANSFORM)
├── data/
│   └── partitions_stage2_scratch_v3/ # Bộ phân hoạch 8 điều kiện non-IID (84 tệp, 264.037.275 bytes)
├── results/fedavg/                   # Dữ liệu & 6 mô hình .pt lịch sử của pipeline Pretrained GĐ1 (LFS)
└── docs/
    └── STAGE2_CODE_AND_MODELS.md     # Hướng dẫn chi tiết về vận hành, cấu trúc và kiểm định
```

---

## 2. Danh Mục Mô Hình Toàn Cục FedAvg (Scratch-v4)

Tất cả mô hình sử dụng kiến trúc **MobileNetV3-Small (38 lớp)**, khởi tạo ngẫu nhiên từ scratch (`weights=None`, `pretrained=False`), tổng hợp theo trọng số mẫu FedAvg ($w = \sum \frac{n_k}{N} w_k$):

| Điều Kiện Non-IID | Seed | Best Round | Tiêu Chí Chọn | Test Accuracy | Test Loss | Tệp Mô Hình (LFS) | File SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`label01_seed42`** | 42 | Round 5 | `validation_accuracy` | **25.5013%** | 2.6569 | `models/stage2_scratch_v4/label01_seed42/best_model.pth` | `04bb4db3...` |
| **`label100_seed42`** | 42 | Round 10 | `validation_accuracy` | **53.7222%** | 1.6366 | `models/stage2_scratch_v4/label100_seed42/best_model.pth` | `c01e858d...` |
| **`label1_seed42`** | 42 | Round 10 | `validation_accuracy` | **54.8576%** | 1.5841 | `models/stage2_scratch_v4/label1_seed42/best_model.pth` | `03d9972b...` |
| **`quantity100_seed42`**| 42 | Round 10 | `validation_accuracy` | **52.4311%** | 1.7454 | `models/stage2_scratch_v4/quantity100_seed42/best_model.pth` | `96f53e34...` |

> [!NOTE]
> Bốn mô hình trên đã được kiểm chứng an toàn: nạp `weights_only=True`, đối chiếu SHA-256 state dict, load strict vào MobileNetV3-Small và forward pass cho ra vector logits hữu hạn `[1, 38]`.
> Kèm theo là 5 tệp resume checkpoint tại `models/stage2_scratch_v4/*/checkpoints/`, trong đó `quantity01_seed42` mang trạng thái **PAUSED_QUOTA** (dừng ở round 5 do hạn ngạch Kaggle, không có final model).

---

## 3. Hướng Dẫn Thực Hiện Suy Luận (Inference)

Sử dụng `scripts/predict_stage2.py` để phân loại ảnh lá cây với bất kỳ mô hình toàn cục nào đã xuất bản:

```bash
python scripts/predict_stage2.py \
  --model models/stage2_scratch_v4/label1_seed42/best_model.pth \
  --image path/to/leaf_image.jpg \
  --top-k 5
```

Script tự động thiết lập đường dẫn namespace `training/stage2_scratch_v4`, kiểm tra nguồn gốc import (tránh fallback nhầm vào root cũ), áp dụng chuẩn tiền xử lý `EVAL_TRANSFORM` (RGB, resize $224 \times 224$, chuẩn hóa ImageNet) và in kết quả top-k dạng JSON.

---

## 4. Hướng Dẫn Tái Hiện Huấn Luyện (Training)

Để tái hiện huấn luyện mà không cần chia lại dữ liệu thô, sử dụng trực tiếp bộ phân hoạch sẵn có:

- Tham số `--suite`: Trỏ trực tiếp tới `data/partitions_stage2_scratch_v3` (84 tệp, **264.037.275 bytes**, SHA-256 `eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15`).
  *(Đính chính: Tài liệu R12 cũ từng ghi nhầm là 263.733.312 bytes; dung lượng thẩm định chính xác là 264.037.275 bytes).*
- Tham số `--dataset`: Trỏ tới thư mục chứa ảnh thô PlantVillage bên ngoài repo (`.../PlantVillage/raw/color`).

```bash
# Thiết lập đường dẫn mã nguồn vận hành
export PYTHONPATH=training/stage2_scratch_v4:$PYTHONPATH

# Kiểm tra tiền điều kiện (Preflight)
python -m stage2_scratch preflight \
  --suite data/partitions_stage2_scratch_v3 \
  --dataset /path/to/PlantVillage/raw/color

# Thực thi mô phỏng huấn luyện FedAvg
python -m stage2_scratch run \
  --suite data/partitions_stage2_scratch_v3 \
  --dataset /path/to/PlantVillage/raw/color \
  --conditions label100 label1 label01 \
  --seeds 42 \
  --output runs/stage2_scratch_v4
```

---

## 5. Nguồn Gốc Huấn Luyện (Training Provenance)

- **Mã nguồn trực tiếp sinh ra 4 mô hình**: Đóng gói tại `provenance/kaggle-scratch-v4-training-source/` (fingerprint: `6f2b6e8c1d3dd3a03dff0127b90aaf73b9dfc83c8c667eaa90d2e8f1a914a3bd`). Khớp hoàn toàn với trường `source_sha256` trong metadata các mô hình.
- **Mã nguồn vận hành hiện hành (Candidate R12)**: Đóng gói tại `training/stage2_scratch_v4/` (fingerprint: `192e40760869a3a00c3da056da67a1b3ef47273d8d9600ba57976d8fd8a572c7`).
- **Mô hình lịch sử**: Thư mục `results/fedavg/` chứa 6 tệp `.pt` lịch sử (tracked LFS) thuộc pipeline Pretrained GĐ1 (accuracy ~99%). Không gộp hay nhầm lẫn số liệu pretrained này cho mô hình scratch.

---

## 6. Giới Hạn Khoa Học & Quy Chuẩn Đánh Giá

1. **`scientific_stage2_complete = false`**: Mới chỉ hoàn tất 4 điều kiện ở seed 42; các điều kiện khác và kiểm chứng đa hạt giống (multi-seed) chưa hoàn tất.
2. **Không gộp trung bình (No Model Averaging)**: Không gộp trọng số giữa các mô hình thuộc các điều kiện phân phối khác nhau.
3. **Không chọn "Mô hình tốt nhất toàn bộ" theo Test Accuracy**: Sự chênh lệch độ chính xác (25,5% ở `label01` so với ~54,8% ở `label1`) phản ánh độ khó của phân bố non-IID, không phải sự vượt trội của thuật toán hay kiến trúc.

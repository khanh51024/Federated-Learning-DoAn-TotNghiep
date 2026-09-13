# Plant disease federated-learning baseline — Giai đoạn 1

Dự án dựng baseline có thể tái lập trên **PlantVillage / raw / color** (38 lớp):

- Centralized Learning: upper bound.
- Local-only trên 5 client: lower bound.
- Flower FedAvg: hệ thống chính.

Cả ba cấu hình sử dụng cùng MobileNetV3-Small và một split stratified cố định (`seed=42`): 72% train, 8% validation, 20% test. PlantDoc và các biến thể grayscale/segmented không dùng ở Giai đoạn 1. Validation dùng để chọn epoch/round; test chỉ được chạy một lần cho checkpoint đã chọn.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_setup.py
```

## Chạy theo thứ tự

```powershell
# 1. Kiểm tra dữ liệu và tạo split train/validation/test cố định
python data/inspect_dataset.py

# 2. Centralized baseline
python centralized/train_centralized.py --epochs 10

# 3. Local-only baseline (5 client)
python local_only/train_local.py --epochs 10 --alpha 1.0

# 4. FedAvg smoke test, sau đó chạy cấu hình báo cáo
python federated/run_simulation.py --rounds 3 --local-epochs 1 --alpha 1.0
python federated/run_simulation.py --rounds 10 --local-epochs 1 --alpha 1.0

# 5. Vẽ biểu đồ và bảng so sánh
python experiments/compare_results.py
```

Kết quả được ghi vào `experiments/results/`; split được lưu ở `experiments/splits/plantvillage_train0.72_val0.08_seed42.json`. Partition Dirichlet được lưu ở `experiments/partitions/plantvillage_alpha1_clients5_seed42.json` và được cả FedAvg lẫn Local-only đọc lại ở các lần chạy sau. Có thể truyền `--partition-file` để chỉ định artifact cụ thể. `fedavg_metrics.json` lưu train loss từ client và validation loss riêng biệt, cùng test metrics cuối của best validation round. File local-only lưu thêm min/max/std theo client và ma trận phân bố lớp. Các heatmap `fedavg_class_distribution.png` và `local_only_class_distribution.png` trực quan hóa phân vùng Dirichlet.

Để báo cáo kết quả ổn định, lặp lại cả ba baseline với `--seed 42`, `--seed 123`, và `--seed 2026`; chỉ kết luận FedAvg vượt centralized khi chênh lệch nhất quán qua các seed. Metadata leaf-to-image chưa có trong repository nên split hiện chưa thể được xác nhận là group-wise theo lá; không nên tuyên bố đã loại bỏ leaf-level leakage nếu chưa cung cấp metadata đó.

> MobileNetV3-Small mặc định dùng trọng số ImageNet. Lần đầu chạy có thể cần tải weights; nếu cần kiểm tra offline, thêm `--no-pretrained`.

Trong terminal, Centralized và Local-only hiển thị thanh tiến trình theo batch/epoch; FedAvg in loss, Accuracy và Macro-F1 sau mỗi server round.

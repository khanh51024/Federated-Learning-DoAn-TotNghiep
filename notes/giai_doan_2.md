# Giai đoạn 2 - Non-IID và chuẩn hóa thực nghiệm

## 1. Mục tiêu

Giai đoạn 2 đo ảnh hưởng của phân phối dữ liệu non-IID lên ba cấu hình:

- **Centralized**: gom toàn bộ `global_train`; dùng làm mốc tham chiếu.
- **FedAvg**: nhiều client train cục bộ, server tổng hợp trọng số theo số mẫu.
- **Local-only**: mỗi client train độc lập, không trao đổi tham số.

Dữ liệu sử dụng là PlantVillage 38 lớp, MobileNetV3-Small, ảnh 224x224 và chuẩn hóa ImageNet. Split toàn cục được tạo trước theo tỷ lệ 72% train, 8% validation và 20% test. Chỉ `global_train` được chia cho client; validation và test không đi qua Dirichlet.

Với Dirichlet, `alpha` càng nhỏ thì phân phối nhãn giữa các client càng lệch:

- `alpha=100`: gần IID.
- `alpha=10`: lệch rất nhẹ.
- `alpha=1`: non-IID vừa.
- `alpha=0.5`: non-IID rõ.
- `alpha=0.1`: non-IID mạnh.

## 2. Chuẩn bị môi trường

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_setup.py
```

Kiểm tra thành công khi có đủ thông tin PyTorch, torchvision, Flower, dataset và dòng:

```text
MobileNetV3-Small forward pass: OK
```

## 3. Tạo và kiểm tra split toàn cục

```powershell
python data/inspect_dataset.py
```

Split được lưu tại:

```text
experiments/splits/plantvillage_train0.72_val0.08_seed42.json
```

Khi đổi seed, sử dụng file split tương ứng. Không dùng test set để chọn epoch hoặc round tốt nhất.

## 4. Chạy Centralized

Centralized không phụ thuộc `alpha`, nên mỗi seed chỉ cần chạy một lần:

```powershell
python centralized/train_centralized.py --epochs 10 --seed 42
```

Có thể chạy nhiều seed:

```powershell
python centralized/train_centralized.py --epochs 10 --seed 123
python centralized/train_centralized.py --epochs 10 --seed 2026
```

Kết quả được ghi vào:

```text
experiments/results/centralized_metrics.json
experiments/results/centralized_best_model.pth
```

Checkpoint được chọn theo `validation_accuracy`; test chỉ được đánh giá sau khi chọn checkpoint.

## 5. Chạy Local-only theo alpha

Ví dụ chạy `alpha=1.0`:

```powershell
python local_only/train_local.py --epochs 10 --clients 5 --alpha 1.0 --seed 42
```

Chạy toàn bộ trục label skew:

```powershell
python local_only/train_local.py --epochs 10 --clients 5 --alpha 100 --seed 42
python local_only/train_local.py --epochs 10 --clients 5 --alpha 10 --seed 42
python local_only/train_local.py --epochs 10 --clients 5 --alpha 1.0 --seed 42
python local_only/train_local.py --epochs 10 --clients 5 --alpha 0.5 --seed 42
python local_only/train_local.py --epochs 10 --clients 5 --alpha 0.1 --seed 42
```

Local-only ghi accuracy và Macro-F1 của từng client trên cùng `global_test`, cùng các thống kê mean, std, min và max.

## 6. Chạy FedAvg theo alpha

Nên chạy smoke test trước:

```powershell
python federated/run_simulation.py --rounds 3 --local-epochs 1 --clients 5 --alpha 1.0 --seed 42
```

Cấu hình baseline hiện tại:

```powershell
python federated/run_simulation.py --rounds 10 --local-epochs 1 --clients 5 --alpha 1.0 --seed 42
```

Chạy toàn bộ trục label skew với cùng hyperparameter:

```powershell
python federated/run_simulation.py --rounds 10 --local-epochs 1 --clients 5 --alpha 100 --seed 42
python federated/run_simulation.py --rounds 10 --local-epochs 1 --clients 5 --alpha 10 --seed 42
python federated/run_simulation.py --rounds 10 --local-epochs 1 --clients 5 --alpha 1.0 --seed 42
python federated/run_simulation.py --rounds 10 --local-epochs 1 --clients 5 --alpha 0.5 --seed 42
python federated/run_simulation.py --rounds 10 --local-epochs 1 --clients 5 --alpha 0.1 --seed 42
```

FedAvg dùng Flower FedAvg và tổng hợp theo `num_examples`. Round tốt nhất được chọn theo validation accuracy, sau đó model mới được đánh giá trên test.

## 7. Partition reproducible

Mỗi cặp `alpha + clients + seed` có một partition JSON riêng, ví dụ:

```text
experiments/partitions/plantvillage_alpha1_clients5_seed42.json
```

FedAvg và Local-only đọc cùng file partition. Nếu file chưa tồn tại, lần chạy đầu tiên sẽ tạo file; các lần chạy sau sẽ tải lại và kiểm tra:

- đúng tổng số mẫu train;
- đúng số client;
- mỗi mẫu xuất hiện đúng một lần;
- có thống kê số mẫu, số lớp, entropy nhãn và tỷ lệ lớp lớn nhất.

Có thể chỉ định partition thủ công:

```powershell
python federated/run_simulation.py --rounds 10 --local-epochs 1 --alpha 0.1 --seed 42 --partition-file experiments/partitions/plantvillage_alpha0_1_clients5_seed42.json
```

## 8. Tổng hợp kết quả

```powershell
python experiments/compare_results.py
```

Các file chính:

```text
experiments/results/centralized_metrics.json
experiments/results/fedavg_metrics.json
experiments/results/local_only_metrics.json
experiments/results/comparison.md
experiments/results/comparison.png
```

Các metrics bắt buộc cần đọc:

- Accuracy và Macro-F1 trên test;
- validation accuracy và best epoch/round;
- accuracy/Macro-F1 từng client;
- mean, std, min, max của Local-only;
- số mẫu, số lớp hiện diện, entropy nhãn của từng client;
- khoảng cách `Centralized - FedAvg`;
- lợi ích cộng tác `FedAvg - Local-only mean`.

## 9. Kết quả baseline hiện có

Kết quả dưới đây là lần chạy đã có trong repository, với `seed=42`, `alpha=1.0`, 5 clients:

| Cấu hình                      | Accuracy | Macro-F1 |
| ----------------------------- | -------: | -------: |
| Centralized (best validation) |   98.90% |   98.46% |
| FedAvg (best validation)      |   99.02% |   98.44% |
| Local-only trung bình         |   90.30% |   86.47% |

Diễn giải hiện tại:

- FedAvg cao hơn Centralized khoảng `0.12` điểm phần trăm về Accuracy trong lần chạy này; chưa đủ cơ sở để kết luận FedAvg tốt hơn ổn định.
- FedAvg cao hơn Local-only khoảng `8.72` điểm phần trăm Accuracy.
- Macro-F1 của FedAvg thấp hơn Centralized khoảng `0.02` điểm phần trăm.
- Kết quả hiện tại mới là một seed và một mức `alpha`; chưa phải kết quả cuối của Giai đoạn 2.

## 10. Tiêu chí hoàn thành Giai đoạn 2

GĐ2 chỉ được kết luận sau khi có:

- partition reproducible cho `alpha=100, 10, 1, 0.5, 0.1`;
- kết quả Centralized, FedAvg và Local-only trên cùng test split;
- ít nhất ba seed: `42, 123, 2026`;
- Accuracy, Macro-F1 và metrics từng client;
- thống kê entropy và heatmap client x class;
- bảng `summary.csv` tổng hợp theo seed và alpha;
- biểu đồ Accuracy/Macro-F1 theo alpha;
- biểu đồ khoảng cách Centralized-FedAvg;
- xác định được mức non-IID khó để đưa sang Giai đoạn 3.

Không thay đổi learning rate, augmentation, backbone hoặc số round riêng cho từng alpha khi so sánh label skew. Các thuật toán FedProx, FedBN, SCAFFOLD và MOON thuộc Giai đoạn 3, không đưa vào để sửa kết quả FedAvg của Giai đoạn 2.

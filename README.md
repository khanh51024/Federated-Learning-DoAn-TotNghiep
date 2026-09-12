# Giai đoạn 2: Federated Learning trên PlantVillage

Repository này là bản chính của GĐ2: phân hoạch non-IID, FedAvg/MobileNetV3,
Centralized/Local-only baseline, evaluate, checkpoint/resume và reporting.

Code và kết quả mô phỏng non-IID của giai đoạn 2 được công bố trên nhánh
[`Stage-2-non-iid-simulation`](https://github.com/khanh51024/Federated-Learning-DoAn-TotNghiep/tree/Stage-2-non-iid-simulation).
Dataset cùng thông tin nguồn được tách riêng trên nhánh `dataset-sources`.
Kết quả của run FedAvg đã hoàn tất được tổng hợp tại
[`TRAINING_RESULTS.md`](TRAINING_RESULTS.md).

## Thiết kế dữ liệu

- PlantVillage màu: 54.305 ảnh, 38 lớp trong `../PlantVillage-Dataset/raw/color`.
- Tách global test khoảng 20% trước khi chia client. Centralized, Federated và
  Local-only dùng chung tập test này.
- Mọi biên chia mặc định giữ nguyên nhóm ảnh cùng một lá từ `leaf-map.json`.
  Leaf map chỉ phủ một phần corpus; ảnh chưa có metadata được xem là singleton,
  vì vậy không thể chứng minh chống rò rỉ lá cho phần này.
- Lệch nhãn: mỗi lớp dùng tỷ lệ client lấy từ `Dirichlet(alpha)`.
- Lệch số lượng: một vector dung lượng client lấy từ
  `Dirichlet(quantity_alpha)`, theo NIID-Bench mục IV-D.
- Lệch đặc trưng: profile ánh sáng/màu/cảm biến xác định theo client, áp dụng lúc
  load. Các bộ đo lệch nhãn và số lượng để `feature_skew: none`; một bộ IID riêng
  dùng `moderate` để đo trục đặc trưng.

`alpha` và `quantity_alpha` càng nhỏ thì độ lệch tương ứng càng mạnh. Cấu hình cũ
dùng `size_sigma` bị từ chối rõ ràng để tránh diễn giải nhầm LogNormal thành
Dirichlet.

## Chạy

```powershell
cd Federated-Learning-DoAn-TotNghiep
python scripts/sweep_alpha.py
python scripts/verify_integrity.py --all
python scripts/verify_source_images.py
python scripts/smoke_test_loader.py
python -m pytest tests -q
```

Luồng huấn luyện và nghiệm thu chi tiết nằm trong [TRAINING.md](TRAINING.md).
`smoke` và `stage2_sweep_smoke.yaml` chỉ kiểm tra code; chúng không phải kết quả
khoa học GĐ2.

Chia một cấu hình riêng:

```powershell
python scripts/partition_dataset.py --scenario label_skew --alpha 0.1
python scripts/partition_dataset.py --scenario quantity_skew --quantity-alpha 0.1
python scripts/partition_dataset.py --scenario iid --feature-skew moderate
```

Mỗi thư mục trong `data/partitions_v3` có manifest client, `centralized_train.csv`,
`global_test.csv`, thống kê, audit, provenance và `fedavg_meta.json`. Tên thư mục
chứa scenario, seed và mọi tham số ảnh hưởng kết quả để không ghi đè thí nghiệm.

## Contract FedAvg và MobileNetV3

```python
from src.data import FedAvgPartition

part = FedAvgPartition("data/partitions_v3/label_skew/<partition-name>")
client_loaders = part.all_client_loaders(batch_size=32)
global_test = part.global_test_loader(batch_size=128)
round_weights = part.aggregation_weights_for([0, 2, 5])
```

`n_k` là số ảnh train sau khi tách validation cục bộ. Trọng số FedAvg cho một
vòng được chuẩn hóa trên đúng tập client tham gia. Ảnh đưa vào MobileNetV3 là
float32 CHW, RGB, crop 224x224 và chuẩn hóa ImageNet. Evaluation dùng resize cạnh
ngắn 256 rồi center crop 224 theo torchvision. Centralized replay đúng profile
client của từng dòng để so sánh công bằng khi bật feature skew.

Nếu có PyTorch và torchvision, smoke test thực hiện một CPU forward qua
`mobilenet_v3_small(weights=None, num_classes=38)` và kiểm tra output `[B, 38]`.
Nếu thiếu hai gói này, phần NumPy/Pillow vẫn được kiểm tra và trạng thái forward
được ghi rõ là chưa chạy.

## Nguồn tham khảo

1. Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He (2021), *Federated Learning on
   Non-IID Data Silos: An Experimental Study* (NIID-Bench), mục IV-D:
   https://arxiv.org/pdf/2102.02079
2. Brendan McMahan et al. (2017), *Communication-Efficient Learning of Deep
   Networks from Decentralized Data*: https://proceedings.mlr.press/v54/mcmahan17a.html
3. Torchvision, `mobilenet_v3_small` weights and preprocessing:
   https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html



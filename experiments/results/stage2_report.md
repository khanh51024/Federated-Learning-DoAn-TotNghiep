# Giai đoạn 2 - Báo cáo thực nghiệm

Số cấu hình đã tổng hợp: **2**.

Các metrics test được tính trên cùng global test split; checkpoint được chọn theo validation accuracy.

| Alpha | Seed | Centralized Acc | FedAvg Acc | Local-only Acc mean | FedAvg - Local-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.9890 | 0.9914 | 0.9030 | 0.0884 |
| 100 | 42 | 0.9890 | - | 0.9669 | - |

## Artefacts

- `summary.csv`: tổng hợp theo alpha và seed.
- `client_metrics.csv`: metrics và thống kê partition theo client.
- `accuracy_by_alpha.png`: Accuracy theo alpha.
- `macro_f1_by_alpha.png`: Macro-F1 theo alpha.

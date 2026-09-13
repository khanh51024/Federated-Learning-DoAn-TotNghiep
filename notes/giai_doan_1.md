# Giai đoạn 1 — ghi chú nền tảng

- **Centralized**: toàn bộ train data được gom về một nơi; là upper bound.
- **Local-only**: mỗi client tự train, không trao đổi tham số; là lower bound.
- **Federated Learning**: server chỉ điều phối và tổng hợp, client giữ ảnh tại chỗ.
- **FedAvg**: server tổng hợp trọng số theo số mẫu: `w(t+1) = Σ (n_k / n) w_k(t+1)`.
- **Non-IID**: phân phối lớp của client khác nhau. Với Dirichlet, `alpha` càng nhỏ thì càng lệch và client drift càng mạnh.

Mọi so sánh GĐ1 phải dùng cùng test split được tạo bởi `data/inspect_dataset.py`.


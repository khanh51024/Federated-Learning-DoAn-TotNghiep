"""Bộ huấn luyện thống nhất cho release dữ liệu v4: Centralized, FedAvg (mô phỏng 5 client) và Local-only dùng CHUNG một công thức.

Lý do không sửa ba bộ chạy cũ ở các nhánh train: chúng gắn cứng release v3 và dùng hai công thức khác nhau (Centralized AdamW, FedAvg SGD không động lượng),
nên so sánh không công bằng (xem docs/DANH_GIA_DU_LIEU_MO_HINH_FEDAVG_2026-10-21.md). Ở đây ba cách có cùng mô hình khởi tạo (w0 theo hạt giống),
cùng bộ tối ưu, cùng tăng cường ảnh, cùng tổng số bước gradient và cùng tiêu chí chọn checkpoint. Checkpoint lưu `model_state` của
torchvision mobilenet_v3_small 38 lớp, cùng định dạng với mô hình đang dùng trên web (scripts/evaluate_published_models.py nạp được).
"""

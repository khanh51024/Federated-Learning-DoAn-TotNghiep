"""Cấu hình chung cho bộ test.

Một luồng cho OpenMP/MKL, đúng như dịch vụ trên máy chủ (deploy/cropfed-demo.service đặt OMP_NUM_THREADS=1 và MKL_NUM_THREADS=1).
Lý do: mỗi lần FastAPI chạy một handler đồng bộ trong luồng mới, PyTorch dựng thêm một đội OpenMP cho luồng đó và trên Windows mỗi đội giữ khoảng
100 MB bộ nhớ cam kết (commit) không trả lại. Một file test gửi vài trăm yêu cầu suy luận làm tiến trình pytest cam kết hơn 14 GB (working set chỉ ~0,8 GB),
rồi các tiến trình con (Flower, OpenBLAS) lỗi cấp phát và cả bộ test rớt ngẫu nhiên. Biến phải được đặt trước khi torch được import lần đầu.
"""
from __future__ import annotations

import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

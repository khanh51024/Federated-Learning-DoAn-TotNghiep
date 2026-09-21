# Đồ Án Tốt Nghiệp: Federated Learning trên PlantVillage với MobileNetV3-Small

> **Nghiên Cứu & Triển Khai Hệ Thống Học Liên Hợp (Federated Learning) Ứng Dụng Chẩn Đoán Bệnh Cây Trồng Trên Nền Tảng Flower Framework**

Hệ thống được thiết kế theo cấu trúc **Lean & Direct** (tinh gọn, trực tiếp, dễ đọc, dễ bảo trì):
- **`train/`**: Toàn bộ lõi AI / Machine Learning / Federated Learning (Centralized, Flower FedAvg, Local-only, Telemetry, Weight Inspector).
- **`web/`**: Giao diện điều khiển, giám sát thời gian thực và suy luận song song (FastAPI Backend + Modern Dark-mode Frontend).
- **`tests/`**: Bộ kiểm thử toán học độc lập cho thuật toán FedAvg, kiến trúc MobileNetV3-Small và phân hoạch non-IID.
- **`configs/`**: Cấu hình YAML chuẩn hóa cho các kịch bản thực nghiệm.
- **`docs/`**: Sơ đồ kiến trúc Mermaid ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) và Báo cáo đối sánh 3 trụ cột ([docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)).

---

## 1. Hướng Dẫn Cài Đặt & Chạy Nhanh (1-Click Quickstart)

### Bước 1: Khởi tạo môi trường ảo
```powershell
# Sử dụng uv (cài đặt siêu nhanh) hoặc python chuẩn
python -m uv venv .venv --python 3.11
.\.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Chạy kiểm thử tự động (Unit Tests)
Chứng minh tính đúng đắn toán học của FedAvg ($n_k$), trích xuất trọng số và tính toàn vẹn SHA-256:
```powershell
python -m pytest tests/test_fedavg.py tests/test_model.py tests/test_partition.py -v
```

### Bước 3: Huấn luyện mô hình tập trung (Centralized Baseline)
```powershell
# Chạy smoke test nhanh (1 epoch)
python -m train.centralized --smoke --epochs 1

# Hoặc huấn luyện đầy đủ
python -m train.centralized --epochs 10 --batch-size 32 --lr 0.01
```

### Bước 4: Huấn luyện Federated Learning (FedAvg Simulation)
```powershell
# Chạy mô phỏng 5 clients x 2 communication rounds
python -m train.server --smoke --rounds 2 --num-clients 5
```

### Bước 5: Khởi động Giao diện Web Dashboard & Chẩn Đoán Ảnh
```powershell
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```
Truy cập trình duyệt tại: **`http://localhost:8000`** để trải nghiệm đầy đủ 6 màn hình:
1. **Dashboard**: Giám sát KPIs toàn cục, đường cong hội tụ, phân phối dữ liệu.
2. **Centralized Baseline**: Điều khiển và quan sát baseline tập trung.
3. **Huấn Luyện FedAvg**: Cấu hình và kích hoạt các vòng huấn luyện liên hợp.
4. **Clients & Telemetry**: Thẻ chi tiết 5 clients (số mẫu $n_k$, chỉ số đói dữ liệu, RTT, Weight Inspector).
5. **So Sánh & Benchmark**: Bảng đối sánh 3 trụ cột (Centralized vs FedAvg vs Local-only) và biểu đồ Dirichlet $\alpha$.
6. **Nhận Diện (Inference)**: Tải ảnh lá cây lên và nhận kết quả đối chiếu song song giữa mô hình Centralized và mô hình Federated Global.

---

## 2. Triển Khai Phân Tán Thực Tế Trên Nhiều Máy / Tiến Trình (Network Deployment)

Flower cho phép chạy máy chủ và các máy khách trên các tiến trình hoặc máy tính vật lý độc lập qua mạng:

### Khởi động Server (Máy chủ Flower):
```powershell
python -m train.server --mode deployment --port 8080 --num-clients 5 --rounds 10
```

### Khởi động 5 Clients (trên các cửa sổ terminal hoặc các máy khác nhau):
```powershell
# Terminal Client 0:
python -m train.client --client-id 0 --server <IP_SERVER>:8080

# Terminal Client 1:
python -m train.client --client-id 1 --server <IP_SERVER>:8080

# (Tương tự cho Client 2, 3, 4...)
```
Server sẽ tự động nhận diện đủ 5 clients, phát tán mô hình toàn cục, thu nhận updates và tổng hợp FedAvg qua mạng.

---

## 3. Cấu Trúc Mã Nguồn (Directory Tree)

```text
Federated-Learning-DoAn-TotNghiep/
├── README.md                # Tài liệu hướng dẫn này
├── requirements.txt         # Thư viện phụ thuộc
├── .env.example             # Biến môi trường mẫu
│
├── train/                   # CORE MACHINE LEARNING & FEDERATED LEARNING
│   ├── model.py             # MobileNetV3-Small (Pretrained & Scratch, 38 classes, SHA-256)
│   ├── data.py              # PlantVillage Dataset loader & Dirichlet Partitioning
│   ├── strategy.py          # Thuật toán FedAvg chuẩn w_(t+1) = sum(n_k * w_k) / sum(n_k)
│   ├── centralized.py       # Huấn luyện Centralized baseline & API predict ảnh
│   ├── local_only.py        # Huấn luyện Local-only baseline (mỗi cơ sở tự học)
│   ├── server.py            # Flower ServerApp (điều phối vòng lặp FedAvg & mạng)
│   ├── client.py            # Flower ClientApp (huấn luyện cục bộ SGD & đo update norm)
│   ├── telemetry.py         # Đo đạc Loss != Latency, RTT, train time, byte trao đổi
│   ├── weight_inspector.py  # Thống kê L2 norm, Update Norm ||Delta_k||_2, mean, std
│   └── faulty_client.py     # Giả lập client lỗi: no-train, random, scaled, noise
│
├── web/                     # GIAO DIỆN QUAN SÁT & SUY LUẬN
│   ├── app.py               # Máy chủ FastAPI: REST API & phục vụ frontend
│   └── static/              # Giao diện Dark-mode, Chart.js, 6 tab trực quan
│       ├── index.html
│       ├── css/style.css
│       └── js/main.js
│
├── configs/                 # CẤU HÌNH THỰC NGHIỆM
│   ├── default.yaml         # Cấu hình tham số mặc định
│   └── non_iid.yaml         # Ma trận Dirichlet alpha = 100, 1.0, 0.1
│
├── tests/                   # BỘ KIỂM THỬ TOÁN HỌC & TÍNH TOÀN VẸN
│   ├── test_fedavg.py       # Kiểm định toán học FedAvg (scalar 2.5, conv tensor, NaN)
│   ├── test_model.py        # Kiểm định MobileNetV3-Small, nạp rút tham số, hash
│   └── test_partition.py    # Kiểm định bảo toàn mẫu Dirichlet & đo lường đói dữ liệu
│
└── docs/                    # BÁO CÁO & PHẢN BIỆN KHOA HỌC
    ├── ARCHITECTURE.md      # Sơ đồ kiến trúc Mermaid (Centralized, Flower, Client, FedAvg)
    └── EXPERIMENT_RESULTS.md# Bảng đối sánh 3 trụ cột & cẩm nang trả lời phản biện
```

---

## 4. Tóm Tắt Kết Quả Đối Sánh 3 Trụ Cột

Đánh giá trên cùng tập kiểm thử chung **`global_test.csv` (10.921 ảnh, 38 lớp)**:

```text
               Centralized Baseline (98.25%)
                           |
                     [ -1.13% gap ]
                           v
               FedAvg Global Model  (97.12%)
                           |
                    [ +15.72% gain ]
                           v
               Local-only Baseline  (81.40%)
```

- **FedAvg mang lại giá trị to lớn (+15.72% so với Local-only)**: Chứng minh việc hợp tác huấn luyện liên hợp giữa các cơ sở giúp vượt qua rào cản thiếu hụt dữ liệu (data scarcity) mà không cần chia sẻ ảnh thô.
- **Sự đánh đổi rất nhỏ (-1.13% so với Centralized)**: Đổi lại việc bảo toàn tính riêng tư dữ liệu và tuân thủ các quy định chủ quyền dữ liệu nông nghiệp.

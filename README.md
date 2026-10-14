# Đồ Án Tốt Nghiệp: Federated Learning trên PlantVillage với MobileNetV3-Small

> **Demo bảo vệ GĐ1/GĐ2 hiện tại:** xem [hướng dẫn web và nguồn số liệu](docs/WEB_DEMO_GD1_GD2.md). Trang web dùng ba nhánh mixed PlantVillage/PlantDoc đã ghim; phiên Flower 5 client localhost là smoke kỹ thuật trên ảnh tổng hợp. Các lệnh và số liệu scratch trong phần lịch sử bên dưới không phải đối chứng chính của demo.

> **Nghiên Cứu & Triển Khai Hệ Thống Học Liên Hợp (Federated Learning) Ứng Dụng Chẩn Đoán Bệnh Cây Trồng Trên Nền Tảng Flower Framework**

Hệ thống được tổ chức theo cấu trúc **Monorepo hoàn chỉnh**:
- **`src/`**: Toàn bộ lõi AI / Machine Learning / Federated Learning chuẩn hóa (`model`, `data`, `centralized`, `federated`, `common`).
- **`web/`**: Giao diện điều khiển, giám sát thời gian thực và suy luận song song (FastAPI Backend + giao diện sáng 6 bước theo trình tự bảo vệ đồ án).
- **`tests/`**: Bộ kiểm thử toán học độc lập (`test_fedavg.py`, `test_model.py`, `test_parameters.py`, `test_partition.py`, `test_integration.py`).
- **`configs/`**: Cấu hình YAML chuẩn hóa cho các kịch bản thực nghiệm (`centralized.yaml`, `federated.yaml`, `non_iid.yaml`).
- **`legacy/`**: Pipeline Kaggle/stage* cũ, notebook, config và test của chúng — chỉ để tra cứu/tái lập lịch sử, chạy riêng ([legacy/README.md](legacy/README.md)).
- **`docs/`**: Hợp đồng model registry ([docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md)), Runbook vận hành ([docs/RUNBOOK.md](docs/RUNBOOK.md)), Giao thức thực nghiệm ([docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)), Sơ đồ kiến trúc Mermaid ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

> **Hướng chính thức (một chiều):** train NGOÀI repo (Kaggle/branch) → bundle + manifest + SHA-256 → `scripts/import_branch_demo.py` kiểm hash → `artifacts/registry.json` → web/báo cáo. Máy dev chỉ CPU, không train; engine trong repo là `src/`. Chi tiết: [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md).

> **Trạng thái tiến độ:** xem [PROJECT_STATUS.md](PROJECT_STATUS.md) — file thống kê duy nhất, tự sinh bởi `python scripts/update_status.py`. Mọi con số accuracy trong repo phải truy được về `artifacts/registry.json`.

---

## 1. Hướng Dẫn Cài Đặt & Chạy Nhanh (1-Click Quickstart)

### Bước 1: Khởi tạo môi trường ảo
```powershell
# Sử dụng uv hoặc python chuẩn (Python 3.11)
python -m venv .venv
.\.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ kiểm thử tự động (Unit & Integration Tests)
Chứng minh tính đúng đắn toán học của FedAvg ($n_k$), trích xuất trọng số, tính toàn vẹn SHA-256 và luồng tích hợp 5 clients:
```powershell
python -m pytest tests/test_fedavg.py tests/test_model.py tests/test_parameters.py tests/test_partition.py tests/test_integration.py -v
```

### Bước 3: Huấn luyện mô hình tập trung (Centralized Baseline)
```powershell
# Chạy smoke test nhanh (1 epoch)
python -m src.centralized.train --smoke --epochs 1

# Hoặc huấn luyện đầy đủ
python -m src.centralized.train --epochs 10 --batch-size 32 --lr 0.01
```

### Bước 4: Huấn luyện Federated Learning (FedAvg Simulation)
```powershell
# Chạy mô phỏng 5 clients x 2 communication rounds
python -m src.federated.server_app --smoke --rounds 2 --num-clients 5
```

### Bước 5: Khởi động Giao diện Web Dashboard & Chẩn Đoán Ảnh
```powershell
python -m uvicorn web.app:app --host 0.0.0.0 --port 8088
```
Truy cập trình duyệt tại: **`http://localhost:8088`** để xem 6 bước trình bày (chi tiết trong [docs/WEB_DEMO_GD1_GD2.md](docs/WEB_DEMO_GD1_GD2.md)):
1. **Tổng quan**: đề tài, cách FedAvg hoạt động, ba cách huấn luyện và kết quả chính.
2. **Dữ liệu phân tán**: mỗi client giữ phần nào của 38 nhóm; 12 cách chia dữ liệu.
3. **Kết quả & so sánh**: bảng 7 mô hình, tập giữ riêng và khoảng tin cậy, ma trận nhầm lẫn, độ tin cậy xác suất.
4. **Chẩn đoán bệnh lá**: một ảnh qua cả 7 mô hình, đúng/sai và lý do, Grad-CAM.
5. **Huấn luyện thử**: chạy Flower 5 client trên máy này để xem quy trình FedAvg từng vòng.
6. **Bằng chứng & hạn chế**: những gì đã kiểm chứng, hạn chế và việc còn lại.

---

## 2. Triển Khai Phân Tán Thực Tế Trên Nhiều Máy / Tiến Trình (Network Deployment)

Flower cho phép chạy máy chủ và các máy khách trên các tiến trình hoặc máy tính vật lý độc lập qua mạng:

### Khởi động Server thật (khuyến nghị dùng launcher):
```powershell
.\scripts\start_real_flower_server.ps1 `
  -Rounds 10 `
  -NumClients 5 `
  -LocalEpochs 1 `
  -Seed 42 `
  -Port 8089 `
  -EvaluationDatasetRoot D:\PlantVillage `
  -RejectSuspicious
```

Server launcher mặc định **bắt buộc client dùng dữ liệu PlantVillage thật**. Probe RTT chạy ở `port + 1` (ví dụ `8090`). `-EvaluationDatasetRoot` là tùy chọn nhưng được khuyến nghị cho nghiệm thu: server chỉ dùng `global_val/global_test` để đánh giá global model sau FedAvg, không dùng dữ liệu training của client. Nếu bật evaluator, mọi đường dẫn held-out trong manifest phải resolve thật.

### Khởi động 5 Clients trên các máy/process độc lập:
```powershell
# Ví dụ Client 0
.\scripts\start_real_flower_client.ps1 `
  -ClientId 0 `
  -Server <IP_SERVER>:8089 `
  -DatasetRoot D:\PlantVillage `
  -PartitionDir data\partitions_stage2_scratch_v3\label100
```

Đổi `-ClientId` thành `1..4` cho các client còn lại. Mỗi client chỉ cần sở hữu **ảnh thuộc local partition của chính nó**. Launcher chạy `preflight_client_partition.py` theo `client_XX.csv`; không bắt mỗi client phải có toàn bộ 54.305 ảnh. Nếu thiếu bất kỳ ảnh local nào, client dừng thay vì fallback synthetic.

Server chỉ aggregate update khi client hợp lệ theo policy hiện tại, đồng thời lưu RTT probe, local train time, update norm, seed, partition ID/hash, client-manifest hash, validator status và checkpoint global model.

---

## 3. Cấu Trúc Mã Nguồn Chuẩn Hóa (Monorepo Directory Tree)

```text
Federated-Learning-DoAn-TotNghiep/
├── README.md                # Hướng dẫn toàn diện đồ án
├── requirements.txt         # Pinned dependencies
├── pyproject.toml           # Cấu hình project & tools
├── .env.example             # Biến môi trường mẫu
│
├── src/                     # MÃ NGUỒN LÕI HỆ THỐNG
│   ├── model/               # Kiến trúc MobileNetV3-Small (Pretrained & Scratch)
│   │   ├── mobilenetv3.py
│   │   └── __init__.py
│   │
│   ├── data/                # Nạp dữ liệu PlantVillage & Phân hoạch Dirichlet
│   │   ├── plantvillage.py
│   │   ├── split.py
│   │   └── partition.py
│   │
│   ├── centralized/         # Baseline học tập trung & suy luận ảnh
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   │
│   ├── federated/           # Flower FedAvg, Telemetry & Weight Inspector
│   │   ├── client_app.py    # Flower ClientApp & vòng đời huấn luyện cục bộ
│   │   ├── server_app.py    # Flower ServerApp & FedAvg điều phối mạng
│   │   ├── strategy.py      # Thuật toán FedAvg w_(t+1) = sum(n_k * w_k) / sum(n_k)
│   │   ├── telemetry.py     # Đo đạc Loss != Latency, RTT, train time
│   │   ├── weight_inspector.py # Thống kê L2 norm, Update Norm ||Delta_k||_2
│   │   └── faulty_client.py # Giả lập client lỗi/tấn công
│   │
│   └── common/              # Tiện ích dùng chung
│       ├── metrics.py       # Macro-F1 & ma trận nhầm lẫn
│       ├── logging.py       # Định dạng log chuẩn
│       └── config.py        # YAML config loader
│
├── web/                     # ỨNG DỤNG WEB QUAN SÁT & SUY LUẬN
│   ├── app.py               # FastAPI: chỉ mount router
│   ├── routers/             # monitoring.py, inference.py, training.py
│   ├── services/model_service.py  # MỘT CỬA inference — chỉ mở cho model đã qua cổng web_visible của registry
│   ├── state.py             # trạng thái background task + Flower demo
│   └── static/              # Giao diện 6 bước (HTML/CSS/ES module, không bước build)
│       ├── index.html       # 1 Tổng quan · 2 Dữ liệu phân tán · 3 Kết quả & so sánh · 4 Chẩn đoán · 5 Huấn luyện thử · 6 Bằng chứng
│       ├── css/{style,pages}.css
│       ├── js/app.js        # khởi động; js/core/ (router, store, api, charts, status, theme, motion, flnet, crosshair), js/pages/ (mỗi bước một file)
│       ├── samples/demo/    # 19 ảnh mẫu có nhãn thật + catalog.json (scripts/build_demo_gallery.py)
│       ├── fonts/           # Be Vietnam Pro + Source Serif 4 (tiêu đề) + JetBrains Mono, tự host
│       └── vendor/          # Chart.js 4.4.1 ghim phiên bản
│
├── legacy/                  # PIPELINE CŨ (Kaggle/stage*) — chỉ tra cứu, chạy riêng
│   ├── fl_training/ stage1_compat/ stage2_matched/ stage2_scratch/ experiments/
│   ├── tests/               # pytest legacy/tests
│   └── configs/ scripts/ *.ipynb release_manifest_v*.json
│
├── tests/                   # BỘ KIỂM THỬ TOÁN HỌC & ĐỘ TOÀN VẸN
│   ├── test_fedavg.py       # Kiểm định trọng số toán học FedAvg
│   ├── test_model.py        # Kiểm định MobileNetV3-Small & 38 lớp
│   ├── test_parameters.py   # Kiểm định trích xuất, nạp & SHA-256 hash
│   ├── test_partition.py    # Kiểm định phân hoạch Dirichlet & đói dữ liệu
│   └── test_integration.py  # Kiểm thử tích hợp luồng 5 clients x 1 round
│
├── configs/                 # CẤU HÌNH PHÂN HOẠCH DỮ LIỆU
│   └── partition_config.yaml
│
├── artifacts/               # SẢN PHẨM THỰC NGHIỆM
│   ├── checkpoints/         # centralized_best.pt, fedavg_best.pt
│   └── metrics/             # centralized_results.json, fedavg_results.json, telemetry_history.json
│
└── docs/                    # BÁO CÁO & PHẢN BIỆN KHOA HỌC
    ├── EXPERIMENT_PROTOCOL.md # Giao thức thực nghiệm đóng băng
    ├── ARCHITECTURE.md      # Sơ đồ kiến trúc Mermaid (Centralized, Flower, Client, FedAvg)
    ├── MODEL_REGISTRY.md   # Hợp đồng nguồn sự thật (registry)
    └── RUNBOOK.md          # Lệnh vận hành hiện hành
```

---

## 4. Trạng Thái Kết Quả Đối Sánh

> **Quan trọng:** nguồn sự thật duy nhất về model/checkpoint/metrics là `artifacts/registry.json` (hợp đồng trong [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md)). Các sự thật đã được registry xác nhận (2026-10-09):
>
> 1. **Checkpoint canonical (FedAvg 99,09% / Centralized 98,25%) KHÔNG còn trên máy** — chỉ còn metrics JSON của run Kaggle. Số liệu này chỉ được trích dẫn kèm nhãn "metrics-only / historical".
> 2. Các checkpoint smoke (acc=1.0) từng ghi đè slot canonical đã được **cách ly** vào `artifacts/checkpoints/smoke/` và code đã chặn smoke ghi lại.
> 3. Demo web dùng họ **published_demo** (pretrained mixed PV+PlantDoc, hash-verified) — hoạt động bình thường, ghi rõ là DEMO.
> 4. Provenance hiện `UNVERIFIED` — đúng như thật, sẽ chuyển `verified` sau khi tái chạy run canonical với dataset gốc.


---

## 5. Quy Trình Nghiệm Thu Khoa Học Khuyến Nghị

### 5.1. Kiểm tra trạng thái trước demo

```powershell
.\.venv\Scripts\python.exe scripts\pre_demo_check.py --dataset-root D:\PlantVillage
```

Kết quả `ready` mới có nghĩa là toàn bộ scientific gates đã hoàn tất. `technical_demo_ready` chỉ có nghĩa code/demo kỹ thuật chạy được nhưng còn thiếu bằng chứng khoa học thật.

### 5.2. Chạy bộ thực nghiệm thật nhiều seed

```powershell
.\scripts\run_real_multiseed_report.ps1 `
  -DatasetRoot D:\PlantVillage `
  -CentralizedEpochs 10 `
  -FederatedRounds 10 `
  -Seeds 42,123,2026
```

Script sẽ preflight toàn bộ PlantVillage, chạy Centralized và các điều kiện FedAvg `label100`, `label1`, `label01`, `quantity01` cho từng seed, sau đó sinh mean/std tại:

```text
artifacts/report_experiments/multiseed_real_summary.json
artifacts/report_experiments/multiseed_real_summary.csv
```

Các seed trên thay đổi model initialization/DataLoader shuffle trên partition manifests hiện có; chưa đại diện cho partition-seed variance.

### 5.3. Khóa một run thật thành verified artifact

Sau khi một run `mode=real` hoàn tất và dataset preflight PASS:

```powershell
.\.venv\Scripts\python.exe scripts\promote_verified_run.py `
  artifacts\report_experiments\<TIMESTAMP> `
  --scenario label1
```

Promotion chỉ thành công khi `mode=real`, `scientific_accuracy_valid=true`, preflight PASS và đủ checkpoint/metrics. Artifact được sao chép vào `artifacts/verified/<promotion_id>/` cùng SHA-256 và `verified_manifest.json`; smoke/demo run không thể promote.

### 5.4. Nguyên tắc báo cáo

- Không lấy accuracy/F1/loss từ smoke/synthetic làm kết quả khoa học.
- Không dùng canonical lịch sử đang `UNVERIFIED` để kết luận.
- Network latency phải phân biệt `probe_rtt_ms` từng client với `estimated_comm_overhead_ms` của round.
- Kết quả chính nên báo cáo `mean ± std` qua nhiều training seeds.
- Faulty-client experiment phải nêu rõ policy `observe-only` hay `reject-suspicious` và ngưỡng validator.


### 5.5. Một lệnh chạy scientific network 1 server + 5 clients trên cùng máy

Để nghiệm thu pipeline Flower network thật bằng 6 process độc lập trên một máy trước khi chuyển sang nhiều máy vật lý:

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 5 `
  -LocalEpochs 1 `
  -Seed 42
```

Script tự thực hiện theo thứ tự:

1. full-path preflight PlantVillage;
2. khởi động Flower server thật trên `127.0.0.1:8089`;
3. khởi động 5 Flower client process độc lập;
4. bắt buộc mọi client báo `data_mode=real`;
5. chạy held-out `global_val/global_test` thật tại server;
6. lưu RTT per-client, update norm, audit status, checkpoint global;
7. chạy `validate_network_artifact.py`;
8. chỉ khi mọi scientific gate PASS mới chạy `promote_verified_network_run.py`;
9. copy artifact hợp lệ vào `artifacts/verified_network/<promotion_id>/` kèm SHA-256 và `verified_network_manifest.json`.

Có thể kiểm tra command flow mà không khởi động training:

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 3 `
  -Seed 123 `
  -DryRun
```

Artifact smoke/synthetic sẽ bị `validate_network_artifact.py` từ chối và không thể promote vào thư mục verified.

# Kiến Trúc Hệ Thống Federated Learning (System Architecture)

Tài liệu này cung cấp các sơ đồ kiến trúc chuẩn hóa phục vụ báo cáo và phản biện đồ án tốt nghiệp với hội đồng.

---

## 1. So Sánh Mô Hình Tập Trung (Centralized) vs Phân Tán (Federated)

```mermaid
graph TD
    subgraph Centralized ["Mô hình Học Tập Trung (Centralized)"]
        D1["Dữ liệu Nông trại 1"] -->|Gửi toàn bộ ảnh thô| S1[("Máy chủ Trung tâm")]
        D2["Dữ liệu Nông trại 2"] -->|Gửi toàn bộ ảnh thô| S1
        D3["Dữ liệu Nông trại 3"] -->|Gửi toàn bộ ảnh thô| S1
        S1 -->|Tập hợp 54.305 ảnh| T1["Huấn luyện MobileNetV3"]
    end

    subgraph Federated ["Mô hình Học Liên Kết (Federated Learning)"]
        F1["Nông trại 1 (Local Data)"] -.->|Chỉ gửi Update Trọng số & n_k| S2[("Flower Server (FedAvg)")]
        F2["Nông trại 2 (Local Data)"] -.->|Chỉ gửi Update Trọng số & n_k| S2
        F3["Nông trại 3 (Local Data)"] -.->|Chỉ gửi Update Trọng số & n_k| S2
        S2 -->|Tổng hợp trọng số mới| G1["Global Model"]
        G1 -.->|Phát tán trọng số toàn cục| F1
        G1 -.->|Phát tán trọng số toàn cục| F2
        G1 -.->|Phát tán trọng số toàn cục| F3
    end
```

---

## 2. Kiến Trúc Triển Khai Flower Thực Tế (Deployment Topology)

Mô hình triển khai chuẩn: Server và các Client chạy trên các process/máy tính độc lập, kết nối qua mạng IP/Port.

```mermaid
graph TB
    subgraph ServerHost ["MÁY CHỦ (SERVER HOST)"]
        SA["Flower ServerApp / SuperLink"]
        FedAvg["FedAvg Strategy Engine"]
        GlobalModel["Global MobileNetV3 Checkpoint"]
        SA --- FedAvg
        FedAvg --- GlobalModel
    end

    SA <== "Mạng IP/Port (gRPC / HTTP)" ==> C1["Client 01 (Process riêng)"]
    SA <== "Mạng IP/Port (gRPC / HTTP)" ==> C2["Client 02 (Process riêng)"]
    SA <== "Mạng IP/Port (gRPC / HTTP)" ==> C3["Client 03 (Process riêng)"]
    SA <== "Mạng IP/Port (gRPC / HTTP)" ==> C4["Client 04 (Process riêng)"]
    SA <== "Mạng IP/Port (gRPC / HTTP)" ==> C5["Client 05 (Process riêng)"]

    subgraph ClientNodes ["CÁC CLIENT NỘI BỘ"]
        C1 --- D1["Local Partition 1 (Không chia sẻ)"]
        C2 --- D2["Local Partition 2 (Không chia sẻ)"]
        C3 --- D3["Local Partition 3 (Không chia sẻ)"]
        C4 --- D4["Local Partition 4 (Không chia sẻ)"]
        C5 --- D5["Local Partition 5 (Không chia sẻ)"]
    end
```

---

## 3. Vòng Đời Huấn Luyện Cục Bộ Của Client (Client Lifecycle)

```mermaid
sequenceDiagram
    autonumber
    actor Server as Flower Server (FedAvg)
    actor Client as Client Node (k)

    Server->>Client: 1. Phát tán Global Weights (w_t)
    Note over Client: 2. Nhận weights & tính RECV_HASH
    Note over Client: 3. Nạp weights vào MobileNetV3 cục bộ
    Note over Client: 4. Nạp phân hoạch dữ liệu riêng (Client CSV)
    Note over Client: 5. Huấn luyện E epochs với fresh SGD
    Note over Client: 6. Đánh giá loss và accuracy cục bộ
    Note over Client: 7. Tính Delta_k = w_k - w_t và ||Delta_k||_2
    Note over Client: 8. Tính SEND_HASH = SHA256(w_k)
    Client->>Server: 9. Gửi Update Trọng số + n_k + Hash + Telemetry
    Note over Server: 10. Server xác thực CLIENT_HASH == SERVER_HASH
```

---

## 4. Giao Thức Tổng Hợp FedAvg Tại Server

$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{\sum_j n_j} w_{k, t+1}$$

```mermaid
flowchart TD
    Start["Bắt đầu Round t"] --> Select["Chọn K Clients (K=5)"]
    Select --> Dispatch["Gửi w_t tới các Clients"]
    Dispatch --> Wait["Chờ K Client hoàn thành local train"]
    Wait --> Collect["Thu nhận updates w_k và số mẫu n_k"]
    Collect --> Verify["Kiểm tra SHA-256 Hash & Shapes & NaN"]
    Verify -->|Hợp lệ| Aggregate["Tính tổng trọng số: sum(n_k * w_k) / sum(n_k)"]
    Verify -->|Không hợp lệ| Reject["Từ chối update dị thường / Log error"]
    Aggregate --> UpdateGlobal["Cập nhật w_(t+1) cho Global Model"]
    UpdateGlobal --> GlobalEval["Đánh giá trên tập chung Global Test (10.921 ảnh)"]
    GlobalEval --> SaveCkpt["Lưu Checkpoint best.pt & last.pt"]
    SaveCkpt --> NextRound{"Còn round tiếp theo?"}
    NextRound -->|Có| Start
    NextRound -->|Hết rounds| Finish["Kết thúc phiên FL & Xuất Telemetry JSON"]
```

---

## 5. Kiến Trúc Tầng Ứng Dụng Web (Web Presentation & Inference)

Web chỉ làm Presentation + API + Inference. Mọi dự đoán đi qua **một cửa**
(`web/services/model_service.py`), cửa này chỉ mở cho model đã qua cổng `web_visible` của registry.

```mermaid
graph LR
    UI["Frontend Web (Dashboard, Charts, Upload)"]
    -->|REST API / HTTP| APP["web/app.py (mount router)"]
    APP --> MON["routers/monitoring.py<br/>status, metrics, clients, weights"]
    APP --> INF["routers/inference.py<br/>/api/models, /api/inference, /api/compare"]
    APP --> TRN["routers/training.py<br/>điều khiển demo Flower"]
    INF --> SVC["services/model_service.py<br/>predict(model_id) — một cửa"]
    MON --> SVC
    SVC -->|đọc| REG[("artifacts/registry.json<br/>(sinh bởi scripts/build_registry.py)")]
    SVC -->|chỉ khi web_visible| BUN["web/demo_evidence.py (bundle đã kiểm hash)<br/>web/plantvillage_reference.py (TorchScript)"]
    TRN --> ENG["src/ (centralized, federated, telemetry)"]
    ENG -->|Lưu/Đọc| ART[("artifacts/ (checkpoints, metrics)")]
    APP -->|Phản hồi JSON| UI
```

Cổng `web_visible` (5 điều kiện, tính tự động trong `build_registry.py`): hash checkpoint khớp manifest;
thứ tự 38 lớp khớp taxonomy; `preprocessing_id` đã khai báo; metrics gắn được với dataset + protocol;
bài forward bắt buộc trên ảnh mẫu đạt. Xem [MODEL_REGISTRY.md](MODEL_REGISTRY.md).

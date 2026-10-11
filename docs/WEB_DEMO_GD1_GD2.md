# Web demo GĐ1/GĐ2: nguồn số liệu và cách chạy

## Khởi động

Tại thư mục gốc repo, với môi trường Python 3.11 đã cài `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m web.app
```

Mở `http://127.0.0.1:8088`. Web chỉ nghe trên localhost. Một lệnh này khởi động API và giao diện; khi bấm **Bắt đầu** ở bước 5 (Huấn luyện thử), web tự tạo một Flower server (`0.0.0.0:8089`, probe RTT `8090`) và đúng năm tiến trình client (`127.0.0.1:8089`). Nút **Dừng tiến trình** dừng toàn bộ tiến trình do phiên web tạo.

## Cấu trúc giao diện: sáu bước

Menu bên trái đi theo đúng trình tự trình bày; mỗi bước có địa chỉ `#<bước>` nên mở thẳng được một bước và tải lại vẫn đúng bước.

| Bước | Địa chỉ | Việc của người trình bày | Dữ liệu |
|---|---|---|---|
| 1 Tổng quan | `#tong-quan` | Nêu câu hỏi, ba trụ cột (Centralized / FedAvg / Local-only), kết quả chính và độ tin cậy | `/api/demo/evidence`, `/api/evidence/summary` |
| 2 Dữ liệu non-IID | `#non-iid` | Cho thấy mỗi client giữ phần nào của 38 lớp: heatmap số ảnh train và recall Local-only | `/api/evidence/summary` (shard, tái đánh giá) |
| 3 Kết quả chi tiết | `#ket-qua` | Bảng 7 mô hình; PlantDoc theo lớp; hiệu chuẩn xác suất (ECE) | `/api/evidence/summary` |
| 4 Chẩn đoán bệnh lá | `#chan-doan` | Chọn ảnh mẫu (dễ → khó) hoặc tải ảnh; một ảnh chạy cả 7 mô hình, hiện đúng/sai và số ảnh train của lớp đó | `/api/inference` (6 lần), `samples/demo/catalog.json` |
| 5 Huấn luyện thử | `#huan-luyen` | Chạy Flower 5 client, theo dõi vòng, RTT, audit | `/api/federated/train`, `/api/status`, `/api/federated/network-metrics` |
| 6 Bằng chứng và hạn chế | `#bang-chung` | Checklist nghiệm thu, danh tính artifact, tái đánh giá, audit rò rỉ, hạn chế | `/api/evidence/summary` |

**Chế độ trình chiếu** (nút ở menu hoặc phím `P`): chữ và số to hơn, ẩn mục kỹ thuật (`.tech-only`), dải bước ở đầu trang; phím `←` `→` chuyển bước, `Esc` thoát. Lựa chọn được nhớ trong trình duyệt.

`GET /api/evidence/summary` chỉ **đọc** các báo cáo đã sinh (`local_reeval_report.json`, `plantdoc_per_class_report.json`, `plantdoc_domain_gap_report.json`, `release_leakage_audit.json`, `client_shard_class_counts.json`, `status_report.json`). Mỗi phần fail-closed riêng: báo cáo thiếu, hỏng hoặc lệch danh tính bundle (protocol, hash release/validation, SHA-256 checkpoint, số công bố) thì phần đó trả `available:false` kèm lý do và giao diện nói rõ chưa có, không hiện số nửa vời. Toàn bộ phần có số liệu mô hình bị khóa khi cổng registry chặn bộ đối chứng.

Ảnh mẫu của bước 4 do `scripts/build_demo_gallery.py` dựng từ `data/demo_images/` (gitignored) vào `web/static/samples/demo/`: kiểm sha256, nhãn thuộc 38 lớp và (nếu có release trên đĩa) ảnh thuộc validation, không thuộc train. Server không có dataset nên dùng catalog đã sinh đi theo git.

## Trình tự demo trước hội đồng

1. **Tổng quan (bước 1):** nêu câu hỏi nghiên cứu; chỉ ba thẻ trụ cột và biểu đồ bảy mô hình. Các chip độ tin cậy ở cuối trang dẫn tới bước 6.
2. **Dữ liệu non-IID (bước 2):** giải thích α=0,1 rồi chỉ heatmap 38 lớp × 5 client. Chuyển sang "Recall PlantVillage" để thấy ô không có ảnh luôn có recall 0%. Không diễn giải trung bình Local-only thành một checkpoint.
3. **Kết quả chi tiết (bước 3):** bảng bảy mô hình (GĐ1: Centralized v6 và năm Local-only v7; GĐ2: FedAvg v7 với Dirichlet α=0,1, seed 42; đây là **một** kịch bản non-IID đã công bố, chưa đủ để tuyên bố hoàn tất nhiều α/nhiều seed). Tab "PlantDoc theo lớp" cho thấy khoảng cách miền; tab "Độ tin cậy xác suất" nói rõ softmax chưa hiệu chuẩn.
4. **Chẩn đoán bệnh lá (bước 4):** bấm một ảnh mẫu (chọn "Vừa" hoặc "Khó" để có cả mô hình đúng lẫn sai); bảng bảy mô hình hiện dần, kèm câu "Vì sao?". Các mô hình dùng checkpoint SHA-256 đã kiểm và resize trực tiếp 224×224 `canonical_v1`. Bộ phân loại nhận ảnh **vùng lá đã cắt**; softmax của một ảnh không phải độ chính xác ngoài thực địa.
5. **Huấn luyện thử (bước 5):** để loại FedAvg qua Flower; web cố định 5 client, 2 vòng, 1 local epoch, smoke. Mỗi client dùng tối đa một ảnh tổng hợp cho mỗi lớp có trong shard để chạy được trên EC2 ít RAM. Bấm Bắt đầu và theo dõi sơ đồ client, vòng hiện tại, RTT localhost, log audit. Phiên này dùng model scratch và partition PlantVillage cũ; không tái tạo checkpoint pretrained/mixed-data từ các nhánh công bố, và không phải kết quả khoa học.
6. **Bằng chứng và hạn chế (bước 6):** dùng khi hội đồng hỏi "làm sao biết số này đúng": checklist 17/20, danh tính artifact, tái đánh giá, audit rò rỉ, danh sách hạn chế.

Nếu artifact thiếu, sai hash, khác class map hoặc khác protocol/split, web ẩn toàn bộ số liệu và khóa chọn checkpoint. Không có fallback sang số liệu scratch cũ.

### Cổng registry cho inference

Mọi dự đoán đi qua một cửa `web/services/model_service.py` và chỉ chạy được với model có `web_visible=true` trong
`artifacts/registry.json` (đủ 5 điều kiện: hash, taxonomy, preprocessing, metrics↔protocol, bài forward — xem
[MODEL_REGISTRY.md](MODEL_REGISTRY.md)). `GET /api/models` liệt kê các model được phép kèm nhãn (Demo / Tham chiếu) và
tóm tắt PlantVillage/PlantDoc công bố. Model bị chặn trả HTTP 503 kèm lý do; registry chưa sinh cũng trả 503 (không
fallback). Sau khi đổi checkpoint hoặc bundle, chạy `python scripts/update_status.py` để cổng tính lại.

## Nguồn đối chứng đã ghim

| Phương pháp | Nhánh | Commit |
|---|---|---|
| Centralized v6 | `train-centralized-v6` | `2294de21c93e28d611013a75413995a128b303b7` |
| FedAvg v7 | `train-fedavg-non-iid-v7` | `b37488fbaf99d94cc0d66e66a9649c99dd9e1257` |
| 5 Local-only v7 | `train-local-only-a01-v7` | `6de4c6b99805399673dc0a41ab97aeefd7099943` |

Bundle đã nhập nằm ở `artifacts/branch_demo/20261003/`. `import_manifest.json` ghi seed, 38 lớp, transform, hash release/validation/W0 và commit. Mọi file trong `artifact_manifest.json` được kiểm SHA-256 mỗi khi web nạp số liệu hoặc checkpoint. Chỉ xác nhận **tính toàn vẹn artifact đã công bố**; ảnh gốc `dataset/mixed/pv_pd_v3`, manifest ảnh và W0 không có ở máy chủ demo, nên máy chủ không tái chạy phép đo hay huấn luyện ảnh thật. Việc tái đánh giá và audit rò rỉ đã chạy ở máy phát triển có dataset (`scripts/evaluate_published_models.py`, `scripts/audit_release_leakage.py`); web chỉ đọc báo cáo của chúng qua `/api/evidence/summary` và đối chiếu hash với bundle.

Trong checkout mới, khi đích `artifacts/branch_demo/20261003/` chưa tồn tại, có thể nhập lại từ ba clone GitHub sạch. Đặt các clone lần lượt ở `.tmp-agent/branch-demo-import/sources/{centralized,fedavg,local}`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts\import_branch_demo.py
```

Importer từ chối commit, SHA-256, seed, W0, split, class map hoặc checkpoint không đúng và không ghi đè bundle đã có. Sau khi nhập, xóa các clone tạm trong `.tmp-agent/branch-demo-import/`; bundle trong `artifacts/branch_demo/20261003/` là bản dùng cho demo.

## Log và giới hạn nghiệm thu

Mỗi lần bấm Start tạo `artifacts/web_runs/network_smoke/<UTC timestamp>_<id>/` gồm `session.json`, `server.log`, năm `client_XX.log`, `progress.json`, `history.json` và `global.pt` nếu cả hai vòng hoàn tất. `session.json` ghi `scientific_valid=false` và `published_result_reproduction=false`. Run qua chính API web đã kiểm có 5 client hợp lệ trong mỗi vòng: `20261004T183127Z_f6f49deb` (thời gian thư mục là UTC).

Sau khi web khởi động lại, bảng Flower đọc phiên smoke **hoàn tất gần nhất** nếu metadata, hai vòng audit và checkpoint đều có đủ. Giao diện gắn nhãn lịch sử, không trình bày là phiên đang chạy. Bản triển khai EC2 `20261005T015850Z_267db2e9` đã hoàn tất hai vòng, mỗi vòng 5/5 update hợp lệ; chỉ dùng tối đa một ảnh tổng hợp mỗi lớp/shard.

Không dùng accuracy từ smoke làm kết quả khoa học. Kết quả validation của ba nhánh không xác nhận hiệu năng trên ảnh người dùng tải lên. Bộ phân loại nhận ảnh **vùng lá đã cắt**; không tìm lá trên ảnh toàn cây. Chế độ huấn luyện ảnh thật trên web đang khóa cho đến khi có release mixed-data, manifest và W0 được kiểm hash. Các thí nghiệm nhiều α, đối chứng phù hợp và nhiều seed là công việc riêng sau mốc demo.

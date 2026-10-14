# Web demo GĐ1/GĐ2: nguồn số liệu và cách chạy

## Khởi động

Tại thư mục gốc repo, với môi trường Python 3.11 đã cài `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m web.app
```

Mở `http://127.0.0.1:8088`. Web chỉ nghe trên localhost. Một lệnh này khởi động API và giao diện; khi bấm **Bắt đầu** ở bước 5 (Huấn luyện thử), web tự tạo một Flower server (`0.0.0.0:8089`, probe RTT `8090`) và đúng năm tiến trình client (`127.0.0.1:8089`). Nút **Dừng tiến trình** dừng toàn bộ tiến trình do phiên web tạo.

## Cấu trúc giao diện: sáu bước

Thanh sáu bước ở đầu trang đi theo đúng trình tự trình bày (bước đã qua được tô xanh, vạch gạch chân chạy theo bước hiện tại; trên màn hẹp thanh này cuộn ngang); mỗi bước có địa chỉ `#<bước>` nên mở thẳng được một bước và tải lại vẫn đúng bước.

| Bước | Địa chỉ | Việc của người trình bày | Dữ liệu |
|---|---|---|---|
| 1 Tổng quan — *Tổng quan đồ án* (câu mở: “Cùng huấn luyện. Ảnh vẫn ở từng nơi.”) | `#tong-quan` | Sơ đồ FedAvg chuyển động một lượt mỗi lần vào bước (client hiện lần lượt, trọng số chạy về máy chủ, mô hình chung hiện ra; nút “Xem lại” chạy lại; tắt khi giảm chuyển động), nêu kết quả chính, giải thích ba bước của FedAvg, so ba cách huấn luyện (Centralized / FedAvg / Local-only) và độ tin cậy của số liệu | `/api/demo/evidence`, `/api/evidence/summary` |
| 2 Dữ liệu phân tán — *Vì sao mỗi nơi tự học là không đủ?* | `#non-iid` | Cho thấy mỗi client giữ phần nào của 38 loại bệnh: heatmap số ảnh và tỉ lệ nhận đúng của mô hình học riêng; mục gập "12 cách chia dữ liệu" (chỉ α=0,1 có mô hình) | `/api/evidence/summary`, `/api/evidence/partitions` |
| 3 Kết quả & so sánh — *Kết quả có đáng tin không?* | `#ket-qua` | Sáu tab: Tổng hợp · Tập giữ riêng & thống kê (khoảng tin cậy, McNemar) · Ảnh ngoài đồng theo bệnh · Ma trận nhầm lẫn · Sai mà rất tự tin · Độ tin cậy xác suất | `/api/evidence/summary`, `/api/evidence/details` |
| 4 Chẩn đoán bệnh lá — *Thử chẩn đoán một chiếc lá* | `#chan-doan` | Chọn ảnh mẫu (dễ → khó) hoặc tải ảnh; một lần gọi chạy cả 7 mô hình, hiện đúng/sai, xác suất gốc + đã hiệu chuẩn, số ảnh bệnh đó mô hình đã học, tùy chọn Grad-CAM; có nút in | `/api/inference/all` (máy chủ cũ: `/api/inference` 6 lần), `samples/demo/catalog.json` |
| 5 Huấn luyện thử — *Xem FedAvg chạy từng vòng* | `#huan-luyen` | Chạy Flower 5 client, theo dõi vòng, RTT, nhật ký từng client | `/api/federated/train`, `/api/status`, `/api/federated/network-metrics` |
| 6 Bằng chứng, hạn chế và hướng phát triển | `#bang-chung` | Thứ tự: đã kiểm chứng gì → hạn chế → việc còn lại; mã băm và lệnh mở khóa nằm trong mục gập "chi tiết kỹ thuật" | `/api/evidence/summary` |

Nút **Nền tối/Nền sáng** ở góc phải thanh trên cùng đổi giao diện (nhớ lựa chọn; khi in luôn dùng nền sáng). Phong cách: nền ngà có lưới chấm mờ, tiêu đề chữ có chân (Source Serif 4), màu theo trụ cột — Centralized xanh dương, FedAvg xanh lá, Local-only cam.

`GET /api/evidence/summary` chỉ **đọc** các báo cáo đã sinh (`local_reeval_report.json`, `plantdoc_per_class_report.json`, `plantdoc_domain_gap_report.json`, `release_leakage_audit.json`, `client_shard_class_counts.json`, `status_report.json`, và `fullframe_eval.json` do `scripts/eval_fullframe_plantdoc.py` sinh: độ chính xác khi đưa ảnh PlantDoc nguyên khung so với ảnh cắt lá, hiện ở bước 6 mục hạn chế). Mỗi phần fail-closed riêng: báo cáo thiếu, hỏng hoặc lệch danh tính bundle (protocol, hash release/validation, SHA-256 checkpoint, số công bố) thì phần đó trả `available:false` kèm lý do và giao diện nói rõ chưa có, không hiện số nửa vời. Toàn bộ phần có số liệu mô hình bị khóa khi cổng registry chặn bộ đối chứng.

Ảnh mẫu của bước 4 do `scripts/build_demo_gallery.py` dựng từ `data/demo_images/` (gitignored) vào `web/static/samples/demo/`: kiểm sha256, nhãn thuộc 38 lớp và (nếu có release trên đĩa) ảnh thuộc validation, không thuộc train. Server không có dataset nên dùng catalog đã sinh đi theo git.

## Trình tự demo trước hội đồng

Khoảng 3 phút cho phần chính (bước 1, 2, 4); các tab của bước 3 và bước 6 mở khi hội đồng hỏi sâu.

1. **Tổng quan (bước 1):** đọc đề tài ở khung đầu, chỉ ba bước FedAvg, rồi ba thẻ Centralized / FedAvg / Local-only (99% — 93% — 45–55% trên ảnh chụp trong phòng). Các chip "Các con số này đáng tin tới đâu?" dẫn tới bước 6.
2. **Dữ liệu phân tán (bước 2):** nói mỗi nơi có số ảnh rất chênh giữa các nhóm (nhiều nhóm gần như chỉ nằm ở một nơi), rồi chỉ heatmap 38 nhóm × 5 client. Chuyển sang "Tỉ lệ nhận đúng" để thấy ô không có ảnh thì mô hình học riêng nhận đúng 0%. Không diễn giải trung bình Local-only thành một checkpoint.
3. **Kết quả & so sánh (bước 3):** bảng bảy mô hình (GĐ1: Centralized v6 và năm Local-only v7; GĐ2: FedAvg v7, α=0,1, seed 42; đây là **một** kịch bản đã công bố). Tab "Tập giữ riêng & thống kê" trả lời "số này có bị đẹp hơn thật không"; tab "Độ tin cậy xác suất" nói rõ softmax gốc tự tin quá mức trên ảnh ngoài đồng.
4. **Chẩn đoán bệnh lá (bước 4):** bấm một ảnh mẫu (chọn "Vừa" hoặc "Khó" để có cả mô hình đúng lẫn sai, ví dụ ảnh #12 Tomato — Bacterial spot); bảy mô hình trả lời cùng lúc, kèm câu "Vì sao?". Bật Grad-CAM để thấy vùng mô hình nhìn (gợi ý trực quan, không phải bằng chứng nhân quả). Các mô hình dùng checkpoint đã kiểm SHA-256 và resize 224×224 (`canonical_v1`). Bộ phân loại nhận ảnh **vùng lá đã cắt**; kết quả một ảnh không phải độ chính xác ngoài thực địa.
5. **Huấn luyện thử (bước 5):** để loại FedAvg qua Flower; web cố định 5 client, 2 vòng, 1 local epoch, smoke. Mỗi client dùng tối đa một ảnh tổng hợp cho mỗi lớp có trong shard để chạy được trên EC2 ít RAM. Bấm Bắt đầu và theo dõi sơ đồ client, vòng hiện tại, RTT localhost, log audit. Phiên này dùng model scratch và partition PlantVillage cũ; không tái tạo checkpoint pretrained/mixed-data từ các nhánh công bố, và không phải kết quả khoa học. Phía trên phần Flower có khối **“Tham gia từ máy của bạn”**: người có mã tham gia huấn luyện lớp cuối ngay trong trình duyệt trên ảnh của mình và gửi trọng số về máy chủ để gộp theo vòng; ảnh không rời máy. Khối này chỉ hoạt động khi máy chủ có artifact và biến môi trường của tính năng — xem [COMMUNITY_FL.md](COMMUNITY_FL.md).
6. **Bằng chứng, hạn chế và hướng phát triển (bước 6):** dùng khi hội đồng hỏi "làm sao biết số này đúng" hoặc "đồ án còn thiếu gì": những gì đã kiểm chứng (chấm lại, tập giữ riêng, kiểm tra trùng ảnh), danh sách hạn chế nêu thẳng, việc còn lại cần GPU. Mã băm SHA-256 và lệnh mở khóa nằm trong mục gập "chi tiết kỹ thuật" (ẩn khi trình chiếu).

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

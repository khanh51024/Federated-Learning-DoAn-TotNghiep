# Web demo GĐ1/GĐ2: nguồn số liệu và cách chạy

## Khởi động

Tại thư mục gốc repo, với môi trường Python 3.11 đã cài `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m web.app
```

Mở `http://127.0.0.1:8088`. Web chỉ nghe trên localhost. Một lệnh này khởi động API và giao diện; khi bấm **Huấn Luyện FedAvg**, web tự tạo một Flower server (`0.0.0.0:8089`, probe RTT `8090`) và đúng năm tiến trình client (`127.0.0.1:8089`). Nút **Stop** dừng toàn bộ tiến trình do phiên web tạo.

## Trình tự demo trước hội đồng

1. **Tổng quan / Đánh giá:** chờ huy hiệu `SHA-256 ĐÃ ĐỐI SOÁT`. So sánh bảy checkpoint trên cùng validation: Centralized v6, FedAvg v7 và năm Local-only v7. PlantDoc (269 ảnh) và PlantVillage (4.344 ảnh) hiển thị riêng.
2. **GĐ1:** chỉ ra Centralized và năm mô hình Local-only độc lập. Không diễn giải trung bình Local-only thành một checkpoint.
3. **GĐ2:** chỉ ra FedAvg v7 với phân hoạch nhãn Dirichlet α=0,1, seed 42. Đây là **một** kịch bản non-IID đã công bố, chưa đủ để tuyên bố hoàn tất nghiên cứu nhiều α/nhiều seed.
4. **Huấn Luyện FedAvg:** để engine `network`; web cố định 5 client, 2 vòng, 1 local epoch, smoke. Mỗi client dùng tối đa một ảnh tổng hợp cho mỗi lớp có trong shard để chạy được trên EC2 ít RAM. Bấm Start và theo dõi client đã kết nối, vòng hiện tại, update được nhận, RTT localhost, log audit. Phiên này dùng model scratch và partition PlantVillage cũ; không tái tạo checkpoint pretrained/mixed-data từ các nhánh công bố.
5. **Chẩn đoán bệnh lá:** chọn ảnh vùng lá minh họa hoặc tải ảnh vùng lá riêng, chọn FedAvg hay một Local-only. Hai cột suy luận dùng checkpoint SHA-256 đã kiểm và resize trực tiếp 224×224 `canonical_v1`. Ảnh minh họa không có nhãn bệnh kiểm chứng. Softmax của một ảnh không phải độ chính xác ngoài thực địa.

Nếu artifact thiếu, sai hash, khác class map hoặc khác protocol/split, web ẩn toàn bộ số liệu và khóa chọn checkpoint. Không có fallback sang số liệu scratch cũ.

## Nguồn đối chứng đã ghim

| Phương pháp | Nhánh | Commit |
|---|---|---|
| Centralized v6 | `train-centralized-v6` | `2294de21c93e28d611013a75413995a128b303b7` |
| FedAvg v7 | `train-fedavg-non-iid-v7` | `b37488fbaf99d94cc0d66e66a9649c99dd9e1257` |
| 5 Local-only v7 | `train-local-only-a01-v7` | `6de4c6b99805399673dc0a41ab97aeefd7099943` |

Bundle đã nhập nằm ở `artifacts/branch_demo/20261003/`. `import_manifest.json` ghi seed, 38 lớp, transform, hash release/validation/W0 và commit. Mọi file trong `artifact_manifest.json` được kiểm SHA-256 mỗi khi web nạp số liệu hoặc checkpoint. Chỉ xác nhận **tính toàn vẹn artifact đã công bố**; ảnh gốc `dataset/mixed/pv_pd_v3`, manifest ảnh và W0 chưa có ở máy demo, nên chưa tái chạy phép đo hoặc huấn luyện ảnh thật.

Trong checkout mới, khi đích `artifacts/branch_demo/20261003/` chưa tồn tại, có thể nhập lại từ ba clone GitHub sạch. Đặt các clone lần lượt ở `.tmp-agent/branch-demo-import/sources/{centralized,fedavg,local}`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts\import_branch_demo.py
```

Importer từ chối commit, SHA-256, seed, W0, split, class map hoặc checkpoint không đúng và không ghi đè bundle đã có. Sau khi nhập, xóa các clone tạm trong `.tmp-agent/branch-demo-import/`; bundle trong `artifacts/branch_demo/20261003/` là bản dùng cho demo.

## Log và giới hạn nghiệm thu

Mỗi lần bấm Start tạo `artifacts/web_runs/network_smoke/<UTC timestamp>_<id>/` gồm `session.json`, `server.log`, năm `client_XX.log`, `progress.json`, `history.json` và `global.pt` nếu cả hai vòng hoàn tất. `session.json` ghi `scientific_valid=false` và `published_result_reproduction=false`. Run qua chính API web đã kiểm có 5 client hợp lệ trong mỗi vòng: `20261004T183127Z_f6f49deb` (thời gian thư mục là UTC).

Sau khi web khởi động lại, bảng Flower đọc phiên smoke **hoàn tất gần nhất** nếu metadata, hai vòng audit và checkpoint đều có đủ. Giao diện gắn nhãn lịch sử, không trình bày là phiên đang chạy. Bản triển khai EC2 `20261005T015850Z_267db2e9` đã hoàn tất hai vòng, mỗi vòng 5/5 update hợp lệ; chỉ dùng tối đa một ảnh tổng hợp mỗi lớp/shard.

Không dùng accuracy từ smoke làm kết quả khoa học. Kết quả validation của ba nhánh không xác nhận hiệu năng trên ảnh người dùng tải lên. Bộ phân loại nhận ảnh **vùng lá đã cắt**; không tìm lá trên ảnh toàn cây. Chế độ ảnh thật trên web đang khóa cho đến khi có release mixed-data, manifest và W0 được kiểm hash. Các thí nghiệm nhiều α, đối chứng phù hợp và nhiều seed là công việc riêng sau mốc demo.

# Web demo GĐ1/GĐ2: nguồn số liệu và cách chạy

## Khởi động

Tại thư mục gốc repo, với môi trường Python 3.11 đã cài `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m web.app
```

Mở `http://127.0.0.1:8088`. Web chỉ nghe trên localhost. Một lệnh này khởi động API và giao diện; khi bấm **Bắt đầu** ở Phụ lục kỹ thuật (mục D, cần đăng nhập quản trị), web tự tạo một Flower server (`0.0.0.0:8089`, probe RTT `8090`) và đúng năm tiến trình client (`127.0.0.1:8089`). Nút **Dừng tiến trình** dừng toàn bộ tiến trình do phiên web tạo.

## Cấu trúc giao diện: báo cáo năm chương, hai trang demo, phụ lục

Web đọc như một báo cáo đồ án chứ không phải một chuỗi bước trình diễn: thanh trên cùng có hai nhóm, **Báo cáo** (năm chương đọc theo thứ tự, chương đã qua được tô xanh) và **Demo** (hai trang độc lập, không đánh số). Phần kiểm chứng kỹ thuật và phần chạy thử hệ thống trên máy chủ nằm ở **Phụ lục kỹ thuật**, không có trên thanh menu, chỉ vào từ liên kết ở chân trang. Mỗi trang có địa chỉ `#<trang>`; địa chỉ cũ của sáu bước (`#tong-quan`, `#non-iid`, `#huan-luyen`, `#bang-chung`) vẫn mở đúng trang mới (router đổi sang địa chỉ mới, mã tham gia đã in dạng `/?code=…#huan-luyen` vẫn dùng được).

| Trang | Địa chỉ | Nội dung | Dữ liệu |
|---|---|---|---|
| Chương 1 · Giới thiệu | `#gioi-thieu` | Bài toán, ba câu hỏi nghiên cứu, kết quả chính (một dải ba số), mục lục báo cáo; sơ đồ FedAvg chuyển động một lượt mỗi lần vào trang (nút “Xem lại”; tắt khi giảm chuyển động) | `/api/demo/evidence`, `/static/samples/demo/catalog.json` |
| Chương 2 · Dữ liệu | `#du-lieu` | Hai nguồn ảnh; **nơi duy nhất định nghĩa tập học, tập kiểm tra, tập giữ riêng**; dữ liệu lệch giữa các client (non-IID): ví dụ, năm thẻ client, bảng 38 nhóm × 5 client (gấp), 12 cách chia (gấp) | `/api/evidence/summary`, `/api/evidence/partitions` |
| Chương 3 · Phương pháp | `#phuong-phap` | Ba cách huấn luyện, thuật toán FedAvg (bốn bước và công thức), thiết lập thí nghiệm, cách đánh giá (Top-1, Macro-F1, khoảng tin cậy, McNemar, ECE), hệ thống demo. Chỉ dùng số có trong bộ số liệu công bố | `/api/demo/evidence` |
| Chương 4 · Kết quả | `#ket-qua` | Bốn tab: Tổng hợp (ba số, biểu đồ và bảng bảy mô hình hiện sẵn, lịch sử Centralized gấp) · Tập giữ riêng và thống kê · Phân tích lỗi (nhóm bệnh PlantDoc, ma trận nhầm lẫn, dự đoán sai tự tin) · Độ tin cậy xác suất | `/api/evidence/summary`, `/api/evidence/details` |
| Chương 5 · Kết luận | `#ket-luan` | Trả lời ba câu hỏi nghiên cứu (số tính từ cùng bộ số liệu), hạn chế (danh sách đầy đủ hiện sẵn), hướng phát triển | `/api/demo/evidence`, `/api/evidence/summary` |
| Demo · Chẩn đoán ảnh | `#chan-doan` | Chọn ảnh mẫu hoặc tải ảnh; một nút chạy **ba cách nhận diện ngang hàng** (tab Local only · PlantVillage · Cộng đồng, mỗi tab có dấu đúng/sai); nút “Tổng hợp 7 mô hình” mở bảng Centralized / FedAvg / Local-only, Grad-CAM tùy chọn, có nút in | `/api/inference/all` (bảy mô hình), `/api/inference/plantvillage` (mô hình ngoài, đã đánh giá lại ở `artifacts/plantvillage_reference_eval.json`), `/api/community/predict` (phiên bản cộng đồng mới nhất) |
| Demo · Học phân tán | `#hoc-lien-ket` | Client học phân tán cộng đồng chạy trong trình duyệt (xem [COMMUNITY_FL.md](COMMUNITY_FL.md)) | `/api/community/*` |
| Phụ lục kỹ thuật | `#phu-luc` | A kiểm chứng số liệu (đối chiếu, tập giữ riêng, trùng ảnh) · B nguồn gốc mô hình (SHA-256, chấm lại) · C hạng mục kỹ thuật (kèm lệnh vận hành) · D kiểm thử hệ thống trên máy chủ: Flower 5 client hoặc Centralized trên ảnh nhiễu, chỉ để kiểm chứng điều phối, không đo độ chính xác · E trọng số mô hình và cách chạy | `/api/evidence/summary`, `/api/federated/train`, `/api/status`, `/api/federated/network-metrics`, `/api/weights/inspect` |

Nút **Nền tối/Nền sáng** ở góc phải thanh trên cùng đổi giao diện (nhớ lựa chọn; khi in luôn dùng nền sáng). Phong cách: nền ngà có lưới chấm mờ, tiêu đề chữ có chân (Source Serif 4), màu theo trụ cột — Centralized xanh dương, FedAvg xanh lá, Local-only cam.

`GET /api/evidence/summary` chỉ **đọc** các báo cáo đã sinh (`local_reeval_report.json`, `plantdoc_per_class_report.json`, `plantdoc_domain_gap_report.json`, `release_leakage_audit.json`, `client_shard_class_counts.json`, `status_report.json`, và `fullframe_eval.json` do `scripts/eval_fullframe_plantdoc.py` sinh: độ chính xác khi đưa ảnh PlantDoc nguyên khung so với ảnh cắt lá, hiện ở chương 5 mục hạn chế). Mỗi phần fail-closed riêng: báo cáo thiếu, hỏng hoặc lệch danh tính bundle (protocol, hash release/validation, SHA-256 checkpoint, số công bố) thì phần đó trả `available:false` kèm lý do và giao diện nói rõ chưa có, không hiện số nửa vời. Toàn bộ phần có số liệu mô hình bị khóa khi cổng registry chặn bộ đối chứng.

Ảnh mẫu của trang Chẩn đoán do `scripts/build_demo_gallery.py` dựng từ `data/demo_images/` (gitignored) vào `web/static/samples/demo/`: kiểm sha256, nhãn thuộc 38 lớp và (nếu có release trên đĩa) ảnh thuộc validation, không thuộc train. Server không có dataset nên dùng catalog đã sinh đi theo git.

## Trình tự demo trước hội đồng

Khoảng 3 phút cho phần chính (chương 1, chương 2, Demo chẩn đoán); chương 3 đến 5 và Phụ lục mở khi hội đồng hỏi sâu.

1. **Giới thiệu (chương 1):** đọc đề tài ở khung đầu, chỉ sơ đồ FedAvg, đọc ba câu hỏi nghiên cứu rồi dải kết quả chính (99% — 93% — 45–55% trên ảnh chụp trong phòng).
2. **Dữ liệu (chương 2):** nói ba tập ảnh và mục đích từng tập, rồi mỗi nơi có số ảnh rất chênh giữa các nhóm (nhiều nhóm gần như chỉ nằm ở một nơi), chỉ ví dụ Tomato — Bacterial spot và bảng 38 nhóm × 5 client. Chuyển sang "Tỉ lệ nhận đúng" để thấy ô không có ảnh thì mô hình học riêng nhận đúng 0%. Không diễn giải trung bình Local-only thành một checkpoint.
3. **Kết quả (chương 4):** biểu đồ và bảng bảy mô hình (GĐ1: Centralized v6 và năm Local-only v7; GĐ2: FedAvg v7, α=0,1, seed 42; đây là **một** kịch bản đã công bố). Tab "Tập giữ riêng và thống kê" trả lời "số này có bị đẹp hơn thật không"; tab "Độ tin cậy xác suất" nói rõ softmax gốc tự tin quá mức trên ảnh ngoài đồng.
4. **Demo chẩn đoán:** bấm một ảnh mẫu (chọn "Nhận đúng trung bình" hoặc "thấp" để có cả mô hình đúng lẫn sai, ví dụ ảnh #12 Tomato — Bacterial spot); bảy mô hình trả lời cùng lúc, kèm câu "Vì sao?". Bật Grad-CAM để thấy vùng mô hình nhìn (gợi ý trực quan, không phải bằng chứng nhân quả). Các mô hình dùng checkpoint đã kiểm SHA-256 và resize 224×224 (`canonical_v1`). Bộ phân loại nhận ảnh **vùng lá đã cắt**; kết quả một ảnh không phải độ chính xác ngoài thực địa.
5. **Demo học phân tán:** người có mã tham gia huấn luyện lớp cuối ngay trong trình duyệt trên ảnh của mình và gửi trọng số về máy chủ để gộp theo vòng; ảnh không rời máy. Chỉ hoạt động khi máy chủ có artifact và biến môi trường của tính năng, và người quản trị đã mở phiên — xem [COMMUNITY_FL.md](COMMUNITY_FL.md). Đây là bằng chứng học thật rõ nhất khi trình bày.
6. **Kết luận và hạn chế (chương 5):** trả lời ba câu hỏi, đọc danh sách hạn chế nêu thẳng, nói hướng phát triển.
7. **Phụ lục kỹ thuật** (liên kết ở chân trang): dùng khi hội đồng hỏi "làm sao biết số này đúng" (mục A, B: chấm lại, tập giữ riêng, kiểm tra trùng ảnh, mã băm SHA-256) hoặc "hệ thống có chạy thật không" (mục D). Mục D chạy FedAvg qua Flower: web cố định 5 client, 2 vòng, 1 local epoch, smoke. Máy chủ demo không có thư mục ảnh nên mỗi client học trên ảnh nhiễu ngẫu nhiên (tối đa một ảnh cho mỗi lớp có trong shard, để chạy được trên máy ít RAM): phần này kiểm chứng hệ thống, không đo độ chính xác, và độ chính xác đứng ở mức đoán ngẫu nhiên 1/38 là đúng kỳ vọng. Khung "Bằng chứng máy đã tính thật" hiện số tiến trình, vòng đã gộp và chuỗi mã băm mô hình chung (xem [REVIEW_WEB_2026-10-19.md](REVIEW_WEB_2026-10-19.md)). Phiên này dùng model scratch và partition PlantVillage cũ; không tái tạo checkpoint pretrained/mixed-data từ các nhánh công bố, và không phải kết quả khoa học.

Nếu artifact thiếu, sai hash, khác class map hoặc khác protocol/split, web ẩn toàn bộ số liệu và khóa chọn checkpoint. Không có fallback sang số liệu scratch cũ.

### Quyền và giới hạn của các nút huấn luyện (Phụ lục kỹ thuật, mục D)

Ba API `POST /api/federated/train`, `POST /api/centralized/train` và `POST /api/train/stop` chạy việc nặng trong cùng tiến trình web,
nên trên máy chủ công khai chúng được bảo vệ ở **backend** (không chỉ khóa nút):

- **Đăng nhập quản trị.** Khi biến `COMMUNITY_ADMIN_TOKEN` được đặt (máy chủ công khai luôn đặt), ba API đòi tiêu đề `Authorization`
  (Basic `admin:<mật khẩu>` hoặc `Bearer <mật khẩu>`); thiếu hoặc sai trả 401, sai nhiều lần trả 429 (chung bộ đếm với `/api/community/admin/*`).
  Khi biến này **không** đặt (chạy web trên máy dev) các API mở như trước. Người xem vẫn thấy tiến trình, bảng và lịch sử; giao diện hiện
  khung "Cần quyền quản trị" kèm liên kết `/quan-tri`, và tự mở khóa khi đăng nhập ở tab quản trị (có "Ghi nhớ trên máy này").
- **Một tác vụ một lúc.** Khối kiểm trạng thái và đặt `running` nằm trong một khóa; yêu cầu thứ hai trong lúc đang chạy nhận **409**.
- **Trần tham số**, trả 422 nếu vượt, và `/api/status` nêu các trần để biểu mẫu đặt `max`: `TRAIN_MAX_CENTRALIZED_EPOCHS` (10),
  `TRAIN_MAX_ROUNDS` (10), `TRAIN_MAX_LOCAL_EPOCHS` (3), `TRAIN_MAX_SECONDS` (900, quá giờ thì tự dừng), `TRAIN_STARTS_PER_10_MIN` (6 mỗi IP).
- **RAM trống tối thiểu** (503 nếu thiếu): `TRAIN_MIN_AVAILABLE_MB` (500, job trong tiến trình) và `TRAIN_MIN_AVAILABLE_MB_NETWORK` (1400, Flower).
  Đo trên EC2 2 GiB: Centralized 10 epoch 36 s / RSS đỉnh ~1,08 GB; simulation 10 vòng × 3 epoch 149 s / ~1,05 GB; **Flower 5 client chiếm
  ~1,6 GB và làm bản web không trả lời trong 30 giây** (swap nặng) — chỉ chạy khi trình diễn và máy rảnh.
- **Nhật ký** `artifacts/web_runs/training_audit.jsonl`: ai (tài khoản, IP), lúc nào, cấu hình, bắt đầu/dừng/hết giờ/kết thúc và trạng thái.

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

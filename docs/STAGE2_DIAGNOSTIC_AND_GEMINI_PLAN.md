# Chẩn đoán GĐ2 và kế hoạch sửa bằng Gemini 3.8 Flash trong Antigravity

Ngày lập: 2026-09-15

## Kết luận ngắn

Điểm Kaggle gần 99% của bản tương thích GĐ1 không thể dùng để kết luận GĐ2. Audit độc lập cho split cũ cho thấy:

- 54.305 ảnh được chia 39.084 train, 4.328 validation và 10.893 test.
- 73,87% ảnh validation và 73,82% ảnh test chia sẻ `leaf_group` với train; các client cũng chia sẻ hàng nghìn nhóm lá.
- Vì các ảnh của cùng một lá thường là các khung gần trùng nhau, mô hình có thể học nhận dạng ảnh/miền chụp thay vì khả năng tổng quát hóa.
- Split cũ dùng 5 client và AdamW; cấu hình v4 dùng 10 client và SGD. Cả hai cấu hình chính đều dùng pretrained ImageNet. Khác biệt split và protocol không cho phép trừ accuracy trực tiếp.
- Log Kaggle được lưu trong workspace ngày 14/09 không phải một run hợp lệ: notebook không tìm thấy package tại `/kaggle/input/datasets/.../fl-training-partitions`, sau đó dừng ở `FileNotFoundError` khi mở `train_fedavg_kaggle.yaml`. Không lấy log này làm bằng chứng accuracy.

## Sửa đã thực hiện

Đã thêm `content_aware: true` tùy chọn cho partitioner. Khi bật, đơn vị chia là thành phần liên thông của:

1. nhóm lá từ `leaf-map.json`; và
2. nhóm ảnh có cùng SHA-256 byte.

Manifest cũ giữ nguyên. Bộ mới nằm tại `data/partitions_train_v4_content_aware/` và có 9 điều kiện:

- IID;
- label skew với alpha 0,1; 1; 100;
- quantity skew với quantity alpha 0,1; 1;
- label + quantity skew với cả hai 0,1;
- IID + feature skew moderate;
- label alpha 0,1 + feature skew moderate.

Audit lại bộ v4 cho cả 9 điều kiện: train/validation/test = 37.852/5.499/10.954; 0 hash trùng giữa mọi cặp split; 0 hash trùng giữa các client; bảo toàn 54.305 ảnh; `group_integrity.leakage_free=true`.

Các tệp quan trọng:

- `src/data/leaf_groups.py`, `src/data/partitioner.py`: gộp nhóm lá + byte hash.
- `fl_training/prepare.py`: lấy hash từ source cache và cho phép split mới không khóa theo hash test cũ.
- `configs/partition_content_aware_v4.yaml`: cấu hình tái lập.
- `data/partitions_train_v4_content_aware/index.json`: danh mục các manifest.

## Kế hoạch dùng Antigravity/Gemini 3.8 Flash

Chọn model mà người dùng yêu cầu trong catalog thực tế của Antigravity. Phiên kiểm tra code này chưa xác minh sự tồn tại của tên model “Gemini 3.8 Flash”, các mức suy luận hay CLI tương ứng; những lệnh `agy` từng ghi ở bản kế hoạch trước không phải lệnh đã được kiểm chứng. Các prompt bên dưới có thể dán vào phiên làm việc trong ứng dụng. Dùng model để review thay đổi có kiểm soát; chọn cấu hình bằng validation và giữ nguyên số liệu đo được.

### Pha 0 - chuẩn bị đầu vào (15-30 phút)

Đưa vào workspace của Antigravity:

- `gd2_federated_learning/` (source, tests, configs);
- `data/partitions_train_v4_content_aware/index.json` và một partition đại diện alpha=0,1;
- `output/stage2-dirichlet-audit-20260915/audit_evidence.json`;
- PDF đề cương GĐ2;
- log Kaggle và `STAGE2_ACCEPTANCE.md`.

Prompt khởi đầu:

> Đọc toàn bộ source và artifacts được cung cấp. Hãy lập bảng nguyên nhân có bằng chứng, phân biệt lỗi rò rỉ dữ liệu, lỗi protocol, lỗi metric và lỗi môi trường Kaggle. Không sửa code ở lượt này. Trả về danh sách file/dòng, mức P0/P1/P2, phép kiểm chứng và tiêu chí pass.

### Pha 1 - review split và test tự động (low/medium)

Yêu cầu Gemini:

1. kiểm tra `content_aware` có đi qua CLI, `prepare-data`, `partition_config.json`, `fedavg_meta.json` và package upload;
2. kiểm tra 4 bất biến: conservation, path disjointness, group disjointness, byte-hash disjointness;
3. thêm test nhỏ với hai ảnh khác path nhưng cùng bytes, và test không làm thay đổi hành vi khi `content_aware=false`;
4. không thay đổi `REFERENCE_TEST_HASH_V3` của manifest cũ.

Prompt:

> Viết test hồi quy cho split content-aware. Test phải chạy trên fixture nhỏ, kiểm tra SHA-256 trùng không thể rơi vào hai split/client, kiểm tra output deterministic theo seed, và chứng minh chế độ mặc định cũ vẫn tương thích. Nếu đề xuất sửa, chỉ tạo patch nhỏ và giải thích lý do.

### Pha 2 - review loader/model/metric (medium/high)

Yêu cầu Gemini kiểm tra theo thứ tự:

- class mapping 38 lớp giống nhau ở train/val/test;
- clean transform cho val/test, feature profile chỉ áp dụng train;
- không dùng test để early stopping, chọn checkpoint hoặc chỉnh hyperparameter;
- FedAvg weight = `n_k / sum(n_k)` trên đúng client hợp lệ;
- metric dùng cùng denominator và báo accuracy, macro-F1, per-class support;
- validation/test được cố định giữa Centralized, FedAvg và Local-only.

Prompt:

> Trace từ manifest đến DataLoader, train_local, FedAvg aggregate và evaluate. Với mỗi đường dữ liệu, ghi rõ file/dòng, split, transform, seed và denominator. Tìm mọi nơi có thể làm accuracy cao giả tạo hoặc làm FedAvg không còn so sánh được với Centralized. Không tối ưu accuracy.

### Pha 3 - chạy pilot có đối chứng (high)

Chạy theo thứ tự, mỗi job một thư mục output riêng:

1. Centralized trên IID v4;
2. FedAvg trên IID v4;
3. Local-only trên IID v4;
4. lặp 3 mode với label alpha=1 và alpha=0,1;
5. thêm quantity alpha=1, rồi feature moderate.

Giữ cố định model, pretrained policy, batch, optimizer, local epochs, round budget, split seed và preprocessing. Chạy 3 training seed (42, 123, 2026) sau khi một seed pilot hoàn tất. Calibration GPU Kaggle phải chạy trước ma trận; nếu không đủ ngân sách, dừng và giữ trạng thái thiếu thay vì giảm round âm thầm.

Prompt:

> Thực thi đúng các config được chỉ định. Trước mỗi job chạy preflight và ghi fingerprint dataset/split/class/model/config. Sau job chỉ collect metric từ checkpoint best được chọn bằng validation. Nếu thiếu artifact hoặc provenance không khớp, đánh dấu INVALID và không zero-fill.

### Pha 4 - phản biện kết quả (high)

Gemini tạo báo cáo nhưng không được tự tuyên bố thành công. Báo cáo phải có:

- bảng mean/std/CI 95% theo training seed;
- paired gap `Centralized - FedAvg` theo cùng condition/seed, tính bằng điểm phần trăm;
- macro-F1 cạnh accuracy;
- client size CV/Gini, TVD/JS/entropy, số lớp mỗi client;
- confusion matrix và per-class recall với support;
- payload ước lượng và số round tới best/stop;
- cảnh báo nếu test hash, class mapping, model hash hoặc preprocessing khác nhau.

Prompt:

> Phân tích chỉ các artifact hợp lệ. Không ép gap dương, không ép đường alpha đơn điệu, không dùng test để chọn ứng viên. Nếu kết quả FedAvg cao hơn Centralized hoặc alpha=0,1 không suy giảm, kiểm tra provenance, split và metric trước khi diễn giải.

### Pha 5 - đóng gói và bàn giao

Antigravity phải xuất:

- patch source;
- config đã resolve;
- index và manifest v4;
- `audit.json`, `comparison.csv/json`, biểu đồ;
- lệnh tái lập và phiên bản runtime;
- danh sách job thiếu/lỗi;
- kết luận `scientific_stage2_complete` chỉ khi đủ ma trận và seed.

Không commit hoặc xóa artifacts cũ tự động. Người dùng duyệt patch và config trước khi chạy full Kaggle.

## Tiêu chí chấp nhận cuối

1. Mọi split/client không chia sẻ path, leaf-group hoặc byte hash.
2. Tổng mẫu và histogram lớp khớp inventory; val/test có đủ 38 lớp.
3. Centralized/FedAvg/Local-only chấm trên cùng test và cùng class mapping.
4. Kết quả được chọn bằng validation; test chỉ báo cáo một lần sau khóa protocol.
5. Có ít nhất 3 training seed cho các điều kiện chính; CI không được làm giả khi thiếu seed.
6. Có paired gap và macro-F1; không dùng accuracy đơn độc.
7. Kaggle notebook xác nhận đúng dataset/package path trước khi cài dependency và train.
8. Nếu một điều kiện lỗi hoặc thiếu quota, trạng thái báo rõ `MISSING/FAILED/INVALID`, không coi là hoàn tất GĐ2.

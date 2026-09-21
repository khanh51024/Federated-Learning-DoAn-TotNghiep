# Báo Cáo Sửa Chữa GĐ2 v4 — FedAvg-Only Clean Architecture

**Ngày thực hiện**: 20/09/2026  
**Phạm vi**: Khắc phục dứt điểm các lỗ hổng đã xác minh trong Review v3 (`BAO_CAO_REVIEW.md` và `KE_HOACH_SUA_V4_ANTIGRAVITY.md`).  
**Mục tiêu**: Chuẩn hóa toàn diện luồng huấn luyện GĐ2 FedAvg-only từ scratch (`weights=None`, MobileNetV3-Small 38 lớp) với bằng chứng thực nghiệm đầy đủ.

---

## 1. Bảng Đối Chiếu Các Lỗ Hổng v3 và Kết Quả Khắc Phục v4

| STT | Lỗ hổng v3 | File / Function đã sửa | Phương pháp khắc phục | Bằng chứng / Test thực tế |
|---|---|---|---|---|
| 1 | `diagnose()` âm thầm xóa kết quả cũ bằng `shutil.rmtree(output)` | `stage2_scratch/experiment.py:diagnose` | Gỡ bỏ `rmtree`; nếu thư mục tồn tại và không rỗng thì ném `FileExistsError` yêu cầu thư mục mới. | Thao tác kiểm chứng từ chối ghi đè; snapshot 90 file giữ nguyên toàn vẹn. |
| 2 | Preflight bỏ qua kiểm tra dữ liệu thật do nhầm `expected_hashes` thành `observed_hashes` (tautology bypass) | `fl_training/content_audit.py:audit_image_content`, `stage2_matched/data.py:preflight` | Bắt buộc `dataset_root.is_dir()`; đọc và hash SHA-256 từ byte thực tế trên đĩa đúng 1 lần cho cả 8 kịch bản; tách biệt tuyệt đối `observed_hashes` và `expected_hashes`. | `tests/test_strict_preflight_regressions.py` (8/8 PASSED); Strict audit 54.305 ảnh PlantVillage thật đạt 54.284 unique hashes, 0 cross-split duplicates. |
| 3 | Không kiểm tra hash và tính toàn vẹn của `duplicate_review.json` và `visually_verified_pairs.json` trong preflight | `stage2_matched/data.py:preflight` | Thêm kiểm tra bắt buộc tồn tại và khớp SHA-256 với `suite.json`. Kiểm tra trực tiếp 6 cặp visual và 1.058 cặp review (934 accepted + 124 uncertain) không vượt split/client trên cả 8 kịch bản. | `test_preflight_fails_when_duplicate_review_tampered` (PASSED); `test_preflight_fails_when_visually_verified_pairs_missing` (PASSED); `test_preflight_fails_when_verified_pair_crosses_splits` (PASSED). |
| 4 | Notebook v3 lặp qua `manifest.get('key_code_files', {})` (không tồn tại trong manifest v3) dẫn đến kiểm tra 0 tệp mà vẫn báo pass | `fl_training/package_verify.py:verify_package_manifest`, `scripts/package_release_v4.py`, `kaggle_stage2_scratch_v4.ipynb` | Tạo hàm xác minh chung `verify_package_manifest` kiểm tra schema `files[path] = {size_bytes, sha256}`, kích thước và SHA-256 từng tệp; in rõ số tệp thực kiểm tra (>0) và SHA-256 manifest. | `tests/test_package_and_notebook_v4.py` (7/7 PASSED). Package release v4 kiểm tra đạt 162 tệp. |
| 5 | GĐ2 dùng runner GĐ1 (`weighted_parameters`), tính validation 5 lần/round (1 lần/client) | `stage2_scratch/runner.py:run_single_fedavg_job`, `stage2_scratch/experiment.py` | Tạo runner GĐ2 tuần tự riêng: gọi `aggregate_fedavg_parameters` (float weighted theo $n_k$, BN counter `base + sum(local - base)`), server validation đúng 1 lần/round tại server. Seed policy theo `(seed, round, client_id)`. | `tests/test_real_flower_parity.py` (PASSED); `tests/test_flower_parity.py` (3/3 PASSED). |
| 6 | Parity giữa Flower và Sequential chỉ dùng tham số giả lập | `tests/test_real_flower_parity.py` | Huấn luyện thật 2 rounds MobileNetV3-Small từ đầu trên cả SequentialRunner và FlowerSimulator, so khớp bitwise/allclose từng tensor, integer BN counter và validation metrics. | `test_real_two_round_flower_sequential_parity` (PASSED 100%). |
| 7 | Notebook dùng cứng `SESSION_MINUTES=420` cho cả smoke và pilot | `kaggle_stage2_scratch_v4.ipynb` | Thiết lập session limits động theo hành động: `smoke` = 5 phút, `pilot` = 45 phút, `run` = 420 phút; kiểm soát deadline an toàn và khôi phục môi trường. | Đã triển khai trong `kaggle_stage2_scratch_v4.ipynb`. |
| 8 | Registry baselines hardcode số mẫu split v2 | `stage2_scratch/baselines.py:load_historical_baselines` | Đọc động counts từ `suite.json` (38.984 train, 4.400 val, 10.921 test), giữ nhãn `historical_reference`, `strict_comparison_eligible=False`. | Đã triển khai và kiểm chứng. |

---

## 2. Nhật Ký Nghiệm Thu Từng Hạng Mục (Local Acceptance)

### 2.1. Strict Preflight trên Dữ Liệu Thật (PlantVillage 54.305 ảnh)
- **Tập dữ liệu**: `PlantVillage-Dataset/raw/color` (38 lớp).
- **Bộ phân hoạch**: `data/partitions_stage2_scratch_v3` (8 kịch bản non-IID).
- **Kết quả kiểm tra**:
  - `label100`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `label1`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `label01`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `quantity100`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `quantity01`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `label_quantity01`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `feature100`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
  - `feature01`: 54.305 verified images, 54.284 unique byte hashes, 0 cross-owner duplicates.
- **Kiểm tra fixtures và review**:
  - 6 cặp kiểm chứng trực quan (`visually_verified_pairs.json`): SHA-256 `257cbcc1b8a77af1c59ab26eff71e191c0763db0cb6c1b5050e2019500a626f1` khớp hoàn toàn.
  - 1.058 cặp review trùng lặp (934 accepted + 124 uncertain trong `duplicate_review.json`): SHA-256 `53bbb477f71e0512535f58d17835e77cd727c65f4354e034ec1761120f0c2f74` khớp hoàn toàn.
  - Cả 1.064 cặp đều giữ nguyên cấu trúc nhóm lá, cùng split và cùng client (nếu ở train) trên cả 8 kịch bản.

### 2.2. Kiểm Thử Hồi Quy (Regression Tests Suite)
Chạy đồng thời 19 bài test tự động qua pytest:
- `tests/test_strict_preflight_regressions.py`: **8/8 PASSED**
- `tests/test_package_and_notebook_v4.py`: **7/7 PASSED**
- `tests/test_real_flower_parity.py`: **1/1 PASSED**
- `tests/test_flower_parity.py`: **3/3 PASSED**
**Tổng cộng: 19/19 PASSED (100%)**.

### 2.3. Đóng Gói Phát Hành v4 (FedAvg-Only Release Package)
- **Tập tin phát hành**: `fl_package_v4.zip`
  - Kích thước nén: ~55.38 MB (58.072.415 bytes)
  - Kích thước giải nén: ~253.41 MB (265.716.473 bytes)
  - Số lượng tệp: 162 tệp
  - SHA-256 package: `f6879497704a9c7afbe747b00233e46fdc099bf1d1afb0203460b7f8e5beb3ac`
- **Manifest phát hành**: `release_manifest_v4.json`
  - Protocol: `stage2_scratch_fedavg_v4`
  - Scope: `fedavg_only`
  - Verification: PASSED (162 files verified against disk bytes and sizes).
- **Notebook Kaggle**: `kaggle_stage2_scratch_v4.ipynb`

### 2.4. Kết Quả Chạy Smoke Check Laptop (Entrypoint Chẩn Đoán Thực Tế)
- **Lệnh chạy**: `python -m stage2_scratch smoke --suite data/partitions_stage2_scratch_v3 --dataset ../PlantVillage-Dataset/raw/color --output runs/diagnostic_v4 --session-minutes 5`
- **Mẫu thực tế**: 368 train samples (nhóm lá nguyên khối, nằm trong khoảng 300–600), 166 val samples (nằm trong khoảng 150–300), 38 lớp, 5 clients.
- **Tập test**: Được bảo vệ bởi `SentinelTestDataset` (ném AssertionError nếu bị nạp), `test_accessed: false`, `test_metrics: null`.
- **Kết quả 2 rounds**:
  - Round 1: train_loss = 3.6365, val_loss = 3.6378, val_acc = 0.0060 (thời gian: 18.45s).
  - Round 2: train_loss = 3.6327, val_loss = 3.6371, val_acc = 0.0542 (thời gian: 17.94s).
  - Tổng thời gian: 40.03 giây (nằm sâu trong ngân sách 5 phút).
  - Trạng thái: `COMPLETED`. Checkpoint atomic đã được commit tại `runs/diagnostic_v4/smoke/label100_seed42/checkpoints/label100_seed42_checkpoint.pth`.

---

## 3. Trạng Thái Nghiệm Thu Hiện Tại

- **LOCAL ACCEPTANCE**: **PASSED**  
  Mọi kiểm tra cục bộ về dữ liệu thật, fixtures, manifest, package integrity, runner decoupling, và Flower-Sequential parity đều đã đạt với bằng chứng định lượng thực tế.
- **KAGGLE GPU VERIFIED**: **PENDING EXECUTION**  
  Chưa chạy full GPU experiment trên Kaggle. Đây là bước tiếp theo khi người dùng tải `fl_package_v4.zip` và `kaggle_stage2_scratch_v4.ipynb` lên môi trường Kaggle để tiến hành kiểm chứng GPU.

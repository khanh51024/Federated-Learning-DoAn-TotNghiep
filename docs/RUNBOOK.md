# RUNBOOK — Các lệnh vận hành hiện hành

> Nguồn: chưng cất từ HANDOFF_CHECKLIST (đã xóa) sau đợt đồng bộ 2026-10-09.
> Chỉ giữ lệnh còn hiệu lực với code hiện tại. Trạng thái tiến độ xem ở `PROJECT_STATUS.md`.

## Quy tắc trước khi làm bất cứ việc gì

1. Đọc `PROJECT_STATUS.md` để biết dự án đang ở đâu.
2. Con số accuracy nào cũng phải truy được về `artifacts/registry.json`.
3. KHÔNG bao giờ để smoke/synthetic run ghi vào slot canonical
   (`artifacts/checkpoints/{centralized,fedavg}_best.pt`) — code đã chặn,
   nhưng nếu thêm script mới phải giữ nguyên tắc này.
4. Sau khi làm xong: `python scripts/update_status.py`.
5. Engine duy nhất là `src/`. Pipeline Kaggle/stage* cũ nằm ở `legacy/` (chạy riêng, xem `legacy/README.md`).
6. Máy này chỉ CPU: **không train**. Train ở ngoài rồi nhận kết quả theo mục 7.

## 0. Kiểm tra dataset gốc (điều kiện cho mọi run khoa học)

Hiện tại **CHƯA CÓ** PlantVillage gốc trên máy này. Khi có, đặt ở một thư mục
chứa trực tiếp 38 thư mục class (ví dụ `D:\PlantVillage`) và preflight:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_plantvillage.py D:\PlantVillage --full-path-check
```

PASS khi: 38 classes, 54,305 images, class names khớp `data/dataset_manifest.json`,
toàn bộ manifest paths resolve.

## 1. Demo kỹ thuật Flower (không phải kết quả khoa học)

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_five_client_network.py `
  tests/test_network_rtt_and_checkpoint.py `
  tests/test_faulty_client_network_detection.py `
  tests/test_attack_matrix_network.py -q
```

Điều kiện PASS: 1 server + 5 client process hoàn thành; ≥2 rounds feedback;
RTT > 0; `no_train` bị gắn suspicious; policy observe-only gộp đủ N client và gắn cờ; policy
reject-suspicious + strict-round hủy cả vòng khi còn <N update hợp lệ (không lặng lẽ gộp phần còn lại);
network checkpoint load/inference được. Test ghi vào `tmp_path`, không động tới `artifacts/`.

Artifact smoke chỉ dùng demo kỹ thuật: `artifacts/metrics/network_flower_history.json`,
`artifacts/checkpoints/federated_network_global.pt`.

## 2. Scientific network run — 1 server + 5 real clients (cần dataset)

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 -Rounds 5 -LocalEpochs 1 -Seed 42 -DryRun   # dry-run
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 -Rounds 5 -LocalEpochs 1 -Seed 42           # chạy thật
```

Script tự động: preflight → Flower server + 5 clients → data_mode=real → gRPC →
RTT từng client → validate update → held-out eval → lưu checkpoint → validate →
chỉ promote khi mọi gate PASS. Sau run, bắt buộc:

```powershell
.\.venv\Scripts\python.exe scripts\validate_network_artifact.py `
  --metrics artifacts\metrics\network_flower_history.json `
  --checkpoint artifacts\checkpoints\federated_network_global.pt `
  --expected-clients 5 --expected-rounds 5
```

Phải trả `"ok": true`. Artifact verified nằm ở `artifacts/verified_network/<promotion_id>/`.
Chỉ artifact trong `verified_network/` mới là bằng chứng network khoa học.

## 3. Centralized + IID/Non-IID report (cần dataset)

```powershell
.\.venv\Scripts\python.exe scripts\run_report_experiments.py `
  --mode real --dataset-root D:\PlantVillage `
  --epochs 10 --rounds 10 --seed 42
```

Experiment: Centralized, label100 (gần IID), label1, label01 (non-IID mạnh),
quantity01 (data starvation). Kết quả: `artifacts/report_experiments/<run_id>/`.

## 4. Multi-seed cho báo cáo khoa học (cần dataset)

```powershell
.\scripts\run_real_multiseed_report.ps1 `
  -DatasetRoot D:\PlantVillage `
  -CentralizedEpochs 10 -FederatedRounds 10 -Seeds 42,123,2026
```

Báo cáo dùng mean ± std cho accuracy/macro-F1, hội tụ theo round, thời gian train,
RTT, data starvation. KHÔNG dùng một seed duy nhất để kết luận.

## 5. Attack experiment

Baseline 5 normal clients; attack: 4 normal + 1 trong {no_train, random_weights,
scaled_update, noise_update}; chạy cả 2 policy {Observe-only, Reject-suspicious}.
Lưu ý hợp đồng strict-round của server: với policy Reject-suspicious, update bị loại làm số hợp lệ <N nên server
hủy vòng thay vì gộp phần còn lại. Kết luận đúng: vanilla FedAvg không tự xác minh client update — validator là lớp
bổ sung của hệ thống, không phải chức năng mặc định của FedAvg.

## 6. Gate cuối trước khi nộp/demo

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py --full-gates   # chạy pytest toàn bộ (chậm, ~15 phút vì test mạng Flower)
.\.venv\Scripts\python.exe scripts\pre_demo_check.py --dataset-root D:\PlantVillage
```

Bật hook chặn commit khi `PROJECT_STATUS.md` lệch: `git config core.hooksPath scripts/hooks`
(hook chạy `update_status.py --check`; sửa bằng `python scripts/update_status.py`).

Mục tiêu: provenance `verified`, scientific blockers = 0, PROJECT_STATUS.md
hiển thị "SẴN SÀNG NGHIỆM THU KHOA HỌC". Hiện tại đang ở mức
`technical_demo_ready` — xem PROJECT_STATUS.md mục 3.

## 6b. Báo cáo suy ra từ bundle (không cần ảnh)

`update_status.py` tự chạy hai script này; chạy tay khi cần:

```powershell
.\.venv\Scripts\python.exe scripts\report_domain_gap.py    # PlantVillage vs PlantDoc theo model  -> artifacts/plantdoc_domain_gap_report.json
.\.venv\Scripts\python.exe scripts\report_per_class.py     # recall theo lớp, phân rã lỗi, lỗi confidence cao -> artifacts/plantdoc_per_class_report.json
```

Đây là phân tích trên số liệu ĐÃ CÔNG BỐ (kèm kiểm nhất quán: recall có trọng số = top-1, phân rã lỗi cộng đúng tổng).
Không thay thế tái đánh giá cục bộ — cần ảnh validation, xem mục BLOCKED trong `PROJECT_STATUS.md`.

## 6c. Dùng dataset đã tải (`data/dataset_sources/`, gitignored)

Nguồn: nhánh `dataset-sources` của `khanh51024/Federated-Learning-DoAn-TotNghiep` (PlantVillage, PlantDoc, release `mixed/pv_pd_v3`).
Lấy lại khi máy chưa có (tải ~4 GB vào `.git`, giải nén phần release cần ~2,2 GB):

```powershell
git fetch origin dataset-sources
git -c core.autocrlf=false archive origin/dataset-sources mixed/pv_pd_v3 mixed/pv_pd_v1 PlantVillage-Dataset/raw/color PlantVillage-Dataset/leaf-map.json PlantVillage-Dataset/leaf_grouping THONG_TIN_NGUON_DATASET.txt | tar -x -C data/dataset_sources
```

Có dataset thì ba script sau chạy được (CPU, không cần GPU):

```powershell
.\.venv\Scripts\python.exe scripts\audit_release_leakage.py       # ~8 phút: train/val/calibration rời nhau + băm 46.076 ảnh -> artifacts/release_leakage_audit.json
.\.venv\Scripts\python.exe scripts\evaluate_published_models.py   # ~10 phút: tái đánh giá 7 model trên val, ECE + temperature scaling -> artifacts/local_reeval_report.json
.\.venv\Scripts\python.exe scripts\report_client_shards.py        # phân bố lớp theo client -> artifacts/client_shard_class_counts.json
.\.venv\Scripts\python.exe scripts\evaluate_published_models.py --holdout   # ~6 phút: 7 model trên test giữ riêng 11.015 ảnh -> artifacts/holdout_eval_report.json (thêm --reuse: dùng lại logits đã lưu)
.\.venv\Scripts\python.exe scripts\report_statistics.py         # khoảng tin cậy theo cụm, McNemar, độ nhạy 'cùng lá' (dùng logits đã lưu) -> artifacts/statistics_report.json
.\.venv\Scripts\python.exe scripts\report_partition_schemes.py  # phân bố lớp của 12 cách chia (chương 2 của web) -> artifacts/partition_schemes_class_counts.json
.\.venv\Scripts\python.exe scripts\build_error_gallery.py       # thumbnail ảnh val bị đoán sai với độ tin cậy >= 0,9 -> web/static/samples/errors/
.\.venv\Scripts\python.exe scripts\build_demo_gallery.py        # 19 ảnh mẫu của trang Chẩn đoán -> web/static/samples/demo/
```

Dùng `--reuse` ở script thứ hai để phân tích lại từ logits đã lưu (`data/reeval/`) mà không chạy lại suy luận.
Server demo không có dataset: các báo cáo trên được sinh ở máy này rồi chép lên cùng mã.

## 7. Nhận kết quả train ngoài (bundle mới)

Train chạy ở Kaggle/branch `train-*`, xuất bundle có `artifact_manifest.json` + SHA-256. Trên máy này:

```powershell
# 1. đặt 3 nhánh nguồn dưới một thư mục, rồi import + kiểm hash vào artifacts/branch_demo/<ngày>/
.\.venv\Scripts\python.exe scripts\import_branch_demo.py --sources-root <thư_mục_nguồn> --target artifacts\branch_demo\<ngày>
# 2. trỏ registry sang bundle mới: sửa DEMO_BUNDLE_DIR / DEMO_PROTOCOL_ID trong src/registry.py
# 3. cập nhật hằng số ghim (RELEASE_SHA, VALIDATION_SHA, SOURCES) trong scripts/import_branch_demo.py cho bundle mới
# 4. đồng bộ: registry + cổng web + PROJECT_STATUS.md
.\.venv\Scripts\python.exe scripts\update_status.py
```

Model không đạt 5 điều kiện của cổng `web_visible` sẽ không hiện trên web; lý do nằm ở `web_gate_failures`
trong `artifacts/registry.json` và mục 2 của `PROJECT_STATUS.md`.

## 8. Chạy test

```powershell
.\.venv\Scripts\python.exe -m pytest -q                 # test hiện hành (tests/)
.\.venv\Scripts\python.exe -m pytest legacy/tests -q    # test pipeline cũ (một số cần dữ liệu gốc/EOL CRLF)
```

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
RTT > 0; `no_train` bị gắn suspicious; reject policy loại được suspicious update;
network checkpoint load/inference được.

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
Kết luận đúng: vanilla FedAvg không tự xác minh client update — validator là lớp
bổ sung của hệ thống, không phải chức năng mặc định của FedAvg.

## 6. Gate cuối trước khi nộp/demo

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py --full-gates
.\.venv\Scripts\python.exe scripts\pre_demo_check.py --dataset-root D:\PlantVillage
```

Mục tiêu: provenance `verified`, scientific blockers = 0, PROJECT_STATUS.md
hiển thị "SẴN SÀNG NGHIỆM THU KHOA HỌC". Hiện tại đang ở mức
`technical_demo_ready` — xem PROJECT_STATUS.md mục 3.

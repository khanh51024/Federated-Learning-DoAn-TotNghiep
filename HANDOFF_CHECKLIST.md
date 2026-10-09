# HANDOFF CHECKLIST — Federated Learning PlantVillage

## Trạng thái hiện tại

- Technical regression: **60 passed, 2 dependency warnings**.
- Flower network thật: PASS.
- 1 server + 5 clients + multi-round: PASS.
- RTT per-client: PASS.
- Fault detection/reject policy: PASS.
- Held-out evaluation guard: PASS.
- Artifact provenance/scientific guard: PASS.
- Raw PlantVillage trên máy hiện tại: **CHƯA CÓ**.
- Scientific artifacts hiện tại: **CHƯA VERIFIED**.

Không dùng accuracy từ smoke/synthetic run cho luận văn.

---

## A. Demo kỹ thuật nhanh — không dùng làm kết quả khoa học

Mục tiêu: chứng minh Flower/FedAvg, server-client, round feedback, RTT, validator, fault modes và inference hoạt động.

Chạy test regression:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_five_client_network.py `
  tests/test_network_rtt_and_checkpoint.py `
  tests/test_faulty_client_network_detection.py `
  tests/test_attack_matrix_network.py -q
```

Điều kiện PASS:

- 1 server + 5 client process hoàn thành.
- >= 2 rounds feedback thành công.
- RTT probe > 0.
- `no_train` bị gắn suspicious.
- random/scaled/noise vượt ngưỡng bị gắn suspicious.
- reject policy loại được suspicious update.
- network checkpoint load/inference được.

Artifact smoke chỉ dùng demo kỹ thuật:

- `artifacts/metrics/network_flower_history.json`
- `artifacts/checkpoints/federated_network_global.pt`

Nếu `data_mode=synthetic` thì **không scientific-valid**.

---

## B. Scientific network run — 1 server + 5 real clients

### 1. Chuẩn bị PlantVillage

Dataset root phải chứa trực tiếp 38 thư mục class PlantVillage.

Preflight:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_plantvillage.py D:\PlantVillage --full-path-check
```

PASS khi:

- 38 classes.
- 54,305 images.
- class names khớp manifest.
- toàn bộ manifest paths resolve.

### 2. Dry-run trước

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 5 `
  -LocalEpochs 1 `
  -Seed 42 `
  -DryRun
```

Dry-run chỉ in command, không train.

### 3. Chạy thật

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 5 `
  -LocalEpochs 1 `
  -Seed 42
```

Script tự động:

1. PlantVillage full preflight.
2. Start Flower server.
3. Start 5 Flower clients.
4. Bắt buộc `data_mode=real`.
5. Train qua gRPC thật.
6. Đo RTT từng client.
7. Server-side validate update.
8. Held-out global validation/test.
9. Lưu global checkpoint.
10. Validate artifact.
11. Chỉ promote nếu mọi gate PASS.

### 4. Scientific artifact PASS

Sau run, validator:

```powershell
.\.venv\Scripts\python.exe scripts\validate_network_artifact.py `
  --metrics artifacts\metrics\network_flower_history.json `
  --checkpoint artifacts\checkpoints\federated_network_global.pt `
  --expected-clients 5 `
  --expected-rounds 5
```

Phải trả:

```text
"ok": true
```

Artifact verified nằm ở:

```text
artifacts/verified_network/<promotion_id>/
```

Gồm:

- `network_metrics.json`
- `global_model.pt`
- `validation.json`
- `verified_network_manifest.json`

Chỉ artifact trong `verified_network/` mới nên dùng làm bằng chứng network khoa học.

---

## C. Centralized + IID/Non-IID/Data-Starvation report

Một run real:

```powershell
.\.venv\Scripts\python.exe scripts\run_report_experiments.py `
  --mode real `
  --dataset-root D:\PlantVillage `
  --epochs 10 `
  --rounds 10 `
  --seed 42
```

Các experiment:

- Centralized.
- `label100`: gần IID.
- `label1`: Non-IID vừa.
- `label01`: Non-IID mạnh.
- `quantity01`: quantity skew / data starvation.

Kết quả lưu trong:

```text
artifacts/report_experiments/<run_id>/
```

Có:

- `plan.json`
- `summary.json`
- `summary.csv`
- checkpoint/metrics từng experiment.

---

## D. Multi-seed cho báo cáo khoa học

Khuyến nghị ít nhất 3 seed:

```text
42
123
2026
```

Chạy:

```powershell
.\scripts\run_real_multiseed_report.ps1 `
  -DatasetRoot D:\PlantVillage `
  -CentralizedEpochs 10 `
  -FederatedRounds 10 `
  -Seeds 42,123,2026
```

Báo cáo dùng:

- mean accuracy ± std.
- mean macro-F1 ± std.
- loss.
- convergence theo round.
- thời gian training.
- RTT/network overhead.
- data starvation metrics.

Không dùng một seed duy nhất để kết luận mô hình nào tốt hơn.

---

## E. Attack experiment

Baseline:

```text
5 normal clients
```

Attack cases:

```text
4 normal + 1 no_train
4 normal + 1 random_weights
4 normal + 1 scaled_update
4 normal + 1 noise_update
```

Chạy cả hai policy:

```text
Observe-only
Reject-suspicious
```

Thu:

- global accuracy/loss.
- suspicious/rejected clients.
- received update norm.
- RTT.
- convergence.

Kết luận đúng về FedAvg:

> Vanilla FedAvg không tự xác minh client update; malicious/corrupted updates có thể làm hỏng global model nếu vẫn được aggregate. Validator là lớp bổ sung của hệ thống, không phải chức năng mặc định của FedAvg.

---

## F. Inference demo

Web inference có hai nguồn:

- Canonical FedAvg.
- Network Global.

Chỉ chọn Network Global làm bằng chứng khoa học sau khi artifact đã được `verified_network` promotion.

Dùng cùng một ảnh để so:

```text
Centralized prediction
vs
Federated Global prediction
```

---

## G. Final gate trước khi nộp/demo

Chạy:

```powershell
.\.venv\Scripts\python.exe scripts\pre_demo_check.py --dataset-root D:\PlantVillage
```

Mục tiêu cuối:

```text
overall = ready
```

Không chỉ:

```text
technical_demo_ready
```

Scientific blockers phải bằng 0:

- real PlantVillage PASS.
- provenance PASS.
- verified network artifact PASS.
- real multi-seed summary PASS.

---

## H. Lưu ý Git

Repo hiện có nhiều modified/untracked artifact từ trước. Không `git add .` và commit toàn bộ một cách máy móc.

Trước commit:

```powershell
git status
git diff
```

Ưu tiên commit source/test/docs riêng; checkpoint, temporary metrics và smoke artifacts cần xem xét trước khi đưa vào Git.

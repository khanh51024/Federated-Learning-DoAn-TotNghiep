# PROJECT STATUS — Federated Learning PlantVillage

> **Cập nhật lúc:** 2026-10-09 17:01 · sinh tự bởi `scripts/update_status.py`
> **Đánh giá hiện tại:** DEMO KỸ THUẬT DÙNG ĐƯỢC — CHƯA nghiệm thu khoa học (checkpoint canonical mất, chưa multi-seed)
>
> File này là nguồn thống kê DUY NHẤT về tiến độ. Con số nào trong repo
> phải truy được về `artifacts/registry.json`. Không sửa phần sinh tự —
> chỉ sửa khối ROADMAP giữa hai marker.

## 1. Trạng thái hệ thống (sự thật từ registry)

- Tổng số model trong registry: **18** (scientific_canonical: 3, published_demo: 7, network_demo: 4, reference: 1, smoke: 3)
- Dataset reproduction cục bộ: `data/partitions_stage2_scratch_v3` — split **38984/4400/10921** (5 clients, seed 42, protocol `stage2_scratch_clean_v3`)
- Run canonical Kaggle: `20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62` — metrics CÓ, **checkpoint KHÔNG còn trên máy**
- Demo web dùng họ published_demo (hash-verified) — hoạt động bình thường

## 2. Kết quả chuẩn theo họ (không trộn lẫn)

### Họ A — Khoa học canonical (scratch, PlantVillage thuần)

_Kết quả khoa học canonical (scratch, PlantVillage thuần) từ run Kaggle 20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62. Chỉ số từ metrics JSON của run; KHÔNG so sánh với họ published_demo._

| Model | Trạng thái | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| FedAvg — scientific canonical | Scientific — metrics-only | ❌ không có trên máy | test acc 99.0932%; macro-F1 98.72%; test 10917 ảnh | artifacts/metrics/fedavg_results.json + results/fedavg/20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62/evaluations/b0cddc167068f46403c13595/test_metrics.json |
| Centralized — scientific canonical | Scientific — historical (chưa tái kiểm chứng) | ❌ không có trên máy | test acc 98.25%; macro-F1 97.81%; test n/a ảnh | artifacts/metrics/centralized_results.json |
| Local-only — scientific canonical | Scientific — historical (chưa tái kiểm chứng) | ❌ không có trên máy | mean acc 81.40% ± 8.70% | artifacts/metrics/local_only_results.json |

### Họ B — Published demo (pretrained, mixed PV+PlantDoc)

_Checkpoint pretrained mixed PlantVillage+PlantDoc (protocol mixed_pv_pd_v3_pretrained_seed42_validation), hash-verified khi import. Dùng cho demo web/inference, KHÔNG phải kết quả khoa học của protocol GĐ2 scratch._

| Model | Trạng thái | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Centralized v6 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 53.90% (n=269); PlantVillage 99.22% (n=4344) | artifacts/branch_demo/20261003/centralized/validation_metrics.json |
| FedAvg v7 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 47.21% (n=269); PlantVillage 93.12% (n=4344) | artifacts/branch_demo/20261003/fedavg/validation_metrics.json |
| Local-only client 00 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 25.65% (n=269); PlantVillage 44.89% (n=4344) | artifacts/branch_demo/20261003/local/client_00/validation_metrics.json |
| Local-only client 01 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 25.28% (n=269); PlantVillage 52.00% (n=4344) | artifacts/branch_demo/20261003/local/client_01/validation_metrics.json |
| Local-only client 02 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 29.00% (n=269); PlantVillage 55.34% (n=4344) | artifacts/branch_demo/20261003/local/client_02/validation_metrics.json |
| Local-only client 03 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 26.77% (n=269); PlantVillage 55.46% (n=4344) | artifacts/branch_demo/20261003/local/client_03/validation_metrics.json |
| Local-only client 04 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) | ✅ có, hash ok | PlantDoc 21.19% (n=269); PlantVillage 55.18% (n=4344) | artifacts/branch_demo/20261003/local/client_04/validation_metrics.json |

### Họ C — Network demo (smoke kỹ thuật)

_Checkpoint từ các bài test mạng Flower (RTT, attack matrix, smoke network). Chỉ dùng demo kỹ thuật, cấm dùng làm chỉ số khoa học._

| Model | Trạng thái | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Flower network global (smoke) | Network demo (không khoa học) | present | (không có số liệu) | — |
| Attack matrix — observe | Network demo (không khoa học) | present | (không có số liệu) | — |
| Attack matrix — reject | Network demo (không khoa học) | present | (không có số liệu) | — |
| RTT + checkpoint test | Network demo (không khoa học) | present | (không có số liệu) | — |

### Model tham chiếu

_Model tham chiếu (TorchScript), không phải checkpoint FedAvg._

| Model | Trạng thái | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| PlantVillage TorchScript (tham chiếu) | Tham chiếu (không phải FedAvg) | present | (không có số liệu) | — |

### Họ D — Smoke (CẤM dùng trong luận văn)

_Checkpoint synthetic smoke (accuracy 1.0 / 0.0263 do dữ liệu dummy). CẤM dùng trong luận văn; đã cách ly vào artifacts/checkpoints/smoke/._

| Model | Trạng thái | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Smoke — centralized_best_smoke | Smoke (cấm luận văn) | quarantined | (không có số liệu) | — |
| Smoke — fedavg_best_smoke | Smoke (cấm luận văn) | quarantined | test_accuracy=1.0 | — |
| Smoke — fedavg_last_smoke | Smoke (cấm luận văn) | quarantined | last_accuracy=1.0 | — |

## 3. Gates kỹ thuật

| Gate | Kết quả | Chi tiết |
|---|---|---|
| Provenance checkpoints | **unverified** | critical: fedavg_state_hash_mismatch |
| Pre-demo check | **technical_demo_ready** | artifacts/pre_demo_check.json |
| Final acceptance | technical **pass** / scientific **blocked** | blockers: 4 |
| Network artifact | **FAIL** | artifacts/network_artifact_validation.json |

## 4. Lộ trình & checklist (khối dưới đây TỰ TAY SỬA)

<!-- ROADMAP-BEGIN (khối này TỰ TAY SỬA - script bảo toàn nội dung) -->
### Checklist nghiệm thu

- [x] Lập model registry thống nhất (artifacts/registry.json) — 2026-10-09
- [x] Cách ly checkpoint smoke khỏi slot canonical + chặn ghi đè từ code — 2026-10-09
- [x] Tái sinh canonical_manifest.json trung thực từ nguồn thật — 2026-10-09
- [x] File thống kê duy nhất PROJECT_STATUS.md (tự sinh) — 2026-10-09
- [ ] Tách web inference/training/monitoring đọc qua registry
- [ ] Dọn tài liệu mâu thuẫn (xóa archive, tạo RUNBOOK/MODEL_REGISTRY)
- [ ] Kiểm chứng provenance & checkpoint (cần tái chạy run canonical hoặc khôi phục weights từ Kaggle)
- [ ] Chốt một protocol khoa học duy nhất cho mỗi thí nghiệm
- [ ] Audit leakage theo từng run (cần dataset gốc)
- [ ] Đánh giá PlantDoc theo từng lớp
- [ ] Phân tích lỗi confidence cao (100 ảnh sai với confidence > 0.9 ở FedAvg v7)
- [ ] Chạy multi-seed non-IID (seed 42/123/2026)
- [ ] Đồng bộ báo cáo & artifacts sau khi có run mới
- [ ] Full tests + release sạch

### Nhật ký

- 2026-10-09: Đồng bộ hóa — lập registry (18 models/5 family), phát hiện & cách ly smoke
  ghi đè slot canonical, xác nhận checkpoint canonical KHÔNG còn trên máy (metrics-only).
<!-- ROADMAP-END -->

## 5. Việc cần làm tiếp theo (sinh từ checklist chưa tick)

1. Tách web inference/training/monitoring đọc qua registry
2. Dọn tài liệu mâu thuẫn (xóa archive, tạo RUNBOOK/MODEL_REGISTRY)
3. Kiểm chứng provenance & checkpoint (cần tái chạy run canonical hoặc khôi phục weights từ Kaggle)
4. Chốt một protocol khoa học duy nhất cho mỗi thí nghiệm
5. Audit leakage theo từng run (cần dataset gốc)
6. Đánh giá PlantDoc theo từng lớp
7. Phân tích lỗi confidence cao (100 ảnh sai với confidence > 0.9 ở FedAvg v7)
8. Chạy multi-seed non-IID (seed 42/123/2026)
9. Đồng bộ báo cáo & artifacts sau khi có run mới
10. Full tests + release sạch

## 6. Cách đồng bộ

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py          # chuẩn
.\.venv\Scripts\python.exe scripts\update_status.py --skip-gates  # chỉ sinh lại file
```

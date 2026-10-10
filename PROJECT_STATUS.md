# PROJECT STATUS — Federated Learning PlantVillage

> **Cập nhật lúc:** 2026-10-10 22:29 · sinh tự bởi `scripts/update_status.py`
> **Đánh giá hiện tại:** DEMO KỸ THUẬT DÙNG ĐƯỢC — CHƯA nghiệm thu khoa học (6 mục BLOCKED cần dữ liệu/máy train, 0 mục kỹ thuật chưa XONG)
>
> File này là nguồn thống kê DUY NHẤT về tiến độ. Checklist do máy chấm từ sự thật trên đĩa;
> con số nào trong repo phải truy được về `artifacts/registry.json`. Không sửa tay — chỉ khối Nhật ký cuối file.

## 1. Hướng chính thức

- **Luồng một chiều:** train ngoài repo (Kaggle/branch) -> bundle + manifest + SHA-256 -> import/kiểm hash -> registry -> status/web/báo cáo
- **Kết quả hiện hành:** họ `published_demo`, protocol `mixed_pv_pd_v3_pretrained_seed42_validation`.
- **Engine trong repo:** `src/` (Flower demo, evaluator, web). Pipeline Kaggle cũ nằm ở `legacy/` — chỉ để tra cứu/tái lập lịch sử.
- **Máy này:** CPU, không train; chỉ nhận kết quả đã train rồi kiểm hash + test.

## 2. Trạng thái hệ thống (sự thật từ registry)

- Registry: **18** model (scientific_canonical: 3, published_demo: 7, network_demo: 4, reference: 1, smoke: 3)
- Web được phép dùng: **8** model (đã qua đủ 5 điều kiện: hash, taxonomy, preprocessing, metrics↔protocol, forward test)
- Dataset cục bộ để tái lập: `data/partitions_stage2_scratch_v3` — split **38984/4400/10921** (5 clients, seed 42, protocol `stage2_scratch_clean_v3`)
- Run Kaggle lịch sử: `20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62` — metrics CÓ, **checkpoint KHÔNG còn trên máy**

## 3. Checklist nghiệm thu — XONG 14/20 · CHƯA 0 · BLOCKED 6

| # | Hạng mục | Trạng thái | Bằng chứng (máy kiểm) |
|---|---|---|---|
| 1 | Registry v2 + danh mục mọi checkpoint trên đĩa | ✅ XONG | schema v2; 18 model; 58 file .pt đều có tên (15 trong model, 43 trong nhóm) |
| 2 | Smoke cách ly khỏi slot canonical | ✅ XONG | slot canonical không chứa smoke; 3 checkpoint smoke nằm ở artifacts/checkpoints/smoke/ |
| 3 | Một engine train (pipeline cũ gom vào legacy/) | ✅ XONG | engine duy nhất là src/; pipeline Kaggle/stage* nằm trong legacy/ (chạy riêng: pytest legacy/tests) |
| 4 | Web tách router/service, inference đi một cửa | ✅ XONG | app.py chỉ mount router; inference chỉ đi qua web/services/model_service.py (cổng web_visible) |
| 5 | Không hardcode đường dẫn artifact | ✅ XONG | mọi đường dẫn artifacts/ trong src/web/scripts lấy từ src/registry.py (quét AST) |
| 6 | Dọn tài liệu mâu thuẫn | ✅ XONG | gốc repo chỉ còn README.md + PROJECT_STATUS.md; docs/ chỉ gồm tài liệu hiện hành |
| 7 | Số liệu trong docs không mâu thuẫn artifact | ✅ XONG | không còn số liệu mâu thuẫn đã biết (97,12, 97.12...) và không ghi tĩnh kết quả test trong README/docs |
| 8 | Chốt một protocol khoa học cho mỗi thí nghiệm | ✅ XONG | mỗi protocol (scratch GĐ2 cục bộ / bundle mixed pretrained) có mục riêng trong EXPERIMENT_PROTOCOL.md |
| 9 | Test không ghi đè artifact đang track | ✅ XONG | test không tham chiếu artifacts/checkpoints/metrics (dùng tmp_path) |
| 10 | Dependency khai báo một nơi (pyproject = requirements) | ✅ XONG | 15 dependency trùng giữa pyproject.toml và requirements.txt; ghim cũ nằm ở legacy/pyproject.toml |
| 11 | Full tests pass trên mã nguồn hiện tại | ✅ XONG | technical_acceptance=pass lúc 2026-10-10T22:29:50 (gồm test mạng, 39 file test) trên đúng mã nguồn hiện tại |
| 12 | Kiểm chứng provenance & checkpoint canonical | ⛔ BLOCKED | provenance=unverified; lỗi: ['fedavg_checkpoint_missing', 'centralized_checkpoint_missing'] |
| 13 | Audit leakage theo từng run | ⛔ BLOCKED | 7/7 model hiện hành có leakage_audit=unknown. Các nhánh train-* trên GitHub (đã kiểm 2026-10-10) chỉ có MÃ ràng buộc group-disjoint giữa client (plant_data_contract/partitions.py) và hash release, KHÔNG có manifest ảnh train/val nên không audit được dữ liệu thật. [mục kiểm tay: chưa có script đọc kết quả audit để tự chuyển trạng thái] |
| 14 | Đánh giá PlantDoc theo từng lớp (từ số công bố) | ✅ XONG | PlantDoc phủ 13/38 lớp; 6 lớp recall<50% (136/269 ảnh); FedAvg tệ hơn Centralized ở 5/13 lớp (số công bố, đã kiểm nhất quán; xem mục 5) |
| 15 | Tái đánh giá cục bộ trên ảnh validation (xác nhận số công bố) | ⛔ BLOCKED | số theo lớp ở trên là số CÔNG BỐ; chưa tái đánh giá cục bộ nên chưa có confusion matrix đầy đủ. [mục kiểm tay] |
| 16 | Phân tích lỗi confidence cao (mức tổng hợp, từ số công bố) | ✅ XONG | mức tổng hợp từ metrics công bố — Centralized v6: 77/269 ảnh PlantDoc sai với confidence>0.9 (62% số lỗi); FedAvg v7: 41/269 ảnh PlantDoc sai với confidence>0.9 (29% số lỗi) (xem mục 5) |
| 17 | Phân tích từng ảnh sai tự tin + đo calibration | ⛔ BLOCKED | chưa phân tích được TỪNG ảnh sai tự tin và chưa đo calibration (softmax chưa hiệu chuẩn, UI cảnh báo is_uncertain chỉ là heuristic). [mục kiểm tay] |
| 18 | Chạy multi-seed non-IID (seed 42/123/2026) | ⛔ BLOCKED | chưa có multi-seed THẬT (chỉ có bản smoke accuracy 0.0263 trên ảnh tổng hợp) |
| 19 | Đồng bộ báo cáo & artifacts sau khi có run mới | ⛔ BLOCKED | phụ thuộc multi-seed thật |
| 20 | Release sạch (mọi gate kỹ thuật XONG) | ✅ XONG | mọi mục kỹ thuật XONG; trước khi nộp vẫn phải `git status` sạch và gắn tag |

### Việc đang BLOCKED — cần gì và mở khóa bằng lệnh nào

**Kiểm chứng provenance & checkpoint canonical**
- Cần: HAI file checkpoint thật: FedAvg best.pt của run canonical (oid sha256:bdfe01879aef693d…) và Centralized best — hiện chỉ là con trỏ Git-LFS / không có, repo không có remote nên không thể `git lfs pull`
- Mở khóa:

```powershell
# 1) tải hai file từ Kaggle, 2) đặt đúng vào slot canonical (đường mà provenance validator và registry đọc):
Copy-Item <fedavg_best.pt tải về> artifacts/checkpoints/fedavg_best.pt
Copy-Item <centralized_best.pt tải về> artifacts/checkpoints/centralized_best.pt
.\.venv\Scripts\python.exe scripts\update_status.py   # so state-hash với canonical_manifest
```

**Audit leakage theo từng run**
- Cần: manifest ảnh train/val của release `dataset/mixed/pv_pd_v3` (hash 6d2c6b40…, nằm ở Kaggle/Drive của nhóm train, không nằm trong repo) hoặc chính dataset gốc để dựng lại split
- Mở khóa:

```powershell
# 1) lấy dataset/mixed/pv_pd_v3 (hoặc riêng manifest train/val) từ nhóm train, đặt ở thư mục cục bộ, rồi:
.\.venv\Scripts\python.exe scripts\verify_integrity.py --all   # hiện chỉ audit split cục bộ partitions_*; cần mở rộng cho manifest bundle
```

**Tái đánh giá cục bộ trên ảnh validation (xác nhận số công bố)**
- Cần: ảnh validation (269 PlantDoc + 4.344 PlantVillage) khớp validation_manifest của bundle + checkpoint đã kiểm hash
- Mở khóa:

```powershell
# chưa có script tái đánh giá — viết scripts/evaluate_published_per_class.py khi có ảnh validation
# (cùng điều kiện dữ liệu với mục audit leakage; checkpoint bundle đã có và đã kiểm hash)
.\.venv\Scripts\python.exe scripts\audit_realworld_inference.py --images <thư_mục_ảnh_ngoài>
.\.venv\Scripts\python.exe scripts\evaluate_inference_audit.py --audit artifacts\inference_audit.json --labels <nhãn.csv>
```

**Phân tích từng ảnh sai tự tin + đo calibration**
- Cần: ảnh validation + checkpoint để liệt kê ảnh sai confidence cao, đo ECE/reliability và thử temperature scaling trên tập validation độc lập
- Mở khóa:

```powershell
# cùng điều kiện dữ liệu với mục 'tái đánh giá cục bộ'; viết scripts/calibration_report.py khi có ảnh validation
```

**Chạy multi-seed non-IID (seed 42/123/2026)**
- Cần: dataset PlantVillage gốc + máy train (máy này CPU, không train)
- Mở khóa:

```powershell
.\scripts\run_real_multiseed_report.ps1 -DatasetRoot D:\PlantVillage -CentralizedEpochs 10 -FederatedRounds 10 -Seeds 42,123,2026
# hoặc train ngoài rồi import bundle: python scripts/import_branch_demo.py (xem docs/MODEL_REGISTRY.md)
```

**Đồng bộ báo cáo & artifacts sau khi có run mới**
- Cần: chạy xong mục multi-seed rồi đồng bộ báo cáo
- Mở khóa:

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py
```

### Việc làm được ngay (chưa BLOCKED)

_(không còn mục kỹ thuật nào mở)_

## 4. Kết quả chuẩn theo họ (không trộn lẫn)

### Hiện hành — bundle đã kiểm hash (pretrained, mixed PV+PlantDoc)

_Checkpoint pretrained mixed PlantVillage+PlantDoc (protocol mixed_pv_pd_v3_pretrained_seed42_validation), hash-verified khi import. Dùng cho demo web/inference, KHÔNG phải kết quả khoa học của protocol GĐ2 scratch._

| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Centralized v6 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 53.90% (n=269); PlantVillage 99.22% (n=4344) | artifacts/branch_demo/20261003/centralized/validation_metrics.json |
| FedAvg v7 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 47.21% (n=269); PlantVillage 93.12% (n=4344) | artifacts/branch_demo/20261003/fedavg/validation_metrics.json |
| Local-only client 00 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 25.65% (n=269); PlantVillage 44.89% (n=4344) | artifacts/branch_demo/20261003/local/client_00/validation_metrics.json |
| Local-only client 01 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 25.28% (n=269); PlantVillage 52.00% (n=4344) | artifacts/branch_demo/20261003/local/client_01/validation_metrics.json |
| Local-only client 02 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 29.00% (n=269); PlantVillage 55.34% (n=4344) | artifacts/branch_demo/20261003/local/client_02/validation_metrics.json |
| Local-only client 03 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 26.77% (n=269); PlantVillage 55.46% (n=4344) | artifacts/branch_demo/20261003/local/client_03/validation_metrics.json |
| Local-only client 04 (published demo) | Demo (hash-verified, pretrained mixed PV+PlantDoc) (web: ✅) | ✅ có, hash ok | PlantDoc 21.19% (n=269); PlantVillage 55.18% (n=4344) | artifacts/branch_demo/20261003/local/client_04/validation_metrics.json |

### Lịch sử — Kaggle scratch, PlantVillage thuần (chỉ còn metrics)

_Kết quả khoa học canonical (scratch, PlantVillage thuần) từ run Kaggle 20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62. Chỉ số từ metrics JSON của run; KHÔNG so sánh với họ published_demo._

| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| FedAvg — scientific canonical | Scientific — metrics-only (web: —) | ❌ không có trên máy | test acc 99.0932%; macro-F1 98.72%; test 10917 ảnh | artifacts/metrics/fedavg_results.json + results/fedavg/20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62/evaluations/b0cddc167068f46403c13595/test_metrics.json |
| Centralized — scientific canonical | Scientific — historical (chưa tái kiểm chứng) (web: —) | ❌ không có trên máy | test acc 98.25%; macro-F1 97.81%; test n/a ảnh | artifacts/metrics/centralized_results.json |
| Local-only — scientific canonical | Scientific — historical (chưa tái kiểm chứng) (web: —) | not_applicable | mean acc 81.40% ± 8.70% | artifacts/metrics/local_only_results.json |

### Network demo (smoke kỹ thuật Flower)

_Checkpoint từ các bài test mạng Flower (RTT, attack matrix, smoke network). Chỉ dùng demo kỹ thuật, cấm dùng làm chỉ số khoa học._

| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Flower network global (smoke) | Network demo (không khoa học) (web: —) | present | (không có số liệu) | — |
| Attack matrix — observe | Network demo (không khoa học) (web: —) | present | (không có số liệu) | — |
| Attack matrix — reject | Network demo (không khoa học) (web: —) | present | (không có số liệu) | — |
| RTT + checkpoint test | Network demo (không khoa học) (web: —) | present | (không có số liệu) | — |

### Model tham chiếu

_Model tham chiếu (TorchScript), không phải checkpoint FedAvg._

| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| PlantVillage TorchScript (tham chiếu) | Tham chiếu (không phải FedAvg) (web: ✅) | ✅ có, hash ok | (không có số liệu) | — |

### Smoke — CẤM dùng trong luận văn

_Checkpoint synthetic smoke (accuracy 1.0 / 0.0263 do dữ liệu dummy). CẤM dùng trong luận văn; đã cách ly vào artifacts/checkpoints/smoke/._

| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Smoke — centralized_best_smoke | Smoke (cấm luận văn) (web: —) | quarantined | (không có số liệu) | — |
| Smoke — fedavg_best_smoke | Smoke (cấm luận văn) (web: —) | quarantined | test_accuracy=1.0 | — |
| Smoke — fedavg_last_smoke | Smoke (cấm luận văn) (web: —) | quarantined | last_accuracy=1.0 | — |

### Checkpoint nằm ngoài danh sách model (nhóm có khai báo)

| Nhóm | Số file | Vai trò | Mô tả |
|---|---|---|---|
| `kaggle_run_lfs_pointer` | 6 | historical | Checkpoint run Kaggle canonical: chỉ còn con trỏ Git-LFS, bytes thật KHÔNG có trên máy (repo không có remote) — cần tải lại từ Kaggle |
| `report_experiments_smoke` | 36 | quarantined | Run report_experiments mode=smoke (dữ liệu tổng hợp, accuracy 0.0263) — cấm dùng trong luận văn |
| `web_runs` | 1 | technical_demo | Checkpoint từ run do web khởi động (smoke Flower hoặc demo huấn luyện); chưa có bằng chứng khoa học |

## 5. Chất lượng nhận diện (số liệu công bố, chưa tái đánh giá)

| Model | PlantVillage | PlantDoc | Chênh | Đúng cây (PlantDoc) | Sai mà confidence > 0.9 |
|---|---|---|---|---|---|
| Centralized v6 | 99.22% | 53.9% | 45.32 điểm | 78.81% | 98 |
| FedAvg v7 | 93.12% | 47.21% | 45.91 điểm | 75.46% | 100 |

- Audit ảnh ngoài (không có nhãn, chỉ dự đoán): 4 ảnh; Centralized và FedAvg **bất đồng trên 2/4** ảnh. Đây không phải accuracy.
- Nhầm lẫn lớn nhất của FedAvg v7 (metrics công bố): Grape___Black_rot → Grape___Esca_(Black_Measles) (55 ảnh); Tomato___Early_blight → Tomato___Late_blight (21 ảnh); Tomato___Bacterial_spot → Tomato___Early_blight (20 ảnh)
- Nhận định: khoảng cách PlantVillage↔PlantDoc ~45 điểm ở cả hai phương pháp ⇒ vấn đề là tổng quát hóa giữa miền dữ liệu, không phải riêng frontend. Chưa đủ bằng chứng để kết luận nguyên nhân (thiếu dữ liệu, nhãn, hay chiến lược train).
- Số liệu trên là kết quả công bố của bundle, **chưa tái đánh giá cục bộ** (evidence: published_artifact_hash_verified_not_locally_reevaluated).

**Vì sao Local-only sai: số ảnh train mỗi client có cho từng lớp cà chua (partition `label_alpha_0_1`. tổng 37.238 ảnh):**

| Lớp | c00 | c01 | c02 | c03 | c04 | Cả 5 |
|---|---:|---:|---:|---:|---:|---:|
| Bacterial_spot | 14 | 0 | 1549 | 17 | 17 | 1597 |
| Early_blight | 0 | 2 | 0 | 0 | 792 | 794 |
| Late_blight | 521 | 203 | 175 | 22 | 463 | 1384 |
| Leaf_Mold | 3 | 393 | 46 | 351 | 39 | 832 |
| Septoria_leaf_spot | 6 | 1258 | 0 | 11 | 180 | 1455 |
| Spider_mites Two-spotted_spider_mite | 0 | 4 | 0 | 0 | 1094 | 1098 |
| Target_Spot | 878 | 0 | 32 | 1 | 0 | 911 |
| Tomato_Yellow_Leaf_Curl_Virus | 0 | 548 | 39 | 2881 | 0 | 3468 |
| Tomato_mosaic_virus | 0 | 0 | 377 | 16 | 0 | 393 |
| healthy | 0 | 47 | 122 | 881 | 13 | 1063 |

- Mỗi lớp bệnh dồn gần như vào một client; Local-only không nhận ra bệnh mà shard của nó không có ảnh (khớp recall công bố, vd. Bacterial_spot: c02 có 1.549 ảnh và đạt 99%, bốn client còn lại 0%). Đây là minh họa trực tiếp cho giá trị của FedAvg.

**Theo lớp trên PlantDoc (FedAvg v7, số công bố; sắp theo recall tăng dần):**

- PlantDoc chỉ phủ **13/38** lớp của taxonomy; 6 lớp có recall < 50% chiếm 136/269 ảnh.
- Phân rã 269 ảnh: đúng 127 · sai loại cây 66 · đúng cây nhưng sai bệnh 76; sai với confidence > 0.9: 41 ảnh (29% số lỗi).

| Lớp | Ảnh PlantDoc | FedAvg PlantDoc | Centralized PlantDoc | FedAvg PlantVillage | Chênh (điểm) |
|---|---:|---:|---:|---:|---:|
| Tomato___Tomato_mosaic_virus | 29 | 0% | 66% | 83% | 83.33 |
| Tomato___Early_blight | 5 | 0% | 20% | 52% | 52.5 |
| Tomato___Bacterial_spot | 16 | 6% | 0% | 81% | 75.15 |
| Apple___Apple_scab | 25 | 44% | 32% | 85% | 41.42 |
| Potato___Late_blight | 22 | 45% | 45% | 95% | 49.55 |
| Tomato___Septoria_leaf_spot | 39 | 49% | 64% | 89% | 40.64 |
| Tomato___Leaf_Mold | 32 | 50% | 44% | 93% | 43.42 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 6 | 50% | 67% | 54% | 3.66 |
| Potato___Early_blight | 36 | 56% | 44% | 100% | 44.44 |
| Tomato___Late_blight | 20 | 60% | 60% | 96% | 36.05 |
| Grape___Black_rot | 6 | 67% | 67% | 38% | -29.17 |
| Corn_(maize)___Common_rust_ | 9 | 89% | 89% | 98% | 9.01 |
| Squash___Powdery_mildew | 24 | 96% | 100% | 100% | 4.17 |

- Không có trong số công bố (cần ảnh validation): confusion matrix đầy đủ trên ảnh PlantDoc; danh sách từng ảnh sai với confidence cao; calibration (ECE / temperature scaling) trên tập validation độc lập; tỷ lệ PlantDoc trong dữ liệu huấn luyện (bundle chỉ công bố validation_manifest, không kèm manifest train).

## 6. Gates kỹ thuật

| Gate | Kết quả | Chi tiết |
|---|---|---|
| Provenance checkpoint canonical | **unverified** | lỗi: fedavg_checkpoint_missing, centralized_checkpoint_missing |
| Pre-demo check | **technical_demo_ready** | artifacts/pre_demo_check.json |
| Final acceptance | technical **pass** / scientific **blocked** | 4 blocker khoa học |
| Network artifact | **CHƯA CÓ run mạng khoa học (artifact hiện tại là smoke kỹ thuật)** | artifacts/network_artifact_validation.json |
| Danh mục checkpoint | **complete** | 58 file .pt, vô danh: 0 |

## 7. Nhật ký (khối dưới đây TỰ TAY SỬA)

<!-- ROADMAP-BEGIN (khối này TỰ TAY SỬA - script bảo toàn nội dung) -->
### Nhật ký

- 2026-10-09: Đồng bộ hóa — lập registry (18 models/5 family), phát hiện & cách ly smoke
  ghi đè slot canonical, xác nhận checkpoint canonical KHÔNG còn trên máy (metrics-only).
- 2026-10-10: Đợt 2 — gom về MỘT hướng. (1) Pipeline Kaggle/stage* + `train/` trùng gom/xóa → `legacy/`, engine duy nhất là `src/`.
  (2) Registry v2: pipeline/role/dataset/taxonomy, danh mục mọi `.pt` trên đĩa, cổng `web_visible` 5 điều kiện tính tự động.
  (3) Web: `app.py` tách router + `model_service` (một cửa inference), thêm `/api/models`; `/api/weights/inspect` cùng cổng.
  (4) File này: checklist do máy chấm, mục BLOCKED kèm lệnh mở khóa, `--check` + hook pre-commit (`git config core.hooksPath scripts/hooks`).
  (5) Test: 5 test mạng từng ghi đè artifact đang track (checkpoint/metrics) nay dùng `tmp_path`; `attack_matrix` và
  `network_heldout` (lỗi sẵn có từ trước) sửa theo hợp đồng strict-round / partition contract hiện hành.
  (6) Review độc lập (harness-reviewer) 12 phát hiện → đã sửa các mục HIGH/MEDIUM.
- 2026-10-10 (tiếp): khai thác số công bố của bundle — `scripts/report_per_class.py` phân tích PlantDoc theo lớp
  (13/38 lớp được phủ; 6 lớp recall<50%; `Tomato___Tomato_mosaic_virus` 0% trên 29 ảnh; FedAvg tệ hơn Centralized ở 5/13 lớp;
  41/142 lỗi PlantDoc có confidence>0.9). Tách mục review thành "làm được từ số công bố" (XONG) và "cần ảnh" (BLOCKED).
- 2026-10-10 (review lại): (A) `/api/compare` và `/api/demo/evidence` chỉ ra số khi MỌI model của bundle qua cổng
  `web_visible` (không lọc nửa vời); ô chọn model trên UI dựng từ `GET /api/models` kèm nhãn. (B) bỏ bảng "60 passed/PASS"
  tĩnh trong TONG_HOP, `check_docs_numbers` chặn quay lại. (C) dependency khai báo một nơi (pyproject = requirements),
  ghim cũ ở `legacy/pyproject.toml`. (D) điều kiện taxonomy lấy từ `contract/taxonomy.py` của bundle, độc lập với manifest.
  Chưa kiểm được UI bằng trình duyệt thật (không có công cụ Chrome trong phiên này): đã kiểm HTTP 8 endpoint + cú pháp JS.
- 2026-10-10 (GitHub + server): đã đẩy nhánh `integration-gd1-gd2` + tag `v0.2.0-sync-20261010` lên
  `khanh51024/Federated-Learning-DoAn-TotNghiep` (remote có `main` riêng, lịch sử không liên quan nên KHÔNG đẩy `main`).
  Nhánh `train-*` của remote đã kiểm: có mã hợp đồng dữ liệu (ràng buộc group-disjoint giữa client) và hash release,
  KHÔNG có manifest ảnh train/val → mục audit leakage vẫn BLOCKED, cần manifest/dataset từ nhóm train.
  Server EC2 (đã chuyển 2026-10-10): bản mới ở `/home/ubuntu/fl-demo-20261010` (registry sinh trên server, 8/8 model qua cổng,
  canary 8091 đúng); `cropfed-demo.service` đã trỏ sang bản này, kiểm qua HTTPS: `/api/models` 8 model, `/api/compare` 14 dòng,
  suy luận FedAvg 200. Bản cũ `fl-demo-20261005` giữ nguyên; quay lui = chép `/home/ubuntu/cropfed-demo.service.bak-20261010`
  về `/etc/systemd/system/cropfed-demo.service` rồi `daemon-reload` + `restart`. Server dùng mã commit `fe8bdd3`
  (các commit sau chỉ đổi tài liệu/nội dung trạng thái).
- 2026-10-10 — số test legacy (`pytest legacy/tests`): 149 pass / 29 fail / 10 error. 22 fail + 10 error trùng baseline trước khi gom
  (thiếu PlantVillage gốc và `data/partitions_train_v*`, thiếu fixture `kaggle_calibration` bị `.gitignore *kaggle*`);
  thêm 7 test `stage1_compat` lỗi vì manifest hash upstream tính trên checkout CRLF còn cây này là LF (không do mã).
<!-- ROADMAP-END -->

## 8. Cách đồng bộ

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py              # chuẩn: registry + gates + file này
.\.venv\Scripts\python.exe scripts\update_status.py --skip-gates # chỉ sinh lại file từ JSON hiện có
.\.venv\Scripts\python.exe scripts\update_status.py --full-gates # + pytest toàn bộ (chậm)
.\.venv\Scripts\python.exe scripts\update_status.py --check      # kiểm file có đang lệch không (dùng cho pre-commit)
```

# PROJECT STATUS — Federated Learning PlantVillage

> **Cập nhật lúc:** 2026-10-19 10:42 · sinh tự bởi `scripts/update_status.py`
> **Đánh giá hiện tại:** DEMO KỸ THUẬT DÙNG ĐƯỢC — CHƯA nghiệm thu khoa học (3 mục BLOCKED cần dữ liệu/máy train, 0 mục kỹ thuật chưa XONG)
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

## 3. Checklist nghiệm thu — XONG 19/22 · CHƯA 0 · BLOCKED 3

| # | Hạng mục | Trạng thái | Bằng chứng (máy kiểm) |
|---|---|---|---|
| 1 | Registry v2 + danh mục mọi checkpoint trên đĩa | ✅ XONG | schema v2; 18 model; 61 file .pt đều có tên (15 trong model, 46 trong nhóm) |
| 2 | Smoke cách ly khỏi slot canonical | ✅ XONG | slot canonical không chứa smoke; 3 checkpoint smoke nằm ở artifacts/checkpoints/smoke/ |
| 3 | Một engine train (pipeline cũ gom vào legacy/) | ✅ XONG | engine duy nhất là src/; pipeline Kaggle/stage* nằm trong legacy/ (chạy riêng: pytest legacy/tests) |
| 4 | Web tách router/service, inference đi một cửa | ✅ XONG | app.py chỉ mount router; inference chỉ đi qua web/services/model_service.py (cổng web_visible) |
| 5 | Không hardcode đường dẫn artifact | ✅ XONG | mọi đường dẫn artifacts/ trong src/web/scripts lấy từ src/registry.py (quét AST) |
| 6 | Dọn tài liệu mâu thuẫn | ✅ XONG | gốc repo chỉ còn README.md + PROJECT_STATUS.md; docs/ chỉ gồm tài liệu hiện hành |
| 7 | Số liệu trong docs không mâu thuẫn artifact | ✅ XONG | không còn số liệu mâu thuẫn đã biết (97,12, 97.12...) và không ghi tĩnh kết quả test trong README/docs |
| 8 | Chốt một protocol khoa học cho mỗi thí nghiệm | ✅ XONG | mỗi protocol (scratch GĐ2 cục bộ / bundle mixed pretrained) có mục riêng trong EXPERIMENT_PROTOCOL.md |
| 9 | Test không ghi đè artifact đang track | ✅ XONG | test không tham chiếu artifacts/checkpoints/metrics (dùng tmp_path) |
| 10 | Dependency khai báo một nơi (pyproject = requirements) | ✅ XONG | 17 dependency trùng giữa pyproject.toml và requirements.txt; ghim cũ nằm ở legacy/pyproject.toml |
| 11 | Full tests pass trên mã nguồn hiện tại | ✅ XONG | technical_acceptance=pass lúc 2026-10-19T10:42:02 (gồm test mạng, 58 file test) với mã nguồn có dấu vân tay 2e2601f3b8d5; thay mã sau thời điểm này thì phải chạy lại |
| 12 | Kiểm chứng provenance & checkpoint canonical | ⛔ BLOCKED | provenance=unverified; lỗi: ['fedavg_checkpoint_missing', 'centralized_checkpoint_missing'] |
| 13 | Audit leakage theo từng run | ✅ XONG | release mixed_pv_pd_v3: train 37.238 / val 4.613 / calibration 4.225 rời nhau trên 5 khóa (giao nhau tổng = 0); 12 partition đều là tập con train; 46.076 ảnh khớp byte_sha256. Phạm vi: split của release, checkpoint ràng buộc qua metadata (chưa kiểm lại từng bước train) |
| 14 | Đánh giá PlantDoc theo từng lớp (từ số công bố) | ✅ XONG | PlantDoc phủ 13/38 lớp; 6 lớp recall<50% (136/269 ảnh); FedAvg tệ hơn Centralized ở 5/13 lớp (số công bố, đã kiểm nhất quán; xem mục 5) |
| 15 | Tái đánh giá cục bộ trên ảnh validation (xác nhận số công bố) | ✅ XONG | 4.613 ảnh validation tái đo trên CPU: Centralized v6 và FedAvg v7 khớp TUYỆT ĐỐI số công bố (số ảnh đúng, macro-F1, crop accuracy, recall theo lớp); tổng 10/14 phép so khớp tuyệt đối; 4 phép so của Local-only lệch ≤1 ảnh: Local 01/plantdoc (+0 ảnh), Local 01/plantvillage (+0 ảnh), Local 03/plantvillage (+1 ảnh), Local 04/plantvillage (-1 ảnh) — chưa rõ nguyên nhân, phù hợp sai số số học (ảnh sát ngưỡng) |
| 16 | Phân tích lỗi confidence cao (mức tổng hợp, từ số công bố) | ✅ XONG | mức tổng hợp từ metrics công bố — Centralized v6: 77/269 ảnh PlantDoc sai với confidence>0.9 (62% số lỗi); FedAvg v7: 41/269 ảnh PlantDoc sai với confidence>0.9 (29% số lỗi) (xem mục 5) |
| 17 | Phân tích từng ảnh sai tự tin + đo calibration (ECE) | ✅ XONG | FedAvg v7: ECE PlantDoc 0.295 → 0.261 sau temperature scaling (T=1.1488, fit trên 4.225 ảnh calibration tách biệt val); 100 ảnh validation sai với confidence ≥ 0,9 đã liệt kê từng ảnh. UI hiển thị softmax thô kèm xác suất đã hiệu chuẩn (T) |
| 18 | Đánh giá trên tập test giữ riêng (không dùng để chọn checkpoint) | ✅ XONG | 11.015 ảnh legacy_diagnostic_test (không dùng để chọn checkpoint): Centralized v6 PV 99.23% / PD 63.37%; FedAvg v7 PV 93.34% / PD 50.00%; Centralized so với số tự ghi trong summary.json: plantdoc khớp, plantvillage khớp |
| 19 | Khoảng tin cậy theo cụm + McNemar + độ nhạy 'cùng lá' | ✅ XONG | khoảng tin cậy 95% bootstrap theo cụm (cảnh/nhóm lá) — val: PlantDoc Centralized 53.9% [44.2%; 63.1%], Centralized−FedAvg +6.7% [-3.7%; +18.0%], McNemar p=0.0414 / test: PlantDoc Centralized 63.4% [52.9%; 73.7%], Centralized−FedAvg +13.4% [+1.6%; +24.8%], McNemar p=0.00762 |
| 20 | Chạy multi-seed non-IID (seed 42/123/2026) | ⛔ BLOCKED | chưa có multi-seed THẬT cho protocol đang công bố (mixed PV+PlantDoc, pretrained); chỉ có bản smoke trên ảnh tổng hợp |
| 21 | Đồng bộ báo cáo & artifacts sau khi có run mới | ⛔ BLOCKED | phụ thuộc multi-seed thật |
| 22 | Release sạch (mọi gate kỹ thuật XONG) | ✅ XONG | mọi mục kỹ thuật XONG; trước khi nộp vẫn phải `git status` sạch và gắn tag |

### Việc đang BLOCKED — cần gì và mở khóa bằng lệnh nào

**Kiểm chứng provenance & checkpoint canonical**
- Cần: HAI file checkpoint thật: FedAvg best.pt của run canonical (oid sha256:bdfe01879aef693d…) và Centralized best — best.pt trên máy này chỉ là con trỏ Git-LFS (object chưa tải về, không có trong .git/lfs/objects); thử `git lfs pull` trước, remote không còn giữ object thì phải tải lại từ Kaggle
- Mở khóa:

```powershell
# 0) thử kéo object LFS về (chỉ được nếu remote còn giữ):
git lfs pull
# 1) nếu vẫn thiếu: tải hai file từ Kaggle, 2) đặt đúng vào slot canonical (đường mà provenance validator và registry đọc):
Copy-Item <fedavg_best.pt tải về> artifacts/checkpoints/fedavg_best.pt
Copy-Item <centralized_best.pt tải về> artifacts/checkpoints/centralized_best.pt
.\.venv\Scripts\python.exe scripts\update_status.py   # so state-hash với canonical_manifest
```

**Chạy multi-seed non-IID (seed 42/123/2026)**
- Cần: GPU (Kaggle/Colab): train lại 7 mô hình × 3 seed với CÙNG optimizer và ngân sách (~10 GPU-giờ mỗi seed); máy này CPU, không train. Xem docs/EXPERIMENT_PROTOCOL.md mục 'Khi nào cần train lại'
- Mở khóa:

```powershell
# train ngoài theo protocol mixed_pv_pd_v3 rồi nhập bundle: python scripts/import_branch_demo.py (docs/MODEL_REGISTRY.md)
# LƯU Ý: scripts/run_real_multiseed_report.ps1 chạy protocol scratch PlantVillage CŨ, không thay được mục này
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
| `web_runs` | 4 | technical_demo | Checkpoint từ run do web khởi động (smoke Flower hoặc demo huấn luyện); chưa có bằng chứng khoa học |

## 5. Chất lượng nhận diện (số công bố, tái đánh giá cục bộ, test giữ riêng)

| Model | PlantVillage | PlantDoc | Chênh | Đúng cây (PlantDoc) | Sai mà confidence > 0.9 |
|---|---|---|---|---|---|
| Centralized v6 | 99.22% | 53.9% | 45.32 điểm | 78.81% | 98 |
| FedAvg v7 | 93.12% | 47.21% | 45.91 điểm | 75.46% | 100 |

- Audit ảnh ngoài (không có nhãn, chỉ dự đoán): 4 ảnh; Centralized và FedAvg **bất đồng trên 2/4** ảnh. Đây không phải accuracy.
- Nhầm lẫn lớn nhất của FedAvg v7 (metrics công bố): Grape___Black_rot → Grape___Esca_(Black_Measles) (55 ảnh); Tomato___Early_blight → Tomato___Late_blight (21 ảnh); Tomato___Bacterial_spot → Tomato___Early_blight (20 ảnh)
- Nhận định: khoảng cách PlantVillage↔PlantDoc ~45 điểm ở cả hai phương pháp ⇒ vấn đề là tổng quát hóa giữa miền dữ liệu, không phải riêng frontend. Chưa đủ bằng chứng để kết luận nguyên nhân (thiếu dữ liệu, nhãn, hay chiến lược train).
- Số liệu trên là số công bố của bundle; đã **tái đánh giá cục bộ** (bảng dưới) và chấm thêm trên tập test giữ riêng.

**Vì sao Local-only sai: số ảnh train mỗi client có cho từng lớp cà chua (partition `label_alpha_0_1`, tổng 37.238 ảnh):**

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

**Tái đánh giá CỤC BỘ trên CPU (4.613 ảnh validation; calibration 4.225 ảnh tách biệt val):**

| Model | PlantVillage | PlantDoc | Khớp số công bố | ECE PV (thô → TS) | ECE PlantDoc (thô → TS) | T |
|---|---:|---:|---|---:|---:|---:|
| Centralized v6 | 99.22% | 53.90% | PV tuyệt đối, PD tuyệt đối | 0.006 → 0.010 | 0.370 → 0.187 | 2.9264 |
| FedAvg v7 | 93.12% | 47.21% | PV tuyệt đối, PD tuyệt đối | 0.013 → 0.019 | 0.295 → 0.261 | 1.1488 |
| Local 00 | 44.89% | 25.65% | PV tuyệt đối, PD tuyệt đối | 0.419 → 0.063 | 0.543 → 0.087 | 3.3933 |
| Local 01 | 52.00% | 25.28% | PV +0 ảnh, PD +0 ảnh | 0.335 → 0.064 | 0.625 → 0.217 | 3.1411 |
| Local 02 | 55.34% | 29.00% | PV tuyệt đối, PD tuyệt đối | 0.327 → 0.100 | 0.578 → 0.169 | 3.1841 |
| Local 03 | 55.48% | 26.77% | PV +1 ảnh, PD tuyệt đối | 0.270 → 0.115 | 0.510 → 0.116 | 2.6643 |
| Local 04 | 55.16% | 21.19% | PV -1 ảnh, PD tuyệt đối | 0.333 → 0.082 | 0.625 → 0.183 | 3.2764 |

- Khớp tuyệt đối 10/14 phép so; Centralized và FedAvg khớp tuyệt đối trên cả hai nguồn. Chênh lệch (nếu có) chỉ ở Local-only và <= 1 ảnh. Mỗi chênh đều có ảnh ứng viên với độ lệch top-1/top-2 rất nhỏ (< 0,01); phù hợp với sai số số học giữa môi trường chạy, CHƯA được chứng minh.
- ECE đo mức lệch giữa độ tin cậy và độ đúng thật (càng thấp càng tốt); TS = temperature scaling với T fit trên tập calibration, KHÔNG fit trên validation. Temperature scaling không đổi nhãn dự đoán, chỉ đổi độ tin cậy hiển thị.
- 100 ảnh validation FedAvg v7 sai với confidence ≥ 0,9 đã liệt kê từng ảnh trong `artifacts/local_reeval_report.json` (mục `detail.fedavg.high_confidence_wrong`).
- Leakage (split của release `mixed_pv_pd_v3`): train/val/calibration giao nhau = 0; 46.076 ảnh khớp byte_sha256; 12 partition là tập con của train. Phạm vi: split của release; checkpoint ràng buộc release qua metadata.

**Tập test giữ riêng `legacy_diagnostic_test` (11.015 ảnh: PlantVillage 10.843, PlantDoc 172; không dùng để chọn checkpoint; [khoảng tin cậy 95% bootstrap theo cụm]):**

| Model | PlantVillage | PlantDoc |
|---|---:|---:|
| Centralized v6 | 99.23% [99.0%; 99.4%] | 63.37% [52.9%; 73.7%] |
| FedAvg v7 | 93.34% [92.7%; 94.0%] | 50.00% [38.4%; 61.4%] |
| Local 00 | 45.54% [43.4%; 47.8%] | 25.00% [17.2%; 34.2%] |
| Local 01 | 52.03% [50.0%; 54.1%] | 24.42% [14.5%; 34.8%] |
| Local 02 | 55.22% [53.2%; 57.3%] | 26.16% [15.7%; 39.0%] |
| Local 03 | 55.48% [53.4%; 57.6%] | 24.42% [17.3%; 32.3%] |
| Local 04 | 55.15% [53.1%; 57.2%] | 25.58% [16.6%; 35.3%] |
- Hiệu ứng chọn checkpoint: Centralized được chọn ở epoch 22/83 theo macro-F1 PlantDoc trên CHÍNH validation; accuracy PlantDoc ở epoch đó 53.90% là cao nhất (hạng 1/83), epoch cuối 51.67%. Chọn trên val có thể đẩy số val lên; số test giữ riêng ở bảng trên không dùng để chọn nên không chịu thiên lệch này (PlantDoc test chỉ 172 ảnh — khoảng tin cậy rộng).
- Độ nhạy 'cùng lá' (test, PlantVillage): 2.633 ảnh không được gom nhóm lá, 2.015 có ảnh train mang số liền kề (ước lượng chụp lại). Bỏ các ảnh đó: Centralized 99.23% → 99.32%, FedAvg 93.34% → 93.94% — không thấy dấu hiệu bị thổi phồng.

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
| Danh mục checkpoint | **complete** | 61 file .pt, vô danh: 0 |

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
- 2026-10-10 (dataset + giải thích Local-only): đã tải nhánh `dataset-sources` của GitHub vào `data/dataset_sources/` (gitignored,
  2,2 GB; 41.851 ảnh train+val khớp `byte_sha256` của manifest release `mixed_pv_pd_v3`). Audit leakage trên split của release:
  train và val không giao nhau ở group_id/scene/byte_sha256/sample_id; partition `label_alpha_0_1` và `iid` là tập con của train
  (số liệu mới, mục audit trong checklist chưa tự chuyển trạng thái — cần script audit riêng). Web hiện thêm "ảnh train của client
  này cho lớp được dự đoán / nhãn thật" (`shard_context`); đã đưa lên server `fl-demo-20261010` (commit 208fc76), sao lưu bản trước ở
  `/home/ubuntu/backup-20261010-shard`.
- 2026-10-10 (tái đánh giá cục bộ + audit leakage): `scripts/audit_release_leakage.py` — release `mixed_pv_pd_v3`: train/val/calibration
  rời nhau trên 5 khóa, 46.076 ảnh khớp byte_sha256, 12 partition là tập con train, không nhóm ảnh nào chia cho nhiều client;
  `RELEASE_SHA` của checkpoint khớp release_manifest.json khi thêm CRLF cuối file (git lưu LF). `scripts/evaluate_published_models.py` —
  7 model × 4.613 ảnh val trên CPU: Centralized và FedAvg khớp TUYỆT ĐỐI số công bố; 10/14 phép so khớp tuyệt đối, 4 phép của
  Local-only lệch ≤1 ảnh (nguyên nhân chưa chứng minh; có ảnh sát ngưỡng quyết định, phù hợp sai số số học). Calibration: PlantVillage
  đã tốt (ECE 0,006–0,013) nhưng PlantDoc rất tự tin quá mức (FedAvg ECE 0,295, TS chỉ hạ còn 0,261). Sai sót đã sửa trong lúc làm:
  một kiểm tra có lỗi (so tên nguồn viết thường với giá trị viết hoa) từng dẫn tới kết luận sai "giả thuyết bị bác bỏ", đã làm lại.
  Việc nặng (audit, tái đánh giá) KHÔNG nằm trong `update_status.py`; chạy tay theo `docs/RUNBOOK.md` mục 6c.
- 2026-10-10 — số test legacy (`pytest legacy/tests`): 149 pass / 29 fail / 10 error. 22 fail + 10 error trùng baseline trước khi gom
  (thiếu PlantVillage gốc và `data/partitions_train_v*`, thiếu fixture `kaggle_calibration` bị `.gitignore *kaggle*`);
  thêm 7 test `stage1_compat` lỗi vì manifest hash upstream tính trên checkout CRLF còn cây này là LF (không do mã).
- 2026-10-11 (làm lại giao diện web): chỉ đổi `web/static/` (+ `web/app.py` thêm mimetype woff2 và cache 1 ngày cho `/static/fonts`,
  `/static/vendor`); mọi ID, `data-tab`, handler và hợp đồng API giữ nguyên. Nền sáng kiểu báo cáo, phông Be Vietnam Pro tự host (có
  subset tiếng Việt), Chart.js 4.4.1 ghim và tự host (demo chạy được khi không có Internet), toast thay cả 7 `alert()`, thanh tiến
  trình/độ tin cậy đổi bằng `transform`, bỏ `backdrop-filter` và `transition: all`, poll 4 giây chỉ ghi DOM khi đổi và dừng khi tab ẩn.
  Kiểm bằng Chrome headless qua CDP: 6 tab, suy luận ảnh mẫu thật, 4 loại toast, không tràn ngang ở 390 px, không lỗi console,
  khung hình 16,7 ms ở render phần mềm (số trên GPU thật chưa đo). Reviewer độc lập: 14 phát hiện (không HIGH), đã sửa các mục đáng kể
  (đồng bộ badge khi chữ ký bảng không đổi, focus bàn phím cho vùng tải ảnh và sidebar mobile, tách lỗi mạng khỏi lỗi dựng giao diện…).
  Sửa kèm: bảng Weight Inspector ở tab Clients trước đây không bao giờ được nạp (hàm `loadLayerList` không ai gọi) — nay nạp khi mở tab.
  Test mới `tests/test_web_static_frontend.py` (9 test hợp đồng giao diện). Chưa đưa lên server public.
- 2026-10-11 (tái cấu trúc giao diện theo nghiệp vụ, 6 bước): sắp lại web theo trình tự bảo vệ — 1 Tổng quan, 2 Dữ liệu non-IID,
  3 Kết quả chi tiết, 4 Chẩn đoán bệnh lá, 5 Huấn luyện thử, 6 Bằng chứng & hạn chế (hash route `#tong-quan`…, chế độ trình chiếu phím `P`,
  `←` `→`). Backend thêm đọc-chỉ `GET /api/evidence/summary` (`web/services/evidence_service.py`): gộp báo cáo tái đánh giá, PlantDoc theo
  lớp, khoảng cách miền, audit rò rỉ, shard, checklist; mỗi phần fail-closed riêng và đối chiếu danh tính với bundle đã kiểm hash.
  Chẩn đoán chạy một ảnh qua cả 7 mô hình (6 lần gọi `/api/inference` hiện có), chấm đúng/sai theo nhãn thật và giải thích bằng số ảnh train
  của lớp đó trong shard; có thư viện 19 ảnh mẫu validation xếp dễ → khó (`scripts/build_demo_gallery.py`, `web/static/samples/demo/`).
  Gỡ `main.js`/`published_demo.js`; frontend là ES module (`js/app.js`, `js/core/`, `js/pages/`). Kiểm bằng Chrome headless: 6 bước, tải
  ảnh lên, chấm lại khi đổi nhãn, trình chiếu, không tràn ngang ở 390 px, không lỗi console. Bỏ hai biểu đồ luôn trống (hội tụ FedAvg v7,
  client lỗi) thay bằng ghi chú; thêm "chưa có thí nghiệm client lỗi" vào danh sách hạn chế. Chưa đưa lên server public.
  Review độc lập (harness-reviewer): 17 phát hiện, 2 mức cao — đã sửa hết. (1) checklist của `/api/evidence/summary` lộ số của mô hình
  (chuỗi `detail`) khi cổng registry chặn bộ đối chứng: nay chỉ trả id/tiêu đề/trạng thái. (2) sidebar `inert` không gỡ khi mở rộng
  cửa sổ. Kèm theo: service tính lại tổng "khớp tuyệt đối" từ từng phép so sánh thay vì tin báo cáo, bắt `passed` của audit phải khớp số
  bên trong, kiểm số ảnh/top-1 của báo cáo PlantDoc, NaN thành null thay vì 500; thư viện ảnh mẫu buộc `byte_sha256` + nhãn với
  `val.jsonl`; frontend chống đua khi chọn ảnh liên tiếp, bỏ qua poll cũ sau khi bấm Bắt đầu, phím mũi tên chỉ chuyển bước khi trình chiếu.
- 2026-10-11 (củng cố đánh giá bằng CPU + cập nhật giao diện lớn; KHÔNG train lại): (1) chấm 7 checkpoint trên tập test giữ riêng
  `legacy_diagnostic_test` của release (11.015 ảnh: PlantVillage 10.843, PlantDoc 172; chưa từng dùng để chọn checkpoint; hash manifest
  ghim trong `src/registry.py`) — Centralized PV 99,23% / PD 63,37% (tái lập đúng số nhóm train tự ghi trong `summary.json`), FedAvg
  93,34% / 50,00%, Local 45,5–55,5% / 24,4–26,2%. (2) `scripts/report_statistics.py`: khoảng tin cậy 95% bootstrap theo cụm (PlantDoc theo
  cảnh, PlantVillage theo nhóm lá), McNemar chính xác. PlantDoc Centralized − FedAvg: trên val CI chứa 0 (p=0,041), trên test +13,4 điểm,
  CI [1,7; 24,8], p=0,008. Hiệu ứng chọn checkpoint: epoch Centralized được chọn có PlantDoc val cao nhất 83 epoch (53,90%; epoch cuối
  51,67%). Bỏ ảnh PlantVillage nghi chụp lại cùng lá (số thứ tự liền kề ảnh train, ước lượng) không làm giảm accuracy. Hạn chế và tiêu chí
  "khi nào cần train lại" ghi ở `docs/EXPERIMENT_PROTOCOL.md` mục 7; mục multi-seed BLOCKED nay ghi đúng cách mở khóa (train lại protocol
  mixed trên GPU). (3) Backend: cache hash/mô hình theo (đường dẫn, kích thước, mtime) có khóa; `POST /api/inference/all` (mỗi mô hình
  chạy một lần, ~0,3 s/7 mô hình khi ấm trên 1 luồng; trước đây 6 lần gọi ~6,8 s); xác suất đã hiệu chuẩn (temperature scaling, không đổi
  dự đoán); Grad-CAM (`web/services/explain_service.py`, ~0,3 s thêm); `GET /api/evidence/details` (ma trận nhầm lẫn val+test, 175 ảnh
  "sai mà rất tự tin" — `scripts/build_error_gallery.py`) và `GET /api/evidence/partitions` (12 cách chia — `scripts/report_partition_schemes.py`);
  mọi phần fail-closed theo danh tính bundle. (4) Giao diện: bước 3 thêm tab Test giữ riêng & thống kê, Ma trận nhầm lẫn, Sai mà rất tự tin;
  bước 4 điều khiển lên đầu cột kết quả, một lần gọi cho 7 mô hình (máy chủ cũ trả 404 thì tự quay về cách gọi cũ), số thô + hiệu chuẩn,
  dải Grad-CAM, nút in; bước 2 trình xem 12 cách chia (chỉ α=0,1 có mô hình); bước 6 gập lệnh mở khóa, thẻ test giữ riêng, hạn chế lấy
  số từ báo cáo; nút nền sáng/tối (nhớ lựa chọn, biểu đồ dựng lại theo màu mới, in luôn dùng nền sáng). Sửa trong lúc làm: phép đối chiếu
  với `summary.json` từng báo "lệch −1 ảnh" vì coi 0,9923 là một số ảnh duy nhất — số làm tròn 4 chữ số ứng với 10.759–10.760 ảnh, nay so
  theo khoảng. Kiểm: pytest các phần liên quan, Chrome headless cả hai nền (tab mới, Grad-CAM 7 ảnh, đổi chế độ xem không gọi lại máy chủ,
  giả lập máy chủ cũ 404 → 6 lần gọi, 390 px không tràn ngang, không lỗi console), tương phản chữ trên ô heatmap/ma trận ≥ 5,5:1 ở cả hai nền.
  Review độc lập (harness-reviewer): 14 phát hiện, 1 mức cao — đã sửa. (1) Grad-CAM gắn hook lên mô hình dùng chung nên predict chạy
  song song (người dùng đổi ảnh khi Grad-CAM còn chạy, hai tab) lỗi 500: nay tính forward + `torch.autograd.grad` không hook (đã tái hiện:
  bản cũ 24 lỗi/30 lần, bản mới 0; có test chạy song song). (2) Lỗi Grad-CAM của một mô hình không còn làm mất 7 dự đoán. (3) Câu "số val
  lạc quan" trái với số đo (PlantDoc test cao hơn val) — viết lại theo số liệu, gọi đúng tên tập "legacy diagnostic test" (summary.json ghi
  `FINAL_HOLDOUT_PENDING`). (4) Test mới `tests/test_evidence_sections.py` cho các nhánh fail-closed. Kèm theo: nạp checkpoint băm chính các
  byte được giải nén, `report_statistics.py` tự băm manifest đối chiếu release, `details`/`partitions` hỏng trả `available: false` thay vì 500,
  checklist so checkpoint theo từng model_id, biểu đồ vẽ không animation khi in, bỏ `aria-pressed` sai nghĩa ở nút nền.
  Chưa đưa lên server public.
- 2026-10-12 (rà soát giao diện toàn diện, viết lại cho dễ hiểu): chỉ đổi `web/static/` + tài liệu. (1) Tiêu đề từng bước thành câu hỏi dẫn dắt,
  bước 1 mở bằng đề tài + ba bước FedAvg bằng lời thường; bước 6 sắp lại: đã kiểm chứng → hạn chế → việc còn lại → chi tiết kỹ thuật (gập).
  (2) Thuật ngữ thống nhất: tập kiểm tra (validation), tập giữ riêng (legacy diagnostic test), nhóm (38 lớp = 26 bệnh + 12 lá khỏe), chấm lại
  (tái đánh giá); bỏ shard/recall/canonical_v1/Telemetry; tên bệnh giữ nguyên. (3) Review độc lập (harness-reviewer) chỉ ra lời nói quá nên đã sửa:
  "tập độc lập" → "tập giữ riêng" kèm câu nói rõ đây là tập chẩn đoán "legacy", chưa phải test cuối khóa trước và chưa nằm trong phép kiểm tra
  trùng ảnh; khôi phục lời cảnh báo ở bước 1; câu "FedAvg gần bằng Centralized" thành câu theo số liệu; "mỗi nơi chỉ có vài loại bệnh" sửa theo
  dữ liệu (mỗi client có 20–27/38 nhóm); "bệnh này" sai với nhóm lá khỏe → "nhóm này"; việc còn lại gắn điều kiện theo trạng thái.
  (4) Số liệu dùng phông chính (tabular-nums), phông code chỉ cho mã băm; test mới chặn thuật ngữ cũ/lời nói quá quay lại. Kiểm bằng Chrome
  headless cả hai nền và 390 px (không tràn ngang, không lỗi console), tương phản chữ đạt ở cả hai nền.
- 2026-10-12 (làm lại giao diện: thanh bước ngang, phong cách "phòng nghiên cứu cây trồng", chuyển động có mục đích): chỉ đổi `web/static/`,
  test, tài liệu và `scripts/fetch_webfonts.py`; không đổi backend/API, không đổi tên bệnh. (1) Khung: thanh sáu bước nằm ngang thay sidebar
  (bước đã qua tô xanh, vạch chỉ báo chạy bằng transform, màn hẹp cuộn ngang và mờ mép), nền ngà + lưới chấm rất mờ, tiêu đề Source Serif 4,
  giữ cả nền sáng và nền tối; bỏ khoảng 100 lớp CSS chết. (2) Bước 1: sơ đồ FedAvg chuyển động một lượt mỗi lần vào (client hiện lần lượt, chấm
  trọng số chạy theo dây về máy chủ, mô hình chung hiện ra), nút "Xem lại", tắt khi giảm chuyển động; ánh sáng bám chuột chỉ trong hero và
  chỉ với chuột thật. (3) Tương tác: heatmap/ma trận nhầm lẫn sáng hàng + cột kèm tooltip (cảm ứng: chạm ô để xem), biểu đồ có cột mọc và
  tooltip theo token, bước 4 chuyển cảnh ảnh gốc → Grad-CAM của FedAvg (chỉ khi bật Grad-CAM), bước 5 sơ đồ bám `/api/status` thật (vòng sáng,
  dây sáng theo client đã kết nối, chấm chạy về máy chủ đúng lúc `round_completed` tăng; Centralized không bao giờ làm sơ đồ nhịp).
  (4) Review độc lập (harness-reviewer) tìm 3 lỗi phải sửa và 9 mục nên sửa — đã xử lý hết: hoạt ảnh hero bị ResizeObserver hủy khi quay lại
  bước 1; lời cảnh báo "một lần chạy / chưa phải ngoài thực địa" bị đẩy vào phần gập và chip tập giữ riêng còn xanh "đạt" (nay ngoài fold,
  chip vàng, gọi đúng bộ chẩn đoán legacy); sơ đồ bước 5 từng nhịp theo epoch Centralized với số client cũ của phiên Flower trước. Sửa thêm:
  `.hidden` bị luật component đè (nút chuyển ảnh Grad-CAM hiện sớm và làm bước 4 tràn 21 px trên 390 px), quầng bóng bị `scale()` phóng
  to trên dải sáng hàng/cột, đổi `font-weight` khi active làm layout lại, viền ô nhập 1,76:1 (nay dùng token ≥ 3:1), chữ trắng trên nền
  amber 3,9:1, mất tiêu điểm khi vào chế độ trình chiếu. Test: 22 test giao diện (token tương phản hai nền, lớp CSS chết thật sự, trạng thái
  không đổi số đo hộp/chữ, chuyển động chỉ transform/opacity, cảnh báo nằm ngoài fold…), đã thử đột biến 9 vi phạm cố ý đều bị bắt.
  Đo bằng Chrome headless (vẽ phần mềm, vsync 60 Hz — cho thấy không có khung bị chặn, chưa phải số liệu GPU): sơ đồ hero, chuyển bước ×6,
  rê chuột 60 ô heatmap và ma trận, tạo biểu đồ, cuộn 2.000 px, chạy chẩn đoán 7 mô hình + Grad-CAM đều p95 16,7–16,8 ms, khung dài
  nhất 16,8 ms, 0 khung > 25 ms, 0 long task; layout cộng dồn 6–8 ms cho 60 lần rê chuột; hero/rê chuột 0 hoạt ảnh không-composite
  (chuyển bước còn các transition màu của thanh bước — chỉ vẽ lại, không layout). CHƯA kiểm: phiên huấn luyện Flower thật (máy 16 GB còn
  ~2 GB trống, trước đó server bị hệ điều hành tắt 4 lần) — sơ đồ bước 5 được kiểm bằng `/api/status` giả lập (idle → chạy → vòng xong
  với 5 và 4 client → hoàn tất → Centralized với số cũ); trình duyệt có giao diện và thiết bị cảm ứng thật; không deploy server.
- 2026-10-12 (ảnh ngoài đời, bước cắt lá): trả lời "99% có thật không / có cần train lại / ảnh trên mạng có nhận được không" bằng số đo, không
  bằng suy đoán. (1) Chấm lại độc lập bằng mã viết mới (không dùng `web/` hay `evaluate_published_models.py`): 7 mô hình × 4.613 ảnh kiểm tra và
  × 11.015 ảnh giữ riêng khớp số web hiển thị (12/14 ô khớp số công bố tuyệt đối, 2 ô lệch đúng 1 ảnh — đã được web ghi); ma trận nhầm lẫn, recall
  từng lớp, 7/7 nhiệt độ T, ECE, 175 ảnh sai-mà-tự-tin, 19 ảnh mẫu đều khớp; 99,22% PlantVillage là thật và không rò rỉ (giao nhau train/val/giữ
  riêng theo mã băm, ảnh, nhóm lá, cảnh = 0). (2) Điểm yếu thật là ảnh ngoài đồng: PlantDoc 47–63%. `scripts/eval_fullframe_plantdoc.py` (ảnh gốc
  tải về, kiểm byte_sha256; 208/211 cảnh, 3 cảnh 404 bị loại và ghi rõ): đưa cả khung ảnh thay vì ảnh cắt lá làm FedAvg tụt từ ~49–50% xuống
  ~30% và Centralized từ 55–62% xuống 45–56%. Số này đọc từ `artifacts/fullframe_eval.json` qua `/api/evidence/summary` (phần `fullframe`,
  fail-closed theo danh tính release và SHA-256 checkpoint) và hiện ở bước 6 trong 5 hạn chế đầu. (3) `scripts/train_leaf_detector.py`:
  Faster R-CNN MobileNetV3 một lớp, chỉ học trên cảnh `split == "train"`; mới thử đường ống trên CPU (12 ảnh, 1 epoch), CHƯA có detector đã
  huấn luyện thật nên chưa tích hợp vào web; tiêu chí chấp nhận đặt trước trong `docs/NEXT_TRAINING_PLAN.md`. (4) Sửa 3 câu chữ web nói quá
  hoặc chưa kiểm chứng (nguồn số vòng FedAvg: 32 vòng chạy, chọn vòng 27, log chỉ có 27 vòng; bỏ "15–25 giây" chưa từng đo). Train lại ba cách
  CHƯA làm (cần GPU): chỉ đáng nếu muốn kết luận FedAvg/Centralized công bằng hơn hoặc tăng ảnh ngoài đồng.
- 2026-10-12 (dữ liệu ngoài đồng bị cách ly): `quarantine.jsonl` của release v3 giữ 5.569 lá PlantDoc có nhãn taxonomy (clean; nhãn
  ngầm định assumed_healthy 4.019 / ambiguous_alias 1.550) thuộc 14 nhóm KHÁC HẲN 14 nhóm đang có ảnh ngoài đồng trong tập học (giao
  nhau = 0): nếu được nhận thì 14 → 28/38 nhóm, còn 10 nhóm không có. `scripts/eval_quarantined_plantdoc.py` (ảnh khớp byte_sha256):
  7 checkpoint hiện có chỉ khớp nhãn ngầm định ở 3.4% (Centralized) và 2.1% (FedAvg), hầu hết nhóm gần 0% —
  KHÔNG phải độ chính xác thật vì nhãn chưa được người kiểm (xem ngẫu nhiên 24 ảnh thì phần lớn trông đúng nhãn). `scripts/
  build_label_review.py` dựng trang kiểm nhãn (14 nhóm × 30 lá, seed cố định, quy tắc nhận ≥ 90% đúng và ≤ 20% không rõ đặt trước);
  chưa có kết quả kiểm nên CHƯA dựng release v4 và CHƯA train lại. Rà mã runner `train-*`: Centralized chỉ chấp nhận AdamW, tỉ lệ
  PlantDoc chỉ 0/0,25 theo spec đã duyệt, lịch lr/dừng sớm khác nhau giữa hai runner → so sánh công bằng cần sửa mã, nên đề xuất
  tách hai giai đoạn (A: chỉ đổi dữ liệu; B: cùng bộ tối ưu và ngân sách) trong `docs/NEXT_TRAINING_PLAN.md`. Sửa số trong kế
  hoạch trước: "29/38, thiếu 9" là sai do tính nhầm lá không nhãn thành một nhóm; đúng là 28/38, thiếu 10.
<!-- ROADMAP-END -->

## 8. Cách đồng bộ

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py              # chuẩn: registry + gates + file này
.\.venv\Scripts\python.exe scripts\update_status.py --skip-gates # chỉ sinh lại file từ JSON hiện có
.\.venv\Scripts\python.exe scripts\update_status.py --full-gates # + pytest toàn bộ (chậm)
.\.venv\Scripts\python.exe scripts\update_status.py --check      # kiểm file có đang lệch không (dùng cho pre-commit)
```

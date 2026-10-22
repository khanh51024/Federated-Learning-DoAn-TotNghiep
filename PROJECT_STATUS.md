# PROJECT STATUS — Federated Learning PlantVillage

> **Cập nhật lúc:** 2026-10-22 15:32 · sinh tự bởi `scripts/update_status.py`
> **Đánh giá hiện tại:** DEMO KỸ THUẬT DÙNG ĐƯỢC — CHƯA nghiệm thu khoa học (0 mục BLOCKED cần dữ liệu/máy train, 0 mục kỹ thuật chưa XONG)
>
> File này là nguồn thống kê DUY NHẤT về tiến độ. Checklist do máy chấm từ sự thật trên đĩa;
> con số nào trong repo phải truy được về `artifacts/registry.json`. Không sửa tay — chỉ khối Nhật ký cuối file.

## 1. Hướng chính thức

- **Luồng một chiều:** train ngoài repo (Kaggle/branch) -> bundle + manifest + SHA-256 -> import/kiểm hash -> registry -> status/web/báo cáo
- **Kết quả hiện hành:** họ `published_demo`, protocol `mixed_pv_pd_v4_pretrained_seed42_validation`.
- **Engine trong repo:** `src/` (Flower demo, evaluator, web). Pipeline Kaggle cũ nằm ở `legacy/` — chỉ để tra cứu/tái lập lịch sử.
- **Máy này:** CPU, không train; chỉ nhận kết quả đã train rồi kiểm hash + test.

## 2. Trạng thái hệ thống (sự thật từ registry)

- Registry: **18** model (scientific_canonical: 3, published_demo: 7, network_demo: 4, reference: 1, smoke: 3)
- Web được phép dùng: **8** model (đã qua đủ 5 điều kiện: hash, taxonomy, preprocessing, metrics↔protocol, forward test)
- Dataset cục bộ để tái lập: `data/partitions_stage2_scratch_v3` — split **38984/4400/10921** (5 clients, seed 42, protocol `stage2_scratch_clean_v3`)
- Run Kaggle lịch sử: `20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62` — metrics CÓ, **checkpoint KHÔNG còn trên máy**

## 3. Checklist nghiệm thu — XONG 22/22 · CHƯA 0 · BLOCKED 0

| # | Hạng mục | Trạng thái | Bằng chứng (máy kiểm) |
|---|---|---|---|
| 1 | Registry v2 + danh mục mọi checkpoint trên đĩa | ✅ XONG | schema v2; 18 model; 100 file .pt đều có tên (15 trong model, 85 trong nhóm) |
| 2 | Smoke cách ly khỏi slot canonical | ✅ XONG | slot canonical không chứa smoke; 3 checkpoint smoke nằm ở artifacts/checkpoints/smoke/ |
| 3 | Một engine train (pipeline cũ gom vào legacy/) | ✅ XONG | engine duy nhất là src/; pipeline Kaggle/stage* nằm trong legacy/ (chạy riêng: pytest legacy/tests) |
| 4 | Web tách router/service, inference đi một cửa | ✅ XONG | app.py chỉ mount router; inference chỉ đi qua web/services/model_service.py (cổng web_visible) |
| 5 | Không hardcode đường dẫn artifact | ✅ XONG | mọi đường dẫn artifacts/ trong src/web/scripts lấy từ src/registry.py (quét AST) |
| 6 | Dọn tài liệu mâu thuẫn | ✅ XONG | gốc repo chỉ còn README.md + PROJECT_STATUS.md; docs/ chỉ gồm tài liệu hiện hành |
| 7 | Số liệu trong docs không mâu thuẫn artifact | ✅ XONG | không còn số liệu mâu thuẫn đã biết (97,12, 97.12...) và không ghi tĩnh kết quả test trong README/docs |
| 8 | Chốt một protocol khoa học cho mỗi thí nghiệm | ✅ XONG | mỗi protocol (scratch GĐ2 cục bộ / bundle mixed pretrained) có mục riêng trong EXPERIMENT_PROTOCOL.md |
| 9 | Test không ghi đè artifact đang track | ✅ XONG | test không tham chiếu artifacts/checkpoints/metrics (dùng tmp_path) |
| 10 | Dependency khai báo một nơi (pyproject = requirements) | ✅ XONG | 17 dependency trùng giữa pyproject.toml và requirements.txt; ghim cũ nằm ở legacy/pyproject.toml |
| 11 | Full tests pass trên mã nguồn hiện tại | ✅ XONG | technical_acceptance=pass lúc 2026-10-22T15:32:22 (gồm test mạng, 91 file test) với mã nguồn có dấu vân tay 7dca56bd0eb0; thay mã sau thời điểm này thì phải chạy lại |
| 12 | Kiểm chứng provenance & checkpoint canonical | ✅ XONG | gói v4b_s42: con trỏ ghim mã băm manifest; 7 checkpoint khớp artifact_manifest; release 68e574519a1a… và validation khớp; mã huấn luyện 9dcbcbf6684d… (commit 93c5f80fd); W0 tính lại từ hạt giống 42. Slot canonical của pipeline Kaggle cũ không còn checkpoint nhưng không phục vụ web. |
| 13 | Audit leakage theo từng run | ✅ XONG | release mixed_pv_pd_v4: train 52.464 / val 6.746 / calibration 7.005 rời nhau trên 5 khóa (giao nhau tổng = 0); 2 partition đều là tập con train; 66.215 ảnh khớp byte_sha256. Phạm vi: split của release, checkpoint ràng buộc qua metadata (chưa kiểm lại từng bước train) |
| 14 | Đánh giá PlantDoc theo từng lớp (từ số công bố) | ✅ XONG | PlantDoc phủ 13/38 lớp; 8 lớp recall<50% (204/269 ảnh); FedAvg tệ hơn Centralized ở 9/13 lớp (số công bố, đã kiểm nhất quán; xem mục 5) |
| 15 | Tái đánh giá cục bộ trên ảnh validation (xác nhận số công bố) | ✅ XONG | 6.746 ảnh validation tái đo trên CPU: Centralized v4b và FedAvg v4b khớp TUYỆT ĐỐI số ghi trong gói (số ảnh đúng, macro-F1, crop accuracy, recall theo lớp); tổng 21/21 phép so khớp tuyệt đối |
| 16 | Phân tích lỗi confidence cao (mức tổng hợp, từ số công bố) | ✅ XONG | mức tổng hợp từ metrics công bố — Centralized v4b: 8/269 ảnh PlantDoc sai với confidence>0.9 (7% số lỗi); FedAvg v4b: 0/269 ảnh PlantDoc sai với confidence>0.9 (0% số lỗi) (xem mục 5) |
| 17 | Phân tích từng ảnh sai tự tin + đo calibration (ECE) | ✅ XONG | FedAvg v4b: ECE PlantDoc 0.077 → 0.245 sau temperature scaling (T=0.5588, fit trên 7.005 ảnh calibration tách biệt val); 1 ảnh validation sai với confidence ≥ 0,9 đã liệt kê từng ảnh. UI hiển thị softmax thô kèm xác suất đã hiệu chuẩn (T) |
| 18 | Đánh giá trên tập test giữ riêng (không dùng để chọn checkpoint) | ✅ XONG | 13.101 ảnh legacy_diagnostic_test (không dùng để chọn checkpoint): Centralized v4b PV 99.32% / PD 59.88%; FedAvg v4b PV 96.84% / PD 40.12%; Centralized so với số tự ghi trong summary.json: plantdoc +2 ảnh, plantvillage +13 ảnh, field_other không suy ra được số ảnh đúng |
| 19 | Khoảng tin cậy theo cụm + McNemar + độ nhạy 'cùng lá' | ✅ XONG | khoảng tin cậy 95% bootstrap theo cụm (cảnh/nhóm lá) — val: PlantDoc Centralized 56.5% [47.4%; 66.0%], Centralized−FedAvg +19.0% [+11.2%; +26.9%], McNemar p=5.19e-09 / test: PlantDoc Centralized 59.9% [51.2%; 68.8%], Centralized−FedAvg +19.8% [+10.4%; +30.0%], McNemar p=1.21e-05 |
| 20 | Chạy multi-seed non-IID (seed 42/123/2026) | ✅ XONG | 3 hạt giống [42, 123, 2026] × 7 mô hình (Centralized, FedAvg, 5 Local-only) cùng release 68e574519a1a…, cùng thiết lập huấn luyện; mean ± độ lệch chuẩn ở trang Kết quả (artifacts/train_v4_results.json). 7/14 ngưỡng đặt trước đạt; chưa đạt: B_centralized_plantwild_65, web1_centralized_plantwild_plus10, web2_centralized_plantdoc_holdout, A_fedavg_field_same_source, B_fedavg_plantwild_65, web1_fedavg_plantwild_plus10, web2_fedavg_plantdoc_holdout. |
| 21 | Đồng bộ báo cáo & artifacts sau khi có run mới | ✅ XONG | 5 báo cáo (tái đánh giá, test giữ riêng, thống kê, cấu hình huấn luyện, shard) khớp protocol và release của gói v4b_s42 |
| 22 | Release sạch (mọi gate kỹ thuật XONG) | ✅ XONG | mọi mục kỹ thuật XONG; trước khi nộp vẫn phải `git status` sạch và gắn tag |

### Việc làm được ngay (chưa BLOCKED)

_(không còn mục kỹ thuật nào mở)_

## 4. Kết quả chuẩn theo họ (không trộn lẫn)

### Hiện hành — bundle đã kiểm hash (pretrained, mixed PV+PlantDoc)

_Checkpoint pretrained mixed PlantVillage+PlantDoc (protocol mixed_pv_pd_v4_pretrained_seed42_validation), hash-verified khi import. Dùng cho demo web/inference, KHÔNG phải kết quả khoa học của protocol GĐ2 scratch._

| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |
|---|---|---|---|---|
| Centralized v4b (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 56.51% (n=269); PlantVillage 99.54% (n=4344) | artifacts/model_bundles/v4b_s42/centralized/validation_metrics.json |
| FedAvg v4b (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 37.55% (n=269); PlantVillage 96.71% (n=4344) | artifacts/model_bundles/v4b_s42/fedavg/validation_metrics.json |
| Local-only client 00 (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 21.56% (n=269); PlantVillage 51.82% (n=4344) | artifacts/model_bundles/v4b_s42/local/client_00/validation_metrics.json |
| Local-only client 01 (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 17.10% (n=269); PlantVillage 48.85% (n=4344) | artifacts/model_bundles/v4b_s42/local/client_01/validation_metrics.json |
| Local-only client 02 (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 29.74% (n=269); PlantVillage 29.24% (n=4344) | artifacts/model_bundles/v4b_s42/local/client_02/validation_metrics.json |
| Local-only client 03 (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 27.14% (n=269); PlantVillage 53.02% (n=4344) | artifacts/model_bundles/v4b_s42/local/client_03/validation_metrics.json |
| Local-only client 04 (mô hình web, đã đánh giá lại cục bộ) | Mô hình web (hash-verified, đánh giá lại cục bộ) (web: ✅) | ✅ có, hash ok | PlantDoc 7.43% (n=269); PlantVillage 55.57% (n=4344) | artifacts/model_bundles/v4b_s42/local/client_04/validation_metrics.json |

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
| `branch_demo_inactive` | 7 | historical | Gói mô hình cũ (Centralized v6, FedAvg v7, Local 00-04 từ ba nhánh train) không phải gói web đang dùng; giữ để lùi (swap_web_models.py --rollback) |
| `kaggle_run_lfs_pointer` | 6 | historical | Checkpoint run Kaggle canonical: chỉ còn con trỏ Git-LFS, bytes thật KHÔNG có trên máy (repo không có remote) — cần tải lại từ Kaggle |
| `report_experiments_smoke` | 36 | quarantined | Run report_experiments mode=smoke (ảnh nhiễu ngẫu nhiên, accuracy 0.0263) — cấm dùng trong luận văn |
| `train_v4_runs` | 21 | technical_demo | Checkpoint đợt train v4 (scripts/train_v4.py, máy GPU thuê) chưa đánh giá/promote — số liệu chỉ dùng sau scripts/eval_trained_v4.py |
| `web_runs` | 15 | technical_demo | Checkpoint từ run do web khởi động (smoke Flower hoặc demo huấn luyện); chưa có bằng chứng khoa học |

## 5. Chất lượng nhận diện (số công bố, tái đánh giá cục bộ, test giữ riêng)

| Model | PlantVillage | PlantDoc | Chênh | Đúng cây (PlantDoc) | Sai mà confidence > 0.9 |
|---|---|---|---|---|---|
| Centralized v4b | 99.54% | 56.51% | 43.03 điểm | 80.3% | 10 |
| FedAvg v4b | 96.71% | 37.55% | 59.16 điểm | 67.66% | 1 |

- Audit ảnh ngoài (không có nhãn, chỉ dự đoán): 4 ảnh; Centralized và FedAvg **bất đồng trên 2/4** ảnh. Đây không phải accuracy.
- Nhầm lẫn lớn nhất của FedAvg v4b (metrics ghi trong gói): Corn_(maize)___Northern_Leaf_Blight → Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot (52 ảnh); Tomato___Late_blight → Tomato___Early_blight (24 ảnh); Tomato___Septoria_leaf_spot → Tomato___Early_blight (22 ảnh)
- Nhận định: khoảng cách PlantVillage↔PlantDoc ~45 điểm ở cả hai phương pháp ⇒ vấn đề là tổng quát hóa giữa miền dữ liệu, không phải riêng frontend. Chưa đủ bằng chứng để kết luận nguyên nhân (thiếu dữ liệu, nhãn, hay chiến lược train).
- Số liệu trên là số công bố của bundle; đã **tái đánh giá cục bộ** (bảng dưới) và chấm thêm trên tập test giữ riêng.

**Vì sao Local-only sai: số ảnh train mỗi client có cho từng lớp cà chua (partition `label_alpha_0_1`, tổng 52.464 ảnh):**

| Lớp | c00 | c01 | c02 | c03 | c04 | Cả 5 |
|---|---:|---:|---:|---:|---:|---:|
| Bacterial_spot | 4 | 355 | 36 | 4 | 1624 | 2023 |
| Early_blight | 0 | 0 | 3 | 738 | 779 | 1520 |
| Late_blight | 1951 | 0 | 0 | 0 | 27 | 1978 |
| Leaf_Mold | 0 | 1507 | 0 | 4 | 0 | 1511 |
| Septoria_leaf_spot | 0 | 0 | 1826 | 53 | 46 | 1925 |
| Spider_mites Two-spotted_spider_mite | 0 | 83 | 69 | 0 | 1134 | 1286 |
| Target_Spot | 258 | 646 | 4 | 0 | 3 | 911 |
| Tomato_Yellow_Leaf_Curl_Virus | 3776 | 0 | 4 | 6 | 1 | 3787 |
| Tomato_mosaic_virus | 0 | 129 | 251 | 0 | 13 | 393 |
| healthy | 0 | 2028 | 1 | 0 | 94 | 2123 |

- Mỗi lớp bệnh dồn gần như vào một client; Local-only không nhận ra bệnh mà shard của nó không có ảnh (khớp recall công bố, vd. Bacterial_spot: c02 có 1.549 ảnh và đạt 99%, bốn client còn lại 0%). Đây là minh họa trực tiếp cho giá trị của FedAvg.

**Tái đánh giá CỤC BỘ trên CPU (6.746 ảnh validation; calibration 7.005 ảnh tách biệt val):**

| Model | PlantVillage | PlantDoc | Khớp số công bố | ECE PV (thô → TS) | ECE PlantDoc (thô → TS) | T |
|---|---:|---:|---|---:|---:|---:|
| Centralized v4b | 99.54% | 56.51% | PV tuyệt đối, PD tuyệt đối | 0.106 → 0.014 | 0.094 → 0.214 | 0.694 |
| FedAvg v4b | 96.71% | 37.55% | PV tuyệt đối, PD tuyệt đối | 0.227 → 0.026 | 0.077 → 0.245 | 0.5588 |
| Local 00 | 51.82% | 21.56% | PV tuyệt đối, PD tuyệt đối | 0.140 → 0.130 | 0.355 → 0.189 | 1.3258 |
| Local 01 | 48.85% | 17.10% | PV tuyệt đối, PD tuyệt đối | 0.170 → 0.166 | 0.381 → 0.217 | 1.3287 |
| Local 02 | 29.24% | 29.74% | PV tuyệt đối, PD tuyệt đối | 0.315 → 0.163 | 0.358 → 0.114 | 1.5022 |
| Local 03 | 53.02% | 27.14% | PV tuyệt đối, PD tuyệt đối | 0.180 → 0.105 | 0.323 → 0.155 | 1.3219 |
| Local 04 | 55.57% | 7.43% | PV tuyệt đối, PD tuyệt đối | 0.140 → 0.147 | 0.440 → 0.262 | 1.3646 |

- Khớp tuyệt đối 21/21 phép so; Centralized và FedAvg khớp tuyệt đối trên cả hai nguồn. Chênh lệch (nếu có) chỉ ở Local-only và <= 1 ảnh. Mỗi chênh đều có ảnh ứng viên với độ lệch top-1/top-2 rất nhỏ (< 0,01); phù hợp với sai số số học giữa môi trường chạy, CHƯA được chứng minh.
- ECE đo mức lệch giữa độ tin cậy và độ đúng thật (càng thấp càng tốt); TS = temperature scaling với T fit trên tập calibration, KHÔNG fit trên validation. Temperature scaling không đổi nhãn dự đoán, chỉ đổi độ tin cậy hiển thị.
- 1 ảnh validation FedAvg v4b sai với confidence ≥ 0,9 đã liệt kê từng ảnh trong `artifacts/local_reeval_report.json` (mục `detail.fedavg.high_confidence_wrong`).
- Leakage (split của release `mixed_pv_pd_v4`): train/val/calibration giao nhau = 0; 66.215 ảnh khớp byte_sha256; 2 partition là tập con của train. Phạm vi: split của release; checkpoint ràng buộc release qua metadata.

**Tập test giữ riêng `legacy_diagnostic_test` (13.101 ảnh: PlantVillage 10.843, PlantDoc 172; không dùng để chọn checkpoint; [khoảng tin cậy 95% bootstrap theo cụm]):**

| Model | PlantVillage | PlantDoc |
|---|---:|---:|
| Centralized v4b | 99.32% [99.1%; 99.5%] | 59.88% [51.2%; 68.8%] |
| FedAvg v4b | 96.84% [96.3%; 97.3%] | 40.12% [30.4%; 50.6%] |
| Local 00 | 52.68% [50.4%; 54.9%] | 11.05% [5.6%; 17.7%] |
| Local 01 | 49.19% [47.1%; 51.4%] | 23.84% [14.9%; 33.3%] |
| Local 02 | 29.09% [26.9%; 31.3%] | 32.56% [19.3%; 46.2%] |
| Local 03 | 53.51% [51.4%; 55.6%] | 34.30% [23.9%; 46.2%] |
| Local 04 | 55.11% [52.9%; 57.3%] | 17.44% [10.4%; 26.3%] |
- Hiệu ứng chọn checkpoint: Centralized được chọn ở epoch 33/40 theo mean macro-F1 over val sources with >= 30 images trên CHÍNH validation; macro-F1 PlantDoc ở epoch đó 59.03% (hạng 6/40), epoch cuối 59.07%. Chọn trên val có thể đẩy số val lên; số test giữ riêng ở bảng trên không dùng để chọn nên không chịu thiên lệch này (PlantDoc test chỉ 172 ảnh — khoảng tin cậy rộng).
- Độ nhạy 'cùng lá' (test, PlantVillage): 2.633 ảnh không được gom nhóm lá, 2.015 có ảnh train mang số liền kề (ước lượng chụp lại). Bỏ các ảnh đó: Centralized 99.32% → 99.29%, FedAvg 96.84% → 97.02% — không thấy dấu hiệu bị thổi phồng.

**Theo lớp trên PlantDoc (FedAvg v4b, số ghi trong gói; sắp theo recall tăng dần):**

- PlantDoc chỉ phủ **13/38** lớp của taxonomy; 8 lớp có recall < 50% chiếm 204/269 ảnh.
- Phân rã 269 ảnh: đúng 101 · sai loại cây 87 · đúng cây nhưng sai bệnh 81; sai với confidence > 0.9: 0 ảnh (0% số lỗi).

| Lớp | Ảnh PlantDoc | FedAvg PlantDoc | Centralized PlantDoc | FedAvg PlantVillage | Chênh (điểm) |
|---|---:|---:|---:|---:|---:|
| Tomato___Tomato_mosaic_virus | 29 | 0% | 34% | 100% | 100.0 |
| Tomato___Early_blight | 5 | 0% | 20% | 74% | 73.75 |
| Tomato___Bacterial_spot | 16 | 6% | 0% | 96% | 89.68 |
| Potato___Early_blight | 36 | 28% | 42% | 99% | 70.97 |
| Apple___Apple_scab | 25 | 28% | 56% | 94% | 65.75 |
| Tomato___Leaf_Mold | 32 | 34% | 62% | 93% | 59.05 |
| Tomato___Septoria_leaf_spot | 39 | 36% | 67% | 79% | 42.83 |
| Potato___Late_blight | 22 | 41% | 45% | 91% | 50.34 |
| Tomato___Late_blight | 20 | 50% | 85% | 90% | 40.13 |
| Grape___Black_rot | 6 | 67% | 67% | 100% | 33.33 |
| Squash___Powdery_mildew | 24 | 88% | 96% | 100% | 12.5 |
| Corn_(maize)___Common_rust_ | 9 | 89% | 89% | 100% | 11.11 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 6 | 100% | 67% | 93% | -7.32 |

- Không có trong số công bố (cần ảnh validation): confusion matrix đầy đủ trên ảnh PlantDoc; danh sách từng ảnh sai với confidence cao; calibration (ECE / temperature scaling) trên tập validation độc lập; tỷ lệ PlantDoc trong dữ liệu huấn luyện (bundle chỉ công bố validation_manifest, không kèm manifest train).

## 6. Gates kỹ thuật

| Gate | Kết quả | Chi tiết |
|---|---|---|
| Provenance checkpoint canonical | **unverified** | lỗi: fedavg_checkpoint_missing, centralized_checkpoint_missing |
| Pre-demo check | **technical_demo_ready** | artifacts/pre_demo_check.json |
| Final acceptance | technical **pass** / scientific **blocked** | 4 blocker khoa học |
| Network artifact | **CHƯA CÓ run mạng khoa học (artifact hiện tại là smoke kỹ thuật)** | artifacts/network_artifact_validation.json |
| Danh mục checkpoint | **complete** | 100 file .pt, vô danh: 0 |

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
- 2026-10-21: Đánh giá dữ liệu, ảnh ngoài đồng và trao đổi trọng số (`docs/DANH_GIA_DU_LIEU_MO_HINH_FEDAVG_2026-10-21.md`), không train lại.
  (1) Sửa giải mã ảnh trên web: xoay theo EXIF, ảnh trong suốt dán nền đen (`src/common/image_io.py`). (2) Tập thử ngoài đồng PlantWild đã loại ảnh trùng,
  khóa SHA-256; 7 mô hình: ảnh bệnh Centralized 33,1% / FedAvg 31,8%, ảnh lá khỏe bị đoán thành bệnh 87–94%. (3) Biến thể suy luận (TTA, giữ tỉ lệ, gộp mô hình)
  không đạt ngưỡng đặt trước; chức năng "chọn loại cây" đã thêm vào trang Chẩn đoán. (4) Trọng số đi qua Flower nguyên từng byte; phép gộp trùng GD tập trung tới 5e-14;
  máy chủ nay kiểm băm chiều client→server; "độ trôi client" trên web tách phần BatchNorm (>98,9%). (5) Tốc độ laptop↔VM Oracle qua Internet: RTT 46,7 ms,
  31–34 MB/s, một vòng 22,8 s (truyền ≈ 0,8 s); gói đo LAN ở `docs/HUONG_DAN_DO_TOC_DO_LAN.md`. Chuẩn bị công thức train mới: `docs/NEXT_TRAINING_PLAN.md` mục 5.
- 2026-10-21: Dựng release dữ liệu v4 `mixed_pv_pd_v4` (`docs/DATASET_V4.md`), không train. Thêm 19.160 ảnh ngoài đồng từ 10 bộ dữ liệu công khai (CC BY/CC0), iNaturalist và Wikimedia,
  loại ảnh gần trùng với các tập đã khóa, kiểm rò rỉ bằng 0, phân vùng 5 client không cắt nhóm; nhãn kiểm bằng AI sàng lọc (không phải chuyên gia). 13/38 lớp vẫn dưới 100 ảnh ngoài đồng
  (cần ảnh tự chụp: `docs/HUONG_DAN_CHUP_ANH.md`). Đo không cần train (kNN DINOv2): ảnh mới nâng nhận cùng nguồn 44,6% → 82,1% và lá khỏe 19,1% → 26,9%, nhưng PlantWild bệnh 50,3% → 47,1%;
  mục tiêu > 90% chỉ khả thi cho ảnh cùng nguồn (Centralized, FedAvg), không cho ảnh nguồn lạ và không cho Local-only. Chưa train; chưa triển khai.
- 2026-10-21 (rà soát lại release v4): bộ chạy của nhánh train sẽ TỪ CHỐI bản dựng đầu (khóa shard thiếu đuôi `.jsonl`, thiếu `partition_spec_v4.json` và `quality_audit_v4.json`); đã sửa, và
  `scripts/verify_release_contract.py` chạy các bộ kiểm của nhánh train trên release: 11/11 qua (SHA release mới ở `artifacts/release_v4_report.json`). Ảnh chụp liên tiếp của một lá từng rơi vào cả train lẫn test
  (ngô 8%, cà chua Bangladesh 34% ảnh thử có ảnh học gần trùng trong 10 bit): nay gộp nhóm theo phiên chụp và ảnh gần trùng cùng nguồn, cùng lớp; cổng rò rỉ = 0; bỏ giới hạn PlantVillage 800 ảnh/lớp.
  Cổng từ chối ảnh không phải lá (`docs/CONG_TU_CHOI_ANH_KHONG_PHAI_LA.md`, 1.513 ảnh Wikimedia): xác suất lớn nhất của mô hình hiện có chỉ loại 7,6–8,4% ảnh không phải lá; kNN trên DINOv2-small
  loại 93,0% ở ngưỡng nhận 95,5% ảnh lá (đạt chỉ tiêu đặt trước), loại lá của loài khác chỉ 41,6%. Chưa đưa vào web; chưa đo RAM trên VM.
- 2026-10-21 (bộ train v4 + bù dữ liệu đợt 2): bộ train thống nhất `scripts/train_v4.py` cho ba cách (cùng công thức, cùng số bước), đóng gói chạy máy GPU,
  đánh giá sau train `scripts/eval_trained_v4.py`. Người dùng không kiểm nhãn được nên bù dữ liệu bằng 6 bộ ngoài đồng mới (ngô khỏe, cà chua ruộng có 307 ảnh nhện đỏ, ớt, đào, nho)
  và kiểm từng lá PlantDoc cách ly (nhận 475/750 lá; AI sàng lọc, không phải chuyên gia). Lớp thiếu ảnh ngoài đồng còn 5/38 (không có nguồn công khai). Chưa train.
- 2026-10-21 (soát lại dữ liệu đợt 2 trước khi commit): phát hiện và sửa hai lỗi gộp nhóm ở lá đặt trên nền đồng màu (ngưỡng dHash 10 nối chuỗi thành nhóm 916 ảnh, 47% ảnh ớt rơi vào test;
  cùng một lá chụp xoay nằm ở cả train lẫn test). Nay: ngưỡng dHash 4 cho các nguồn đó, gộp ảnh cùng lá theo cosine DINOv2 >= 0,95, cổng cân bằng split, soát lại 18 ảnh mỗi lớp mới: qua.
- 2026-10-21 (đợt 3: gộp phần dùng được từ phiên làm việc song song): tải ba nhóm ứng viên, xem từng ảnh, nhận 45 ảnh (Cherry powdery mildew 15, Strawberry leaf scorch 30; AI sàng lọc, không phải chuyên gia);
  loại FieldVitis Isariopsis (nhãn chưa xác minh) và cà chua đa lớp (không có từ điển lớp). Phiên kia đúng ở hai điểm, đã nhận: "ngoài đồng sẵn sàng" đếm trước khi chia split (đếm lại trong train: chỉ 12/38 lớp có
  >= 300 ảnh từ >= 2 nguồn; tập thử ngoài đồng gần như không có nguồn lạ) và 286 cặp cùng lá cosine >= 0,95 nằm ở hai split ở các nguồn chưa trích đặc trưng (nay gộp nhóm cho mọi nguồn, còn 0 cặp).
  Release dựng lại, SHA `68e57451...`, 52464 ảnh train, hợp đồng bộ chạy 12/12. 5 lớp vẫn thiếu ảnh ngoài đồng (< 100). Chưa train, chưa triển khai.
- 2026-10-22 (đợt train chính v4b, 21 run trên máy GPU thuê): công thức mới (tăng cường trên GPU, batch 128, bf16) nhanh gấp khoảng 3 lần đợt đầu; 3 hạt giống song song mất 1 giờ 50 phút.
  Trung bình 3 hạt giống: Centralized PlantVillage 99,4%, ngoài đồng cùng nguồn 92,2%, PlantDoc giữ riêng 57,6%, PlantWild 36,9%, lá khỏe bị đoán thành bệnh 41% (cũ 87%).
  FedAvg tương ứng 96,9 / 83,1 / 40,5 / 29,8 / 44%. Không đạt ngưỡng PlantWild và PlantDoc nên chưa nhập lên web (`artifacts/train_v4_results.json`, `docs/NEXT_TRAINING_PLAN.md` mục 5.8).
<!-- ROADMAP-END -->

## 8. Cách đồng bộ

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py              # chuẩn: registry + gates + file này
.\.venv\Scripts\python.exe scripts\update_status.py --skip-gates # chỉ sinh lại file từ JSON hiện có
.\.venv\Scripts\python.exe scripts\update_status.py --full-gates # + pytest toàn bộ (chậm)
.\.venv\Scripts\python.exe scripts\update_status.py --check      # kiểm file có đang lệch không (dùng cho pre-commit)
```

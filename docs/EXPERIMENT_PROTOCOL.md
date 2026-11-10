# Giao Thức Thực Nghiệm Chuẩn Hóa (Experiment Protocol)
## Đồ Án Tốt Nghiệp: Federated Learning Trên Dữ Liệu Nông Nghiệp PlantVillage

Tài liệu này đóng băng toàn bộ thông số kỹ thuật, cấu hình mô hình, phân hoạch dữ liệu và quy trình thực nghiệm nhằm đảm bảo tính tái lập 100% (Strict Reproducibility) theo tiêu chuẩn của GVHD và hội đồng phản biện.

---

## 0. Các protocol đang tồn tại — mỗi con số chỉ thuộc về MỘT protocol

| Protocol ID | Vai trò | Dữ liệu | Nơi train | Trạng thái |
|---|---|---|---|---|
| `mixed_pv_pd_v3_pc_r4_seed42_validation` | **Kết quả hiện hành** (13 mô hình PC v3: Centralized AdamW và SGD, FedAvg, SCAFFOLD, 5 Local-only, 4 bản int8; chỉ đánh giá trên validation) | Release `pv_pd_v3`: validation 4.613 (4.344 PlantVillage + 269 PlantDoc); seed 42; pretrained; non-IID label alpha=0.1 | Máy PC của nhóm train (nhánh `pc-v3-ket-qua-20261110`), nhập bằng `scripts/import_pc_package.py`, đổi gói bằng `scripts/swap_web_models.py --validation-only` | Đã kiểm hash; **đã tái đánh giá cục bộ trên validation** (117 phép so khớp tuyệt đối với số của nhóm PC); không có số kiểm thử/calibration/ngoài miền trong đồ án |
| `mixed_pv_pd_v4_pretrained_seed42_validation` | Gói trước, giữ để lùi (Centralized v4b, FedAvg v4b, 5 Local-only v4b; web phục vụ hạt giống 42, 3 hạt giống 42/123/2026 đã chạy) | Release `mixed_pv_pd_v4` (SHA `68e57451…`): train 52.464, validation 6.746 (4.344 PlantVillage + 269 PlantDoc + 2.133 ngoài đồng bộ khác), calibration 7.005, test giữ riêng `legacy_diagnostic_test` 13.101; pretrained; non-IID label alpha=0.1 | Máy GPU thuê (`scripts/train_v4.py`), nhập gói bằng `scripts/import_run_bundle.py`, đổi gói bằng `scripts/swap_web_models.py` | Đã kiểm hash; **đã tái đánh giá cục bộ** (21/21 phép so khớp tuyệt đối); có kết quả trên test giữ riêng và 3 hạt giống (`artifacts/train_v4_results.json`) |
| `mixed_pv_pd_v3_pretrained_seed42_validation` | Lịch sử (gói trước, giữ để lùi: Centralized v6, FedAvg v7, 5 Local-only v7) | Mixed PlantVillage + PlantDoc; validation 4.344 PV + 269 PlantDoc; seed 42; pretrained; non-IID label alpha=0.1 | Ngoài repo (branch `train-*`), import bundle bằng `scripts/import_branch_demo.py` | Đã kiểm hash; **đã tái đánh giá cục bộ** (10/14 phép so khớp tuyệt đối, hai mô hình chính khớp hết); **có kết quả trên test giữ riêng** `legacy_diagnostic_test` (11.015 ảnh); chưa multi-seed |
| `stage2_scratch_clean_v3` | Tái lập khoa học cục bộ GĐ2 (scratch, 5 client) | PlantVillage 54.305 ảnh, split 38.984/4.400/10.921 (`data/partitions_stage2_scratch_v3/`) | Cần dataset gốc + máy train | **BLOCKED**: chưa có run cục bộ |
| `kaggle_pv_v1_label_skew_a0.1` | Lịch sử (run Kaggle 99,09% / 98,25% / 81,40%) | PlantVillage, partition v1, 10 client, test 10.917 ảnh | Kaggle (pipeline cũ ở `legacy/`) | Chỉ còn metrics; checkpoint mất; chỉ trích dẫn kèm nhãn "historical" |

Quy tắc:

1. Không so sánh hay lập bảng chung giữa số liệu của hai protocol khác nhau.
2. "Quy tắc vàng" ở mục 1 (cùng một tập kiểm thử cố định) áp dụng **trong phạm vi từng protocol**.
3. Mọi con số trong báo cáo phải truy được về một entry của `artifacts/registry.json` (xem `PROJECT_STATUS.md`).

---

## 1. Khóa Phạm Vi & Tập Dữ Liệu (Dataset Protocol)

> Mục 1–6 dưới đây mô tả protocol `stage2_scratch_clean_v3` (tái lập cục bộ).

- **Tập dữ liệu duy nhất**: PlantVillage (Color Variant), định dạng RGB.
- **Tổng số ảnh thực tế trên máy**: **54.305 ảnh**, phân bổ đầy đủ trên **38 lớp bệnh cây trồng**.
- **Tập phân chia cố định (Fixed Splits)**:
  - **Tập huấn luyện tập trung (Centralized Train)**: 38.984 ảnh (~71.79%)
  - **Tập kiểm định (Global Validation)**: 4.400 ảnh (~8.10%)
  - **Tập kiểm thử dùng chung (Global Test)**: **10.921 ảnh (~20.11%)**
- **Quy tắc vàng (Golden Rule)**: Toàn bộ mô hình trong mọi kịch bản thực nghiệm (**Centralized**, **FedAvg**, **Local-only**) đều phải được đánh giá trên **cùng một tập kiểm thử cố định duy nhất (`global_test.csv`, 10.921 ảnh)** để bảo đảm tính công bằng tuyệt đối.
- **Tiền xử lý & Chuẩn hóa (Preprocessing)**:
  - Input Resolution: $224 \times 224$ pixels.
  - Channels: 3 (RGB).
  - Normalization: ImageNet Mean $[0.485, 0.456, 0.406]$ và Std $[0.229, 0.224, 0.225]$.
  - Training Augmentation: Resize $(224, 224) \to$ RandomHorizontalFlip.
  - Evaluation / Test / Inference: Resize $(224, 224) \to$ ToTensor $\to$ Normalize.

---

## 2. Kiến Trúc Mạng & Khởi Tạo Trọng Số (Model & P0 Policy)

- **Kiến trúc xương sống (Backbone)**: `MobileNetV3-Small`.
- **Lớp phân loại cuối (Classifier Head)**: 38 lớp đầu ra (`Linear(in_features=1024, out_features=38)`).
- **Chính sách P0 - Tách biệt rõ ràng giữa Giai đoạn 1 và Giai đoạn 2**:
  - **Giai đoạn 1 (Lịch sử)**: Sử dụng trọng số nạp sẵn `ImageNet1K_V1` (Pretrained Transfer Learning).
  - **Giai đoạn 2 (Khoa học - Phân tích non-IID)**: Khởi tạo ngẫu nhiên từ đầu (**Scratch**, `weights=None`).
  - **Nguyên tắc đối chứng**: Tuyệt đối không lấy kết quả 98–99% của mô hình Pretrained GĐ1 để so sánh với 80–90% của Scratch GĐ2 rồi kết luận do non-IID. Mọi so sánh trong GĐ2 giữa **Centralized**, **FedAvg** và **Local-only** đều bắt buộc chạy trên cùng chế độ Scratch với cùng một seed khởi tạo trọng số ban đầu.

---

## 3. Siêu Tham Số Huấn Luyện (Hyperparameters)

- **Thuật toán tối ưu (Optimizer)**: SGD (Stochastic Gradient Descent).
- **Learning Rate ($\eta$)**: $0.01$ (cố định trong toàn bộ thí nghiệm so sánh).
- **Momentum**: $0.9$.
- **Weight Decay**: $1 \times 10^{-4}$.
- **Hàm mất mát (Loss Function)**: Categorical CrossEntropy Loss.
- **Kích thước Batch (Batch Size)**: $32$.
- **Số epoch huấn luyện cục bộ tại client ($E$)**: $1$ epoch / round (mỗi round client sử dụng Fresh SGD state theo đúng chuẩn FedAvg nguyên bản).
- **Số vòng giao tiếp toàn cục (Global Communication Rounds)**: $10$ rounds.
- **Số lượng Clients tham gia**: $K = 5$ clients đồng thời (100% participation rate).

---

## 4. Giao Thức Phân Hoạch Dữ Liệu Phân Tán (Non-IID Partitioning)

Phân hoạch dữ liệu giữa 5 clients được thực hiện dựa trên phân phối Dirichlet trên nhãn bệnh (Label Skew):
$$\mathbf{p}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$$

Các mức độ thực nghiệm:
1. **Gần IID (Baseline)**: $\alpha = 100.0$ (Tỷ lệ phân bổ các lớp bệnh giữa 5 client xấp xỉ tương đương).
2. **Non-IID Vừa Phải (Moderate)**: $\alpha = 1.0$ (Xuất hiện chênh lệch số lượng mẫu giữa các lớp).
3. **Non-IID Gay Gắt (Severe)**: $\alpha = 0.1$ (Mỗi client chỉ nắm giữ dữ liệu của một số ít lớp bệnh).

---

## 5. Chỉ Số Đánh Giá Đo Lường (Evaluation & Telemetry Metrics)

### A. Hiệu năng học máy (ML Performance)
- **Accuracy**: Tỷ lệ mẫu phân loại chính xác trên toàn bộ tập test 10.921 ảnh.
- **Macro-F1 Score**: Trung bình cộng không trọng số của F1-Score trên cả 38 lớp, đảm bảo phản ánh đúng độ chính xác trên các lớp có ít mẫu:
  $$\text{Macro-F1} = \frac{1}{38} \sum_{c=1}^{38} F1_c$$
- **CrossEntropy Loss**: Giá trị hàm mất mát trung bình.
- **Per-class Metrics**: Precision, Recall, F1 và Support cho từng lớp.
- **Confusion Matrix**: Ma trận nhầm lẫn kích thước $38 \times 38$.

### B. Quan sát trọng số & Toàn vẹn (Weight Inspection & Integrity)
- **Chuẩn L2 của trọng số**: $\|w_k\|_2 = \sqrt{\sum w_{k, i}^2}$.
- **Chuẩn cập nhật (Update Norm)**:
  $$\Delta_k = w_k - w_{global}$$
  $$\text{UpdateNorm} = \|\Delta_k\|_2$$
- **Mã băm toàn vẹn**: SHA-256 xác thực `CLIENT_HASH == SERVER_HASH`.

### C. Đo lường truyền thông & Mạng (LOSS != LATENCY)
- `LocalTrainTime`: Thời gian huấn luyện cục bộ bằng `time.perf_counter()`.
- `AggregationTime`: Thời gian server tổng hợp trọng số.
- `NetworkRTT`: Độ trễ khứ hồi giữa Server và Client qua mạng IP/Port.
- `BytesUploaded` / `BytesDownloaded`: Dung lượng bản tin trao đổi.

---

## 6. Tiêu Chuẩn Nghiệm Thu & Phản Biện (Defense Criteria)

| Câu Hỏi Trọng Tâm Của Thầy | Giải Pháp & Bằng Chứng Khoa Học Của Nhóm |
| :--- | :--- |
| **"Trọng số gửi về server có bị tính sai hay không?"** | Trả lời bằng Unit Test toán học độc lập (`tests/test_fedavg.py`) kiểm tra scalar $100(1) + 300(3) / 400 = 2.500000$ và tensor conv 4D. Toàn bộ quá trình kiểm soát bằng SHA-256. |
| **"Loss khác Latency như thế nào?"** | Tách bạch rõ ràng: Loss là hàm mục tiêu toán học (CrossEntropy), Latency là thời gian vật lý (giây / ms). |
| **"Confidence của 1 ảnh có phải Accuracy mô hình không?"** | Không. Web hiện xác suất softmax thô của ảnh đó (gói v4b còn kèm xác suất đã hiệu chuẩn bằng temperature scaling, T fit trên tập calibration riêng; gói PC v3 hiện hành chỉ có softmax thô, mục 8); accuracy là tỷ lệ đúng trên tập test giữ riêng 13.101 ảnh (protocol v4b, mục 8a; 11.015 ảnh ở protocol v3 cũ; gói PC v3 hiện hành không chạy lại trên tập này, mục 8) hoặc 10.921 ảnh (protocol scratch cũ). |
| **"Federated Learning có bảo mật tuyệt đối không?"** | Không có bảo mật tuyệt đối; FL chỉ bảo vệ ảnh thô không rời biên, vẫn cần DP/SecAgg để chống lại tấn công tái tạo đặc trưng (Reconstruction Attack). |
| **"Số validation có bị chọn cho đẹp không?"** | Có hiệu ứng chọn: checkpoint được chọn theo macro-F1 PlantDoc trên chính validation (Centralized: epoch 22 là epoch có accuracy PlantDoc cao nhất trong 83 epoch). Vì vậy nhóm chấm thêm trên tập test giữ riêng chưa từng dùng để chọn và báo cáo khoảng tin cậy theo cụm (mục 7). |

## 7. Hạn chế của phép đánh giá (protocol `mixed_pv_pd_v3_pretrained_seed42_validation`, bản lịch sử)

> Mục này giữ nguyên để tra cứu số của gói v3 (nay chỉ dùng để lùi). Protocol hiện hành là PC v3: xem mục 8 (v4b ở mục 8a). Các hạn chế "một hạt giống", "so sánh chưa công bằng" và số ảnh tập giữ riêng dưới đây **không còn đúng với v4b** (3 hạt giống, cùng công thức, 13.101 ảnh).

Số liệu kiểm bằng `scripts/evaluate_published_models.py` (`--holdout`) và `scripts/report_statistics.py`; mọi con số dưới đây truy được
về `artifacts/holdout_eval_report.json` và `artifacts/statistics_report.json`.

1. **Chọn checkpoint trên chính validation.** "Best" là epoch/vòng có macro-F1 PlantDoc cao nhất trên val (trong các epoch đạt ngưỡng
   F1 PlantVillage); với Centralized, epoch được chọn có PlantDoc val cao nhất trong 83 epoch. Cách chọn này có thể đẩy số val lên.
   Bù lại: tập `legacy_diagnostic_test` của release (PlantVillage 10.843 + PlantDoc 172) không giao train/val/calibration và không dùng
   để chọn, nên không chịu thiên lệch này. Thực tế đo được: số test KHÔNG thấp hơn val (Centralized PlantDoc test 63,37% so với val 53,90%;
   PlantVillage gần như bằng nhau) — chưa thấy val bị thổi phồng, nhưng PlantDoc test chỉ 172 ảnh nên khoảng tin cậy rộng. Release
   gọi đây là tập chẩn đoán "legacy" (summary.json ghi `FINAL_HOLDOUT_PENDING`), chưa phải test cuối cùng được khóa trước.
2. **PlantDoc nhỏ và tập trung theo cảnh.** Val 269 ảnh từ 115 cảnh, test 172 ảnh từ 96 cảnh; khoảng tin cậy 95% bootstrap theo cảnh
   rộng khoảng ±9–10 điểm. Chênh Centralized−FedAvg trên PlantDoc val có McNemar p≈0,04 nhưng khoảng tin cậy theo cụm vẫn chứa 0;
   trên test thì chênh có ý nghĩa. Chỉ 13/38 lớp có ảnh PlantDoc, không có lớp healthy.
3. **Nhóm lá PlantVillage chưa đủ.** Khoảng 24% ảnh PlantVillage là nhóm một ảnh (8 lớp không có nhóm lá nào), nên ảnh chụp lại cùng
   một lá có thể nằm ở cả train và val/test (dHash không phát hiện được kiểu này). Phân tích độ nhạy: bỏ các ảnh có ảnh train mang số
   thứ tự liền kề (ước lượng chụp lại) thì accuracy Centralized/FedAvg không giảm — chưa thấy bằng chứng bị thổi phồng.
4. **So sánh chưa hoàn toàn công bằng.** Centralized: AdamW lr 1e-3, tối đa 100 epoch (chạy 83). FedAvg: SGD không momentum lr 0,01,
   E=1, dừng sớm ở vòng 32 trong khi F1 PlantVillage còn tăng; ngưỡng chọn khác nhau. Khoảng cách Centralized−FedAvg vì vậy gồm cả
   phần do cấu hình, không chỉ do FL.
5. **Một seed, một mức α.** Chưa có độ lệch giữa các lần chạy.

### Khi nào cần train lại

- **Không cần** để demo và trình bày kết quả hiện có, nếu nêu đủ năm hạn chế trên và trích số test giữ riêng kèm khoảng tin cậy.
- **Cần** nếu muốn khẳng định "FedAvg gần Centralized" như một kết quả khoa học: train lại 7 mô hình với CÙNG optimizer và ngân sách
  (ví dụ cùng số epoch-tương-đương, cùng quy tắc chọn), 3 seed; ước tính ~10 GPU-giờ mỗi seed (Centralized ~5,2 h, FedAvg ~1,7 h,
  Local ~3,3 h theo thời gian đã ghi), tức ~30 GPU-giờ trên Kaggle/Colab. Máy demo chỉ có CPU, không train được.
- **Sạch nhất**: gom lại nhóm lá cho ảnh PlantVillage một-ảnh (theo tên/số thứ tự), dựng release mới, chia lại split rồi train lại
  — mọi số liệu sẽ đổi. Kết quả nhập lại qua `scripts/import_branch_demo.py` (bundle có kiểm hash).
- `scripts/run_real_multiseed_report.ps1` thuộc protocol scratch PlantVillage cũ (`stage2_scratch_clean_v3`), không thay được việc trên.

---

## 8. Protocol hiện hành `mixed_pv_pd_v3_pc_r4_seed42_validation`

Mục này mô tả gói mô hình web đang dùng (`artifacts/active_bundle.json` → `artifacts/model_bundles/pc_v3_s42`), nhập từ kết quả huấn luyện PC v3 của nhóm train (nhánh `pc-v3-ket-qua-20261110`, gói `pc_v3_a01_s42_r4`). Cách thay gói: `docs/THAY_MO_HINH_WEB.md`; riêng gói này dùng `scripts/import_pc_package.py` rồi `scripts/swap_web_models.py --name pc_v3_s42 --validation-only`.

**Dữ liệu.** Release `pv_pd_v3` (`data/dataset_sources/mixed/pv_pd_v3`). Validation 4.613 ảnh (4.344 PlantVillage + 269 PlantDoc). Phân vùng non-IID `label_alpha_0_1` (Dirichlet α = 0,1, 5 client, n_k = 5.434 / 7.858 / 5.376 / 8.495 / 10.075, tổng 37.238). Hạt giống 42. Mã băm release (bản có xuống dòng CRLF cuối tệp) nằm trong checkpoint và `import_manifest.json`; bản ở máy này kết thúc bằng LF và web chấp nhận cả hai.

**Mô hình (13).** Centralized AdamW (id `centralized`), Centralized SGD, FedAvg, SCAFFOLD, Local 00–04 và bốn bản int8 (động, chỉ lớp Linear) của bốn mô hình đầu. Mọi checkpoint là epoch/vòng cuối, không chọn theo validation. FedProx, MOON và SCAFFOLD-BN có trong gói của nhóm PC nhưng chưa đưa lên web.

**Cách học.** Do nhóm train quy định và ghi trong `resolved_config` của từng checkpoint (đặc tả ở nhánh kết quả, `pc_v3_ket_qua/docs/specs`); đồ án không huấn luyện lại. Ví dụ FedAvg: 40 vòng, 1 epoch cục bộ mỗi vòng, SGD lr 0,01 không động lượng, weight decay 1e-4, batch 32, AMP; Centralized AdamW: lr 1e-3, weight decay 1e-4, batch 32, 35 epoch.

**Phạm vi bằng chứng: chỉ validation.** Tập kiểm thử cuối được nhóm train mở đúng một lần (bước P6, số ở nhánh kết quả của họ) và tập `calibration` được giữ niêm phong, nên đồ án KHÔNG chạy mô hình PC v3 trên tập kiểm thử, tập calibration hay tập ngoài miền. Hệ quả trên web: không có xác suất đã hiệu chuẩn (hiện softmax thô) và các mục holdout, khoảng tin cậy, công bằng giữa client, ảnh nguyên khung, tham chiếu PlantVillage, mô phỏng cộng đồng, audit rò rỉ và giao thức huấn luyện bị giữ lại kèm lý do; gói ghi `evidence_scope = "validation_only"` trong `import_manifest.json`.

**Int8.** Bản int8 được đo từng ảnh một (lượng tử hóa động tính thang theo cả lô), nên con số khớp cách web suy luận; xem `docs/TICH_HOP_INT8_PC_V3.md`. Không gọi là nhanh hơn: độ trễ CPU tương đương, tệp nhỏ hơn khoảng 30%.

**Kiểm chứng.** 117 phép so (13 mô hình × tổng/PlantDoc/PlantVillage × top-1, macro-F1, độ đúng loại cây) giữa số tính lại trên CPU của đồ án và `validation_metrics.json` của nhóm PC: cả 117 khớp tuyệt đối (`pc_v3_ket_qua/kiem_lai_web/` ở nhánh kết quả).

**Hạn chế.** Số validation, không phải số kiểm thử, và không dùng để thay các số P6. Gói không kèm lịch sử huấn luyện đầy đủ như gói chạy bằng `scripts/train_v4.py` (chỉ `history` trong checkpoint nên chưa có độ lệch cập nhật từng client).

---

## 8a. Protocol của gói trước `mixed_pv_pd_v4_pretrained_seed42_validation` (v4b, giữ để lùi)

Mục này mô tả protocol của gói v4b (`artifacts/model_bundles/v4b_s42`, lùi bằng `scripts/swap_web_models.py --rollback artifacts/model_bundles/v4b_s42`). Cách thay gói: `docs/THAY_MO_HINH_WEB.md`.

**Dữ liệu.** Release `mixed_pv_pd_v4` (`data/dataset_sources/mixed/pv_pd_v4`, mô tả ở `docs/DATASET_V4.md`). Ảnh được chia theo nhóm (cùng lá, cùng cảnh, cùng phiên chụp) rồi
kiểm trùng bằng băm byte và dHash; mã băm manifest của release và của từng tập được ghim trong `import_manifest.json` của gói và kiểm lại ở mỗi lần suy luận. Báo cáo chia ảnh theo
`source_dataset` thành ba nhóm: `plantvillage` (ảnh trong phòng), `plantdoc` (ngoài đồng, nền phức tạp) và `field_other` (ảnh ngoài đồng từ các bộ công khai khác, nhãn do tác giả bộ
dữ liệu gán hoặc sàng lọc bằng mắt). Release v4 gán `source_domain = "plantdoc"` cho mọi ảnh ngoài đồng nên không dùng `source_domain` để chia nhóm.

**Tập đánh giá.** `val` (chọn checkpoint, báo cáo kèm cảnh báo là đã tham gia chọn), `calibration` (fit nhiệt độ T; đồng thời là tập kiểm thử niêm phong, xem mục 9), `legacy_diagnostic_test` (không dùng để chọn;
chưa nằm trong kiểm tra trùng ảnh của `audit_release_leakage.py`), PlantWild (ảnh bệnh nguồn lạ, đo một lần sau khi train, không dùng để chọn). Chọn checkpoint theo trung bình
macro-F1 trên các nhóm nguồn của `val` có từ 30 ảnh trở lên.

**Cách học.** MobileNetV3-Small, khởi tạo ImageNet-1K + đầu 38 lớp theo hạt giống (W0 tính lại từ hạt giống, ghi trong manifest). Cả ba cách (Centralized, FedAvg, Local-only) dùng cùng
công thức: SGD động lượng 0,9, weight decay 1e-4, lr 0,02 cosine có khởi động 1 epoch hoặc vòng, batch 128, label smoothing 0,1, bf16, tăng cường ảnh `gpu_v2`, trọng số mất mát cân bằng
hai miền và 40 epoch hoặc vòng. FedAvg mô phỏng 5 client trong một tiến trình (1 epoch cục bộ mỗi vòng, bộ tối ưu của client làm mới mỗi vòng, gộp theo n_k); Local-only là phần bước
của một client. Phân vùng non-IID: `label_alpha_0_1` (Dirichlet α = 0,1, n_k = 13.107 / 11.691 / 6.647 / 8.863 / 12.156). Hạt giống 42, 123, 2026.

**Điều chưa ghi lại được.** Độ lệch cập nhật ‖w_k − w‖₂ của từng client và lưu lượng mạng đo thật (web ghi `delta_l2_recorded: false`); FedAvg chưa chạy qua Flower.

**Kiểm tra trên web.** `scripts/web_model_check.py` gửi ảnh qua `/api/inference/all` và đối chiếu từng dự đoán với tính toán offline cùng đường tiền xử lý; kết quả ở `artifacts/web_model_check.json`.

**Hạn chế.** Kết quả trên PlantDoc và PlantWild thấp hơn nhiều so với ảnh cùng nguồn (PlantWild: Centralized 36,9%, FedAvg 29,8% trung bình 3 hạt giống); FedAvg chưa dò riêng tốc độ học trên dữ liệu lệch;
chưa thử FedProx, SCAFFOLD, FedBN; năm client là phân vùng giả lập.

## 9. Tập đánh giá: lựa chọn và lý do (N1.4, N5.3)

**Vấn đề.** Tập PlantDoc giữ riêng chỉ có 172 ảnh, 13/38 lớp, khoảng tin cậy 95% rộng khoảng ±7 điểm; mọi tập thử hiện có đã được đọc nhiều lần nên không còn là tập niêm phong.

**Lựa chọn: niêm phong split `calibration` của release v4, và khi chạy niêm phong thì gộp PlantDoc của split này với 172 ảnh giữ riêng.**

| Phần | Số ảnh | Ghi chú |
|---|---:|---|
| PlantVillage | 3.934 | |
| PlantDoc | 388 | 19 lớp, 144 cảnh; 97 ảnh là lá cách ly được nhận lại, nhãn chỉ qua sàng lọc bằng AI nên báo thêm một số không tính chúng |
| Ngoài đồng khác | 2.683 | 35 lớp, 20 nguồn |
| PlantDoc gộp với tập giữ riêng | 560 | khoảng tin cậy hẹp lại còn khoảng ±4 điểm nếu các ảnh độc lập; bootstrap theo cụm cảnh sẽ rộng hơn một chút |

**Vì sao chọn cách này.**
- Split này chưa từng dùng để học các mô hình trong phép so sánh ba mốc và các thuật toán, chưa dùng để chọn checkpoint hay chọn cấu hình, và chưa có báo cáo độ chính xác nào của các mô hình đó đọc nó (`artifacts/statistics_report.json` chỉ có `val` và `test`).
- Logit đã có sẵn trong `logits_best.npz` của mọi run, nên không tốn thêm giờ GPU.

**Những lần đã đụng tới split này (ghi đủ để người đọc tự đánh giá).** Không lần nào trong đây tác động tới mô hình hay lựa chọn trong phép so sánh, nên đây là "chưa dùng để quyết định", không phải "chưa ai từng thấy".
- Fit nhiệt độ T cho xác suất hiển thị trên web; T không đổi dự đoán top-1.
- Hiệu chuẩn dải giá trị của bản int8 ngày 26/10 (256 ảnh); xem giới hạn bên dưới.
- Thí nghiệm học cộng đồng: `scripts/simulate_community.py` và `scripts/aggregation_equivalence.py` học lớp cuối trên ảnh của kho này. Đó là các mô hình riêng, không vào phép so sánh; số báo cáo của chúng đo trên tập khác.
- Ngày 2026-11-02, khi rà soát mã, một tác nhân review đã in top-1 và macro-F1 của split này từ `final.json` của 3 run v4b (`fedavg_s42`, `centralized_s42`, `local_c0_s42`). Số không được chép lại, không được đưa vào báo cáo và không dùng cho quyết định nào; người ra quyết định chưa thấy chúng.

**Vì sao không chọn hai hướng còn lại.**
- *Kiểm tra chéo trên PlantDoc:* PlantDoc nằm trong tập học của release v4, mỗi phần gập phải huấn luyện lại toàn bộ ma trận, tốn gấp nhiều lần ngân sách GPU.
- *Tập mới từ kho `data/external/sources`:* 5.017 ảnh chưa vào release và không chung nhóm với release, nhưng đa số là ảnh đã **bị loại** khi kiểm nhãn (ví dụ thư mục đào "healthy" toàn ảnh quả, nhóm iNaturalist lá khỏe phần lớn là cả bụi cây hoặc cành trụi; xem `docs/DATASET_V4.md` mục 5). Lấy chúng làm tập thử sẽ cho số thấp giả tạo. Ảnh tự chụp hiện có 0 ảnh.

**Giới hạn cần nói khi báo cáo.**
- Tập này cùng nguồn với tập học, nên nó đo khả năng tổng quát trên ảnh chưa thấy của các nguồn đã biết. Khả năng với nguồn lạ vẫn đo bằng PlantWild (2.304 ảnh bệnh, không niêm phong).
- Một phiên chụp ngô Common rust 465 ảnh nằm trọn trong split này (`docs/DATASET_V4.md` mục 4), nên phần "ngoài đồng khác" nghiêng về lớp đó; macro-F1 và bootstrap theo cụm giảm bớt ảnh hưởng nhưng không xóa hết.
- `final.json` của 21 run v4b có sẵn tóm tắt split này từ trước khi có quy định. Không xóa vì gói web kiểm mã băm các run này; không báo cáo nào được đọc mục đó. `scripts/finetune_personal.py` không chép mục này sang run mới.
- Chỉ đo mô hình toàn cục: mô hình cá nhân hóa không có logit trên split này nên `scripts/run_sealed_test.py` từ chối thay vì báo nhầm số của mô hình chung.
- Bản int8 đo ngày 26/10 đã dùng 256 ảnh của split này để hiệu chuẩn dải giá trị (không dùng để đo độ chính xác). Mô hình float32 không bị ảnh hưởng. `scripts/eval_gd4_compression.py` nay hiệu chuẩn bằng `val`; nếu báo số niêm phong cho bản int8 thì phải dựng lại bản int8 trước.

**Khóa.** `scripts/run_sealed_test.py` là đường được dùng để tính độ chính xác trên split này: cần `--confirm-final`, bắt buộc đủ 3 hạt giống và đủ ba mốc, kết quả ghi `artifacts/sealed_test_result.json` và đã có thì từ chối chạy lại, không có cờ ghi đè. `--dry-run` kiểm đủ run mà không mở logit và không tính số nào. `scripts/train_v4.py` không còn tóm tắt split này; `tests/test_run_sealed_test.py` khóa các điều trên. Đây là khóa theo kỷ luật, không phải mã hóa: logit vẫn đọc được, và xóa tệp kết quả thì chạy lại được; tệp nằm trong git nên việc đó để lại dấu vết.

## 10. Sinh lại kịch bản phân hoạch non-IID

Mọi cách chia dựng từ cùng tập train của release `68e57451…`, không đổi mã băm release; mã băm từng shard và kết quả kiểm toán (tách rời, hợp đúng tập train, không cắt nhóm) ghi ở `<release>/partitions_extra/extra_manifest.json`. Dựng lại một cách chia đã có mà ra shard khác thì script dừng.

```bash
python scripts/build_extra_partitions.py          # dựng toàn bộ bảng dưới
```

| Dạng lệch | Tên cách chia | Mức đo được | Trong đợt |
|---|---|---|---|
| Lệch nhãn | `label_alpha_0_1` (có sẵn trong release), `label_alpha_0_5`, `label_alpha_1_0`, `label_alpha_10_0`, `label_alpha_100_0`; `iid` (có sẵn) làm điểm neo | α = 100: TVD nhãn 0,03–0,04; α = 0,1: khoảng 0,6 | N2.1 |
| Lệch nhãn, đo dao động do cách chia | `label_alpha_0_1_p123`, `label_alpha_0_1_p2026` | | N2.1 |
| Lệch số lượng | `quantity_alpha_0_5` | client lớn nhất gấp 7,7 lần client nhỏ nhất (18.554 so với 2.413 ảnh), mọi client đủ 38 lớp, TVD nhãn ≤ 0,02 | N2.2 |
| Lệch đặc trưng (nguồn ảnh) | `source_skew` | tỉ lệ ảnh ngoài đồng theo client 0%, 8%, 25%, 50%, 83%; cỡ client gần bằng nhau | N2.2 |

**Giới hạn của `source_skew`:**
- Ảnh ngoài đồng và ảnh PlantVillage không có cùng phân phối lớp, nên chia lệch nguồn kéo theo lệch nhãn: TVD 0,04–0,26, mức cao nhất đã gần α = 1 (0,28–0,33). Đây không phải lệch đặc trưng thuần túy, không được quy hết chênh lệch cho nguồn ảnh.
- Trọng số mất mát cân bằng hai miền tính theo từng shard (đúng như mọi kịch bản khác). Client 0 chỉ có ảnh phòng thí nghiệm nên khối lượng ảnh ngoài đồng toàn hệ thống là 40%, so với 50% của Tập trung. Đây là một khác biệt nhỏ ngoài cách chia dữ liệu, phải ghi khi so với Tập trung.

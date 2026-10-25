# Rà soát mức hoàn thiện đồ án (2026-10-25)

Phạm vi chấm: **toàn bộ 5 giai đoạn của đề cương** (GĐ1–GĐ5), đề tài thực hiện cá nhân.
Mốc dữ liệu: nhánh `ui-upgrade-2026-10-14`, commit `47fb78f9e`, bản triển khai `fl-demo-20261022-facts` trên máy chủ Oracle.
Tài liệu này chỉ đọc và đối chiếu, không train lại, không triển khai.

## 1. Kết luận

- **Mức hoàn thiện ước tính: khoảng 46% (sai số ±5 điểm)** theo phạm vi GĐ1–GĐ5. Nếu chỉ tính phần GĐ1–GĐ2 thì khoảng 56%.
- Phần hệ thống phần mềm đã hoàn chỉnh. Phần còn thiếu nhiều nhất là thực nghiệm GĐ2 (mới một mức non-IID), GĐ3 (chưa có thuật toán nào), GĐ5 (chưa có ma trận đánh giá tổng hợp) và quyển báo cáo đồ án.
- Điểm số là phán đoán của người rà soát, không phải số đo. Trọng số ở mục 2 có thể điều chỉnh theo yêu cầu của hội đồng.

## 2. Bảng chấm

| Hạng mục | Trọng số | Đã xong | Điểm |
|---|---:|---:|---:|
| A. Hệ thống phần mềm (web, Flower, triển khai, kiểm thử) | 20% | 90% | 18,0 |
| B. GĐ1 – Mốc so sánh (Centralized, Local-only, FedAvg) | 10% | 95% | 9,5 |
| C. GĐ2 – Mô phỏng non-IID nhiều mức | 15% | 35% | 5,3 |
| D. GĐ3 – FedProx, FedBN, SCAFFOLD, MOON | 12% | 5% | 0,6 |
| E. GĐ4 – Tối ưu phía client | 8% | 40% | 3,2 |
| F. GĐ5 – Đánh giá tổng hợp và công bố | 10% | 30% | 3,0 |
| G. Quyển báo cáo đồ án | 15% | 25% | 3,8 |
| H. Chuẩn bị bảo vệ | 10% | 30% | 3,0 |
| **Tổng** | 100% | | **46,3** |

Phạm vi chỉ GĐ1–GĐ2 gồm các hạng mục A, B, C, G, H (trọng số 70): điểm 39,5 trên 70, tức khoảng 56%.

## 3. Hiện trạng từng hạng mục

### A. Hệ thống phần mềm (khoảng 90%)

Đã có:
- Kiểm thử đầy đủ trên mã hiện tại đạt, gồm cả nhóm kiểm thử mạng Flower (`artifacts/final_acceptance.json`: `technical_acceptance = pass`, sinh lúc 2026-10-22 15:32). Số lượng kiểm thử lấy từ file này, không ghi tĩnh trong tài liệu.
- Danh mục kiểm tra của máy: 22/22 hạng mục kỹ thuật XONG (`PROJECT_STATUS.md` mục 3).
- Web tách router và service; suy luận đi qua một cửa `web/services/model_service.py`, chỉ cho phép mô hình qua cổng `web_visible`.
- Hệ thống đang chạy tại `https://datn.tranngochung.id.vn`; `/api/models` trả 8 mô hình, bundle `artifacts/model_bundles/v4b_s42` (`artifacts/active_bundle.json`).
- Bộ huấn luyện thống nhất `scripts/train_v4.py` + `src/training/` cho ba cách huấn luyện.

Còn thiếu hoặc tồn đọng:
- Chưa có CI (không có `.github/`) và chưa có công cụ kiểm tra kiểu mã (lint).
- Mã không còn dùng: `src/data/split.py`; hai bản cài đặt Dirichlet (`src/data/partition.py` và `src/data/dirichlet_split.py`); ba module metrics (`src/common/metrics.py`, `src/data/metrics.py`, `src/training/metrics.py`); hai bộ huấn luyện (`src/centralized/` và `src/training/`).
- Chưa có tập lệnh triển khai; mọi bước triển khai thực hiện thủ công theo `docs/SERVER_DEPLOY_20261005.md`.
- Quản trị cộng đồng chỉ dựa vào `COMMUNITY_ADMIN_TOKEN`; danh sách IP cho phép trong nginx đang để chú thích.
- Phiên học cộng đồng đã đặt lại về phiên bản 0 sau lần chuyển v4b, quản trị viên cần mở lại.

### B. GĐ1 – Mốc so sánh (khoảng 95%)

Đã có: Centralized, FedAvg và 5 mô hình Local-only, cùng công thức huấn luyện, cùng số bước, 3 hạt giống (42, 123, 2026), 21 lượt chạy (`artifacts/train_v4_results.json`).

| Chỉ số (trung bình 3 hạt giống) | Centralized | FedAvg | Local-only (TB 5 client) |
|---|---:|---:|---:|
| PlantVillage, tập test | 99,38% | 96,85% | 48,09% |
| Ngoài đồng cùng nguồn | 92,23% | 83,13% | 40,46% |
| PlantDoc giữ riêng (172 ảnh) | 57,56% | 40,50% | 23,68% |
| PlantWild (bệnh) | 36,88% | 29,80% | 16,24% |

Hệ thống đánh giá mạnh: khoảng tin cậy bootstrap theo cụm, kiểm định McNemar, hiệu chỉnh xác suất (ECE, temperature scaling), kiểm tra rò rỉ dữ liệu bằng 0 (`artifacts/statistics_report.json`, `artifacts/release_leakage_audit.json`).

Còn thiếu: khoảng tin cậy và McNemar mới tính cho hạt giống 42; chưa có kiểm định ghép cặp giữa các hạt giống.

### C. GĐ2 – Mô phỏng non-IID (khoảng 35%)

Đã có: release dữ liệu v4 gồm hai cách chia client là `iid` và `label_alpha_0_1` (`data/dataset_sources/mixed/pv_pd_v4/partitions/`); toàn bộ 18 lượt chạy FedAvg và Local-only dùng `label_alpha_0_1`.

Còn thiếu:
- Chưa có mức α nào khác α = 0,1 được huấn luyện. Phân hoạch `iid` có sẵn nhưng chưa dùng. Vì vậy chưa vẽ được đường "khoảng cách FedAvg – Centralized theo mức non-IID", là kết quả trung tâm của GĐ2.
- FedAvg mới chạy dạng mô phỏng trong một tiến trình. Phía Flower chỉ có kiểm thử khói trên dữ liệu ngẫu nhiên (`artifacts/network_artifact_validation.json`: `ok = false`) và một lần đo 3 vòng qua Internet (RTT 46,7 ms, một vòng 22,8 s).
- Chưa lưu độ lệch trọng số từng client (`delta_l2_recorded = false`) và chưa đo số byte truyền thực tế khi huấn luyện.
- Chỉ có một cấu hình K = 5 client, E = 1 epoch cục bộ, 40 vòng, tham gia đầy đủ.

### D. GĐ3 – Thuật toán chịu non-IID (khoảng 5%)

Không có FedProx, FedBN, SCAFFOLD, MOON hay FedAvgM trong `src/`, `scripts/`, `web/`. Hiện chỉ có phần lý thuyết trong `docs/TONG_HOP_FEDERATED_LEARNING.md` và các nguồn tham khảo [10]–[12] trong báo cáo Word. `scripts/train_v4.py` chỉ chấp nhận `--method` là `centralized`, `fedavg`, `local`.

### E. GĐ4 – Tối ưu phía client (khoảng 40%)

Đã có: MobileNetV3-Small làm nền; phép đo độ nhạy của việc nén trọng số khi truyền (fp16, bf16, int8) trong `scripts/audit_weight_exchange.py`; học cộng đồng trên trình duyệt chỉ gửi lớp cuối (khoảng 156 KB mỗi lượt).

Còn thiếu: lượng tử hóa mô hình suy luận (kích thước, độ trễ, mức giảm độ chính xác); nén cập nhật khi huấn luyện liên kết; đường cong số byte – độ chính xác.

### F. GĐ5 – Đánh giá tổng hợp và công bố (khoảng 30%)

Đã có: hạ tầng đánh giá (tập PlantWild loại trùng, cổng từ chối ảnh không phải lá, đánh giá mô hình nền DINOv2/BioCLIP, độ công bằng giữa các client trong `artifacts/client_fairness_report.json`).

Còn thiếu:
- Ma trận mức non-IID × thuật toán × hạt giống.
- Cổng từ chối ảnh không phải lá mới được đo trên mô hình v3; chưa hiệu chỉnh lại cho v4b và chưa đưa vào web (`artifacts/train_v4_results.json`, mục `not_checked_here`).
- Chưa có bài viết công bố.

### G. Quyển báo cáo đồ án (khoảng 25%)

- Không có quyển báo cáo cuối (Word/LaTeX/PDF) trong kho mã. Chỉ có báo cáo tiến độ `docs/Bao-cao-tien-do-do-an.docx` ngày 19/10.
- Báo cáo tiến độ còn số liệu cũ (bản v3: FedAvg 50,00%, Centralized 63,37% trên 172 ảnh PlantDoc, một hạt giống), 13 lỗi đã biết trong `docs/REPORT_WORD_WEB_2026-10-20.md` chưa sửa trong file Word, và phần bìa còn để trống tên, mã số sinh viên, lớp, giảng viên hướng dẫn.
- Chưa có các chương cơ sở lý thuyết, phân tích thiết kế hệ thống, triển khai và kiểm thử phần mềm; chưa có mục lục, danh mục hình/bảng, lời cảm ơn.
- Ba danh sách tài liệu tham khảo riêng (`docs/REFERENCES_IEEE.md`, báo cáo Word, chương 5.4 của web) chưa gộp.
- Báo cáo dạng web (5 chương, 3 trang minh họa, 5 phụ lục) đã khớp v4b và không có nội dung giữ chỗ. Đây là nguồn nội dung tốt để chuyển sang quyển báo cáo.

### H. Chuẩn bị bảo vệ (khoảng 30%)

Đã có: bộ 52 câu hỏi đáp (`docs/CAU_HOI_BAO_VE_2026-10-19.md`), kịch bản trình bày (`docs/WEB_DEMO_GD1_GD2.md`), sơ đồ kiến trúc (`docs/ARCHITECTURE.md`).

Còn thiếu:
- Không có slide trong kho mã. Bản slide cũ nằm ngoài kho (8 trang, ngày 21/09), còn giữ chỗ trống cho trường, khoa, giảng viên, ngày trình bày và ba ô thành viên.
- Hai tài liệu trên còn dùng số của mô hình v6/v7 và protocol v3.
- Kho mã chưa sẵn sàng nộp: nhánh `main` cục bộ cách nhánh đang làm 91 commit; `origin/main` có hai commit khác lịch sử; chưa có tag cho bản v4b; còn bốn tệp móc Git LFS chưa được theo dõi trong `scripts/hooks/`.
- Repo không ghi ngày nộp hay ngày bảo vệ.

## 4. Rủi ro cần xử lý trước khi bảo vệ

1. **Mô hình v4b đã lên web dù chưa đạt ngưỡng đặt trước.** Kết quả 7/14 ngưỡng đạt (`artifacts/train_v4_results.json`, mục `checks`). Các ngưỡng không đạt: PlantWild ≥ 65% và mức tăng ≥ 10 điểm (cả Centralized và FedAvg), PlantDoc giữ riêng không giảm quá 2 điểm (cả hai), FedAvg ngoài đồng cùng nguồn ≥ 90% (đạt 83,13%). `docs/NEXT_TRAINING_PLAN.md` dòng 291 và `PROJECT_STATUS.md` dòng 388 vẫn ghi "chưa nhập lên web", nhưng `artifacts/active_bundle.json` đã trỏ sang v4b từ 2026-10-22. Chưa có văn bản nào giải thích quyết định này. Hội đồng có thể hỏi.
2. **Cổng nghiệm thu khoa học kiểm tra sai đối tượng.** Kết luận "scientific blocked" đến từ artifact của pipeline Kaggle cũ (`src/registry.py`, `scripts/validate_artifact_provenance.py`, `scripts/pre_demo_check.py`), không phải từ bundle v4b mà web đang dùng. Cổng này không thể chuyển sang đạt nếu không sửa các kiểm tra.
3. **Số liệu Kaggle không được trích dẫn.** Các file `artifacts/metrics/{centralized,local_only,fedavg}_results.json` (99,09 / 98,25 / 81,40) không còn checkpoint đối chiếu, và dãy giá trị có dấu hiệu ghi tay (ví dụ `aggregation_time_ms` hằng số 78,4 ở mọi vòng). Registry đã gắn nhãn "chưa tái kiểm chứng".
4. **Tài liệu lỗi thời so với v4b:**
   - `docs/CAU_HOI_BAO_VE_2026-10-19.md` (câu 24, 26, 27, 29, 31, 33, 34)
   - `docs/WEB_DEMO_GD1_GD2.md`
   - `docs/COMMUNITY_FL.md` (đường cơ sở FedAvg v7)
   - `docs/DATASET_V4.md` mục 6 ("chưa train")
   - `docs/EXPERIMENT_PROTOCOL.md` mục 6–7 (tập giữ riêng 11.015 ảnh, hạn chế của v3)
   - `docs/RUNBOOK.md` mục 0, 6, 7
   - `docs/AWS_DEPLOYMENT_GUIDE.md` (mô tả EC2, hệ thống đã chuyển sang Oracle)
   - `docs/Bao-cao-tien-do-do-an.docx`
5. **Giới hạn của dữ liệu.** Nhãn ảnh ngoài đồng mới chỉ được sàng lọc bằng AI, chưa có chuyên gia kiểm; 5 lớp còn dưới 100 ảnh ngoài đồng và 2 lớp không có; chỉ 12/38 lớp có từ 300 ảnh từ ít nhất 2 nguồn; chưa có ảnh tự chụp; tập PlantDoc giữ riêng chỉ 172 ảnh phủ 13/38 lớp, không có lớp lá khỏe.
6. **Một số kết quả cần nói đúng mức.** FedAvg trên PlantDoc không hơn Local-only có ý nghĩa thống kê (chênh +7,6 điểm, khoảng tin cậy [−7,3; +24,3], p = 0,20). Sau hiệu chỉnh nhiệt độ, ECE trên PlantDoc xấu đi (FedAvg 0,077 → 0,245). Mô hình trên ảnh nguyên khung PlantDoc tụt rõ so với ảnh cắt lá. Bộ phát hiện lá chưa được huấn luyện.

## 5. Lộ trình các bước tiếp theo

Thứ tự theo phụ thuộc. Mỗi bước nên xin duyệt riêng trước khi thực hiện.

### P0 – Gỡ mâu thuẫn (khoảng 1 ngày, không cần GPU)

1. Ghi lại quyết định đưa v4b lên web và lý do (v4b tốt hơn v3 ở lá khỏe bị đoán thành bệnh: 87–94% xuống 41–44%, và có 3 hạt giống), kèm nêu rõ 7/14 ngưỡng chưa đạt. Cập nhật `docs/NEXT_TRAINING_PLAN.md`.
2. Chuyển các cổng nghiệm thu khoa học sang kiểm tra bundle v4b; bỏ phụ thuộc vào slot Kaggle cũ.
3. Cập nhật các tài liệu ở mục 4.4.
4. Xử lý bốn tệp móc LFS chưa theo dõi: thêm vào kho hoặc đưa vào `.gitignore`.
5. Hợp nhất mã vào `main`, gắn tag cho bản hiện hành.

### P1 – Hoàn tất GĐ2 (cần máy GPU thuê)

1. Tạo phân hoạch v4 cho thêm α ∈ {1,0; 0,5}, dùng cách chia nhóm của `scripts/build_release_v4.py` và `src/data/dirichlet_split.py` để không rò rỉ giữa các client.
2. Chạy FedAvg và Local-only × 3 hạt giống cho `iid`, α = 1,0, α = 0,5 bằng `scripts/train_v4.py` (một lượt FedAvg khoảng 1.760 giây trong đợt v4b).
3. Ghi độ lệch trọng số từng client và số byte truyền; vẽ đường khoảng cách FedAvg – Centralized theo mức non-IID.
4. Chạy một lượt Flower thật với công thức v4b (có thể ít vòng hơn) và so với bản mô phỏng cùng hạt giống.

### P2 – GĐ3: thuật toán chịu non-IID

Mở rộng `--method` trong `scripts/train_v4.py`, dùng lại `src/training/loop.py` và `src/federated/strategy.py`:
- **FedProx**: thêm số hạng ràng buộc μ/2·‖w − w_g‖²; thử μ ∈ {0,001; 0,01; 0,1}.
- **FedBN**: không gộp tham số và thống kê BatchNorm; đánh giá theo BatchNorm của từng client.
- **SCAFFOLD**: thêm biến điều khiển; lượng truyền tăng gấp đôi, cần ghi nhận.
- **MOON**: thêm đầu chiếu và hàm mất mát đối chiếu; nặng nhất nên làm sau cùng.
- **FedAvgM**: chi phí thấp, làm kèm.

Kiểm thử đơn vị cho từng thuật toán: FedProx với μ = 0 phải trùng FedAvg; FedBN giữ nguyên BatchNorm cục bộ; SCAFFOLD đúng công thức cập nhật. Chạy trên α = 0,1 và α = 0,5, mỗi cấu hình 3 hạt giống.

### P3 – GĐ4: tối ưu phía client

- Lượng tử hóa int8 mô hình suy luận bằng ONNX Runtime; đo kích thước, độ trễ CPU và trên trình duyệt (đã có onnxruntime-web), và mức giảm độ chính xác.
- Nén cập nhật khi huấn luyện liên kết (fp16, int8; tái sử dụng hàm `quantize` trong `scripts/audit_weight_exchange.py`); vẽ đường số byte – độ chính xác.
- Tùy chọn: thử một backbone nhỏ hơn.

### P4 – GĐ5: đánh giá tổng hợp và công bố

- Ma trận mức non-IID × thuật toán × hạt giống trên PlantVillage, ngoài đồng cùng nguồn, PlantDoc giữ riêng và PlantWild; khoảng tin cậy và McNemar bằng `scripts/report_statistics.py`, mở rộng sang kiểm định giữa các hạt giống.
- Hiệu chỉnh lại cổng từ chối ảnh không phải lá cho v4b (`scripts/eval_ood_gate.py`) và cân nhắc đưa vào web.
- Cập nhật chương 4 của báo cáo web, registry và trạng thái.
- Viết bản thảo bài báo hoặc báo cáo khoa học theo yêu cầu của khoa.

### P5 – Quyển báo cáo đồ án

Cấu trúc đề xuất: Mở đầu; Chương 1 cơ sở lý thuyết (học liên kết, FedAvg, non-IID, các thuật toán GĐ3, MobileNetV3); Chương 2 dữ liệu; Chương 3 phân tích và thiết kế hệ thống; Chương 4 thực nghiệm GĐ1–GĐ5; Chương 5 triển khai và kiểm thử; Kết luận và hướng phát triển. Tái sử dụng nội dung 5 chương của báo cáo web. Sửa 13 lỗi trong `docs/REPORT_WORD_WEB_2026-10-20.md`, gộp ba danh sách tài liệu tham khảo, điền bìa.

### P6 – Bảo vệ

Làm slide theo số liệu cuối; cập nhật bộ 52 câu hỏi đáp và kịch bản trình bày; tập dượt; chốt kho mã (README, tag, cây làm việc sạch).

### Tùy chọn sau cùng

Bộ phát hiện lá cho ảnh nguyên khung; ảnh tự chụp theo `docs/HUONG_DAN_CHUP_ANH.md`; CI bằng GitHub Actions kèm công cụ lint; dọn mã không còn dùng ở mục 3.A.

## 6. Phương pháp và giới hạn của bản rà soát

- Đọc `PROJECT_STATUS.md`, `README.md`, toàn bộ nhóm tài liệu trong `docs/`, các file kết quả trong `artifacts/`, báo cáo tiến độ Word, cấu trúc mã nguồn và kiểm thử, lịch sử git, bản ghi triển khai.
- Đối chiếu trực tiếp: kết quả 14 ngưỡng đặt trước, bundle đang dùng, số mô hình web đang phục vụ (8), khác biệt giữa `main` và nhánh hiện tại, phân hoạch có trong release v4.
- Chưa chạy lại bộ kiểm thử đầy đủ và chưa kiểm thử giao diện; số lượng kiểm thử lấy từ `artifacts/final_acceptance.json` (mã nguồn chưa đổi từ thời điểm đó).
- Phần trăm hoàn thiện là ước tính; các trọng số do người rà soát chọn.

## 7. Tiến độ P0 (cập nhật 2026-10-26)

| Việc P0 | Trạng thái | Ghi chú |
|---|---|---|
| Ghi quyết định đưa v4b lên web | Xong | `docs/NEXT_TRAINING_PLAN.md` mục 5.9 (số hai phía, cách nói với hội đồng, đường lùi) |
| Cổng nghiệm thu khoa học kiểm gói v4b | Xong một phần | `scripts/pre_demo_check.py` nay kiểm dữ liệu release, nguồn gốc gói và đủ 3 hạt giống của v4b. Còn một cảnh báo thật: chưa có run mạng Flower bằng dữ liệu thật (thuộc P1) |
| Cập nhật tài liệu còn số cũ | Xong | Bộ hỏi đáp, `COMMUNITY_FL`, `WEB_DEMO_GD1_GD2`, `DATASET_V4`, `EXPERIMENT_PROTOCOL`, `RUNBOOK`, `AWS_DEPLOYMENT_GUIDE` |
| Tệp móc LFS chưa theo dõi | Xong | Bốn tệp trong `scripts/hooks/` được thêm vào kho |
| Hợp nhất vào `main`, gắn tag | Xong (2026-10-26) | `origin/main` chỉ có 2 commit và 1 tệp README của Khanh, lịch sử không chung. Đã gộp bằng merge commit `925630b70` (README của nhánh này được giữ, bản cũ còn trong lịch sử `f1e7388a2`) và đẩy lên `main` dạng fast-forward, không ghi đè. Tag `v0.3.0-v4b-20261026` |
| Báo cáo Word và slide | Chưa | Thuộc P5 và P6 |

### Tiến độ P1, P2 (cập nhật 2026-10-26, phần không cần GPU)

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Cài FedProx, FedBN, SCAFFOLD, FedAvgM, MOON | Xong mã, chưa có số thí nghiệm | `src/federated/algorithms.py`, nối vào `scripts/train_v4.py --algo`; 22 test đơn vị, chạy thử trên CPU cả sáu thuật toán và chạy tiếp từ checkpoint. Giới hạn của cách cài ở `docs/NEXT_TRAINING_PLAN.md` mục 5.10 |
| Ghi độ lệch trọng số và số byte từng client | Xong | `history.jsonl` có `delta_l2`, `delta_l2_without_bn`, `bytes_up`, `bytes_down` cho run học liên kết mới |
| Cách chia client cho α = 0,5; 1,0; 10 và hai hạt giống chia khác | Xong | `scripts/build_extra_partitions.py`; mã băm release không đổi; kiểm toán rò rỉ đạt |
| Kịch bản chạy ma trận trên máy GPU | Xong | `scripts/remote/run_sweep.sh` |
| Chạy thí nghiệm GĐ2 và GĐ3 | Chưa | Cần máy GPU, khoảng 14–17 giờ GPU (mục 5.10) |
| Bộ gộp kết quả đợt này (`scripts/eval_sweep_v4.py`) | Chưa | Viết khi có run |

Phát hiện thêm khi làm P0:
- `PROJECT_STATUS.md` có câu minh họa Local-only viết cứng trong `scripts/update_status.py` ("Bacterial_spot: c02 có 1.549 ảnh và đạt 99%") sai với phân hoạch v4 (client 04 có 1.624/2.023 ảnh, đạt 100%; client 02 chỉ 36 ảnh, đạt 26%). Câu này nay được sinh từ số liệu.
- Số lớp mỗi client ghi trong tài liệu (19–26 và 20–27) sai; đo từ phân hoạch là 22–26/38.
- Cổng `plantvillage_real_data` trước đây thuộc protocol scratch cũ (`data/partitions_stage2_scratch_v3`), không liên quan tới gói v4b.

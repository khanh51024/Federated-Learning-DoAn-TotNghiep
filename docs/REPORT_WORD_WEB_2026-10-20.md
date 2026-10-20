# Đối chiếu báo cáo tiến độ Word và báo cáo web

Ngày kiểm tra: 20/10/2026. Nguồn Word: `C:/Users/JiiPii/Downloads/Bao-cao-tien-do-do-an.docx` (ngày trên bìa 19/10/2026; 182 đoạn, 15 bảng, 18 hình). Bản web đối chiếu: website năm chương trong checkout `D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old`. Giữ cấu trúc và các phép đánh giá đã có trên web.

## Kết luận kiểm tra

Các số chính trong Word khớp artifact: validation 4.613 ảnh (269 PlantDoc + 4.344 PlantVillage); FedAvg v7 chọn vòng 27, mốc hoàn tất 32/60; Centralized v6 chọn epoch 22, chạy 83/100; legacy test có 172 PlantDoc, Centralized đúng 109 (63,37%), FedAvg đúng 86 (50,00%). Local-only có năm checkpoint riêng. Cùng seed 42 và fingerprint W0 không đồng nghĩa cùng optimizer hoặc ngân sách huấn luyện.

| Nội dung Word | Kết quả đối chiếu và cách dùng trên web |
|---|---|
| “Kiểm thử độc lập”, “khách quan nhất” | Legacy diagnostic test không được dùng chọn checkpoint, nhưng chưa phải test thực địa mới niêm phong. Audit `release_leakage_audit.json` chỉ kiểm train/val/calibration (46.076 ảnh), không chứng minh legacy test không trùng. Giữ tên “tập giữ riêng (legacy)” và hạn chế này. |
| Audit PASS trên 57.091 bản ghi | 57.091 = toàn bộ release trừ quarantine; không đồng nhất với phạm vi audit hiện có trên web. Không mở rộng kết luận PASS sang test. |
| Wilson 95% khoảng ±7 điểm | Wilson theo ảnh không tính tương quan nhiều crop cùng cảnh. Web đã dùng bootstrap theo 96 cảnh: FedAvg 38,4–61,4%; Centralized 52,9–73,7%. Giữ bootstrap và ghi rõ phương pháp; p McNemar 0,0076 theo ảnh cần đọc cùng CI theo cảnh. |
| Calibration “chưa dùng” | Đúng với summary huấn luyện ban đầu (`NOT_REQUESTED`), đã cũ so với web: web có temperature scaling fit trên calibration riêng. Cập nhật vai trò calibration theo trạng thái hiện tại. |
| “Mỗi client thiếu lớp nên không thể dự đoán đúng” | Khẳng định tuyệt đối không được suy ra từ shard thiếu nhãn, nhất là mạng pretrained. Chỉ mô tả quan sát recall thấp và dữ liệu thiếu lớp; không tuyên bố bất khả thi. |
| “Ba phương pháp chỉ khác cách dùng dữ liệu” / “mốc sàn công bằng” | Centralized dùng AdamW 0,001; FedAvg SGD 0,01, momentum 0; ngân sách khác. Bổ sung bảng cấu hình và giới hạn suy luận nhân quả. Local-only checkpoint chỉ lưu round + model state và fingerprint, không đủ metadata để xác minh riêng sampler/optimizer/AMP. |
| V7 chạy một tiến trình; Flower là bước tiếp theo | Đúng: telemetry 27 vòng ghi `in_process_simulation`, không có RTT mạng thật. Demo Flower localhost ảnh nhiễu là bằng chứng hệ thống riêng, không phải kết quả v7. |
| Nhật ký chỉ đến 27 | Có ngay trong checkpoint FedAvg đã kiểm SHA; thêm biểu đồ và bảng telemetry từng vòng. Mốc 32 từ `fedavg/RESULTS.md`; không bịa log vòng 28–32. |
| Chưa đo ảnh toàn cảnh | Web đã có `fullframe_eval.json`; giữ phép đo đó, nhưng không suy rộng thành đánh giá trên ảnh thực địa mới. |
| Trọng số và dung lượng | Xác minh bằng checkpoint SHA và byte thật: 19,02 MB Centralized; 7,88 MB FedAvg; khoảng 6,35 MB mỗi Local. Phân biệt tensor mô hình với hệ số gộp p_k, và byte tệp với payload mô phỏng. |
| EXIF trong canonical_v1 | Bundle transform xác minh resize/normalize/augmentation, chưa xác minh được decoder huấn luyện có xoay EXIF; decoder web hiện chuyển RGB chưa xoay EXIF. Báo cáo web ghi rõ khác biệt này và hướng dẫn xoay ảnh đúng trước khi tải; không tuyên bố web hoàn toàn tương đương tiền xử lý huấn luyện. |
| 16 ảnh minh họa chọn ngẫu nhiên | Word có hình nhưng không kèm danh sách sample_id/nhãn và phép tái tạo trong bundle hiện tại. Không nhập ảnh chụp biểu đồ như số liệu gốc. Giữ thư viện ảnh có nguồn và ma trận nhầm lẫn đang có trên web. |

## Nội dung đã bổ sung vào web

- Chương 2: đủ năm phần release với số ảnh theo nguồn từ manifest đã kiểm hash; phân biệt validation, calibration và legacy test; tiền xử lý, crop 8%, sampler 25% và giới hạn EXIF hiện tại.
- Chương 3: cấu hình thực lưu của Centralized/FedAvg; metadata Local-only thiếu thì ghi thiếu; bảng bảy checkpoint, mốc chọn, dung lượng và SHA rút gọn; phân biệt runtime v7 với Flower + Ray legacy và phiên kiểm thử localhost.
- Chương 4: lịch sử FedAvg 27 vòng từ checkpoint; top-1 validation và macro-F1 PlantDoc; chọn vòng để đọc n_k, p_k, train loss, delta L2, số bước cập nhật; payload mô phỏng và thời gian chỉ cho 27 vòng.
- Chương 5: trạng thái tiến độ, phần còn thiếu (multi-alpha/seed, rerun Flower, thuật toán chịu non-IID, triển khai thực địa); nguồn nền tảng có liên kết bài gốc. Bổ sung giới hạn McNemar theo ảnh và giữ CI theo cảnh đã có.

## Nguồn có thể tái tạo

`scripts/report_training_protocol.py` sinh `artifacts/training_protocol_report.json` từ bảy checkpoint đã kiểm SHA, `centralized/summary.json`, `local/combined_summary.json`, `fedavg/RESULTS.md`, từng `validation_metrics.json` và năm manifest của release. Không train hoặc đổi checkpoint. API `/api/evidence/summary` thêm phần `training`, khóa theo protocol/release/W0/seed/SHA và kiểm nguồn trước khi hiển thị; thiếu hoặc sai thì phần đó không hiện số.

Nguồn thống kê hiện có: `artifacts/statistics_report.json`, `holdout_eval_report.json`, `release_leakage_audit.json`, `fullframe_eval.json`. Thông tin thư mục bài báo đối chiếu với trang chính thức PMLR/arXiv ngày 20/10/2026; không lấy độ chính xác trong bài khác làm kết quả đồ án.

Không chỉnh file Word gốc. Không chạy lại huấn luyện hay triển khai bản web lên máy chủ bên ngoài.

## Đợt kiểm tra thứ hai (số liệu Word đối chiếu với artifact)

Nguồn Word: `docs/Bao-cao-tien-do-do-an.docx`. Phần lớn số liệu khớp artifact. Những chỗ sai hoặc nói quá (không sửa file Word, chỉ ghi để người viết chỉnh):

1. Mục 2.1: “loss kiểm định thấp nhất 0,3055 ở epoch 22” sai. Thấp nhất là 0,2811 ở epoch 4; 0,3055 là loss của epoch được chọn.
2. Mục 2.1: lịch giảm tốc độ học không phải “0,001 ở epoch 10, 0,0005 ở epoch 22, …”. Thực tế (ReduceLROnPlateau theo macro-F1 PlantDoc): giảm một nửa lần đầu ở epoch 19, tổng 11 lần, còn 1,53×10⁻⁸ ở epoch 83.
3. Mục 2.3: tốc độ học client đổi 0,01 → 0,005 từ vòng 22 (`lr_next_round` đổi ở dòng vòng 21), không phải “sau vòng 20”.
4. Mục 2.3: “loss cục bộ mọi client dưới 0,09” chỉ đúng ở vòng 27; vòng 22–24 có client 0,092–0,104.
5. Bảng 2.5: thời gian một vòng là 191–246 giây, không phải 192–246.
6. Bảng 2.7: “đúng loại cây” trung bình Local-only là 67,21% (không phải 67,33%); macro-F1 trung bình Local-only tính được là 19,45% (Word để “—”).
7. Bảng 1.3: calibration “chưa dùng” đã cũ; web dùng tập này để fit temperature scaling.
8. “Kiểm tra chất lượng PASS trên 57.091 bản ghi” và “kiểm thử độc lập, khách quan nhất” nói quá: audit trùng ảnh hiện có chỉ phủ train, validation và calibration (46.076 ảnh), chưa phủ tập legacy test.
9. Wilson theo ảnh (±7 điểm) bỏ qua tương quan giữa các ảnh cùng cảnh; bootstrap theo 96 cảnh cho khoảng rộng hơn (FedAvg 38,4–61,4%).
10. “Ba phương pháp chỉ khác cách dùng dữ liệu” mâu thuẫn Bảng 2.1 (AdamW 0,001 và SGD 0,01, ngân sách khác nhau).
11. “Không thể dự đoán đúng các lớp thiếu” là khẳng định tuyệt đối; nên viết là nhận diện kém.
12. Số ảnh gốc PlantDoc “2.593” chưa kiểm được từ dữ liệu trong repo (bài báo gốc ghi 2.598), nên web không ghi số này. PlantVillage 54.305 khớp `data/dataset_manifest.json`.
13. Bìa còn trống tên sinh viên, MSSV, giảng viên hướng dẫn.

### Bổ sung lên web đợt này

- Chương 4 (4.1): đoạn diễn biến Centralized sinh từ lịch sử 83 epoch (dừng sớm, loss thấp nhất, mốc giảm lr, chênh lệch train/val PlantDoc) và biểu đồ macro-F1 PlantDoc/PlantVillage theo epoch có đánh dấu epoch được chọn; bảng “Toàn tập kiểm tra 4.613 ảnh” (top-1, macro-F1, đúng loại cây của Centralized, FedAvg, trung bình Local-only) và mức FedAvg thu hẹp khoảng cách.
- Chương 4 (4.2): bảng 172 ảnh PlantDoc của tập giữ riêng (đúng/172, top-1, macro-F1, đúng loại cây, lỗi tự tin ≥ 0,9) và lớp đóng góp nhiều nhất vào khoảng cách (Cà chua – khảm lá: 32 so với 17); `/api/evidence/summary` có thêm `holdout.models[].plantdoc_per_class`.
- Mục lịch sử FedAvg: biểu đồ độ trôi client ‖w_k − w‖₂ theo vòng, nhận xét kèm Zhao và cộng sự (2018), payload mỗi chiều / mỗi vòng và khoảng thời gian một vòng.
- Chương 2 (2.1): quy mô PlantVillage và cách cắt lá PlantDoc. Chương 3 (3.4): khối thuật ngữ và viết tắt.
- Chương 5: thêm ba hạn chế (chênh lệch miền, BatchNorm, nhãn PlantDoc và công bằng giữa client) và sáu tài liệu tham khảo (Mohanty, Noyan, Zhao, FedProx, SCAFFOLD, FedBN).
- Không đưa bộ 16 ảnh minh họa (seed 20261017) vì không có danh sách `sample_id` để tái tạo.
- `training_protocol_report.json` có thêm `lr`, `status`, `train_accuracy_by_source` cho Centralized; phần `training` của API giờ trả cả lịch sử Centralized (bỏ `duration_sec`), số thực làm tròn 6 chữ số có nghĩa, nên ngưỡng kích thước payload trong `tests/test_evidence_summary.py` nâng từ 120.000 lên 145.000 byte.

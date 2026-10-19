# Bộ câu hỏi chuẩn bị bảo vệ đồ án (52 câu)

Bộ câu hỏi bám theo website đồ án (5 chương báo cáo, 2 trang minh họa, phụ lục kỹ thuật) và mã nguồn hiện hành. Mọi con số thuộc
protocol `mixed_pv_pd_v3_pretrained_seed42_validation` (một seed, một mức lệch dữ liệu α = 0,1), lấy từ `/api/demo/evidence`,
`/api/evidence/summary` và `docs/EXPERIMENT_PROTOCOL.md`. Mức độ: **[CB]** cơ bản, **[TB]** trung bình, **[NC]** nâng cao.
Cột "Trên web" chỉ trang có thể mở ra khi trả lời.

---

## A. Lý thuyết học liên kết (8 câu)

**1. [CB] Học liên kết (federated learning) là gì, khác học tập trung ở điểm nào?**
Gợi ý: mỗi client huấn luyện trên dữ liệu của mình, chỉ gửi trọng số (hoặc phần cập nhật) lên máy chủ; máy chủ gộp thành mô hình chung
rồi gửi lại. Dữ liệu thô không rời client. Học tập trung (Centralized) gom toàn bộ ảnh về một máy.
Trên web: chương 1 (sơ đồ FedAvg), chương 3 mục 3.1.

**2. [CB] Trình bày thuật toán FedAvg và công thức gộp.**
Gợi ý: bốn bước (gửi xuống → học tại chỗ E epoch → gửi lên → gộp). Công thức: w_mới = Σ_k (n_k / n) · w_k, n_k là số ảnh học của
client k. Client nhiều ảnh đóng góp nhiều hơn. Trong thí nghiệm, số ảnh 5 client là 5.434; 7.858; 5.376; 8.495; 10.075.
Trên web: chương 3 mục 3.2.

**3. [TB] Vì sao trọng số gộp theo n_k mà không lấy trung bình đều?**
Gợi ý: trung bình theo n_k tương đương tối ưu hàm mất mát trên toàn bộ dữ liệu khi dữ liệu IID; lấy trung bình đều sẽ để client ít ảnh
có ảnh hưởng ngang client nhiều ảnh. Đổi lại, client nhiều ảnh có thể áp đảo; ở phần học liên kết cộng đồng, trọng số bị chặn ở
n_cap = 200 ảnh để một người không chiếm cả vòng.

**4. [TB] Non-IID là gì, vì sao làm khó FedAvg?**
Gợi ý: phân bố dữ liệu khác nhau giữa các client (lệch nhãn, lệch số lượng, lệch nguồn ảnh). Mỗi client tối ưu về hướng của dữ liệu
riêng (client drift), trung bình các hướng đó lệch khỏi lời giải chung, mô hình hội tụ chậm hơn và kém hơn Centralized.
Trong dự án: 18/38 nhóm có từ 90% ảnh trở lên nằm ở một client.
Trên web: chương 2 mục 2.2, ví dụ Tomato — Bacterial spot.

**5. [TB] Phân chia Dirichlet α hoạt động thế nào? α = 0,1 nghĩa là gì?**
Gợi ý: với mỗi nhóm, rút tỉ lệ chia cho K client từ Dirichlet(α·1_K); α nhỏ thì tỉ lệ dồn về một vài client (lệch nặng), α = 100 gần
IID. Dự án dùng α = 0,1 (lệch nặng), có bước sửa để mỗi client đạt số ảnh tối thiểu (`_repair_min_size` trong
`src/data/dirichlet_split.py`). Bộ dữ liệu có sẵn 12 cách chia nhưng chỉ α = 0,1 có mô hình đã huấn luyện.
Trên web: chương 2, mục gấp "Xem 12 cách chia dữ liệu có sẵn".

**6. [TB] Học liên kết có bảo đảm riêng tư tuyệt đối không?**
Gợi ý: không. FL chỉ giữ ảnh gốc ở client; trọng số vẫn có thể bị tấn công suy luận thành viên hoặc tái tạo ảnh (gradient inversion).
Muốn mạnh hơn cần differential privacy, secure aggregation. Dự án chưa có secure aggregation; phần cộng đồng có tùy chọn DP (tắt
mặc định) và cắt chuẩn cập nhật ở 1,0.
Trên web: chương 5, hạn chế "Riêng tư chưa tuyệt đối".

**7. [NC] Khác nhau giữa FedAvg và FedSGD, FedProx, SCAFFOLD?**
Gợi ý: FedSGD gửi gradient mỗi bước (tốn truyền thông); FedAvg cho client học nhiều bước rồi gửi trọng số. FedProx thêm số hạng
μ/2·‖w − w_global‖² để kéo client không trôi xa; SCAFFOLD dùng biến điều khiển để hiệu chỉnh client drift. Dự án chỉ cài FedAvg;
FedProx/SCAFFOLD là hướng phát triển cho dữ liệu lệch nặng.

**8. [NC] BatchNorm xử lý thế nào khi gộp FedAvg?**
Gợi ý: trong `aggregate_fedavg` (`src/federated/strategy.py`), tensor số thực (weight, bias, running_mean, running_var) lấy trung bình
có trọng số; bộ đếm nguyên `num_batches_tracked` lấy giá trị cha cộng phần tăng của các client. Với non-IID, thống kê BatchNorm lệch
giữa client là một nguồn sai số đã biết (hướng khác: FedBN giữ BN cục bộ).

---

## B. Dữ liệu (7 câu)

**9. [CB] Đồ án dùng bộ dữ liệu nào? Bao nhiêu nhóm?**
Gợi ý: PlantVillage (ảnh một lá chụp trong phòng, nền đơn giản) và PlantDoc (ảnh ngoài đồng, nền phức tạp). 38 nhóm lá khỏe/bệnh
theo nhãn PlantVillage; PlantDoc chỉ phủ 13/38 nhóm và không có nhóm lá khỏe.
Trên web: chương 2 mục 2.1.

**10. [CB] Chia tập thế nào? Tập kiểm tra và tập giữ riêng khác nhau ra sao?**
Gợi ý: tập học 37.238 ảnh chia cho 5 client; tập kiểm tra (validation) 4.613 ảnh (4.344 PlantVillage + 269 PlantDoc) dùng trong lúc
huấn luyện để chọn checkpoint; tập hiệu chuẩn 4.225 ảnh để học nhiệt độ T; tập giữ riêng 11.015 ảnh (10.843 + 172) không dùng để
chọn mô hình. Tập giữ riêng là bộ chẩn đoán có sẵn của bản phát hành dữ liệu (legacy), chưa nằm trong kiểm tra trùng ảnh.
Trên web: chương 2 mục 2.1 (định nghĩa), chương 4 mục 4.2.

**11. [TB] Làm sao biết không có ảnh trùng giữa tập học và tập kiểm tra (data leakage)?**
Gợi ý: kiểm tra trùng ảnh theo mã băm byte, mã ảnh, nhóm lá, cảnh chụp trên 46.076 ảnh (học + kiểm tra + hiệu chuẩn): 0 phần tử trùng,
mọi cách chia client đều là tập con của tập học. Giới hạn: tập giữ riêng chưa nằm trong phép kiểm này; khoảng 24% ảnh PlantVillage là
nhóm một ảnh nên ảnh chụp lại cùng một lá có thể lọt qua (đã làm phân tích độ nhạy, chưa thấy số bị thổi phồng).
Trên web: Phụ lục A, thẻ "Kiểm tra trùng ảnh".

**12. [TB] Tiền xử lý ảnh gồm những gì? Vì sao phải khớp giữa huấn luyện, máy chủ và trình duyệt?**
Gợi ý: đưa về 224×224 theo quy trình `canonical_v1`, chuẩn hóa theo trung bình/độ lệch ImageNet. Nếu cách resize khác nhau, đặc trưng
lệch và dự đoán đổi: đo được `createImageBitmap` của trình duyệt cho cosine tối thiểu 0,948 và đổi 2/49 dự đoán, nên dự án tự viết
resize kiểu PIL trong JavaScript, khớp từng byte, có test đối chiếu Python ↔ JS.

**13. [TB] Vì sao PlantDoc chỉ dùng ảnh đã cắt theo khung lá?**
Gợi ý: PlantDoc gốc là ảnh cả cảnh có khung bao từng lá. Mô hình học ảnh một lá nên đầu vào phải là vùng lá. Đo trên cùng ảnh PlantDoc
kiểm tra: FedAvg đạt 49,0% với ảnh cắt nhưng còn 29,7% khi đưa cả khung ảnh; Centralized 54,8% → 45,2%. Vì vậy web yêu cầu ảnh một lá
đã cắt.
Trên web: chương 5, hạn chế "Ảnh cả cây hoặc nhiều lá làm mô hình kém đi rõ rệt".

**14. [TB] Phân bố dữ liệu 5 client trông thế nào? Có client nào thiếu hẳn nhóm bệnh không?**
Gợi ý: client có từ 20 đến 27 trên 38 nhóm có ảnh; số ảnh 14,4%–27,1% tổng. Ví dụ Tomato — Bacterial spot: client 02 giữ 1.549 ảnh
(97%), client 01 có 0 ảnh. Ở các ô (client × nhóm) không có ảnh, mô hình học riêng nhận đúng 0%.
Trên web: chương 2, bảng ví dụ và mục 2.3.

**15. [NC] Dữ liệu có cân bằng không? Ảnh hưởng gì tới chỉ số?**
Gợi ý: không cân bằng giữa nhóm và giữa nguồn (PlantVillage chiếm phần lớn). Vì vậy báo cáo thêm Macro-F1 (mỗi nhóm trọng số như nhau)
và tách kết quả theo nguồn PlantVillage/PlantDoc; Macro-F1 chỉ tính trên các nhóm có ảnh kiểm tra.

---

## C. Mô hình và huấn luyện (8 câu)

**16. [CB] Dùng mô hình gì? Vì sao chọn MobileNetV3-Small?**
Gợi ý: MobileNetV3-Small, đầu ra 38 nhóm, khởi tạo từ trọng số ImageNet (pretrained, theo mã giao thức). Nhẹ, ít tham số, phù hợp truyền
trọng số nhiều vòng trong FL và chạy được trên CPU/trình duyệt (phần trích đặc trưng xuất ONNX chỉ khoảng 6 MB).

**17. [TB] Ba cách huấn luyện được so sánh là gì và mục đích của từng cách?**
Gợi ý: Centralized (gom ảnh, mức tham chiếu), FedAvg (học liên kết, phương pháp đề xuất), Local-only (5 mô hình, mỗi client tự học,
không trao đổi). Local-only cho thấy một nơi tự học thì thiếu nhóm nào; Centralized cho thấy mức có thể đạt nếu được gom dữ liệu.
Trên web: chương 3 mục 3.1.

**18. [TB] Cấu hình huấn luyện cụ thể?**
Gợi ý (EXPERIMENT_PROTOCOL mục 7): Centralized dùng AdamW lr 1e-3, tối đa 100 epoch, dừng sớm ở epoch 83, chọn epoch 22. FedAvg dùng SGD
không momentum lr 0,01, 1 epoch cục bộ mỗi vòng, 5 client, dừng ở vòng 32, chọn vòng 27. Huấn luyện trên GPU Kaggle
(Centralized khoảng 5,2 giờ, FedAvg 1,7 giờ, Local 3,3 giờ mỗi seed).

**19. [TB] Chọn checkpoint "tốt nhất" theo tiêu chí nào? Có thiên lệch không?**
Gợi ý: chọn epoch/vòng có Macro-F1 PlantDoc cao nhất trên tập kiểm tra (trong các epoch đạt ngưỡng F1 PlantVillage). Có hiệu ứng chọn:
số trên tập kiểm tra có thể hơi đẹp hơn thực tế. Vì vậy chấm thêm trên tập giữ riêng 11.015 ảnh không dùng để chọn.
Trên web: chương 4 mục 4.2, chương 5 hạn chế số 4.

**20. [NC] Vì sao Centralized dùng AdamW còn FedAvg dùng SGD? So sánh có công bằng không?**
Gợi ý: hai nhánh huấn luyện được làm riêng, cấu hình chưa cố định chung; FedAvg cũng dừng sớm khi F1 PlantVillage còn tăng. Vì vậy
khoảng cách Centralized − FedAvg gồm cả phần do cấu hình, không chỉ do học liên kết. Muốn kết luận chắc cần huấn luyện lại cùng bộ tối
ưu và ngân sách, 3 seed (ước tính khoảng 30 GPU-giờ).
Trên web: chương 5 hạn chế "So sánh Centralized và FedAvg chưa hoàn toàn công bằng".

**21. [TB] Hàm mất mát và vì sao dùng nó?**
Gợi ý: cross-entropy cho phân loại đa lớp; mô hình xuất logit, softmax ra xác suất. Lưu ý: loss là hàm mục tiêu, khác độ trễ mạng;
xác suất của một ảnh không phải độ chính xác của mô hình.

**22. [NC] Làm sao chắc phép gộp FedAvg trong mã đúng toán?**
Gợi ý: unit test độc lập `tests/test_fedavg.py` (ví dụ (100·1 + 300·3)/400 = 2,5, và tensor 4D), kiểm tra khóa/shape/hữu hạn trước khi
gộp, mã băm SHA-256 mô hình mỗi vòng; ở phiên Flower, mô hình chung của vòng trước là mô hình cha của vòng sau (chuỗi mã băm khớp).

**23. [NC] Huấn luyện một client trong FL mất bao lâu, truyền bao nhiêu dữ liệu mỗi vòng?**
Gợi ý: mỗi lượt gửi là toàn bộ trọng số MobileNetV3-Small dạng float32, cỡ vài MB (ước tính khoảng 6 MB theo kích thước mô hình); 5
client × 2 chiều × số vòng. Phần cộng đồng chỉ gửi lớp cuối 155.800 byte (khoảng 156 KB) nên nhẹ hơn khoảng 40 lần. Thời gian và RTT
từng client của phiên chạy thử nằm trong nhật ký Flower.
Trên web: Phụ lục D, "Nhật ký chi tiết từng client".

---

## D. Đánh giá và kết quả (9 câu)

**24. [CB] Kết quả chính là gì?**
Gợi ý: trên tập kiểm tra PlantVillage (4.344 ảnh), Top-1: Centralized 99,22%, FedAvg 93,12%, Local-only 44,9–55,5%. Trên PlantDoc
(269 ảnh): Centralized 53,90%, FedAvg 47,21%, Local-only 21,2–29,0%.
Trên web: chương 1 mục 1.3, chương 4 mục 4.1.

**25. [CB] Top-1, Macro-F1, "đúng loại cây" khác nhau thế nào?**
Gợi ý: Top-1 là tỉ lệ ảnh mà lựa chọn đầu tiên đúng nhóm. Macro-F1 trung bình F1 theo nhóm, không bị nhóm nhiều ảnh lấn át (FedAvg
PlantVillage 0,9019, PlantDoc 0,4687). "Đúng loại cây" tính cả khi đúng cây sai bệnh (FedAvg PlantDoc 75,5%).
Trên web: chương 3 mục 3.4, chương 4 bảng 7 mô hình.

**26. [TB] Ba câu hỏi nghiên cứu được trả lời thế nào?**
Gợi ý: (1) FedAvg thấp hơn Centralized 6,1 điểm trên PlantVillage, chỉ mang tính tham khảo vì cấu hình khác; (2) FedAvg cao hơn
Local-only tốt nhất 37,7 điểm, vì mỗi client thiếu nhiều nhóm bệnh; (3) từ PlantVillage sang PlantDoc FedAvg giảm 45,9 điểm, nguyên nhân
chưa kiểm chứng (khác kiểu ảnh, ít ảnh PlantDoc, nhãn).
Trên web: chương 5 mục 5.1.

**27. [TB] Kết quả trên tập giữ riêng có khác tập kiểm tra không?**
Gợi ý: PlantVillage gần như bằng (Centralized 99,23%, FedAvg 93,34%). PlantDoc giữ riêng chỉ 172 ảnh: Centralized 63,37%, FedAvg 50,00%,
không thấp hơn tập kiểm tra nên chưa thấy số bị đẩy lên do cách chọn checkpoint; khoảng tin cậy rộng (FedAvg 38,4–61,4%).
Trên web: chương 4 mục 4.2.

**28. [TB] Khoảng tin cậy 95% tính thế nào? Vì sao "bootstrap theo cụm"?**
Gợi ý: lấy mẫu lại có hoàn lại nhiều lần để ước lượng độ dao động; theo cụm vì nhiều ảnh PlantDoc cùng một cảnh chụp và nhiều ảnh
PlantVillage cùng một lá không độc lập, lấy mẫu từng ảnh sẽ cho khoảng hẹp giả tạo. PlantDoc kiểm tra 269 ảnh từ 115 cảnh, giữ riêng
172 ảnh từ 96 cảnh.

**29. [NC] Chênh lệch Centralized − FedAvg có ý nghĩa thống kê không?**
Gợi ý: PlantVillage có (+6,1 điểm, khoảng [4,9; 7,2], McNemar p < 0,001). PlantDoc tập kiểm tra +6,7 điểm, khoảng [−3,7; 18,0] chứa 0,
McNemar p ≈ 0,041 nhưng chưa qua hiệu chỉnh Bonferroni (0,05/8 ≈ 0,006) nên chưa đủ chắc chắn; tập giữ riêng +13,4 điểm [1,7; 24,8],
p ≈ 0,008 thì rõ hơn.
Trên web: chương 4 mục 4.2, "So sánh theo cặp".

**30. [TB] Mô hình hay nhầm những nhóm nào? Vì sao?**
Gợi ý: trên PlantDoc, FedAvg kém nhất ở 6 nhóm (136 ảnh) có tỉ lệ nhận đúng dưới 50%, ví dụ Tomato mosaic virus, Tomato early blight,
Tomato bacterial spot. Các cặp nhầm nhiều nhất của FedAvg trên PlantDoc (tập kiểm tra): Tomato mosaic virus → Tomato Leaf Mold (15 ảnh), Tomato Bacterial spot → Tomato Septoria leaf spot (12 ảnh), chủ yếu giữa các bệnh có vết lá giống nhau. FedAvg kém Centralized ở 5/13 nhóm PlantDoc.
Trên web: chương 4 mục 4.3 (bảng theo nhóm, ma trận nhầm lẫn, ảnh sai tự tin).

**31. [NC] Hiệu chuẩn xác suất (calibration) là gì? Kết quả?**
Gợi ý: mô hình báo chắc 90% thì phải đúng khoảng 90%. Đo bằng ECE (15 khoảng). Temperature scaling chia logit cho T học trên tập hiệu
chuẩn 4.225 ảnh (FedAvg T = 1,15; Centralized T = 2,93), không đổi dự đoán. PlantVillage đã sát (ECE FedAvg 0,013); PlantDoc tự tin quá
mức: FedAvg báo trung bình 76,5% nhưng chỉ đúng 47,2%, ECE 0,295 → 0,261 sau hiệu chuẩn.
Trên web: chương 4 mục 4.4.

**32. [TB] Số liệu trên web có đáng tin, có bị nhập tay không?**
Gợi ý: mọi số đọc từ registry đã kiểm SHA-256 của checkpoint và manifest; sai một byte là web ẩn số liệu (fail-closed, không có số dự
phòng). Đã chấm lại 4.613 ảnh kiểm tra trên máy: Centralized và FedAvg khớp tuyệt đối, 10/14 phép so khớp tuyệt đối, phần còn lại lệch
không quá 1 ảnh.
Trên web: Phụ lục A, B.

---

## E. Điểm yếu và hạn chế (7 câu)

**33. [CB] Hạn chế lớn nhất của nghiên cứu?**
Gợi ý: một kịch bản, một seed, một mức α = 0,1, chưa có độ dao động giữa các lần chạy; chưa có kiểm thử thực địa độc lập.
Trên web: chương 5 mục 5.2 (13 hạn chế).

**34. [TB] Vì sao không chạy nhiều seed?**
Gợi ý: mỗi seed cần khoảng 10 GPU-giờ cho 7 mô hình; máy phát triển và máy chủ chỉ có CPU. Đây là việc nghiên cứu còn lại (đã ghi trong
danh mục kiểm tra là chưa làm được).
Trên web: Phụ lục C.

**35. [TB] 5 client là thật hay giả lập?**
Gợi ý: giả lập: năm phần của cùng một bộ dữ liệu, không phải năm nông hộ; không có độ trễ mạng hay thiết bị khác nhau. Phiên Flower trên
máy chủ chạy 5 tiến trình trên cùng một máy (localhost).

**36. [TB] Mô hình có dùng ngoài thực địa được không?**
Gợi ý: chưa. Ảnh ngoài đồng PlantDoc chỉ đạt khoảng 47–54% và chỉ phủ 13 nhóm; mô hình cần ảnh một lá đã cắt; xác suất trên ảnh ngoài
đồng tự tin quá mức. Cần thêm dữ liệu thực địa và bước phát hiện lá.

**37. [NC] Máy chủ web có thật sự huấn luyện không? Vì sao độ chính xác phiên chạy thử chỉ 1/38?**
Gợi ý: có tính toán thật (Centralized dùng gần 10 lõi CPU; Flower sinh 6 tiến trình Python trao đổi trọng số qua gRPC, mỗi vòng có mã
băm mô hình mới) nhưng máy chủ không có bộ ảnh nên học trên ảnh nhiễu ngẫu nhiên ở chế độ smoke; tham số học được chỉ đổi 0,002–0,007%,
độ chính xác 1/38 đúng kỳ vọng. Mục tiêu phần này là kiểm chứng hệ thống, không đo độ chính xác.
Trên web: Phụ lục D.

**38. [NC] Nguồn gốc các checkpoint đã được kiểm chứng đầy đủ chưa?**
Gợi ý: đã kiểm mã băm của tệp công bố và chấm lại cục bộ, nhưng chưa kiểm được nguồn huấn luyện gốc trên Kaggle (cần tệp gốc); đây là
một trong 3 việc nghiên cứu còn lại cùng chạy nhiều seed và cập nhật báo cáo sau lần chạy mới.
Trên web: Phụ lục C.

**39. [NC] Có thử client lỗi hoặc client tấn công (poisoning) chưa?**
Gợi ý: có cơ chế kiểm bản cập nhật (khóa/shape/hữu hạn, mã băm cha, ngưỡng chuẩn L2 tùy chọn, chế độ chỉ ghi nhận hoặc loại bỏ) và ở
phần cộng đồng có cắt chuẩn, cổng chấp nhận; e2e đã thử một head "đầu độc" chuẩn 380 bị cắt về 1,0. Chưa có thí nghiệm đo độ chính xác
khi có tấn công, nên chưa kết luận về độ bền.

---

## F. Website, máy chủ và triển khai (8 câu)

**40. [CB] Kiến trúc website gồm những thành phần nào?**
Gợi ý: backend FastAPI (routers: suy luận, giám sát, huấn luyện, học liên kết cộng đồng; services: registry mô hình, Grad-CAM, bằng
chứng), một tiến trình uvicorn sau nginx (HTTPS); frontend HTML + ES module thuần, không bước build, router theo hash, tải lười từng
trang; Chart.js 4.4.1 và onnxruntime-web 1.20.1 tự host. Lưu trữ cộng đồng bằng SQLite.

**41. [TB] Vì sao uvicorn chỉ chạy một worker?**
Gợi ý: trạng thái vòng học cộng đồng, khóa "một tác vụ huấn luyện một lúc" và trạng thái phiên Flower nằm trong bộ nhớ tiến trình;
nhiều worker sẽ có nhiều bản trạng thái lệch nhau. Đủ cho quy mô lớp học; muốn mở rộng phải đưa trạng thái ra ngoài (Redis, cơ sở dữ
liệu) và tách huấn luyện ra hàng đợi.

**42. [TB] Trang chẩn đoán xử lý một ảnh như thế nào?**
Gợi ý: trình duyệt thu nhỏ ảnh, gửi một yêu cầu `/api/inference/all`; máy chủ tiền xử lý `canonical_v1`, chạy 7 mô hình, trả dự đoán,
xác suất gốc và đã hiệu chuẩn, số ảnh nhóm đó mỗi mô hình đã học, tùy chọn Grad-CAM (độ phân giải gốc 7×7). Ảnh tải lên tối đa 10 MB.
Trên web: Demo · Chẩn đoán ảnh.

**43. [TB] Grad-CAM cho biết gì và không cho biết gì?**
Gợi ý: vùng ảnh ảnh hưởng mạnh tới nhóm mà mô hình chọn (gradient của lớp đó theo bản đồ đặc trưng cuối). Là gợi ý trực quan độ phân
giải thấp, không phải bằng chứng nhân quả; dùng để phát hiện mô hình nhìn vào nền thay vì vết bệnh.

**44. [TB] Bảo mật website: ai được huấn luyện trên máy chủ? Chống lạm dụng thế nào?**
Gợi ý: ba API huấn luyện đòi đăng nhập quản trị (Basic/Bearer, giới hạn số lần sai); một tác vụ một lúc; trần tham số (10 epoch, 10
vòng, 3 epoch cục bộ, 900 giây, 6 lần khởi chạy mỗi 10 phút); kiểm RAM trống (500 MB cho tác vụ trong tiến trình, 1.400 MB cho Flower);
Flower chỉ nghe 127.0.0.1. Trang có CSP chặt (không `unsafe-inline`), nạp checkpoint bằng `weights_only=True`.

**45. [TB] Triển khai ở đâu, cập nhật phiên bản và quay lui thế nào?**
Gợi ý: VM Oracle Cloud (Ubuntu, 64 GB RAM, chỉ CPU), dịch vụ systemd `cropfed-demo`, nginx dùng chung với site khác. Phát hành bằng thư
mục mới (`cp -a` bản đang chạy, thay mã bằng `git archive`, dựng lại registry), chạy thử ở cổng phụ 8099 với bí mật tạm, rồi mới đổi
unit; quay lui bằng cách chép lại unit sao lưu (`docs/SERVER_DEPLOY_20261005.md`).

**46. [TB] Kiểm thử phần mềm gồm những gì?**
Gợi ý: pytest 400 test (toán FedAvg, phân chia dữ liệu, registry/mã băm, API, giao diện tĩnh, CSP, học liên kết cộng đồng); kịch bản
trình duyệt thật bằng Playwright (`e2e_community.py` 52 kiểm tra, `e2e_community_ux.py`, `e2e_training.py`), quét CSP trên Chromium,
Firefox, WebKit, kiểm tra trợ năng axe và thao tác bàn phím (`ui_audit.py`).

**47. [NC] Website hoạt động khi không có Internet không? Khi thiếu mô hình thì sao?**
Gợi ý: có, phông chữ và thư viện tự host, không gọi CDN. Thiếu hoặc sai mã băm artifact thì từng phần tắt kèm lý do (fail-closed),
không hiển thị số nửa vời; trang chủ vẫn mở được.

---

## G. Học liên kết cộng đồng từ trình duyệt (5 câu)

**48. [TB] Người dùng tham gia học liên kết từ trình duyệt như thế nào?**
Gợi ý: nhận mã từ người tổ chức, tham gia, thêm ít nhất 5 ảnh lá và tự gán nhóm, huấn luyện trên máy (Web Worker), gửi trọng số. Máy
chủ gộp theo FedAvg khi đủ K người (cấu hình hiện tại: 3, tối thiểu 2 khi hết giờ vòng; quản trị viên đổi được); mỗi mã gửi tối đa một lần mỗi vòng để chống một người
chiếm cả vòng.
Trên web: Demo · Học liên kết, trang `/quan-tri`.

**49. [NC] Vì sao trình duyệt chỉ huấn luyện lớp cuối?**
Gợi ý: phần trích đặc trưng (MobileNetV3-Small + classifier[0..1] của FedAvg v7) đóng băng, chạy bằng onnxruntime-web ra vector 1024
chiều; chỉ `classifier.3` (38 × 1024 + 38 = 38.950 tham số, 155.800 byte) được học bằng softmax regression (3 epoch, lr 0,01, batch 16).
Lý do: trình duyệt yếu (8–21 ms/ảnh trên máy bàn), truyền nhẹ (156 KB so với cỡ vài MB cả mạng), ít rủi ro phá mô hình. Đây là
federated fine-tuning lớp cuối, không phải huấn luyện lại toàn mạng.

**50. [NC] Máy chủ không có ảnh, vậy kiểm tra một vòng gộp có tốt lên hay không bằng cách nào?**
Gợi ý: trích sẵn đặc trưng của toàn bộ ảnh kiểm tra (float16, khoảng 9,5 MB) bằng đúng tệp ONNX trình duyệt dùng; đánh giá head ứng viên
chỉ là một phép nhân ma trận. Cổng từ chối nếu độ chính xác tổng giảm quá 0,5 điểm, macro-recall giảm quá ngưỡng hoặc độ chính xác
PlantDoc giảm quá 2 điểm. Ví dụ đã gặp: một người gửi 5 ảnh của 2 nhóm làm độ chính xác giảm 90,4% → 84,8%, vòng bị từ chối, mô hình cũ
giữ nguyên.

**51. [NC] Rủi ro nhãn sai hoặc người tham gia phá hoại?**
Gợi ý: nhãn do người dùng tự gán, chưa có chuyên gia kiểm. Phòng thủ: kiểm độ dài/hữu hạn/mã băm cha, đủ số ảnh tối thiểu, tổng nhãn
khớp số ảnh, cắt chuẩn cập nhật ở 1,0, chặn trọng số mỗi người ở 200 ảnh, cổng chấp nhận, hạn mức theo IP và người tham gia, khôi phục
phiên bản. Mô hình cộng đồng tách khỏi 7 mô hình đã kiểm hash và không dùng làm số liệu khoa học.

---

## H. Mở rộng (1 câu tổng hợp)

**52. [NC] Nếu có thêm thời gian, em sẽ làm gì trước?**
Gợi ý: theo thứ tự ảnh hưởng tới kết luận: (1) huấn luyện lại 3 cách cùng điều kiện, 3 seed, thêm α = 1,0 và 100; (2) thu ảnh ngoài
đồng đủ nhóm, khóa trước một tập kiểm tra cuối; (3) bước phát hiện và cắt lá; (4) FedProx/SCAFFOLD cho non-IID; (5) secure aggregation
hoặc DP; (6) thí nghiệm client lỗi/tấn công.
Trên web: chương 5 mục 5.3.

---

## Ghi nhớ khi trả lời

- Luôn nói rõ số thuộc tập nào (kiểm tra hay giữ riêng, PlantVillage hay PlantDoc) và đây là **một lần chạy**.
- Phân biệt "xác suất của một ảnh" với "độ chính xác của mô hình"; "loss" với "độ trễ".
- Khi bị hỏi về điểm yếu, nêu hạn chế kèm bằng chứng đã đo và việc cần làm tiếp, không né tránh.
- Phần huấn luyện trên máy chủ là kiểm chứng hệ thống; bằng chứng học thật là học liên kết cộng đồng và các checkpoint đã công bố.

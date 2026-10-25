# Bộ câu hỏi chuẩn bị bảo vệ đồ án (52 câu)

Bộ câu hỏi bám theo website đồ án (5 chương báo cáo, 3 trang minh họa, phụ lục kỹ thuật) và mã nguồn hiện hành. Mọi con số thuộc
protocol `mixed_pv_pd_v4_pretrained_seed42_validation` (release dữ liệu `mixed_pv_pd_v4`, một mức lệch dữ liệu α = 0,1; mô hình
web là hạt giống 42, bảng 3 hạt giống 42/123/2026 ở `artifacts/train_v4_results.json`), lấy từ `/api/demo/evidence`,
`/api/evidence/summary` và `docs/EXPERIMENT_PROTOCOL.md`. Bản cập nhật ngày 2026-10-25 thay các số của protocol v3 cũ.
Mức độ: **[CB]** cơ bản, **[TB]** trung bình, **[NC]** nâng cao. Dòng "Trên web" chỉ trang có thể mở ra khi trả lời.

---

## A. Lý thuyết học liên kết (8 câu)

**1. [CB] Học liên kết (federated learning) là gì, khác học tập trung ở điểm nào?**
Gợi ý: mỗi client huấn luyện trên dữ liệu của mình, chỉ gửi trọng số (hoặc phần cập nhật) lên máy chủ; máy chủ gộp thành mô hình chung
rồi gửi lại. Dữ liệu thô không rời client. Học tập trung (Centralized) gom toàn bộ ảnh về một máy.
Trên web: chương 1 (sơ đồ FedAvg), chương 3 mục 3.1.

**2. [CB] Trình bày thuật toán FedAvg và công thức gộp.**
Gợi ý: bốn bước (gửi xuống → học tại chỗ E epoch → gửi lên → gộp). Công thức: w_mới = Σ_k (n_k / n) · w_k, n_k là số ảnh học của
client k. Client nhiều ảnh đóng góp nhiều hơn. Trong thí nghiệm, số ảnh train của 5 client là 13.107; 11.691; 6.647; 8.863; 12.156
(tổng 52.464).
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
`src/data/dirichlet_split.py`). Release v4 có 2 cách chia (`iid` và `label_alpha_0_1`) và chỉ α = 0,1 có mô hình đã huấn luyện; các
mức α khác (1,0; 0,5; 100) chưa chạy và nằm trong việc cần làm tiếp (câu 52).
Trên web: chương 2, trình xem các cách chia dữ liệu.

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
Gợi ý: PlantVillage (ảnh một lá chụp trong phòng, nền đơn giản), PlantDoc (ảnh ngoài đồng, nền phức tạp) và nhóm "ngoài đồng, bộ
khác" gồm các bộ ảnh công khai (Mendeley, Zenodo, HuggingFace, iNaturalist, Wikimedia; 19.160 ảnh ở đợt đầu, bổ sung thêm sau đó;
chi tiết ở `docs/DATASET_V4.md`). 38 nhóm lá khỏe/bệnh theo nhãn PlantVillage; PlantDoc chỉ phủ 13/38 nhóm và không có nhóm lá
khỏe. Nhãn ảnh ngoài đồng mới chỉ được sàng lọc bằng công cụ AI, chưa có chuyên gia kiểm, và 5 nhóm còn dưới 100 ảnh ngoài đồng.
Trên web: chương 2 mục 2.1.

**10. [CB] Chia tập thế nào? Tập kiểm tra và tập giữ riêng khác nhau ra sao?**
Gợi ý: tập học 52.464 ảnh chia cho 5 client; tập kiểm tra (validation) 6.746 ảnh (4.344 PlantVillage + 269 PlantDoc + 2.133 ngoài
đồng bộ khác) dùng trong lúc huấn luyện để chọn checkpoint; tập hiệu chuẩn 7.005 ảnh để học nhiệt độ T; tập giữ riêng 13.101 ảnh
(10.843 + 172 + 2.086) không dùng để chọn mô hình. Tập giữ riêng là bộ chẩn đoán có sẵn của bản phát hành dữ liệu (legacy), chưa nằm
trong kiểm tra trùng ảnh.
Trên web: chương 2 mục 2.1 (định nghĩa), chương 4 mục 4.2.

**11. [TB] Làm sao biết không có ảnh trùng giữa tập học và tập kiểm tra (data leakage)?**
Gợi ý: kiểm tra trùng ảnh theo mã băm byte, mã ảnh, nhóm lá, cảnh chụp trên 66.215 ảnh (học + kiểm tra + hiệu chuẩn): 0 phần tử trùng,
mọi cách chia client đều là tập con của tập học, mọi ảnh khớp `byte_sha256` của manifest. Giới hạn: tập giữ riêng chưa nằm trong phép
kiểm này; 2.633 ảnh PlantVillage của tập giữ riêng không gom được nhóm lá nên ảnh chụp lại cùng một lá có thể lọt qua (đã làm phân
tích độ nhạy: bỏ các ảnh nghi chụp lại thì Centralized 99,32% → 99,29%, chưa thấy số bị thổi phồng). Với ảnh ngoài đồng, các ảnh
chụp liên tiếp của cùng một lá được gộp thành một nhóm trước khi chia tập.
Trên web: Phụ lục A, thẻ "Kiểm tra trùng ảnh".

**12. [TB] Tiền xử lý ảnh gồm những gì? Vì sao phải khớp giữa huấn luyện, máy chủ và trình duyệt?**
Gợi ý: đưa về 224×224 theo quy trình `canonical_v1`, chuẩn hóa theo trung bình/độ lệch ImageNet. Nếu cách resize khác nhau, đặc trưng
lệch và dự đoán đổi: đo được `createImageBitmap` của trình duyệt cho cosine tối thiểu 0,948 và đổi 2/49 dự đoán, nên dự án tự viết
resize kiểu PIL trong JavaScript, khớp từng byte, có test đối chiếu Python ↔ JS.

**13. [TB] Vì sao PlantDoc chỉ dùng ảnh đã cắt theo khung lá?**
Gợi ý: PlantDoc gốc là ảnh cả cảnh có khung bao từng lá. Mô hình học ảnh một lá nên đầu vào phải là vùng lá. Đo trên cùng ảnh PlantDoc
(263 khung của tập kiểm tra, 113 cảnh): FedAvg đạt 37,6% với ảnh cắt nhưng còn 25,9% khi đưa cả khung ảnh; Centralized 57,4% → 36,9%.
Trên tập giữ riêng (171 khung): FedAvg 39,8% → 33,3%, Centralized 59,7% → 50,9%. Đây là giới hạn trên (khung do người gán), không phải
độ chính xác ngoài thực địa. Vì vậy web yêu cầu ảnh một lá đã cắt; bộ phát hiện lá chưa được huấn luyện.
Trên web: chương 5, hạn chế "Ảnh cả cây hoặc nhiều lá làm mô hình kém đi rõ rệt".

**14. [TB] Phân bố dữ liệu 5 client trông thế nào? Có client nào thiếu hẳn nhóm bệnh không?**
Gợi ý: client có từ 22 đến 26 trên 38 nhóm có ảnh; số ảnh chiếm 12,7%–25,0% tổng. Ví dụ Tomato — Bacterial spot (2.023 ảnh train):
client 04 giữ 1.624 ảnh (80%), client 01 có 355, client 02 có 36, client 00 và client 03 chỉ có 4 ảnh mỗi nơi. Trên ảnh PlantVillage
kiểm tra, mô hình học riêng của client 04 nhận đúng 100% nhóm này, client 01 đạt 97%, còn client 00 chỉ 0,6%. Ở các ô (client × nhóm)
không có ảnh, mô hình học riêng không nhận ra nhóm đó.
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
Gợi ý (gói `v4b_s42`, `import_manifest.json`): cả ba cách dùng cùng một công thức: SGD động lượng 0,9, weight decay 1e-4, lr 0,02
(khởi động 1 epoch hoặc vòng rồi giảm cosine về 0), batch 128, label smoothing 0,1, bf16, tăng cường ảnh trên GPU (`gpu_v2`), trọng số hàm mất
mát `domain_balanced`, khởi tạo từ trọng số ImageNet. Ngân sách 40 đơn vị: 40 epoch (Centralized, Local-only) hoặc 40 vòng
FedAvg với 5 client, 1 epoch cục bộ mỗi vòng. Chạy trên một GPU RTX 3090 thuê, ba hạt giống song song, 21 lượt chạy mất khoảng 1 giờ
50 phút.

**19. [TB] Chọn checkpoint "tốt nhất" theo tiêu chí nào? Có thiên lệch không?**
Gợi ý: chọn epoch/vòng có trung bình Macro-F1 cao nhất trên các nguồn của tập kiểm tra có ít nhất 30 ảnh. Cả Centralized và FedAvg
đều chọn bước 33/40. Có hiệu ứng chọn: Macro-F1 PlantDoc ở bước được chọn của Centralized xếp thứ 6/40 (59,03%, cao nhất là 60,69%),
của FedAvg xếp 16/40. Vì vậy chấm thêm trên tập giữ riêng 13.101 ảnh không dùng để chọn.
Trên web: chương 4 mục 4.2, chương 5 hạn chế số 4.

**20. [NC] So sánh Centralized và FedAvg có công bằng không?**
Gợi ý: ở đợt v4b hai cách dùng cùng bộ tối ưu, cùng lr, cùng batch, cùng số bước (40) và cùng dữ liệu, nên khác biệt do cách học chứ
không phải do cấu hình (đợt v3 cũ dùng AdamW cho Centralized và SGD cho FedAvg nên không so sánh được). Phần chưa công bằng còn lại:
công thức (lr 0,02, batch 128) được chọn trên Centralized, chưa chỉnh riêng cho FedAvg; giả thuyết lr lớn làm client trôi xa nhau khi
dữ liệu lệch nhãn chưa được kiểm chứng. Cần thử lr nhỏ hơn, số epoch cục bộ khác và FedProx/FedBN cho FedAvg.
Trên web: chương 3 mục 3.3 và chương 5 mục 5.2.

**21. [TB] Hàm mất mát và vì sao dùng nó?**
Gợi ý: cross-entropy cho phân loại đa lớp; mô hình xuất logit, softmax ra xác suất. Lưu ý: loss là hàm mục tiêu, khác độ trễ mạng;
xác suất của một ảnh không phải độ chính xác của mô hình.

**22. [NC] Làm sao chắc phép gộp FedAvg trong mã đúng toán?**
Gợi ý: unit test độc lập `tests/test_fedavg.py` (ví dụ (100·1 + 300·3)/400 = 2,5, và tensor 4D), kiểm tra khóa/shape/hữu hạn trước khi
gộp, mã băm SHA-256 mô hình mỗi vòng; ở phiên Flower, mô hình chung của vòng trước là mô hình cha của vòng sau (chuỗi mã băm khớp).

**23. [NC] Huấn luyện một client trong FL mất bao lâu, truyền bao nhiêu dữ liệu mỗi vòng?**
Gợi ý: mỗi lượt gửi là toàn bộ trọng số MobileNetV3-Small dạng float32, khoảng 6,28 MB (6.275.944 byte, tính từ kích thước mô hình, chưa
đo trong lúc huấn luyện); 5 client × 2 chiều × 40 vòng ≈ 2,5 GB cho cả lần huấn luyện. Phần cộng đồng chỉ gửi lớp cuối 155.800 byte
(khoảng 156 KB) nên nhẹ hơn khoảng 40 lần. Đã đo thật một lần giữa laptop và máy chủ Oracle qua Internet (2 client × 380 ảnh × 3 vòng):
RTT 46,7 ms, tốc độ 31–34 MB/s, một vòng 22,8 s trong đó truyền khoảng 0,8 s, nên thời gian chủ yếu là tính toán. Chưa có phép đo
trong mạng LAN và chưa ghi số byte thật trong lúc huấn luyện 5 client.
Trên web: Phụ lục D, "Nhật ký chi tiết từng client".

---

## D. Đánh giá và kết quả (9 câu)

**24. [CB] Kết quả chính là gì?**
Gợi ý (hạt giống 42, tập kiểm tra): PlantVillage (4.344 ảnh) Top-1: Centralized 99,54%, FedAvg 96,71%, Local-only 29,2–55,6%.
PlantDoc (269 ảnh): Centralized 56,51%, FedAvg 37,55%, Local-only 7,4–29,7%. Ngoài đồng bộ khác (2.133 ảnh): Centralized 94,56%,
FedAvg 86,73%. Trung bình 3 hạt giống (tập giữ riêng): PlantVillage C 99,38% ± 0,16 và F 96,85% ± 0,04; ảnh ngoài đồng cùng nguồn
C 92,23% và F 83,13%; PlantDoc giữ riêng C 57,56% và F 40,50%; PlantWild (ảnh ngoài đồng bộ lạ) C 36,88% và F 29,80%.
Trên web: chương 1 mục 1.3, chương 4 mục 4.1.

**25. [CB] Top-1, Macro-F1, "đúng loại cây" khác nhau thế nào?**
Gợi ý: Top-1 là tỉ lệ ảnh mà lựa chọn đầu tiên đúng nhóm. Macro-F1 trung bình F1 theo nhóm, không bị nhóm nhiều ảnh lấn át (FedAvg
tập kiểm tra: PlantVillage 0,9602, PlantDoc 0,4541). "Đúng loại cây" tính cả khi đúng cây sai bệnh (PlantDoc: FedAvg 67,7%,
Centralized 80,3%).
Trên web: chương 3 mục 3.4, chương 4 bảng 7 mô hình.

**26. [TB] Ba câu hỏi nghiên cứu được trả lời thế nào?**
Gợi ý (tập giữ riêng, hạt giống 42): (1) FedAvg thấp hơn Centralized 2,5 điểm trên PlantVillage, 9,4 điểm trên ảnh ngoài đồng bộ khác
và 19,8 điểm trên PlantDoc, cùng công thức và số bước; (2) FedAvg cao hơn Local-only tốt nhất 41,7 điểm trên PlantVillage và 36,8 điểm
trên ảnh ngoài đồng bộ khác vì mỗi client thiếu nhiều nhóm bệnh, nhưng trên PlantDoc chỉ hơn 7,6 điểm và chưa có ý nghĩa thống kê
(khoảng [−7,3; +24,3], p ≈ 0,20); (3) từ PlantVillage sang PlantDoc FedAvg giảm 56,7 điểm (96,84% → 40,12%) và Centralized giảm 39,4
điểm (99,32% → 59,88%); nguyên nhân chưa kiểm chứng (khác kiểu ảnh, ít ảnh PlantDoc, nhãn).
Trên web: chương 5 mục 5.1.

**27. [TB] Kết quả trên tập giữ riêng có khác tập kiểm tra không?**
Gợi ý: PlantVillage gần như bằng (giữ riêng: Centralized 99,32%, FedAvg 96,84%; kiểm tra: 99,54% và 96,71%). Ảnh ngoài đồng bộ khác:
92,38% và 82,93% (kiểm tra: 94,56% và 86,73%). PlantDoc giữ riêng chỉ 172 ảnh: Centralized 59,88%, FedAvg 40,12%, không thấp hơn nhiều so
với tập kiểm tra (56,51% và 37,55%) nên chưa thấy số bị đẩy lên do cách chọn checkpoint; khoảng tin cậy rộng (FedAvg 30,4–50,6%).
Trên web: chương 4 mục 4.2.

**28. [TB] Khoảng tin cậy 95% tính thế nào? Vì sao "bootstrap theo cụm"?**
Gợi ý: lấy mẫu lại có hoàn lại nhiều lần để ước lượng độ dao động; theo cụm vì nhiều ảnh PlantDoc cùng một cảnh chụp và nhiều ảnh
PlantVillage cùng một lá không độc lập, lấy mẫu từng ảnh sẽ cho khoảng hẹp giả tạo. PlantDoc kiểm tra 269 ảnh từ 115 cảnh, giữ riêng
172 ảnh từ 96 cảnh; PlantVillage giữ riêng 10.843 ảnh từ 4.147 nhóm lá; ngoài đồng bộ khác giữ riêng 2.086 ảnh từ 1.021 nhóm. Khoảng
tin cậy hiện chỉ tính cho mô hình hạt giống 42; chưa có kiểm định ghép cặp giữa các hạt giống.

**29. [NC] Chênh lệch Centralized − FedAvg có ý nghĩa thống kê không?**
Gợi ý (tập giữ riêng, hạt giống 42): có ở cả ba nguồn. PlantVillage +2,5 điểm, khoảng [2,1; 2,9], McNemar p < 0,001; ngoài đồng bộ
khác +9,4 điểm [7,1; 12,2], p < 0,001; PlantDoc +19,8 điểm [10,4; 30,0], p ≈ 1,2e-5, vẫn nhỏ hơn ngưỡng Bonferroni 0,05/8 ≈ 0,006.
Trên tập kiểm tra PlantDoc +19,0 điểm [11,2; 26,9], p ≈ 5,2e-9. Ngược lại FedAvg so với Local-only tốt nhất trên PlantDoc chỉ +7,6
điểm [−7,3; +24,3], p ≈ 0,20: chưa có bằng chứng FedAvg hơn Local-only ở nguồn này.
Trên web: chương 4 mục 4.2, "So sánh theo cặp".

**30. [TB] Mô hình hay nhầm những nhóm nào? Vì sao?**
Gợi ý: trên PlantDoc (tập kiểm tra), FedAvg nhận đúng dưới 50% ở 8 nhóm (204/269 ảnh), ví dụ Tomato mosaic virus 0% (29 ảnh), Tomato
early blight 0%, Tomato bacterial spot 6%, Potato early blight 28%, Apple scab 28%. Các cặp nhầm nhiều nhất của FedAvg trên toàn tập
kiểm tra: Corn Northern leaf blight → Corn Gray leaf spot (52 ảnh), Tomato Late blight → Tomato Early blight (24 ảnh), Tomato
Septoria leaf spot → Tomato Early blight (22 ảnh), chủ yếu giữa các bệnh có vết lá giống nhau. FedAvg kém Centralized ở 9/13 nhóm
PlantDoc. Trên PlantDoc, 87/269 ảnh bị đoán sai cả loại cây, 81 ảnh đúng cây nhưng sai bệnh.
Trên web: chương 4 mục 4.3 (bảng theo nhóm, ma trận nhầm lẫn, ảnh sai tự tin).

**31. [NC] Hiệu chuẩn xác suất (calibration) là gì? Kết quả?**
Gợi ý: mô hình báo chắc 90% thì phải đúng khoảng 90%. Đo bằng ECE (15 khoảng). Temperature scaling chia logit cho T học trên tập hiệu
chuẩn 7.005 ảnh (FedAvg T = 0,559; Centralized T = 0,694; T < 1 nghĩa là xác suất thô còn thấp hơn độ chính xác ở ảnh cùng nguồn), không
đổi dự đoán. Trên PlantVillage hiệu chuẩn có ích (ECE FedAvg 0,227 → 0,026; Centralized 0,106 → 0,014) nhưng trên PlantDoc lại làm
xấu đi (FedAvg 0,077 → 0,245; Centralized 0,094 → 0,214). Nguyên nhân chưa kiểm chứng; giả thuyết là tập hiệu chuẩn chủ yếu là ảnh
cùng nguồn nên nhiệt độ không chuyển sang ảnh ngoài đồng. Giao diện hiển thị cả xác suất thô lẫn xác suất đã hiệu chuẩn.
Trên web: chương 4 mục 4.4.

**32. [TB] Số liệu trên web có đáng tin, có bị nhập tay không?**
Gợi ý: mọi số đọc từ registry đã kiểm SHA-256 của checkpoint và manifest; sai một byte là web ẩn số liệu (fail-closed, không có số dự
phòng). Đã chấm lại 6.746 ảnh kiểm tra trên CPU: Centralized và FedAvg khớp tuyệt đối số công bố, 21/21 phép so khớp tuyệt đối.
Trên web: Phụ lục A, B.

---

## E. Điểm yếu và hạn chế (7 câu)

**33. [CB] Hạn chế lớn nhất của nghiên cứu?**
Gợi ý: chỉ một mức non-IID (α = 0,1) và một cấu hình (5 client, 1 epoch cục bộ, 40 vòng); chưa có đường cong theo mức lệch dữ liệu;
FedAvg chạy mô phỏng trong một tiến trình, chưa có lần chạy Flower với dữ liệu thật; chưa có thuật toán chịu non-IID (FedProx, FedBN,
SCAFFOLD); ảnh ngoài đồng chưa được chuyên gia gán nhãn và PlantDoc giữ riêng chỉ 172 ảnh; PlantWild chỉ đạt 36,9% (Centralized)
và 29,8% (FedAvg). Cũng chưa đạt 7/14 ngưỡng đặt trước (PlantWild, PlantDoc giữ riêng, FedAvg ngoài đồng cùng nguồn).
Trên web: chương 5 mục 5.2.

**34. [TB] Chỉ có ba hạt giống và một mức α, vậy kết luận chắc đến đâu?**
Gợi ý: ba hạt giống (42, 123, 2026) cho độ lệch chuẩn nhỏ (PlantVillage ±0,04–0,16 điểm, PlantDoc giữ riêng ±1,5–2,4 điểm, PlantWild
±0,3 điểm), nên chênh Centralized − FedAvg không phải do may rủi hạt giống. Nhưng ba hạt giống chỉ đổi khởi tạo và thứ tự dữ liệu,
còn cách chia client (partition) giữ nguyên, và khoảng tin cậy chỉ tính cho hạt giống 42. Kết luận về mức lệch dữ liệu cần chạy thêm các
mức α khác (việc đang lên kế hoạch, mỗi lượt FedAvg khoảng 30 phút trên GPU thuê).
Trên web: chương 4 mục 4.1 (bảng 3 hạt giống), Phụ lục C.

**35. [TB] 5 client là thật hay giả lập?**
Gợi ý: giả lập: năm phần của cùng một bộ dữ liệu, không phải năm nông hộ; 40 vòng FedAvg v4b chạy mô phỏng trong một tiến trình trên
GPU thuê, không đi qua Flower. Phiên Flower trên máy chủ chạy 5 tiến trình trên cùng một máy (localhost) với ảnh nhiễu để kiểm tra hệ
thống; mới đo mạng thật một lần giữa laptop và máy chủ Oracle (2 client, 3 vòng).

**36. [TB] Mô hình có dùng ngoài thực địa được không?**
Gợi ý: chưa. Ảnh PlantDoc chỉ đạt 40–60% (FedAvg 40,1%, Centralized 59,9% trên tập giữ riêng) và chỉ phủ 13 nhóm; ảnh PlantWild từ nguồn
lạ chỉ đạt 29,8% (FedAvg) và 36,9% (Centralized), lá khỏe bị đoán thành bệnh ở 41–44% ảnh; ảnh ngoài đồng cùng nguồn với dữ liệu học
đạt 83–92% nhưng đó không phải nguồn lạ. Mô hình cần ảnh một lá đã cắt, bộ phát hiện lá chưa được huấn luyện, và chưa có ảnh tự chụp.
Cần thêm dữ liệu thực địa có chuyên gia gán nhãn và bước phát hiện lá.

**37. [NC] Máy chủ web có thật sự huấn luyện không? Vì sao độ chính xác phiên chạy thử chỉ 1/38?**
Gợi ý: có tính toán thật (Centralized dùng gần 10 lõi CPU; Flower sinh 6 tiến trình Python trao đổi trọng số qua gRPC, mỗi vòng có mã
băm mô hình mới) nhưng máy chủ không có bộ ảnh nên học trên ảnh nhiễu ngẫu nhiên ở chế độ smoke; tham số học được chỉ đổi 0,002–0,007%,
độ chính xác 1/38 đúng kỳ vọng. Mục tiêu phần này là kiểm chứng hệ thống, không đo độ chính xác.
Trên web: Phụ lục D.

**38. [NC] Nguồn gốc các checkpoint đã được kiểm chứng đầy đủ chưa?**
Gợi ý: với gói v4b đang dùng thì đã kiểm: con trỏ `artifacts/active_bundle.json` ghim mã băm manifest; 7 checkpoint khớp
`artifact_manifest`; mã băm release (68e57451…) và manifest kiểm tra khớp; mã huấn luyện được ghi kèm (commit 93c5f80fd); trọng số khởi
tạo W0 tính lại được từ hạt giống 42; chấm lại cục bộ 21/21 phép so khớp tuyệt đối. Điều chưa kiểm được là từng bước huấn luyện đã dùng
đúng split (ràng buộc qua metadata của checkpoint). Các checkpoint của lần chạy Kaggle cũ (PlantVillage thuần) không còn, chỉ còn số đo,
nên không dùng làm kết quả.
Trên web: Phụ lục B, C.

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
Gợi ý: pytest hơn 600 test (số chính xác và thời điểm chạy ở `artifacts/final_acceptance.json`: toán FedAvg, phân chia dữ liệu,
registry/mã băm, API, giao diện tĩnh, CSP, học liên kết cộng đồng, các phiên Flower nhiều tiến trình); kịch bản
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
Gợi ý: phần trích đặc trưng (MobileNetV3-Small + classifier[0..1] của FedAvg v4b) đóng băng, chạy bằng onnxruntime-web ra vector 1024
chiều; chỉ `classifier.3` (38 × 1024 + 38 = 38.950 tham số, 155.800 byte) được học bằng softmax regression (3 epoch, lr 0,01, batch 16).
Lý do: trình duyệt yếu (8–21 ms/ảnh trên máy bàn), truyền nhẹ (156 KB so với cỡ vài MB cả mạng), ít rủi ro phá mô hình. Đây là
federated fine-tuning lớp cuối, không phải huấn luyện lại toàn mạng.

**50. [NC] Máy chủ không có ảnh, vậy kiểm tra một vòng gộp có tốt lên hay không bằng cách nào?**
Gợi ý: trích sẵn đặc trưng của toàn bộ ảnh kiểm tra (float16, khoảng 9,5 MB) bằng đúng tệp ONNX trình duyệt dùng; đánh giá head ứng viên
chỉ là một phép nhân ma trận. Cổng từ chối nếu độ chính xác tổng giảm quá 0,5 điểm, macro-recall giảm quá ngưỡng hoặc độ chính xác
PlantDoc giảm quá 2 điểm. Ví dụ đã gặp ở đợt thử trước khi chuyển sang v4b (nền FedAvg v7): một người gửi 5 ảnh của 2 nhóm làm độ
chính xác giảm 90,4% → 84,8%, vòng bị từ chối, mô hình cũ giữ nguyên. Số đo của cổng trên nền v4b nằm ở
`artifacts/community_fl/simulation_report.json`.

**51. [NC] Rủi ro nhãn sai hoặc người tham gia phá hoại?**
Gợi ý: nhãn do người dùng tự gán, chưa có chuyên gia kiểm. Phòng thủ: kiểm độ dài/hữu hạn/mã băm cha, đủ số ảnh tối thiểu, tổng nhãn
khớp số ảnh, cắt chuẩn cập nhật ở 1,0, chặn trọng số mỗi người ở 200 ảnh, cổng chấp nhận, hạn mức theo IP và người tham gia, khôi phục
phiên bản. Mô hình cộng đồng tách khỏi 7 mô hình đã kiểm hash và không dùng làm số liệu khoa học.

---

## H. Mở rộng (1 câu tổng hợp)

**52. [NC] Nếu có thêm thời gian, em sẽ làm gì trước?**
Gợi ý: theo thứ tự ảnh hưởng tới kết luận: (1) chạy FedAvg và Local-only ở thêm các mức α (1,0; 0,5; `iid`), 3 hạt giống, để vẽ đường
khoảng cách với Centralized theo mức lệch dữ liệu, kèm một lần chạy Flower thật có đo byte và độ lệch trọng số từng client; (2)
FedProx, FedBN, SCAFFOLD, MOON so với FedAvg trên cùng phân hoạch; (3) lượng tử hóa mô hình và nén cập nhật, đo đánh đổi dung lượng –
độ chính xác; (4) thu ảnh ngoài đồng đủ nhóm có chuyên gia gán nhãn, khóa trước một tập kiểm tra cuối; (5) bước phát hiện và cắt lá;
(6) secure aggregation hoặc DP; (7) thí nghiệm client lỗi/tấn công. Lộ trình chi tiết ở `docs/REVIEW_HOAN_THIEN_2026-10-25.md`.
Trên web: chương 5 mục 5.3.

---

## Ghi nhớ khi trả lời

- Luôn nói rõ số thuộc tập nào (kiểm tra hay giữ riêng, PlantVillage, PlantDoc hay ngoài đồng bộ khác), thuộc hạt giống 42 hay trung
  bình 3 hạt giống, và đây là **một mức lệch dữ liệu (α = 0,1)**.
- Web đang dùng v4b dù 7/14 ngưỡng nhập đặt trước chưa đạt: nói rõ đây là lựa chọn của người thực hiện (`docs/NEXT_TRAINING_PLAN.md` mục 5.9).
- Phân biệt "xác suất của một ảnh" với "độ chính xác của mô hình"; "loss" với "độ trễ".
- Khi bị hỏi về điểm yếu, nêu hạn chế kèm bằng chứng đã đo và việc cần làm tiếp, không né tránh.
- Phần huấn luyện trên máy chủ là kiểm chứng hệ thống; bằng chứng học thật là học liên kết cộng đồng và các checkpoint đã công bố.

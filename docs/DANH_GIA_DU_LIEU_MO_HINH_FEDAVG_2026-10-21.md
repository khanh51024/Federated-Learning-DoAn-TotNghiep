# Đánh giá dữ liệu, khả năng nhận diện ảnh ngoài đồng và trao đổi trọng số FedAvg

Ngày lập: 2026-10-21. Phạm vi: ba nhánh huấn luyện `train-local-only`, `train-centralized`, `train-federated` cùng nhánh dữ liệu
`dataset` của kho GitHub của đề tài. Mọi số đo trong tài liệu được sinh bởi các tập lệnh trong thư mục `scripts/` và lưu ở
`artifacts/`; cột "Nguồn" của từng bảng nêu tên tệp. Không có mô hình nào được huấn luyện lại trong đợt này.

## 1. Tóm tắt kết luận

1. **Quy mô dữ liệu đủ cho một đồ án so sánh Centralized, FedAvg và Local-only**, nhưng chưa đủ cho mục tiêu nhận diện ảnh bất kỳ.
   Tập huấn luyện có 37.238 ảnh, trong đó 94,3% là ảnh PlantVillage chụp trong phòng thí nghiệm và chỉ 2.115 ảnh PlantDoc
   (5,7%, thuộc 14 trong 38 lớp, không có lá khỏe chụp ngoài đồng).
2. **Trên ảnh ngoài đồng độc lập (PlantWild, đã loại ảnh trùng), độ chính xác top-1 của bảy mô hình hiện có chỉ từ 10,7% đến 33,1%**
   với ảnh bệnh (21 lớp, 2.304 ảnh); Centralized đạt 33,1%, FedAvg 31,8%. Với ảnh lá khỏe ngoài vườn, các mô hình đoán thành có
   bệnh trong 87% đến 94% số ảnh. Con số 99% trên PlantVillage không phản ánh khả năng này.
3. **Cải thiện được mà không cần huấn luyện lại ở mức hạn chế.** Kỹ thuật tăng cường lúc suy luận (TTA), giữ tỉ lệ ảnh và gộp hai mô hình
   không vượt ngưỡng chấp nhận đặt trước. Biện pháp hiệu quả nhất là để người dùng chọn loại cây: top-1 của Centralized tăng từ 7,6 đến 13,3 điểm và của FedAvg
   từ 10,4 đến 21,2 điểm trên các tập PlantDoc, và tăng khoảng 25 điểm trên ảnh PlantWild (33,1% lên 58,5% với Centralized). Chức năng này đã được thêm vào trang
   Chẩn đoán.
4. **Mô hình nền mạnh hơn có số đo thật.** Không huấn luyện thêm lớp nào, DINOv2-small với kNN trên thư viện 3.635 ảnh (PlantDoc cộng PlantVillage) đạt 50,3% top-1 trên ảnh bệnh
   PlantWild, so với 33,1% của Centralized; BioCLIP zero-shot đạt 35,6% mà không dùng ảnh huấn luyện nào. Trên ảnh lá khỏe ngoài vườn mọi cách đều yếu (tối đa 25,8% với BioCLIP zero-shot).
5. **Nguyên nhân chính nằm ở dữ liệu và cách huấn luyện, không nằm ở đường truyền hay phép gộp trọng số.** Dữ liệu huấn luyện
   lệch về ảnh phòng thí nghiệm, tăng cường ảnh chỉ có lật ngang, ảnh PlantDoc bị học thuộc (độ chính xác 99,7% trên ảnh đã học,
   53,9% trên ảnh kiểm tra), và đã tồn tại một lỗi giải mã ảnh trên web (không xoay theo EXIF) nay đã được sửa.
6. **Trọng số gửi qua lại giữa server và client không bị thay đổi.** Cả 244 tensor của bảy mô hình đi qua đường tuần tự hóa của
   Flower và giao thức protobuf mà không khác một byte nào. Phép gộp float64 so với float32 làm 38,2% phần tử trọng số lệch tối đa
   3,8·10⁻⁶ và không đổi dự đoán của ảnh nào trong 4.613 ảnh kiểm tra. Nếu trọng số bị nén thì phép kiểm phát hiện được: float16 đổi dự đoán
   6 ảnh, bfloat16 đổi 71 ảnh, int8 làm độ chính xác tụt còn 2,8%.
7. **Khoảng cách giữa FedAvg và Centralized không do việc truyền hay gộp trọng số.** Với FedSGD, mô hình gộp trùng gradient descent
   tập trung tới 4,9·10⁻¹⁴ (float64). Chênh lệch chỉ xuất hiện khi mỗi client chạy nhiều bước cục bộ trên dữ liệu lệch nhãn và tăng
   khi α giảm. Hai checkpoint đã công bố còn khác nhau ở bộ tối ưu (AdamW so với SGD không động lượng), nên khoảng cách trọng số
   58% giữa chúng gồm cả phần do cấu hình.
8. **Tốc độ giữa hai máy**: laptop và VM Oracle qua Internet có độ trễ khứ hồi 46,7 ms, băng thông 31 đến 34 MB/s mỗi chiều. Một vòng
   FedAvg mất 22,8 giây, trong đó truyền trọng số chiếm khoảng 0,8 giây (3,4%) và phần còn lại là thời gian tính của máy chậm nhất.

## 2. Bộ dữ liệu và ba nhánh huấn luyện

### 2.1 Ba nhánh dùng chung gì và khác gì

| Nội dung | Centralized | FedAvg | Local-only |
|---|---|---|---|
| Mô hình | MobileNetV3-Small, khởi tạo ImageNet, cùng trọng số khởi đầu W0 | giống | giống |
| Dữ liệu | `pv_pd_v3` train 37.238 ảnh | cùng ảnh, chia 5 client lệch nhãn (Dirichlet α=0,1) | mỗi client chỉ dùng shard của mình |
| Tiền xử lý | `canonical_v1`: resize thẳng 224×224, chỉ lật ngang | giống | giống |
| Bộ tối ưu | AdamW lr 1e-3, chọn epoch 22 trên 83 | SGD lr 0,01, không động lượng, 1 epoch mỗi vòng, chọn vòng 27 trên 32 | SGD như FedAvg, 60 vòng |
| Nơi chạy | Google Colab | Kaggle | laptop có GPU |
| Top-1 PlantVillage / PlantDoc (kiểm tra) | 99,22% / 53,90% | 93,12% / 47,21% | 44,9–55,5% / 21,2–29,0% |
| Top-1 PlantVillage / PlantDoc (giữ riêng) | 99,23% / 63,37% | 93,34% / 50,00% | 45,5–55,5% / 24,4–26,2% |

Nguồn: `artifacts/branch_demo/20261003/*/validation_metrics.json`, `artifacts/holdout_eval_report.json`, `PROJECT_STATUS.md`. Các nhánh
`train-centralized-v6`, `train-fedavg-non-iid-v7`, `train-local-only-a01-v7` đã bị xóa trên GitHub nhưng commit của chúng trùng
đúng đầu nhánh hiện có (xem `artifacts/branch_demo/20261003/import_manifest.json`).

Hai điểm khiến phép so sánh Centralized với FedAvg chưa thật công bằng: bộ tối ưu khác nhau và ngân sách huấn luyện khác nhau. Hạn chế này
đã được ghi trong `docs/EXPERIMENT_PROTOCOL.md` mục 7 và được lượng hóa thêm ở mục 4.4 dưới đây.

### 2.2 Quy mô so với các nghiên cứu khác

| Bộ dữ liệu / nghiên cứu | Quy mô | Điều kiện ảnh | Mốc độ chính xác đã công bố |
|---|---|---|---|
| PlantVillage (Mohanty và cộng sự, 2016) | 54.306 ảnh, 38 lớp | phòng thí nghiệm, nền đơn giản | 99,35% trên tập giữ riêng; 31,4% trên ảnh từ nguồn khác |
| PlantDoc | 2.598 điểm dữ liệu, 13 loài, tối đa 17 bệnh | ảnh Internet, ngoài đồng | CNN đơn thường 60–75%; ViT/ConvNeXt 76–94% tùy bài và cách chia (trích từ các bài khác nhau, không cùng quy trình) |
| PlantWild (Wei và cộng sự, 2024) | 18.542 ảnh, 89 lớp (bản 2: 11.488 ảnh, 115 lớp) | ảnh Internet, ngoài đồng | baseline CoOp 61,2%, CLIP-Adapter 56,5%, Tip-Adapter-F 58,8% |
| PlantVillage, thiên lệch nền (Noyan, 2022) | 54.306 ảnh | phòng thí nghiệm | mô hình chỉ dùng 8 điểm ảnh nền đã đạt 49,0% trên tập giữ riêng, so với 2,6% khi đoán ngẫu nhiên |
| Plantix, thử ngoài đồng tại Ấn Độ (CABI, MSSRF, ICRISAT) | 900 ca | ảnh nông dân chụp | đúng ở lần trả lời đầu 51%; trong danh sách rút gọn, đúng ở vị trí đầu 75% và trong ba vị trí đầu 80% |

Nguồn đối chiếu: các liên kết ở cuối tài liệu. Các số của PlantDoc và Plantix là số công bố thứ cấp hoặc từ đơn vị phát triển, nên chỉ dùng làm mốc
tham khảo, không dùng để kết luận về mô hình của đề tài.

Nhận xét: PlantVillage ở quy mô hàng chục nghìn ảnh là kích thước chuẩn của các bài báo học liên kết về bệnh cây trồng, nên quy mô dữ liệu không phải
điểm yếu cho phần so sánh ba phương pháp huấn luyện. Điểm yếu nằm ở thành phần dữ liệu: các nghiên cứu độc lập đều ghi nhận độ chính xác
sụt mạnh khi mô hình học trên ảnh phòng thí nghiệm gặp ảnh ngoài đồng, và đề tài tái hiện đúng hiện tượng đó (mục 3.1).

## 3. Nhận diện ảnh ngoài đồng

### 3.1 Kết quả trên tập ảnh ngoài đồng độc lập

Tập thử được dựng từ PlantWild (`scripts/build_external_eval_set.py`). Chỉ giữ các lớp khớp đúng nghĩa với 38 lớp của đề tài; loại 288 ảnh gần trùng
cảnh PlantDoc, 42 ảnh gần trùng PlantVillage (dHash 64 bit, khoảng cách Hamming không quá 6) và 110 ảnh trùng nhau trong PlantWild. Còn lại 5.045 ảnh
thuộc 33 lớp, khóa bằng SHA-256 `5adba619…`. Giấy phép của PlantWild là CC BY-NC-ND 4.0, vì vậy ảnh chỉ dùng để đánh giá và không đưa vào git.

| Mô hình | Ảnh bệnh, top-1 (2.304 ảnh, 21 lớp) | Khoảng tin cậy 95% | Top-3 | Ảnh lá khỏe: đoán thành có bệnh (2.741 ảnh) |
|---|---|---|---|---|
| Centralized | 33,07% | 31,25–35,03 | 51,30% | 86,9% |
| FedAvg | 31,81% | 29,95–33,64 | 52,04% | 94,0% |
| Gộp Centralized và FedAvg | 34,81% | 32,90–36,89 | 53,99% | 89,9% |
| Local-only (năm mô hình) | 10,68–23,91% | — | 23,3–35,8% | 93,0–95,0% |

Nguồn: `artifacts/external_eval_results.json`. Nhãn "lá khỏe" của PlantWild là nhãn ngầm định (thư mục "<cây> leaf"), nên tầng này chỉ nên đọc như
tín hiệu về xu hướng báo bệnh cho lá không có triệu chứng, không phải độ chính xác. Các cặp nhầm lẫn nhiều nhất: Tomato Bacterial spot thành Septoria
leaf spot (50 ảnh), Apple scab thành Tomato Septoria leaf spot (46 ảnh), Apple healthy thành Apple scab (99 ảnh).

Kết quả trên ảnh PlantDoc (đã cắt lá, tập kiểm tra và tập giữ riêng) và trên ảnh PlantDoc nguyên khung nằm ở `artifacts/inference_variants_eval.json` và
`artifacts/fullframe_eval.json`: Centralized đạt 53,9% và 63,4% trên ảnh cắt, 54,9% và 57,9% trên ảnh nguyên khung; FedAvg đạt 47,2% và 50,0% trên ảnh cắt,
43,4% và 43,2% trên ảnh nguyên khung. Các số nguyên khung ở đây tính theo từng cảnh (mỗi cảnh một phiếu), còn `fullframe_eval.json` tính theo từng khung lá
(cảnh có nhiều lá có trọng số lớn hơn), nên con số FedAvg 29,7% và 30,4% ở mục 3.2 khác với số này mà không mâu thuẫn.

### 3.2 Nguyên nhân, kèm bằng chứng

| Nguyên nhân | Bằng chứng |
|---|---|
| Dữ liệu huấn luyện lệch về ảnh phòng thí nghiệm | PlantDoc chỉ chiếm 2.115 trong 37.238 ảnh (5,7%), 14 trong 38 lớp; không có lá khỏe ngoài đồng |
| Lớp không có ảnh ngoài đồng gần như không nhận được | 5.569 lá PlantDoc bị cách ly: Centralized 3,4%, FedAvg 2,1% (`artifacts/quarantined_plantdoc_eval.json`) |
| Học thuộc ảnh ngoài đồng ít ỏi | Độ chính xác PlantDoc trên ảnh đã học 99,7% so với 53,9% trên ảnh kiểm tra; bộ lấy mẫu có hoàn lại lặp một ảnh tới 14 lần mỗi epoch |
| Tăng cường ảnh tối thiểu | Chỉ `RandomHorizontalFlip`; các bộ tăng cường mạnh hơn có trong mã nhưng không được dùng khi huấn luyện |
| Đầu vào là một lá đã cắt, nhưng người dùng gửi cả ảnh | Ảnh nguyên khung làm FedAvg tụt từ 49,0% xuống 29,7% trên tập kiểm tra |
| Độ tin cậy quá cao khi sai | Trên PlantDoc, 77 trong 269 ảnh bị Centralized đoán sai với độ tin cậy trên 0,9 |
| Lỗi giải mã ảnh trên web | Ảnh chụp dọc bằng điện thoại mang thẻ xoay EXIF không được xoay về thẳng; ảnh PNG trong suốt không được dán lên nền đen như lúc huấn luyện |

### 3.3 Các cách cải thiện không cần huấn luyện lại

Đo bằng `scripts/eval_inference_variants.py` trên sáu tập (PlantDoc đã cắt, PlantDoc nguyên khung, mẫu PlantVillage; mỗi loại có tập kiểm tra và tập giữ riêng),
với sáu góc nhìn cho mỗi ảnh. Điều kiện chấp nhận được đặt trước khi chạy: một biến thể chỉ được đưa lên web nếu so với "gốc" cùng mô hình thỏa đồng thời
(1) tăng ít nhất 2,0 điểm top-1 trên cả hai tập ảnh nguyên khung, (2) không giảm quá 2,0 điểm trên ảnh cắt, (3) không giảm quá 0,3 điểm trên PlantVillage.

| Biến thể (Centralized, top-1 %) | PlantDoc cắt, kiểm tra | PlantDoc cắt, giữ riêng | Nguyên khung, kiểm tra | Nguyên khung, giữ riêng | Đạt ngưỡng |
|---|---|---|---|---|---|
| Gốc | 53,90 | 63,37 | 54,87 | 57,89 | — |
| Lật ngang và lấy trung bình (TTA 2 góc) | 54,65 | 65,70 | 49,56 | 56,84 | không |
| TTA 4 góc (thêm cắt giữa 80%) | 57,25 | 65,12 | 51,33 | 58,95 | không |
| Sáu góc nhìn | 57,62 | 65,70 | 50,44 | 61,05 | không |
| Giữ tỉ lệ (đệm đen thành hình vuông) | 46,84 | 56,98 | 41,59 | 52,63 | không (giảm) |
| Tự đoán loại cây rồi giới hạn | 53,90 | 63,95 | 53,98 | 58,95 | không |
| Gộp Centralized và FedAvg | 55,39 | 63,37 | 52,21 | 64,21 | không |
| **Người dùng chọn đúng loại cây** | **66,54** | **70,93** | **68,14** | **70,53** | cận trên, báo riêng |

Kết quả của FedAvg cùng xu hướng: chọn đúng loại cây nâng top-1 từ 47,21% lên 57,62% và từ 50,00% lên 60,47% trên ảnh cắt; từ 43,36% lên 64,60% và từ 43,16% lên
57,89% trên ảnh nguyên khung. Không biến thể nào trong số các biến thể tự động đạt ngưỡng, vì mức tăng không nhất quán giữa tập kiểm tra và tập giữ riêng và các khoảng tin cậy
theo cụm cảnh đều chứa 0 (ví dụ TTA 4 góc của Centralized: +3,4 [−0,4; +6,9] trên PlantDoc cắt kiểm tra, nhưng −3,5 [−8,8; +1,8] trên ảnh nguyên khung kiểm tra).
Giữ tỉ lệ làm giảm độ chính xác vì mô hình được học trên ảnh bị ép về hình vuông.

Chức năng "chọn loại cây" nằm ngoài ngưỡng tự động vì nó dựa vào thông tin do người dùng cung cấp: nếu chọn sai cây thì kết quả sai, và giao diện nói rõ điều đó.
Trên PlantWild (Centralized, ảnh bệnh), chọn đúng cây nâng top-1 từ 33,07% lên 58,46%.

Xác suất chuẩn hóa lại trong một nhóm cây luôn cao, và nhóm chỉ có một lớp (Blueberry, Orange, Raspberry, Soybean, Squash) luôn ra 100%, nên giao diện không dùng con số đó một mình.
Máy chủ trả thêm phần xác suất gốc (38 lớp) mà mô hình đặt vào cây được chọn; dưới 50% thì kết quả bị gắn cờ "không chắc" và giao diện ghi rõ mô hình gần như không thấy cây đó trong ảnh.
Nhiệt độ hiệu chuẩn T được học trên phân bố 38 lớp nên không áp cho kết quả đã giới hạn theo cây.

Từ chối trả lời khi không chắc cũng cải thiện độ chính xác của phần được trả lời: với PlantWild (ảnh bệnh, Centralized), khi chỉ trả lời 80%, 60% và 40% số ảnh tự tin
nhất, độ chính xác là 37,2%, 43,6% và 53,3% (so với 33,1% khi trả lời tất cả). AUROC của độ tin cậy trong việc phát hiện ảnh sai trên các tập PlantDoc nằm trong khoảng
0,68 đến 0,78, nghĩa là độ tin cậy có giá trị phân biệt nhưng không đủ để dùng làm cơ chế duy nhất. Quy tắc cờ "không chắc" hiện có trên web được giữ nguyên.

### 3.4 Lỗi giải mã ảnh trên web (đã sửa)

Đường đọc ảnh tải lên của web chỉ gọi `convert("RGB")`, trong khi lúc huấn luyện và chấm điểm ảnh được xoay theo EXIF và ảnh có kênh trong suốt được dán lên nền đen. Hàm dùng chung
`src/common/image_io.py::to_rgb` nay áp dụng đúng quy tắc huấn luyện ở bốn nơi (tải lên, Grad-CAM, suy luận đã công bố, dự đoán cộng đồng), có sáu kiểm thử trong
`tests/test_image_io.py`, gồm kiểm thử gọi hai lần không xoay hai lần. Ba thư mục ảnh thử `D:\demo*` không mang thẻ EXIF nên không bị ảnh hưởng; lỗi chỉ tác động tới ảnh chụp dọc từ điện thoại.

### 3.5 Phương pháp mạnh hơn: số đo thật và số công bố

**Số đo thật** (`scripts/eval_foundation_models.py` → `artifacts/foundation_model_eval.json`; không huấn luyện lớp nào; cùng các tập và cùng khóa SHA-256 PlantWild như trên).
BioCLIP zero-shot so ảnh với câu mô tả 38 lớp. kNN bỏ phiếu k láng giềng cosin trên đặc trưng đóng băng, thư viện tham chiếu gồm 2.115 lá PlantDoc trong tập học và 40 ảnh PlantVillage mỗi lớp.
Cột "chọn đúng cây" chỉ xét lớp cùng loại cây với nhãn thật (cận trên, như mục 3.3).

| Phương pháp (top-1 %) | PlantWild bệnh | chọn đúng cây | PlantWild lá khỏe (nhãn ngầm định) | PlantDoc cắt, kiểm tra | PlantDoc cắt, giữ riêng |
|---|---|---|---|---|---|
| MobileNetV3-Small Centralized (đã huấn luyện, 1,5 triệu tham số) | 33,07 | 58,46 | 2,77 | 53,90 | 63,37 |
| MobileNetV3-Small FedAvg | 31,81 | 56,29 | 2,23 | 47,21 | 50,00 |
| BioCLIP zero-shot (ViT-B/16, không ảnh huấn luyện) | 35,59 | 52,26 | 25,83 | 20,07 | 23,84 |
| BioCLIP + kNN, k=10 | 42,32 | 59,81 | 16,27 | 49,81 | 55,23 |
| DINOv2-small + kNN, k=1 | 46,05 | 58,29 | 15,72 | 52,42 | 41,86 |
| DINOv2-small + kNN, k=10 | **50,30** | 60,07 | 19,08 | **55,39** | 53,49 |

Đọc kết quả:
- Một bộ đặc trưng đóng băng cỡ 22 triệu tham số (DINOv2-small) kết hợp kNN trên thư viện nhỏ vượt MobileNetV3 đã huấn luyện đầy đủ 17 điểm trên ảnh bệnh ngoài đồng, và tương đương hoặc hơn trên PlantDoc kiểm tra.
  Khi cả hai đều biết đúng loại cây, khoảng cách thu hẹp còn 1,6 điểm (60,07% so với 58,46%): phần lớn lợi thế của mô hình nền nằm ở việc phân biệt loại cây.
- BioCLIP zero-shot không dùng ảnh huấn luyện nào mà vẫn đạt 35,6% trên PlantWild, cao hơn MobileNetV3, nhưng chỉ đạt 20–24% trên ảnh PlantDoc đã cắt (ảnh nhỏ, bị cắt sát lá), nên kết quả không đồng đều giữa các nguồn ảnh.
- Với ảnh lá khỏe ngoài vườn, mọi phương pháp đều yếu (16–26%); BioCLIP zero-shot tốt nhất vì không phụ thuộc thư viện không có lá khỏe ngoài đồng. MobileNetV3 chỉ đạt 2–3% vì gần như luôn đoán thành có bệnh.
- Thư viện kNN dùng ảnh PlantDoc và PlantVillage cũng nằm trong tập học của MobileNetV3, nên so sánh theo cùng nguồn dữ liệu; hai mô hình nền khác MobileNetV3 ở số tham số và dữ liệu tiền huấn luyện nên đây là mốc tham khảo, không thay mô hình của đề tài.
- Hàm ý cho học phân tán: nếu dùng đặc trưng đóng băng của một mô hình nền thì chỉ cần học và gửi phần đầu phân loại (384 chiều × 38 lớp ≈ 14,6 nghìn tham số, khoảng 58 KB mỗi lần truyền, nhỏ hơn bản MobileNetV3 đầy đủ 6,3 MB hơn 100 lần). Đây là giả thuyết chưa huấn luyện thử.

**Số công bố của các nghiên cứu khác** (không phải kết quả đo trên mô hình của đề tài):

| Hướng | Mốc công bố | Nhận xét cho đề tài |
|---|---|---|
| Bộ phân loại ViT/ConvNeXt tinh chỉnh trên ảnh ngoài đồng | PlantDoc 72–94% tùy bài | cần huấn luyện lại; kích thước mô hình lớn hơn nhiều, ảnh hưởng tới lượng trọng số truyền qua mạng |
| Hai tầng: tìm lá bằng bộ phát hiện rồi phân loại | PlantDoc 78,5% (YOLO + ECA-NFNet) | `scripts/train_leaf_detector.py` đã có; kế hoạch cần GPU |
| Mô hình nền (BioCLIP) không huấn luyện thêm | khoảng 28% trên PlantDoc theo một bản chép lại bảng của bài báo (chưa đối chiếu với bài gốc); đề tài đo được 20–24% trên PlantDoc cắt và 35,6% trên PlantWild | xem số đo thật ở trên |
| Thêm dữ liệu ngoài đồng đã kiểm nhãn | PlantVillage cộng ảnh PlantDoc cắt cho kết quả tổng quát hóa tốt hơn rõ rệt theo nhiều nghiên cứu | khả thi nhất: 5.569 lá PlantDoc đang cách ly và PlantWild |

Khuyến nghị theo thứ tự ưu tiên: (0) cân nhắc dùng đặc trưng đóng băng của mô hình nền (DINOv2-small) làm backbone và chỉ học phần đầu phân loại, vì số đo thật cho thấy hướng này vượt MobileNetV3 trên ảnh ngoài đồng và cho phép truyền rất ít byte;
khi có máy huấn luyện: (1) dựng release dữ liệu mới có ảnh ngoài đồng đã kiểm nhãn và lá khỏe; (2) dùng bộ tăng cường mạnh `src/data/robust_transforms.py`
(đã kiểm khô bằng `scripts/preview_augmentations.py`); (3) dùng lấy mẫu không lặp và làm trơn nhãn (label smoothing); (4) so sánh công bằng ba phương pháp với cùng họ bộ tối ưu và cùng ngân sách; (5) ba hạt giống ngẫu nhiên;
(6) dùng tập PlantWild đã khóa mã băm làm tập thử ngoài đồng cố định. Chi tiết và ngưỡng chấp nhận nằm ở `docs/NEXT_TRAINING_PLAN.md`.

## 4. Trọng số trao đổi giữa server và client

### 4.1 Đường truyền giữ nguyên từng byte

`scripts/audit_weight_exchange.py` đưa mỗi state_dict (244 tensor: 210 float32 và 34 int64 `num_batches_tracked`) qua đúng chuỗi mà Flower dùng khi gửi qua mạng:
mảng numpy, `Parameters` (`np.save`), protobuf (tuần tự hóa như trên dây gRPC) rồi ngược lại, và qua `get_parameters`/`set_parameters` của mô hình. Kết quả cho cả bảy checkpoint:
so từng byte, kiểu dữ liệu, hình dạng, thứ tự khóa và SHA-256 `compute_model_hash` đều trùng. Một mô hình chiếm 6.275.944 byte tensor và 6.307.958 byte trên dây (phụ phí 32.014 byte, 0,51%).
Không có float16, lượng tử hóa hay nén ở bất kỳ bước nào trên đường đi của FedAvg.

### 4.2 Làm tròn khi gộp và độ nhạy của phép kiểm

| Phép đo (4.613 ảnh kiểm tra) | Kết quả |
|---|---|
| Gộp 5 mô hình Local-only theo n_k của FedAvg: tích lũy float64 rồi làm tròn một lần, so với tích lũy thuần float32 | 38,2% phần tử khác nhau, lệch tuyệt đối tối đa 3,8·10⁻⁶; 0 ảnh đổi dự đoán; logit lệch tối đa 1,9·10⁻⁴ |
| FedAvg v7 sau khi nén float16 | 6 ảnh đổi dự đoán; top-1 chung giảm 0,02 điểm |
| FedAvg v7 sau khi nén bfloat16 | 71 ảnh đổi dự đoán; top-1 chung giảm 0,37 điểm |
| FedAvg v7 sau khi lượng tử hóa int8 mỗi tensor | 4.480 ảnh đổi dự đoán; top-1 chung còn 2,82% |
| Centralized sau float16, bfloat16, int8 | 3, 8 và 4.525 ảnh đổi dự đoán |

Nguồn: `artifacts/weight_exchange_audit.json`. Phép gộp ở `src/federated/strategy.py` (float64 rồi làm tròn một lần) đúng như thiết kế; nén xuống float16 gần như vô hại nhưng không bit-exact,
còn int8 phá hủy mô hình. Đồng thời, các phép nén cho thấy phép so byte đủ nhạy để phát hiện sai lệch thật.

### 4.3 Phép gộp không gây sai số; chênh lệch đến từ học cục bộ

Thí nghiệm trên lớp cuối softmax 1024→38 với đặc trưng của FedAvg v7 (`scripts/aggregation_equivalence.py`, 4.225 ảnh học, 11.015 ảnh thử giữ riêng; không tạo mô hình mới):

- **FedSGD** (mỗi client một bước gradient đầy đủ trên shard Dirichlet α=0,1, rồi gộp theo n_k) trùng 100 bước gradient descent tập trung: chênh tối đa 4,9·10⁻¹⁴ ở float64 và 7,5·10⁻⁷ ở float32 (cỡ vài lần ε của float32). Độ chính xác hai bên bằng nhau (94,498%).
- Với tổng 100 bước cục bộ không đổi, chia thành R vòng × E bước:

| α | E bước cục bộ | Khoảng cách tương đối tới lời giải tập trung | Độ chính xác mô hình gộp | Trung bình mô hình client trước khi gộp |
|---|---|---|---|---|
| 100 (gần IID) | 1 | 1,3·10⁻¹⁴ | 94,50% | 94,45% |
| 100 | 5 | 0,110 | 94,65% | 94,58% |
| 1,0 | 1 | 8,3·10⁻¹⁵ | 94,50% | 94,43% |
| 1,0 | 20 | 0,399 | 94,54% | 93,75% |
| 0,1 | 1 | 1,4·10⁻¹⁴ | 94,50% | 94,18% |
| 0,1 | 5 | 0,596 | 94,04% | 92,70% |
| 0,1 | 20 | 0,573 | 93,49% | 90,03% |

Với E = 1 mô hình gộp trùng mô hình tập trung ở mọi mức lệch nhãn. Khi E > 1, mô hình gộp lệch khỏi lời giải tập trung và lệch nhiều hơn khi dữ liệu lệch hơn; mô hình gộp tốt hơn trung bình các mô hình client
trước khi gộp (93,49% so với 90,03% ở α=0,1, E=20), tức phép gộp sửa bớt một phần độ trôi của client. Đây là hiện tượng client drift đã được mô tả trong tài liệu về dữ liệu non-IID,
không phải sai số truyền.

### 4.4 Centralized, FedAvg và Local-only về trọng số và hành vi

Nguồn: `artifacts/model_weight_comparison.json` (trọng số tách nhóm lớp, thống kê BatchNorm tách riêng; 4.613 ảnh kiểm tra).

| Đại lượng | Kết quả |
|---|---|
| Trọng số học được, Centralized so với FedAvg | khoảng cách tương đối 0,581; cosine 0,815; khối đặc trưng thấp (0–3) cosine 0,986; lớp đầu ra cosine 0,485 |
| Khoảng cách tới trọng số ImageNet (trọng số học được, không tính lớp đầu ra) | Centralized 57,8%; FedAvg 2,4%; Local-only 1,6–2,7% |
| Trùng khớp dự đoán Centralized và FedAvg | 90,6% (PlantVillage 93,0%, PlantDoc 53,2%) |
| CKA tuyến tính đặc trưng 1024 chiều | Centralized–FedAvg 0,383; FedAvg–Local-only 0,80–0,84 |
| Nội suy trọng số Centralized → FedAvg (top-1 trên mẫu 1.000 ảnh) | α=0: 86,7%; 0,25: 87,1%; 0,5: 80,4%; 0,75: 63,9%; 1: 80,7% |

Đọc kết quả: FedAvg v7 chỉ rời khỏi trọng số ImageNet 2,4% (các Local-only 1,6–2,7%), tức là mới được tinh chỉnh nhẹ (SGD lr 0,01 không động lượng), trong khi Centralized đi xa hơn 24 lần (AdamW lr 1e-3, 22 epoch).
Phần lớn khoảng cách 58% giữa hai mô hình vì thế phản ánh khác biệt cấu hình huấn luyện, không phải riêng việc học phân tán. Phép nội suy chỉ là phép thăm dò: thống kê BatchNorm được nội suy tuyến tính và không được tính lại theo từng α, nên rào cản ở α=0,75 có thể bị phóng đại.
Điểm α=0,25 cho top-1 PlantDoc 56,5% trên mẫu này, cao hơn hai đầu mút (53,9% và 47,2%); với mẫu nhỏ như vậy không đủ để kết luận hai mô hình cùng hay khác một thung lũng. Để tách riêng ảnh hưởng của học phân tán cần huấn luyện lại với cùng bộ tối ưu và ngân sách (xem mục 7).

### 4.5 Độ trôi của client: phần lớn số liệu cũ là thống kê BatchNorm

Biểu đồ "độ trôi client" của web trước đây vẽ ‖w_k − w‖₂ tính trên mọi tensor float, kể cả `running_mean` và `running_var` của BatchNorm. Tách hai phần từ lịch sử 27 vòng của FedAvg v7
(`scripts/report_training_protocol.py`, 135 cặp vòng–client): thống kê BatchNorm chiếm ít nhất 98,9% bình phương độ lớn cập nhật, trung bình 99,7%. Phần trọng số học được giảm từ khoảng
1,8–2,1 ở vòng 1 xuống 0,33–0,43 ở vòng 27, còn tổng số cũ nằm ở khoảng 58–62 rồi 9–12. Biểu đồ và bảng trên trang Kết quả nay vẽ phần trọng số học được và hiển thị riêng thống kê BatchNorm,
kèm lời giải thích. Một hệ quả cho thiết kế: ngưỡng chuẩn cập nhật của máy chủ phải tính theo phần trọng số học được nếu muốn phát hiện client bất thường, vì phần BatchNorm lấn át số tổng.

### 4.6 Kiểm mã băm hai chiều: tài liệu và mã nguồn đã được đồng bộ

`docs/ARCHITECTURE.md` ghi rằng server xác thực băm client gửi lên, nhưng máy chủ Flower trước đây chỉ kiểm băm mô hình cha (chiều xuống). Máy chủ nay tính lại SHA-256 trên đúng các mảng vừa nhận
và so với băm client khai; khác nhau thì loại update với lý do `upload_hash_mismatch`; client không khai băm được ghi là chưa kiểm (`null`), không ghi là đã xác thực; băm sai định dạng (không phải 64 ký tự hex) bị loại (`tests/test_weight_exchange.py`).
Băm do client tự khai chỉ bảo đảm toàn vẹn (bắt lỗi truyền và lỗi phía client), không phải xác thực: máy chủ Flower của đề tài không dùng TLS, nên bên nào sửa được byte trên đường truyền cũng tính lại được băm.
Số byte tải lên, tải xuống nay là byte đã tuần tự hóa thật; thời gian đường truyền chưa đo được ghi `null` thay cho 0.

## 5. Tốc độ giữa hai máy

Cách đo: máy chủ Flower trên VM Oracle (`149.118.51.118`, ràng buộc cổng loopback); client 0 trên laptop kết nối qua đường hầm `ssh -L` (hai cổng Flower và cổng đo mạng), client 1 chạy ngay trên VM.
Mỗi client có 380 ảnh PlantVillage thật, 2 epoch mỗi vòng, 3 vòng, mô hình 6,3 MB. Mọi thời gian đo trên đồng hồ của máy tương ứng, không cần đồng bộ giờ giữa hai máy. Môi trường chạy là thư mục tạm riêng trên VM
và đã được xóa sau khi đo; dịch vụ web đang chạy không bị ảnh hưởng (kiểm tra HTTP 200 sau đó). Nguồn: `artifacts/network_speedtest/*/summary.json`.

| Đại lượng (trung vị 3 vòng) | Loopback, một laptop (mốc) | Laptop ↔ VM Oracle qua Internet |
|---|---|---|
| Độ trễ khứ hồi trên kết nối giữ sẵn | 0,12–0,17 ms | 46,7 ms |
| Độ trễ khứ hồi cũ (kết nối mới, gồm bắt tay TCP) | 2,4–25 ms (dao động theo tranh chấp CPU) | 93,4–93,9 ms (≈ 2 lần độ trễ khứ hồi) |
| Băng thông tải lên / tải xuống của laptop | 618–916 / 787–1.065 MB/s | 33,7 / 31,2 MB/s |
| Thời gian tính một vòng, laptop | 39,9 s (hai client cùng tranh CPU) | 20,9 s (18,2 ảnh/s) |
| Thời gian tính một vòng, VM | — | 5,2 s (73,7 ảnh/s) |
| Thời gian một vòng ở server | 40,7 s | 22,8 s |
| Phần truyền và xếp hàng (suy ra) | 0,37 s | 0,77 s (3,4% thời gian vòng) |
| Thời gian truyền dự đoán từ băng thông đo được | 0,02 s | 0,38 s |
| Trọng số hai chiều giữ nguyên (băm khớp, chuỗi băm liên vòng) | có | có |

Máy VM tính nhanh hơn laptop khoảng 4 lần cho cùng khối lượng. Trong một vòng, máy chậm nhất quyết định thời gian vòng; truyền trọng số chiếm phần nhỏ nhờ băng thông 31–34 MB/s cho khối 6,3 MB.
Phần chênh 0,4 s giữa thời gian truyền suy ra và thời gian dự đoán từ băng thông gồm tuần tự hóa, gRPC và việc server kiểm tra update. Với băng thông thấp hơn hoặc mô hình lớn hơn (ví dụ một backbone mạnh hơn)
phần truyền sẽ tăng tương ứng, vì lượng byte tỉ lệ với số tham số.

Hai điểm cần lưu ý khi đọc số: (1) đường hầm SSH là TCP chạy trong TCP nên độ trễ và băng thông có thể kém hơn kết nối trực tiếp; (2) hai máy đều là CPU, các tiến trình client không chia sẻ máy với tác vụ khác lúc đo
trên WAN, còn ở loopback hai client tranh nhau CPU nên thời gian tính không so sánh trực tiếp được. Phép đo trong mạng LAN chưa thực hiện vì chưa có máy thứ hai; hướng dẫn từng bước
nằm ở `docs/HUONG_DAN_DO_TOC_DO_LAN.md`, dùng cùng tập lệnh và cùng định dạng kết quả.

Một lỗi đo đạc có sẵn được sửa trong đợt này: đồng hồ vòng của máy chủ bắt đầu trước lúc chờ đủ client kết nối, nên vòng 1 bị cộng thêm 15–22 giây khởi động client và bị đọc nhầm là thời gian truyền.

## 6. Hạn chế

- Các tập PlantDoc nhỏ (269 và 172 ảnh cắt; 113 và 95 cảnh nguyên khung), khoảng tin cậy rộng; kết quả của biến thể suy luận không đủ sức để kết luận về mức tăng dưới 3–4 điểm. Phép đo mô hình nền chưa có khoảng tin cậy theo cụm.
- PlantWild được lấy từ ảnh Internet như PlantDoc; phép loại ảnh trùng chỉ dùng dHash 64 bit, so với cảnh PlantDoc nguyên khung (không so với từng lá đã cắt), và không bắt ảnh bị lật, xoay hoặc cắt xén,
  nên ảnh trùng vẫn có thể còn sót và kết quả trên PlantWild có thể cao hơn thực tế. Ảnh trùng giữa hai lớp khác nhau của PlantWild giữ lại bản đầu tiên.
- Nhãn "lá khỏe" của PlantWild là nhãn ngầm định; chỉ tầng ảnh bệnh (nhãn đã được chuyên gia nông nghiệp rà soát ở bản 2) được dùng để nêu độ chính xác.
- Thí nghiệm ở mục 4.3 làm trên lớp cuối với đặc trưng đóng băng; nó chứng minh phép gộp không gây sai số và cho thấy xu hướng client drift, nhưng không thay thế một lần huấn luyện FedAvg đầy đủ với cùng bộ tối ưu như Centralized.
- Chưa có phép đo mạng LAN; phép đo WAN dùng đường hầm SSH và một lần chạy ba vòng.
- Một hạt giống ngẫu nhiên; chưa có độ lệch giữa các lần huấn luyện.

## 7. Việc cần làm khi có máy huấn luyện

Không thực hiện trong đợt này theo yêu cầu. Danh mục, cấu hình đề xuất và ngưỡng chấp nhận đặt trước nằm ở `docs/NEXT_TRAINING_PLAN.md`; các công cụ đã sẵn sàng: tập thử ngoài đồng khóa mã băm
(`artifacts/external_eval_set.json`), bộ tăng cường ảnh (`src/data/robust_transforms.py`), tập lệnh đo lại các biến thể suy luận và tập thử ngoài đồng sau mỗi lần huấn luyện.

## Phụ lục: lệnh chạy lại

```powershell
python scripts/eval_inference_variants.py                         # mục 3.3
python scripts/build_external_eval_set.py --download-scenes       # mục 3.1, dựng tập PlantWild đã loại ảnh trùng
python scripts/eval_external_set.py                               # mục 3.1, đo bảy mô hình
tmp/venv-fm/Scripts/python.exe scripts/eval_foundation_models.py # mục 3.5, BioCLIP và DINOv2 (môi trường riêng có open_clip, transformers)
python scripts/audit_weight_exchange.py                           # mục 4.1, 4.2
python scripts/compare_model_weights.py                           # mục 4.4
python scripts/aggregation_equivalence.py                         # mục 4.3
python scripts/report_training_protocol.py                        # mục 4.5
python scripts/prepare_network_speedtest.py --out tmp/speedtest   # mục 5
python scripts/summarize_network_speedtest.py --run wan=...       # mục 5
```

## Nguồn tham khảo

- Mohanty, Hughes, Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*: https://arxiv.org/pdf/1604.03169
- Noyan (2022), *Uncovering bias in the PlantVillage dataset*: https://arxiv.org/pdf/2206.04374
- Singh và cộng sự, *PlantDoc: A Dataset for Visual Plant Disease Detection*: https://ar5iv.labs.arxiv.org/html/1911.10317
- Wei và cộng sự (2024), *Benchmarking In-the-wild Multimodal Disease Recognition and A Versatile Baseline* (PlantWild): https://arxiv.org/pdf/2408.03120
- Stevens và cộng sự (2024), *BioCLIP: A Vision Foundation Model for the Tree of Life*: https://arxiv.org/pdf/2311.18803
- ConMamba (bảng ViT, EfficientNet, ResNet50 trên PlantDoc): https://arxiv.org/pdf/2506.03213
- ConvGeM-NeXt, Frontiers in Plant Science (2026): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1763739/full
- Hai tầng YOLO và ECA-NFNet trên PlantDoc: https://vu.edu.bd/publications/2864/an-explainable-ai-based-plant-disease-identification-using-a-two-stage-detection-classification-pipeline-with-yolo-and-eca-nfnet-framework
- Thử Plantix ngoài đồng (PlantwisePlus): https://blog.plantwise.org/2020/06/19/teaching-an-app-to-help-farmers/
- Đánh giá các ứng dụng chẩn đoán bệnh cây: https://arxiv.org/pdf/2208.02446
- Luận văn học liên kết trên PlantVillage và PlantWild (Minnesota State University): https://cornerstone.lib.mnsu.edu/etds/1595
- Zhao và cộng sự (2018), dữ liệu non-IID làm lệch trọng số FedAvg: https://arxiv.org/abs/1806.00582

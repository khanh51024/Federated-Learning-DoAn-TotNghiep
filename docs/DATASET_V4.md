# Release dữ liệu v4 (`mixed_pv_pd_v4`): thêm ảnh ngoài đồng đã kiểm nhãn

Tài liệu này mô tả cách dựng release v4, nguồn và giấy phép của từng phần dữ liệu, quy tắc loại ảnh, số liệu thực đo và những giới hạn còn lại.
Chưa có lần huấn luyện nào trên v4; mọi số liệu ở đây là số liệu về dữ liệu, không phải độ chính xác của mô hình đồ án.

## 1. Vì sao cần v4

Đợt đánh giá ngày 2026-10-21 (`docs/DANH_GIA_DU_LIEU_MO_HINH_FEDAVG_2026-10-21.md`) cho thấy tập học v3 có 37.238 ảnh, trong đó khoảng 94% là ảnh PlantVillage chụp
trong phòng thí nghiệm, chỉ có 2.115 lá PlantDoc trên 14/38 lớp và không có lá khỏe chụp ngoài đồng. Cả ba cách (Local-only, Centralized, FedAvg) vì vậy đạt trên 90%
với ảnh PlantVillage nhưng chỉ còn 33,1% (Centralized) và 31,8% (FedAvg) top-1 trên ảnh bệnh ngoài đồng của PlantWild. Nút thắt nằm ở dữ liệu, không nằm ở cách huấn luyện.

## 2. Nguyên tắc dựng

1. **Khóa các tập thử trước khi thêm dữ liệu**: PlantVillage val/test, PlantDoc val/test của v3, PlantWild 5.045 ảnh (`artifacts/external_eval_set.json`, SHA-256 `5adba619…`)
   và phần thử của ảnh tự chụp. PlantWild không bao giờ dùng để học.
2. **Chỉ nhận nhãn có cơ sở**: nhãn `trusted` (do nhóm tác giả bộ dữ liệu gán, có DOI) hoặc nhóm (nguồn, lớp) đã qua kiểm nhãn mẫu.
   Quy tắc nhận nhóm đặt trước khi xem kết quả: mẫu 30 ảnh, "đúng" ≥ 90% và "không rõ" ≤ 20%. Nhóm nhỏ (nguồn tìm kiếm Wikimedia, iNaturalist nhóm ≤ 150 ảnh) kiểm **từng ảnh**.
3. **Loại ảnh gần trùng** (dHash 64 bit, Hamming ≤ 6, cả bản lật ngang) với PlantVillage, cảnh và lá PlantDoc của v3, PlantWild và phần thử ảnh tự chụp.
4. **Gộp nhóm gần trùng giữa các nguồn** (Hamming ≤ 4) vào cùng một `group_id` để chia train/val/test theo nhóm, không theo ảnh.
   Nhóm gần trùng mà mang **hai nhãn khác nhau** bị bỏ hẳn (nhãn mâu thuẫn).
5. **Chia 70/10/10/10** (train, val, calibration, legacy_diagnostic_test) theo mã băm của nhóm; PlantVillage train giới hạn 800 ảnh mỗi lớp.
6. **Ghi nguồn thật ở `source_dataset`**; `source_domain` của mọi ảnh ngoài đồng là `"plantdoc"` vì bộ chạy huấn luyện hiện chỉ biết hai miền (`plantvillage`, `plantdoc`).
7. Phân vùng 5 client chia theo nhóm: `label_alpha_0_1` (Dirichlet α = 0,1) và `iid` (chỉ để minh họa trần của Local-only).

## 3. Nguồn dữ liệu và giấy phép

Cấu hình ở `configs/external_sources.yaml`; mỗi ảnh mang giấy phép trong `manifest.jsonl` của nguồn. Ảnh nằm ngoài git (`data/` bị bỏ qua).

| Nguồn | Giấy phép | Ảnh đã nhập | Vào release (train / val / calib / test) | Kiểm nhãn |
|---|---|---:|---|---|
| [Tomato Leaf Disease Classification Dataset in Pakistan (ngoài ruộng)](https://data.mendeley.com/datasets/3mbnb82mxd/1) | CC BY 4.0 | 4697 | 2920 / 490 / 436 / 365 | nhãn của tác giả bộ dữ liệu |
| [Tomato Leaf Disease Classification Dataset (Bangladesh)](https://data.mendeley.com/datasets/jttrv2w27r) | CC BY 4.0 | 1367 | 774 / 104 / 87 / 108 | nhãn của tác giả bộ dữ liệu |
| [Corn Leaf Disease Classification Dataset (5 lớp, điều kiện ngoài ruộng)](https://data.mendeley.com/datasets/hmkd6nbngr) | CC BY 4.0 | 8737 | 4891 / 809 / 732 / 828 | mẫu 30/nhóm: nhận 3/3 nhóm |
| [Open-Source Multi-Format Maize Leaf Image Dataset (ruộng Bangladesh)](https://data.mendeley.com/datasets/sb4wwjs8bz) | CC BY 4.0 | 0 | 0 / 0 / 0 / 0 | chưa tải |
| [Potato Leaf Disease Dataset (iPhone 15, trạm nghiên cứu Chattogram)](https://data.mendeley.com/datasets/d5b3fzpw3g/1) | CC BY 4.0 | 24 | 16 / 2 / 3 / 3 | nhãn của tác giả bộ dữ liệu |
| [Potato Leaf (Healthy and Late Blight) - ruộng khoai tây Holeta, Ethiopia (máy ảnh và điện thoại)](https://data.mendeley.com/datasets/v4w72bsts5/2) | CC BY 4.0 | 429 | 288 / 39 / 54 / 41 | nhãn của tác giả bộ dữ liệu |
| [ESCA dataset (vườn nho Ý, 1.770 ảnh)](https://data.mendeley.com/datasets/89cnxc58kj) | CC BY 4.0 | 1770 | 1231 / 172 / 180 / 187 | nhãn của tác giả bộ dữ liệu |
| [Orange Leaves Images Dataset for the Detection of Huanglongbing (nền đã xóa)](https://data.mendeley.com/datasets/jgkh2jxbwt) | CC BY 4.0 | 465 | 306 / 54 / 47 / 38 | nhãn của tác giả bộ dữ liệu |
| [Plant Pathology 2021 FGVC8 (táo, vườn thật; 3 mảnh parquet đầu)](https://huggingface.co/datasets/timm/plant-pathology-2021) | Apache-2.0 (dữ liệu gốc Kaggle FGVC8, dùng cho nghiên cứu) | 3464 | 2404 / 352 / 366 / 337 | nhãn của tác giả bộ dữ liệu |
| [Auburn Soybean Disease Image Dataset (ASDID) - lá khỏe, ruộng Alabama](https://zenodo.org/records/7304859) | CC0 | 199 | 143 / 21 / 21 / 14 | nhãn của tác giả bộ dữ liệu |
| [Grape Leaf dataset for leaf Health Classification and Disease Detection (phần ảnh gốc, nho)](https://data.mendeley.com/datasets/378c69nwk5) | CC BY 4.0 | 0 | 0 / 0 / 0 / 0 | chưa tải |
| [CherryLeaf-KG (vườn anh đào Kyrgyzstan, iPhone 14, lá cắt 224x224)](https://data.mendeley.com/datasets/wp3b6pz9gc) | CC BY 4.0 | 200 | 138 / 16 / 18 / 27 | nhãn của tác giả bộ dữ liệu |
| iNaturalist (cấp research; CC0/CC BY/CC BY-NC theo từng ảnh) | theo ảnh | 1907 | 60 / 7 / 10 / 9 | kiểm từng ảnh: chỉ ảnh được đánh dấu "đúng" mới vào |
| Wikimedia Commons (giấy phép theo từng tệp) | theo ảnh | 399 | 7 / 1 / 2 / 2 | kiểm từng ảnh: chỉ ảnh được đánh dấu "đúng" mới vào |

## 4. Số liệu release

Release `mixed_pv_pd_v4`, SHA-256 của `release_manifest.json`: `cc5356bd2e867678c043c5df9ae18aa35dcf76d361402c89e8f8796e6db585c8`.

| Tập | Ảnh |
|---|---:|
| train | 39232 |
| validation | 6680 |
| calibration | 6220 |
| legacy_diagnostic_test | 12974 |
| cách ly (quarantine) | 5505 |

- Ảnh ngoài đồng thêm vào: 19160; lá PlantDoc cách ly được nhận lại: 592; ảnh tự chụp: 0.
- PlantVillage train: 35123 → giới hạn 800 ảnh mỗi lớp.
- Loại 2010 ảnh gần trùng với các tập đã khóa/PlantVillage; bỏ 47 ảnh thuộc 12 nhóm gần trùng có nhãn mâu thuẫn.
- Kiểm rò rỉ giữa bốn tập: 0 giao nhau theo `group_id`, `parent_scene_id`, `byte_sha256`, `sample_id`, `image_id`.
- Phân vùng `label_alpha_0_1`: số ảnh mỗi client [5494, 9616, 10638, 8113, 5371], rời nhau và hợp bằng đúng tập train, 0 nhóm bị chia cho nhiều client.
- Phân vùng `iid`: số ảnh mỗi client [7866, 7874, 7837, 7848, 7807], rời nhau và hợp bằng đúng tập train, 0 nhóm bị chia cho nhiều client.

Số ảnh ngoài đồng đã kiểm nhãn (dùng để học) theo lớp, từ `artifacts/data_gap_report.json`. Số này đếm **trước** khi loại ảnh gần trùng với tập đã khóa và trước khi chia split,
nên cao hơn số ảnh thực vào tập học (bảng mục 3); ảnh tăng cường ngoại tuyến của tác giả cũng được đếm từng ảnh:

| Lớp | PlantVillage train | Ngoài đồng sẵn sàng | Số nguồn | Trạng thái |
|---|---:|---:|---:|---|
| Apple___Apple_scab | 404 | 1275 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Apple___Black_rot | 378 | 773 | 1 | đủ tối thiểu (≥ 100) |
| Apple___Cedar_apple_rust | 175 | 435 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Apple___healthy | 1079 | 1128 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Blueberry___healthy | 983 | 0 | 0 | thiếu (< 100) |
| Cherry_(including_sour)___Powdery_mildew | 618 | 0 | 0 | thiếu (< 100) |
| Cherry_(including_sour)___healthy | 582 | 200 | 1 | đủ tối thiểu (≥ 100) |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 332 | 2426 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Corn_(maize)___Common_rust_ | 769 | 3120 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Corn_(maize)___Northern_Leaf_Blight | 640 | 3338 | 1 | đủ tối thiểu (≥ 100) |
| Corn_(maize)___healthy | 756 | 0 | 0 | thiếu (< 100) |
| Grape___Black_rot | 772 | 93 | 2 | thiếu (< 100) |
| Grape___Esca_(Black_Measles) | 896 | 888 | 1 | đủ tối thiểu (≥ 100) |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 684 | 0 | 0 | thiếu (< 100) |
| Grape___healthy | 274 | 1098 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Orange___Haunglongbing_(Citrus_greening) | 3595 | 465 | 1 | đủ tối thiểu (≥ 100) |
| Peach___Bacterial_spot | 1510 | 0 | 0 | thiếu (< 100) |
| Peach___healthy | 228 | 0 | 0 | thiếu (< 100) |
| Pepper,_bell___Bacterial_spot | 626 | 0 | 0 | thiếu (< 100) |
| Pepper,_bell___healthy | 921 | 0 | 0 | thiếu (< 100) |
| Potato___Early_blight | 644 | 232 | 1 | đủ tối thiểu (≥ 100) |
| Potato___Late_blight | 632 | 246 | 3 | đủ tối thiểu (≥ 100) |
| Potato___healthy | 108 | 366 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Raspberry___healthy | 236 | 35 | 1 | thiếu (< 100) |
| Soybean___healthy | 3295 | 199 | 1 | đủ tối thiểu (≥ 100) |
| Squash___Powdery_mildew | 1176 | 189 | 1 | đủ tối thiểu (≥ 100) |
| Strawberry___Leaf_scorch | 708 | 0 | 0 | thiếu (< 100) |
| Strawberry___healthy | 312 | 492 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Tomato___Bacterial_spot | 1379 | 335 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Tomato___Early_blight | 660 | 1199 | 3 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Tomato___Late_blight | 1221 | 800 | 3 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Tomato___Leaf_Mold | 608 | 1310 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Tomato___Septoria_leaf_spot | 1144 | 1040 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |
| Tomato___Spider_mites Two-spotted_spider_mite | 1096 | 2 | 1 | thiếu (< 100) |
| Tomato___Target_Spot | 911 | 0 | 0 | thiếu (< 100) |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 3468 | 639 | 1 | đủ tối thiểu (≥ 100) |
| Tomato___Tomato_mosaic_virus | 240 | 153 | 1 | đủ tối thiểu (≥ 100) |
| Tomato___healthy | 1063 | 1557 | 2 | đạt mục tiêu (≥ 300, ≥ 2 nguồn) |

Tóm tắt: 14 lớp đạt mục tiêu, 11 lớp đủ tối thiểu, 13 lớp thiếu (tổng 24033 ảnh ngoài đồng sẵn sàng).

### Đo giá trị dữ liệu khi chưa train

`scripts/eval_gallery_value.py` đo kNN (k = 10, top-1 %) trên đặc trưng DINOv2-small đóng băng, không huấn luyện lớp nào; `scripts/ablate_gallery_value.py` tách tác động từng phần trên đặc trưng đã lưu
(`artifacts/dataset_v4_value.json`, `artifacts/dataset_v4_value_ablation.json`). Số đo ứng với bản dựng có SHA-256 `3b592c44ab0b528bba1308156f0aaa4ed5954371bd3f238b619b713569a491ac`, **trước khi** thêm CherryLeaf-KG (200 ảnh lá anh đào khỏe, không nằm trong phép đo).

| Tập đo | n | Thư viện v3 | v3 + ảnh mới | v4 (như đã đo) |
|---|---:|---:|---:|---:|
| PlantWild bệnh (nguồn lạ) | 2304 | 50,3 | 47,1 | 39,4 |
| PlantWild lá khỏe (nguồn lạ, nhãn ngầm định) | 2741 | 19,1 | 26,9 | 31,7 |
| PlantDoc kiểm tra (nguồn lạ) | 269 | 55,4 | 52,8 | 43,1 |
| PlantDoc giữ riêng (nguồn lạ) | 172 | 53,5 | 48,8 | 38,4 |
| Ngoài đồng cùng nguồn (v4_field_test) | 1932 | 44,6 | 82,1 | 81,4 |

- "v3" = PlantVillage 40 ảnh mỗi lớp + mọi lá PlantDoc của tập học (2.115). "v3 + ảnh mới" = v3 giữ nguyên rồi thêm ảnh ngoài đồng mới (mẫu tối đa 120 ảnh mỗi lớp, 2086 ảnh).
  "v4 (như đã đo)" = PlantVillage 40 + mẫu ngoài đồng tối đa 120 ảnh mỗi lớp (gồm cả PlantDoc), nên chỉ còn 906 lá PlantDoc.
- **Mức giảm ở "v4 (như đã đo)" phần lớn do trần 120 ảnh mỗi lớp làm mất lá PlantDoc (2.115 → 906), không phải do nguồn mới**: giữ đủ lá PlantDoc rồi thêm ảnh mới thì PlantWild bệnh chỉ còn 47,1% (v3: 50,3%).
- Ảnh mới **giúp** nhận cùng nguồn (44,6% → 82,1%) và nhận lá khỏe (19,1% → 26,9%; lá khỏe PlantWild là điểm yếu lớn nhất ở báo cáo 2026-10-21).
- Ảnh mới **chưa giúp** nhận bệnh ở nguồn lạ trong phép đo này: PlantWild bệnh giảm 3,2 điểm và PlantDoc giảm 2,6–4,7 điểm. Riêng Plant Pathology 2021 đã kéo PlantWild bệnh xuống 46,7%, tức cả mức giảm có thể quy cho nguồn này. Giải thích khả dĩ, chưa kiểm chứng: 465 ảnh táo đồng nhất lấn át các lớp táo trong kNN
  (kNN trên đặc trưng đóng băng ưu tiên láng giềng cùng nguồn). Chưa kết luận được cho mô hình huấn luyện có cân bằng theo nguồn; chỉ phép train mới trả lời.
- Số ngoài đồng cùng nguồn (82%) **không** chứng minh được mục tiêu > 90% cho ảnh bất kỳ: tập đó cùng nguồn và cùng bộ ảnh tăng cường với thư viện.
- Bài học cho lần train: không giới hạn lá PlantDoc, cân bằng theo nguồn (trọng số mất mát hoặc lấy mẫu), và báo riêng nguồn lạ.
- Phân tích theo từng nguồn là chẩn đoán trên tập thử đã khóa, **không** dùng để chọn nguồn (chọn theo PlantWild là học trên tập thử).
- Lần chạy đo thứ hai bị hệ thống dừng vì thiếu bộ nhớ; kết quả lấy từ lần chạy đầu đã hoàn tất. Chưa đo lại trên bản dựng có CherryLeaf-KG.

## 5. Vì sao không dùng một số nguồn

| Nguồn | Lý do không dùng |
|---|---|
| PlantSeg / PlantWild (cùng nhóm tác giả, ảnh ngoài đồng từ web) | PlantWild đã khóa làm tập thử ngoài đồng; số liệu công bố của hai bộ gần như trùng nhau (PlantSeg 11.458 ảnh, PlantWild v2 11.488 ảnh, cùng 115 lớp bệnh; chưa đối chiếu từng ảnh), nên dùng PlantSeg để học có nguy cơ làm hỏng phép đo. Giấy phép CC BY-NC-ND 4.0 (không cho phép biến đổi) cũng không hợp với việc cắt, đổi kích thước và nối vào release. |
| Pepper Bell Leaf Disease (Mendeley `w9drvtxg9g`) | Đã xem ảnh mẫu: lá rời trên nền xám kiểu phòng thí nghiệm, một phần là ảnh PlantVillage; không thêm đa dạng ngoài đồng. |
| Tomato leaf diseases (Mendeley `93h9p62kg4`) | 2.627 ảnh ngoài đồng nhưng nằm chung một thư mục, không có nhãn theo tệp; không dùng được cho học có giám sát. |
| Strawberry Disease Detection (Mendeley `k2ptxmfjhj`) | Nhãn nằm trong tệp COCO (nhiều khung mỗi ảnh), không có lớp Leaf scorch, chủ yếu ảnh tăng cường và ảnh quả; không phục vụ lớp nào trong 38 lớp. |
| Tomato plant spider mite damage (WUR, 4TU) | Ảnh chụp ban đêm trong nhà kính; xem 30 ảnh mẫu thì 0/30 thấy được tổn thương nhện đỏ. |
| Wikimedia Commons: truy vấn Orange HLB, Pepper bacterial spot, Tomato bacterial spot | Kiểm từng ảnh: toàn tàu hỏa, máy bay, logo công ty "HLB", ảnh chó nghiệp vụ dò bệnh loét cam quýt; 0 ảnh lá dùng được. Cơ chế kiểm từng ảnh đã chặn các ảnh này, không vào release. |
| iNaturalist: nhóm lá khỏe của Blueberry, Cherry, Peach, Pepper, Soybean, Corn | Mẫu 30 ảnh không đạt quy tắc nhận nhóm: phần lớn là cả bụi cây, hoa, quả, thân cây hoặc cành trụi mùa đông; nhãn "khỏe" chỉ có nghĩa là người đăng không ghi bệnh. Lá ngô khỏe chỉ đạt 86,7% "đúng" (ngưỡng 90%) nên cũng loại, không nới ngưỡng. |
| Tomato Yellow Leaf Curl (Mendeley Bangladesh) | Mẫu 30 ảnh: 12 ảnh "không rõ" (> 20%), loại cả nhóm. |
| Mendeley Grape Leaf 18k (`378c69nwk5`) | Đã khai báo trong cấu hình (nho: Healthy, Black Rot; 3 GB) nhưng chưa tải xong: tác vụ tải bị hệ thống dừng vì thiếu bộ nhớ, nên chưa có ảnh nào của nguồn này trong release. |

## 6. Giới hạn và mức khả thi của mục tiêu > 90%

Bảng nói trước những gì khả thi, để không hứa quá mức. Căn cứ: trần Local-only tính từ `artifacts/client_shard_class_counts.json`; các số tham khảo từ tài liệu ở `docs/REFERENCES_IEEE.md` và báo cáo ngày 2026-10-21.

| | PlantVillage | Ảnh ngoài đồng **cùng nguồn** (v4 field test, ảnh tự chụp phần thử) | Ảnh web/bất kỳ **nguồn chưa từng thấy** (PlantWild) |
|---|---|---|---|
| Centralized | > 90%: đã đạt (99,2%) | > 90% khả thi nếu mỗi lớp có ≥ 100–300 ảnh ngoài đồng từ ≥ 2 nguồn | > 90% top-1: **không thực tế**. Mục tiêu 65–75% top-1; > 90% chỉ khi có điều kiện kèm (top-3, chọn loại cây, hoặc chỉ tính ảnh mô hình tự tin) |
| FedAvg | > 90% khả thi nếu cấu hình công bằng (93,3% hiện có, run Kaggle cũ 99,09%) | thấp hơn Centralized vài điểm khi dữ liệu lệch nhãn (α = 0,1) | như Centralized, trừ vài điểm |
| Local-only | **không thể** với cách chia α = 0,1: trần 65–80% | **không thể**: trần 47–96% tùy client | **không thể**: trần 50–79% |

- Mỗi client Local-only chỉ thấy 20–27/38 lớp nên không thể đoán đúng lớp chưa từng học; đây là điều thí nghiệm muốn chứng minh. Muốn Local-only cao thì phải chia IID (phân vùng `iid` có sẵn trong release để minh họa), khi đó bài toán non-IID mất ý nghĩa.
- Bằng chứng cho cột ảnh nguồn chưa từng thấy: nghiên cứu 221 nghìn ảnh cho khoảng 99% khi cùng ruộng và 64–88% khi ruộng mới; baseline tốt nhất đã công bố trên PlantWild khoảng 61%.
- **Chưa đo được độ chính xác của mô hình trên v4** (chưa train). Phép đo không cần train (kNN trên đặc trưng DINOv2-small đóng băng, thư viện v3 so với v4) ghi ở mục 4.
- **Nhãn kiểm bằng AI không phải nhãn chuyên gia.** Mọi quyết định nhận/loại nhóm trong `data/label_review_ext/decisions_ai.json` mang `reviewer = ai_screening` (xem ảnh lưới bằng mắt). Quy tắc `apply_label_review.py` cho phép tệp quyết định của người đứng sau để ghi đè; nên để người có chuyên môn xem lại ít nhất các nhóm đã nhận.
- Nhãn ánh xạ gần đúng: "frog eye leaf spot" (Plant Pathology 2021) → `Apple___Black_rot` (cùng tác nhân *Botryosphaeria obtusa*) là alias, không phải khớp tên.
- Lớp còn thiếu ảnh ngoài đồng sau các vòng tìm (iNaturalist, Wikimedia, bộ dữ liệu công khai): xem bảng ở mục 4; cần ảnh tự chụp (`docs/HUONG_DAN_CHUP_ANH.md`, kiểm bằng `scripts/check_own_photos.py`). Báo cáo độ chính xác sau này phải báo riêng các lớp dưới 100 ảnh, không gộp vào trung bình.
- Ảnh cà chua Pakistan và ngô Mendeley phần lớn là ảnh tăng cường ngoại tuyến của tác giả (gom theo ảnh gốc để chia train/test); số ảnh nhiều nhưng số cảnh độc lập ít hơn nhiều, nên `field_ready` của các lớp này đánh giá cao hơn đa dạng thật.
- CherryLeaf-KG: 100 ảnh gốc, mỗi ảnh có hai bản cắt trong hai thư mục (200 ảnh, 100 nhóm).

## 7. Cách chạy lại

```
python scripts/fetch_external_sources.py --source <tên nguồn>        # tải và giải nén
python scripts/fetch_inaturalist.py ; python scripts/fetch_wikimedia.py   # nguồn tìm kiếm có giấy phép từng ảnh
python scripts/import_external_source.py --source <tên nguồn>        # chuẩn hóa EXIF/RGB, dHash, ánh xạ nhãn
python scripts/build_label_review.py --external --per-class 30       # sinh trang kiểm nhãn + mẫu
python scripts/make_review_sheets.py                                 # ảnh lưới để xem nhanh
python scripts/record_review_decisions.py --spec ... --reviewer ...  # đổi kết quả xem thành quyết định
python scripts/apply_label_review.py --samples data/label_review_ext/samples.json --decisions ...
python scripts/build_release_v4.py --pv-cap 800 --seed 20261022      # dựng release + kiểm rò rỉ
python scripts/report_data_gaps.py                                   # lớp nào còn thiếu ảnh
tmp/venv-fm/Scripts/python.exe scripts/eval_gallery_value.py         # đo giá trị dữ liệu bằng kNN, không train
```

Khi huấn luyện trên v4 phải truyền `--expected-release-sha <release_manifest_sha256 trong artifacts/release_v4_report.json>`, vì bộ chạy chỉ ghim mã băm v1–v3.
Bộ tăng cường ảnh mạnh (`src/data/robust_transforms.py`) chưa nằm trong danh sách preset của bộ chạy; lúc train cần thêm preset vào `plant_data_contract/transforms.py`.

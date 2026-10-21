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
4. **Gộp nhóm để chia train/val/test theo nhóm, không theo ảnh**:
   - ảnh gần trùng giữa các nguồn (Hamming ≤ 4) vào cùng một `group_id`;
   - ảnh CÙNG nguồn và CÙNG lớp gần nhau (Hamming ≤ 10, tính cả bản lật ngang) vào cùng nhóm;
   - ảnh cùng nguồn chụp liên tiếp trong 60 giây (mốc giờ trong tên tệp như `20250404_154210`, mỗi phiên tối đa 10 phút) vào cùng nhóm, để các lần chụp liên tiếp của một lá
     không rơi vào cả train lẫn test (phát hiện khi rà soát: trước đó 8% ảnh thử của bộ ngô, 34% của cà chua Bangladesh có ảnh học gần trùng trong 10 bit);
   - ảnh đã xóa nền (bộ HLB) có dHash chỉ phản ánh nền đen nên không gộp theo dHash, chỉ gộp theo phiên chụp;
   - hai ảnh gần trùng (Hamming ≤ 4) mang **hai nhãn khác nhau**: ít nhất một nhãn sai, bỏ **cả hai ảnh đó** (không bỏ cả nhóm), không bao giờ gộp hai lớp vào một nhóm.
   Cổng rò rỉ: tỉ lệ ảnh có ảnh cùng nguồn, cùng lớp, trong 10 bit ở split khác phải ≤ 2% (kết quả: 0).
5. **Chia 70/10/10/10** (train, val, calibration, legacy_diagnostic_test) theo mã băm của nhóm. PlantVillage train giữ đủ (không giới hạn, `--pv-cap 0`); cân bằng giữa
   ảnh phòng thí nghiệm và ảnh ngoài đồng làm lúc train bằng trọng số lấy mẫu (xem `docs/NEXT_TRAINING_PLAN.md` mục 5.2).
6. **Ghi nguồn thật ở `source_dataset`**; `source_domain` của mọi ảnh ngoài đồng là `"plantdoc"` vì bộ chạy huấn luyện hiện chỉ biết hai miền (`plantvillage`, `plantdoc`).
7. Phân vùng 5 client chia theo nhóm: `label_alpha_0_1` (Dirichlet α = 0,1) và `iid` (chỉ để minh họa trần của Local-only).
8. **Đúng hợp đồng của bộ chạy huấn luyện**: `partition_shard_hashes` khóa theo tên tệp (`client_00.jsonl`), có `partition_spec_v4.json` (trạng thái PASS, SHA từng shard, thống kê
   entropy và JS theo bit như v3) và `quality_audit_v4.json` (cổng rò rỉ trước khi train đầy đủ). `scripts/verify_release_contract.py` chạy các bộ kiểm của nhánh `origin/train-centralized`
   trên release và hiện đạt 11/11. Lần dựng đầu thiếu hai tệp trên và khóa shard sai, nên bộ chạy sẽ từ chối release; lỗi này phát hiện khi rà soát và đã sửa.

## 3. Nguồn dữ liệu và giấy phép

Cấu hình ở `configs/external_sources.yaml`; mỗi ảnh mang giấy phép trong `manifest.jsonl` của nguồn. Ảnh nằm ngoài git (`data/` bị bỏ qua).

| Nguồn | Giấy phép | Ảnh đã nhập | Vào release (train / val / calib / test) | Kiểm nhãn |
|---|---|---:|---|---|
| [Tomato Leaf Disease Classification Dataset in Pakistan (ngoài ruộng)](https://data.mendeley.com/datasets/3mbnb82mxd/1) | CC BY 4.0 | 4697 | 2949 / 458 / 395 / 417 | nhãn của tác giả bộ dữ liệu |
| [Tomato Leaf Disease Classification Dataset (Bangladesh)](https://data.mendeley.com/datasets/jttrv2w27r) | CC BY 4.0 | 1367 | 870 / 73 / 60 / 77 | nhãn của tác giả bộ dữ liệu |
| [Corn Leaf Disease Classification Dataset (5 lớp, điều kiện ngoài ruộng)](https://data.mendeley.com/datasets/hmkd6nbngr) | CC BY 4.0 | 8737 | 4619 / 748 / 1176 / 717 | mẫu 30/nhóm: nhận 3/3 nhóm |
| [Open-Source Multi-Format Maize Leaf Image Dataset (ruộng Bangladesh)](https://data.mendeley.com/datasets/sb4wwjs8bz) | CC BY 4.0 | 0 | 0 / 0 / 0 / 0 | chưa tải |
| [Potato Leaf Disease Dataset (iPhone 15, trạm nghiên cứu Chattogram)](https://data.mendeley.com/datasets/d5b3fzpw3g/1) | CC BY 4.0 | 24 | 18 / 2 / 3 / 1 | nhãn của tác giả bộ dữ liệu |
| [Potato Leaf (Healthy and Late Blight) - ruộng khoai tây Holeta, Ethiopia (máy ảnh và điện thoại)](https://data.mendeley.com/datasets/v4w72bsts5/2) | CC BY 4.0 | 429 | 282 / 38 / 58 / 44 | nhãn của tác giả bộ dữ liệu |
| [ESCA dataset (vườn nho Ý, 1.770 ảnh)](https://data.mendeley.com/datasets/89cnxc58kj) | CC BY 4.0 | 1770 | 1230 / 173 / 180 / 187 | nhãn của tác giả bộ dữ liệu |
| [Orange Leaves Images Dataset for the Detection of Huanglongbing (nền đã xóa)](https://data.mendeley.com/datasets/jgkh2jxbwt) | CC BY 4.0 | 465 | 268 / 14 / 100 / 63 | nhãn của tác giả bộ dữ liệu |
| [Plant Pathology 2021 FGVC8 (táo, vườn thật; 3 mảnh parquet đầu)](https://huggingface.co/datasets/timm/plant-pathology-2021) | Apache-2.0 (dữ liệu gốc Kaggle FGVC8, dùng cho nghiên cứu) | 3464 | 2404 / 352 / 366 / 337 | nhãn của tác giả bộ dữ liệu |
| [Auburn Soybean Disease Image Dataset (ASDID) - lá khỏe, ruộng Alabama](https://zenodo.org/records/7304859) | CC0 | 199 | 143 / 21 / 21 / 14 | nhãn của tác giả bộ dữ liệu |
| [Grape Leaf dataset for leaf Health Classification and Disease Detection (phần ảnh gốc, nho)](https://data.mendeley.com/datasets/378c69nwk5) | CC BY 4.0 | 0 | 0 / 0 / 0 / 0 | chưa tải |
| [CherryLeaf-KG (vườn anh đào Kyrgyzstan, iPhone 14, lá cắt 224x224)](https://data.mendeley.com/datasets/wp3b6pz9gc) | CC BY 4.0 | 200 | 138 / 16 / 18 / 27 | nhãn của tác giả bộ dữ liệu |
| iNaturalist (cấp research; CC0/CC BY/CC BY-NC theo từng ảnh) | theo ảnh | 1907 | 60 / 7 / 10 / 9 | kiểm từng ảnh: chỉ ảnh được đánh dấu "đúng" mới vào |
| Wikimedia Commons (giấy phép theo từng tệp) | theo ảnh | 399 | 7 / 1 / 2 / 2 | kiểm từng ảnh: chỉ ảnh được đánh dấu "đúng" mới vào |

## 4. Số liệu release

Release `mixed_pv_pd_v4`, SHA-256 của `release_manifest.json`: `7c90d9661d1ebecb0dabe3d8035ce3ed0483218e910f8146e441eb7ab874f729`.

| Tập | Ảnh |
|---|---:|
| train | 50779 |
| validation | 6516 |
| calibration | 6653 |
| legacy_diagnostic_test | 12910 |
| cách ly (quarantine) | 5505 |

- Ảnh ngoài đồng thêm vào: 19175; lá PlantDoc cách ly được nhận lại: 592; ảnh tự chụp: 0.
- PlantVillage train giữ đủ 35123 ảnh (không giới hạn; cân bằng miền làm lúc train).
- Loại 2010 ảnh gần trùng với các tập đã khóa/PlantVillage; bỏ 32 ảnh có ảnh gần trùng (Hamming ≤ 4) mang nhãn khác.
- Nhóm ngoài đồng: 9425 nhóm cho 19175 ảnh (nhóm lớn nhất 461 ảnh); 636 cặp ảnh chụp liên tiếp được gộp.
- Rò rỉ cùng nguồn và cùng lớp (cổng ≤ 2%): ảnh có ảnh gần trùng ở split khác ≤ 10 bit: 0; chỉ báo cáo, không gộp: ≤ 14 bit: 4277 (22,3% ảnh ngoài đồng).
- Kiểm rò rỉ giữa bốn tập: 0 giao nhau theo `group_id`, `parent_scene_id`, `byte_sha256`, `sample_id`, `image_id`.
- `quality_audit_v4.json`: PASS (băm lại mọi tệp ảnh, 0 lệch byte; 0 ảnh trùng byte trong cùng split), SHA-256 `9b23575e6f95aa1764253ee8bc8f2102bad2c550304bec058b3a465afb6c8e68`; `partition_spec_v4.json`: label_alpha_0_1 PASS, iid PASS, SHA-256 `58fed86412e5c906f62e4a253c8f6cf7efd488f1ddbe371ef2fc73e25c3ae8fa`.
- `scripts/verify_release_contract.py` chạy các bộ kiểm của nhánh huấn luyện trên release này: 11/11 qua (`artifacts/release_v4_contract_check.json`).
- Phân vùng `label_alpha_0_1`: số ảnh mỗi client [6286, 13131, 9519, 11087, 10756], rời nhau và hợp bằng đúng tập train, 0 nhóm bị chia cho nhiều client.
- Phân vùng `iid`: số ảnh mỗi client [10188, 10168, 10161, 10116, 10146], rời nhau và hợp bằng đúng tập train, 0 nhóm bị chia cho nhiều client.

Số ảnh ngoài đồng đã kiểm nhãn (dùng để học) theo lớp, từ `artifacts/data_gap_report.json`:

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
(`artifacts/dataset_v4_value.json`, `artifacts/dataset_v4_value_ablation.json`). Số đo ứng với bản dựng có SHA-256 `7c90d9661d1ebecb0dabe3d8035ce3ed0483218e910f8146e441eb7ab874f729` (hai tệp kết quả đều ghi SHA này).

| Tập đo | n | Thư viện v3 | v3 + ảnh mới | v4 (như đã đo) |
|---|---:|---:|---:|---:|
| PlantWild bệnh (nguồn lạ) | 2304 | 50,3 | 46,9 | 40,4 |
| PlantWild lá khỏe (nguồn lạ, nhãn ngầm định) | 2741 | 19,1 | 30,2 | 35,1 |
| PlantDoc kiểm tra (nguồn lạ) | 269 | 55,4 | 52,4 | 43,5 |
| PlantDoc giữ riêng (nguồn lạ) | 172 | 53,5 | 48,8 | 40,7 |
| Ngoài đồng cùng nguồn (v4_field_test) | 1895 | 41,1 | 80,0 | 80,6 |

- "v3" = PlantVillage 40 ảnh mỗi lớp + mọi lá PlantDoc của tập học (2.115). "v3 + ảnh mới" = v3 giữ nguyên rồi thêm ảnh ngoài đồng mới (mẫu tối đa 120 ảnh mỗi lớp, 2207 ảnh).
  "v4 (như đã đo)" = PlantVillage 40 + mẫu ngoài đồng tối đa 120 ảnh mỗi lớp (gồm cả PlantDoc), nên chỉ còn 905 lá PlantDoc.
- **Mức giảm ở "v4 (như đã đo)" phần lớn do trần 120 ảnh mỗi lớp làm mất lá PlantDoc (2.115 → 905), không phải do nguồn mới**: giữ đủ lá PlantDoc rồi thêm ảnh mới thì PlantWild bệnh chỉ còn 46,9% (v3: 50,3%).
- Ảnh mới **giúp** nhận cùng nguồn (41,1% → 80,0%) và nhận lá khỏe (19,1% → 30,2%; lá khỏe PlantWild là điểm yếu lớn nhất ở báo cáo 2026-10-21).
  Riêng 120 ảnh lá anh đào khỏe của CherryLeaf-KG đã nâng lá khỏe PlantWild từ 19,1% lên 23,6%.
- Ảnh mới **chưa giúp** nhận bệnh ở nguồn lạ trong phép đo này: PlantWild bệnh giảm 3,4 điểm và PlantDoc giảm 3,0–4,7 điểm.
  Riêng Plant Pathology 2021 đã kéo PlantWild bệnh xuống 46,7%, tức gần như cả mức giảm có thể quy cho nguồn này. Giải thích khả dĩ, chưa kiểm chứng: 465 ảnh táo đồng nhất lấn át các lớp táo trong kNN
  (kNN trên đặc trưng đóng băng ưu tiên láng giềng cùng nguồn). Chưa kết luận được cho mô hình huấn luyện có cân bằng theo nguồn; chỉ phép train mới trả lời.
- Tập "cùng nguồn" (1895 ảnh) đã loại rò rỉ chụp liên tiếp (nhóm theo phiên chụp và theo ảnh gần trùng cùng nguồn): lần dựng đầu (chưa gộp theo phiên chụp) cho 82,1% trên một tập đo khác, chênh khoảng 2 điểm; chưa quy riêng được mức chênh đó cho rò rỉ.
  Dù vậy 80% **không** chứng minh được mục tiêu > 90% cho ảnh bất kỳ: đây là kNN, và tập đo vẫn cùng nguồn, cùng bộ ảnh tăng cường với thư viện.
- Bài học cho lần train: không giới hạn lá PlantDoc, cân bằng theo nguồn (trọng số mất mát hoặc lấy mẫu), và báo riêng nguồn lạ.
- Phân tích theo từng nguồn là chẩn đoán trên tập thử đã khóa, **không** dùng để chọn nguồn (chọn theo PlantWild là học trên tập thử).

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
| Local-only | **không thể** với cách chia α = 0,1 | **không thể** | **không thể** |

Trần cụ thể của từng client tính từ phân vùng v4 `label_alpha_0_1` (mỗi client chỉ thấy 19–26/38 lớp) nằm ở `artifacts/data_gap_report.json`, mục `local_only_ceiling_top1_percent`.

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
python scripts/build_release_v4.py --seed 20261022                   # dựng release, kiểm rò rỉ, ghi partition_spec_v4.json và quality_audit_v4.json
python scripts/verify_release_contract.py                            # chạy các bộ kiểm của nhánh huấn luyện trên release (thêm --images: kiểm cả byte và giải mã)
python scripts/report_data_gaps.py                                   # lớp nào còn thiếu ảnh, trần Local-only
tmp/venv-fm/Scripts/python.exe scripts/eval_gallery_value.py         # đo giá trị dữ liệu bằng kNN, không train
```

### Việc cần làm ở nhánh huấn luyện trước khi train trên v4 (phát hiện khi rà soát, chưa sửa vì nằm ngoài repo này)

1. Truyền `--expected-release-sha <release_manifest_sha256 trong artifacts/release_v4_report.json>`; bộ chạy chỉ ghim mã băm v1–v3.
2. `training-workflows/shared/data_selection.py` gắn cứng `RELEASE_NAME = "pv_pd_v3"` và đường dẫn `mixed/pv_pd_v3`; cần tham số hóa. `scripts/verify_release_contract.py` đặt tên này trong bộ nhớ để kiểm tra phần còn lại.
3. Ghim `quality_audit_v4.json` (tên tệp và SHA-256 trong `artifacts/release_v4_report.json`) vào spec huấn luyện, mục `quality_gate`.
4. Bộ chạy chọn checkpoint theo `plantdoc_macro_f1_supported` của miền `plantdoc`. Ở v4 mọi ảnh ngoài đồng mang miền `plantdoc`, nên chỉ số này không còn chỉ đo PlantDoc mà trộn mọi nguồn ngoài đồng (nhiều ảnh là bản tăng cường của tác giả), và tiêu chí chọn đổi nghĩa so với v3.
   Cần tính chỉ số theo `source_dataset` và chọn theo macro-F1 cân bằng giữa các nguồn.
5. Thêm preset tăng cường mạnh (`src/data/robust_transforms.py`) vào `plant_data_contract/transforms.py`, và lấy mẫu cân bằng giữa ảnh phòng thí nghiệm với ảnh ngoài đồng.

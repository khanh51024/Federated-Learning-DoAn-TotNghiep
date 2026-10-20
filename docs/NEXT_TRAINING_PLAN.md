# Kế hoạch tiếp theo: ảnh ngoài đời, bước cắt lá, train lại

Tài liệu này ghi lại (1) số đã đo về ảnh nguyên khung, (2) cách chạy detector lá trên máy có GPU, (3) cách train lại cả
ba cách (Centralized / FedAvg / Local-only) nếu muốn. Mọi số trong mục 1 sinh bởi
`scripts/eval_fullframe_plantdoc.py` → `artifacts/fullframe_eval.json`; không có số nào được viết tay.

## 1. Đã đo: cả ảnh so với ảnh đã cắt lá (PlantDoc, cùng 7 checkpoint đã công bố)

Đo trên các khung PlantDoc thuộc **tập kiểm tra** (113 cảnh) và **tập giữ riêng** (95 cảnh); các cảnh này không nằm trong
train. Ảnh gốc tải từ kho PlantDoc công khai và khớp `byte_sha256` của release (208/211 cảnh; 3 cảnh không còn ở kho nên bị
loại, ghi trong `excluded_unavailable`).

| Cách đưa ảnh vào | Centralized (val / giữ riêng) | FedAvg (val / giữ riêng) |
|---|---|---|
| Ảnh đã cắt có sẵn trong release (số công bố) | 55,1% / 63,2% | 48,3% / 49,7% |
| Tự cắt lại theo khung người gán + lề 8% | 54,8% / 62,0% | 49,0% / 50,3% |
| **Cả ảnh nguyên khung** | 45,2% / 55,6% | **29,7% / 30,4%** |
| Hình vuông ở giữa ảnh | 40,3% / 57,3% | 33,5% / 41,5% |

Đọc kết quả:
- Cắt lại từ ảnh gốc cho ra số gần số công bố (kích thước crop trùng ở mọi khung; 89–97% dự đoán trùng giữa ảnh tự cắt
  và ảnh đã công bố, tùy mô hình và tập). Phần dự đoán còn lại khác nhau chưa rõ nguyên nhân (chưa kiểm chứng
  chuyện nén/resize lúc dựng release), nên khoảng ±1–2 điểm giữa hai dòng đầu là nhiễu, không phải khác biệt thật.
- Đưa cả ảnh vào làm **FedAvg tụt ~19–20 điểm**, Centralized tụt ~6–10 điểm (so với dòng "tự cắt lại"). Một bước cắt lá tốt có thể lấy lại phần này
  (giới hạn trên chính là dòng "tự cắt lại").
- Chỉ ~100 cảnh mỗi tập nên khoảng tin cậy rộng; chưa phải độ chính xác ngoài thực địa.

Vì sao: ảnh học là MỘT lá đã cắt gọn (PlantVillage: một lá giữa khung; PlantDoc: cắt theo khung + lề 8%), và transform
`canonical_v1` ép ảnh về 224×224 bằng cách resize thẳng (không giữ tỉ lệ), chỉ có lật ngang khi học.

## 2. Detector lá (đặt ở bước 4, trước khi chẩn đoán)

Dữ liệu có sẵn: `data/dataset_sources/mixed/pv_pd_v3/annotations/detection.json` (2.556 ảnh PlantDoc, 8.908 khung lá,
trung vị một lá chiếm 6,7% diện tích ảnh). Chỉ ảnh `split == "train"` (1.763 ảnh) được dùng để học; các cảnh dùng để đo
bộ phân loại bị loại nên không rò rỉ.

```bash
python scripts/train_leaf_detector.py --download --epochs 12 --device cuda --pretrained-backbone
python scripts/eval_fullframe_plantdoc.py --detector data/leaf_detector/leaf_detector.pt   # thêm chế độ detector_top1
```

Tiêu chí chấp nhận (đặt trước khi chạy, để không "chọn số đẹp" sau):
1. Recall khung lá ≥ 0,85 tại IoU 0,5 trên cảnh val + test (in ở cuối lúc train).
2. `detector_top1` (FedAvg) cao hơn `full_frame` ≥ 8 điểm trên cả val lẫn giữ riêng, và không thấp hơn "tự cắt lại" quá
   5 điểm.

Chỉ khi đạt mới tích hợp vào web: `/api/inference/all` thêm tham số bật detector, bước 4 vẽ khung lá và cho chọn lá.
Chưa tích hợp vì chưa có checkpoint đã kiểm chứng; không đưa mã chưa kiểm chứng vào đường chẩn đoán.

## 3. Dữ liệu ngoài đồng đang bị cách ly: thiếu dữ liệu, không phải thiếu cách train

Release `mixed_pv_pd_v3` chỉ đưa 2.115 lá PlantDoc (14 nhóm) vào tập học. Trong `quarantine.jsonl` còn **5.569 lá PlantDoc có
nhãn trong taxonomy** (quarantine_status = clean; không tính 11 lá có `class_id = -1`, 473 lá xung đột nhãn và 6 lá nghi
trùng cảnh). Chúng bị cách ly vì nhãn chỉ "ngầm định", không phải vì ảnh hỏng:

| Lý do | Số lá | Nghĩa |
|---|---|---|
| `assumed_healthy` | 4.019 | nhãn PlantDoc kiểu "Apple leaf" được hiểu là lá khỏe |
| `ambiguous_alias` | 1.550 | tên nhãn PlantDoc khớp gần đúng với nhãn PlantVillage |

Chúng thuộc **14 nhóm khác hoàn toàn** với 14 nhóm đang có ảnh ngoài đồng (giao nhau = 0). Nếu nhận được: 14 → **28/38 nhóm**
có ảnh ngoài đồng; còn **10 nhóm** vẫn không có (Apple Black rot, Cherry Powdery mildew, Corn healthy, Grape Esca, Grape Leaf
blight, Orange Haunglongbing, Peach Bacterial spot, Potato healthy, Strawberry Leaf scorch, Tomato Target Spot).

**Đã đo** (`scripts/eval_quarantined_plantdoc.py` → `artifacts/quarantined_plantdoc_eval.json`; ảnh khớp `byte_sha256`):
7 checkpoint hiện có chỉ đạt **Centralized 3.4%, FedAvg 2.1%** trên 5.569 lá này, và hầu hết nhóm gần 0%:

| Nhóm | Số lá | Centralized | FedAvg |
|---|---|---|---|
| `Blueberry___healthy` | 830 | 4% | 2% |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 788 | 1% | 1% |
| `Peach___healthy` | 568 | 2% | 0% |
| `Raspberry___healthy` | 550 | 4% | 1% |
| `Strawberry___healthy` | 485 | 5% | 0% |
| `Corn_(maize)___Northern_Leaf_Blight` | 333 | 8% | 4% |
| `Pepper,_bell___healthy` | 320 | 2% | 2% |
| `Tomato___healthy` | 305 | 0% | 0% |
| `Soybean___healthy` | 258 | 10% | 5% |
| `Pepper,_bell___Bacterial_spot` | 255 | 0% | 1% |
| `Apple___healthy` | 247 | 4% | 13% |
| `Cherry_(including_sour)___healthy` | 240 | 0% | 0% |
| `Grape___healthy` | 216 | 9% | 11% |
| `Apple___Cedar_apple_rust` | 174 | 0% | 1% |

Đọc kết quả cho đúng:
- Đây là "khớp với nhãn ngầm định", **không phải độ chính xác thật**: nhãn chưa được người kiểm, nên một phần "sai" có thể do
  nhãn. Xem ngẫu nhiên 24 ảnh (Tomato healthy, Peach healthy, Tomato Yellow Leaf Curl Virus) thì phần lớn trông đúng nhãn;
  ảnh khỏe ngoài đồng thường là ảnh stock nền trắng/nền đất, rất khác ảnh PlantVillage.
- Dù vậy mô hình hiện tại gần như **không nhận được** loại ảnh này, và thường đoán thành một bệnh (ví dụ lá cà chua khỏe bị
  đoán Leaf Mold). Đây là chỗ train lại có thể giúp thật, vì hiện không có ảnh nào của các nhóm này trong tập học.
- Chưa dùng làm bằng chứng và chưa đưa lên web cho tới khi nhãn được kiểm.

### Kiểm nhãn bằng mắt (bước cần người)
```bash
python scripts/build_label_review.py        # ghi data/label_review/index.html (14 nhóm, 30 lá mỗi nhóm = 420 lá)
```
Mở `data/label_review/index.html` bằng trình duyệt, bấm Đúng / Sai / Không rõ cho từng ảnh, rồi "Tải kết quả" →
`decisions.json` đặt cạnh `index.html`. Lấy mẫu có seed cố định (tái lập). Quy tắc nhận nhóm đặt trước khi xem kết quả:
**≥ 90% "đúng" và ≤ 20% "không rõ"**; nhóm không đạt vẫn ở lại cách ly. Trang tự lưu trong trình duyệt nên dừng giữa chừng
được.

## 4. Train lại ba cách: chia hai giai đoạn

Giữ nguyên để còn so được với số hiện tại: W0 đã ghim, MobileNetV3-Small, 38 nhóm, 5 client, phân vùng
`label_alpha_0_1`, `canonical_v1` khi suy luận, tập kiểm tra và tập giữ riêng của v3.

Kiểm tra mã train hiện có (`train_production.py`, `fedavg_mixed_runner.py` ở các nhánh `train-*`) cho thấy các runner bị
**ràng buộc rất chặt bởi "spec đã duyệt"** (hash release, hash pilot, ngưỡng chất lượng, resume guard):

| Thiết lập | Runner hiện tại | Đổi được bằng cấu hình? |
|---|---|---|
| Dữ liệu / release | `--release-dir`, `--release-name`, `--expected-release-sha` | Có (cần release v4 + hash) |
| Seed | `--seed` (khớp spec) | Có |
| Tỉ lệ ảnh PlantDoc khi lấy mẫu | chỉ 0 hoặc 0,25 theo `selected_variant` của spec | Không (sửa kiểm tra trong runner) |
| Transform lúc học | chọn được `light_augment_v2`, nhưng phải trùng `preprocessing` trong spec | Một phần (sửa spec đã duyệt) |
| Bộ tối ưu Centralized | chỉ AdamW (từ chối cái khác) | **Không** (sửa mã) |
| Bộ tối ưu FedAvg | chỉ SGD (momentum đọc từ spec) | Có (momentum), không đổi họ |
| Lịch lr / dừng sớm | plateau theo PlantDoc, patience khác nhau (3 so với 4), dừng sớm | **Không** (sửa mã) |

Vì vậy đề xuất hai giai đoạn, quyết định riêng:

- **Giai đoạn A (ít rủi ro): chỉ đổi dữ liệu.** Dùng nguyên runner và thiết lập đã duyệt, thay release v3 bằng v4 (v3 + lá
  đã kiểm nhãn), 3 seed. Câu trả lời cho "thiếu dữ liệu ngoài đồng thì bổ sung có cải thiện không" sạch nhất ở đây vì chỉ đổi
  một yếu tố. Hạn chế so sánh Centralized/FedAvg (AdamW so với SGD, 83 epoch so với 27 vòng) vẫn còn và vẫn phải ghi.
- **Giai đoạn B (nhiều việc): so sánh công bằng.** Cùng họ bộ tối ưu, cùng ngân sách, cùng chỉ số chọn checkpoint, không dừng
  sớm. Cần sửa runner Centralized (và kiểm lại cổng chất lượng/resume của cả hai runner), rồi chạy thử trên GPU trước khi
  chạy 21 lượt. Không nên gộp với A: gộp thì không tách được phần cải thiện do dữ liệu và do cách train.

Khối lượng giai đoạn A: 21 lượt (3 seed × (1 Centralized + 1 FedAvg + 5 Local)), cỡ 10 GPU-giờ mỗi seed theo ước lượng
trong `docs/EXPERIMENT_PROTOCOL.md`.

Chốt trước khi train (để đây là lần cuối): ghi cấu hình vào một tệp, lấy mã băm, commit trước khi chạy; chọn checkpoint chỉ
trên tập kiểm tra; báo cáo tập giữ riêng đúng một lần. Lưu ý trung thực: tập giữ riêng của v3 đã được dùng để chẩn đoán nên
không còn "chưa ai nhìn"; nếu thêm nhóm mới thì cần một tập kiểm tra ngoài đồng mới khóa trước khi train.

Việc cần làm sau khi có kết quả: đưa bundle về bằng `scripts/import_branch_demo.py` (cập nhật nhánh/commit ghim và hash; web
kiểm hash từng checkpoint trước khi hiện số), chấm lại độc lập, so từng nhóm cũ/mới. Không hứa trước con số.

## 5. Công thức train mới và ngưỡng chấp nhận (chuẩn bị 2026-10-21, chưa chạy)

Cập nhật sau đợt đo ở `docs/DANH_GIA_DU_LIEU_MO_HINH_FEDAVG_2026-10-21.md`. Mọi mô hình hiện có đạt 33,1% (Centralized) và 31,8% (FedAvg) top-1
trên ảnh bệnh ngoài đồng của PlantWild và đoán thành có bệnh trong 87–94% ảnh lá khỏe, nên mục tiêu của lần train sau là ảnh ngoài đồng, không phải PlantVillage.

### 5.1 Dữ liệu
- Release v4 = v3 cộng các lá PlantDoc đã kiểm nhãn (5.569 lá đang cách ly; quy tắc nhận nhóm ≥ 90% "đúng" và ≤ 20% "không rõ" ở mục 3) và ảnh lá khỏe ngoài đồng.
- **PlantWild không được dùng để học**: giấy phép CC BY-NC-ND 4.0 và đã khóa làm tập thử ngoài đồng cố định (`artifacts/external_eval_set.json`, manifest SHA-256 `5adba619…`).
  Nếu thêm nguồn ảnh mới thì phải loại trùng với manifest này bằng `scripts/build_external_eval_set.py` trước khi học.
- Cảnh PlantDoc nguyên khung đi qua bộ phát hiện lá (`scripts/train_leaf_detector.py`) để cắt lá; tiêu chí chấp nhận bộ phát hiện giữ như mục 2.

### 5.2 Cách học
| Hạng mục | Hiện tại | Đề xuất |
|---|---|---|
| Tăng cường ảnh | chỉ lật ngang | `src/data/robust_transforms.py` (cắt-phóng ngẫu nhiên, xoay, đổi màu, mờ, nén JPEG, độ phân giải thấp, xóa mảng); suy luận vẫn `canonical_v1` |
| Lấy mẫu | có hoàn lại, một ảnh lặp tới 14 lần mỗi epoch | không hoàn lại; cân bằng miền bằng trọng số mất mát hoặc trọng số lớp |
| Hàm mất mát | cross-entropy | thêm label smoothing 0,1 để giảm độ tin cậy quá cao khi sai |
| Chọn checkpoint | macro-F1 PlantDoc trên chính tập kiểm tra | giữ nguyên quy tắc đã ghi trước, chọn chỉ trên tập kiểm tra; báo tập giữ riêng và PlantWild đúng một lần |
| So sánh Centralized và FedAvg | AdamW so với SGD không động lượng, 83 epoch so với 27 vòng | cùng họ bộ tối ưu (SGD động lượng 0,9) và cùng số bước gradient: số epoch của Centralized = số vòng × số epoch cục bộ của FedAvg |
| BatchNorm trong FedAvg | gộp trung bình theo n_k | đối chứng thêm FedBN (BatchNorm giữ cục bộ) hoặc GroupNorm; ghi rõ phương án nào dùng |
| Hạt giống | một | ba (42, 123, 2026); báo trung bình ± độ lệch chuẩn |

Phép đo ở mục 4.3 của báo cáo cho thấy phép gộp không gây sai số và chênh lệch với tập trung đến từ số bước cục bộ cùng độ lệch dữ liệu; vì vậy nên thử E nhỏ và FedAvgM trước khi tăng số client.

### 5.3 Backbone
Giữ MobileNetV3-Small cho so sánh ba phương pháp (một lần truyền 6,3 MB). Thêm một mô hình trần chỉ huấn luyện tập trung (ConvNeXt-Tiny hoặc EfficientNet-B0) để biết giới hạn của dữ liệu;
mô hình cỡ 28 triệu tham số tương ứng khoảng 112 MB mỗi lần truyền, gấp khoảng 18 lần, nên không dùng cho học phân tán trong phạm vi đồ án.

### 5.4 Ngưỡng chấp nhận (đặt trước khi train)
Một đợt train mới chỉ được nhập vào web nếu đồng thời:
1. top-1 trên PlantWild (tầng bệnh) tăng ít nhất 10 điểm so với mô hình hiện có (Centralized ≥ 43,1%, FedAvg ≥ 41,8%) và cận dưới khoảng tin cậy 95% cao hơn mốc cũ;
2. top-1 PlantDoc tập giữ riêng không giảm quá 2 điểm so với 63,4% (Centralized) và 50,0% (FedAvg);
3. top-1 PlantVillage không giảm quá 0,5 điểm trên tập kiểm tra và tập giữ riêng;
4. tỉ lệ ảnh lá khỏe PlantWild bị đoán thành bệnh giảm xuống dưới 60% (chỉ báo xu hướng, nhãn ngầm định);
5. kết quả báo ba hạt giống; chênh Centralized và FedAvg đi kèm khoảng tin cậy.

### 5.5 Việc đã làm sẵn
- Tập thử ngoài đồng và công cụ đo: `scripts/build_external_eval_set.py`, `scripts/eval_external_set.py`, `scripts/eval_inference_variants.py`.
- Bộ tăng cường ảnh và kiểm khô (vẽ lưới ảnh, chạy một batch qua MobileNetV3-Small trên CPU): `src/data/robust_transforms.py`, `scripts/preview_augmentations.py`.
- Chức năng chọn loại cây ở trang Chẩn đoán (cận trên đã đo: +7,6 đến +21,2 điểm trên PlantDoc khi chọn đúng cây).
- Đo tốc độ hai máy: `scripts/prepare_network_speedtest.py`, `scripts/summarize_network_speedtest.py`, hướng dẫn LAN `docs/HUONG_DAN_DO_TOC_DO_LAN.md`.

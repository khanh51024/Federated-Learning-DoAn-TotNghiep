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

## 3. Train lại ba cách (chỉ khi muốn kết luận chắc hơn hoặc tăng ảnh ngoài đồng)

Giữ nguyên để kết quả còn so sánh được: release `mixed_pv_pd_v3`, khởi tạo W0 (`w0_fingerprint`), phân vùng
`label_alpha_0_1`, 5 client, MobileNetV3-Small, `canonical_v1` khi suy luận, chọn checkpoint trên tập kiểm tra và báo cáo
trên tập giữ riêng.

Thay đổi theo mục tiêu:
- **So sánh công bằng (luận văn):** cùng bộ tối ưu và cùng ngân sách cho Centralized và FedAvg; FedAvg chạy đến khi hội tụ
  thật (bản hiện tại dừng ở vòng 32, chọn vòng 27 khi F1 PlantVillage còn tăng); ≥ 3 seed; báo cáo trung bình ± độ lệch.
- **Ảnh ngoài đồng:** tăng `plantdoc_sampling_ratio` (hiện 0,25) và dùng augment mạnh hơn có sẵn trong contract
  (`light_augment_v2`, `moderate_augment_v1` trong `contract/transforms.py`; transform lúc suy luận vẫn là resize thẳng
  224×224 nên web không phải đổi).

Việc cần làm: mã huấn luyện nằm ở các nhánh `train-centralized-v6`, `train-fedavg-non-iid-v7`,
`train-local-only-a01-v7` (cấu hình mẫu: `configs/central_spec_v3.json`, `configs/quality_spec_v4_FEDAVG_FULL.json`);
chạy trên Kaggle/Colab, rồi đưa bundle mới về bằng `scripts/import_branch_demo.py` (cập nhật nhánh/commit ghim và hash,
web kiểm hash từng checkpoint trước khi hiện số). Không hứa trước con số: chạy xong mới biết tăng được bao nhiêu.

# Kiểm lại số validation của gói PC v3 bằng đường suy luận của web

Thư mục này ghi kết quả đồ án (phía web) **tính lại từ đầu** 13 mô hình của `pc_v3_ket_qua/models/` rồi đối chiếu với `validation_metrics.json` mà nhóm train đã công bố.

## Đã làm gì
- Chạy trên **CPU**, torch 2.14.0+cpu, qua đúng đường suy luận của web (tiền xử lý `canonical_v1` của hợp đồng đi kèm gói, từng mô hình nạp từ đúng byte đã băm).
- Chỉ dùng **tập validation 4.613 ảnh** (PlantVillage 4.344 + PlantDoc 269). Không mở tập kiểm thử, tập calibration hay tập ngoài miền; tập kiểm thử cuối vẫn là số P6 của nhóm train, chạy một lần.
- 13 mô hình: Centralized AdamW, Centralized SGD, FedAvg, SCAFFOLD, Local 00–04 và 4 bản int8 (động, chỉ lớp Linear). Bản int8 được đo **từng ảnh một**, giống cách web suy luận (lượng tử hóa động tính thang theo cả lô, nên đo theo lô sẽ ra số khác).
- FedProx, MOON và SCAFFOLD-BN có trong gói nhưng chưa đưa lên web nên không nằm trong bảng này.

## Kết quả
- 117 phép so (13 mô hình × {tổng, PlantDoc, PlantVillage} × {top-1, macro-F1 các lớp có mặt, độ đúng loại cây}).
- **Cả 117 phép khớp tuyệt đối** (chênh lệch bằng 0), nên đều nằm trong ngưỡng 0,002 mà nhóm train đặt ra.
- Chi tiết từng phép: `KIEM_LAI_VALIDATION.csv`; tóm tắt, môi trường và commit nguồn: `KIEM_LAI_VALIDATION.json`.

## Cách chạy lại
Ở repo đồ án (nhánh `web-int8-pc-v3`):

    python scripts/import_pc_package.py --package <thư mục pc_v3_ket_qua> --name pc_v3_s42 --report-dir <thư mục pc_v3_ket_qua>/kiem_lai_web

Cần dữ liệu `data/dataset_sources/mixed/pv_pd_v3` (release `pv_pd_v3`) và Git LFS đã kéo các tệp `.pt`. Máy có torch mới (2.14) chỉ còn engine lượng tử `onednn`; mã web đã chấp nhận `x86`, `fbgemm` và `onednn`.

## Giới hạn
- Đây là số validation, không phải số kiểm thử. Không dùng để thay các số P6.
- Khớp tuyệt đối trên một máy CPU không chứng minh cùng kết quả trên mọi phần cứng; số float có thể lệch ở chữ số cuối trên CPU/GPU khác.

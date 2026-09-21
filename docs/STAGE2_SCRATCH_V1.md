# GĐ2 train từ đầu — random initialization v1

Theo yêu cầu ngày 19/09/2026, dùng `stage2_scratch` cho các lần train mới.
MobileNetV3-Small được tạo với `weights=None` (`pretrained=False`) cho **mỗi
điều kiện và seed**. Không dùng weights ImageNet hoặc model PlantVillage đã train.
Luồng `stage2_matched` vẫn giữ nguyên để kiểm tra/tái lập kết quả pretrained cũ.

## Cách chạy

Từ thư mục `gd2_federated_learning`, với môi trường trong
`requirements-stage2-matched.txt`:

```powershell
python -m stage2_scratch preflight
python -m stage2_scratch run --conditions label100 label1 label01 --seeds 42 --output runs/stage2_scratch_v1_seed42
python -m stage2_scratch collect --output runs/stage2_scratch_v1_seed42
```

`run` mặc định yêu cầu đường dẫn output **chưa tồn tại**, kể cả thư mục rỗng.
Không tự skip kết quả cũ, không xóa kết quả cũ và không tự resume. Lần train độc
lập tiếp theo phải chọn output mới. Khi cần tiếp tục **chính thí nghiệm scratch
đang dở**, chỉ định rõ:

```powershell
python -m stage2_scratch run --conditions label100 label1 label01 --seeds 42 --output runs/stage2_scratch_v1_seed42 --resume
```

Resume kiểm tra protocol, code, dữ liệu, trainer, identity và checkpoint. Không
nhận output pretrained v5. Khi copy phiên Kaggle, giữ toàn bộ output scratch.

Notebook cho luồng mới: `kaggle_stage2_scratch_v1.ipynb`, mặc định preflight.
Đặt `ACTION='run'` để train; `RESUME=False` là mặc định. Package Kaggle phải có
thư mục `stage2_scratch` mới. Bộ package v5 cũ chưa có module này.

## Giao thức cố định và ý nghĩa

- Dùng lại split sạch v5: 38.916 train / 4.435 validation / 10.954 test, 38 lớp.
- 5 client, AdamW lr=0,001, weight decay=0,0001, batch 32, 10 vòng × 1 local epoch.
- Tất cả trọng số học được khởi tạo ngẫu nhiên; buffer/BatchNorm theo mặc định
  của kiến trúc. Seed giống nhau cho các điều kiện cho cùng trạng thái khởi đầu
  để so sánh công bằng, nhưng mỗi model là một instance độc lập.
- Trong cùng một lượt train, các client nhận **global weights của vòng hiện
  tại** trước local fit. Không reset ngẫu nhiên ở mỗi vòng: làm vậy sẽ phá FedAvg.
- Mỗi job ghi `initialization/<job_id>.json`; runner kiểm tra hash model thực
  tế với hash random đã ghi trước khi train. Optimizer AdamW mới ở mỗi local fit.
- Test chỉ đánh giá sau khi chọn checkpoint tốt nhất bằng validation accuracy.
- `scratch_protocol.json` và `scratch_comparison.json` có scope riêng;
  collector không trộn kết quả pretrained hoặc smoke vào so sánh scratch.

Giữ 10 vòng để thí nghiệm đầu có ngân sách thống nhất; train từ đầu có thể chưa
hội tụ trong 10 vòng. Không điều chỉnh vòng/alpha nhằm ép accuracy xuống thấp.
Nếu cần tăng ngân sách, dựa trên validation và áp dụng đồng nhất ở một protocol
mới. Không quy toàn bộ chênh lệch pretrained–scratch cho non-IID.

Muốn so sánh GĐ1–GĐ2 hoàn toàn theo scratch, cần chạy lại cả các mốc label100,
label1 và label01 với scratch; không trừ trực tiếp số ~99% pretrained cũ.
Toàn bộ GĐ2 vẫn cần multi-seed, Centralized/Local-only cùng protocol và fairness.

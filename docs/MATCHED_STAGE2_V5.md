# FedAvg GĐ2: Dirichlet và phép so sánh với GĐ1

## Mục đích và phạm vi

Đã triển khai luồng `stage2_matched`, gọi trực tiếp
`stage1_compat.runner.run_single_fedavg_job`. Không sao chép hoặc thay thuật toán
FedAvg. Các kết quả GĐ1 cũ (~99%) giữ nguyên để tham khảo lịch sử; chúng dùng split
cũ có nhóm lá xuất hiện ở cả train và test, nên không tính chênh lệch với v5.

Mốc so sánh mới là **cấu hình GĐ1 chạy lại trên dữ liệu sạch**. Đây không phải
khôi phục chính xác số đo lịch sử. Notebook v4 và các kết quả K=10/SGD cũng không
được gộp với v5. FedProx, FedBN, SCAFFOLD, MOON không nằm trong luồng v5 này.

## Những yếu tố được cố định

- MobileNetV3-Small, ImageNet weights, 38 lớp, Resize 224 và augmentation GĐ1.
- 5 client tham gia mỗi vòng; 10 vòng, 1 local epoch, batch 32.
- AdamW, LR 0,001, weight decay 0,0001; khởi tạo optimizer mới mỗi local fit như GĐ1.
- Cùng cách tổng hợp state_dict, trọng số bằng số ảnh thực tế của client.
- Cùng seed khởi tạo theo từng cặp so sánh; báo cáo kiểm tra hash weights ban đầu.
- Chọn checkpoint theo validation accuracy, giữ checkpoint sớm hơn khi hòa.
- Test chỉ dùng sau khi train xong và chọn checkpoint; không điều chỉnh alpha theo test.
- Cùng ảnh train, validation, test và mapping nhãn cho tất cả điều kiện.

Giữ local epoch như nhau có thể tạo chênh lệch số batch do làm tròn ở client có
kích thước khác nhau. Engine lưu `sample_count`, `step_count` và payload ước tính.
Các seed 42/123/2026 thay ngẫu nhiên khi train; split seed cố định 42. Độ lệch chuẩn
qua ba seed này phản ánh biến thiên train, không phản ánh nhiều lần bốc phân hoạch.

## Dữ liệu và Dirichlet

Manifest đầy đủ tại `data/partitions_stage2_matched_v5`:
**38.916 train / 4.435 validation / 10.954 test** từ 54.305 ảnh. Tỷ lệ mục tiêu
72/8/20 theo GĐ1; số thực tế chênh nhẹ do không cắt nhóm lá.

Tách holdout trước khi phân bổ client. Một nhóm là hợp của leaf-map và ảnh trùng
SHA-256. Các kiểm tra chặn trùng đường dẫn, nhóm lá và byte giữa train/val/test
hoặc giữa client; kiểm tra nhãn thực tế, đủ 38 lớp ở holdout và bảo toàn mọi ảnh.
Leaf-map không phủ toàn bộ dữ liệu, nên chưa chứng minh loại được mọi ảnh gần
giống về thị giác hoặc mọi lá chưa có metadata.

Tám điều kiện được định trước trong `stage2_matched/data.py`:

- `label100`, `label1`: mốc GĐ1 chạy lại; mỗi lớp được phân bổ bằng Dirichlet(α).
- `label01`: lệch nhãn α=0,1; so sánh cùng seed với cả hai mốc trên.
- `quantity100`, `quantity01`: một vector tỷ lệ client Dirichlet dùng chung cho
  mọi lớp, β=100 và β=0,1; so sánh hai mức này với nhau.
- `label_quantity01`: lệch nhãn α=0,1 kết hợp lệch số lượng β=0,1.
- `feature100`, `feature01`: gán mỗi nhóm train vào một trong 5 miền ảnh mô phỏng,
  rồi phân bổ từng miền cho client bằng Dirichlet(α=100 hoặc 0,1). Cùng ảnh giữ
  cùng domain ở cả hai mức; nhãn bệnh không đổi. Miền là các biến đổi màu/sáng/mờ/
  nhiễu mức moderate cố định, **không phải metadata thiết bị thực**. Holdout sạch.

`domain_id` thực sự được đọc khi train, trước augmentation GĐ1. Không chỉ ghi cấu
hình feature vào báo cáo. `client_domain_counts` cho biết lượng ảnh từng miền.
Các điều kiện feature có thể kéo theo lệch nhãn/số lượng; các histogram được lưu
để mô tả mức lệch thực tế, không giả định các trục hoàn toàn độc lập.

Client tối thiểu 10 ảnh, sửa bằng chuyển nguyên nhóm; số nhóm sửa được lưu ở
`split_diagnostics`. Với quantity β=0,1, seed 42 có kích thước
10/6.268/31.299/1.329/10: đây là phân phối rất cực đoan, cần báo cáo rõ. Không loại
client nhỏ hoặc đổi seed sau khi xem điểm để tạo kết quả mong muốn.

## Chạy local / Kaggle

Từ thư mục `gd2_federated_learning`, dùng môi trường PyTorch 2.6/torchvision 0.21
và các phụ thuộc trong `requirements-stage2-matched.txt`. Bản local hiện dùng CPU.
Notebook `kaggle_stage2_matched_v5.ipynb` nhận đường dẫn package/dataset rõ ràng;
chỉ train khi đặt `ACTION = "run"`.

```powershell
# Manifests đã được tạo. Kiểm tra cả byte ảnh thực trước lượt train.
python -m stage2_matched preflight

# Ưu tiên hoàn tất một bộ đối chiếu seed 42 trước khi chạy thêm seed.
python -m stage2_matched run --conditions label100 label1 label01 --seeds 42
python -m stage2_matched run --conditions quantity100 quantity01 label_quantity01 feature100 feature01 --seeds 42

# Lặp các điều kiện theo seed 123/2026 khi còn đủ ngân sách.
python -m stage2_matched run --seeds 123 2026
python -m stage2_matched collect
```

Output mặc định `runs/stage2_matched_v5`; có thể đổi bằng `--output` sang thư mục
mới có quyền ghi. Khi bị ngắt, chạy lại nguyên lệnh với cùng output để resume từ
vòng đã commit. Copy toàn bộ output khi chuyển phiên Kaggle. Hash dữ liệu, code,
trainer hoặc runtime không khớp sẽ bị từ chối; không sửa JSON để ép resume.
Lệnh mặc định giới hạn phiên 420 phút, kiểm tra deadline từng batch và giữ tối đa
vòng đã commit. Đây là giới hạn phiên, **không tự biết quota tài khoản Kaggle**.
Vòng đang dở có thể phải chạy lại. Phải kiểm tra thời lượng thực và quota trước
khi tiếp tục; không tự giảm vòng, epoch hoặc tập train riêng cho một điều kiện.

Tạo lại manifest khi cần một phiên bản mới (không ghi đè thư mục hiện có):

```powershell
python -m stage2_matched prepare --suite data/partitions_stage2_matched_new
```

Smoke test giữ 5 client và nhóm nguyên vẹn, nhưng 2 vòng và không pretrained;
được đánh dấu `smoke=true`, collector không cho vào so sánh khoa học:

```powershell
python -m stage2_matched prepare --smoke --suite data/matched_smoke_new
python -m stage2_matched run --suite data/matched_smoke_new --output runs/matched_smoke_new --conditions label100 label01 --device cpu
python -m stage2_matched collect --suite data/matched_smoke_new --output runs/matched_smoke_new
```

CLI đặt TEMP/TMP/TORCH_HOME dưới `.stage2_matched_runtime` trong project trên ổ D;
không tải pretrained về cache ổ C cho các lần chạy mới.

## Đọc kết quả

`matched_comparison.json` chứa accuracy, macro-F1, best round và chênh lệch
**target − reference, đơn vị điểm phần trăm** theo seed. Cặp thiếu dữ liệu được
liệt kê; không điền bằng số lịch sử hoặc kết quả chạy thử. Giá trị dương cũng
được giữ nguyên. Chưa có kết quả đầy đủ trên Kaggle thì chưa thể kết luận mức
suy giảm hoặc xếp hạng mô hình.

Engine dùng lại vẫn ghi `profile=stage1_compat`; nhận diện thí nghiệm mới bằng
`identity.context.scope=stage2_matched_clean_v5` và collector `stage2_matched`.
Không dùng collector lịch sử để tính chênh lệch mới.

Accuracy cao tự nó không phải lỗi và Dirichlet không bảo đảm điểm sẽ thấp.
Mục tiêu là đo trung thực trên holdout sạch, không hạ điểm. Nếu kết quả vẫn cao,
báo cáo đúng và kiểm tra thêm ảnh gần trùng/độ khó PlantVillage thay vì tăng độ
nhiễu hoặc đổi alpha dựa trên test.

Luồng này hoàn thiện phần **FedAvg so với cấu hình GĐ1 chạy lại**. Để nghiệm thu
toàn bộ GĐ2 theo đề cương vẫn cần Centralized/Local-only và phân tích fairness
trên chính protocol matched; không lấy đối chứng v4 khác trainer ghép vào v5.
Collector luôn ghi `scientific_stage2_complete=false` để tránh đánh dấu nhầm.

## Kiểm chứng local ngày 16/09/2026

- 41 kiểm thử đạt: bộ train GĐ1, content guard, Dirichlet bảo toàn nhóm, feature
  chỉ áp dụng khi train, và từ chối so sánh khác trainer/holdout/khởi tạo.
- Audit byte ảnh thực cho đủ 54.305 ảnh trên cả 8 điều kiện đạt; không phát hiện
  nhóm lá hoặc ảnh trùng byte vượt ranh giới split/client trong các manifest này.
- Chạy ảnh thật với `label100`, `label1`, `label01`, `feature01`: 324 train,
  176 validation, 156 test, 5 client, 2 vòng, không pretrained. Cả bốn hoàn thành;
  test accuracy 7,05–9,62%. Đây chỉ là smoke test, không phải kết quả nghiên cứu.
- Ngắt ngay sau checkpoint vòng 1 (trước cập nhật ledger) ở `feature01`, resume
  vòng 2: cả 244 tensor và best weights khớp tuyệt đối lần chạy liên tục.
- Các cell Python notebook đã compile; chưa chạy notebook trên GPU Kaggle.

Bằng chứng nằm ở `../output/matched-v5-validation/`: `tests-final.log`,
`full-preflight.json`, `run/matched_comparison.json`,
`resume/resume_verification.json` và `verification.json`.

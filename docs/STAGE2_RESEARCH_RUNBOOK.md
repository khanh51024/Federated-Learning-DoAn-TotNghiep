# GĐ2: ma trận nghiên cứu nhiều seed

Đây là cấu hình bổ sung cho nghiệm thu nghiên cứu, khác pilot 18 job một seed trong `kaggle_30h.yaml`. Không coi test pass, dry-run hoặc việc tạo manifest là kết quả huấn luyện. Chưa chạy ma trận đầy đủ, chưa đủ dữ liệu để chốt kịch bản GĐ3.

## Ma trận và protocol

`configs/stage2_research.yaml` có 13 điều kiện, training seeds `[42, 123, 2026]`, ba phương pháp Centralized/FedAvg/Local-only: 117 job, 390 model Local-only.

- IID, feature none.
- Label skew với alpha 100, 10, 1, 0.5, 0.1 (bắt buộc); thêm 5 và 0.05 giữ từ sweep phân hoạch cũ.
- Quantity skew với quantity alpha 1 và 0.1, feature none.
- Label + quantity skew với cả hai alpha 0.1, feature none.
- IID với feature moderate.
- Label alpha 0.1 với feature moderate.

Đây là hợp của các điều kiện đã khai báo trong sweep cũ và yêu cầu mới; không tự mở rộng thành mọi tích Descartes với mild/strong. Split seed luôn 42; chỉ training seed thay đổi. CI phản ánh biến thiên do huấn luyện trên cùng phân hoạch, không phản ánh bất định do lấy mẫu dataset/phân hoạch. Giữ MobileNetV3-Small, 38 lớp, K=10, C=1, E=1, chạy tuần tự client, initialization chung theo từng seed, best theo validation loss, early stopping tắt.

Manifest v3 ở `data/partitions_train_v3_research` được tạo riêng từ dữ liệu thật. Không ghi đè v1/v2. Quantity/mixed skew vẫn giữ diagnostics repair và histogram thực tế; không giả định alpha nhỏ phải cho gap lớn hơn.

## Ngân sách

Runner vẫn giới hạn tổng 30 giờ trừ giờ đã dùng và 3 giờ dự phòng, session 8 giờ, hệ số 1.5, round chung 10–20. 117 job có thể bị calibration từ chối vì không đủ 10 round; điều đó là kết quả kiểm tra tài nguyên hợp lệ. Không có bảo đảm ma trận đầy đủ vừa 30 giờ. Muốn tăng ngân sách hoặc thay protocol phải thỏa thuận riêng; không xóa seed/kịch bản để làm đẹp trạng thái hoàn tất.

Code được cập nhật sẽ đổi source fingerprint. Không chép code mới vào output có protocol đã khóa của bản cũ rồi bỏ guard để resume. ZIP 30h cũ vẫn giữ nguyên và chỉ nên dùng với ledger được tạo bởi chính bản đó. Dùng output mới cho bản nghiên cứu và nhập tổng quota đã tiêu thụ từ mọi phiên trước.

## Lệnh chạy

Giải nén ZIP nghiên cứu vào `/kaggle/working`, mở `kaggle_stage2_research.ipynb`. Notebook dùng cùng trainer package. Hoặc chạy các lệnh dưới từ thư mục package; thay chính xác các đường dẫn và số giờ quota.

```bash
export GD2_DATASET=/kaggle/input/YOUR-DATASET/color
export GD2_INDEX=/kaggle/working/gd2_federated_learning/data/partitions_train_v3_research/index.json
export GD2_OUTPUT=/kaggle/working/stage2_research_output
python -m fl_training.kaggle_budget --config configs/stage2_research.yaml --action plan --dataset-root "$GD2_DATASET" --partition-index "$GD2_INDEX" --output-root "$GD2_OUTPUT"
python -m fl_training.kaggle_budget --config configs/stage2_research.yaml --action calibrate --dataset-root "$GD2_DATASET" --partition-index "$GD2_INDEX" --output-root "$GD2_OUTPUT" --already-used-hours 0 --quota-remaining-hours 30
python -m fl_training.kaggle_budget --config configs/stage2_research.yaml --action run --dataset-root "$GD2_DATASET" --partition-index "$GD2_INDEX" --output-root "$GD2_OUTPUT" --already-used-hours 0 --quota-remaining-hours 30
python -m fl_training.kaggle_budget --config configs/stage2_research.yaml --action collect --dataset-root "$GD2_DATASET" --partition-index "$GD2_INDEX" --output-root "$GD2_OUTPUT"
```

Thay `0` và `30` bằng số tài khoản thực tế. Lệnh run cũng là resume; exit 75 là pause. `/kaggle/working` không bền: lưu toàn bộ output/ledger/checkpoint, gắn backup phiên trước và restore nguyên cây trước khi resume. Collect chỉ đọc artifact và tạo báo cáo, không train; cần ledger đã khóa protocol. Không dùng lệnh sweep cũ để thay thế runner có deadline/ledger này.

## Tổng hợp và GĐ3

- `comparison.csv/json/md`: từng condition/seed/method đã được kiểm tra identity checkpoint/evaluation.
- `summary.csv/json`: mỗi condition/method/metric có n, số seed kỳ vọng, trạng thái đủ/thiếu, mean, sample std (ddof=1), CI 95% khi n>=3. Accuracy/F1 dùng phần trăm. Local-only: một mean của các model client trên global test cho mỗi seed; dispersion giữa client ở comparison không phải std giữa seed hoặc fairness FedAvg.
- `accuracy_gap_paired.csv`: Centralized minus FedAvg cùng condition và seed, theo điểm phần trăm; giữ gap âm.
- `accuracy_gap_summary.csv` và đồ thị `accuracy_gap_vs_alpha_test.png/pdf`: thống kê trên các gap đã ghép cặp, không lấy std của hai phương pháp cộng trực tiếp; không ép đường đơn điệu.
- `stage3_candidate.json`: bị chặn khi thiếu bất kỳ job/seed/validation cần thiết. Khi đủ, đề xuất ứng viên có mean gap validation lớn nhất; cần đánh giá khoa học thêm, không tự tuyên bố mức suy giảm nghiêm trọng hay thuật toán GĐ3 sẽ cải thiện. Gap test không tham gia lựa chọn.
- `collection_status.json`, `research_status.json`: expected/completed/missing/failed/invalid và coverage nhiều seed; `scientific_stage2_complete=false` luôn giữ.

CI dùng `mean +/- t(0.975,n-1) * std / sqrt(n)`; với 3 seed, CI có thể rất rộng. Đây là CI mô tả theo giả định training seeds độc lập, chưa điều chỉnh so sánh nhiều kịch bản. Không cắt CI vào [0,100] hoặc làm giả số khi thiếu seed. Tham chiếu API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html

Sau mỗi collect, file summary/gap/candidate cũ bị thay thế hoặc dọn nếu không còn dữ liệu hợp lệ. Chưa có số liệu thật thì không có đồ thị kết quả hay ứng viên GĐ3 giả.

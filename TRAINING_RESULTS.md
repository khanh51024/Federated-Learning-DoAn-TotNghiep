# Kết quả huấn luyện FedAvg

## Run được công bố

- Run ID: `20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62`
- Mô hình: MobileNetV3 Small, huấn luyện bằng FedAvg
- Seed huấn luyện: `42`
- Trạng thái: `completed_with_warnings`
- Số vòng đã chạy: `62/100`
- Vòng tốt nhất: `52`
- Validation loss tốt nhất: `0.0283217734`
- Thời gian chạy: `10.952,50` giây (khoảng 3 giờ 2 phút 33 giây)
- Lý do dừng: early stopping sau 10 vòng không cải thiện tối thiểu `0.0001`
- Cảnh báo runtime: Flower `run_simulation` API đã phát cảnh báo deprecation

## Đánh giá trên global test

Checkpoint `best.pt` được đánh giá trên 10.917 ảnh:

| Chỉ số | Giá trị |
| --- | ---: |
| Loss | 0,030665 |
| Accuracy | 99,0932% |
| Macro F1 | 98,7155% |
| Weighted F1 | 99,0932% |

SHA-256 của model state: `a5af501306293146f21d5d3ce9446f66bc23355d1cbb4df196d55a50d4abe288`.

## Phạm vi kết luận

Run này xác nhận pipeline FedAvg có thể huấn luyện, dừng sớm, lưu checkpoint và
đánh giá end-to-end. Metadata của run đánh dấu `scientific_stage2_complete: false`;
vì vậy các chỉ số trên chưa đại diện cho toàn bộ giao thức thí nghiệm Stage 2 có
đối chứng nhiều seed và nhiều kịch bản non-IID.

Artifact đầy đủ nằm trong
[`results/fedavg/20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62`](results/fedavg/20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62),
gồm cấu hình đã resolve, lịch sử theo vòng, checkpoint, metric test, biểu đồ,
confusion matrix và metric theo lớp. Các checkpoint `.pt` được quản lý bằng Git LFS.

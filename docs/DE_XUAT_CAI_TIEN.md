# Đề xuất cải tiến baseline local-only trên năm client

**Trạng thái:** kế hoạch thí nghiệm, **chưa sửa code train hoặc tạo mô hình mới**. Baseline: mỗi client train độc lập 60 vòng từ cùng W0, không trao đổi trọng số, partition lệch nhãn `α=0,1`. [Tổng kết 5 client](../results/20261003/combined_summary.json) · [metrics riêng](../results/20261003/RESULTS.md) · [runner](../train-gd-2/kaggle-workspace/campaign-mixed-pv-pd-v1/local_only_mixed_runner.py).

## 1. Vì sao mô hình local nhận sai nhiều?

**Số đã đo:** PlantDoc top-1 từng client từ **21,19% đến 29,00%**, macro-F1 supported **11,67% đến 25,64%**. PlantVillage top-1 **44,89% đến 55,46%**. Client 02 chỉ có **20/38** nhãn trong train; các client khác có 23, 24, 27, 24 nhãn. Đánh giá dùng đủ validation 38 lớp, nên mô hình local gặp nhiều nhãn chưa có ví dụ trong shard. Đây là một giới hạn có thể xác minh từ `class_coverage`, không phải lỗi của MobileNetV3-Small riêng.

**Giả thuyết cần kiểm:** mỗi client chỉ có **366–495 ảnh PlantDoc gốc**, nhưng sampler nhắm 25% lượt rút có hoàn lại, nên có thể lặp ảnh/cảnh; dấu bệnh ngoài đồng và ảnh PlantVillage khác miền; ảnh cắt vùng lá 224×224 có thể mất chi tiết. Chưa có phép đo cô lập tác động của từng yếu tố, vì vậy không quy mọi lỗi cho Dirichlet hoặc chất lượng ảnh.

Năm mô hình có cùng W0 nhưng **không cùng dữ liệu** và không tổng hợp. FedAvg nhận cập nhật từ cả năm shard, do đó so local-only với FedAvg nhằm lượng hóa lợi ích cộng tác dưới cùng partition, không để xếp hạng “kiến trúc nào tốt hơn”. [FedAvg gốc](https://proceedings.mlr.press/v54/mcmahan17a.html).

## 2. Việc nên làm theo thứ tự

### P0 — Bổ sung chứng cứ và tránh kết luận sai

1. Lưu mỗi vòng vào `history.jsonl` hoặc CSV: round, train loss, PlantDoc/PlantVillage F1, crop accuracy, LR, số lượt/cảnh duy nhất sampler rút. Hiện `summary.json` chỉ giữ mốc tốt nhất; biểu đồ [mốc train](figures/training_milestones.png) không phải đường cong. Ghi log bằng một dòng cập nhật/thanh tiến trình, giữ tương thích resume và kiểm không trùng round.
2. Báo cáo **per-client** theo nhãn từng client đã thấy/chưa thấy. Tính accuracy trên các nhãn seen và unseen riêng; không trình bày điểm unseen như khả năng nhận diện bệnh đã học. Kiểm ma trận nhầm lẫn PlantDoc, ví dụ sai cây so với đúng cây sai bệnh. Chụp lại ảnh lỗi và bbox gốc để rà chất lượng nhãn.
3. Tạo external test độc lập theo nhóm cảnh/địa điểm, giữ nhãn chuyên gia và nguồn ảnh. Tách khỏi validation; không dùng ảnh trên Internet chưa có ground truth để tính “độ chính xác thực tế”.

### P1 — Làm baseline so sánh có ý nghĩa

- Giữ **cùng W0 và cùng validation** cho local-only/FedAvg, cùng số ảnh gốc toàn chiến dịch. Với mỗi client, thêm baseline IID shard cùng cỡ `n_k` để tách tác động *thiếu nhãn* khỏi tác động mô hình. Tạo partition mới thì phải audit group-disjointness và lưu SHA, không sửa shard baseline `label_alpha_0_1`.
- `train_client` tạo SGD mới mỗi vòng nhưng giữ trọng số mô hình đã học, tổng cộng **60 lượt qua phần dữ liệu của client**. Với momentum baseline bằng 0 và learning rate cố định, việc tạo lại SGD tự nó **chưa phải nguyên nhân đã chứng minh** cho độ chính xác thấp. Nếu muốn thay đổi cách giữ trạng thái optimizer, chỉ thử ở run mới và so số học; ưu tiên phân tích thiếu nhãn và mất cân bằng dữ liệu trước.
- Đừng chỉ tăng từ 60 lên nhiều vòng: client 03 đạt checkpoint PlantDoc tốt nhất ở **vòng 11**, trong khi mọi client vẫn chạy đủ 60. Trước khi thử thêm epoch, xem đường cong mới ghi, train/val gap và overfit. Có thể thêm early stop **ở run mới** và báo tài nguyên tiết kiệm.
- Nếu cần tăng số client, tạo partition mới có số ảnh/tỷ lệ nhãn tối thiểu đủ kiểm. Trên cùng 37.238 ảnh, thêm client thường giảm dữ liệu mỗi client; không mặc định cải thiện local-only. Ghi rõ số lớp có trong từng client.

### P2 — Ảnh cây nhiều lá, độ sáng và độ mờ

MobileNetV3-Small ở đây là classifier **một crop → một nhãn**. Ảnh cả cây cần phát hiện lá, crop từng lá, chạy classifier rồi hiển thị kết quả theo lá. Với cảnh có nhiều bệnh, quy tắc gộp cấp cây phải được định nghĩa và kiểm riêng; không gán tất cả lá cùng một nhãn. PlantDoc có bbox phục vụ thử pipeline detector → classifier; [tài liệu TorchVision cho dataset bbox](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial).

So resize vuông 224 với giữ tỷ lệ rồi đệm viền và ảnh cắt lớn hơn; dùng cùng phép biến đổi khi train và suy luận. Thử ánh sáng, tương phản, blur, JPEG, che khuất theo mức độ và **đo hiệu năng từng client**; không có sẵn ngưỡng lux/blur/% che khuất từ cấu trúc mạng. Không tăng cường ảnh quá mạnh làm mất đốm bệnh. [TorchVision transforms](https://docs.pytorch.org/vision/stable/transforms.html) · [benchmark biến dạng ảnh](https://openreview.net/pdf?id=HJz6tiCqYm).

### P3 — Cân bằng và hiệu chuẩn có kiểm soát

- So sampler PlantDoc `25% with_replacement` với 15% hoặc `cycle_without_replacement` trên **từng client**, ghi số scene lặp. Không tăng tỷ lệ một cách mù quáng, đặc biệt client có chỉ vài trăm ảnh PlantDoc. Rà nhãn/bbox của lớp hiếm trước khi đổi loss.
- Fit temperature riêng trên calibration nếu định sử dụng mô hình local để trả xác suất; ngưỡng “không chắc” phải được kiểm trên external test. `softmax` cao chưa đảm bảo đúng. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).

## 3. Nghiệm thu và báo cáo

Một thay đổi chỉ được coi là cải thiện nếu so trên **cùng shard, split, W0**, có metric PlantDoc và PlantVillage cho *mỗi* client, gồm top-1, macro-F1 supported, đúng cây, seen/unseen labels, số ảnh/cảnh rút, thời gian, VRAM. Nếu tạo partition mới, đánh giá lại cả local-only và FedAvg trên partition đó rồi mới so. Chạy pilot trước full, thay **một biến mỗi lần**, lặp nhiều seed cho ứng viên. Không trộn trọng số của năm mô hình local và gọi đó là FedAvg nếu chưa thực hiện thuật toán tổng hợp và đánh giá model mới.

Đề xuất này không cam kết một mức tăng độ chính xác; các lợi ích chỉ có thể xác nhận sau thí nghiệm. Vấn đề miền ảnh thực tế được thảo luận trong [PlantDoc](https://arxiv.org/abs/1911.10317) và [nghiên cứu PlantVillage](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full).

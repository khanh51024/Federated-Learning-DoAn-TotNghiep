# Đề xuất cải tiến train tập trung v6

**Trạng thái:** đề xuất dựa trên code và kết quả, **chưa chạy thí nghiệm mới**. Baseline: Colab, 37.238 ảnh train, MobileNetV3-Small 38 lớp, seed 42, checkpoint epoch 22/83, giới hạn 100 epoch. [Kết quả chi tiết](../results/20261003/RESULTS.md) · [lịch sử epoch](../results/20261003/summary.json) · [runner](../train-tap-trung/scripts/train_production.py).

## 1. Điều đã thấy

- PlantDoc validation **53,90% top-1**, **53,47% macro-F1 supported**, **78,81% đúng cây**; PlantVillage **99,22%**, **98,99%**, **99,72%**. Tổng 96,57% top-1 bị PlantVillage chi phối vì 4.344/4.613 ảnh validation thuộc miền này. Không dùng điểm tổng làm bằng chứng đủ tốt ngoài đồng.
- PlantDoc có **57** ảnh sai cây, **67** ảnh đúng cây sai bệnh, **77** ảnh sai nhưng confidence ≥0,9. Cần kiểm lỗi loài cây, dấu bệnh và hiệu chuẩn xác suất riêng.
- Trong `summary.json`, checkpoint tốt nhất ở **epoch 22** nhưng chạy tới **83** rồi hội tụ sớm. Tăng số epoch lên 120 hoặc tắt early stopping **không có cơ sở** để cải thiện. Điểm train gần tuyệt đối trong khi PlantDoc validation còn thấp cho thấy khoảng cách lớn giữa train và validation; sampler lặp ảnh/cảnh và chênh miền là **giả thuyết cần kiểm định**, chưa xác nhận nguyên nhân.
- Pipeline hiện tại co mọi crop thành vuông 224×224, lật ngang lúc train; chưa có thử nghiệm định lượng theo ánh sáng, độ mờ hay che khuất. Không thể tuyên bố mô hình chịu được một mức lux hoặc blur cụ thể.

[PlantDoc gốc](https://arxiv.org/abs/1911.10317) được xây cho ảnh cây bệnh ngoài điều kiện phòng thí nghiệm. [Nghiên cứu PlantVillage](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full) đã ghi nhận giảm hiệu năng khi chuyển sang ảnh khác điều kiện thu thập. Đây là cơ sở để ưu tiên đánh giá theo miền.

## 2. Kế hoạch theo thứ tự ưu tiên

### P0 — Xác định bài toán đánh giá và tập lỗi

1. Lưu dự đoán `sample_id`, `group_id`, nhãn thật/đoán, 38 logit, confidence, nguồn và bbox cho **validation PlantDoc**. Tạo ma trận nhầm lẫn theo *cây* và theo *bệnh trong cùng cây*, lọc các cặp Tomato bacterial spot/Septoria, Potato late blight, Apple scab… có lỗi nổi bật trong metrics. Xem ảnh thật trước khi sửa nhãn.
2. Đánh giá theo **scene** ngoài theo crop. Scene PlantDoc có thể chứa nhiều lá; một ảnh toàn cây phải được phát hiện lá/crop và báo riêng từng lá, sau đó mới xây quy tắc tổng hợp cấp cảnh. So `bbox chuẩn + classifier` với `bbox dự đoán + classifier` để tách lỗi detector và classifier. [Hướng dẫn object detection TorchVision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial).
3. Tạo external test ảnh chụp/ảnh nguồn độc lập có nhãn chuyên gia và giấy phép dùng; khóa split trước khi thử thuật toán. Tách nhóm cảnh/nguồn khi chia để tránh cùng cây/lá xuất hiện nhiều phần.

### P1 — Thử đúng cách tiền xử lý và tỷ lệ lấy mẫu

- So resize vuông `canonical_v1` với resize giữ tỷ lệ và padding, rồi với crop độ phân giải **288/320 px** cho vết bệnh nhỏ. Kiến trúc có adaptive pooling nhưng **checkpoint v6 học ở 224 px**; phải fine-tune lại, đo VRAM/tốc độ và dùng cùng transform lúc suy luận. [Tài liệu MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
- Dữ liệu gốc PlantDoc là **2.115/37.238 = 5,68%**, còn sampler rút mục tiêu **25% lượt học** bằng cách lấy có hoàn lại. So 15% và 25%, `with_replacement` và `cycle_without_replacement`; ghi số ảnh/cảnh duy nhất và số lượt lặp. Tăng tỷ lệ không mặc nhiên tốt vì PlantDoc có ít cảnh.
- Dùng augmentation **có phạm vi đo được**: sáng/tương phản nhẹ, JPEG, blur nhẹ, hình học vừa phải. Thiết kế “corruption sweep” theo mức độ, báo top-1/F1 trên ảnh sạch và từng mức. Tránh augment xóa đốm bệnh hoặc đổi màu dấu bệnh tới mức sai nhãn. [TorchVision transforms](https://docs.pytorch.org/vision/2.0/transforms.html) · [benchmark nhiễu ảnh](https://openreview.net/pdf?id=HJz6tiCqYm).
- Với ảnh mờ/thiếu sáng/che khuất: đo phân tầng trước, rồi chọn ngưỡng cảnh báo trên calibration. **Không có ngưỡng vật lý cố hữu của MobileNetV3-Small**; độ chịu lỗi phụ thuộc ảnh, tần suất kiểu lỗi trong train và phiên bản preprocessing.

### P2 — Cải thiện học nhãn bệnh trong cùng loại cây

- Dùng phân tích nhầm lẫn để thử loss theo nhánh: đầu ra cây và bệnh riêng hoặc loss phụ cho cây, nhưng kiểm mapping 38 nhãn để bệnh không bị gán cho cây không hỗ trợ. Đây là thí nghiệm kiến trúc mới, phải so với baseline 38-logit dưới cùng split/W0 và chi phí train. Không suy từ crop accuracy 78,81% rằng head phân cấp chắc chắn tăng F1.
- Với lớp ít ảnh PlantDoc, xem lại chất lượng nhãn/bbox và cân bằng theo lớp **sau** khi đã cân bằng nguồn; tránh kết hợp nhiều trọng số và sampling mạnh gây lặp cùng ảnh quá mức. Có thể thử loss có trọng số, focal hoặc mixup ở pilot, mỗi lần **một biến**. Mixup có thể làm nhòe dấu bệnh rất nhỏ nên phải kiểm ảnh minh họa và per-class recall. [Nghiên cứu mixup](https://openreview.net/pdf?id=r1Ddp1-Rb).
- Kiểm vùng chú ý Grad-CAM trên tập ảnh đúng/sai để phát hiện dựa vào nền hoặc vùng ngoài lá; sau đó mới cân nhắc mask/crop nền hoặc domain adaptation. Grad-CAM là công cụ chẩn đoán, không chứng minh nhân quả. [Bài báo Grad-CAM](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) · [domain adaptation](https://www.jmlr.org/papers/v17/15-239.html).

### P3 — Hiệu chuẩn confidence và chọn checkpoint

Fit một hệ số nhiệt độ T trên **calibration 4.225 ảnh** rồi đo NLL/ECE/Brier theo PlantDoc và PlantVillage. `softmax(logits/T)` với T>0 không đổi top-1 nhưng có thể cải thiện ý nghĩa xác suất; kiểm trên test khóa kín. [Guo và cộng sự, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).

Kiểm lịch sử 83 epoch: checkpoint epoch 22 được chọn theo chất lượng PlantDoc có ngưỡng bảo vệ PlantVillage. Nếu thay quy tắc chọn, tạo run mới; không chọn epoch bằng test hoặc ảnh ngoài mạng. Xuất `sampler_epoch_stats`, learning rate, train/val loss và PlantDoc/PlantVillage F1 trong đồ thị để nhìn rõ sự phân kỳ. Biểu đồ baseline hiện tại: [đường cong](../results/20261003/centralized_training_curve.png).

## 3. Tiêu chí nghiệm thu thí nghiệm

Giữ nguyên release, W0, validation manifest và cùng ngân sách tài nguyên cho từng ablation. So **PlantDoc macro-F1 supported, crop accuracy**, PlantVillage macro-F1, per-class recall, metric cấp scene, tỷ lệ sai tự tin, độ trễ suy luận và VRAM. Không dùng validation để vừa thử nhiều cấu hình vừa tuyên bố độ chính xác thực tế; xác nhận ứng viên trên external test chưa đụng tới. Sau pilot, lặp nhiều seed và công bố biến thiên. Mọi số tăng/giảm chỉ được viết sau khi có kết quả, không suy diễn từ tài liệu tham khảo.

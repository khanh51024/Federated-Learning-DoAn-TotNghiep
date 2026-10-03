# Đề xuất cải tiến FedAvg v7 theo kết quả đã đo

**Trạng thái:** tài liệu nghiên cứu, **chưa sửa runner hoặc train lại**. Baseline cố định: `label_alpha_0_1`, 5 client, seed 42, checkpoint global vòng 27, kết thúc vòng 32, release `pv_pd_v3`. Nguồn số: [metrics](../results/20261003/validation_metrics.json), [spec](../configs/quality_spec_v4_FEDAVG_FULL.json), [runner](../train-gd-2/kaggle-workspace/campaign-mixed-pv-pd-v1/fedavg_mixed_runner.py).

Phiên này mô phỏng năm client **tuần tự trong một tiến trình**, với dữ liệu tập trung trong môi trường Kaggle. Các kết quả dưới đây không đo quyền riêng tư, chi phí truyền mạng hay khả năng vận hành trên thiết bị phân tán thật.

## 1. Triệu chứng quan sát và giới hạn suy luận

- PlantDoc validation: **47,21% top-1**, **46,87% macro-F1 supported**, **75,46% đúng cây**; PlantVillage: **93,12%**, **90,19%**, **98,69%**. Chênh lệch lớn giữa hai nguồn là bằng chứng về **hiệu năng khác miền**, nhưng chưa chứng minh nguyên nhân là nền, ánh sáng hay lỗi nhãn.
- Trên 269 ảnh PlantDoc, sai cây **66**, đúng cây sai bệnh **76**, sai mà điểm cao nhất ≥0,9 có **41**. Bệnh cần phân biệt tinh tế hơn sau khi nhận đúng cây; xác suất cao chưa đáng tin nếu chưa hiệu chuẩn.
- PlantDoc gốc chỉ **2.115/37.238 (5,68%)**; sampler tăng lên khoảng **25% lượt rút**, có hoàn lại. α = 0,1 làm client chỉ thấy **20–27/38** lớp và chênh `n_k` **5.376–10.075**. Việc lặp cảnh và lệch nhãn có thể góp phần vào lỗi, cần ablation mới kết luận.
- Chưa có dữ liệu từng vòng trong gói công bố, chưa có thử nghiệm độ sáng, mờ, che khuất, kích thước lá hoặc ảnh Internet độc lập. **Không được đưa ra ngưỡng chịu lỗi bằng lux, sigma blur hay % che khuất** từ hiện trạng.

PlantVillage vốn chủ yếu là lá đơn điều kiện kiểm soát, còn PlantDoc nhắm ảnh trong điều kiện thực tế hơn; phân bố khác nhau đã được mô tả trong [bài báo PlantDoc](https://arxiv.org/abs/1911.10317) và [nghiên cứu PlantVillage gốc](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full). Đây là lý do cần báo cáo tách nguồn, không dùng top-1 tổng làm chỉ số duy nhất.

## 2. Ưu tiên thử nghiệm, không sửa lan sang baseline

### P0 — Đo đúng lỗi trước khi chỉnh thuật toán

1. Xuất dự đoán theo `sample_id`, `group_id`, nguồn, nhãn thật/đoán, độ tin cậy dự đoán và bbox của vùng lá; dựng ma trận nhầm lẫn **PlantDoc riêng**, phân nhóm *sai cây*, *đúng cây sai bệnh*. Kiểm tra thủ công mẫu đại diện, nhất là Tomato bacterial spot/early blight/Septoria, Potato early/late blight và Grape black rot/Esca. Không tự động coi ảnh sai là lỗi nhãn.
2. Từ ảnh scene gốc PlantDoc, đánh giá hai điều kiện riêng: **bbox chuẩn → classifier** và **detector dự đoán bbox → classifier**. Chênh lệch cho biết phần lỗi do định vị lá. Với ảnh nhiều lá, trả dự đoán từng lá và thống kê cấp cảnh; không ép cả ảnh thành một nhãn.
3. Tạo bộ ảnh ngoài mạng có giấy phép/nguồn và nhãn chuyên gia, giữ tách hẳn khỏi train/tuning. Báo cáo theo loại cây, bệnh, nguồn/cảnh; chỉ gọi đó là external test sau khi bộ này được cố định.

### P1 — Cải thiện ảnh đầu vào và giảm lặp vô ích

- So `canonical_v1` với **giữ tỷ lệ rồi padding** và với **crop lá độ phân giải cao hơn** (ví dụ 288 hoặc 320 px), đồng nhất train/inference. MobileNetV3-Small có adaptive pooling nhưng chi phí GPU, bộ nhớ và kích thước đặc trưng tăng; cần pilot trên cùng ảnh và đo tốc độ. Với lá nhỏ trong ảnh toàn cây, thử detector/crop nhiều lá trước classifier, đánh giá bbox và nhãn riêng. [TorchVision detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial).
- So sampler 25% `with_replacement` với `cycle_without_replacement` và 15%/25% PlantDoc **từng biến một**. Ghi số ảnh/cảnh duy nhất được rút mỗi epoch, số lần lặp tối đa, recall từng lớp. Tăng tỷ lệ PlantDoc quá mức có thể overfit cảnh ít.
- Thử tăng cường ảnh có kiểm soát: biến thiên sáng/tương phản vừa phải, Gaussian blur nhẹ, JPEG, thay đổi góc hoặc vùng cắt nhưng **không xóa dấu bệnh hoặc đổi nhãn**. Đo trước/sau trên ảnh sạch và trên dãy mức độ biến dạng cố định. [TorchVision transforms](https://docs.pytorch.org/vision/stable/transforms.html) · [benchmark nhiễu ảnh](https://openreview.net/pdf?id=HJz6tiCqYm).
- Phân tầng kết quả theo ánh sáng, blur, che khuất và kích thước vùng lá. **Chỉ sau phép thử này** mới đặt ngưỡng cảnh báo chất lượng ảnh; phải ghi phương pháp đo (ví dụ Laplacian variance cho blur) và tỷ lệ lỗi theo từng mức, không suy ra một ngưỡng “khả năng tiếp nhận” từ cấu trúc mạng.

### P2 — Giảm lệch client trong FedAvg

- Giữ baseline 5 client để đối chiếu; **tăng số client đòi hỏi tạo partition/audit mới**. Nhiều client hơn trên cùng 37.238 ảnh thường làm mỗi shard nhỏ hơn, không mặc định tăng độ chính xác. So IID, `label_alpha_0_1`, feature-skew đã audit trên cùng W0, split và ngân sách tính toán; không gọi quantity-skew thuần túy nếu audit chưa đạt.
- Thử FedProx với phạt `(μ/2)||w−w_t||²` ở client; so μ=0 (FedAvg) với một vài giá trị pilot. Hoặc thử SCAFFOLD để hạn chế *client drift*. Chúng **chưa được áp dụng** trong checkpoint hiện tại. [FedProx](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) · [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html).
- Kiểm BatchNorm theo miền: hiện code **gộp running statistics bằng `n_k`**, còn bộ đếm nguyên được cộng delta. Với thử nghiệm **feature skew** mới, có thể so FedBN giữ BatchNorm local; phải thiết kế rõ cách suy luận global khi không biết miền client. Không áp FedBN mặc định cho baseline lệch nhãn. [FedBN](https://openreview.net/pdf?id=6YEQUn0QICG).
- Nếu chuyển sang Flower, cố định phiên bản được test, viết phép so tensor sau **1 vòng** giữa runner hiện tại và `FedAvg` Flower với `weighted_by_key="num-examples"`. So số ảnh gốc, thứ tự lớp, BatchNorm, checkpoint resume, AMP và cùng W0. Chỉ chuyển khi sai khác số học được giải thích; không ghi checkpoint v7 này là “đã chạy Flower”. [Tài liệu Flower](https://flower.ai/docs/framework/1.36/en/how-to-use-strategies.html).

### P3 — Độ tin cậy khi suy luận

Hiệu chuẩn nhiệt độ trên tập **calibration 4.225 ảnh** được tách riêng theo manifest; đo ECE, Brier hoặc NLL, đặc biệt PlantDoc. Việc chia logit cho T không đổi top-1 nếu T>0; mức độ tin cậy dự đoán có thể cải thiện nhưng phải đo. Sau đó chọn ngưỡng “không chắc” trên calibration và xác nhận trên bộ test ngoài miền sau khi xây dựng và khóa nhãn; không coi `softmax≥0,9` là chứng nhận đúng. [Nghiên cứu calibration](https://proceedings.mlr.press/v70/guo17a.html). Dùng Grad-CAM để kiểm liệu vùng được chú ý nằm trên lá hay nền, như công cụ chẩn đoán chứ không là bằng chứng nhân quả. [Grad-CAM](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).

## 3. Cách quyết định một thay đổi có tốt hơn

Giữ nguyên release, manifest validation, seed/W0 và giới hạn tài nguyên trong mỗi cặp so sánh; train pilot trước full. Báo cáo **PlantDoc macro-F1 supported và crop accuracy**, PlantVillage macro-F1 để tránh hy sinh miền cũ, thêm per-class recall, đúng cấp cảnh, tỷ lệ tự tin sai, thời gian và VRAM. Nếu đổi kích thước ảnh, nhãn, detector hoặc partition, lập phiên bản thí nghiệm mới; không ghi đè checkpoint v7. Lặp nhiều seed sau khi chốt ứng viên; khoảng biến thiên giữa seed phải được báo cáo thay vì công bố một mức tăng chưa kiểm chứng.

**Ưu tiên thực tế:** P0 phân tích lỗi/cảnh → P1 crop, sampler và robustness theo từng ablation → P2 thử FedProx/Flower khi đã có metric tin cậy → P3 hiệu chuẩn trước triển khai. Không có đề xuất nào ở đây được khẳng định chắc chắn sẽ tăng độ đúng nếu chưa qua thí nghiệm.

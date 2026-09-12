# Tài liệu tham khảo

Đánh số dùng chung với `STAGE2_REPAIR_PLAN.md`. Truy cập và kiểm tra nguồn web ngày 09/09/2026. Không tải lại toàn bộ bài báo vào repository; các tệp Markdown này ghi thông tin thư mục, liên kết nguồn và ghi chú áp dụng.

[1] “Nghiên cứu và xây dựng hệ thống Học liên kết (Federated Learning) cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, không đồng nhất,” đề cương đề tài tốt nghiệp, tài liệu nội bộ, không ghi tác giả và năm, tr. 1–4. Tệp PDF gốc nằm tại thư mục gốc workspace; yêu cầu GĐ2 ở tr. 3, mục 5; trục so sánh ở tr. 1–3. Không suy đoán thông tin xuất bản còn thiếu.

[2] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in *Proc. 20th Int. Conf. Artificial Intelligence and Statistics*, PMLR, vol. 54, 2017, pp. 1273–1282. [Online]. Available: https://proceedings.mlr.press/v54/mcmahan17a.html. [Accessed: Sep. 9, 2026].

[3] T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification,” arXiv:1909.06335, 2019, doi: 10.48550/arXiv.1909.06335. [Online]. Available: https://arxiv.org/abs/1909.06335. [Accessed: Sep. 9, 2026].

[4] Flower Labs, “Run simulations,” *Flower Framework Documentation*. [Online]. Available: https://flower.ai/docs/framework/how-to-run-simulations.html. [Accessed: Sep. 9, 2026].

[5] Flower Labs, “run_simulation,” *Flower Framework Documentation*, ver. 1.36.x. [Online]. Available: https://flower.ai/docs/framework/ref-api/flwr.simulation.run_simulation.html. [Accessed: Sep. 9, 2026].

[6] PyTorch Contributors, “Reproducibility,” *PyTorch Documentation*, ver. 2.6. [Online]. Available: https://docs.pytorch.org/docs/2.6/notes/randomness.html. [Accessed: Sep. 9, 2026].

[7] PyTorch Contributors, “Saving and Loading Models,” *PyTorch Tutorials*. [Online]. Available: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html. [Accessed: Sep. 9, 2026].

[8] Torchvision Contributors, “mobilenet_v3_small,” *Torchvision Documentation*. [Online]. Available: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html. [Accessed: Sep. 9, 2026].

[9] PyTorch Contributors, “torch.optim.lr_scheduler,” source code, class `ReduceLROnPlateau`, *PyTorch Documentation*, ver. 2.6. [Online]. Available: https://docs.pytorch.org/docs/2.6/_modules/torch/optim/lr_scheduler.html. [Accessed: Sep. 9, 2026].

[10] Scikit-learn Developers, “precision_recall_fscore_support,” *Scikit-learn Documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html. [Accessed: Sep. 10, 2026].

[11] PyTorch Contributors, “Automatic Mixed Precision examples,” *PyTorch Documentation*, ver. 2.6. [Online]. Available: https://docs.pytorch.org/docs/2.6/notes/amp_examples.html. [Accessed: Sep. 10, 2026].

[12] Ray Contributors, “ray.init,” *Ray Documentation*. [Online]. Available: https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html. [Accessed: Sep. 10, 2026].

## Phạm vi sử dụng và giá trị hiện tại

- [1] xác định yêu cầu cần nghiệm thu. GĐ2 không đòi triển khai FedProx/FedBN/SCAFFOLD/MOON; đó là GĐ3.
- [2] là nguồn gốc FedAvg, vẫn dùng để kiểm tra tổng hợp theo số mẫu và phân biệt local epoch với communication round. Không dùng năm xuất bản cũ để suy ra API phần mềm hiện hành.
- [3] hỗ trợ thiết kế khảo sát nhiều mức không đồng nhất trong phân loại ảnh. Đây là preprint và thí nghiệm trên CIFAR-10; không chuyển trực tiếp độ chính xác, mức cải thiện hay siêu tham số của bài sang PlantVillage.
- [4] hỗ trợ kiểm soát tài nguyên mô phỏng và lựa chọn WSL2 khi Ray gặp vấn đề trên Windows. Giới hạn tài nguyên Ray là cơ chế lập lịch, không bảo đảm tiến trình chỉ dùng đúng lượng bộ nhớ đó.
- [5] dùng để đối chiếu API với Flower 1.36.0 đang cài. Log thực tế còn cảnh báo `run_simulation` deprecated; giữ phiên bản đã kiểm chứng trong bản sửa nhỏ, tách việc chuyển sang `flwr run` thành bước tương thích riêng.
- [6] dùng cho seed Python/NumPy/PyTorch, DataLoader và phạm vi tái lập. Không cam kết kết quả bitwise giống nhau giữa CPU/GPU hoặc các phiên bản phần mềm.
- [7] dùng cho checkpoint và trạng thái cần thiết khi resume. Chính sách optimizer cục bộ reset mỗi round của dự án phải được ghi rõ; không gọi việc khôi phục LR là khôi phục toàn bộ optimizer.
- [8] dùng cho MobileNetV3-Small, pretrained weights, resize/crop/normalization và tắt progress tải weights phụ. Trang stable đã mới hơn Torchvision 0.21.0 cài tại máy, nên khi sửa phải đối chiếu chữ ký API trong `.venv`.
- [9] dùng cho semantics `threshold_mode='abs'`, `patience` và LR của round kế tiếp. Không cần thay controller đang đúng chỉ để đổi phong cách code.
- [10] dùng để đối chiếu accuracy, precision/recall/F1, macro/weighted averaging với 38 nhãn cố định và `zero_division=0`; không bỏ lớp thiếu support khỏi mẫu số macro.
- [11] xác nhận GradScaler có thể bỏ bước SGD khi gradient không hữu hạn. Log phải phân biệt bước cập nhật thực tế và bước bị bỏ qua; loss hữu hạn chưa đủ để suy ra SGD đã chạy.
- [12] hỗ trợ cấu hình `object_store_memory` theo byte. Mặc định 256 MiB trong dự án là lựa chọn vận hành đã thử trên máy này, không phải tổng RAM Ray được phép dùng. Đối chiếu API cài tại máy (Ray 2.55.1) vì trang latest đã mới hơn.

Các giá trị batch size, local epochs, patience, alpha và số seed trong kế hoạch là cấu hình thử nghiệm đề xuất, không phải kết quả tối ưu được các nguồn bảo đảm.

# DEPLOY — bàn giao Q7

## 2.1 Vai trò và kết quả

Centralized là model web app (trần, chọn trước). Ba checkpoint FedAvg là ba kết quả kịch bản ngang hàng; không chọn một alpha làm “sản phẩm FL”. Cùng một pipeline áp dụng cho cả 19 checkpoint. Số liệu Q6 test mô tả với một seed, không xếp hạng phương pháp; Centralized dùng AdamW, client dùng SGD. Headline luôn là cặp accuracy toàn tập + accuracy PlantDoc, kèm macro-F1 PlantDoc.

| Nhãn | Vai trò | SHA256 checkpoint | Epoch/round | T | Acc toàn tập | Acc PV | Acc PD | PD macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| central | web app model (ceiling, fixed in advance) | `c45fa9937b636d815ef99f4bd21f078cad86521cf67a5b118f7c64e0a7809998` | 22 | 2.9258 | 0.9867 | 0.9923 | 0.6337 | 0.5685 |
| fedavg_iid | FedAvg scenario result: iid | `bd5a56151904f8b0b7adfd59aa38f63e741ed23ffe9c46b7a4c7a4dadd4cb238` | 13 | 1.0938 | 0.9629 | 0.9702 | 0.5000 | 0.5266 |
| fedavg_a10 | FedAvg scenario result: label alpha 1.0 | `085701c3d19d3106167d4eb8a7d943880e5077e07b0766b5f1a8409937d58e42` | 6 | 0.8415 | 0.9379 | 0.9450 | 0.4884 | 0.5400 |
| fedavg_a01 | FedAvg scenario result: label alpha 0.1 | `2031d4d291b69c391065febe5e6a7c0da7d91e25a8efd8490ffac24d77c74941` | 27 | 1.1481 | 0.9266 | 0.9334 | 0.5000 | 0.5435 |

## 2.2 Model và tải trọng số

torchvision.models.mobilenet_v3_small(weights=None), classifier[-1] = Linear(1024, 38), load_state_dict(strict=True), model.eval(), torch.no_grad(). Khóa state: `model_state` cho Centralized và FedAvg; `model_state_dict` cho Local-only. Với checkpoint train, xác minh SHA256 trước, rồi `torch.load(path, map_location="cpu", weights_only=False)`. Unpickling chỉ an toàn khi SHA khớp. Ưu tiên file `deploy/<label>_state_dict.pt`: xác minh SHA trong handover.json rồi tải bằng `weights_only=True`.

## 2.3 Tiền xử lý chung

Theo đúng thứ tự: PIL.Image.open → ImageOps.exif_transpose → nếu RGBA/LA hoặc P có transparency thì chuyển RGBA và paste trên nền RGB đen bằng alpha mask; các mode khác chuyển RGB → torchvision Resize((224,224), BILINEAR, antialias=True) trên PIL (ép vuông, không giữ tỉ lệ, không padding) → ToTensor → Normalize mean (0.485,0.456,0.406), std (0.229,0.224,0.225) → batch 1. EXIF cần xử lý vì dataset train đã xử lý. Bảng field_aug_v1 / aspect_pad trong PLAN là lịch sử; không checkpoint Q5 nào dùng chúng.

## 2.4 Đầu ra và hiệu chỉnh

`p = softmax(logits / T)` với T ở temperatures.json, fit chỉ trên calibration, max_iter=500 theo Q6. Phần trăm hiển thị là độ tự tin của mô hình cho dự đoán, không phải accuracy.

## 2.5 Ngưỡng từ chối

Chưa cung cấp ngưỡng (phương án a). Ngưỡng tương lai phải đăng ký trước và fit chỉ trên calibration; cấm dùng test hoặc S4-clean96 (cùng 96 cảnh test).

## 2.6 Giới hạn

Accuracy PD chỉ 0,49–0,63 trên 172 crop / 13 lớp; mỗi nhánh một seed. Khoảng cách PD Centralized–FedAvg 13–15 pp đã tồn tại ở IID, không quy cho non-IID. Không phát biểu FedAvg tiệm cận Centralized trên ảnh thực tế. FedAvg α=1,0 có T<1, tăng tự tin và ECE PD xấu đi; ghi nhận, không sửa kết quả.

## 2.7 FedAvg

Đây là mô phỏng trong một tiến trình với 5 client (`fedavg_mixed_runner.py`), không phải triển khai Flower (V-1).

## 2.8 Web app hiện có

`train-tap-trung/web_app/per_leaf_engine.py` là file chưa commit và Q7 chỉ đọc. Giao diện: `PerLeafDualInferenceEngine(detector_path, centralized_path, federated_path)`, `predict_per_leaf`, `_classify_crop`. Mã hiện tại dùng softmax không T; ngưỡng top1/margin classifier và ngưỡng detector có nguồn dữ liệu **unknown**; detector tạo crop lá với biên 8%. Resize classifier `Resize((224,224))` mặc định có thể tương đương BILINEAR trên PIL (luôn antialias), nhưng chưa kiểm nghiệm: **unknown**. Ngưỡng web đặt trên xác suất chưa hiệu chỉnh; thêm T (Centralized T≈2,93) đổi tỉ lệ từ chối. Tích hợp T vào web là việc riêng cần người dùng phê duyệt.

## 2.9 T trong trainer

`train_production --prepare-calibration` fit T với max_iter=50, có thể chưa hội tụ. Dùng `temperatures.json`.

## 2.10 Trọng số xuất

Mỗi file `<label>_state_dict.pt` là bản clone từng tensor của state key tương ứng. G5 so khóa, dtype, `torch.equal` với checkpoint gốc sau khi tải `weights_only=True`; SHA256 ghi trong handover.json.

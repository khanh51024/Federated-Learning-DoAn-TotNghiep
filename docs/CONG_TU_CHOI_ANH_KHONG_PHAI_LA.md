# Cổng từ chối ảnh không phải lá cây

Mục tiêu "nhận ảnh bất kỳ" chỉ có nghĩa khi hệ thống từ chối được ảnh không thuộc phạm vi 38 lớp. Hiện mô hình luôn trả một tên bệnh, kể cả khi ảnh là xe, người hay phong cảnh.
Tài liệu này ghi phép đo bốn cổng từ chối trên tập ảnh không phải lá. Chưa có cổng nào được đưa vào ứng dụng web và chưa huấn luyện lớp nào.

## 1. Cách đo

- **Tập ảnh đúng (cần nhận)**: lá PlantWild bệnh (2.304) và lá khỏe (2.741), nguồn chưa từng thấy; lá PlantDoc kiểm tra và giữ riêng (441).
- **Tập ảnh sai (cần từ chối)**: 1.513 ảnh Wikimedia Commons (giấy phép mở, ghi tác giả và URL từng ảnh trong `data/external/ood/metadata.jsonl`; tải bằng `scripts/fetch_ood_images.py`):
  - nhóm `general` (1.285 ảnh): xe, người, động vật, nhà cửa, trong nhà, thức ăn, quả và rau, hoa, phong cảnh, giấy tờ, đất đá, và 170 ảnh Wikimedia đã bị loại ở bước kiểm nhãn (tàu hỏa, máy bay, logo, ảnh chó nghiệp vụ);
  - nhóm `other_leaf` (228 ảnh): lá của loài **ngoài** 38 lớp (sồi, phong, hoa hồng, chuối, dương xỉ...), nhóm khó; chỉ báo cáo, không có chỉ tiêu.
  Manifest khóa bằng SHA-256 `59d4aaa6293d1b9550c76145d1ad36aa4c5c20d43b442ff7c222667f5abf0cb2` (`artifacts/ood_eval_set.json`). Các ảnh này không bao giờ dùng để học.
- **Chỉ tiêu đặt trước khi đo**: ở ngưỡng nhận đúng ≥ 95% ảnh lá, phải từ chối ≥ 90% ảnh `general`.
- **Chọn ngưỡng và báo kết quả tách rời**: mỗi tập chia đôi theo mã băm của ảnh; ngưỡng chọn trên nửa hiệu chỉnh, mọi số dưới đây là của nửa báo cáo (2.727 ảnh lá, 657 ảnh `general`, 101 ảnh `other_leaf`).
- Bốn cổng, điểm càng cao càng giống "lá của 38 lớp":
  - `msp_centralized`, `msp_fedavg`: xác suất lớn nhất của mô hình đã công bố (cách web hiện gắn cờ "không chắc" khi xác suất dưới 50%);
  - `maxlogit_centralized`: logit lớn nhất của Centralized;
  - `dino_knn_sim`: độ tương tự cosine trung bình của 5 láng giềng gần nhất trong thư viện v3 (PlantVillage 40 ảnh mỗi lớp cộng lá PlantDoc), đặc trưng DINOv2-small đóng băng;
  - `bioclip_leaf`: xác suất BioCLIP zero-shot của nhóm câu "ảnh lá" so với các câu loại, dùng đúng bộ câu của `scripts/filter_leaf_photos.py` (thiết kế để lọc ảnh iNaturalist, chưa thử bộ câu khác).

## 2. Kết quả

Số liệu từ `artifacts/ood_gate_eval.json`, nửa báo cáo. Khoảng tin cậy 95% (Wilson) ở trong tệp.

| Cổng | AUROC (lá so với `general`) | AUROC (lá so với `other_leaf`) | Nhận lá | Loại `general` | Loại `other_leaf` | Đạt chỉ tiêu |
|---|---:|---:|---:|---:|---:|:---:|
| msp_centralized | 0,569 | 0,518 | 96,1% | 7,6% | 4,0% | không |
| msp_fedavg | 0,596 | 0,538 | 94,8% | 8,4% | 8,9% | không |
| maxlogit_centralized | 0,669 | 0,584 | 95,3% | 16,0% | 4,0% | không |
| **dino_knn_sim** | **0,981** | 0,782 | 95,5% | **93,0%** (90,8–94,7) | 41,6% | **có** |
| bioclip_leaf | 0,559 | 0,482 | 94,7% | 2,1% | 4,0% | không |

Cột "Nhận lá", "Loại `general`" và "Loại `other_leaf`" ở ngưỡng chọn để nhận 95% ảnh lá trên nửa hiệu chỉnh.

- **Độ tự tin của mô hình hiện tại không dùng được làm cổng.** Với ngưỡng nhận 95% ảnh lá, xác suất lớn nhất chỉ loại 7,6% (Centralized) và 8,4% (FedAvg) ảnh không phải lá. Muốn loại 90% ảnh không phải lá thì ngưỡng
  phải đặt rất cao (Centralized 0,9996; FedAvg 0,92) và chỉ còn nhận 18,9% (Centralized) và 22,4% (FedAvg) ảnh lá. Mô hình rất tự tin trên ảnh nó chưa từng học.
- **`dino_knn_sim` đạt chỉ tiêu** trên nửa báo cáo: nhận 95,5% ảnh lá (94,7–96,2) và loại 93,0% ảnh `general`. Ở ngưỡng chọn để loại 90% ảnh `general`, cổng nhận 98,0% ảnh lá.
- **`bioclip_leaf` với bộ câu hiện có không phân biệt được** (AUROC 0,559; ngưỡng suy biến về 0). Bộ câu loại chỉ gồm quả, hoa, cây cả... nên không bao quát xe, người, nhà cửa; chưa thử bộ câu khác, nên chưa kết luận về BioCLIP nói chung.
- Chi tiết của `dino_knn_sim` ở ngưỡng nhận 95% ảnh lá (nửa báo cáo):

| Nhóm ảnh lá | Nhận | | Chủ đề không phải lá | Loại |
|---|---:|---|---|---:|
| PlantWild bệnh (1.151 ảnh) | 98,1% | | xe, người, nhà cửa, giấy tờ, thức ăn, trong nhà, phong cảnh | 100% |
| PlantWild lá khỏe (1.352) | 92,6% | | động vật | 98,3% |
| PlantDoc kiểm tra (132) | 100% | | đất đá | 91,1% |
| PlantDoc giữ riêng (92) | 100% | | hoa | 74,1% |
| | | | quả và rau | 68,6% |
| | | | lá loài khác: cây gỗ / cây vườn và rau thơm | 50,0% / 31,9% |

## 3. Giới hạn

1. **Lá của loài ngoài 38 lớp vẫn bị nhận nhầm** (loại 41,6%). Cổng "giống lá của 38 lớp" không biết loài nào nằm ngoài phạm vi; muốn loại các ảnh này cần dữ liệu lá loài khác trong thư viện hoặc huấn luyện thêm lớp "khác".
2. Hoa, quả và rau còn lọt qua 26–31%: nền lá xung quanh làm điểm tương tự cao.
3. Tập sai là ảnh Wikimedia (chất lượng tương đối tốt, cỡ 640 px). Ảnh người dùng thật thường mờ, lệch sáng, cắt dở; chưa đo.
4. Ngưỡng gắn với thư viện v3 và đặc trưng DINOv2-small. Khi đổi thư viện (v4) hoặc mô hình, phải chọn lại ngưỡng trên nửa hiệu chỉnh và đo lại; số ở đây **không** tự chuyển sang cấu hình khác.
5. Nửa báo cáo của nhóm `general` có 657 ảnh nên khoảng tin cậy còn rộng (±2 điểm); từng chủ đề chỉ có 35–87 ảnh.
6. Chưa đo chi phí triển khai: cổng cần chạy DINOv2-small (khoảng 22 triệu tham số) cùng thư viện đặc trưng cho mỗi ảnh gửi lên. Chưa đo RAM và độ trễ trên máy chủ Oracle dùng chung; chưa đưa vào web.

## 4. Việc tiếp theo nếu muốn đưa vào ứng dụng

1. Đo RAM và độ trễ của DINOv2-small (dạng ONNX) trên máy chủ; nếu quá nặng thì thử thư viện đặc trưng nhỏ hơn hoặc chạy cổng ở trình duyệt như phần học cộng đồng.
2. Thêm lá của loài ngoài 38 lớp vào tập "từ chối" khi hiệu chỉnh, và ảnh người dùng thật (đã được phép) vào tập đo.
3. Hiển thị "ảnh không giống lá của 38 lớp, không chẩn đoán" khi điểm dưới ngưỡng; ghi ngưỡng và SHA-256 của manifest vào cấu hình.
4. Báo cáo kèm tỉ lệ từ chối nhầm lá thật (hiện khoảng 4,5%) vì cổng làm giảm độ phủ của chẩn đoán.

## 5. Cách chạy lại

```
python scripts/fetch_ood_images.py                                        # tải ảnh Wikimedia không phải lá (ngoài git)
python scripts/eval_ood_gate.py prepare                                   # manifest + khóa SHA
python scripts/eval_ood_gate.py logits                                    # logit Centralized và FedAvg trên ảnh sai
tmp/venv-fm/Scripts/python.exe scripts/eval_ood_gate.py features          # đặc trưng BioCLIP và DINOv2 của ảnh sai
python scripts/eval_ood_gate.py evaluate                                  # ghi artifacts/ood_gate_eval.json
```

Chạy lần lượt, không song song: bước `features` nạp hai mô hình (từ 2 GB RAM trở lên); máy 16 GB đã từng buộc hệ thống dừng tác vụ nền khi chạy cùng lúc với việc tải nhiều GB.

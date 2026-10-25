# GĐ4: nén trọng số khi truyền và lượng tử hóa mô hình suy luận (2026-10-26)

Phần này trả lời hai câu hỏi của GĐ4 trong đề cương (tối ưu phía client, giảm chi phí truyền thông):

1. Gửi trọng số mô hình bằng ít byte hơn thì mô hình nhận được kém đi bao nhiêu?
2. Mô hình suy luận lượng tử hóa int8 nhỏ hơn bao nhiêu, nhanh hơn bao nhiêu trên CPU, và mất bao nhiêu độ chính xác?

Mọi số liệu đo trên CPU, không cần huấn luyện lại. Nguồn số: `artifacts/gd4_compression.json`, sinh bằng `scripts/eval_gd4_compression.py`.

## 1. Thiết lập đo

- **Mô hình:** Centralized và FedAvg của gói web `v4b_s42` (hạt giống 42), nạp qua đường đã kiểm SHA-256 của `web/demo_evidence.py`.
- **Tập đánh giá:** tập giữ riêng `legacy_diagnostic_test` của release `mixed_pv_pd_v4`, gồm 13.101 ảnh (PlantVillage 10.843, PlantDoc 172, ngoài đồng bộ khác 2.086). Tập này không dùng để chọn checkpoint cũng không dùng để chọn cấu hình nén.
- **Tiền xử lý:** giống lúc huấn luyện và trên web (`canonical_v1`, 224 × 224).
- **Máy đo:** laptop Intel Core i5-12500H (12 nhân, 16 luồng), Windows 11, PyTorch 2.14 (CPU), ONNX Runtime 1.31. Khi đo máy đang chạy các ứng dụng khác nên số đo độ trễ có dao động.

### 1.1. Phần A: nén trọng số khi truyền

- **Phạm vi nén:** chỉ nén các tensor số thực từ 2 chiều trở lên (trọng số conv, depthwise, tuyến tính). Bias, tham số và thống kê BatchNorm, cùng bộ đếm, được giữ nguyên dạng float32/int64; phần này rất nhỏ.
- **Các cách nén được so:**
  - fp16;
  - int8 đối xứng với một hệ số cho cả tensor;
  - int8 đối xứng với một hệ số cho mỗi kênh đầu ra;
  - hai cách trên áp lên **độ lệch** w − w0, trong đó w0 là trọng số khởi tạo (ImageNet cộng đầu phân loại theo hạt giống). Hai bên đều biết w0 nên không phải gửi lại; phía nhận dựng lại w0 + giải_lượng_tử(độ lệch).
- **Đây là phép dựng lại một lần** từ mô hình đã huấn luyện xong. Nó đo sai số do nén, chưa đo ảnh hưởng của việc nén ở **mỗi vòng** lên quá trình hội tụ của FedAvg (việc đó cần chạy huấn luyện trên GPU).

### 1.2. Phần B: lượng tử hóa mô hình suy luận

- **Định dạng:** xuất ONNX (opset 17), rồi lượng tử hóa tĩnh int8 theo định dạng QDQ: trọng số int8 theo kênh, đầu ra từng lớp uint8 theo cả tensor, hiệu chuẩn MinMax.
- **Ảnh hiệu chuẩn:** 256 ảnh rút ngẫu nhiên (hạt giống 0) từ split `calibration`, tách rời với tập đánh giá.
- **Độ trễ:** đo trên một ảnh, một luồng CPU, lấy trung vị của 100 lần chạy.
- **Ba biến thể:**
  - lượng tử hóa mọi lớp (mặc định của ONNX Runtime);
  - chỉ khối SE và lớp tuyến tính (Gemm);
  - khối SE, Gemm và các lớp pointwise ở nửa sau mạng.
  Hai biến thể sau được chọn bằng phép đo độ nhạy ở mục 4, trên 200 ảnh của split `val`, không dùng tập giữ riêng.

## 2. Kết quả A: nén trọng số khi truyền

Dung lượng gốc mỗi lượt gửi: 6,28 MB (6.275.944 byte, khớp số byte ghi trong `history.jsonl` của run FedAvg). Cột "Ảnh đổi dự đoán" đếm số ảnh có dự đoán khác mô hình gốc, trên 13.101 ảnh.

**Centralized v4b**

| Cách | Dung lượng gửi (MB) | So với fp32 | Top-1 chung | PlantVillage | PlantDoc | Ngoài đồng bộ khác | Ảnh đổi dự đoán |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 (gốc) | 6,28 | 100,0% | 97,69% | 99,32% | 59,88% | 92,38% | 0 |
| fp16 | 3,19 | 50,9% | 97,71% | 99,34% | 59,88% | 92,38% | 3 |
| int8, một hệ số mỗi tensor | 1,65 | 26,4% | 97,43% | 99,12% | 60,47% | 91,66% | 106 |
| int8, một hệ số mỗi kênh | 1,69 | 27,0% | 97,61% | 99,28% | 57,56% | 92,23% | 51 |
| int8 độ lệch w − w0, mỗi tensor | 1,65 | 26,4% | 97,72% | 99,35% | 59,30% | 92,43% | 14 |
| int8 độ lệch w − w0, mỗi kênh | 1,69 | 27,0% | 97,71% | 99,35% | 59,30% | 92,33% | 10 |

**FedAvg v4b**

| Cách | Dung lượng gửi (MB) | So với fp32 | Top-1 chung | PlantVillage | PlantDoc | Ngoài đồng bộ khác | Ảnh đổi dự đoán |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 (gốc) | 6,28 | 100,0% | 93,88% | 96,84% | 40,12% | 82,93% | 0 |
| fp16 | 3,19 | 50,9% | 93,89% | 96,85% | 40,12% | 82,93% | 5 |
| int8, một hệ số mỗi tensor | 1,65 | 26,4% | 93,53% | 96,25% | 40,12% | 83,80% | 319 |
| int8, một hệ số mỗi kênh | 1,69 | 27,0% | 93,44% | 96,29% | 41,86% | 82,89% | 302 |
| int8 độ lệch w − w0, mỗi tensor | 1,65 | 26,4% | 93,89% | 96,82% | 40,12% | 83,13% | 14 |
| int8 độ lệch w − w0, mỗi kênh | 1,69 | 27,0% | 93,89% | 96,82% | 40,12% | 83,08% | 10 |

Nhận xét:
- **fp16 giảm một nửa dung lượng gửi**, chỉ đổi 3–5 dự đoán trên 13.101 ảnh.
- **int8 trên trọng số tuyệt đối giảm dung lượng xuống khoảng 27%**, nhưng mất 0,1–0,4 điểm top-1 chung và đổi 51–319 dự đoán. FedAvg nhạy hơn Centralized.
- **int8 trên độ lệch so với w0 cũng giảm dung lượng xuống khoảng 27%** nhưng gần như không mất gì: đổi 10–14 dự đoán, top-1 chung chênh không quá 0,03 điểm.
  - Lý do: huấn luyện chỉ dịch trọng số một khoảng nhỏ so với w0, nên miền giá trị cần biểu diễn hẹp hơn nhiều và bước lượng tử nhỏ hơn.
  - Đây là cơ sở để dùng cách "gửi độ lệch, không gửi trọng số" khi nén ở mỗi vòng FedAvg. Hiệu quả trên cả quá trình huấn luyện chưa đo.
- **PlantDoc chỉ có 172 ảnh**, nên mỗi ảnh tương ứng 0,58 điểm. Các chênh lệch PlantDoc ở bảng trên chỉ là 1–4 ảnh, không đủ để kết luận cách nào tốt hơn ở nguồn này.

## 3. Kết quả B: lượng tử hóa mô hình suy luận

**Centralized v4b**

| Biến thể | Tệp (MB) | Top-1 chung | PlantVillage | PlantDoc | Ngoài đồng bộ khác | Ảnh đổi dự đoán | Độ trễ 1 ảnh, 1 luồng (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX fp32 | 6,24 | 97,69% | 99,32% | 59,88% | 92,38% | 0 | 6,48 |
| ONNX int8, mọi lớp | 1,89 | 13,07% | 11,48% | 6,40% | 21,86% | 11.360 | 8,44 |
| ONNX int8, khối SE + Gemm | 3,09 | 97,66% | 99,32% | 59,30% | 92,23% | 13 | 5,41 |
| ONNX int8, khối SE + Gemm + pointwise nửa sau | 2,12 | 97,66% | 99,30% | 59,30% | 92,33% | 43 | 5,18 |

**FedAvg v4b**

| Biến thể | Tệp (MB) | Top-1 chung | PlantVillage | PlantDoc | Ngoài đồng bộ khác | Ảnh đổi dự đoán | Độ trễ 1 ảnh, 1 luồng (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX fp32 | 6,24 | 93,88% | 96,84% | 40,12% | 82,93% | 0 | 5,77 |
| ONNX int8, mọi lớp | 1,89 | 9,98% | 7,95% | 1,74% | 21,24% | 11.748 | 7,83 |
| ONNX int8, khối SE + Gemm | 3,09 | 93,92% | 96,83% | 39,53% | 83,27% | 47 | 5,20 |
| ONNX int8, khối SE + Gemm + pointwise nửa sau | 2,12 | 93,81% | 96,71% | 40,70% | 83,13% | 138 | 5,14 |

Để tham chiếu, PyTorch fp32 trên cùng máy (một luồng): 24,91 ms (Centralized) và 17,25 ms (FedAvg).

Nhận xét:
- **Lượng tử hóa mọi lớp làm mô hình sụp đổ:** độ chính xác chỉ còn 10–13% và đổi gần 87–90% dự đoán, dù tệp nhỏ nhất (1,89 MB). Biến thể này còn **chậm hơn** bản fp32. Nhiều khả năng là do phải đổi qua lại giữa int8 và float32 quanh các lớp HardSwish và HardSigmoid của MobileNetV3; điểm này chưa kiểm bằng công cụ đo từng phép tính.
- **Lượng tử hóa có chọn lớp giữ nguyên độ chính xác:**
  - Biến thể khối SE + Gemm + pointwise nửa sau còn 2,12 MB (34% của 6,24 MB). Top-1 chung giảm 0,03 điểm (Centralized) và 0,07 điểm (FedAvg).
  - Biến thể khối SE + Gemm còn 3,09 MB (một nửa), top-1 gần như không đổi.
- **Không kết luận được về tốc độ:**
  - Các biến thể int8 có chọn lớp đo được 5,1–5,4 ms, so với 5,8–6,5 ms của bản fp32.
  - Hai mô hình cùng kiến trúc mà bản fp32 lệch nhau 5,77 và 6,48 ms, tức khoảng 12%. Chênh lệch của int8 nằm trong mức dao động đó.
  - Vậy chưa có bằng chứng int8 nhanh hơn trên máy này. Lợi ích chắc chắn là **dung lượng tệp**.
- **Chuyển sang ONNX nhanh gấp 3–4 lần PyTorch eager** trên CPU một luồng. Lợi ích về tốc độ của phía triển khai đến chủ yếu từ bước này, không phải từ int8.

## 4. Vì sao lượng tử hóa mọi lớp sụp đổ

**Bảng 4.1. Độ nhạy theo nhóm lớp.** Mô hình FedAvg v4b, 200 ảnh split `val` (mỗi ảnh thứ hai trong 400 ảnh đầu), cùng cấu hình QDQ của mục 1.2. Đây là phép đo dùng để chọn cấu hình, không phải số báo cáo. Mô hình có 1 lớp conv đầu, 11 lớp depthwise, 22 lớp pointwise, 18 lớp conv trong khối SE và 2 lớp Gemm.

| Lớp được lượng tử hóa | Top-1 trên 200 ảnh | Tệp (MB) |
|---|---:|---:|
| không lớp nào (fp32) | 78,0% | 6,24 |
| mọi lớp | 4,5% | 1,89 |
| chỉ lớp conv đầu | 5,0% | 6,25 |
| chỉ 11 lớp depthwise | 4,5% | 6,13 |
| chỉ 22 lớp pointwise | 69,5% | 5,16 |
| chỉ 18 lớp trong khối SE | 79,5% | 4,96 |
| chỉ 2 lớp Gemm | 78,5% | 4,38 |
| khối SE + Gemm | 78,0% | 3,09 |
| khối SE + Gemm + pointwise nửa đầu | 71,5% | 2,97 |
| khối SE + Gemm + pointwise nửa sau | 79,5% | 2,12 |

Đổi cách hiệu chuẩn không cứu được bản lượng tử hóa mọi lớp: Percentile 99,99% đạt 12,5%, Entropy 4,5%, định dạng QOperator 4,5%.

Nhận xét:
- **Hai nhóm làm hỏng mô hình là lớp conv đầu và các lớp depthwise.** Chỉ lượng tử hóa riêng một nhóm này là đủ làm mô hình sụp đổ. Các lớp pointwise ở nửa đầu mạng làm giảm khoảng 6–9 điểm. Khối SE, Gemm và pointwise ở nửa sau chịu được.
- **Nhiều khả năng lỗi đến từ phần đầu ra của các lớp, không phải trọng số:**
  - Phần A cho thấy chỉ lượng tử hóa trọng số (đầu ra vẫn float32) thì không sụp đổ.
  - Phần B lượng tử hóa cả đầu ra theo **một hệ số cho cả tensor**. Sau khi BatchNorm được gộp vào conv, các kênh của lớp depthwise có biên độ chênh nhau rất lớn, nên kênh biên độ nhỏ bị làm tròn về 0. Đây là hiện tượng đã được mô tả cho MobileNet (Nagel và cộng sự, "Data-Free Quantization Through Weight Equalization and Bias Correction", ICCV 2019).
  - Phép đo ở đây chưa tách riêng trọng số và đầu ra của từng lớp để kiểm chứng trực tiếp giả thuyết này.
- **Hướng để lượng tử hóa được cả các lớp nhạy cảm:** cân bằng trọng số giữa các lớp (cross-layer equalization, cùng bài báo trên) hoặc huấn luyện có tính đến lượng tử hóa (quantization-aware training, cần GPU). Cả hai chưa thử.

## 5. Đính chính kết luận ngày 2026-10-21

`docs/DANH_GIA_DU_LIEU_MO_HINH_FEDAVG_2026-10-21.md` viết "int8 phá hủy mô hình". Kết luận đó dựa trên `scripts/audit_weight_exchange.py`, vốn lượng tử hóa int8 một hệ số mỗi tensor lên **mọi** tensor số thực, kể cả thống kê BatchNorm.

**Bảng 5.1.** Kiểm lại trên gói v4b, 200 ảnh `val` như Bảng 4.1, int8 một hệ số mỗi tensor (không phải số báo cáo):

| Tensor bị lượng tử hóa | Centralized | FedAvg |
|---|---:|---:|
| không tensor nào | 92,5% | 78,0% |
| mọi tensor số thực (như audit cũ) | 3,5% | 5,5% |
| chỉ trọng số từ 2 chiều trở lên | 92,0% | 80,0% |
| chỉ `running_var` của BatchNorm | 4,0% | 4,0% |
| chỉ `running_mean` của BatchNorm | 92,5% | 79,0% |
| chỉ tham số BatchNorm và bias | 93,0% | 82,0% |

**Kết luận sửa lại:**
- Thứ phá hủy mô hình là việc lượng tử hóa thống kê phương sai `running_var` của BatchNorm, không phải việc lượng tử hóa trọng số.
  - Ở các lớp đầu mạng (ví dụ `features.2`, `features.3`), phương sai lớn nhất của một tensor là 18–51 trong khi có kênh gần bằng 0.
  - Với một hệ số chung, tới 31–33% số kênh của một tensor có phương sai nhỏ hơn nửa bước lượng tử và bị làm tròn về 0.
  - Sau đó BatchNorm chia cho căn của phương sai gần 0, nên đầu ra các kênh đó phình lớn và làm hỏng các lớp sau.
- Chỉ lượng tử hóa trọng số thì gần như không mất độ chính xác (mục 2, trên 13.101 ảnh).
- Các số của bản audit cũ vẫn đúng với cách nén nó đã dùng.

## 6. Giới hạn

- **Nén mới đo một lần, chưa đo trong huấn luyện:** phần A là dựng lại một lần sau huấn luyện, chưa phải huấn luyện liên kết có nén ở mỗi vòng. Cần chạy FedAvg với nén độ lệch ở mỗi vòng trên GPU để biết ảnh hưởng tới hội tụ.
- **Độ trễ mới đo trên một laptop:** máy đang chạy ứng dụng khác lúc đo. Chưa đo trên điện thoại hay trên trình duyệt (onnxruntime-web).
- **Cấu hình chọn lớp lấy từ FedAvg:** cấu hình được chọn bằng 200 ảnh `val` của mô hình FedAvg rồi áp nguyên cho Centralized. Đánh giá trên tập giữ riêng nên không bị thiên lệch do chọn, nhưng chưa thử cấu hình khác cho Centralized.
- **Chưa tính khoảng tin cậy:** chưa có khoảng tin cậy cho các chênh lệch nhỏ. Với PlantDoc (172 ảnh), chênh 1–4 ảnh không có ý nghĩa thống kê.

## 7. Cách chạy lại

```powershell
.\.venv\Scripts\python.exe scripts\eval_gd4_compression.py --split legacy_diagnostic_test --threads 6 --workers 0   # khoảng 30 phút trên máy trên
.\.venv\Scripts\python.exe scripts\eval_gd4_compression.py --limit 200 --no-onnx                                     # thử nhanh, không ghi vào artifacts/
```

Tệp ONNX trung gian nằm ở `tmp/gd4/` (ngoài git). Test của các hàm nén, cách tính byte và cách chia nhóm lớp ONNX: `tests/test_gd4_compression.py`.

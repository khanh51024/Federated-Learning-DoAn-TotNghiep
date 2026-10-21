# Q6 — Đánh giá Centralized · FedAvg · Local-only trên cùng tập kiểm thử (ĐÃ RÀ, đóng 21-10)

> Một seed cho mỗi nhánh ⇒ **mọi so sánh dưới đây là mô tả**, không phải kết luận thống kê về phương pháp.
> Tập test: `legacy_diagnostic_test` (11.015 ảnh = 10.843 PV + **172 crop PD / 96 cảnh / 13 trong 38 lớp**),
> mở **một lần** 21-10 (02:22–02:36Z). G-LEAK = PASS. Spec `Q6_SPEC.md` cf7c5e89…, script phân tích 40694c29…,
> kết quả `q6/analysis_test/Q6_RESULTS.json` 0657359f…, receipt `q6/test_RECEIPT.jsonl` 777148ee…

## 1. Kết quả chính (cặp headline: accuracy toàn tập + accuracy PD, kèm macro-F1 PD)

| Mô hình | Acc toàn tập | Acc PV | **Acc PD** | macro-F1 PD | Checkpoint |
|---|---:|---:|---:|---:|---|
| Centralized | **0,9867** | 0,9923 | **0,6337** | 0,5685 | epoch 22 |
| FedAvg iid | 0,9629 | 0,9702 | 0,5000 | 0,5266 | round 13 |
| FedAvg α=1,0 | 0,9379 | 0,9450 | 0,4884 | 0,5400 | round 6 |
| FedAvg α=0,1 | 0,9266 | 0,9334 | 0,5000 | 0,5435 | round 27 |
| Local-only iid (TB đều 5 client) | 0,9328 | 0,9410 | 0,4140 | 0,4286 | — |
| Local-only α=1,0 | 0,8207 | 0,8283 | 0,3442 | 0,3844 | — |
| Local-only α=0,1 | 0,5225 | 0,5268 | 0,2512 | 0,1946 | — |

Local-only gộp theo luật **đăng ký trước khi mở test**: trung bình đều 5 client (LEAF, arXiv:1812.01097).
Kèm theo (không thay số chính): trung bình theo n_k iid 0,9328 · α=1,0 0,8237 · α=0,1 0,5273;
min–max acc toàn tập iid 0,902–0,955 · α=1,0 0,775–0,903 · α=0,1 0,453–0,550. Không báo "client tốt nhất".

## 2. Khoảng cách Centralized − FedAvg theo mức non-IID (pp)

| Mức α | Acc toàn tập | Acc PV | Acc PD | macro-F1 PD |
|---|---:|---:|---:|---:|
| iid | 2,38 | 2,21 | 13,37 | 4,19 |
| α=1,0 | 4,88 | 4,73 | 14,53 | 2,85 |
| α=0,1 | 6,01 | 5,89 | 13,37 | 2,49 |

- **Đường cong đề cương:** khoảng cách toàn tập **tăng theo mức non-IID: 2,4 → 4,9 → 6,0 pp**. Vì 98,4% ảnh test là
  PV, đường cong này **chủ yếu là đường cong trên PV**.
- Ngay khoảng cách 2,38 pp ở IID cũng đã trộn khác optimizer (D-C2) và việc train phân tán. Phần có thể quy cho
  non-IID một cách sạch hơn là **mức tăng so với IID: +2,5 pp (α=1,0), +3,6 pp (α=0,1)** — 3 điểm, một seed, mô tả.
- **Khoảng cách PD ~13–15 pp đã có ngay ở IID** và gần như không đổi theo α ⇒ **không được quy cho non-IID**.
  Nó trộn hai yếu tố: khác optimizer (D-C2: Centralized AdamW, client FedAvg/Local-only SGD) và việc train phân tán.
  Tách riêng từng yếu tố: **unknown** (không có thí nghiệm nào cô lập).
- Sàn Local-only: FedAvg − Local-only toàn tập 3,0 / 11,7 / 40,4 pp (iid / α=1,0 / α=0,1) ⇒ **trên tập test chung**,
  lợi ích của liên kết tăng mạnh khi dữ liệu càng lệch. FedAvg và Local-only cùng dùng client SGD (cùng hàm
  `train_client`) ⇒ khoảng cách này không lẫn yếu tố optimizer. Trên phân bố lớp riêng của từng client thì bức tranh
  ngược lại ở α=0,1 (Local-only 0,960 > FedAvg 0,914, xem §6).
- PD test chỉ có 13/38 lớp: **không diễn giải xu hướng macro-F1 PD** theo α.

## 3. McNemar exact trên 172 crop PD (Centralized vs FedAvg; mô tả)

| Mức α | b (chỉ Central đúng) | c (chỉ FedAvg đúng) | p (2 phía) | Mức cảnh (96, cảnh đúng = mọi crop đúng): b / c |
|---|---:|---:|---:|---|
| iid | 41 | 18 | 0,0038 | 23 / 14 |
| α=1,0 | 46 | 21 | 0,0031 | 22 / 15 |
| α=0,1 | 46 | 23 | 0,0076 | 21 / 14 |

**Trên tập test này**, Centralized đúng nhiều hơn FedAvg ở ảnh PD. Một seed mỗi nhánh ⇒ **không** claim phương pháp
nào tốt hơn. MDE ~7 pp (crop) đến ~9,5 pp (cảnh).

## 4. Val không dự báo test với PD

Ở chạy thử trên val (đã dùng để chọn checkpoint), khoảng cách PD ở iid = **0,00 pp**; trên test = **13,37 pp**.
Val có 269 crop PD, test 172 crop / 13 lớp. Vì vậy mọi kết luận về PD phải dựa trên test, và chỉ ở mức mô tả.

## 5. Chi phí truyền thông (FedAvg, byte thật từ telemetry)

| Mức α | Round chạy | Round tốt nhất | Payload tới round tốt nhất | Tổng thực sự đã truyền |
|---|---:|---:|---:|---:|
| iid | 23 | 13 | 778 MiB | 1.377 MiB |
| α=1,0 | 23 | 6 | 359 MiB | 1.377 MiB |
| α=0,1 | 32 | 27 | 1.616 MiB | 1.915 MiB |

Mỗi lượt client gửi/nhận đúng state_dict 6.275.944 B (gồm BN buffer); tổng byte telemetry khớp chính xác
6.275.944 × 2 × số lượt client. Centralized và Local-only không trao đổi model.

## 6. Công bằng (hai góc nhìn, báo cả hai)

**Ước lượng, giả định label shift:** acc_k = Σ_c p_k(c)·recall_c (p_k = phân bố lớp tập train của client k).

| Mức α | Centralized TB (min) | FedAvg TB (min, sd) | Local-only trên phân bố của chính nó TB (min) | Local-only trên test chung TB (min) |
|---|---|---|---|---|
| iid | 0,986 (0,986) | 0,961 (0,961, 0,000) | 0,931 (0,900) | 0,933 (0,902) |
| α=1,0 | 0,986 (0,985) | 0,937 (0,917, 0,013) | 0,943 (0,927) | 0,821 (0,775) |
| α=0,1 | 0,986 (0,980) | 0,914 (**0,803**, 0,064) | **0,960 (0,941)** | **0,523 (0,453)** |

Ở α=0,1: nếu mỗi client chỉ quan tâm đến phân bố lớp của chính mình, model Local-only (ước lượng 0,960, client tệ
nhất 0,941) **cao hơn** FedAvg tái cân (0,914, client tệ nhất 0,803). Nhưng trên tập test chung, Local-only chỉ
đạt 0,523. Hai góc nhìn trả lời hai câu hỏi khác nhau; không góc nào thay được góc kia.

## 7. Hiệu chỉnh độ tin cậy (T fit chỉ trên tập calibration; báo cả hai chiều)

| Mô hình | T | ECE15 PD trước → sau | PD sai & tin ≥0,9 | PV sai & tin ≥0,9 | Đúng & tin <0,5 (PV / PD) |
|---|---:|---|---|---|---|
| Centralized | 2,93 | 0,278 → 0,102 | 33 → 6 | 43 → 4 | 1 → 38 / 4 → 18 |
| FedAvg iid | 1,09 | 0,250 → 0,224 | 9 → 8 | 35 → 29 | 56 → 70 / 7 → 7 |
| FedAvg α=1,0 | **0,84** | **0,134 → 0,204** | 5 → 6 | **30 → 54** | 220 → 134 / 16 → 13 |
| FedAvg α=0,1 | 1,15 | 0,247 → 0,221 | 18 → 9 | 115 → 75 | 144 → 226 / 6 → 10 |

- Centralized: hiệu chỉnh giảm mạnh ca "sai mà rất tự tin" (PD 33→6), **nhưng** tăng ca "đúng mà không tự tin"
  (PV 1→38, PD 4→18). ECE PV nhích 0,005 → 0,011.
- FedAvg α=1,0 có T<1 (làm model tự tin hơn): ECE PD tệ đi 0,134→0,204, PV sai-tự-tin 30→54. Không phải lỗi:
  một T chung fit trên tập calibration 93% PV khiến miền PD (bị dịch chuyển) chịu thiệt (Ovadia 2019,
  arXiv:1906.02530). Giả thuyết tier 2: round tốt nhất 6/23 ⇒ model còn thiếu tự tin trên PV.
- Local-only: T 1,0–1,4 (iid), 1,4–2,0 (α=1,0), 2,7–3,4 (α=0,1); ECE PD trung bình 0,31→0,26 · 0,41→0,24 · 0,56→0,15.
- ECE chia bin đều số mẫu cho cùng chiều. **Không dùng test để chọn T hay ngưỡng từ chối cho DEPLOY**; DEPLOY dùng
  T của Q6 (fit trên calibration).

## 8. Nguồn gốc và giới hạn

- 3 nền tảng train: Centralized Colab T4 (cu130), FedAvg Kaggle T4 (cu128), Local-only laptop RTX 2050 (cu124);
  **đánh giá toàn bộ trên laptop**. FedAvg chạy lại trên val lệch ≤0,02 pp acc, ≤0,28 pp PD F1 so với lúc train.
- FedAvg dừng sớm theo luật đã đăng ký (round 23/23/32); Centralized có resume (epoch tốt nhất 22, trước resume).
- Local-only α=1,0 `client_02` = **RESUMED_NOT_BIT_EXACT** (bị cắt ở round 40, resume từ checkpoint); trần 14 h/scheme
  trên laptop được người dùng cho phép. Không scheme nào chạm trần: 0 sự kiện `CAP_REACHED`, mọi `segment_end`
  `stop_reason = null`, giờ ghi nhận 7,21 / 5,88 / 4,48 h (receipt iid b2c9653b…, α=1,0 652473ae…, α=0,1 a1599081…).
  Riêng α=1,0 có một đoạn bị cắt không có `segment_end` (~43 phút không ghi) ⇒ tổng thật ~6,6 h, vẫn dưới trần.
- FedAvg dùng `plantdoc_sampling_ratio = 0,25` như Centralized (G-PIPE-1).
- **F-B, luật chọn checkpoint khác nhau:** Centralized và FedAvg chọn best theo PD-F1 có ưu tiên sàn PV
  (quality control dùng chung); Local-only chọn best theo PD-F1 thuần, không có sàn PV
  (`local_only_mixed_runner.py`:192). So sánh với Local-only vì vậy chỉ ở mức mô tả.
- Sự cố 20-10: iid `client_00` bị 2 tiến trình cùng train trong một thư mục ~4 phút ⇒ thư mục cách ly vào
  `runs_rtx2050/runs_invalid/local_iid_collision_20261020`, iid chạy lại **từ đầu**; kết quả iid hiện tại không có
  client resume.
- Một seed; PD test 172 crop / 13 lớp.

## 9. Không được phát biểu

- Bất kỳ claim thống kê nào về phương pháp.
- "FedAvg tiệm cận Centralized trên ảnh thực tế" — khoảng cách PD 13–15 pp ở mọi α.
- Quy khoảng cách PD cho non-IID.

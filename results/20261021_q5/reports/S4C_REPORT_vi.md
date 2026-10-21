# S4-clean96 — Model nhận diện ảnh chụp thực tế nguyên cảnh (đã rà, 21-10)

> **Mô tả, không phải kết luận thống kê.** Mỗi model 1 seed; 96 ảnh / 13 lớp; **cùng 96 cảnh với tập test Q6**.
> Hai classifier được train trên **crop lá** (crop PlantDoc theo bbox GT + lá PlantVillage); ảnh nguyên cảnh là
> kịch bản người dùng thật mà model **chưa từng được train**. Không chỉnh model, luật, ngưỡng hay T theo kết quả.
> Spec `S4C_SPEC.md` ea98a5b0…, script `s4_run.py` 0fcdfe63…, kết quả `s4clean96/main/RESULTS.json` 29c46d02…
> (chạy 1 lần, CPU, 21-10 06:51–06:59Z, rc 0).

## 1. Kết quả chính — ảnh nguyên cảnh đưa thẳng vào model (không detector)

| Model | Đúng / 96 | Accuracy |
|---|---:|---:|
| v7 Centralized (epoch 42) | 54 | 0,563 |
| Q5 Centralized (epoch 22) | 56 | 0,583 |

- **Trên 96 ảnh chụp nguyên cảnh, model đúng khoảng 56–58%.**
- v7 và Q5 **không khác biệt rõ**: chỉ v7 đúng 8 cảnh, chỉ Q5 đúng 10 cảnh (b/c = 8/10, không tính p).
  Không phát biểu "Q5 tốt hơn".
- So với Q6 trên crop lá GT của cùng các cảnh (acc PD Centralized Q5 = 0,634): ảnh nguyên cảnh thấp hơn (~0,58) —
  phù hợp với việc ảnh nguyên cảnh có nền và nhiều vật thể mà model chưa được train. Đây là mô tả, không phải đo
  nguyên nhân.

## 2. Theo lớp (ảnh nguyên cảnh, số cảnh đúng)

| Lớp | n | v7 | Q5 |
|---|---:|---:|---:|
| Apple___Apple_scab | 10 | 6 | 6 |
| Corn___Common_rust | 10 | 9 | 9 |
| Tomato___Tomato_mosaic_virus | 10 | 4 | 7 |
| Tomato___Late_blight | 9 | 6 | 5 |
| Tomato___Septoria_leaf_spot | 9 | 6 | 8 |
| Grape___Black_rot | 8 | 7 | 5 |
| Tomato___Early_blight | 8 | 2 | 4 |
| Tomato___Bacterial_spot | 7 | 3 | 2 |
| Squash___Powdery_mildew | 6 | 5 | 6 |
| Tomato___Leaf_Mold | 6 | 1 | 0 |
| Potato___Early_blight | 5 | 2 | 2 |
| Potato___Late_blight | 5 | 1 | 1 |
| Corn___Cercospora_leaf_spot | 3 | 2 | 1 |

Mỗi lớp chỉ 3–10 cảnh: không diễn giải xu hướng theo lớp.

## 3. Độ tin cậy của dự đoán (ảnh nguyên cảnh; T fit trên tập calibration: v7 3,63 · Q5 2,93)

| Model | | Sai mà tin ≥ 0,9 | Đúng mà tin < 0,5 | Độ tin TB |
|---|---|---:|---:|---:|
| v7 | không T | 26 | 1 | 0,903 |
| v7 | có T | 4 | 8 | 0,648 |
| Q5 | không T | 18 | 2 | 0,892 |
| Q5 | có T | 4 | 9 | 0,686 |

T không đổi nhãn dự đoán. Không T, model rất tự tin cả khi sai (18–26 cảnh sai mà tin ≥ 0,9); có T, ca này còn 4
nhưng ca "đúng mà không tự tin" tăng lên 8–9. Báo cả hai chiều.

## 4. Phụ — hệ thống detector + model (mô tả hệ thống, KHÔNG phải năng lực model)

Detector GroundingDINO-tiny ("a leaf.", ngưỡng 0,15; detector của chiến dịch crop, **không thuộc DEPLOY Q7**) →
crop lá biên 8% → classifier. Cảnh không có lá nào: 0/96; số lá mỗi cảnh: trung vị 3, tối đa 44.

| Luật | v7 | Q5 | b/c |
|---|---:|---:|---|
| Bỏ phiếu theo lá | 52 (0,542) | 59 (0,615) | 5/12 |
| Lá tin cậy nhất (không T) | 52 | 55 | 7/10 |
| Lá tin cậy nhất (có T) | 54 | 56 | — |

Con số 0,615 là của **hệ thống detector + model**, không dùng để kết luận về model. Không báo precision/recall của
detector (không có nhãn người đầy đủ cho mọi lá).

## 5. Kết luận cho giảng viên

Hệ thống **nhận được ảnh bất kỳ** và trả về dự đoán, nhưng trên ảnh chụp thực tế nguyên cảnh mức đúng chỉ khoảng
**6/10** (96 ảnh, 13 lớp). Giả thuyết (chưa đo): model học trên crop lá, ít ảnh thực địa có nhãn (PlantDoc
~1.460 crop). Hướng cải thiện cần **thêm ảnh thực địa có nhãn**; không chỉnh tham số trên chính 96 ảnh này
(chúng thuộc tập test).

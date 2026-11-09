# Tích hợp mô hình int8 (PC v3) vào web app — tài liệu bàn giao

Nhánh: `web-int8-pc-v3` (tách từ `main` @ `925630b`). **Chưa merge vào `main`.** Thành viên phát triển web đọc tài liệu này, chạy test,
rồi tự quyết định merge.

Nhánh này làm hai việc:

1. Thêm vào web app khả năng nạp **một định dạng mô hình mới**: TorchScript int8 (`format: "torchscript_int8"`). Gói hiện tại (`v4b_s42`)
   không bị đổi gì. Đã kiểm: `load_evidence`, `predict_published` và `build_published_demo` cho kết quả như cũ.
2. Đưa lên **4 tệp mô hình int8** của chiến dịch train trên PC (release v3, α = 0,1, seed 42), kèm số đo. Các tệp nằm ở
   `artifacts/model_bundles/pc_v3_a01_s42_int8/`. **Chúng chưa được nối vào gói web đang chạy** (xem mục 4).

---

## 1. Mô hình được dùng

Các tệp int8 được dựng từ **chính** các checkpoint fp32 của gói PC v3 (gói `web_pc_v3_a01_s42_r2`, đã gửi trước đây). Thông số chung:

- release `pv_pd_v3`, phân hoạch non-IID nhãn α = 0,1, seed 42;
- checkpoint lấy ở epoch / round cuối;
- kiến trúc MobileNetV3-Small, 38 lớp, tiền xử lý `canonical_v1`.

| Thư mục | Mô hình gốc (fp32) | SHA-256 checkpoint fp32 (8 ký tự đầu) | SHA-256 tệp int8 (8 ký tự đầu) |
|---|---|---|---|
| `centralized_adamw` | Centralized AdamW | `0a8293a3` | `ee6a1349` |
| `centralized_sgd` | Centralized SGD | `7bf7cd80` | `156d9240` |
| `fedavg` | FedAvg | `5f923b6d` | `69ca4217` |
| `scaffold` | SCAFFOLD (giải pháp chống non-IID được chọn theo validation) | `4817fbac` | `6e6ca73a` |

SHA đầy đủ, số đo và độ trễ nằm trong `artifacts/model_bundles/pc_v3_a01_s42_int8/INT8_E5_SUMMARY.json`.

### 1.1 Cách lượng tử hoá

- Dùng `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`. Đây là **int8 động, chỉ áp cho các lớp
  Linear** (bộ phân loại). Các lớp conv **vẫn là fp32**.
- Sau đó `torch.jit.trace` rồi `torch.jit.freeze`, và lưu bằng `torch.jit.save`.
- Môi trường xuất: torch 2.6.0, engine `x86`, CPU Intel i5-13400F.

**Không dùng int8 tĩnh (static PTQ).** Đã thử theo yêu cầu ban đầu: torch FX, qconfig x86, hiệu chỉnh trên 512 ảnh train. Kết quả làm
**sập** MobileNetV3-Small: macro-F1 validation chỉ còn 0,03–0,06, so với 0,88–0,96 của fp32. Lỗi xảy ra cả với trọng số ImageNet gốc,
nên nguyên nhân nằm ở kiến trúc, không phải ở mô hình của đồ án. Muốn int8 đầy đủ phải train có lượng tử hoá (QAT), việc này nằm ngoài
phạm vi. Vì vậy tệp static **không** được đưa lên nhánh này.

### 1.2 Số đo

Số đo trên tập validation 4.613 ảnh (PlantVillage 4.344 + PlantDoc 269), đánh giá trên CPU. Đây là số mô tả, không phải số trên tập test.

| Mô hình | macro-F1 fp32 | macro-F1 int8 | Δ | accuracy fp32 | accuracy int8 | PlantDoc macro-F1 (lớp có mặt) fp32 → int8 |
|---|---|---|---|---|---|---|
| Centralized AdamW | 0,9629 | 0,9627 | −0,0002 | 0,9686 | 0,9684 | 0,5435 → 0,5425 |
| Centralized SGD | 0,9549 | 0,9554 | +0,0005 | 0,9629 | 0,9631 | 0,4729 → 0,4722 |
| FedAvg | 0,8832 | 0,8832 | −0,0000 | 0,9116 | 0,9113 | 0,4652 → 0,4631 |
| SCAFFOLD | 0,9013 | 0,9028 | +0,0014 | 0,9207 | 0,9217 | 0,4834 → 0,4846 |

- **Kích thước:** tệp TorchScript int8 4,36 MB, so với 6,24 MB của TorchScript fp32 (giảm khoảng 30 %).
- **Độ trễ:** CPU i5-13400F, batch 1, median của 200 lần đo, không gồm tiền xử lý.
  - Ở 1 luồng: int8 7,90–8,25 ms, fp32 8,06–8,61 ms. **Tương đương**, vì chênh lệch nhỏ hơn độ dao động (p90 − median).
  - **Không** được ghi là int8 "nhanh hơn".
  - Đây không phải độ trễ đo trên server web.
- **Luật đưa lên web** (đăng ký trước khi có số đo): mô hình int8 chỉ được hiển thị nếu đạt cả ba điều kiện.
  - fp32 đánh giá lại khớp số lúc train (|d| ≤ 0,002);
  - tệp nạp lại cho cùng argmax;
  - macro-F1 giảm không quá 0,01.

  Cả 4 mô hình đều đạt.

### 1.3 Nhãn bắt buộc khi hiển thị trên web

- Ghi đúng câu: **"int8 động, chỉ lớp Linear (classifier); các lớp conv vẫn fp32"**. Không gọi tắt là "mô hình int8".
- Số trên web là số **validation**, trừ số nào ghi rõ lấy từ bước kiểm thử cuối P6.
- Mô hình fp32 vẫn là **mặc định**. int8 chỉ là biến thể phụ, dùng để minh hoạ việc nén mô hình.

---

## 2. Các phần đã sửa

### 2.1 `src/bundle.py`

- Thêm các hằng `FORMAT_STATE_DICT = "state_dict"`, `FORMAT_TORCHSCRIPT_INT8 = "torchscript_int8"` và `MODEL_FORMATS`.
- `model_specs()` thêm khóa `format` cho mỗi mục. Thiếu khóa thì mặc định là `state_dict`, nên gói cũ và gói v4b không phải sửa gì.
- Với mục int8, `model_specs()` chép thêm `fp32_of`, `fp32_checkpoint_sha256` và `quantization`. Hàm này chỉ chép dữ liệu; việc kiểm tra
  nằm ở `web/demo_evidence.py`.

### 2.2 `web/demo_evidence.py`

- **`_check_int8_specs()`** (gọi trong `load_evidence`). Kiểm từng mục:
  - định dạng phải nằm trong danh sách hỗ trợ;
  - mục int8 phải có `fp32_of` trỏ tới một mô hình **fp32 trong cùng gói**;
  - `fp32_checkpoint_sha256` phải là SHA-256 hợp lệ.

  Sai bất kỳ điều nào thì gói bị từ chối (fail-closed, như mọi kiểm tra khác của tệp này).
- **`_check_int8_parents()`.** TorchScript không mang metadata release / seed / W0 như checkpoint fp32, nên tệp int8 được **buộc vào
  checkpoint fp32 gốc**: `fp32_checkpoint_sha256` phải trùng SHA của tệp gốc trong `artifact_manifest.json` của gói. Bản thân tệp int8
  cũng được kiểm hash qua `artifact_manifest.json` của thư mục nó, như mọi tệp khác.
- **`select_quantized_engine()`.** Bật engine `x86`, hoặc `fbgemm`, hoặc `onednn` (bản torch mới như 2.14 chỉ còn `onednn`; đã kiểm tệp fedavg int8 nạp được, ra 1×38 logits). Nếu bản torch không có engine nào trong ba (ví dụ CPU ARM chỉ
  có `qnnpack`), hàm ném `ValueError` với thông báo rõ ràng.
- **`_build_torchscript_int8()`.** Gồm các bước:
  - nạp bằng `torch.jit.load(..., map_location="cpu")` từ **đúng các byte đã băm**, giống đường fp32;
  - chạy thử một ảnh 0;
  - kiểm đầu ra là 1 × 38 logits hữu hạn.

  Lỗi nạp (thường do torch trên server cũ hơn bản đã xuất) được đổi thành `ValueError` có ghi phiên bản torch.
- **`_build_model()`.** Rẽ nhánh theo `format`. Đường fp32 giữ nguyên từng dòng.
- **`predict_published()`.** Với mô hình int8, kết quả có thêm khối `model_format` và `quantization` (gồm phương pháp, `fp32_of` và câu
  nhãn). Mô hình fp32 trả **đúng các khóa như cũ**, để các test hợp đồng không bị ảnh hưởng.
- **`explain_published()`.** Từ chối Grad-CAM cho mô hình int8, vì TorchScript lượng tử hoá không có gradient. Thông báo lỗi hướng người
  dùng sang mô hình fp32 gốc.
- **`load_evidence()`.** Trong `models[]`, mục int8 có thêm `format`, `fp32_of` và `quantization`. Mục fp32 không đổi.

### 2.3 `scripts/build_registry.py` (`build_published_demo`)

- Trước đây đường tệp được viết cứng là `<dir>/checkpoint_best.pt`. Nay đường tệp lấy theo khóa `file` của manifest, mặc định vẫn là
  `checkpoint_best.pt`. Local-only giữ nguyên đường `local/client_xx/checkpoint_best.pt`.
- Mục int8 trong registry có thêm `model_format` và một câu ghi chú: không có Grad-CAM, cần engine x86/fbgemm/onednn.

### 2.4 `tests/test_int8_torchscript.py` (mới, 8 test)

Các test bao gồm:
- gói cũ mặc định là `state_dict`;
- thiếu mô hình cha, mô hình cha sai, SHA sai hoặc định dạng lạ thì bị từ chối;
- int8 được buộc vào SHA của checkpoint fp32 gốc;
- nạp từ byte và cho đúng 38 logits;
- số lớp sai thì bị từ chối;
- `_build_model` rẽ nhánh đúng;
- engine int8 có sẵn, hoặc báo lỗi rõ khi không có;
- **end-to-end**: dựng bản sao gói v4b, thêm một mục int8 làm đúng theo mục 3 dưới đây, rồi kiểm `load_evidence`, `predict_published`,
  và việc Grad-CAM bị từ chối. Sửa `fp32_checkpoint_sha256` thì gói phải bị từ chối.

**Trạng thái kiểm thử (ghi trung thực):**
- Cả 8 test đã **pass** trên máy PC: Python 3.11, torch 2.6.0+cu124, chạy trên CPU.
- Máy PC không cài `pytest`, nên các test được chạy bằng một trình chạy tối giản, có hỗ trợ `pytest.raises` và `tmp_path`.
- **Chưa chạy toàn bộ bộ test của repo.** Ngoài 8 test mới, đã kiểm thủ công trên gói mặc định v4b: `load_evidence` có `available=True`
  với 7 mô hình, `predict_published('fedavg')` chạy được, và `build_published_demo` cho cả 7 mục `hash_verified`.
- Việc cần làm ngay: chạy `pytest -q` đầy đủ trên máy dev trước khi merge.

---

## 3. Cách thêm một mô hình int8 vào gói web

Làm sau khi đã nhập gói PC v3 (fp32) thành một gói trong `artifacts/model_bundles/`. Test end-to-end
(`_add_int8_to_bundle_copy` trong `tests/test_int8_torchscript.py`) làm đúng các bước dưới đây và có thể dùng làm mẫu.

1. **Thư mục riêng cho mỗi mô hình int8.** Tên thư mục phải trùng `model_id`. `build_registry` dùng tên thư mục làm khóa suy luận, và
   `predict_published` tìm mô hình theo `model_id`. Ví dụ: `fedavg_int8/model_int8_dynamic_torchscript.pt`. Tệp lấy từ
   `artifacts/model_bundles/pc_v3_a01_s42_int8/<mô hình>/`.
2. **Mục trong `import_manifest.json`** (`models[]`):
   ```json
   {"model_id": "fedavg_int8", "dir": "fedavg_int8", "file": "model_int8_dynamic_torchscript.pt",
    "display_name": "FedAvg int8 (động, Linear)", "phase": "pc_v3", "method": "FedAvg int8 (dynamic, Linear)",
    "registry_id": "web_fedavg_int8", "format": "torchscript_int8", "fp32_of": "fedavg",
    "fp32_checkpoint_sha256": "<SHA-256 đầy đủ của checkpoint fp32 FedAvg trong gói>", "quantization": "dynamic_linear_qint8"}
   ```
   Thêm cả `sources["fedavg_int8"] = {"branch": "...", "commit": "..."}`.
3. **`fedavg_int8/artifact_manifest.json`.** Ghi `branch`, `release_manifest_sha256` và `validation_manifest_sha256` giống gói.
   `files` gồm tệp int8 (`bytes`, `sha256`) và `validation_comparison.csv`.
4. **`validation_comparison.csv`.** Thêm hai hàng cho mô hình int8 (nguồn `plantdoc` và `plantvillage`), với `checkpoint_sha256` là SHA
   của **tệp int8**. Mọi thư mục của gói phải mang **cùng một tệp CSV**, vì `load_evidence` từ chối nếu hash CSV khác nhau giữa các thư
   mục. Do đó phải chép lại CSV và cập nhật mục `validation_comparison.csv` trong `artifact_manifest.json` của **mọi** thư mục.
5. **`fedavg_int8/validation_metrics.json`.** Dùng cùng định dạng với các mô hình fp32 (`validation.overall` và `by_source`). Cổng số 4 của
   registry cần tệp này, nếu thiếu thì mô hình không hiển thị.
6. **Đổi con trỏ** `artifacts/active_bundle.json` (`manifest_sha256` mới) bằng `scripts/swap_web_models.py`. Sau đó chạy
   `python scripts/update_status.py`.

**Số đo cho bước 4 và 5:**
- `INT8_E5_SUMMARY.json` đã có số tổng và số PlantDoc của từng mô hình int8 và fp32 gốc.
- Số **PlantVillage riêng** và **crop_accuracy** theo đúng định dạng CSV của web chưa có trong tệp này. Nhóm train sẽ gửi chúng trong gói
  PC v3 r3, tính trên cùng tập validation.
- **Không** chép số của fp32 sang hàng int8. Test end-to-end có chép, nhưng chỉ để kiểm cấu trúc gói.

---

## 4. Các vấn đề có thể gặp

1. **Kiến trúc CPU của server.**
   - Tệp int8 được xuất với engine x86/fbgemm, cần **torch bản x86_64**.
   - Trên CPU ARM (ví dụ Graviton, Ampere, Apple Silicon), torch thường chỉ có `qnnpack`. Khi đó `select_quantized_engine()` báo lỗi và
     mô hình không hiển thị (fail-closed). Trang web không bị sập.
   - Repo hiện không có Dockerfile hay cấu hình triển khai nên không xác định được kiến trúc server. Cần kiểm trên server:
     `python -c "import torch; print(torch.__version__, torch.backends.quantized.supported_engines)"`.
2. **Phiên bản torch.**
   - `requirements.txt` chỉ ghi `torch>=2.1.0`, không ghim bản cụ thể.
   - Tệp TorchScript được lưu bằng torch 2.6.0, nên torch cũ hơn có thể không nạp được. Khi đó `_build_torchscript_int8` báo
     `ValueError` kèm phiên bản.
   - Khuyến nghị: server dùng torch ≥ 2.6. Nếu không nâng được, xuất lại tệp int8 bằng đúng bản torch của server, từ checkpoint fp32 gốc
     theo mục 1.1, rồi cập nhật SHA.
3. **Grad-CAM.** Không hỗ trợ cho int8. Giao diện nên ẩn hoặc vô hiệu nút "giải thích" khi mô hình có `model_format = torchscript_int8`.
   Phía server đã trả lỗi rõ ràng.
4. **Không nhanh hơn.** Chỉ lớp Linear được lượng tử hoá, nên độ trễ gần như bằng fp32. Lợi ích là tệp nhỏ hơn khoảng 30 %. Không quảng
   cáo là "tăng tốc".
5. **Số luồng.** Trên PC, 4 luồng chậm hơn 1 luồng (khoảng 12,5–13,6 ms so với 8 ms), có thể do CPU có lõi P và lõi E. Đây là giả thuyết,
   chưa kiểm. Nếu server có độ trễ lạ, thử `torch.set_num_threads(1)`.
6. **Hash của gói.**
   - Thêm mô hình làm đổi `import_manifest.json` và `validation_comparison.csv`, nên con trỏ `active_bundle.json` phải được ghi lại.
   - Không sửa tay manifest sau khi đã ghim hash: web sẽ từ chối toàn bộ gói. Đây là hành vi đúng, không phải lỗi.
7. **Danh mục tệp `.pt`.**
   - `build_registry` quét mọi tệp `.pt` trong repo. 4 tệp trong `artifacts/model_bundles/pc_v3_a01_s42_int8/` được xếp vào nhóm
     `model_bundles_inactive`, vì chúng chưa thuộc gói đang dùng.
   - Sau khi merge, chạy `python scripts/update_status.py`, để `PROJECT_STATUS.md` và `artifacts/status_report.json` khớp lại. Nếu không,
     hook pre-commit sẽ chặn commit tiếp theo.
8. **Git LFS.** Tệp `.pt` đi qua LFS (`.gitattributes`). Cần `git lfs install` trước khi pull. 4 tệp có tổng khoảng 17 MB.
9. **Cache bộ nhớ.** Mỗi mô hình int8 dựng xong được giữ trong `_MODEL_CACHE`, như mô hình fp32 (khoảng vài MB mỗi mô hình).
10. **Không dùng tệp static.** Nếu ai đó xuất lại bằng static PTQ, mô hình vẫn nạp được, nhưng sẽ dự đoán gần như ngẫu nhiên (xem mục
    1.1). Luôn kiểm macro-F1 trên validation trước khi hiển thị.
11. **Không trộn số giữa các gói.** Không so số của gói PC v3 với gói v4b trực tiếp: khác release, khác cách phân hoạch.

---

## 5. Nguồn

- Bước đo: E5 (N4.1) của chiến dịch train trên PC. Spec `E5_SPEC.md` r1 (sha `e5ea64c2`), script `e5_int8.py` (sha `27c1a905`), kết quả
  `E5_RESULT.json` (sha `d8f983d3`). Các tệp này nằm ở máy train, không nằm trong repo này.
- `INT8_E5_SUMMARY.json` (trong nhánh này) là bản trích từ `E5_RESULT.json`. Bản trích không chứa ảnh hay đường dẫn dữ liệu.
- Nhánh này không chứa dữ liệu ảnh, không chứa tập test và không chứa checkpoint nào ngoài 4 tệp int8 nói trên.

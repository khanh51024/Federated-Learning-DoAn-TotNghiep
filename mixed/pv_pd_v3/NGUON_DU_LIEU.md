# Nguồn dữ liệu của pv_pd_v3

Thư mục `mixed/pv_pd_v3/` là bản phát hành dữ liệu trộn PlantVillage và PlantDoc (38 lớp). Nguồn gốc, commit đã tải, giấy phép và kết quả kiểm tra chi tiết
nằm sẵn trong nhánh này, nên không chép lại ở đây:

- [`README.md`](../../README.md) ở gốc nhánh;
- [`THONG_TIN_NGUON_DATASET.txt`](../../THONG_TIN_NGUON_DATASET.txt) ở gốc nhánh.

## Điểm cần ghi nhớ

- **Ảnh PlantDoc trong `mixed/pv_pd_v1/images/classification/` là bản dẫn xuất** của PlantDoc (giấy phép CC BY 4.0): mỗi ảnh được **cắt theo hộp bao lá (bbox), có lề 8%**
  (`crop_policy_version` = `bbox_margin_8pct_v1`). Cần ghi công tác giả PlantDoc khi dùng lại; xem `CITATION.cff`.
- **Dựng lại và kiểm tra ảnh cắt:** `rebuild_pd_crops.py` (cùng thư mục) cắt lại từ ảnh PlantDoc gốc trong nhánh này và đối chiếu sha256 với manifest.
  Đã kiểm trên 14 ảnh train: 14/14 ảnh gốc khớp sha256 và 14/14 ảnh cắt trùng từng byte.
- **Giấy phép PlantVillage:** repository GitHub tại commit đã tải không có file LICENSE riêng. Dataset card chính thức của tài khoản `mohanty` trên Hugging Face
  khai CC BY-SA 3.0, nên kiểm tra lại điều khoản tại nguồn trước khi phân phối lại hoặc dùng cho mục đích thương mại.
- Các manifest trong thư mục `manifests/` có cột `original_path` là đường dẫn của máy tạo dữ liệu; hãy dùng cột `relative_path`.

## Trích dẫn

- Singh D., Jain N., Jain P., Kayal P., Kumawat S., Batra N. *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS COMAD 2020, tr. 249–253. DOI 10.1145/3371158.3371196.
- Mohanty S. P., Hughes D. P., Salathé M. *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science 7, 2016. DOI 10.3389/fpls.2016.01419.

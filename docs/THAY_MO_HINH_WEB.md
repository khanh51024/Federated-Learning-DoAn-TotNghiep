# Thay mô hình của web bằng một lệnh

Tài liệu này mô tả cách web nhận mô hình mới sau mỗi đợt huấn luyện (ví dụ v4b → v5) mà không phải sửa mã từng chỗ.

## 1. Nguyên tắc: nhận dạng nằm ở một nơi

Web phục vụ **một gói mô hình**: thư mục chứa 7 checkpoint (Centralized, FedAvg, 5 Local-only), số liệu đánh giá của chúng và `import_manifest.json`.
Mọi thông tin nhận dạng gói chỉ nằm ở `import_manifest.json`, đọc qua `src/bundle.py`:

| Thông tin | Trường trong manifest |
|---|---|
| release và mã băm release, tập validation, tập test giữ riêng | `release_dir`, `release_manifest_sha256`, `validation_manifest_sha256`, `holdout` |
| số ảnh validation theo nhóm nguồn (`plantdoc`, `plantvillage`, `field_other`) | `validation_counts` |
| tổng ảnh train, hạt giống, W0, phân vùng client | `shard_total`, `seed`, `w0_fingerprint`, `partition_scheme` |
| danh sách và tên mô hình, id trong registry | `models[]` (`model_id`, `display_name`, `registry_id`) |
| giao thức huấn luyện hiển thị trên web | `training{}` (lấy từ `config.json`, `final.json` của từng run) |

Mã web, báo cáo, trạng thái và giao diện **không viết cứng** SHA release, tên "Centralized v6", số 37.238, 269/4.344...; đổi gói thì mọi nơi đổi theo.
Hai thứ khác cũng suy ra từ gói, không viết sẵn trong giao diện:

- **Nhóm nguồn ảnh khi đo** (`validation_counts`, `holdout.counts`): `src/bundle.py::source_groups` trả danh sách nhóm (gói v4: `plantvillage`, `field_other`, `plantdoc`; gói cũ: hai nhóm). Mọi bảng, biểu đồ và nút chuyển nguồn trên trang Kết quả và Chẩn đoán lặp theo danh sách này (`identity.source_groups`, `SOURCE_GROUPS` ở `js/core/dom.js`). Nhóm lạ của gói sau lấy nhãn bằng tên nhóm; muốn nhãn đẹp thì thêm vào `SOURCE_PRESENTATION`.
- **Câu về cách huấn luyện** (số hạt giống, các cách học đã cùng thiết lập chưa, độ trôi client có được ghi không, tập giữ riêng đã kiểm trùng chưa, hệ số lệch α): `web/services/evidence_service.py::_facts` tính từ khối `training{}` của gói và các báo cáo, trả ở `/api/evidence/summary` khóa `facts`; `js/core/facts-text.js` dựng câu từ đó. Đổi gói thì câu đổi theo. Test `tests/test_report_facts.py` chặn các câu cố định của bản cũ quay lại.
Con trỏ `artifacts/active_bundle.json` chỉ ra gói đang dùng và ghim mã băm của `import_manifest.json`: sửa manifest sau khi nhập thì web từ chối (chặn mọi kết quả).
Không có con trỏ thì web dùng gói cũ `artifacts/branch_demo/20261003`.

## 2. Thay mô hình

Điều kiện: các run của `scripts/train_v4.py` đã kéo về máy (có `DONE`, `final.json`, `config.json`, `history.jsonl`, `checkpoint_best.pt`), release có ở `data/dataset_sources/mixed/pv_pd_v4`.

```
python scripts/swap_web_models.py --runs runs/v4b --seed 42 --name v4b_s42 --label v4b
```

Lệnh làm theo thứ tự và dừng ở bước lỗi:

1. **Nhập gói** (`scripts/import_run_bundle.py`): kiểm 7 run cùng release, cùng mã train, cùng công thức, không phải run thử; chép checkpoint nguyên byte; đánh giá lại 7 mô hình trên CPU trên tập validation (cùng đường suy luận với web); ghi `validation_metrics.json`, bảng so sánh, `summary.json`, `artifact_manifest.json` (mã băm từng tệp) và `import_manifest.json`.
2. **Chụp bản lưu** các báo cáo, thư viện ảnh sai và tệp học cộng đồng của gói đang dùng vào `artifacts/report_snapshots/<gói>/` và `data/report_snapshots/<gói>/` để lùi nguyên vẹn.
3. **Đổi con trỏ** sang gói mới.
4. **Sinh lại** danh mục mô hình, báo cáo (shard, phân vùng, rò rỉ, theo lớp, khoảng cách miền, cấu hình huấn luyện, tái đánh giá trên validation và test giữ riêng, thống kê, độ chính xác theo lớp mà client có hoặc không có ảnh, PlantVillage tham chiếu, thư viện ảnh sai), rồi phần học cộng đồng (mô hình đặc trưng ONNX, đầu phân loại gốc, đặc trưng đánh giá, mô phỏng) và trạng thái dự án. Hai bước chậm (ảnh nguyên khung PlantDoc, bộ ảnh ngoài PlantWild) bỏ được bằng `--skip-slow`.
5. Một bước lỗi thì lệnh **tự lùi** (khôi phục con trỏ và báo cáo đã chụp) và thoát với mã khác 0. Log ở `tmp/swap_<tên>.log`.

`python scripts/swap_web_models.py --steps` liệt kê các bước; `--only <bước>` chạy riêng một bước.

Sau khi thay, chạy `pytest` và `python scripts/update_status.py --full-gates`, rồi kiểm web ở máy này (`uvicorn web.app:app --port 8088`) trước khi triển khai lên VM.

Sau khi triển khai, kiểm mô hình ngay trên web bằng `python scripts/web_model_check.py` (mặc định gọi production; `--base-url http://127.0.0.1:8088` cho máy chủ cục bộ). Script gửi ảnh val, PlantWild và ảnh mẫu qua `/api/inference/all`, đối chiếu dự đoán với tính toán offline cùng đường tiền xử lý (phải trùng ~100%), đo độ chính xác theo nhóm nguồn, lọc loại cây, Grad-CAM, ảnh lớn và ảnh xấu, thời gian; kết quả ở `artifacts/web_model_check.json`.

## 3. Lùi về gói trước

```
python scripts/swap_web_models.py --rollback artifacts/branch_demo/20261003
```

Khôi phục con trỏ, các báo cáo, thư viện ảnh sai và tệp cộng đồng đã chụp của gói đó rồi dựng lại danh mục mô hình. Gói chưa từng được chụp (chưa có bản lưu) thì lệnh báo rõ và các báo cáo phải sinh lại.
Trạng thái học cộng đồng của gói đang dùng được chụp trước khi lùi, nên thay lại được.

## 4. Những điều gói train bằng `train_v4.py` chưa ghi lại

- Độ lệch cập nhật `‖w_k − w_toàn cục‖₂` của từng client FedAvg: web ghi `delta_l2_recorded: false` và ẩn biểu đồ độ trôi thay vì vẽ số không có.
- Số byte truyền mỗi vòng là kích thước float32 của trạng thái mô hình (mô phỏng trong tiến trình), không phải lưu lượng đo trên mạng.
- Checkpoint không mang `w0_fingerprint`; W0 được tính lại từ hạt giống khi nhập và ghi vào manifest (`w0_in_checkpoints: false`).
- Release v4 gán `source_domain = "plantdoc"` cho mọi ảnh ngoài đồng, nên báo cáo nhóm theo `source_dataset`: `plantvillage`, `plantdoc` và `field_other` (ảnh ngoài đồng từ các bộ khác). Thư viện ảnh sai chỉ lấy ảnh PlantVillage và PlantDoc (nguồn và giấy phép ghi rõ).
- Release v4 chỉ có hai cách chia (`label_alpha_0_1`, `iid`) nên trình xem non-IID hiện hai cách chia.

## 5. Tệp liên quan

`src/bundle.py` (đọc manifest), `src/registry.py` (hằng số suy ra từ gói), `scripts/import_run_bundle.py`, `scripts/swap_web_models.py`, `web/demo_evidence.py` (kiểm tra gói khi suy luận), `.gitattributes` (`artifacts/model_bundles/** -text`: mã băm theo byte, Git không đổi LF/CRLF).

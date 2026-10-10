# MODEL REGISTRY — Hợp đồng nguồn sự thật duy nhất

> Hiệu lực từ 2026-10-09, schema v2 từ 2026-10-10. Quy tắc này ràng buộc mọi code, script và tài liệu.

## 1. Ba lớp sự thật

| Lớp | File | Ai ghi | Ai đọc |
|---|---|---|---|
| Máy đọc | `artifacts/registry.json` | CHỈ `scripts/build_registry.py` | web, scripts, tests |
| Người đọc | `PROJECT_STATUS.md` | CHỈ `scripts/update_status.py` (trừ khối Nhật ký) | con người |
| Code dùng chung | `src/registry.py` | tay (hằng số đường dẫn + loader) | mọi module Python |

**Cấm:** viết tay `registry.json`; hardcode đường dẫn checkpoint/metrics trong `src/`, `web/`, `scripts/`
(checklist của `PROJECT_STATUS.md` tự kiểm điều này); sinh `PROJECT_STATUS.md` bằng tay ngoài khối Nhật ký.

## 2. Luồng một chiều (hướng chính thức)

```
train NGOÀI repo (Kaggle / branch train-*)
   -> bundle + artifact_manifest + SHA-256
   -> import & kiểm hash   (scripts/import_branch_demo.py)
   -> artifacts/registry.json  (scripts/build_registry.py)
   -> PROJECT_STATUS.md / web / báo cáo
```

Máy dev chỉ CPU: **không train**, chỉ nhận kết quả đã train rồi kiểm hash + test. Kết quả mới = import bundle mới
và đổi `DEMO_BUNDLE_DIR` / `DEMO_PROTOCOL_ID` trong `src/registry.py`, rồi chạy `update_status.py`.

## 3. Schema `artifacts/registry.json` (schema_version 2)

```
schema_version: 2
generator, canonical_run_id, official_direction{current_family, protocol_id, flow}
dataset_truth: { protocol, total_images, split{train,val,test}, num_clients, split_seed, conditions{...} }
families: { <family>: {description} }
models: [ <model>, ... ]
artifact_groups: [ {group_id, role, description, count, paths[], lfs_oids?} ]   # .pt ngoài danh sách model
evidence_reports: { domain_gap, per_class, client_shards, inference_audit }   # suy ra từ bundle: report_domain_gap.py, report_per_class.py
gates: { provenance, pre_demo_check, final_acceptance, network_artifact, checkpoint_catalog }
```

Mỗi `model`:

| Trường | Ý nghĩa |
|---|---|
| `model_id` | định danh duy nhất (khóa tra cứu) |
| `family` | `scientific_canonical` \| `published_demo` \| `network_demo` \| `reference` \| `smoke` |
| `pipeline` | `external_branch` (train ngoài, import bundle) \| `legacy_kaggle` \| `src` |
| `role` | `current` (hiện hành) \| `historical` \| `technical_demo` \| `reference` \| `quarantined` |
| `dataset_id` / `protocol_id` | dữ liệu và protocol mà số liệu thuộc về |
| `validation_manifest_sha256` / `taxonomy_sha256` | hash manifest validation công bố / hash thứ tự 38 lớp |
| `leakage_audit` / `leakage_audit_scope` | `passed` \| `unknown` (mặc định `unknown`). `passed` chỉ cho model của bundle khi `artifacts/release_leakage_audit.json` đạt, val hash khớp và bài forward xác nhận checkpoint ràng buộc release; scope ghi rõ giới hạn |
| `checkpoint_path` / `checkpoint_exists` / `file_sha256` / `state_sha256` | sự thật trên đĩa |
| `expected_state_sha256` | hash khai báo trong metrics gốc |
| `checkpoint_status` | `hash_verified` \| `matches_manifest` \| `missing_on_this_machine` \| `smoke_overwrite_detected` \| `hash_mismatch` \| `quarantined` \| `present` \| `not_applicable` |
| `preprocessing_id` | `imagenet_eval_v1` \| `canonical_v1` \| `get_default_transform(eval)` |
| `metrics` / `metrics_source` / `metrics_waiver` | số liệu + file nguồn; waiver chỉ cho model `reference` |
| `inference` | backend nạp model: `published_bundle` (+ `key`) \| `torchscript_reference` |
| `inference_test` | kết quả bài forward bắt buộc trên `artifacts/sample_test_leaf.jpg` |
| `web_visible` / `web_gate_failures` | có được phép dùng trên web không và lý do nếu bị chặn |
| `status`, `display_label`, `notes` | trạng thái, nhãn hiển thị, ghi chú trung thực |

## 4. Cổng `web_visible` (tính tự động, không gán tay)

Chỉ model có `role` ∈ {`current`, `reference`} mới được xét. Phải đạt **đủ 5 điều kiện**:

1. Checkpoint tồn tại và hash khớp manifest của nguồn (`checkpoint_status = hash_verified`; model `reference`
   được ghim bằng `PLANTVILLAGE_TRACED_SHA256`, trùng oid Git-LFS).
2. `taxonomy_sha256` khớp thứ tự 38 lớp của `data/dataset_manifest.json`.
3. `preprocessing_id` thuộc danh sách đã khai báo (`KNOWN_PREPROCESSING`).
4. Có `metrics_source` + `protocol_id` + `dataset_id` (hoặc `metrics_waiver` cho model tham chiếu).
5. Bài forward bắt buộc đạt: một ảnh mẫu đi qua đúng đường inference của web, ra top-3 hữu hạn, đúng lớp thuộc taxonomy.

Không đạt → `web_visible=false`, lý do ghi ở `web_gate_failures`, và `web/services/model_service.predict()` từ chối
(HTTP 503). Model chỉ minh họa phải để `display_label` ghi rõ "Demo", không được trình bày như hệ thống chẩn đoán
đã nghiệm thu.

## 5. Danh mục checkpoint trên đĩa

`build_registry.py` quét mọi `*.pt` (trừ `.venv`, `.git`, thư mục tạm). Mỗi file phải thuộc một model **hoặc** một
`artifact_groups` có khai báo (`report_experiments_smoke`, `web_runs_smoke`, `kaggle_run_lfs_pointer`...). File lạ vào
`gates.checkpoint_catalog.unclassified` và checklist báo CHƯA. Run `report_experiments` mode=real chưa được promote
cũng bị coi là file lạ.

## 6. Slot canonical & chống ghi đè

- `artifacts/checkpoints/{centralized_best,fedavg_best,fedavg_last}.pt` là slot canonical — CHỈ run khoa học thật
  (data_mode=real, không smoke) được ghi.
- Pipeline `src/` chặn: smoke không có output_dir riêng → ghi vào `artifacts/checkpoints/smoke/` với hậu tố `_smoke`.
- Checkpoint nhị phân KHÔNG track trong git (trừ bundle/LFS đã khai báo); quản lý bằng registry + hash.
- Test không được ghi vào `artifacts/checkpoints|metrics` (dùng `tmp_path`) — checklist tự kiểm.

## 7. Bài học đã đóng thành cơ chế

Smoke run từng ghi đè `fedavg_best.pt`/`centralized_best.pt` làm provenance FAIL và checkpoint canonical (99,09%) mất
trên máy này. Registry giờ tự phát hiện (`smoke_overwrite_detected`), tách "không có file" khỏi "sai hash"
(`fedavg_checkpoint_missing` vs `fedavg_state_hash_mismatch`), và code chặn tận nguồn.

## 8. Đồng bộ

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py              # registry + gates + PROJECT_STATUS.md
.\.venv\Scripts\python.exe scripts\update_status.py --check      # file thống kê có đang lệch không (pre-commit)
```

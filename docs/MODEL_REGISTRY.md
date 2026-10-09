# MODEL REGISTRY — Hợp đồng nguồn sự thật duy nhất

> Hiệu lực từ 2026-10-09. Quy tắc này ràng buộc mọi code, script và tài liệu.

## 1. Ba lớp sự thật

| Lớp | File | Ai ghi | Ai đọc |
|---|---|---|---|
| Máy đọc | `artifacts/registry.json` | CHỈ `scripts/build_registry.py` | web, scripts, tests |
| Người đọc | `PROJECT_STATUS.md` | CHỈ `scripts/update_status.py` (trừ khối ROADMAP) | con người |
| Code dùng chung | `src/registry.py` | tay (hằng số + loader) | mọi module Python |

**Cấm:** viết tay `registry.json`; hardcode đường dẫn checkpoint/metrics trong code;
sinh `PROJECT_STATUS.md` bằng tay ngoài khối ROADMAP.

## 2. Schema `artifacts/registry.json` (schema_version 1)

```
schema_version: 1
generator: "scripts/build_registry.py"
canonical_run_id: <run Kaggle canonical>
dataset_truth: { protocol, total_images, split{train,val,test}, num_clients,
                 split_seed, conditions{...} }        # từ data/partitions_stage2_scratch_v3/suite.json
families: { <family>: {description} }
models: [ <model>, ... ]
gates: { provenance, pre_demo_check, final_acceptance, network_artifact }
```

Mỗi `model`:

| Trường | Ý nghĩa |
|---|---|
| `model_id` | định danh duy nhất (khóa tra cứu) |
| `family` | `scientific_canonical` \| `published_demo` \| `network_demo` \| `reference` \| `smoke` |
| `display_name` / `display_label` | tên hiển thị / nhãn ngữ cảnh (Demo / Scientific / Smoke...) |
| `checkpoint_path` | đường dẫn tương đối POSIX từ repo root |
| `checkpoint_exists` / `file_sha256` / `state_sha256` | sự thật trên đĩa (state hash cùng scheme với provenance validator) |
| `expected_state_sha256` | hash khai báo trong metrics gốc (so khớp khi có checkpoint) |
| `checkpoint_status` | `hash_verified` \| `matches_manifest` \| `missing_on_this_machine` \| `smoke_overwrite_detected` \| `hash_mismatch` \| `quarantined` \| `present` \| `not_applicable` |
| `preprocessing_id` | `imagenet_eval_v1` \| `canonical_v1` \| `get_default_transform(eval)` |
| `protocol_id` | protocol thực nghiệm của số liệu |
| `metrics` / `metrics_source` | số liệu + file nguồn (truy vết được) |
| `status` | `verified_checkpoint` \| `verified_published_artifact` \| `unverified_metrics_only` \| `historical_unverified` \| `smoke_demo_only` \| `smoke_quarantined` \| `reference_only` |
| `web_visible` | model có được phép hiện trên web hay không |
| `notes` | ghi chú trung thực (mất checkpoint, chưa tái kiểm chứng...) |

## 3. Quy tắc thêm model mới

Một model CHỈ được thêm vào registry (và do đó xuất hiện trên web) khi:

1. Checkpoint tồn tại và `file_sha256` khớp manifest của nguồn (hoặc được ghi
   `missing_on_this_machine` trung thực).
2. Class mapping (38 lớp) đúng thứ tự taxonomy.
3. `preprocessing_id` khai báo và khớp với phiên bản model.
4. Metrics liên kết được `metrics_source` + `protocol_id` cụ thể.
5. Đạt bài kiểm tra inference bắt buộc (test pytest).

Không đạt → không vào registry → không hiện web. Model minh họa phải để
`display_label` ghi rõ "Demo", không được trình bày như hệ thống chẩn đoán
đã nghiệm thu.

## 4. Slot canonical & chống ghi đè

- `artifacts/checkpoints/{centralized_best,fedavg_best,fedavg_last}.pt` là slot
  canonical — CHỈ run khoa học thật (data_mode=real, không phải smoke, có
  output_dir riêng khi chạy thử) được ghi.
- Tất cả pipeline train (train/ và src/) đã chặn: smoke không có output_dir riêng
  → ghi vào `artifacts/checkpoints/smoke/` với hậu tố `_smoke`.
- Checkpoint nhị phân KHÔNG track trong git; quản lý bằng registry + hash.

## 5. Bài học đã đóng thành cơ chế (25/09)

Smoke run từng ghi đè `fedavg_best.pt`/`centralized_best.pt` làm provenance FAIL
và checkpoint canonical (99,09%) mất vĩnh viễn trên máy.registry giờ tự phát hiện
(`smoke_overwrite_detected`) và code chặn tận nguồn.

## 6. Đồng bộ

```powershell
.\.venv\Scripts\python.exe scripts\update_status.py
```

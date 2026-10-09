# Trạng thái triển khai kế hoạch sửa GĐ2

> Cập nhật nghiệm thu 10/09/2026: [docs/STAGE2_ACCEPTANCE.md](docs/STAGE2_ACCEPTANCE.md) thay thế kết luận nghiệm thu của bản triển khai dưới đây. Đã sửa thêm lỗi và kiểm chứng đủ 244 tensor; vẫn chưa hoàn tất thí nghiệm khoa học GĐ2.

Ngày: 09/09/2026. Phạm vi: triển khai A–F trong
`docs/STAGE2_REPAIR_PLAN.md`, giữ FedAvg, MobileNetV3-Small và ảnh PlantVillage
gốc; không triển khai thuật toán GĐ3.

## Kết luận

Code path GĐ2 đã sẵn sàng để chạy protocol đã cấu hình: prepare/preflight,
FedAvg train/resume/evaluate, Centralized, Local-only, report/diagnostics và
sweep. Bounded smoke đã chạy đủ ba mode. **Chưa hoàn tất nghiệm thu khoa học
GĐ2** vì chưa chạy full-data 100 round, toàn bộ ma trận nhiều seed và chưa có
khoảng tin cậy/kết luận accuracy gap.

## A — tính đúng

- Train transform nhận profile feature-skew đã lưu; augmentation/noise dùng seed
  ổn định theo sample, client và epoch. Validation/test tiếp tục dùng clean
  resize + center crop.
- Server và client seed Python/NumPy/Torch/CUDA; server restore RNG checkpoint.
  Cùng môi trường CPU, continuous 3 round và 2 round + resume tới round 3 có
  `max_abs_weight_diff=0.0`; metric không tính duration khớp hoàn toàn.
- Node→partition được query qua Flower. Reply bị từ chối nếu error, source,
  client, round, `n_k`, processed count, key, shape, dtype hoặc tensor hữu hạn
  không hợp lệ. `num_batches_tracked` vẫn lấy max; đây là FedAvg hiện tại, không
  phải FedBN.
- `fraction_train` chọn deterministic; `num_workers`, `output.progress` và giới
  hạn Ray CPU theo `max_concurrent_clients * cpus_per_client` có hiệu lực. CUDA
  một GPU từ chối concurrency khác 1. Validator kiểm tra finite/range, số
  manifest, quorum và tài nguyên.
- Evaluate resolve theo chính file config; checkpoint cũ có thể relocate bằng
  config mới hoặc `--dataset-root`/`--partition-dir`.

## B–C — tiến độ, log, checkpoint và resume

- Launcher sở hữu một renderer; reader thread ghi stdout vào `runtime.log` nên
  polling event không chờ `readline()`. JSONL giữ phần dòng chưa hoàn chỉnh.
- Có phase prepare/discovery/train/aggregate/validation/checkpoint/evaluate/report;
  local epoch được ghi riêng và hợp nhất thành `client_epochs.jsonl`.
- `rounds.jsonl` và `history.csv` commit sau mỗi round; `weights.jsonl` ghi
  global/update L2, relative norm, min/max, non-finite count, `n_k`, FedAvg
  weight và tensor bytes.
- `last.pt`, `best.pt`, `round_NNNN.pt` có retention, và `model_final.pt` chứa
  best weights + class/preprocessing/provenance. Fresh SGD mỗi client mỗi round
  vẫn là chính sách FedAvg; momentum client không được resume qua round.
- Resume tạo run mới có `parent_run_id`, materialize `best.pt`, không âm thầm
  tiếp tục checkpoint đã early-stop nếu thiếu `--force-resume-stopped`.
- Run mới `20260909_144922...` đo observed max overlapping local epochs = 1,
  đúng `max_concurrent_clients=1`.

## D — report và diagnostics

- `report --run-dir` tạo learning curves, LR, update norm, raw/row-normalized
  confusion matrix PNG/PDF/CSV, per-class CSV, `diagnostics.json/.md` và
  `report.json`.
- Quy tắc có điều kiện/evidence/suggestion cho non-finite, divergence, plateau,
  không cải thiện round 0, recall thấp, client nhỏ, update outlier, OOM và cảnh
  báo shutdown. Không tự chỉnh config từ test set và không gọi LLM theo epoch.
- Hàng confusion matrix support=0 chuẩn hóa thành 0; hàng có support cộng bằng 1.

## E — baseline và sweep

- Centralized và Local-only tái sử dụng `train_local`, evaluation, controller,
  checkpoint và report. Mọi mode seed cùng initialization; Centralized giữ
  optimizer state trong run, còn FedAvg dùng fresh client SGD mỗi round.
- `configs/partition_training.yaml` đã sinh 7 partition: IID; label alpha
  0.1/1/100; quantity alpha 0.1/1; và label alpha 0.1 + feature moderate.
  Test/val path hash vẫn lần lượt là
  `340198e0...6e79e` và `9dc22863...f6b112`.
- `sweep` mặc định chỉ dry-run; `--execute` mới train, mỗi mode chạy process
  riêng để cô lập Ray Windows. `--collect` tạo lại comparison từ artifact.
- Smoke sweep tại `runs/sweeps/stage2_smoke` chạy 64 train/64 val/100 test,
  2 client × 3 round, tạo comparison và payload estimate. Accuracy cả ba mode
  là 0 trên test smoke một lớp; số này chỉ là bằng chứng luồng chạy.

## F — rename và tương thích đường dẫn

- Đã rename một lần `gd2_noniid_partition` → `gd2_federated_learning`; không copy
  dataset/runs và không sửa artifacts lịch sử hàng loạt.
- `.vscode`, README, tài liệu vận hành và comment path đã cập nhật.
- `.venv` được `venv --upgrade`, editable install được reinstall; `pyvenv.cfg`,
  editable mapping và activation scripts trỏ tên mới. Lượt pip đầu bị sandbox
  chặn network; chạy lại có phê duyệt và thành công.
- Config mới + compatibility field check cho phép resume checkpoint format 1;
  protocol fingerprint mới hash nội dung manifest/class/profile/preprocessing và
  bỏ timestamp/library version dễ thay đổi. Format 1 phát `RuntimeWarning` vì
  artifact cũ chưa chứa content fingerprint; checkpoint format 2 mismatch bị từ chối.
- `pyproject.toml` giới hạn Flower 1.36.x, Ray 2.55.x, Torch 2.6.x và
  Torchvision 0.21.x theo API/môi trường đã kiểm chứng; `TRAINING.md` ghi CUDA
  12.4 wheel index khi recreate venv.

## Kiểm thử và lệnh đã thực sự chạy

1. Unit/regression A–E:

   `python -m pytest tests -k 'not flower_simulation_integration_smoke and not smoke_and_evaluate_cli' ...`

   Kết quả trước rename: **76 passed, 2 deselected, 2 deprecation warnings**, 18.78 s.
   Hai test bỏ chọn là integration cũ; các luồng tương ứng được chạy trực tiếp.

2. FedAvg smoke: 2 client × 2 round, CPU, exit 0, run
   `runs/fedavg/20260909_143236__...`; sau commit chỉ có deprecation warning.
3. Evaluate best checkpoint có/không `--config`: 100 sample, loss/accuracy/F1
   khớp tuyệt đối; xác nhận lỗi root cũ đã sửa.
4. Resume 2→3 so với continuous 3 round: weights exact (`0.0` max diff), metric
   exact, best tồn tại, history không lặp.
5. Centralized và Local-only smoke riêng: exit 0.
6. Prepare-data: 7 partition, giữ nguyên test/val hashes.
7. Full Stage-2 dry-run: 21/21 tổ hợp mode-condition sẵn sàng, không train.
8. Smoke sweep execute: Centralized + Local-only + FedAvg đều exit 0 và tạo
   `comparison.csv/json/png/pdf`. FedAvg ghi `completed_with_warnings` vì
   Flower deprecated API và Windows/Ray access violation lúc shutdown.
9. Sau rename: compileall qua; targeted regression **12 passed, 1 deselected**;
   preflight exit 0; evaluate checkpoint cũ exit 0; resume cũ tạo run
   `20260909_150141...` exit 0; fresh smoke tạo run `20260909_150228...` exit 0;
   full sweep dry-run vẫn 21/21 mục ready.
10. Sau chỉnh metadata log/progress cuối: targeted A/C/D **19 passed,
    1 deselected**, 2 warning từ Typer/Click; CLI help liệt kê đủ 8 lệnh.
11. Nghiệm thu chốt từ tên mới: full regression **76 passed, 2 deselected**;
    fresh smoke `20260909_150941...` exit 0, observed concurrency=1, không còn
    process Python/Ray sau run. Evaluate cùng run có CM 38×38 tổng 100 sample;
    mọi hàng có support của normalized CM cộng bằng 1, hàng support=0 toàn 0;
    report có 13 artifact.
12. Diagnostics thresholds đã chuyển vào resolved YAML; compileall và
    `pip check` qua, targeted config/report/checkpoint **26 passed**, preflight
    full config exit 0.

## Blocker và giới hạn còn lại

- Chưa chạy full train/sweep/multi-seed theo yêu cầu giới hạn tài nguyên; vì vậy
  chưa có kết luận khoa học GĐ2.
- Flower 1.36 deprecate `run_simulation`; chưa chuyển sang `flwr run` vì cần một
  lượt migration riêng để không mất launcher progress/control.
- Ray trên Windows vẫn có thể access violation khi shutdown dù checkpoint đã
  commit và exit tổng là 0. Trước full run nên dùng WSL2 hoặc nghiệm thu cleanup
  thêm. TTY live bar chưa được quan sát bằng terminal tương tác; parser và
  single-renderer đã có regression non-TTY/partial-line.
- Không có Git repository trong workspace, nên danh sách thay đổi và evidence
  được ghi bằng file/artifact thay vì commit/diff.

## File triển khai chính

- Sửa: `fl_training/{checkpoint,cli,client_app,config,data,progress,server_app,strategy,task}.py`.
- Thêm: `fl_training/{baselines,reporting,reproducibility,sweep}.py` và
  `tests/test_stage2_repair.py`.
- Config: `train_fedavg.yaml`, `train_smoke.yaml`, `train_smoke_resume.yaml`,
  `partition_training.yaml`, `stage2_sweep.yaml`, `stage2_sweep_smoke.yaml`.
- Docs/path: README gốc, `README.md`, `TRAINING.md`, `pyproject.toml`, file status này,
  `docs/STAGE2_REPAIR_PLAN.md`, `.vscode/settings.json`, venv metadata/activation.
- Generated: 7 thư mục partition + `index.json`, các run smoke/baseline/resume,
  report và comparison dưới `runs/`.

## Lệnh bàn giao từ tên thư mục mới

```powershell
cd D:\university\do-an-tot-nghiep\train-gd-2\gd2_federated_learning

# Preflight
.\.venv\Scripts\python.exe -m fl_training.cli preflight --config configs/train_fedavg.yaml

# FedAvg train / resume
.\.venv\Scripts\python.exe -m fl_training.cli train --config configs/train_fedavg.yaml
.\.venv\Scripts\python.exe -m fl_training.cli train --config configs/train_fedavg.yaml --resume runs/fedavg/<run_id>/last.pt

# Evaluate / relocate checkpoint cũ
.\.venv\Scripts\python.exe -m fl_training.cli evaluate --checkpoint runs/fedavg/<run_id>/best.pt --config configs/train_fedavg.yaml

# Report không train lại
.\.venv\Scripts\python.exe -m fl_training.cli report --run-dir runs/fedavg/<run_id>

# Baseline
.\.venv\Scripts\python.exe -m fl_training.cli baseline --config configs/train_fedavg.yaml --mode centralized
.\.venv\Scripts\python.exe -m fl_training.cli baseline --config configs/train_fedavg.yaml --mode local-only

# Sweep: dry-run trước, execute chỉ khi chủ động cấp ngân sách
.\.venv\Scripts\python.exe -m fl_training.cli sweep --config configs/stage2_sweep.yaml
.\.venv\Scripts\python.exe -m fl_training.cli sweep --config configs/stage2_sweep.yaml --execute
.\.venv\Scripts\python.exe -m fl_training.cli sweep --config configs/stage2_sweep.yaml --collect
```

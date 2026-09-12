# Nghiệm thu code huấn luyện GĐ2

Ngày hoàn tất kiểm tra: 10/09/2026. Phạm vi: nghiệm thu và sửa lỗi triển khai [STAGE2_REPAIR_PLAN.md](STAGE2_REPAIR_PLAN.md), đối chiếu PDF đề cương tr. 1–3 [1]. Tài liệu này cập nhật kết luận của lần triển khai 09/09; chỉ số [n] tra tại [REFERENCES_IEEE.md](REFERENCES_IEEE.md).

**Kết luận: tính đúng của tổng hợp trọng số và luồng train đã qua nghiệm thu nhỏ trên CPU/CUDA. Chưa đủ bằng chứng nghiệm thu kết quả nghiên cứu GĐ2.** Cần chạy protocol đầy đủ để đo suy giảm FedAvg theo mức non-IID so với Centralized và Local-only. Accuracy smoke không chứng minh chất lượng mô hình.

## 1. Trọng số và chỉ số đã kiểm chứng

- FedAvg dùng `n_k / sum(n_k)` trên các client tham gia round; `n_k` lấy từ manifest, không lấy số batch hoặc tổng mẫu đã xử lý qua nhiều epoch [2]. Với hai client 16 và 32 ảnh, hệ số thực đo là **1/3 và 2/3**. Reply sai client/round, số mẫu, keys, shape, dtype, NaN/Inf và metric không nhất quán bị chặn trước aggregate.
- Oracle NumPy float64 kiểm tra **toàn bộ 244 tensor**, gồm 1.568.918 phần tử dấu phẩy động và các buffer số nguyên. Test dữ liệu tổng hợp với `n_k = 11, 23, 47`: sai số tuyệt đối lớn nhất **3,4586×10⁻⁷**, trong tolerance `rtol=2e-6, atol=5e-7`. Không đổi tensor đầu vào; ghi/nạp checkpoint khớp tuyệt đối.
- Replay SGD thật từng client ở round 1 rồi tổng hợp độc lập bằng NumPy: kiểm tra đủ 244 tensor; sai số lớn nhất **1,1921×10⁻⁷**. Đây là kiểm tra giá trị tensor thực, không chỉ đọc các hệ số trong log.
- Tensor dấu phẩy động, gồm BatchNorm running mean/variance, được tổng hợp theo mẫu. Buffer `num_batches_tracked` lấy **max** và giữ dtype nguyên; đây là chính sách đã công bố của dự án, không phải FedBN và không phải trung bình thống kê BN chính xác của toàn bộ dữ liệu tập trung.
- CPU: 2 round liên tục và 1 round + resume đến round 2 có **max absolute difference = 0** cho toàn bộ model state trong cùng môi trường. `model_final.pt` bằng `best.pt` trên từng tensor. `best` có thể ở round 0 khi validation không cải thiện; `last` vẫn chứa trọng số round 2 đã train.
- Accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support và confusion matrix được kiểm tra với scikit-learn [10]. Macro tính trên đủ 38 lớp, lớp không có support dùng `zero_division=0`. Tổng CM và đường chéo phải khớp số mẫu và accuracy.
- `optimizer_steps` đếm cập nhật thực; `skipped_optimizer_steps` đếm bước bị AMP bỏ qua do gradient không hữu hạn [11]. Test CUDA chủ động tạo gradient Inf nhưng loss hữu hạn xác nhận không ghi nhận nhầm bước SGD.

Bằng chứng tổng hợp: [acceptance.json](acceptance_20260909/integration_224036/acceptance.json). Oracle số học: [oracle.json](acceptance_20260909/final_all/test_all_mobilenet_tensors_mat0/oracle.json). Script replay: [verify_smoke_weights.py](../scripts/verify_smoke_weights.py).

## 2. Lỗi được sửa trong lần nghiệm thu

1. **Tổng hợp và dữ liệu:** kiểm tra schema tensor ngay cả khi caller không truyền reference; sửa NumPy dtype và giữ độ chính xác float64; chặn ép số mẫu thập phân thành nguyên. Audit chặn ảnh lặp làm tăng sai `n_k`, kiểm tra train union khớp cả nhãn/group/client và các split không giao nhau.
2. **Metric và GPU:** kiểm tra tính hữu hạn/range và tổng metric qua từng local epoch; tách đếm AMP skip và thêm diagnostic tương ứng. Tạo optimizer sau khi model đến thiết bị train; baseline dùng model CPU riêng cho validation. Thiết lập CuBLAS trước khởi tạo CUDA để tránh cảnh báo tái lập đã quan sát [6], [11].
3. **Checkpoint/resume:** chỉ công bố log round/weights/epoch sau khi lưu checkpoint; test lỗi ghi đĩa không công bố round chưa commit. Resume bảo toàn best, lịch sử, trạng thái điều khiển và tạo last tại run tiếp nối. Từ chối dùng baseline/inference artifact để resume FedAvg; force resume trạng thái stopped đặt lại bộ đếm dừng.
4. **Evaluate/report:** model xuất cuối chứa cấu hình và fingerprint trọng số; kiểm tra protocol trước evaluate. Kết quả gắn với đường dẫn, round, hash tensor và test manifest; lưu từng evaluation trong thư mục riêng. Cả hai tên checkpoint có cùng tensor vẫn được phân biệt, tránh mất provenance khi evaluate lại. Bổ sung đồ thị F1/support từng lớp và kiểm tra CM trước vẽ.
5. **Sweep:** không trộn seed, protocol, cấu hình train hoặc ngân sách round; không chọn run lỗi; kiểm tra metric thực sự thuộc `best.pt`. Giữ split seed cố định khi đổi training seed, cô lập output từng mode/seed; dry-run không xóa comparison. Có thống kê mean/std qua seed và đường cong alpha khi đủ điểm.
6. **Cấu hình/runtime/tiến độ:** resolve root theo package nên YAML ngoài `configs` chạy được; chặn kiểu YAML và giá trị không hợp lệ; run ID thêm hậu tố tránh trùng. Giới hạn Ray object store mặc định 256 MiB, startup timeout 120 giây và phát hiện lỗi khởi động [12]. Một renderer xử lý phase và local epoch cho FedAvg/baseline; tắt progress tải pretrained phụ. Generator smoke giữ class mapping và sửa khóa `client_01` của metadata trọng số.

Trọng số đầy đủ nằm trong checkpoint; `weights.jsonl` ghi số mẫu, hệ số tổng hợp, norm/update norm, min/max, nonfinite count và số byte. `client_epochs.jsonl`, `rounds.jsonl`, `history.csv` ghi các epoch/round đã commit. Không ghi hàng triệu giá trị tensor vào log văn bản.

## 3. Chạy thử và kiểm tra thực tế

- Hồi quy cuối: **99 passed, 2 deselected, 2 warnings trong 13,88 giây**. Hai test Flower integration cũ được bỏ khỏi lượt pytest này để chạy integration riêng có giới hạn dữ liệu/tài nguyên. Hai warning còn lại là deprecation Typer/Click. [Log cuối](acceptance_20260909/final_all.txt).
- Integration CPU: Centralized, Local-only và FedAvg đều hoàn tất 2 round; train 1 round, resume và evaluate inference đều exit 0. Hai client có 48 ảnh train tổng cộng; validation/test mỗi tập 38 ảnh, một ảnh mỗi lớp. [Lệnh và thời gian](acceptance_20260909/integration_224036/commands.json).
- Integration CUDA sau sửa: FedAvg 2 round và Centralized 2 round hoàn tất; mỗi mode có 6 SGD steps, 0 AMP skips. Baseline không còn warning CuBLAS của lượt trước. FedAvg có trạng thái `completed_with_warnings` do API Flower `run_simulation` deprecated. [FedAvg log](acceptance_20260909/integration_224036/cuda_fedavg_final.log), [baseline log](acceptance_20260909/integration_224036/cuda_baseline_final.log).
- Test TTY bằng spy xác nhận chỉ tạo một tqdm, cùng bar nhận epoch và các phase aggregate/validation/checkpoint/evaluate/plot. Chưa nghiệm thu hình thức hiển thị live trong terminal IDE của người dùng.
- Đã kiểm tra ảnh CM chuẩn hóa: bố cục không cắt trục. Fixture integration được giữ nguyên có metadata cũ dùng ID lớp số; generator đã sửa giữ tên lớp và có regression test. Metadata mẫu cũ còn ghi 32/32; phép train thực đọc CSV 16/32, đã được replay xác nhận. Không sửa hồi tố bundle vì sẽ làm sai fingerprint của bằng chứng cũ.
- **7/7 phân hoạch đầy đủ** qua audit: mỗi phân hoạch 37.887 train, 5.501 validation, 10.917 test; tổng 54.305; validation/test đủ 38 lớp. [Audit dữ liệu](acceptance_20260909/partition_audits.json).
- Preflight cấu hình đầy đủ `train_fedavg.yaml` exit 0: 10 client, tối đa 100 round, batch 16, client CUDA/server CPU; trạng thái `READY FOR SIMULATION / TRAINING`. Đây là kiểm tra cấu hình, môi trường và manifest, chưa phải chạy 100 round. [Log preflight](acceptance_20260909/preflight_final.txt).
- Dry-run tạo **21 cấu hình** cho 7 điều kiện × 3 mode, seed 42. Collect lại kết quả smoke thành công với kiểm tra identity mới. [Kiểm tra artifacts](acceptance_20260909/artifact_verification.txt), [comparison](acceptance_20260909/integration_224036/comparison/comparison.csv).

Smoke hiện chọn best round 0, test accuracy 1/38 ≈ 2,63%, macro-F1 ≈ 0,00135. Mẫu rất nhỏ, khởi tạo không pretrained; không dùng kết quả này để xếp hạng phương pháp hoặc kết luận FedAvg tốt/xấu. Payload FedAvg 2 round là 50.207.552 byte tensor upload + download, **ước tính**, chưa gồm overhead giao thức. Concurrency quan sát từ khoảng thời gian local epoch là 1, khớp cấu hình thử; chưa kiểm chứng mức concurrency lớn hơn 1.

## 4. Phần còn thiếu để nghiệm thu nghiên cứu GĐ2

1. Chưa chạy full-data hội tụ trên toàn bộ điều kiện label/quantity/feature skew, nhiều training seed và chưa có khoảng tin cậy/accuracy gap đủ ý nghĩa [1], [3]. Pilot cấu hình hiện có 100 round tối đa, early stopping theo validation, chỉ một seed.
2. Local-only đã có accuracy từng model và min/max/std trên **cùng global test**. Đây chưa phải fairness của global FedAvg trên từng miền/client. Global test hiện không có mapping test-client hợp lệ; cần định nghĩa held-out per-client protocol trước khi nghiệm thu tiêu chí đó. Không dùng dữ liệu train làm test client hoặc gán miền giả để lấp chỉ số.
3. `results-gd1` dùng test 10.893 ảnh, khác test GĐ2 10.917 ảnh. Không đưa trực tiếp accuracy cũ vào phép trừ suy giảm; cần baseline cùng split/preprocessing hoặc đánh giá checkpoint GĐ1 trên protocol tương thích có kiểm tra provenance.
4. Early stopping plateau và lỗi checkpoint đã test ở server với Grid giả lập; chưa chạy plateau chủ động qua Flower end-to-end. Resume bitwise chỉ được chứng minh trên CPU cùng môi trường; không cam kết giữa CPU/CUDA hoặc phiên bản thư viện [6].
5. Runtime startup từng thất bại do bộ nhớ ở lượt đầu; lượt giới hạn object store đã chạy được. Cleanup khi timeout/Ctrl+C và tính ổn định Windows dài hạn chưa được chứng minh đầy đủ. Cần kiểm tra lại trước full train dài; giữ phương án WSL2 theo [4]. Không nâng toàn bộ dependency trong lần sửa này.

Các mục trên là điều kiện còn mở, không bị đổi thành “đạt” chỉ vì pytest hoặc smoke pass. Không triển khai thuật toán GĐ3 trong lần nghiệm thu.

## 5. Lệnh tái kiểm tra

Chạy từ `gd2_federated_learning` bằng `.venv/Scripts/python.exe`, đặt `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` trên máy nghiệm thu. Không chạy pytest nặng đồng thời với Ray để tránh tranh RAM.

```powershell
.venv/Scripts/python.exe -m pytest tests -q -k 'not smoke_and_evaluate_cli and not flower_simulation_integration_smoke' -p no:cacheprovider
.venv/Scripts/python.exe -m scripts.acceptance_smoke
.venv/Scripts/python.exe -m scripts.verify_smoke_weights --config <CPU-config-cua-run> --run-dir <FedAvg-run>
.venv/Scripts/python.exe -m fl_training.cli preflight --config configs/train_fedavg.yaml
.venv/Scripts/python.exe -m fl_training.cli sweep --config configs/stage2_sweep.yaml
```

Lệnh sweep cuối chỉ tạo cấu hình, không train 21 lượt. Sau khi kiểm tra pilot/tài nguyên, dùng `--execute` để chạy ma trận. `max_rounds` là communication round; mỗi round có `local_epochs` epoch tại mỗi client được chọn. `checkpoint.every_n_rounds` điều khiển snapshot; best/last vẫn được lưu. Momentum client FedAvg được tạo mới mỗi round theo chính sách hiện có; baseline giữ optimizer trong run nhưng chưa hỗ trợ resume baseline.

Các file nguồn có hash trước/sau tại `acceptance_20260909/source_before.json` và `source_after.json`; không có Git repository trong workspace để xuất diff Git. Ảnh PlantVillage gốc và các phân hoạch đầy đủ không bị sửa trong lần kiểm tra.

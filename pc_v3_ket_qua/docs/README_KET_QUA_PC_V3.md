# Kết quả huấn luyện PC v3 (đợt 2–3) — 10/11/2026

Thư mục này gom **mô hình đã huấn luyện, bảng kết quả, mã nguồn và cấu hình** của đợt chạy trên máy PC (release dữ liệu `pv_pd_v3`, MobileNetV3-Small, 38 lớp, 5 client).
Trục so sánh xuyên suốt của đề tài: **Tập trung (trần) — FedAvg (hệ thống xây dựng) — Local-only (sàn)**, trên cùng tập kiểm định.

## 1. Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `models/` | 16 mô hình: Central AdamW / SGD, FedAvg, FedProx, MOON, SCAFFOLD, SCAFFOLD-BN, Local-only (5 client), 4 mô hình int8 động. Mỗi thư mục có `model_info.json` (nguồn run, SHA-256) và `validation_metrics.json` (theo nguồn PlantVillage / PlantDoc, `crop_accuracy`). File `.pt` lưu bằng Git LFS. |
| `results/` | `KET_QUA_TONG_HOP_PC.md/.json` (các câu bắt buộc và bảng P6), E1 (công bằng giữa client), E1 phase D, E2/E3, E4 (client drift theo vòng), E5 (lượng tử hóa), R-DIAG, P6 (`p6/P6_RESULT.json`), `noniid_deploy_select_20261109.json`. |
| `code/` | Mã đã sinh ra các số trên (xem `code/CODE_MANIFEST.json`). |
| `configs/` | Cấu hình FedAvg (q17, q18), cấu hình R-DIAG, đặc tả phân hoạch non-IID. |
| `docs/specs/` | Các đặc tả đã đăng ký trước khi chạy (E1, E4, E5, E2/E3, R-DIAG, FLW, P6 và phụ lục). |

Mọi mô hình trong gói: α = 0,1, seed 42, round/epoch cuối (quy tắc `checkpoint_last`), trừ khi `model_info.json` ghi khác.
Số trong `validation_metrics.json` là **số trên tập validation (4.613 ảnh: PlantVillage 4.344 + PlantDoc 269)**, tính lại từ logit đã lưu; không phải kết quả test cuối.

## 2. Cài đặt

- Python 3.11.9, torch 2.6.0+cu124, Windows 11, GPU RTX 2060 SUPER (8 GiB). Suy luận CPU cũng chạy được.
- Thư viện: xem `requirements.txt` / `requirements-train.txt` của repo. Mã trong `code/` đã được thay đường dẫn tuyệt đối bằng đường dẫn tương đối hoặc `<WORKSPACE_ROOT>`; khi chạy lại, thay `<WORKSPACE_ROOT>` bằng thư mục gốc làm việc của bạn. `CODE_MANIFEST.json` ghi SHA-256 gốc và SHA-256 sau khi thay đường dẫn của từng file.
- Dữ liệu (`pv_pd_v3`) **không** nằm trong nhánh này.

## 3. Sinh lại phân hoạch non-IID

- Đặc tả: `configs/partition_spec_v4.json` (12 kịch bản: iid, lệch nhãn α ∈ {10; 1; 0,5; 0,1}, lệch số lượng, lệch đặc trưng, hỗn hợp), `configs/PARTITION_SPEC_V5_SUPPLEMENT.json` và khóa SHA `configs/PARTITION_V5_LOCK.json` (bổ sung `label_alpha_100_0`).
- Mã bổ sung α = 100: `code/partition/gen_alpha100_v5.py` (seed `42 + int(α × 100)`, chạy hai lần và so sánh từng byte).
- Lưu ý trung thực: `gen_alpha100_v5.py` gọi bộ sinh gốc `generate_partitions_v4.py` (SHA `d872830d`). Bộ sinh này **không có trong bản sao làm việc trên máy PC**, nên nhánh này không kèm nó; muốn sinh lại từ đầu cần lấy bản đó từ nơi lưu trữ ban đầu. Manifest ảnh của từng client cũng không kèm theo.

## 4. Tái lập các bảng chính

Mọi lệnh chạy từ thư mục gốc làm việc, với `python -I -B`:

| Bảng | Lệnh |
|---|---|
| E1 (độ chính xác theo client, công bằng) | `code/pc_r9/e1_fairness.py <out_dir>` |
| E4 (client drift theo vòng) | `code/pc_r9/e4_divergence_r2.py <out_dir>` |
| E5 (lượng tử hóa int8) | `code/pc_r9/e5_int8.py` — xem `docs/specs/E5_SPEC.md` |
| E2/E3 (suy luận chung, rồi phân tích) | `code/pc_r9/e23_predict.py <out_dir>` rồi `code/pc_r9/e23_analysis.py` |
| Danh sách mô hình cho P6 | `code/pc_r9/build_p6_list.py <out_json>` |
| P6 (test cuối, chạy đúng một lần) | `code/pc_r9/p6_final_test_v3.py {dryrun|final|analysis_only} <out_dir> <cache_dir> <list_json> <list_sha>` |
| Gói web | `code/pc_r9/build_web_pkg_r4.py [DST]` |

Huấn luyện: `code/runners_q17/` và `code/runners_q18/` (FedAvg, Local-only, các thuật toán trong `q8_methods.py`, `q8r2_methods.py`, `q18_methods.py`), cấu hình ở `configs/`. Hạ tầng chạy nền: `code/pc_r9/laptop_sched_v2s4.py`, `ram_guard_launcher.py`, `queue_watch_r2.py`, `eco_watch.py`, `thermal_builtin_logger.py`.

## 5. Kết quả chính (và các câu bắt buộc)

Chi tiết và bảng đầy đủ: `results/KET_QUA_TONG_HOP_PC.md`. Mọi ô có seed là trung bình ± độ lệch chuẩn.

- **Giải pháp được chọn trên validation:** SCAFFOLD (α = 0,1, 3 seed): G = 0,9008 ± 0,0157, so với FedAvg 0,8829 ± 0,0034 (dG = +0,0179).
- **Test cuối P6 (chạy một lần, mô tả):** lựa chọn triển khai dựa trên validation; trên EXT-FINAL, `aug2_lap` (0,145) và `bal_lap` (0,141) cao hơn `r3data_pc` (0,105) có ý nghĩa sau Holm (p_holm 0,045); lựa chọn không đổi.
- **Tổng quát hóa ra ảnh hiện trường:** EXT-FINAL (ảnh hiện trường, ngoài miền, 7 lớp): accuracy mọi nhóm từ 3 seed trở lên nằm trong 0,10–0,17, tức mô hình **không tổng quát hóa tốt** ra ảnh hiện trường.
- **Tập legacy:** PD-172, PD-463 và PV-legacy là "legacy test đã dùng 3 lần (Q5 / Q6 / Q11b)".
- **int8 động (chỉ lớp Linear):** file TorchScript 4,36 so với 6,24 MB (−30%), macro-F1 validation thay đổi ≤ 0,0015, độ trễ CPU tương đương; |Δ accuracy| ≤ 0,006 trên mọi tập so với fp32 CPU. Lượng tử hóa tĩnh sụp đổ (macro-F1 0,03–0,06) nên không đưa vào.
- **E2 (PlantDoc, 269 ảnh):** không bác bỏ H0 sau Holm (min p_holm 0,135).
- **E3:** FedAvg 40 vòng truyền 3,1× dung lượng ảnh thô (2.510 MB so với 806 MB) — không phải tiết kiệm băng thông.
- **R-DIAG (chẩn đoán mô tả, 3 seed, không kiểm định):** ở α = 100, lấy mẫu PlantDoc theo tỉ lệ tự nhiên thay vì 0,25 đổi PD macro-F1 +0,014 ± 0,016 (2/3 seed dương); FedBN (BN cục bộ) đổi −0,003 ± 0,009 (runner q18 so với baseline runner q17).
- **FLW-A:** 3/3 seed bit-exact (40 round + checkpoint cuối); `flower_exact` = cổng tương đương của điều phối và truyền tải; harness giao DataLoader, RNG và GradScaler (không phải triển khai Flower độc lập).

## 6. Đối chiếu yêu cầu thực hiện tiếp (tóm tắt)

| Hạng mục | Trạng thái |
|---|---|
| N1.1 Tập trung công bằng (SGD cùng cấu hình client) | Có: AdamW và SGD, 5 seed. |
| N1.2 FedAvg trên Flower | Có kiểm chứng tương đương bit-exact (FLW-A); không phải triển khai Flower độc lập. |
| N1.4 Củng cố tập đánh giá | Có: PD-463, tập niêm phong EXT-FINAL; tập legacy được coi là đã lộ. |
| N2.1 Quét α | Có: α ∈ {100, 10, 1, 0,5, 0,1}, 3 seed. |
| N2.2 Lệch số lượng, lệch đặc trưng | Có kịch bản `qs` và `ss` (phase D), chạy FedAvg và Local-only. |
| N2.3 Độ chính xác theo client | Có (E1, E1 phase D). |
| N3.1 FedProx, FedBN, SCAFFOLD, MOON | Có, 5 seed ở α = 0,1 và 0,5. |
| N3.3 Cải tiến đề xuất | SCAFFOLD-BN (A1): không thắng FedAvg trên validation (0/4); báo cáo trung thực, không nâng thành kết luận. |
| N4.1 Lượng tử hóa | Có: int8 động (E5); lượng tử hóa tĩnh sụp đổ. |
| N5 Chi phí truyền thông, client drift, phân tích lỗi | Có (E3, E4). |

**Chưa làm:** N4.4 demo trên thiết bị thật; huấn luyện nhận biết lượng tử hóa (QAT); FLW-B (Flower độc lập).

## 7. An toàn dữ liệu

Nhánh này không chứa ảnh, manifest dữ liệu, tên ảnh của tập ngoài (EXT), logit của P6 hay các checkpoint ngoài 16 mô hình ở `models/`. Tập kiểm thử cuối chỉ được mở một lần trong P6.

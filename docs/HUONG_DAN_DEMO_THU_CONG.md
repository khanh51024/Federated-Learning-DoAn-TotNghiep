# Hướng dẫn demo thủ công 9 mục

Mở PowerShell tại `D:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old`. Mọi lệnh dưới đây đã chạy thử trên laptop này, trừ Cách A của mục 8 (lệnh đó cần máy thuê còn bật, xem chú thích ở mục 8). Số trong "Sẽ thấy" là số của lần chạy thử; số của bạn có thể khác chút ở thời gian, các kết quả KHỚP thì phải giống. Mỗi cửa sổ chạy trước lệnh này một lần:

```powershell
$env:PYTHONUTF8 = 1; $env:OMP_NUM_THREADS = 2; $py = ".\.venv\Scripts\python.exe"
```

Đóng bớt Chrome và game: máy cần khoảng 3 GB RAM trống. Số báo cáo đầy đủ nằm ở `docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md`.

**Các lệnh này chỉ để trình diễn, số độ chính xác của chúng không dùng để báo cáo:** chạy trên CPU với rất ít ảnh, ít vòng. Số báo cáo lấy từ các run 40 vòng trên GPU trong `runs/v4_sweep` và `runs/v4b`.

## Mục 4 và 5: Chạy Flower trên nhiều client, xem thông số

Cần 3 cửa sổ PowerShell, chạy theo đúng thứ tự, mỗi vòng khoảng 15–20 giây.

**Cửa sổ 1, máy chủ.** Chờ dòng `Starting Flower Server` rồi mới mở client:

```powershell
& $py -m src.federated.server_app --mode deployment --host 127.0.0.1 --port 18089 --rounds 2 --num-clients 2 --local-epochs 1 --seed 42 --require-real-clients --reject-suspicious-updates --max-update-norm 500 --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 --metrics-file tmp/demo/history.json --checkpoint-file tmp/demo/global.pt
```

**Cửa sổ 2, client 0:**

```powershell
& $py -m src.federated.client_app --client-id 0 --server 127.0.0.1:18089 --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 1 --seed 42
```

**Cửa sổ 3, client 1:**

```powershell
& $py -m src.federated.client_app --client-id 1 --server 127.0.0.1:18089 --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_01 --require-real-data --local-epochs 1 --seed 42
```

Thông số Flower hiện ngay ở cửa sổ 1 khi kết thúc: `round_fit_elapsed_ms`, `mean_probe_rtt_ms`, `rejected_clients`... Xem cho dễ đọc:

```powershell
& $py scripts/demo_show.py flower --history tmp/demo/history.json
```

Sẽ thấy:

```text
Vòng 1: 2 client được nhận, 0 bị từ chối; cả vòng 18.85 giây; gộp 52.2 ms; truyền ước tính 0.22 giây
  client 0: 380 ảnh, học 18.38 giây, RTT 23.6 ms, độ lệch cập nhật 0.0945; băm client gửi c80a19e146b7 / máy chủ tính lại c80a19e146b7 -> KHỚP
```

**Nói khi demo:** đây là Flower thật qua gRPC, mỗi client là một tiến trình riêng, học trên ảnh của mình. Đây là 2 client trên một máy; mục 8 mới là hai máy thật.

Thông số của run thật 5 client, 40 vòng trên GPU:

```powershell
& $py scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 20
```

## Mục 1: Gửi trọng số có sai lệch không

**Cách kiểm:** client tính SHA-256 của trọng số ngay trước khi gửi, máy chủ tính lại ngay sau khi nhận rồi so. SHA-256 đổi hoàn toàn nếu chỉ một bit khác, nên khớp nghĩa là không sai bit nào.

Xem kết quả: các dòng `KHỚP` của lệnh `demo_show flower` ở trên, và dòng cuối của lệnh `demo_show run`:

```text
Băm hai chiều: 400/400 lượt khớp.
```

**Chứng minh phép kiểm thật sự bắt được lỗi**, tự đổi một bit thì phải báo sai:

```powershell
& $py -m pytest tests/test_flower_app.py -k changed_weights -v
```

Sẽ thấy: `PASSED`. Test này đổi trọng số từ `2.0` thành `2.0000002` và yêu cầu run phải dừng ở cả hai chiều.

## Mục 2: Có sai lệch so với tập trung không

**Bước 1, hai bên xuất phát cùng một mô hình (W0):**

```powershell
& $py scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower
```

Sẽ thấy: `-> KHỚP`.

**Bước 2, chứng minh phép gộp không gây sai lệch** (khoảng 1 phút): cho mỗi client học đúng 1 bước rồi gộp, về toán học phải trùng học tập trung. Nếu truyền hoặc gộp có lỗi thì sẽ lệch.

```powershell
& $py scripts/verify_fedsgd_equivalence.py --quick --device cpu
```

Sẽ thấy:

```text
A_fedsgd label_alpha_0_1 float64 K=1: lệch L2 5.892e-14 ...      <- 1 bước: trùng, chỉ còn sai số làm tròn
B_local_steps label_alpha_0_1 float64 K=5: lệch L2 1.318e-01 ... <- 5 bước cục bộ: lệch thật
```

**Nói khi demo:** gửi và gộp không làm lệch. Lệch xuất hiện khi client học nhiều bước trên dữ liệu riêng, đó là bản chất của học phân tán.

**Bước 3, so trọng số hai mô hình đã train xong:**

```powershell
& $py scripts/demo_show.py compare --a runs/v4b/centralized_s42 --b runs/v4b/fedavg_s42
```

Sẽ thấy: L2 tham số 29,17; 94,72% dự đoán trùng trên 13.101 ảnh. **Đây là khác biệt do cách học, không phải lỗi truyền**, vì mục 1 đã chứng minh truyền đúng.

## Mục 3: Trình bày phương pháp kiểm tra

| Kiểm cái gì | Cách làm | Lệnh | Đạt khi |
|---|---|---|---|
| Truyền trọng số | SHA-256 hai đầu, mọi client, mọi vòng | `demo_show run` | khớp hết |
| Phép gộp | Tính lại trung bình theo n_k bằng NumPy, độc lập với mã gộp | `demo_show run`, cột sai số gộp | chỉ còn mức làm tròn |
| Xuất phát giống nhau | SHA-256 của W0 | `demo_show w0` | KHỚP |
| Công thức | FedSGD một bước so với tập trung | `verify_fedsgd_equivalence` | lệch dưới 1e-10 |
| Phép kiểm có nhạy không | Cố tình đổi một bit | `pytest ... -k changed_weights` | test đạt |
| Sau khi train | Khoảng cách trọng số, tỉ lệ dự đoán trùng | `demo_show compare` | đọc số, không có ngưỡng đạt |

## Mục 6: Huấn luyện tổng so với huấn luyện client

```powershell
& $py scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 20
```

Sẽ thấy:

```text
vòng  21: lr 0.0104, val top-1 90.17%, 14.8 giây, sai số gộp 7.45e-09, ...
    client 0: loss 0.821, 103 bước, học 2.9 giây, đã đi 1.048, cách mô hình chung 0.966
```

Cách đọc:
- **Sai số gộp:** sai số của phép tính gộp, chỉ ở mức làm tròn.
- **Đã đi:** client đi xa bao nhiêu so với mô hình nhận đầu vòng.
- **Cách mô hình chung:** sau khi gộp, client vẫn cách mô hình chung bao nhiêu. Đây là độ trôi của client, là khác biệt thật giữa mô hình client và mô hình chung, không phải lỗi.

## Mục 7: Code backend, một phần tập trung, một phần phân tán

**Chạy thật hai đường cùng một lệnh train, chỉ khác cờ** (khoảng 1,5 phút):

```powershell
& $py scripts/train_v4.py --method centralized --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs
& $py scripts/train_v4.py --method fedavg --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag flower
```

Chạy lại phải xóa `tmp/demo/runs` trước, nếu không lệnh sẽ báo đã xong và bỏ qua.

**Mở code để show:**

| Phần | Vị trí | Việc |
|---|---|---|
| Tập trung | `scripts/train_v4.py`, nhánh `else` trong `training_loop` | học trên toàn bộ ảnh bằng `train_epoch` |
| Phân tán, máy chủ | `src/federated/flower_app.py`: `make_round_fn` | gửi trọng số xuống, nhận về, kiểm băm |
| Phân tán, client | `src/federated/flower_app.py`: `client_step` | nhận trọng số, học, gửi lên |
| Một client học | `src/training/client_round.py`: `train_one_client` | gọi đúng `train_epoch` của nhánh tập trung |
| Gộp | `src/federated/strategy.py`: `aggregate_fedavg` | trung bình có trọng số theo số ảnh |
| Flower qua mạng (mục 4, 8) | `src/federated/server_app.py`, `src/federated/client_app.py` | máy chủ và client gRPC |

**Nói khi demo:** hai đường dùng chung một hàm học `train_epoch`, nên khác biệt chỉ nằm ở cách dùng dữ liệu và cách trao đổi trọng số. Đoạn code đã trích sẵn ở mục 7 của `docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md`.

## Mục 8: Đo tốc độ giữa 2 máy khác nhau

Đo được bằng cách cho server gửi một khối byte đúng cỡ mô hình (`--probe-payload-bytes 6275944`) mỗi vòng: RTT là thời gian khứ hồi, tốc độ lên và xuống tính từ thời gian gửi khối đó. Thời gian học do chính mỗi máy tự đo.

**Cách A, máy thuê còn bật:** laptop là client 0, máy thuê là server và client 1. Đo khi máy thuê đã train xong, vì GPU bận làm sai số thời gian học. Lệnh này ghi đè kết quả cũ trên máy thuê; bản đã lưu ở laptop không ảnh hưởng. Các bước này chưa được chạy lại trong lúc viết hướng dẫn: đang dùng lại đúng cấu hình của lần đo ngày 02/11.

1. Cửa sổ 1, mở đường hầm, giữ nguyên cửa sổ này (mật khẩu máy thuê):
   ```powershell
   ssh -N -p 3000 -L 18089:127.0.0.1:18089 -L 18090:127.0.0.1:18090 root@n2.ckey.vn
   ```
2. Cửa sổ 2: `ssh -p 3000 root@n2.ckey.vn`, rồi `bash ~/cropfed/speedtest_server.sh`.
3. Cửa sổ 3: `ssh -p 3000 root@n2.ckey.vn`, rồi `bash ~/cropfed/speedtest_client1.sh`.
4. Cửa sổ 4, trên laptop:
   ```powershell
   & $py -m src.federated.client_app --client-id 0 --server 127.0.0.1:18089 --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 2 --seed 42
   ```
5. Xong thì kéo kết quả về và xem:
   ```powershell
   scp -P 3000 root@n2.ckey.vn:cropfed/speedtest/out/wan_history.json tmp/demo/wan_history.json
   & $py scripts/demo_show.py flower --history tmp/demo/wan_history.json
   ```

**Cách B, không còn máy thuê:** xem lại kết quả đã đo thật ngày 02/11:

```powershell
& $py scripts/demo_show.py flower --history artifacts/network_speedtest/20261102T143913Z_wan_may_thue/wan_history.json
```

Kết quả đã đo, 3 vòng:

| Máy | Học 1 vòng | Ảnh/giây | RTT | Lên | Xuống |
|---|---:|---:|---:|---:|---:|
| Laptop i5-12500H CPU | 18,0 giây | 42 | 1.425 ms | 2,5 MiB/s | 0,45 MiB/s |
| Máy thuê RTX 3090 | 0,9 giây | 845 | 0,1 ms | không phải đường truyền thật | không phải đường truyền thật |

**Nói rõ khi demo:** kết nối đi qua SSH qua relay nên RTT cao và không đại diện cho mạng thường; client trên máy thuê chạy cùng máy với server. Muốn đo hai máy cùng mạng LAN: `docs/HUONG_DAN_DO_TOC_DO_LAN.md`.

## Mục 9: Khác biệt giữa phân tán và tập trung

Bảng đầy đủ ở mục 9 của `docs/KIEM_CHUNG_PHAN_TAN_TAP_TRUNG.md`, nguồn `artifacts/sweep_v4/table_main.csv`.

| | Tập trung | Phân tán |
|---|---|---|
| Dữ liệu | Gom toàn bộ ảnh về một nơi | Ảnh ở lại client, chỉ gửi trọng số |
| Bước gộp | Không có | Máy chủ gộp theo số ảnh mỗi vòng |
| Top-1 toàn tập thử (3 seed) | 97,69% | 96,80% khi đồng nhất; 94,03% khi lệch mạnh (α = 0,1) |
| Ảnh ngoài đồng | 92,23% | 90,09% khi đồng nhất; 84,08% khi lệch mạnh |
| Truyền | 1.524 MB ảnh, một lần | khoảng 2.523 MB trọng số, hai chiều, 40 vòng |

**Ý chính:** gửi và gộp trọng số không gây sai lệch (mục 1, 2). Khoảng cách với tập trung tăng khi dữ liệu các client lệch nhau: từ 0,89 điểm khi đồng nhất lên 3,66 điểm ở α = 0,1. Phân tán không tiết kiệm băng thông trong cấu hình này; lợi ích là dữ liệu không rời client.

## Nếu lỗi khi demo

| Hiện tượng | Xử lý |
|---|---|
| Client báo không nối được | Chưa thấy `Starting Flower Server`, hoặc cổng đang bị chiếm: đóng hết cửa sổ cũ rồi chạy lại từ máy chủ |
| `tmp/speedtest` không có | `& $py -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10` |
| Lệnh train báo "đã xong, bỏ qua" | Xóa `tmp/demo/runs` |
| Hết RAM, máy treo | Đóng Chrome, game; 2 client cần khoảng 3 GB |

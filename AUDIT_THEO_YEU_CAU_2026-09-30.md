# Audit theo yêu cầu đồ án Federated Learning

Ngày audit: 2026-09-30

## 1. Yêu cầu chuẩn đã chốt

- Dataset: PlantVillage.
- Backbone: MobileNetV3-Small, 38 lớp.
- Có Centralized baseline và Federated Learning để so sánh trên cùng kiến trúc/model contract.
- Federated Learning dùng Flower + FedAvg.
- Mục tiêu demo: 1 server + 5 clients.
- Server gửi global model/parameters xuống clients; client train local trên dữ liệu riêng rồi gửi model update/parameters + số mẫu + metrics về server.
- Cần quan sát được cách client train, số mẫu local, thời gian train, update norm/trọng số, client participation.
- Cần đo communication latency/RTT thực qua network, không đánh đồng loss với latency.
- Cần có IID/Non-IID và đo mức thiếu dữ liệu/data starvation của client.
- Cần thử faulty clients: no-train, random weights, scaled update, noise update.
- Cần kiểm tra FedAvg toán học độc lập và kiểm tra update bất thường trước khi aggregate.
- Global model sau FL cần dùng được để inference ảnh thật; nên so cùng ảnh với centralized model.

## 2. Kết quả audit hiện tại

### Đạt / đã có

1. `src/model/mobilenetv3.py`: có MobileNetV3-Small và utilities get/set/hash parameters.
2. `src/data/plantvillage.py` + hệ thống partition: có PlantVillage và nhiều kịch bản non-IID.
3. `src/centralized/`: có train/evaluate/inference cho centralized baseline.
4. `src/federated/client_app.py`: client nhận global weights, load model, local train bằng SGD, tính loss/accuracy/update norm và trả weights + num_examples.
5. `src/federated/strategy.py`: có FedAvg tự viết theo `n_k`, kiểm tra key/shape/non-finite.
6. `src/federated/faulty_client.py`: có `normal`, `no_train`, `random_weights`, `scaled_update`, `noise_update`.
7. `src/federated/weight_inspector.py`: có mean/std/min/max/L2/update norm/sample values theo layer.
8. `src/federated/telemetry.py`: có schema train time, upload/download bytes, RTT, loss, accuracy, num_examples, update norm.
9. `web/app.py`: có dashboard/API clients, weight inspection, compare, centralized train, federated train, inference artifacts.
10. Có checkpoint centralized/fedavg và metrics artifacts.
11. Có test toán học FedAvg, model, parameters, partition, integration, round-trip feedback loop.

### Test đã chạy trong audit

Lệnh:

```powershell
python -m pytest tests/test_fedavg.py tests/test_model.py tests/test_parameters.py tests/test_partition.py tests/test_integration.py tests/test_round_trip_feedback_loop.py -q
```

Kết quả: **17 passed**.

Điều này chứng minh phần lõi toán học, model contract, parameter handling, partition và feedback loop simulation đang hoạt động ở môi trường hiện tại.

## 3. Các gap quan trọng cần sửa trước demo

### G1 — Web đang chạy simulation, chưa điều khiển Flower network thật

`web/app.py` gọi `run_federated_simulation(...)`. Vì vậy nút train trên web hiện chứng minh được mô phỏng nhiều client, nhưng chưa chứng minh 5 process/máy Flower thật giao tiếp qua IP/port.

**Mức độ:** Critical.

**Cần làm:** bổ sung chế độ `simulation | network` trên web; network mode cần start/monitor Flower server thật và hiển thị clients thật đang connected.

### G2 — Network server dùng Flower FedAvg thật nhưng thiếu validator/telemetry nâng cao

`src/federated/server_app.py::start_flower_server()` hiện khởi tạo `flwr.server.strategy.FedAvg` trực tiếp. Nhánh này chưa có:

- per-client update validation;
- NaN/Inf/shape/hash/version verification ở server;
- anomaly detection;
- update norm validation;
- per-client telemetry lưu vào dashboard;
- checkpoint/evaluation logic giống simulation;
- manual-vs-Flower FedAvg verification trong runtime.

**Mức độ:** Critical.

### G3 — RTT/latency hiện có field nhưng simulation ghi bằng 0

`ClientRoundTelemetry` có `network_rtt_ms`, `download_time_ms`, `upload_time_ms`, nhưng `PlantVillageClient.fit()` không đo các đại lượng network này. Trong simulation, các giá trị đó mặc định 0.

Do đó dashboard hiện chưa chứng minh được communication latency thật.

**Mức độ:** Critical cho yêu cầu của thầy.

**Cần làm:** đo RTT/round-trip ở network server; tách `total request duration - local_train_time` thành communication/serialization overhead, hoặc instrument client/server timestamps rõ ràng.

### G4 — `update_norm` được tính trước khi áp dụng faulty mode

Trong `src/federated/client_app.py`:

1. client train;
2. tính `total_update_norm` trên `updated_weights` bình thường;
3. sau đó mới `apply_fault_mode(...)` để tạo `final_weights`.

Vì vậy client `random_weights`, `scaled_update`, `noise_update` vẫn report update norm của model bình thường trước khi bị phá.

Điều này làm telemetry/anomaly detection sai về mặt thực nghiệm.

**Mức độ:** Critical.

**Cần sửa:** tính và lưu ít nhất hai giá trị:

- `trained_update_norm` trước fault;
- `sent_update_norm` sau fault (giá trị server thực sự nhận).

Server chỉ dùng `sent_update_norm`/tự tính lại từ received weights để quyết định suspicious update.

### G5 — Hash hiện chỉ kiểm tra tính nguyên vẹn của chính payload client tạo, chưa chứng minh lineage

Simulation server tính lại SHA-256 của `reply["weights"]` và so với `reply["hash"]`. Điều này phát hiện payload bị thay đổi sau khi client tạo hash, nhưng **không chứng minh client train từ đúng global model** và không phát hiện client cố ý gửi random weights có hash hợp lệ.

**Cần làm:** gửi/ghi:

- `round_id`;
- `parent_model_hash`;
- `model_version`;
- server tự tính `received_update_norm = ||W_client - W_parent||`;
- server validate parent hash/version trước aggregate.

### G6 — Flower network client chưa nhận round config thực

`FlowerClientAdapter.fit()` đọc `config.get("round", 1)`, nhưng `start_flower_server()` hiện không cấu hình `on_fit_config_fn`, nên network clients có khả năng luôn thấy round mặc định 1.

**Mức độ:** High.

**Cần sửa:** Flower strategy phải phát `round`, `local_epochs`, `learning_rate`, `model_version`, `parent_model_hash` trong fit config mỗi round.

### G7 — Môi trường hiện tại thiếu Flower

`requirements.txt` có `flwr>=1.30.0`, nhưng Python đang chạy là 3.13.1 và `pip show flwr` trả về không cài đặt.

`.venv` có Python 3.11.9 nhưng chưa có `pip` khả dụng.

Các test sau chưa chạy được vì `ModuleNotFoundError: flwr`:

- `test_flower_parity.py`
- `test_real_flower_parity.py`
- `test_flower_live_socket.py`
- `test_flower_multi_round_network.py`

**Mức độ:** Critical trước demo network.

### G8 — README mô tả mạnh hơn bằng chứng runtime hiện tại

README nói network deployment, telemetry và dashboard khá đầy đủ, nhưng môi trường hiện tại chưa chứng minh được network Flower thật do thiếu `flwr`, và web train hiện dùng simulation.

Nên tách rõ label:

- `Simulation verified`;
- `Network verified`;
- `Pending environment/deployment verification`.

## 4. Test plan đề xuất

### Gate A — Environment

1. Dùng Python 3.11.
2. Rebuild `.venv` sạch.
3. `pip install -r requirements.txt`.
4. Xác nhận `python -c "import flwr; print(flwr.__version__)"`.

### Gate B — Core correctness

Phải pass:

```powershell
python -m pytest tests/test_fedavg.py tests/test_model.py tests/test_parameters.py tests/test_partition.py tests/test_integration.py tests/test_round_trip_feedback_loop.py -q
```

Hiện tại: PASS 17 tests.

### Gate C — Flower parity

Phải chạy:

```powershell
python -m pytest tests/test_flower_parity.py tests/test_real_flower_parity.py -q
```

Mục tiêu: manual FedAvg và Flower FedAvg gần như tương đương trong sai số floating-point.

### Gate D — Real socket

```powershell
python -m pytest tests/test_flower_live_socket.py tests/test_flower_multi_round_network.py -q
```

Sau đó chạy tay 1 server + 5 clients bằng 6 process khác nhau.

Acceptance:

- server thấy đủ 5 clients;
- mỗi client nhận model;
- mỗi client train local;
- server nhận 5 updates;
- FedAvg tạo global model mới;
- round 2 clients nhận đúng model mới của round 1.

### Gate E — Latency

Chạy network thật và lưu cho từng client/round:

- request start;
- client receive time;
- local train start/end;
- response received;
- local train duration;
- total round-trip duration;
- estimated communication overhead;
- bytes upload/download.

Không chấp nhận RTT = 0 khi claim là network experiment.

### Gate F — Faulty client

Chạy cùng một seed/config:

1. 5 normal clients.
2. 4 normal + 1 no_train.
3. 4 normal + 1 random_weights.
4. 4 normal + 1 scaled_update.
5. 4 normal + 1 noise_update.

Server phải tự tính received update norm, không tin metric do client tự khai.

Log cần có:

- accepted/rejected/suspicious;
- reason;
- update norm;
- cosine similarity (nếu bổ sung);
- ảnh hưởng đến global accuracy/loss khi có và không có validator.

### Gate G — Data starvation / Non-IID

Tạo tối thiểu:

- IID-ish / alpha lớn;
- Non-IID alpha = 1;
- Severe Non-IID alpha = 0.1;
- quantity-skew có một client rất ít samples.

Dashboard/report cần hiển thị:

- samples/client;
- classes present/38;
- missing classes;
- dominant-class ratio;
- entropy;
- local accuracy/loss;
- contribution `n_k / N`;
- cảnh báo data-starved client.

### Gate H — End-to-end demo

Demo cuối:

1. Centralized inference trên ảnh thật.
2. Start Flower server thật.
3. 5 clients connect bằng IP/port.
4. Chạy >= 2 rounds.
5. Quan sát weights/update norms/train time/latency.
6. Lưu global checkpoint.
7. Dùng cùng ảnh test với global FL model.
8. So sánh Centralized vs Federated trên cùng held-out test set.

## 5. Kết luận nghiệm thu hiện tại

**Simulation/core:** khá tốt, có cấu trúc nghiên cứu và test rõ ràng.

**Network Flower thật:** chưa đủ bằng chứng để kết luận đạt, chủ yếu do môi trường chưa có Flower và code network chưa tích hợp đầy đủ telemetry/validator/checkpoint logic như simulation.

**Đánh giá tổng quát theo yêu cầu thầy:** khoảng **70–75% phần nền tảng đã có**, nhưng 25–30% còn lại chính là phần quan trọng nhất khi demo: Flower network thật, latency thật, validator trọng số bất thường, lineage model theo round, và web quan sát network thật.

Không nên đóng gói release cuối trước khi hoàn thành Gate C–H.


## 6. Tiến độ sửa sau audit (2026-09-30)

### Đã sửa và xác minh

1. **Môi trường Flower**
   - Khôi phục `pip` cho `.venv` Python 3.11.9.
   - Cài `flwr==1.36.0` đúng constraint `pyproject.toml`.

2. **Faulty-client telemetry đúng payload thực gửi**
   - Tách `trained_update_norm` và `sent_update_norm`.
   - `update_norm` backward-compatible hiện phản ánh payload cuối cùng thực sự gửi server.
   - `random_weights`, `scaled_update`, `noise_update`, `no_train` không còn báo nhầm norm trước fault injection.

3. **Server-side update validator**
   - Thêm `src/federated/update_validator.py`.
   - Server tự kiểm tra parameter keys, shape, dtype, NaN/Inf, parent-model hash.
   - Server tự tính `received_update_norm`; không tin giá trị client tự khai.
   - Có trạng thái `ok / suspicious / rejected` và lý do.

4. **Model lineage theo round**
   - Client trả `parent_model_hash`, `round`, `model_version`.
   - Flower server gửi metadata round/model version xuống client.
   - Audited Flower strategy lưu parent model từng round và đối chiếu khi nhận update.

5. **Audited Flower FedAvg network**
   - `start_flower_server()` không còn dùng FedAvg thuần trực tiếp.
   - Dùng audited strategy wrapper nhưng vẫn gọi Flower FedAvg để aggregate.
   - Có thể giữ suspicious update để nghiên cứu ảnh hưởng hoặc bật policy reject trong simulation.

6. **Network tests trở thành pytest thật**
   - `test_flower_live_socket.py`: real gRPC socket 1 client / 1 round.
   - `test_flower_multi_round_network.py`: 2 clients / 2 rounds, có round-2 feedback.
   - `test_audited_flower_network.py`: production audited server path / 2 rounds.
   - Tất cả process đều được assert timeout + exit code.

7. **Web backend tách simulation/network**
   - `FederatedTrainRequest.engine = simulation | network`.
   - Network mode schedule `_run_federated_network_bg`, không gọi nhầm simulation.
   - Có `num_clients` và `server_address`.
   - Validate HOST:PORT.

8. **Data starvation**
   - `calculate_data_scarcity()` trả thêm:
     - `class_count_missing`
     - `entropy_normalized`
     - `starvation_score`
     - `starvation_level`
   - `/api/clients` đọc CSV partition thật từng client để tính scarcity, không dùng số mẫu giả định.
   - Trả thêm `largest_class_ratio`, classes missing, entropy normalized và starvation level.

9. **Network communication metrics**
   - Audited strategy lưu `round_fit_elapsed_ms`.
   - Lưu `estimated_comm_overhead_ms = round elapsed - max client local train time`.
   - Ghi kết quả Flower network vào `artifacts/metrics/network_flower_history.json`.
   - Metric này được ghi rõ là **ước lượng communication/serialization overhead**, không được gọi là one-way latency chính xác.

### Regression sau sửa

Lệnh regression tổng hợp đã chạy trên core, Flower parity, real sockets, audited server, validator, data scarcity và web backend.

Kết quả cuối đợt này:

```text
36 passed, 2 warnings
```

2 warnings còn lại đến từ dependency `typer/click`, không phải lỗi logic project.

### Việc còn lại nên làm tiếp

1. Frontend thêm lựa chọn Simulation/Network và hiển thị trạng thái network server/client thật.
2. Dashboard đọc `network_flower_history.json` và vẽ `round_fit_elapsed_ms`, `estimated_comm_overhead_ms`, accepted/suspicious/rejected clients.
3. Nếu cần **per-client RTT chính xác hơn**, bổ sung protocol probe/heartbeat riêng; không gán `estimated_comm_overhead_ms` thành RTT.
4. Bổ sung UI bật fault mode cho từng client và policy `observe-only | reject-suspicious`.
5. Chạy thử production: 1 server + 5 client process thật >= 2 rounds, không chỉ 1–2 client test.
6. Sau khi 5-client network pass mới đóng gói release/demo cuối.

### Tiến độ tiếp theo: UI/network/fault policy

- Frontend Federated đã có `Simulation` và `Flower Network thật` tách biệt.
- Network mode có cấu hình `server_address`, `num_clients`, `local_epochs`.
- `local_epochs` hiện được server gửi xuống client qua Flower fit config và client thực sự áp dụng cho round đó.
- Frontend có bảng Flower Network Metrics đọc `/api/federated/network-metrics`, hiển thị accepted/suspicious/rejected, mean update norm, round fit time và communication-overhead ước lượng.
- CLI client hỗ trợ `--fault-mode normal|no_train|random_weights|scaled_update|noise_update` và `--local-epochs`.
- Web network mode có validator policy `observe only` hoặc `reject suspicious`, cùng ngưỡng `max_update_norm` tùy chọn.
- Đã thêm test production 1 server + 5 client process thật + 2 rounds: PASS.
- Đã thêm test network với 1 normal + 1 `no_train`: server ghi đúng `suspicious_clients=1`: PASS.
- Regression tổng trước policy UI: `38 passed, 2 warnings`.
- Regression trọng yếu sau policy UI: `6 passed, 2 warnings`; JavaScript `node --check` PASS.

Ghi chú: `estimated_comm_overhead_ms` vẫn là ước lượng ở mức round, chưa phải one-way RTT từng client. Nếu cần chứng minh per-client RTT chính xác hơn, nên bổ sung heartbeat/probe protocol riêng hoặc timestamps đồng bộ giữa server-client.


## 7. Hoàn thiện RTT per-client, attack matrix và network inference

### RTT riêng từng client

Đã thêm `src/federated/network_probe.py`:

- Flower server mở một TCP echo probe riêng tại `Flower port + 1`.
- Client đo đúng PING -> PONG round-trip time trước local training.
- Metric này được lưu là `probe_rtt_ms` để không nhầm với Flower model-transfer time.
- Server lưu audit từng client/từng round gồm:
  - `client_id`
  - `probe_rtt_ms`
  - `local_train_time_s`
  - `received_update_norm`
  - `status`
  - `validation_reason`
- Server vẫn giữ `estimated_comm_overhead_ms` riêng ở mức round.

Test `tests/test_network_rtt_and_checkpoint.py`: PASS.

Một lần chạy thực nghiệm localhost:

- Client 0 RTT probe: ~1.54 ms
- Client 1 RTT probe: ~13.17 ms
- Mean RTT: ~7.36 ms
- Max RTT: ~13.17 ms
- Cả 2 update: status `ok`

Các giá trị localhost chỉ chứng minh cơ chế đo hoạt động; khi demo nhiều máy thật, RTT sẽ phản ánh mạng vật lý thực tế.

### Lưu Global Model sau Flower network

`start_flower_server()` hiện lưu global model cuối vào:

`artifacts/checkpoints/federated_network_global.pt`

Checkpoint chứa:

- `model_state`
- source `flower_network_fedavg`
- số rounds
- số clients
- local epochs
- SHA-256 model hash

Đã test load checkpoint vào MobileNetV3-Small và forward tensor `1x3x224x224`; output đúng `(1, 38)`.

Đã test inference trực tiếp với ảnh:

`web/static/samples/sample_tomato.jpg`

Kết quả kỹ thuật:

- `ready=True`
- checkpoint: `federated_network_global.pt`
- model thực sự chạy inference được.

Do checkpoint smoke chỉ train cực ít dữ liệu, confidence thấp (~2.63%) và prediction chưa có ý nghĩa chất lượng. Đây là đúng kỳ vọng; mục tiêu test này là chứng minh artifact Flower network có thể dùng cho inference end-to-end. Khi huấn luyện đầy đủ PlantVillage, mới dùng accuracy/confidence để đánh giá chất lượng.

### Attack matrix network thật

Đã thêm test `tests/test_attack_matrix_network.py` với 4 Flower clients qua gRPC thật:

- Client 0: normal
- Client 1: random_weights
- Client 2: scaled_update
- Client 3: noise_update

Ngưỡng validator: `max_update_norm = 5.0`.

Một lần chạy observe-only ghi nhận:

- normal norm ~1.755 -> OK
- scaled norm ~17.56 -> suspicious
- noise norm ~125.31 -> suspicious
- random norm ~1262.10 -> suspicious
- accepted=4, suspicious=3, rejected=0

Vì observe-only vẫn aggregate cả malicious updates, distributed loss đã trở thành `NaN`. Đây là evidence trực tiếp cho giới hạn của vanilla FedAvg khi server tin update xấu.

Chạy lại cùng experiment với `reject_suspicious_updates=True`:

- accepted=1
- suspicious=3
- rejected=3
- chỉ update normal được đưa vào FedAvg.

Test attack matrix: PASS.

### Dashboard/Web

Frontend hiện có:

- Engine `Simulation` / `Flower Network thật`.
- Server address, num clients, local epochs.
- Validator policy `observe` / `reject`.
- Optional max update norm.
- Network metrics table:
  - accepted/suspicious/rejected
  - mean update norm
  - mean/max RTT probe
  - round fit time
  - estimated communication overhead
- Per-client audit table:
  - round
  - client id
  - RTT probe
  - local train time
  - received update norm
  - validator status/reason
- Inference selector:
  - Canonical FedAvg
  - Network Global checkpoint vừa train qua Flower thật.

### Regression cuối

Regression tổng hợp bao gồm core math, model, parameters, partition, integration, parity, Flower sockets, audited strategy, validator, data scarcity, web engine, 5-client network, faulty detection, RTT/checkpoint và attack matrix.

Kết quả:

```text
40 passed, 2 warnings in 218.07s
```

2 warnings đến từ dependency `typer/click`, không phải project logic.

### Trạng thái hiện tại

Các yêu cầu cốt lõi đã được chứng minh bằng test tự động:

- PlantVillage / MobileNetV3-Small contract: PASS
- FedAvg correctness/parity: PASS
- Flower network socket thật: PASS
- 1 server + 5 client processes + 2 rounds: PASS
- Round-to-round feedback: PASS
- Server-side weight validation: PASS
- No-train client detection: PASS
- Random/scaled/noise detection: PASS
- Observe vs reject policy: PASS
- Per-client RTT probe: PASS
- Data starvation metrics: PASS
- Global model checkpoint từ Flower network: PASS
- Network global checkpoint inference: PASS
- Frontend/backend network mode separation: PASS

Phần cần làm tiếp nếu chuẩn bị bản nghiệm thu chính thức không còn là sửa core logic, mà là chạy experiment thật dài trên full PlantVillage, nhiều máy/network thật, rồi đóng gói số liệu báo cáo/biểu đồ reproducible.


## 8. Chuẩn hóa runner lấy số liệu báo cáo

Đã thêm `scripts/run_report_experiments.py` với hai mode:

- `--mode smoke`: kiểm thử pipeline nhanh; cho phép synthetic fallback và ghi rõ `scientific_accuracy_valid=false`.
- `--mode real`: bắt buộc có `--dataset-root` hoặc biến môi trường `DATASET_ROOT`/`PLANTVILLAGE_DATASET_ROOT`; fail-fast nếu thiếu ảnh thật, không âm thầm dùng dummy tensor.

Runner hiện chạy tuần tự:

1. Centralized baseline.
2. FL `label100` (gần IID).
3. FL `label1` (Non-IID vừa).
4. FL `label01` (Non-IID mạnh).
5. FL `quantity01` (data starvation/quantity skew).

Mỗi run lưu:

- `plan.json`
- `summary.json`
- `summary.csv`
- checkpoint/metrics riêng theo từng experiment.

Smoke batch đã chạy thành công tại:

`artifacts/report_experiments/20260930_144815`

Kết quả smoke đều quanh random baseline 1/38 (~2.63%) vì máy hiện không có PlantVillage raw images và loader dùng deterministic dummy tensors. Các accuracy này **không được dùng làm kết quả khoa học**.

Đã test `--mode real` khi không có dataset root: runner dừng ngay với thông báo yêu cầu đường dẫn PlantVillage, đúng thiết kế fail-fast.

Máy hiện tại:

- PyTorch CPU-only (`torch 2.14.0+cpu`)
- CUDA unavailable
- RAM khoảng 15.7 GB, thời điểm kiểm tra còn khoảng 2.2 GB available

Do đó chưa chạy full 54,305 ảnh trên máy này để tránh tạo benchmark quá chậm/thiếu ổn định.

### Sửa thống nhất định nghĩa update norm

Smoke suite phát hiện client `SentNorm` và server `ServerNorm` lệch khoảng 10x do client trước đây tính cả integer buffers như `BatchNorm.num_batches_tracked`, trong khi server chỉ tính floating tensors.

Đã sửa client update norm để chỉ tính floating model tensors, thống nhất với server-side validator. Regression liên quan validator + faulty network + attack matrix: `8 passed`.


## 9. Real-data guard và launcher nhiều máy

### Quét máy tìm raw PlantVillage

Đã quét C:/D:/E:/F:/G: không tìm thấy raw PlantVillage image tree theo tên/class folders. Repo staging cũ ở F: xác nhận ảnh gốc cố ý không được commit/đóng gói; chỉ còn manifest/profile.

### Preflight dataset

Đã thêm `scripts/preflight_plantvillage.py`.

Script xác minh:

- dataset root tồn tại;
- đúng 38 class directories;
- tổng số ảnh đúng protocol 54,305;
- tên class khớp manifests;
- manifest paths resolve;
- có fingerprint SHA-256 của bảng class-count để khóa experiment.

### Chống synthetic fallback trong network experiment

`PlantVillageClient` hiện có:

- `data_mode = real | synthetic`;
- `require_real_data=True` để fail-fast nếu ảnh thật không resolve;
- Flower metrics gửi `data_mode` về server.

Flower network server hiện có `require_real_clients=True`:

- nếu client báo `synthetic`/`unknown`, server gắn `rejected_synthetic_data`;
- update bị loại trước FedAvg;
- dashboard audit per-client hiển thị thêm `Data Mode`.

Web network mode có checkbox `Bắt buộc client dùng ảnh PlantVillage thật`.

### CLI/PowerShell deployment

Đã nâng CLI server/client và thêm:

- `scripts/start_real_flower_server.ps1`
- `scripts/start_real_flower_client.ps1`

Server launcher mặc định bật `--require-real-clients`.

Client launcher:

1. chạy `preflight_plantvillage.py` trước;
2. chỉ khởi động nếu preflight PASS;
3. truyền `--dataset-root`;
4. bật `--require-real-data`;
5. hỗ trợ fault mode để làm malicious-client experiment thật.

Ví dụ server:

```powershell
.\scripts\start_real_flower_server.ps1 -Rounds 5 -NumClients 5 -LocalEpochs 1 -Port 8089 -RejectSuspicious
```

Ví dụ client 0:

```powershell
.\scripts\start_real_flower_client.ps1 -ClientId 0 -Server 192.168.1.10:8089 -DatasetRoot D:\PlantVillage -PartitionDir data\partitions_stage2_scratch_v3\label100
```

PowerShell syntax: PASS.

`python -m src.federated.server_app --help`: PASS.

`python -m src.federated.client_app --help`: PASS.

Đã chuyển `src/federated/__init__.py` sang lazy import để CLI `python -m` không còn runpy warning do pre-import client/server modules.

Regression trọng yếu sau real-data guard + launcher: `4 passed, 2 dependency warnings`.


## 10. Provenance, multi-seed và scientific promotion gate

### Phát hiện canonical artifact không còn khớp provenance

Đã thêm `scripts/validate_artifact_provenance.py` và kiểm tra artifact hiện tại.

Kết quả hiện tại: **UNVERIFIED**.

Các lỗi critical:

- `fedavg_state_hash_mismatch`: state hash của `artifacts/checkpoints/fedavg_best.pt` không khớp `canonical_manifest.json`.
- `fedavg_accuracy_metadata_mismatch`: checkpoint hiện ghi `test_accuracy=1.0` trong khi canonical manifest ghi giá trị khác.
- `centralized_epoch_mismatch`: checkpoint Centralized hiện ghi 2 epochs trong khi manifest mô tả 10 epochs.
- Manifest cũ còn ghi `scientific_stage2_complete=false`.

Vì vậy README, API và web đã được sửa để không trình bày các số canonical 98–99% cũ như kết quả đã nghiệm thu. Inference vẫn có thể dùng checkpoint để demo kỹ thuật nhưng hiển thị `UNVERIFIED` nếu provenance chưa pass.

### Reproducibility seed

Đã nối `seed` vào:

- Centralized model initialization + DataLoader shuffle;
- Federated simulation global model + client DataLoader;
- Flower network server config;
- CLI server/client;
- PowerShell launcher;
- Web Centralized/Federated controls;
- checkpoint/metrics metadata;
- report runner.

Smoke pipeline đã chạy đủ seed `42`, `123`, `2026` và `scripts/summarize_multiseed.py` sinh mean/std. Smoke accuracy vẫn ~1/38 do synthetic fallback và được đánh dấu `scientific_valid=false`.

Lưu ý: multi-seed hiện thay đổi training/model-init seed trên fixed partition manifests; chưa đo partition-seed variance.

### Pre-demo readiness gate

Đã thêm `scripts/pre_demo_check.py`.

Trạng thái hiện tại:

```text
overall = technical_demo_ready
fail_count = 0
warn_count = 4
```

Các warning hiện tại:

1. chưa có PlantVillage raw dataset root để xác minh scientific training;
2. canonical provenance đang UNVERIFIED;
3. network checkpoint hiện tại được sinh từ smoke/synthetic run, không phải real-client run;
4. chưa có `multiseed_real_summary.json` hoàn chỉnh.

### Verified-artifact promotion

Đã thêm `scripts/promote_verified_run.py`.

Script chỉ cho phép promote nếu:

- run là `mode=real`;
- `scientific_accuracy_valid=true`;
- PlantVillage preflight `ok=true`;
- đủ Centralized/Federated checkpoint và metrics.

Khi pass, artifact được copy bất biến vào:

```text
artifacts/verified/<promotion_id>/
```

và có:

- `verified_manifest.json`;
- checkpoint SHA-256;
- metrics SHA-256;
- dataset class-count fingerprint;
- training seed;
- scenario;
- `LATEST_VERIFIED.json` pointer.

Smoke run, real run có scientific flag false, hoặc preflight fail đều bị từ chối. Bộ test fail-safe mới: PASS.

### Real multi-seed one-command workflow

Đã thêm:

```powershell
.\scripts\run_real_multiseed_report.ps1 -DatasetRoot D:\PlantVillage -CentralizedEpochs 10 -FederatedRounds 10 -Seeds 42,123,2026
```

Workflow:

```text
PlantVillage full preflight
        ↓
seed 42 / 123 / 2026
        ↓
Centralized
label100 (near IID)
label1 (moderate Non-IID)
label01 (severe Non-IID)
quantity01 (data starvation)
        ↓
mean/std summary
        ↓
promote selected verified run
```

### Regression sau thay đổi provenance/seed/promotion

- PowerShell launcher syntax: PASS.
- JavaScript syntax: PASS.
- Python compile: PASS.
- Critical network/provenance/promotion suite: **13 passed, 2 dependency warnings**.
- Warnings vẫn là deprecation từ `typer/click`, không phải project logic.

Kết luận hiện tại: project **đã sẵn sàng cho demo kỹ thuật**, nhưng chưa được gắn nhãn scientific-ready cho đến khi raw PlantVillage được cung cấp, real multi-seed chạy hoàn tất và verified artifact được promote thành công.


## 10. Final technical acceptance gate

Đã thêm `scripts/final_acceptance.py` để gom readiness + core tests + scientific guards + network tests thành một acceptance workflow.

Do một lần chạy full kéo dài gây timeout ở lớp tool transport, acceptance được xác minh theo 2 nhóm độc lập nhưng cùng tiêu chuẩn:

### Core + scientific guards

`python scripts/final_acceptance.py --skip-network`

Kết quả:

- technical_acceptance: PASS
- scientific_acceptance: BLOCKED
- core tests: 32 passed
- scientific guard tests: 16 passed
- pre-demo gate: technical_demo_ready

### Network acceptance

Nhóm real Flower/network tests chạy riêng:

- Flower live socket
- multi-round network
- audited Flower server
- 5-client network
- faulty client detection
- RTT + checkpoint
- attack matrix
- web network routing

Kết quả: `10 passed`.

Held-out evaluation + reproducibility guard chạy thêm: `3 passed`.

Tổng bằng chứng kỹ thuật ở pha acceptance này:

- 32 core tests PASS
- 16 scientific/provenance guard tests PASS
- 10 network tests PASS
- 3 held-out/reproducibility tests PASS

### Scientific blockers còn lại

`pre_demo_check.py` hiện báo 4 warning, không có technical failure:

1. `plantvillage_real_data`: chưa có raw PlantVillage trên máy hiện tại.
2. `canonical_provenance`: các checkpoint/metadata canonical lịch sử hiện không khớp manifest nên bị đánh dấu UNVERIFIED. Không được tự sửa hash/metadata để hợp thức hóa.
3. `network_global_artifact`: checkpoint network hiện là smoke/synthetic artifact, chưa phải run có `require_real_clients + require_real_evaluation + held-out test metrics`.
4. `multiseed_real_summary`: chưa có đủ 3 seed real-data.

Web/API đã có provenance guard: artifact canonical vẫn xem được như historical/legacy nhưng `scientific_valid=false` cho đến khi provenance thực sự pass.

### Multi-seed protocol

Smoke multi-seed đã chạy đủ seeds `42, 123, 2026`; `summarize_multiseed.py` tổng hợp đúng mean/std và ghi `scientific_valid=false` cho smoke data.

Real multi-seed launcher đã sẵn sàng:

```powershell
.\scripts\run_real_multiseed_report.ps1 -DatasetRoot D:\PlantVillage -CentralizedEpochs 10 -FederatedRounds 10 -Seeds 42,123,2026
```

### Scientific network run

Server launcher hỗ trợ:

- Seed cố định
- Require real clients
- Held-out global validation/test dataset
- Require real evaluation
- RTT probe
- validator observe/reject
- checkpoint + metrics + audit trail

Client launcher hỗ trợ:

- preflight partition riêng của client
- DatasetRoot thật
- require-real-data
- seed matching server
- fault mode

Held-out evaluation/reproducibility tests: PASS.

### Trạng thái cuối của code

**Technical demo ready: YES.**

**Scientific reporting ready: NO, do thiếu raw PlantVillage + real multi-seed run + verified artifact promotion, không phải do thiếu core code.**


## 11. One-command scientific network orchestration + verified promotion

Đã thêm:

- `scripts/run_local_5client_scientific_network.ps1`
- `scripts/validate_network_artifact.py`
- `scripts/promote_verified_network_run.py`

### Orchestrator

Một lệnh có thể chạy 1 Flower server + 5 Flower client process độc lập trên cùng máy với:

- real-data preflight đầy đủ;
- require-real-clients;
- require-real-evaluation;
- held-out global val/test;
- seed đồng bộ server/client;
- RTT probe;
- validator update norm;
- checkpoint + metrics + audit trail;
- artifact validation;
- verified promotion tự động nếu mọi gate PASS.

Dry-run đã PASS với scenario `label1`, seed `123`, 3 rounds; command server và cả 5 clients được sinh đúng.

### Network artifact validator

`validate_network_artifact.py` yêu cầu đồng thời:

- metrics/checkpoint tồn tại;
- `require_real_clients=true`;
- `require_real_evaluation=true`;
- đúng số client/round;
- final held-out test metrics tồn tại cả metrics JSON và checkpoint;
- seed tồn tại và khớp;
- source checkpoint đúng `flower_network_fedavg`;
- audit đủ round/client;
- tất cả client có `data_mode=real`;
- không có synthetic rejection.

Checkpoint smoke hiện tại đã được test và validator từ chối đúng (`ok=false`).

### Verified network promotion

`promote_verified_network_run.py` chỉ chạy promotion sau khi validator PASS. Artifact hợp lệ được copy vào:

`artifacts/verified_network/<promotion_id>/`

bao gồm:

- `network_metrics.json`
- `global_model.pt`
- `validation.json`
- `verified_network_manifest.json`

Manifest lưu SHA-256 cho từng file, seed, rounds, số client, held-out test metrics và scientific flags. Smoke artifact hiện tại đã được test và promotion bị từ chối đúng.

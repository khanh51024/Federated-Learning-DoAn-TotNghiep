# TÀI LIỆU TỔNG HỢP ĐỒ ÁN FEDERATED LEARNING

## PlantVillage + MobileNetV3-Small + Flower/FedAvg

**Project:** `D:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old`  
**Mục tiêu:** xây dựng, kiểm thử và đánh giá hệ thống nhận diện bệnh cây bằng **Centralized Learning** và **Federated Learning**, sử dụng **PlantVillage**, **MobileNetV3-Small**, **Flower** và **FedAvg**.

---

# 1. Mục tiêu của đồ án

Đồ án không chỉ nhằm huấn luyện một mô hình có accuracy cao, mà quan trọng hơn là **xây dựng và chứng minh đầy đủ quy trình Federated Learning**.

Các yêu cầu cốt lõi:

1. Dataset: **PlantVillage**.
2. Backbone: **MobileNetV3-Small**.
3. Bài toán phân loại: **38 lớp bệnh/cây**.
4. Có mô hình **Centralized Learning** làm baseline.
5. Có hệ thống **Federated Learning thật bằng Flower**.
6. Sử dụng chiến lược **FedAvg**.
7. Demo mục tiêu: **1 server + 5 clients**.
8. Client phải:
   - nhận global model;
   - train local trên dữ liệu riêng;
   - gửi model update/weights về server.
9. Server phải:
   - phát global model;
   - nhận update từ các clients;
   - kiểm tra update;
   - aggregate bằng FedAvg;
   - tạo global model mới cho round tiếp theo.
10. Cần đo và quan sát:
    - local loss;
    - local accuracy;
    - training time;
    - số mẫu của client;
    - update norm;
    - latency/RTT;
    - trạng thái accepted/suspicious/rejected.
11. Cần nghiên cứu:
    - IID;
    - Non-IID;
    - data scarcity/data starvation;
    - faulty/malicious clients.
12. Cần dùng global model cuối để inference ảnh thật.
13. Cần phân biệt rõ **technical demo** và **scientific experiment**.

---

# 2. Kiến trúc tổng thể

## 2.1. Centralized Learning

Luồng Centralized:

```text
PlantVillage dataset
        |
        v
Centralized train/val/test
        |
        v
MobileNetV3-Small
        |
        v
centralized checkpoint
        |
        v
Inference / Evaluation
```

Centralized model được dùng làm baseline để so với Federated Learning.

Các thành phần chính:

```text
src/centralized/train.py
src/centralized/evaluate.py
src/centralized/inference.py
```

---

## 2.2. Federated Learning

Kiến trúc mục tiêu:

```text
                  +--------------------+
                  |   Flower Server    |
                  |    Global Model    |
                  |      FedAvg        |
                  +---------+----------+
                            |
              global model parameters
                            |
          +-----------------+------------------+
          |        |        |        |         |
          v        v        v        v         v
      Client 0 Client 1 Client 2 Client 3 Client 4
          |        |        |        |         |
       Local     Local     Local     Local     Local
       Data      Data      Data      Data      Data
          |        |        |        |         |
       Train     Train     Train     Train     Train
          |        |        |        |         |
          +--------+--------+--------+---------+
                            |
                    model updates
                            |
                            v
                       Flower Server
                            |
                          FedAvg
                            |
                    New Global Model
                            |
                        Next Round
```

---

# 3. Công nghệ sử dụng

## Model

- MobileNetV3-Small
- Output: 38 classes
- PyTorch

## Federated Learning

- Flower `flwr==1.36.0`
- FedAvg
- Flower gRPC network

## Web

- FastAPI backend
- HTML/CSS/JavaScript frontend

## Environment

Project đang dùng virtual environment:

```text
.venv
Python 3.11.x
```

---

# 4. Dataset PlantVillage

Protocol dataset hiện được khóa theo manifest:

```text
Total images: 54,305
Classes:      38
Train:        38,984
Validation:    4,400
Test:         10,921
```

Các file quản lý dataset:

```text
data/dataset_manifest.json
data/partitions_stage2_scratch_v3/
```

Các partition quan trọng:

```text
label100
label1
label01
quantity01
label_quantity01
```

Ý nghĩa thực nghiệm:

| Partition | Ý nghĩa |
|---|---|
| `label100` | gần IID / label distribution tương đối cân bằng |
| `label1` | Non-IID vừa |
| `label01` | Non-IID mạnh |
| `quantity01` | quantity skew / data starvation |
| `label_quantity01` | kết hợp label-skew + quantity-skew |

---

# 5. MobileNetV3-Small

Model nằm tại:

```text
src/model/mobilenetv3.py
```

Module hỗ trợ:

- tạo model;
- get parameters;
- set parameters;
- compute model hash;
- kiểm tra model state.

Global model và local model phải có cùng:

- architecture;
- 38 output classes;
- preprocessing;
- tensor order;
- parameter shape.

Điều này rất quan trọng để FedAvg aggregate đúng.

---

# 6. Quy trình Federated Learning theo từng round

Giả sử tại round `t`, server đang có global weights:

```text
W_t
```

## Bước 1 — Server gửi global model

Server gửi:

```text
global model parameters
round id
model version
parent model hash
local epochs
probe port
```

Client nhận và load vào MobileNetV3-Small.

---

## Bước 2 — Client local training

Client dùng dữ liệu local riêng:

```text
D_k
```

Train model trong một số local epochs.

Sau local training:

```text
W_k^(t+1)
```

Client tính:

```text
loss
accuracy
train duration
num examples
trained update norm
sent update norm
model hash
```

---

## Bước 3 — Client gửi model update

Client gửi về server:

```text
trained model parameters
num_examples
loss
accuracy
training time
update norm
parent model hash
round
model version
probe RTT
client id
data mode
```

Trong FL, raw images **không được gửi về server**.

---

## Bước 4 — Server kiểm tra update

Server không tin trực tiếp các metric client tự khai.

Server tự kiểm tra:

```text
parameter keys
shape
dtype
NaN / Inf
parent model hash
received update norm
```

Server tự tính:

```text
||W_client - W_global||_2
```

Giá trị này được gọi là:

```text
server_received_update_norm
```

---

## Bước 5 — FedAvg

FedAvg weighted average:

```text
W_(t+1) = Σ_k (n_k / N) * W_k
```

Trong đó:

```text
n_k = số mẫu của client k
N   = tổng số mẫu của các client tham gia
```

Client có nhiều dữ liệu hơn sẽ có trọng số lớn hơn trong aggregation.

---

## Bước 6 — Global model mới

Sau aggregation:

```text
W_(t+1)
```

Global model mới được:

- evaluate;
- checkpoint;
- dùng làm model đầu vào của round tiếp theo.

---

# 7. Kiểm chứng FedAvg

Project có cả:

1. FedAvg tự triển khai;
2. Flower FedAvg.

Các test parity dùng để chứng minh kết quả hai bên tương đương trong sai số floating point.

Các test liên quan:

```text
tests/test_fedavg.py
tests/test_flower_parity.py
tests/test_real_flower_parity.py
```

Mục tiêu:

```text
manual FedAvg ≈ Flower FedAvg
```

---

# 8. Model lineage và hash

Để biết client train từ đúng global model, hệ thống sử dụng:

```text
round
model_version
parent_model_hash
```

Server lưu parent global model theo round.

Client trả:

```text
parent_model_hash
```

Server kiểm tra hash này trước khi aggregate.

Điều này giúp phát hiện:

- stale model;
- wrong round;
- client train từ model khác;
- lỗi truyền model.

---

# 9. Update norm

Hệ thống phân biệt hai metric:

```text
trained_update_norm
sent_update_norm
```

## trained_update_norm

Update hợp lệ sau local training nhưng trước fault injection.

## sent_update_norm

Update thực tế client gửi server sau khi apply fault mode.

Server tiếp tục tự tính:

```text
server_received_update_norm
```

Đây mới là metric authoritative để anomaly detection.

Integer buffers như:

```text
BatchNorm.num_batches_tracked
```

không được đưa vào L2 update norm để tránh làm sai metric.

---

# 10. Faulty / malicious clients

Project hỗ trợ các fault mode:

```text
normal
no_train
random_weights
scaled_update
noise_update
```

## 10.1. no_train

Client không train và gửi lại global model gần như nguyên vẹn.

Kỳ vọng:

```text
update norm ≈ 0
```

Server gắn cờ:

```text
near_zero_update
```

---

## 10.2. random_weights

Client gửi random model parameters.

Kỳ vọng:

```text
update norm rất lớn
```

---

## 10.3. scaled_update

Client lấy update hợp lệ và nhân hệ số lớn.

Ví dụ:

```text
ΔW_attack = 10 * ΔW_normal
```

---

## 10.4. noise_update

Client thêm noise vào weights.

---

# 11. Validator phía server

Validator nằm tại:

```text
src/federated/update_validator.py
```

Các kiểm tra:

```text
parameter key mismatch
shape mismatch
dtype mismatch
NaN / Inf
parent hash mismatch
near-zero update
update norm above threshold
```

Kết quả:

```text
ok
suspicious
rejected
```

Có hai policy:

## Observe only

Suspicious client vẫn được aggregate.

Dùng để nghiên cứu ảnh hưởng của malicious updates.

## Reject suspicious

Suspicious update bị loại trước FedAvg.

---

# 12. Kết quả attack experiment đã kiểm thử

Một experiment với 4 clients:

```text
Client 0 = normal
Client 1 = random_weights
Client 2 = scaled_update
Client 3 = noise_update
```

Ngưỡng:

```text
max_update_norm = 5.0
```

Một lần chạy ghi nhận xấp xỉ:

```text
Normal :   1.755  -> OK
Scaled :  17.56   -> Suspicious
Noise  : 125.31   -> Suspicious
Random :1262.10   -> Suspicious
```

## Observe-only

```text
accepted   = 4
suspicious = 3
rejected   = 0
```

Do server vẫn aggregate malicious updates, distributed loss đã có thể trở thành:

```text
NaN
```

Đây là bằng chứng thực nghiệm quan trọng:

> Vanilla FedAvg không tự phát hiện hoặc loại malicious client updates.

## Reject policy

```text
accepted   = 1
suspicious = 3
rejected   = 3
```

Chỉ update normal được dùng cho FedAvg.

---

# 13. RTT và latency

Project không gọi training time là network latency.

Có hai metric khác nhau.

## 13.1. Per-client RTT probe

Module:

```text
src/federated/network_probe.py
```

Flower server mở một TCP echo probe tại:

```text
Flower port + 1
```

Ví dụ:

```text
Flower = 8089
Probe  = 8090
```

Client đo:

```text
PING -> PONG
```

Metric:

```text
probe_rtt_ms
```

Một lần localhost test:

```text
Client 0 RTT ≈ 1.54 ms
Client 1 RTT ≈ 13.17 ms
Mean RTT     ≈ 7.36 ms
Max RTT      ≈ 13.17 ms
```

Các số localhost chỉ chứng minh cơ chế đo hoạt động, không đại diện cho mạng vật lý nhiều máy.

---

## 13.2. Communication overhead

Server cũng đo:

```text
round_fit_elapsed_ms
```

và ước lượng:

```text
estimated_comm_overhead_ms
    = round elapsed
    - max client local train time
```

Metric này chỉ là **ước lượng communication + serialization overhead**.

Không được gọi là one-way latency chính xác.

---

# 14. Data starvation / data scarcity

Hệ thống tính cho từng client:

```text
num_samples
class_count_present
class_count_missing
largest_class_ratio
label_entropy
entropy_normalized
starvation_score
starvation_level
```

Các mức:

```text
low
medium
high
```

`starvation_score` là heuristic dashboard, không phải một metric học thuật chuẩn.

Trong báo cáo nên công bố các thành phần gốc:

```text
sample count
class coverage
missing classes
entropy
dominant class ratio
```

---

# 15. IID và Non-IID

Các experiment chính:

## Gần IID

```text
label100
```

## Non-IID vừa

```text
label1
```

## Non-IID mạnh

```text
label01
```

## Quantity skew / starvation

```text
quantity01
```

Các experiment phải so sánh:

```text
accuracy
loss
macro-F1
convergence
training time
RTT
client contribution
```

---

# 16. Real-data guard

Một vấn đề nguy hiểm trong project ML là dataset loader có thể fallback sang dummy/synthetic data.

Project hiện có guard rõ ràng.

Client có:

```text
data_mode = real | synthetic
```

Option:

```text
--require-real-data
```

Nếu ảnh thật không resolve, client dừng ngay.

Server có:

```text
--require-real-clients
```

Nếu client báo:

```text
data_mode = synthetic
```

server sẽ reject update với trạng thái:

```text
rejected_synthetic_data
```

Điều này ngăn việc vô tình dùng synthetic experiment làm kết quả luận văn.

---

# 17. PlantVillage preflight

Script:

```text
scripts/preflight_plantvillage.py
```

Chạy:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_plantvillage.py D:\PlantVillage --full-path-check
```

Preflight kiểm tra:

```text
dataset root tồn tại
38 classes
54,305 images
class names khớp
manifest paths resolve
class-count fingerprint SHA-256
```

Chỉ khi preflight PASS mới nên chạy scientific experiment.

---

# 18. Network checkpoint

Flower network server lưu global model cuối tại:

```text
artifacts/checkpoints/federated_network_global.pt
```

Checkpoint chứa:

```text
model_state
source
rounds
num_clients
local_epochs
seed
model_hash
test_metrics
```

Model đã được test load lại và forward:

```text
input  = (1, 3, 224, 224)
output = (1, 38)
```

---

# 19. Inference

Web inference hỗ trợ:

```text
Canonical FedAvg
Network Global
```

Cùng một ảnh có thể so:

```text
Centralized prediction
vs
Federated prediction
```

Chỉ nên dùng Network Global làm bằng chứng khoa học khi artifact đã được verified.

---

# 20. Web dashboard

Web đã tách rõ:

```text
Simulation
Flower Network thật
```

Network mode hỗ trợ:

```text
server address
num clients
rounds
local epochs
require real clients
validator policy
max update norm
```

Dashboard hiển thị:

## Global/network

```text
accepted clients
suspicious clients
rejected clients
mean update norm
mean RTT
max RTT
round fit time
estimated communication overhead
```

## Per client

```text
round
client id
RTT
local training time
data mode
received update norm
status
validation reason
```

---

# 21. Simulation và Network phải được phân biệt

## Simulation

Dùng để:

- test logic;
- debug nhanh;
- kiểm tra FedAvg;
- chạy smoke test.

Không dùng simulation để chứng minh network latency thật.

## Network

Dùng Flower gRPC thật với nhiều process hoặc nhiều máy.

Chỉ network mode mới dùng để báo cáo:

```text
server-client communication
RTT
network experiment
```

---

# 22. Held-out evaluation

Scientific network run phải evaluate global model trên dữ liệu held-out tại server.

Các split:

```text
global_val
global_test
```

Server hỗ trợ:

```text
--evaluation-dataset-root
--evaluation-partition-dir
--require-real-evaluation
```

Nếu real evaluation không chạy được thì artifact không được coi là scientific-valid.

---

# 23. Reproducibility / seed

Các experiment phải có seed rõ ràng.

Recommended seeds:

```text
42
123
2026
```

Seed phải thống nhất giữa:

```text
server
clients
centralized experiment
federated experiment
```

Không nên kết luận dựa trên một seed duy nhất.

---

# 24. Multi-seed reporting

Launcher:

```text
scripts/run_real_multiseed_report.ps1
```

Ví dụ:

```powershell
.\scripts\run_real_multiseed_report.ps1 `
  -DatasetRoot D:\PlantVillage `
  -CentralizedEpochs 10 `
  -FederatedRounds 10 `
  -Seeds 42,123,2026
```

Kết quả nên báo cáo:

```text
mean accuracy ± std
mean macro-F1 ± std
loss
convergence
training time
RTT
```

---

# 25. Report experiment runner

Script:

```text
scripts/run_report_experiments.py
```

## Smoke mode

```powershell
.\.venv\Scripts\python.exe scripts\run_report_experiments.py --mode smoke
```

Dùng kiểm thử pipeline.

Nếu không có raw PlantVillage, có thể dùng synthetic fallback.

Output sẽ đánh dấu:

```text
scientific_accuracy_valid = false
```

Không dùng accuracy smoke trong luận văn.

## Real mode

```powershell
.\.venv\Scripts\python.exe scripts\run_report_experiments.py `
  --mode real `
  --dataset-root D:\PlantVillage `
  --epochs 10 `
  --rounds 10 `
  --seed 42
```

Real mode fail-fast nếu thiếu ảnh thật.

---

# 26. Scientific network orchestration

Script một lệnh:

```text
scripts/run_local_5client_scientific_network.ps1
```

Ví dụ:

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 5 `
  -LocalEpochs 1 `
  -Seed 42
```

Script tự động:

```text
full PlantVillage preflight
start Flower server
start 5 Flower clients
require real clients
held-out val/test
RTT measurement
update validation
checkpoint save
artifact validation
verified promotion
```

Dry-run:

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 3 `
  -Seed 123 `
  -DryRun
```

---

# 27. Artifact provenance

Các checkpoint/metrics lịch sử không tự động được coi là kết quả khoa học.

Project có provenance guard để phân biệt:

```text
legacy / unverified
scientific-valid
```

Không được sửa thủ công hash/metadata để làm artifact cũ trở thành verified.

---

# 28. Network artifact validation

Script:

```text
scripts/validate_network_artifact.py
```

Validator yêu cầu:

```text
metrics file tồn tại
checkpoint tồn tại
require_real_clients = true
require_real_evaluation = true
đúng số clients
đúng số rounds
held-out test metrics tồn tại
seed checkpoint/server khớp
checkpoint source đúng
model_state tồn tại
recomputed model hash khớp checkpoint model_hash
audit đủ rounds
audit đủ clients
mọi client data_mode = real
không synthetic rejection
```

Nếu bất kỳ điều kiện nào fail:

```text
ok = false
```

Artifact không được promote.

---

# 29. Artifact tamper protection

Validator tự tính lại:

```text
compute_model_hash(model_state)
```

sau đó so với:

```text
checkpoint["model_hash"]
```

Nếu một tensor bị sửa nhưng metadata/hash cũ vẫn giữ nguyên:

```text
checkpoint_model_hash_matches = false
```

Artifact bị reject.

Test:

```text
tests/test_network_artifact_validation.py
```

đã xác minh:

```text
valid artifact -> PASS
tampered checkpoint -> FAIL đúng thiết kế
```

---

# 30. Verified artifact promotion

Script:

```text
scripts/promote_verified_network_run.py
```

Promotion chỉ chạy khi validator PASS.

Output:

```text
artifacts/verified_network/<promotion_id>/
```

Gồm:

```text
network_metrics.json
global_model.pt
validation.json
verified_network_manifest.json
```

Manifest lưu:

```text
seed
rounds
num_clients
real-client flags
real-evaluation flags
held-out test metrics
SHA-256 của từng artifact
```

Chỉ artifact đã promote vào `verified_network` mới nên dùng làm bằng chứng scientific network.

---

# 31. Các test quan trọng

Các nhóm test chính:

## Core

```text
test_fedavg.py
test_model.py
test_parameters.py
test_partition.py
test_integration.py
test_round_trip_feedback_loop.py
```

## Flower

```text
test_flower_parity.py
test_real_flower_parity.py
test_flower_live_socket.py
test_flower_multi_round_network.py
test_audited_flower_network.py
```

## 5-client network

```text
test_five_client_network.py
```

## Faulty clients

```text
test_faulty_client_network_detection.py
test_attack_matrix_network.py
```

## RTT/checkpoint

```text
test_network_rtt_and_checkpoint.py
```

## Data starvation

```text
test_data_scarcity.py
```

## Scientific/provenance guard

```text
test_artifact_scientific_status.py
test_web_provenance_guard.py
test_promote_verified_run.py
test_network_evaluation_guard.py
test_network_heldout_evaluation.py
test_network_reproducibility_guard.py
test_dataset_real_mode.py
test_client_partition_preflight.py
```

## Artifact validation

```text
test_network_artifact_validation.py
```

---

# 32. Regression hiện tại

Regression tổng gần nhất:

```text
60 passed
2 warnings
```

Hai warnings là dependency warnings từ `typer/click`, không phải lỗi logic project.

Các khả năng đã được chứng minh bằng automated tests:

```text
FedAvg correctness                     PASS
Flower parity                          PASS
Real Flower socket                     PASS
Multi-round network                    PASS
1 server + 5 client processes          PASS
Round-to-round feedback                PASS
Server-side update validator           PASS
No-train detection                     PASS
Random/scaled/noise detection          PASS
Observe/reject policy                  PASS
Per-client RTT                         PASS
Data starvation metrics                PASS
Network checkpoint                     PASS
Checkpoint inference                   PASS
Held-out evaluation guard              PASS
Reproducibility/seed guard              PASS
Real-data guard                        PASS
Web simulation/network separation      PASS
Artifact provenance guard              PASS
Artifact tamper detection              PASS
```

---

# 33. Technical-ready và Scientific-ready

Đây là điểm rất quan trọng.

## Technical demo ready

Hiện tại:

```text
YES
```

Tức là có thể demo:

- Flower server;
- clients;
- FedAvg;
- RTT;
- validator;
- faulty clients;
- checkpoint;
- inference;
- dashboard.

## Scientific reporting ready

Hiện tại:

```text
NO
```

Lý do chính:

1. Raw PlantVillage chưa có trên máy đang audit.
2. Canonical historical artifacts chưa verified provenance.
3. Network global artifact hiện tại chủ yếu từ smoke/synthetic run.
4. Chưa chạy full real-data multi-seed experiment.

Đây không còn là lỗi core architecture.

---

# 34. Những số liệu KHÔNG được dùng làm kết quả luận văn

Không dùng:

```text
smoke accuracy
synthetic-data accuracy
localhost RTT như đại diện Internet/LAN thật
checkpoint unverified
legacy canonical metrics chưa provenance-pass
```

Ví dụ smoke accuracy khoảng:

```text
2.63% ≈ 1/38
```

chỉ phản ánh random/synthetic baseline.

---

# 35. Những số liệu nên dùng trong báo cáo cuối

Sau khi có real PlantVillage:

## Centralized

```text
accuracy
macro-F1
loss
training time
```

## Federated

```text
global accuracy
global macro-F1
global loss
convergence per round
training duration
RTT per client
communication overhead
client sample count
FedAvg contribution
```

## Non-IID / starvation

```text
classes present
classes missing
entropy
dominant class ratio
num samples
starvation level
```

## Attack experiment

```text
received update norm
accepted/suspicious/rejected
accuracy before/after attack
loss before/after attack
convergence degradation
observe vs reject
```

---

# 36. Đề xuất bảng kết quả cho luận văn

## Bảng 1 — Centralized vs FL

| Method | Accuracy | Macro-F1 | Loss | Train Time |
|---|---:|---:|---:|---:|
| Centralized | ... | ... | ... | ... |
| FL IID | ... | ... | ... | ... |
| FL Non-IID | ... | ... | ... | ... |

## Bảng 2 — Non-IID

| Scenario | Accuracy | Macro-F1 | Rounds to converge | Loss |
|---|---:|---:|---:|---:|
| label100 | ... | ... | ... | ... |
| label1 | ... | ... | ... | ... |
| label01 | ... | ... | ... | ... |

## Bảng 3 — Data starvation

| Client | Samples | Classes | Missing classes | Entropy | Starvation level |
|---|---:|---:|---:|---:|---|
| C0 | ... | ... | ... | ... | ... |
| C1 | ... | ... | ... | ... | ... |

## Bảng 4 — Attack

| Attack | Update norm | Suspicious | Rejected | Global accuracy |
|---|---:|---|---|---:|
| Normal | ... | No | No | ... |
| No-train | ... | Yes | ... | ... |
| Random | ... | Yes | ... | ... |
| Scaled | ... | Yes | ... | ... |
| Noise | ... | Yes | ... | ... |

---

# 37. Quy trình demo đề xuất

Thứ tự demo nên là:

## Bước 1

Giới thiệu architecture:

```text
Centralized vs Federated
```

## Bước 2

Mở dashboard server.

## Bước 3

Khởi động 5 clients.

## Bước 4

Cho thấy:

```text
client connected
round id
model received
local train
update sent
```

## Bước 5

Cho thấy server:

```text
update norm
RTT
accepted/suspicious/rejected
FedAvg
```

## Bước 6

Chạy ít nhất 2 rounds để chứng minh feedback loop.

## Bước 7

Bật một faulty client.

Ví dụ:

```text
no_train
```

Cho thấy server flag suspicious.

## Bước 8

Nếu còn thời gian, dùng:

```text
random_weights
```

so sánh observe vs reject.

## Bước 9

Dùng final global model inference cùng ảnh với centralized model.

---

# 38. Các câu giải thích quan trọng khi bảo vệ

## Vì sao dùng FedAvg?

FedAvg là baseline tiêu chuẩn trong Federated Learning, aggregate model updates dựa trên số mẫu của từng client.

## Raw data có gửi về server không?

Không. Client train local và chỉ gửi model parameters/update + metadata cần thiết.

## FedAvg có tự phát hiện malicious client không?

Không. Vanilla FedAvg chỉ aggregate updates. Validator trong project là lớp bổ sung để nghiên cứu update bất thường.

## RTT đo như thế nào?

Project dùng TCP PING/PONG probe riêng để đo per-client round-trip time, tránh nhầm với local training time.

## Vì sao có data starvation?

Federated clients có thể sở hữu lượng dữ liệu và phân phối label rất khác nhau. Client thiếu sample hoặc thiếu lớp có thể tạo local update kém đại diện.

## Vì sao phải có Centralized baseline?

Để đánh giá trade-off giữa học tập trung và federated trong cùng model/dataset protocol.

---

# 39. Lệnh quan trọng

## Full dataset preflight

```powershell
.\.venv\Scripts\python.exe scripts\preflight_plantvillage.py D:\PlantVillage --full-path-check
```

## Scientific 5-client local network

```powershell
.\scripts\run_local_5client_scientific_network.ps1 `
  -DatasetRoot D:\PlantVillage `
  -Scenario label1 `
  -Rounds 5 `
  -LocalEpochs 1 `
  -Seed 42
```

## Real report experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_report_experiments.py `
  --mode real `
  --dataset-root D:\PlantVillage `
  --epochs 10 `
  --rounds 10 `
  --seed 42
```

## Multi-seed

```powershell
.\scripts\run_real_multiseed_report.ps1 `
  -DatasetRoot D:\PlantVillage `
  -CentralizedEpochs 10 `
  -FederatedRounds 10 `
  -Seeds 42,123,2026
```

## Pre-demo gate

```powershell
.\.venv\Scripts\python.exe scripts\pre_demo_check.py --dataset-root D:\PlantVillage
```

Mục tiêu cuối:

```text
overall = ready
```

---

# 40. Trạng thái hiện tại của project

## Đã hoàn thành

```text
Centralized pipeline
MobileNetV3-Small contract
FedAvg math
Flower server/client
5-client network
multi-round feedback
RTT measurement
weight inspection
fault modes
server-side validator
observe/reject policy
Non-IID partitions
data starvation metrics
network checkpoint
network inference
held-out evaluation guard
real-data guard
artifact provenance
artifact tamper guard
verified promotion pipeline
web simulation/network separation
```

## Còn cần làm để hoàn tất luận văn khoa học

```text
1. Đưa raw PlantVillage đúng protocol vào máy/GPU node.
2. Full preflight PASS.
3. Chạy Centralized real-data nhiều seed.
4. Chạy FL label100/label1/label01/quantity01 nhiều seed.
5. Chạy network scientific 5 clients.
6. Chạy attack experiments real-data.
7. Promote verified artifacts.
8. Tổng hợp mean/std và biểu đồ.
9. Dùng chỉ verified results trong báo cáo cuối.
```

---

# 41. Kết luận

Project hiện đã đạt mức **technical demo ready**.

Các phần quan trọng của Federated Learning đã được xây dựng và kiểm thử:

```text
Flower/FedAvg
1 server + 5 clients
real socket communication
round-to-round feedback
per-client RTT
server-side update validation
fault injection
Non-IID/data starvation
checkpoint/inference
scientific/provenance guard
```

Regression gần nhất:

```text
60 tests passed
```

Phần còn thiếu để đạt **scientific reporting ready** chủ yếu là tài nguyên/dữ liệu thực nghiệm:

```text
raw PlantVillage
real-data training
multi-seed runs
verified final artifacts
```

Không nên tiếp tục thay đổi lớn core architecture nếu không có yêu cầu mới. Bước tiếp theo nên tập trung vào chạy real experiments, thu thập metrics và hoàn thiện báo cáo/biểu đồ.

---

# 42. File tham chiếu quan trọng

```text
README.md
HANDOFF_CHECKLIST.md
AUDIT_THEO_YEU_CAU_2026-09-30.md
TAI_LIEU_TONG_HOP_FEDERATED_LEARNING.md

src/model/mobilenetv3.py
src/data/plantvillage.py
src/data/partition.py
src/centralized/train.py
src/centralized/evaluate.py
src/centralized/inference.py
src/federated/client_app.py
src/federated/server_app.py
src/federated/strategy.py
src/federated/update_validator.py
src/federated/network_probe.py
src/federated/faulty_client.py
src/federated/telemetry.py
src/federated/weight_inspector.py

scripts/preflight_plantvillage.py
scripts/run_report_experiments.py
scripts/run_real_multiseed_report.ps1
scripts/start_real_flower_server.ps1
scripts/start_real_flower_client.ps1
scripts/run_local_5client_scientific_network.ps1
scripts/validate_network_artifact.py
scripts/promote_verified_network_run.py
scripts/pre_demo_check.py
scripts/final_acceptance.py
```

---

**Tài liệu này là tài liệu tổng hợp chính của project tại trạng thái hiện tại.**

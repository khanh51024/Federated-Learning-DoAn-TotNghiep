# Kế hoạch hoàn thiện huấn luyện GĐ2 cho GPT-5.6 Sol

> Nghiệm thu và sửa tiếp ngày 10/09/2026: xem [STAGE2_ACCEPTANCE.md](STAGE2_ACCEPTANCE.md) để biết bằng chứng trọng số, CPU/CUDA, các lỗi đã sửa và điều kiện GĐ2 còn mở.

> Kế hoạch gốc đã được triển khai ngày 09/09/2026. Trạng thái, bằng chứng chạy
> và giới hạn hiện hành nằm tại `../IMPLEMENTATION_STATUS.md`. Tên thư mục sau
> triển khai là `gd2_federated_learning`; các tên cũ bên dưới được giữ để bảo
> toàn ngữ cảnh audit trước khi rename.

Ngày: 09/09/2026. Trạng thái: **chỉ kế hoạch, chưa triển khai sửa code hoặc đổi tên**. Các đường dẫn code dưới đây tính từ `gd2_noniid_partition/`. Kết quả thực đo nằm trong [bằng chứng kiểm tra](audit_20260909/README.md), trích dẫn [n] tra ở [danh mục IEEE](REFERENCES_IEEE.md).

## 1. Kết luận đối chiếu đề cương

**Chưa đáp ứng đầy đủ GĐ2.** Đã có nền tảng mô phỏng FedAvg chạy được ở quy mô nhỏ; chưa có bằng chứng đo suy giảm so với Centralized trên nhiều mức non-IID. PDF tr. 3 yêu cầu lệch nhãn, lệch số lượng, lệch đặc trưng và phép đo này; tr. 1–3 yêu cầu báo cáo thêm Local-only làm mốc [1]. Chạy mô phỏng Flower trên một máy đúng phạm vi đề cương; không bắt buộc triển khai nhiều máy thật ở bước này.

Các mốc hiện có:

- **Có và đã kiểm tra nhỏ:** MobileNetV3-Small 38 lớp; FedAvg cân theo số mẫu; local training CPU và test bước AMP CUDA; validation; early stopping; LR scheduler; ghi checkpoint atomic; evaluate độc lập ra metrics và confusion matrix CSV [2].
- **Một phần:** một renderer tqdm; config YAML/preflight; lịch sử từng round; resume; log lỗi. Chưa đáp ứng đầy đủ tiến độ mọi thao tác, log từng local epoch, cấu hình có hiệu lực và resume tái lập.
- **Thiếu:** nối feature skew vào loader train; workflow Centralized/Local-only + FedAvg sweep cùng protocol; biểu đồ train và so sánh; thống kê weights/update; cảnh báo kèm gợi ý dựa trên kết quả; snapshot theo khoảng round tùy chọn.
- **Chưa nghiệm thu:** full train hội tụ, simulation CUDA đầy đủ, early stop qua Flower end-to-end, resume liên tục so với bị ngắt, TTY live progress và runtime cleanup sạch.

Không đưa FedProx, FedBN, SCAFFOLD, MOON, lượng tử hóa hay differential privacy vào bản sửa GĐ2; chúng thuộc các giai đoạn sau [1]. Có thể ghi gợi ý GĐ3 trong báo cáo, không tự triển khai hoặc tự kết luận sẽ tăng accuracy.

## 2. Phát hiện cần sửa, theo mức ưu tiên

### P0 — ảnh hưởng tính đúng hoặc chặn luồng sử dụng

**F01. Feature skew và augmentation không đến được đầu vào train.** `fl_training/data.py:74–96` chỉ giữ vài trường manifest; `build_training_loader:141` dùng transform mặc định; `__getitem__:123` gọi `t(img)` không truyền seed. `src/data/transforms.py:218` chỉ augmentation khi `rng_seed` tồn tại. `client_app.py:89` không truyền profile. Kết quả: nhãn thí nghiệm feature skew có thể không phản ánh ảnh thực client nhận; crop/flip cũng không diễn ra như dự định. Phải dùng profile đã lưu và seed theo sample/client/epoch, tái sử dụng logic `src/data` thay vì tạo pipeline cạnh tranh. Không sửa ảnh gốc.

**F02. Evaluate với config sai root.** `cli.py:282` dùng `cp_file.parents[2]`; đã tái hiện lỗi với checkpoint smoke. Auto-evaluate sau train gọi cùng đường này. Resolve đường dẫn theo config/package rõ ràng, không suy root từ độ sâu thư mục output. Tạo output directory khi cần; phản ánh lỗi evaluate vào exit/status của cả workflow.

**F03. Seed/resume chưa tái lập.** Server khởi tạo model không seed trước; client seed loader nhưng chưa seed khởi tạo/dropout; `restore_rng_state` có định nghĩa nhưng server resume không gọi. Trong simulation, actor được tạo/hủy và worker được tái sử dụng [4], nên RNG ngầm không đáng tin. Đặt seed server trước model; seed mỗi client theo training seed/client/round; seed augmentation theo ảnh/epoch; restore server RNG sau các thao tác khởi tạo khi resume. Kiểm tra trong cùng môi trường CPU trước; không hứa bitwise trên mọi nền tảng [6].

**F04. Reply validation chưa bảo vệ tổng hợp.** `server_app.py:199–225` chỉ kiểm tra số reply, không xác minh error/source/client/round và số mẫu so với manifest. `strategy.py:17` nhận `expected_shapes` nhưng không dùng; không chặn n_k âm, client trùng, keys/shape khác. Probe n_k âm đã trả kết quả tổng hợp sai. Xác nhận node→partition bằng query/context thay vì thứ tự `grid.get_node_ids()`; kiểm tra toàn bộ reply trước cập nhật model. Mặc định fail round khi thiếu client bắt buộc; không commit nửa round. Nếu hỗ trợ partial participation, định nghĩa rõ sampling, quorum và chuẩn hóa trọng số chỉ trên client hợp lệ [2].

**F05. Cấu hình hiển thị nhưng bị bỏ qua.** `fraction_train` không dùng khi chọn node; `max_concurrent_clients` không truyền vào cơ chế giới hạn runtime; client hard-code `num_workers=0`; renderer chỉ nhận `not no_progress`, bỏ `output.progress`. Hoặc thực thi đúng, hoặc từ chối giá trị chưa hỗ trợ; không chấp nhận rồi âm thầm bỏ qua. Bổ sung validation số hữu hạn, miền LR/batch/patience/factor, min_train_nodes ≤ số client được chọn, CUDA/resources và số client manifest.

### P1 — quan sát, khôi phục, sản phẩm đầu ra và nghiệm thu GĐ2

**F06. Progress có thể bị chặn.** `cli.py:222` gọi blocking `stdout.readline()` trước `renderer.poll()`. Không có stdout mới thì tiến độ event bị trì hoãn. Chỉ có phase/round event, chưa có trạng thái từng client/epoch, checkpoint, evaluate và plot. `progress.py` bỏ dòng JSON chưa hoàn chỉnh mà vẫn tăng offset, có thể mất event. Chưa test TTY trong lần này.

**F07. Run identity và resume best không nhất quán.** Launcher và server đọc YAML rồi tạo timestamp riêng; smoke xác nhận ID lệch 5 giây. Resume tạo run mới, mang best weights trong last nhưng nếu không có best mới thì không tạo `best.pt` tại run mới. Auto-test chỉ chạy nếu thấy best.pt, nên có thể bị bỏ qua. Trạng thái đã early-stopped cũng cần quy tắc rõ khi resume; không tự tiếp tục nếu người chạy chưa yêu cầu.

**F08. Logging/checkpoint chưa đủ yêu cầu epoch.** `task.py` cộng metric qua mọi local epoch rồi trả chung; server bỏ metric từng client sau tổng hợp; history CSV chỉ xuất cuối train (`server_app.py:299`). Chưa có weight/update norm, snapshot định kỳ, manifest ánh xạ model/class/preprocessing độc lập. Semantic hash bỏ một số thiết lập ES/LR/AMP; bundle hash chỉ theo tên, không bảo vệ nội dung. Cần ghi incremental, schema/version/provenance và resume theo round đã commit [7].

**F09. Chưa có biểu đồ và diagnostic train.** `scripts/generate_plots.py` và `sweep_alpha.py` hiện phục vụ phân hoạch, không vẽ learning curve/accuracy gap từ train run. Evaluate có CM CSV nhưng chưa heatmap, normalized CM, biểu đồ per-class hoặc so sánh baseline. Warning hiện chủ yếu thông báo lỗi kỹ thuật, chưa tổng hợp gợi ý từ metric.

**F10. Thiếu protocol thực nghiệm GĐ2.** `configs/partition_training.yaml` chỉ chuẩn bị IID và label-skew alpha=0.1. Có nhiều manifest phân hoạch cũ nhưng không đồng nghĩa đã train chúng. CLI chưa có Centralized/Local-only/sweep training. Cần baseline cùng train union, val/test, class mapping, initialization và preprocessing; thay alpha có kiểm soát [1], [3].

**F11. Môi trường/smoke có điểm yếu đã quan sát.** Test nhỏ validation chỉ 15 lớp, test chỉ 1 lớp; không đại diện hiệu quả. Ray có access violation lúc shutdown dù code 0; cần ghi “completed_with_warnings” hoặc trạng thái tương đương, kiểm tra worker cleanup. `run_simulation` deprecated trong Flower 1.36.0; tài liệu khuyến nghị WSL2 cho Windows [4], [5]. `pyproject.toml` cho phép phiên bản Flower quá rộng so với API đang dùng, lock CUDA cần hướng dẫn index wheel và Python tương thích. Không nâng toàn bộ dependencies chỉ vì có phiên bản mới.

## 3. Quy ước round, epoch và model đầu ra

Một local epoch = một lượt qua dữ liệu của một client. Một round = các client tham gia train `local_epochs=E`, gửi weights, server FedAvg, validation, checkpoint. Early stopping toàn cục tính theo **round**, không cộng epoch giữa các client. Tổng local epoch mỗi client bằng số round client thực sự tham gia × E.

Tổng hợp FedAvg sau E local epoch mỗi round đã có; không cần thêm một thuật toán “tổng hợp sau N epoch” riêng. Thêm `checkpoint.every_n_rounds` (đề xuất 5, smoke dùng 1), `keep_last_n` (đề xuất 3); luôn lưu last và best. Đặt tên `round_0005.pt`, kèm local_epochs và participation để người đọc hiểu mốc epoch. Không lấy trung bình checkpoint khác round trừ khi sau này có thí nghiệm riêng.

`best.pt` là tốt nhất theo validation, có thể là round 0; `last.pt` là trạng thái cuối đã commit. Xuất `model_final.pt` là bản inference lấy từ best, kèm class names, input shape, normalization, seed, partition fingerprint, round và metric. Không nạp pretrained qua mạng khi evaluate checkpoint đã đủ weights [7], [8].

## 4. Trình tự triển khai và điều kiện nghiệm thu

### Bước A — sửa tính đúng, giữ phạm vi nhỏ

Sửa F01–F05 trong `data.py`, `client_app.py`, `server_app.py`, `strategy.py`, `config.py`, `cli.py`. Giữ MobileNet/FedAvg/BN policy hiện có; integer `num_batches_tracked` lấy max là quyết định triển khai cần ghi rõ, không đổi thành FedBN.

Nghiệm thu: cùng sample/client/epoch cho cùng tensor; epoch khác có augmentation thay đổi ở phép thử có kiểm soát; bật/tắt profile tạo khác biệt đúng; clean validation/test không đổi. Test reply trùng/round sai/error/n_k âm/shape sai đều bị từ chối trước aggregate. Test config không dùng được phải bị báo lỗi. Evaluate cùng checkpoint có/không config cho kết quả khớp trên cùng manifest. Chạy một bước train thật, không chỉ test bằng mock.

### Bước B — một thanh tiến độ và log bền vững

Launcher sở hữu đúng một renderer. Tách đọc stdout sang reader thread/queue hoặc ghi subprocess trực tiếp vào runtime.log để polling event không blocking. Có phase prepare/preflight/train client+epoch/aggregate/validation/checkpoint/evaluate/plot; chỉ hiển thị % khi có tổng công việc biết được. Với thao tác chưa biết tổng dùng trạng thái hoạt động/thời gian trên cùng thanh, không tạo thanh phụ.

Mô phỏng một máy có thể dùng log append riêng từng client và launcher đọc, tránh nhiều worker ghi chung một JSONL; log mỗi epoch hoặc heartbeat được throttle, không log tensor hoặc mỗi batch mặc định. Thiết kế này chỉ áp dụng simulation; không mô tả shared filesystem như cơ chế cho nhiều thiết bị thật.

Schema tối thiểu: schema_version, run_id, attempt_id, seq, UTC, phase, round, client_id, local_epoch, processed_examples, loss/accuracy, LR. Round log thêm val_loss/accuracy/macro-F1, best_round, bad_rounds, duration, LR dùng và LR round tới. Weight log ghi global/update L2 norm, delta L2/relative norm có epsilon, min/max, nonfinite count và n_k/aggregation weight; full tensor chỉ lưu trong checkpoint.

Ghi `client_epochs.jsonl`, `rounds.jsonl`, `weights.jsonl`; đồng bộ history CSV sau round commit hoặc tạo lại từ checkpoint/log. Run ID do launcher tạo một lần, truyền server. Giữ phần dòng event chưa hoàn chỉnh cho lần đọc sau. Mọi lỗi phải có failed/interrupted status và ngữ cảnh client/round, không chỉ nằm trong runtime.log.

Nghiệm thu: subprocess im lặng nhưng event vẫn cập nhật; TTY chỉ một bar; non-TTY đọc được; `progress=false` và `--no-progress` có hiệu lực; LR log khớp history. Ngắt sau round đã commit vẫn đọc được metric/checkpoint và không nhân đôi dữ liệu khi resume.

### Bước C — resume, snapshot và giới hạn runtime

Sửa F07–F08: last là commit source; best có thể khôi phục từ last khi tạo run tiếp nối. Có parent_run_id, attempt_id và mapping checkpoint. Hash nội dung manifest/class mapping/preprocessing cùng cấu hình ảnh hưởng train; cho phép override các trường vận hành được tài liệu hóa như output path, tăng max_rounds. Không cho checkpoint thiếu hash mặc nhiên vượt kiểm tra khi cần bảo vệ protocol.

Giữ fresh SGD mỗi client mỗi round như hiện tại, ghi rõ momentum không được duy trì giữa round. Nếu sau này chọn persistent optimizer phải có checkpoint riêng cho state client. ES: giữ global validation loss/min_delta/warmup/patience; test plateau và cải thiện nhỏ. LR controller hiện dùng `bad_rounds > patience`, khớp semantics cần đối chiếu [9]; không sửa vô cớ.

Nghiệm thu: 2 round liên tục so với 1 round + resume tới 2 trong cùng CPU env có weights/history khớp trong tolerance công bố. Test resume không tạo best mới vẫn có best.pt để evaluate; resume từ trạng thái stopped không âm thầm train tiếp. Test snapshot retention chỉ xóa snapshot được quản lý của run hiện tại. Test lỗi/timeout/Ctrl+C giữ last hợp lệ và dọn process của chính run.

Ray concurrency phải được kiểm tra bằng số actor thực tế, không chỉ xem YAML. Giữ venv/lock đang chạy được; xác minh lỗi shutdown trước full train. Nếu Windows tiếp tục lỗi, chuẩn bị lệnh WSL2 và ánh xạ đường dẫn trong tài liệu; không tự cài lại hệ thống trong bản sửa tối thiểu. Chuyển `run_simulation` sang `flwr run` chỉ khi đã có kiểm chứng tương thích và không làm mất progress/control hiện có [4], [5].

### Bước D — báo cáo/biểu đồ và gợi ý

Thêm module reporting và diagnostics cùng lệnh `report --run-dir ...` (lệnh **đề xuất**, chưa tồn tại). Chạy reporting tự động sau train/evaluate thành công, đồng thời hỗ trợ tạo lại từ artifacts mà không train lại.

Đầu ra: learning curves train/val loss và accuracy; val macro-F1; LR; update norm; raw CM và row-normalized CM 38×38 có nhãn; per-class F1/support; bảng CSV/JSON kết quả. PNG để xem, PDF/vector cho báo cáo khi cần; ghi metric, đơn vị, run/checkpoint và denominator. Với local train metric trước aggregate và global validation sau aggregate, chú thích chúng không đo đúng cùng một model; không suy overfit chỉ từ chênh lệch đó.

Diagnostics dùng quy tắc minh bạch, ngưỡng cấu hình được: NaN/Inf; plateau; divergence có nhiều điểm dữ liệu; lớp ít support/recall kém; client ít mẫu; update norm lệch; không có cải thiện so với round 0; OOM; runtime shutdown warning. Mỗi cảnh báo có evidence, severity, điều kiện kích hoạt và gợi ý cụ thể (LR/batch/kiểm tra dữ liệu/chạy thêm validation). Không tự sửa siêu tham số hoặc file code dựa trên test set; không gọi LLM ở mỗi epoch. Khi thiếu dữ liệu thì không kết luận.

Nghiệm thu: số liệu CSV khớp checkpoint/evaluate; tổng CM bằng số mẫu, mỗi hàng normalized có support cộng bằng 1; xử lý hàng support=0; ảnh không cắt nhãn; report chạy lại không overwrite kết quả checkpoint khác mà mất danh tính. Synthetic diagnostic tests cho từng quy tắc đủ điều kiện, không ép smoke phải accuracy cao.

### Bước E — hoàn thành khả năng thực nghiệm GĐ2

Thêm mode/lệnh train Centralized và Local-only tái sử dụng `train_local`/evaluate/checkpoint/report, tránh ba trainer sao chép. Nếu có baseline GĐ1 ngoài workspace, chỉ tái sử dụng sau khi xác minh checkpoint/metrics khớp data hash/class/preprocessing/model/seed; hiện chưa có bằng chứng baseline đó trong workspace.

Protocol tối thiểu để pilot: IID + label skew alpha={0.1, 1, 100}; quantity skew qalpha={0.1, 1}; feature skew={none, moderate} trên cùng phân hoạch nhãn đã chọn. Tái sử dụng `none` và baseline khi fingerprints giống nhau, không chạy tích Descartes mọi tham số. Alpha≈100 là mức gần IID theo đề cương, vẫn giữ IID riêng để kiểm soát [1], [3]. Mỗi thay đổi chỉ điều chỉnh một trục.

Giữ train union, global val/test group-aware cố định; client data không giao nhau; class order 38 lớp thống nhất. Với feature skew, baseline Centralized phải dùng cùng profile theo nguồn client của từng ảnh, Local-only dùng profile tương ứng; val/test clean dùng chung. Nếu sau này đánh giá domain-shift test, báo cáo riêng, không âm thầm thay protocol chuẩn.

So sánh có cùng initialization/pretrained policy và ngân sách lượt đi qua dữ liệu; ghi thêm optimizer steps và số mẫu xử lý thực tế vì số batch khác nhau. Full participation: R round × E local epochs tương đương R×E lượt qua union, nhưng không đảm bảo số optimizer steps bằng Centralized. Early stopping chọn model bằng validation trong cùng budget tối đa. Local-only báo từng client và trung bình/độ lệch trên cùng global test; nếu cần per-client domain metric thì định nghĩa tập đánh giá riêng, không lấy train accuracy làm fairness.

Tạo `comparison.csv` và biểu đồ accuracy/macro-F1 theo alpha, gap Centralized−FedAvg tính bằng điểm phần trăm cho accuracy. Báo Local-only bên cạnh; thêm thời gian, rounds tới best/stop, chi phí payload ước lượng từ tensor bytes và số client gửi/nhận (ghi rõ chưa bao gồm protocol overhead). Chuẩn bị multi-seed, đề xuất 3 seed ở thí nghiệm chính; pilot 1 seed tiết kiệm tài nguyên, không gọi là kết luận thống kê.

Nghiệm thu code: sweep nhỏ trên manifest nhỏ chạy đủ mode và tạo comparison đúng từ run có provenance. Nghiệm thu khoa học GĐ2: có train thật trên dữ liệu phù hợp và báo cáo khoảng cách theo mức non-IID. **Chỉ sửa code và chạy smoke không hoàn tất nghiệm thu khoa học.** Không bắt buộc kết quả phải đẹp hoặc đơn điệu; báo số liệu quan sát trung thực.

### Bước F — đổi tên và đồng bộ tài liệu

Đổi `gd2_noniid_partition` thành **`gd2_federated_learning`**, phản ánh cả phân hoạch, huấn luyện và đánh giá trong GĐ2. Tên `gd_noniid_partition` trong yêu cầu không tồn tại ở workspace; không tạo thêm một thư mục cho biến thể đó.

Thực hiện sau khi code ổn để tránh trộn lỗi path và lỗi thuật toán. Kiểm tra source/destination tuyệt đối nằm trong workspace và destination chưa tồn tại; dùng một thao tác rename, không copy dataset/runs. Cập nhật README gốc, TRAINING/IMPLEMENTATION_STATUS, `.vscode/settings.json`, tài liệu và path tests cần thiết. Giữ tên package/import. Không thay hàng loạt path trong artifacts lịch sử.

Lưu ý `.venv`/editable install/launcher scripts có thể chứa path tuyệt đối cũ. Sau rename kiểm tra python, imports và metadata; reinstall editable vào đường dẫn mới bằng interpreter đã kiểm chứng nếu cần. Tài liệu hóa recreate venv từ lock khi không portable. Checkpoint cũ chứa absolute dataset/partition path: bổ sung cơ chế relocation override/root mapping được kiểm tra hash, không chỉnh tensor hay phá artifact lịch sử. Nghiệm thu preflight, smoke, evaluate và resume checkpoint trước rename tại tên mới.

## 5. Cấu hình khởi đầu cho bản sửa

Tái sử dụng train_fedavg.yaml hiện có: MobileNetV3-Small pretrained, SGD lr=0.01, momentum=0.9, weight_decay=1e-4; 10 client, E=1, batch 16/eval 32; max_rounds=100, ES warmup=10/patience=10/min_delta=1e-4; LR factor=0.5/patience=3/min_lr=1e-6. Đây là baseline cần pilot, chưa phải cấu hình tối ưu.

Một GPU RTX 2050: khởi đầu concurrency=1, client auto/CUDA, server CPU, num_workers=0, AMP khi CUDA. Phải sửa cấu hình concurrency có hiệu lực trước khi dựa vào nó; khi OOM giảm batch 8 rồi kiểm chứng, ghi thay đổi vào run mới. Không tự giảm batch giữa run làm lệch so sánh mà không log.

Smoke sửa mới: 2 client, 2 round, weights=null, CPU, no download, 32–64 ảnh/client; val/test lấy stratified theo lớp nếu budget cho phép hoặc gắn nhãn smoke thiếu lớp rõ ràng. Có chế độ preflight smoke riêng, giữ kiểm định đủ lớp cho benchmark thật. Timeout ngoài 180 giây; có thể tăng có lý do nếu runtime startup chậm. Early-stop integration riêng dùng fixture val_loss plateau để kiểm tra điều phối, không cố đợi plateau tự nhiên.

## 6. Giảm usage và chi phí chạy

1. Đọc kế hoạch này và evidence summary trước, chỉ mở module liên quan bước đang làm. Không đọc lại toàn bộ PDF hoặc quét ảnh nguồn nếu hash không đổi.
2. Một agent GPT-5.6 Sol; không tạo subagent hay gọi LLM cho diagnostics. Không bàn luận lại kiến trúc đã chốt nếu chưa có blocker cụ thể.
3. Tái sử dụng loader/partition/controller/test có sẵn; thêm regression test cho lỗi thực, không viết lại test chỉ để khớp code. Mỗi bước chạy test liên quan; sau A–D chạy một integration smoke/resume; không lặp full suite sau thay đổi tài liệu.
4. Giữ môi trường đã cài; không tải pretrained khi weights=null; không sao chép 54k ảnh; không hash ảnh lại nếu cache/provenance đủ tin cậy.
5. Chạy report từ logs; ghi weight summary, không dump tensor ra terminal. Snapshot có retention; full benchmark/multi-seed là lệnh bàn giao cho người chạy, không tự chạy trong lượt sửa nếu chưa được yêu cầu.
6. Mỗi bước cập nhật ngắn `IMPLEMENTATION_STATUS.md`: đã sửa gì, test/lệnh/kết quả, blocker. Trạng thái cũ “67/67 pass” không chứng minh tiêu chí mới; giữ evidence thực tế.

## 7. Prompt bàn giao

> Hãy triển khai `gd2_noniid_partition/docs/STAGE2_REPAIR_PLAN.md` ngày 09/09/2026 bằng GPT-5.6 Sol. Đây là lượt thực hiện sửa code, khác với lượt audit chỉ lập kế hoạch. Đọc kế hoạch và evidence summary trước; tiếp tục từ trạng thái thực tế, không viết lại hệ thống. Làm lần lượt A–F: sửa feature skew/augmentation, evaluate path, seed/resume, reply validation và config; hoàn thiện một progress bar/log epoch; checkpoint định kỳ; report/diagnostics; baseline/sweep GĐ2; cuối cùng rename sang gd2_federated_learning và xử lý path/venv/checkpoint cũ. Giữ thuật toán FedAvg/MobileNetV3 và dữ liệu gốc, không triển khai thuật toán GĐ3. Test từng lỗi liên quan và chạy bounded smoke/resume với dữ liệu nhỏ; không tự chạy full train/sweep lớn, không tạo subagent. Dùng nguồn IEEE đã ghi trong docs, chỉ mở lại phần API khi cần đối chiếu phiên bản. Ghi bằng chứng và giới hạn vào IMPLEMENTATION_STATUS.md. Kết thúc bằng file đã thay đổi, các test thực sự chạy, blocker và lệnh preflight/train/resume/evaluate/report/sweep dùng được ở tên thư mục mới. Phân biệt code sẵn sàng chạy với GĐ2 đã có kết quả khoa học; không tuyên bố hoàn tất GĐ2 chỉ vì smoke qua.

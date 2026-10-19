# Review toàn diện đồ án — 2026-10-16

> Phạm vi: ~17.000 dòng mã đang ship (`src/`, `web/`, `scripts/`, `tests/`, `configs/`, `docs/`)
> và một lớp đối chiếu sang `legacy/` khi cần xác minh số liệu lịch sử.
> Cách làm: 6 tầng review (chuyên môn/học thuật → file train → client → web backend+bảo mật →
> UI/UX → test, docs, deploy, vệ sinh repo). Mọi mục có dẫn chứng `file:line`.
> Ký hiệu: **✓** = người review tự mở file, chạy `grep`/`Read` kiểm chứng, không tin nguyên báo cáo.
> Trạng thái sửa: **ĐÃ SỬA** / **CÒN LẠI**. Bản gốc tính đến hết 2026-10-16; hai đợt sửa sau đó được
> ghi ngày trong nhãn: **§0.2** (16/10, bổ sung test + đính chính 18/10) và **§0.3** (19/10, nhóm P1
> cục bộ). Mọi mục "XONG 19/10" đều trỏ về §0.3.

---

## Kết luận

Repo ở mức **rất tốt về kỷ luật kỹ thuật và liêm chính khoa học** — phần dữ liệu, thống kê và
cách tự ghi hạn chế là thứ hiếm thấy ở một đồ án tốt nghiệp. Nhưng có **3 nhóm rủi ro có thể bị
đánh trượt** nếu để muộn:

1. Validator FL mặc định *chấp nhận* cập nhật Byzantine (nay đã siết thành chính sách tường minh).
2. Các "cổng nghiệm thu" và số liệu trong docs không thực sự được đối chiếu về nguồn.
3. **Nhánh mặc định trên GitHub (`origin/main`) chỉ có `README.md`** và *không chung tổ tiên* với code
   thật — người mở link repo sẽ thấy một repo trống dù code đã nằm ở `origin/integration-gd1-gd2`.

> **Đính chính 2026-10-18.** Bản đầu của tài liệu này (16/10) có 6 chỗ sai hoặc phóng đại, đã được
> kiểm chứng lại từng chỗ và sửa ngay trong văn bản; danh sách ở **§9 — Đính chính**. Nguyên nhân
> chung của cả 6 chỗ: suy ra kết luận từ một bằng chứng cục bộ rồi khái quát hóa (ví dụ thấy
> `origin/main` trống ⇒ nói "chưa push"; thấy `weights_only=False` ⇒ nói "RCE" mà không đọc cổng kiểm
> hash ngay phía trên nó). Bản đính chính cũng mắc lại đúng lỗi đó một lần — §9.5 gọi đường tràn `n_k`
> là "đường thật" trong khi mới chỉ dựng được ở **mức hàm**; đã sửa sau khi đi hết các đường gọi thật.
> Phần còn lại của tài liệu đã được rà lại theo đúng tiêu chuẩn đó.

Ưu tiên xếp theo **rủi ro mất điểm bảo vệ × chi phí sửa**, không theo sở thích kỹ thuật.

---

## 0. P0 — ba rủi ro chí mạng trước hội đồng

| # | Vấn đề | Bằng chứng | Trạng thái |
|---|---|---|---|
| 1 ✓ | **Validator FL mặc định *cho qua* cập nhật độc hại.** `accepted = parent_hash_ok`, nên `update_norm_above_threshold` và `near_zero_update` chỉ bị *gắn cờ suspicious* rồi vẫn vào FedAvg. Với cấu hình launcher mặc định (không `-RejectSuspicious`), client `scaled_update ×10` được aggregate bình thường. Test của chính repo khóa con số `accepted_clients == 4, suspicious_clients == 3`. **Phạm vi chính xác:** NaN/Inf, sai shape, sai dtype và parent-hash lệch thì đã bị reject/cờ từ trước ([update_validator.py:84-90](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/update_validator.py#L84)); lỗ thật là *độ lớn* cập nhật, và `max_update_norm` mặc định `None` ⇒ khi không truyền ngưỡng thì norm **không bao giờ** bị gắn cờ. | [src/federated/update_validator.py:108](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/update_validator.py#L108) | **ĐÃ SỬA (theo cách khác dự kiến — xem §0.1)** |
| 2 ✓ | **Không kiểm tra finiteness sau aggregate** — *mức độ thật thấp hơn bản đầu đã ghi* (xem §9.5). `aggregate_fedavg` đã kiểm hữu hạn **từng client** ở [strategy.py:85](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/strategy.py#L85), nên đây là lớp phòng thủ thứ hai, không phải lỗ chí mạng. **Phạm vi chính xác:** tràn chỉ dựng được **ở mức hàm** — `num_examples` là số *client tự khai*, `n_k ≈ 1e308` làm tích `n_k · w_k` tràn float64 thành `inf` dù mọi trọng số đều hữu hạn — nhưng **không đường nào từ ngoài tới được**: Flower khóa `n_k` đúng bằng số dòng manifest ([server_app.py:216](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L216), [:521](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L521)), luồng cộng đồng từ chối `num_examples > MAX_EXAMPLES_ACCEPTED` ([community_store.py:646](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/community_store.py#L646)) rồi kẹp `n_cap` trước khi gộp ([community.py:215](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/community.py#L215)). ⇒ Gate giữ lại làm **phòng thủ chiều sâu** cho người gọi `aggregate_fedavg` sau này. | [src/federated/strategy.py:144](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/strategy.py#L144) | **ĐÃ SỬA + có test khóa (mức hàm)** |
| 3 ✓ | **Nhánh mặc định trên GitHub trống.** `git ls-tree origin/main` = **`README.md` duy nhất**, và `origin/main` (`f1e7388`) **không có tổ tiên chung** với HEAD (`git merge-base` = rỗng) ⇒ con số "39 commit" là phép trừ vô nghĩa, không phải độ trễ. Code **đã** nằm trên `origin/integration-gd1-gd2` (`df491b3b5`, đủ `src/ web/ tests/ docs/ scripts/ artifacts/`); phần chưa lên là **12 commit** (339 object, blob lớn nhất 11,2 MB `ort-wasm-simd-threaded.wasm` ⇒ cỡ ~22 MB) **cộng** các thay đổi chưa commit. Tag `v0.2.0-sync-20261010` trỏ đúng `main` cục bộ (`fe8bdd37`). | `git ls-tree origin/main`, `git merge-base origin/main HEAD`, `git rev-list --objects origin/integration-gd1-gd2..HEAD` | **CÒN LẠI** — việc thật cần làm là đổi nhánh mặc định / hợp nhất, không phải "push toàn bộ code" |


### 0.1. Hai quyết định thiết kế khi sửa P0 #1 (đọc trước khi "sửa tiếp")

Ý định ban đầu là đổi `accepted = parent_hash_ok` thành `accepted = parent_hash_ok and not suspicious`.
Kiểm tra test trước khi sửa cho thấy **đừng làm thế**, và report này ghi lại lý do để lần sau không
đi vào vết xe đổ:

- **Chế độ observe là một ablation có chủ đích, không phải bug.** [tests/test_attack_matrix_network.py:87](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_attack_matrix_network.py#L87)
  khóa `accepted_clients == 4` / `suspicious_clients == 3`, và [README.md:239](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/README.md#L239)
  yêu cầu run faulty-client phải **nêu rõ** `observe-only` hay `reject-suspicious`. Đây chính là bảng
  so sánh "validator có tác dụng gì / không có tác dụng gì". Đổi nghĩa `accepted` sẽ xóa luôn bằng
  chứng đó và làm lệch số liệu đã báo cáo.
- ⇒ **Giải pháp đã áp dụng: dịch rủi ro sang phía mặc định.** Thuật toán giữ nguyên; *không còn*
  đường nào âm thầm chạy chế độ observe-only nữa:
  - [src/federated/server_app.py:826-849](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L826)
    — guard tách thành hàm `check_deployment_policy(args)` (gọi từ khối `__main__` ở [`:884`](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L884))
    để **test khóa được mà không phải khởi động Flower server**: `--mode deployment` bắt buộc chọn một
    trong hai chính sách (`--reject-suspicious-updates` hoặc `--observe-only` mới thêm ở [`:865`](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L865));
    hai flag loại trừ nhau; `--observe-only` **chỉ được bind loopback** (`LOOPBACK_HOSTS`); chưa đặt
    `--max-update-norm` thì trả cảnh báo để in ra (không chặn, vì web demo không truyền flag này).
  - [scripts/start_real_flower_server.ps1:41](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/start_real_flower_server.ps1#L41):
    mặc định là chính sách **an toàn**; muốn ablation phải bật `-ObserveOnly` tường minh. Bind mặc
    định `127.0.0.1`, muốn mở LAN phải `-AllowRemoteClients` kèm `Write-Warning`.
  - [scripts/run_local_5client_scientific_network.ps1:43](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/run_local_5client_scientific_network.ps1#L43)
    và [web/network_demo.py:183](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/network_demo.py#L183): mọi đường spawn server đều
    truyền đúng một trong hai flag, không còn trạng thái "để trống".
  - Test mới: [tests/test_deployment_policy.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_deployment_policy.py)
    (7 test, gồm 1 test chạy thật `python -m src.federated.server_app --mode deployment` bằng subprocess
    để kiểm tra khối `__main__` **có nối** guard, thoát mã 2).
- **P0 #1 vẫn còn một nửa chưa sửa:** `accepted` vẫn = `parent_hash_ok`. Nếu muốn ba mức
  `accepted` / `quarantined` / `rejected` thật sự, phải **sửa test + attack matrix + docs song song**
  rồi chạy lại run nghiệm thu, vì số liệu hiện tại được sinh từ hành vi cũ.

### 0.2. Các sửa đổi P0 đã áp dụng (16/10, bổ sung test + đính chính 18/10)

| Mục | Vị trí sau sửa | Nội dung |
|---|---|---|
| Finiteness gate sau FedAvg | [src/federated/strategy.py:144](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/strategy.py#L144) | `check_parameters_finite(aggregated)` trước khi trả; comment ghi đúng phạm vi — tràn **ở mức hàm** do `num_examples` client tự khai, còn đường vào thật đã bị manifest (`n_k` phải khớp) và `n_cap` khóa. Test: [tests/test_fedavg.py:119](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_fedavg.py#L119) (từ chối khi tràn) + `:132` (không chặn oan khi `n_k` hợp lệ). |
| Chính sách deployment tường minh | [src/federated/server_app.py:826](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L826) (`check_deployment_policy`), [:865](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L865) (`--observe-only`), [:884](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L884) (nối vào `__main__`) | gate phản-đối-mặc-định + siết bind; tách hàm để test được. Test: [tests/test_deployment_policy.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_deployment_policy.py) — 7/7 đạt. |
| Loopback mặc định cho Flower | [src/federated/server_app.py:649](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L649), [web/network_demo.py:152](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/network_demo.py#L152), [web/static/index.html:629](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/index.html#L629), [web/routers/training.py:444](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/training.py#L444) | `0.0.0.0` → `127.0.0.1` cả ở code lẫn lời mô tả trên web. |
| Training API fail-closed cho IP xa | [web/services/admin_auth.py:37](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/admin_auth.py#L37), [:94](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/admin_auth.py#L94) | thiếu `COMMUNITY_ADMIN_TOKEN` chỉ còn mở cho `LOCAL_CLIENT_HOSTS`; request từ xa nhận **503 `admin_disabled`**. Giữ nguyên hành vi local mà [tests/test_training_auth.py:49](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_training_auth.py#L49) khóa (TestClient gửi host `testclient`). Test mới: IP xa → 503, admin xa **có** mật khẩu → 200, `X-Forwarded-For: 127.0.0.1` giả mạo → vẫn 503. |
| Ghi rõ điều kiện tin `client_ip` | [web/services/admin_auth.py:9](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/admin_auth.py#L9), [.env.example](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/.env.example) | docstring cũ vẫn nói "KHÔNG bị khóa, đúng hành vi cũ" (mâu thuẫn code mới) ⇒ viết lại, kèm cảnh báo: sau reverse proxy mà thiếu `--proxy-headers` thì **mọi** request hiện ra là IP proxy và bị coi là cục bộ. |
| Tài liệu hóa biến môi trường | [.env.example](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/.env.example) | viết lại theo **đúng** biến code thật đọc (`COMMUNITY_ADMIN_TOKEN`, `CORS_*`, `TRAIN_MAX_*` — nguồn: [web/services/training_guard.py:24](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/training_guard.py#L24)); khôi phục dấu tiếng Việt như bản gốc; khối `FL_*` được ghi chú rõ **chưa mã nào đọc** (giữ làm tham khảo cho launcher). |
| Kéo-thả: `preventDefault` + đọc `dataTransfer` | [web/static/js/pages/diagnosis.js:820-836](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/diagnosis.js#L820) | Xem §9.4: mức độ lỗi thấp hơn bản đầu đã ghi. `cancelDrag` là hàm không trạng thái nên nếu `mount()` chạy lại sau lỗi thì listener trùng cũng vô hại. |
| Nhãn smoke trung thực | [web/static/js/pages/training.js:271-283](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/training.js#L271) | `final_test_metrics` của phiên smoke giờ hiển thị kèm "chạy thử · dữ liệu tổng hợp · không phải kết quả khoa học", đúng cờ `scientific_valid` mà [web/routers/training.py:93](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/training.py#L93) gán. |
| Bỏ số hardcode trong JS | [web/static/js/pages/training.js:256](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/training.js#L256) | "2 vòng, 5 client" thay bằng `configured_rounds` / `num_clients_required` (có sẵn trong payload ở [web/routers/training.py:99](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/training.py#L99)). |

Kiểm chứng: cụm test liên quan (`test_fedavg`, `test_update_validator`, `test_attack_matrix_network`,
`test_training_auth`, `test_web_static_frontend`, `test_demo_aggregation`) **72/72 đạt** ngày 16/10;
cụm network + hợp đồng server (12 file: `test_demo_server_contract`, `test_network_demo_stop`,
`test_audited_flower_network`, `test_five_client_network`, `test_flower_multi_round_network`,
`test_network_evaluation_guard`, `test_network_reproducibility_guard`, `test_network_rtt_and_checkpoint`,
`test_attack_matrix_network`, `test_faulty_client_network_detection`, `test_fedavg`,
`test_network_artifact_validation`) **26/26 đạt** trong 7 phút 30 giây — tức gate `--observe-only`
mới không phá đường spawn của web demo. Sau khi thêm test ngày 18/10: `test_fedavg` +
`test_deployment_policy` + `test_training_auth` = **38/38 đạt**, trong đó **12 test mới** khóa đúng
những hành vi vừa thêm (trước đó các hành vi này chỉ được chứng minh là "không phá cái cũ").

### 0.3. Các sửa đổi P1 đã áp dụng (19/10 — nhóm "cục bộ, không đi ra ngoài")

| Mục | Vị trí sau sửa | Nội dung |
|---|---|---|
| `weights_only=True` cho 5 chỗ `torch.load` | [src/common/checkpoint_io.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/common/checkpoint_io.py) (mới) + 5 nơi gọi: [web/services/model_service.py:296](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/model_service.py#L296), [scripts/build_registry.py:98](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/build_registry.py#L98), [scripts/promote_verified_run.py:32](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/promote_verified_run.py#L32), [scripts/validate_artifact_provenance.py:55](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/validate_artifact_provenance.py#L55), [scripts/validate_network_artifact.py:52](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/validate_network_artifact.py#L52) | Một helper chung thay vì nhân bản allowlist 5 lần. Allowlist gồm 4 global numpy xuất hiện thật trong 55/55 `.pt` của `artifacts/` (đo bằng quét opcode pickle) cộng 7 lớp `numpy.dtypes.*DType` mà `demo_evidence.py` đã chấp nhận từ trước — 11 mục, không allowlist cả 33 DType. Archive TorchScript được nhận diện **trước** khi gọi `torch.load` (thành viên `constants.pkl`) rồi đưa cho `torch.jit.load` — không bắt lỗi rồi đoán câu chữ của torch. Test: [tests/test_checkpoint_io.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_checkpoint_io.py) (4 test). |
| Bỏ tick sẵn "Ghi nhớ trên máy này" | [web/static/admin/admin.html:67](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.html#L67) | `checked` đã gỡ ⇒ mật khẩu chỉ vào `sessionStorage` (mất khi đóng tab) trừ khi người dùng **chủ động** bật. [scripts/e2e_community.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/e2e_community.py) phải `page.check("#remember-token")` tường minh trước khi kiểm luồng "ghi nhớ". |
| Gỡ đường link `#token=<mật khẩu>` | [web/static/admin/admin.html:59](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.html#L59), [web/static/admin/admin.js:621](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.js#L621), [docs/COMMUNITY_FL.md:127](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/COMMUNITY_FL.md#L127) | Không còn nhận mật khẩu từ URL (lộ khi chia sẻ màn hình, lọt vào bookmark). Fragment vẫn được xóa để link cũ ai đã lưu không để mật khẩu nằm lại trên thanh địa chỉ. 3 check `#token=` trong e2e được thay bằng **1 check khóa việc đã gỡ** (không đăng nhập + không ghi `localStorage`/`sessionStorage`). |
| README mô tả file không tồn tại | [README.md:11](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/README.md#L11), [:107](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/README.md#L107), cây thư mục §3 | `configs/` chỉ còn `partition_config.yaml`; `src/centralized/inference.py` đã bỏ khỏi cây (không mã nào import nó — kiểm bằng grep); `requirements.txt` ghi đúng là **sàn phiên bản `>=`**, lock thật nằm ở `legacy/requirements-train.lock`; `artifacts/checkpoints/` liệt kê theo đĩa thật và ghi rõ hai slot canonical `centralized_best.pt`/`fedavg_best.pt` **không có trên máy này** (`checkpoint_status=missing_on_this_machine`, `web_visible=false`), model web_visible thật đến từ `artifacts/branch_demo/20261003/`. Các danh sách chỉ là một phần được nói rõ ("`ls tests/`"). Đã quét toàn bộ 31 đường dẫn trong README: 4 đường còn thiếu đều là **đích đến của script chưa chạy** (`multiseed_real_summary.*`, `artifacts/verified*/`), không phải file hứa sai. |
| `PROJECT_STATUS.md` nói sai về môi trường git | [scripts/update_status.py:360](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/update_status.py#L360) | Sửa ở **generator**, không sửa file (dòng đó nằm ngoài khối Nhật ký viết tay ⇒ sửa tay sẽ bị ghi đè). Câu "repo không có remote" đã bỏ; thay bằng sự thật kiểm được: `best.pt` là con trỏ LFS, object không có trong `.git/lfs/objects`, và bước mở khóa **thêm `git lfs pull`** trước khi tải lại từ Kaggle. |
| Số tham số trong kịch bản demo | [docs/AWS_DEPLOYMENT_GUIDE.md:243](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/AWS_DEPLOYMENT_GUIDE.md#L243) | `930K` → **1.556.806** toàn mô hình (đo bằng `count_parameters(create_model(38))`), kèm tách `features` 927.008 / `classifier` 629.798 để ai quen con số cũ biết nó là backbone. Ghi thêm: Tab 1 **không** in tổng tham số ([web/static/js/pages/overview.js:276](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/overview.js#L276) chỉ hiện tên model + số lớp); số tham số theo lớp nằm ở Tab 5. |
| `run_in_threadpool` cho `/api/inference/plantvillage` | [web/routers/inference.py:178](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/inference.py#L178) | Route là `async def` nên thân nó chạy trên event loop; torch forward giờ đi qua threadpool theo đúng cách [web/routers/community_fl.py:30](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/community_fl.py#L30) đã làm. Quét lại toàn bộ `async def` trong `web/`: đây là route **duy nhất** còn chặn loop (`/api/inference/all` và `/api/inference` là hàm đồng bộ nên FastAPI tự bỏ vào threadpool). |
| Ghi chú giới hạn của test XFF | [tests/test_training_auth.py:265](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_training_auth.py#L265) | Không sửa test, chỉ ghi đúng phạm vi: `TestClient` không chạy `ProxyHeadersMiddleware` của uvicorn nên test vẫn đạt kể cả khi middleware tin XFF — cái test khóa thật là `require_admin` đọc `request.client.host`, không đọc header. |

**Việc phát sinh khi sửa `weights_only` (đáng ghi lại vì không thấy được bằng đọc mã):**
`web/demo_evidence.py` từng mở `with torch.serialization.safe_globals([...])` — hàm này gỡ global theo
**hiệu hai tập** khi thoát khối, không đếm tham chiếu, nên nó **âm thầm xóa** allowlist mà
`checkpoint_io` đăng ký lúc import, trên toàn tiến trình. Triệu chứng: `test_web_model_service.py` chỉ
hỏng khi chạy **cùng** file khác, còn chạy một mình thì đạt. Đã gỡ chỗ đó (dùng `load_checkpoint`) và
đổi helper thành **hợp hai tập mỗi lần gọi, không bao giờ gỡ** — an toàn khi web phục vụ request trong
threadpool. `tests/test_checkpoint_io.py:21` khóa đúng regression này.

**Còn `weights_only=False` ở đâu (có chủ ý):** toàn bộ `legacy/` (pipeline đóng băng, chạy riêng);
`tests/test_network_rtt_and_checkpoint.py:75` và `tests/test_network_artifact_validation.py:101` (test
tự nạp artifact do chính nó vừa ghi); `scripts/evaluate_published_models.py:252` đã dùng
`weights_only=True` với allowlist scoped riêng và chạy ở **tiến trình khác** nên không đụng ai — giữ
nguyên để diff tối thiểu.

**Kiểm chứng 19/10:** 10 file (`test_deployment_policy`, `test_fedavg`, `test_training_auth`,
`test_update_validator`, `test_web_static_frontend`, `test_community_fl`, `test_checkpoint_io`,
`test_inference_all`, `test_plantvillage_reference`, `test_web_upload_limits`) = **149/149 đạt**, chạy
lại lần cuối **sau khi** mọi sửa đổi ở bảng trên đã vào (kể cả nhận diện TorchScript trước `torch.load`);
`test_flower_live_socket` + `test_attack_matrix_network` = **2/2 đạt** (3 phút 46 giây).

Sau đó chạy **toàn bộ gate** của repo (`scripts/update_status.py --full-gates`, 8 phút 41 giây):
`technical_acceptance = pass` trên **58 file test** — bốn nhóm `core_tests` / `scientific_guard_tests` /
`network_tests` / `other_tests` đều rc=0, tổng **373 test đạt, 1 skipped**; `scientific_acceptance` vẫn
`blocked` (đúng, vì thiếu checkpoint canonical — mục 12). Chi tiết nằm ở
`artifacts/final_acceptance.json` (dấu vân tay mã nguồn `db69b7071165`), không chép số tĩnh vào đây.
`PROJECT_STATUS.md` sau gate: **XONG 19/22 · CHƯA 0 · BLOCKED 3** — hai mục 11 ("full tests pass trên mã
nguồn hiện tại") và 22 ("release sạch") đã hết bị đánh dấu **CŨ**. Về artifact: `canonical_manifest.json`
**không đổi một byte**, `git diff artifacts/registry.json` chỉ còn bốn dòng `generated_at` và một dòng `code_fingerprint` (đổi vì mã đổi, đúng kỳ vọng), và không
`model_hash` nào đổi ⇒ helper nạp checkpoint mới không làm lệch kết quả gate.

**Tự kiểm `/quan-tri` bằng trình duyệt thật** (uvicorn cục bộ `127.0.0.1:8123`, `COMMUNITY_ADMIN_TOKEN`
đặt trong env của tiến trình, không ghi ra đĩa): ô "Ghi nhớ trên máy này" **không** tick sẵn; đăng nhập
với ô đó bỏ trống ⇒ mật khẩu chỉ vào `sessionStorage`, `localStorage` rỗng; tick ô đó ⇒ có thêm bản
`localStorage` và tải lại trang vẫn vào thẳng; "Đăng xuất" ⇒ **cả hai** sạch; sai mật khẩu ⇒ hiện đúng
"Tài khoản hoặc mật khẩu không đúng." và không lưu gì; `window.__errs` rỗng (không lỗi console).
Link `/quan-tri#token=<mật khẩu>` không tự đăng nhập và `location.hash` bị xóa thành rỗng.

Chỗ này suýt bị ghi sai: lần đo đầu `location.hash` **vẫn còn** `#token=`, tưởng strip ở
[admin.js:623-626](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.js#L623)
không chạy. Nguyên nhân thật là **cách đo**: điều hướng tới *đúng URL đang mở, chỉ khác fragment* là
fragment-navigation — trình duyệt không tải lại tài liệu nên `init()` không chạy lại, còn hash thì vừa
được chính lệnh điều hướng đặt vào. Đo lại bằng cách sang trang khác rồi mới vào URL có `#token=` (tải
tài liệu thật) thì hash rỗng ngay. `scripts/e2e_community.py:209` dùng context mới + `wait_until="load"`
nên không dính bẫy này.

**Chạy lại e2e Playwright (19/10, sau khi sửa trang quản trị):** lần đầu **hỏng 1/49** — bước 5 ("đăng nhập
ở tab quản trị thì khung khóa tự biến mất") dựa ngầm vào việc "Ghi nhớ" tick sẵn để mở khóa chéo tab
qua `localStorage`; chỗ đăng nhập đầu đã được sửa nhưng chỗ này bị sót. Thêm
`login.check("#remember-token")` ở [scripts/e2e_community.py:256](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/e2e_community.py#L256);
chạy lại **49/49 đạt** (Chromium thật, máy chủ uvicorn bind `127.0.0.1` bằng `.venv` của dự án, mật khẩu
thử nằm trong env tiến trình). Hệ quả cho người dùng thật: nếu không tick "Ghi nhớ" thì tab bước 5 không
tự mở khóa khi đăng nhập ở tab khác — đó là đánh đổi có chủ ý, không phải lỗi.

*Một lần duy nhất* `test_inference_all.py::test_explain_returns_one_gradcam_overlay_per_model` hỏng
trong buổi chạy 19/10; 4 lần chạy lại **đúng tổ hợp file đó** đều đạt (không cài `pytest-randomly` nên
thứ tự test là cố định ⇒ không phải do thứ tự), và lần chạy `--full-gates` ở trên cũng đạt. Chưa bắt
được traceback nên chưa kết luận nguyên nhân.
Đã thêm `assert "error" not in item` vào test: router cố tình nuốt lỗi Grad-CAM thành `{"error": …}`
nên trước đây test chỉ hỏng vì thiếu khóa, lần sau sẽ in ra lỗi thật.

---

## 1. Chuyên môn & phương pháp khoa học

### CAO

- **Tuyên bố privacy rộng hơn code chứng minh.** [docs/COMMUNITY_FL.md:4](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/COMMUNITY_FL.md#L4)
  ghi "Ảnh không rời máy", nhưng `POST /api/community/predict` nhận upload và chạy backbone trên
  server — [web/routers/community_fl.py:325](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/community_fl.py#L325).
  Test được viện dẫn để chứng minh chỉ chặn request của browser context nên *cấu trúc không thể thấy*
  request multipart. Đề xuất câu chữ: "đường **đóng góp** chỉ gửi head 38.950 tham số; đường
  **chẩn đoán** có nhận ảnh, không lưu ảnh". **CÒN LẠI.**
- **Không có client selection.** `fraction_fit = 1.0`, `min_fit_clients = min_available_clients = num_clients`
  — [src/federated/server_app.py:402](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L402).
  FedAvg được định nghĩa bởi cặp (C, K); participation luôn = 1 thì không kiểm chứng được hành vi thật
  của thuật toán, và mọi nhận định "mở rộng xuống N thiết bị" mất cơ sở. Đây là câu hỏi bảo vệ dễ bị
  hỏi nhất. **CÒN LẠI.**
- **Vòng all-or-nothing, không timeout.** Đủ `num_clients` mới nhận; một client chết là treo/abort
  ⇒ **mất trắng cả run**. Sửa: `round_timeout` + `min_accept` + ghi `partial_round`. **CÒN LẠI.**
- **`num_examples` tự khai điều khiển trọng số FedAvg ở đường cộng đồng.** `n_cap = 200` trong khi
  client trung thực chỉ có 5 ảnh ⇒ 10 account gian lận ≈ 97% khối lượng trung bình.
  [scripts/e2e_community.py:376](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/e2e_community.py#L376)
  là PoC chạy được và server chỉ clip rồi trả 200. Fix rẻ nhất: **weight = 1 cho mỗi participant**
  ở đường cộng đồng. **CÒN LẠI.**
- **"Artifact đã kiểm chứng" đứng trên chuỗi tự khai của client.** Server chỉ so chuỗi
  `data_mode == "real"`; chưa bao giờ xác minh trọng số *đến từ ảnh thật*. Một client sửa `data_mode`
  là đủ để run synthetic được promote thành artifact có SHA-256. **CÒN LẠI.**
- **`evaluate()` của Flower client đo trên chính loader huấn luyện** —
  [src/federated/client_app.py:287](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/client_app.py#L287)
  dùng `self.core.loader` (train, `shuffle=True`). Mọi bảng "client-side performance" là in-sample. **CÒN LẠI.**
- **BatchNorm không đóng băng, không thảo luận.** 34 lớp BN, running stats được trung bình như tham số
  thường; grep `track_running_stats|freeze_bn` = **0 kết quả** ở cả `src/` lẫn `legacy/`. Đây là điểm yếu
  kinh điển của FedAvg+BN mà hội đồng ML sẽ hỏi (FedBN / FedMD / FedProx) — và **không có baseline FL
  nào khác FedAvg**. **CÒN LẠI.**
- **Kết luận "non-IID" chưa được chứng minh ở mức mô hình.** Ở mức *dữ liệu* thì rất tốt (TVD đơn điệu
  0,0039 → 0,6955). Nhưng protocol `stage2_scratch_clean_v3` vẫn `BLOCKED: chưa có run cục bộ`,
  `final_acceptance.scientific = "blocked"`, **single seed 42**, và CI PlantDoc **đổi dấu**
  (C−F +6,7% [-3,7%; +18,0%]). Không thể nói "FedAvg kém hơn Centralized" với 1 seed và CI chứa 0 —
  hoặc chạy multi-seed, hoặc viết lại kết luận. **CÒN LẠI.**
- **Số tham số ghi trong tài liệu deploy không phải của toàn mô hình:**
  [docs/AWS_DEPLOYMENT_GUIDE.md:243](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/AWS_DEPLOYMENT_GUIDE.md#L243)
  (kịch bản nói khi demo Tab 1) ghi "MobileNetV3-Small (930K tham số)". Đo thật: **toàn mô hình
  38 lớp = 1.556.806** tham số (6,28 MB fp32), trong đó `features` (backbone) = **927.008** và
  `classifier` = 629.798 ⇒ "930K" là *số của backbone*, ghi nhầm thành số của mô hình. Ba hệ quả cần
  biết trước khi sửa: (a) chỉ xuất hiện **một lần** trong toàn bộ docs, (b) **không có** phép tính
  băng thông/chi phí nào trong repo dựa vào nó (FL truyền toàn bộ 1,55M tham số), (c) giao diện web
  hiện **không hiển thị** tổng số tham số ở đâu cả, nên câu "Tab 1 cho thấy … 930K" còn sai ở chỗ
  Tab 1 không cho thấy gì. Test chỉ assert `> 500_000` nên cả hai cách đọc đều lọt. **ĐÃ SỬA (19/10,
  xem §0.3): cả ba nhịp — số đúng, số backbone ghi chú riêng, và nói rõ Tab 1 không in tổng tham số.**
- **So sánh chưa công bằng và repo tự biết:** Centralized AdamW/83 epoch vs FedAvg SGD-not-momentum/32
  vòng; [legacy/stage2_scratch/baselines.py:37](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/legacy/stage2_scratch/baselines.py#L37)
  khai ~7.900 ảnh test GĐ2 nằm trong train GĐ1. Nếu số GĐ1 lên chung bảng ⇒ Centralized được train
  trên ~72% tập test. **CÒN LẠI.**
- **Tài liệu tham khảo quá mỏng:** [docs/REFERENCES_IEEE.md](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/REFERENCES_IEEE.md)
  có **2/12** là học thuật (McMahan AISTATS 2017 + 1 preprint arXiv), còn lại là doc phần mềm. Đồ án AI
  cần 15–25 nguồn peer-reviewed + chương Related Work + chương Limitations riêng. **CÒN LẠI.**

### TRUNG

- `quantity01` lẫn cả lệch nhãn lẫn dung lượng (TVD 0,3944) ⇒ không nên gọi là "pure quantity skew".
- `_repair_min_size` dịch nhóm lá *sau* khi rút Dirichlet ⇒ câu hỏi "α=0,1 có còn là Dir(0,1) không?"
  là chính đáng và cần một paragraph trả lời trong protocol.
- `label100` thực chất là IID (TVD 0,0377, 38/38 lớp mỗi client) ⇒ đừng gọi là "non-IID nhẹ".
- 4 công thức transform khác nhau nhau và tài liệu không khớp cái nào được dùng cho run nào.
- `std_accuracy 0.087` không tái tạo được từ 5 giá trị per-client (đo lại: 0,060/0,067).
- `label_screening` do **AI xem, không phải chuyên gia nông nghiệp**, nhưng hạn chế này chỉ nằm trong
  comment code chứ không xuất hiện trong tài liệu hạn chế.

---

## 2. File train (`src/centralized`, `src/data`, `configs`)

- **CAO — `python -m src.centralized.train --epochs 5` có thể ghi đè artifact khoa học bằng số từ ảnh
  nhiễu.** CLI không có `--dataset-root`/`--require-real-data` ⇒ `real_mode=False` ⇒ `__getitem__` trả
  tensor ngẫu nhiên (chỉ warn 1 lần), và nhánh chống ghi đè chỉ bật khi `--smoke`. Repo từng tai nạn:
  [scripts/build_registry.py:195](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/build_registry.py#L195)
  ghi "fedavg_best.pt **từng bị smoke ghi đè acc=1.0**". Sửa: raise khi `not real_mode` ngoài smoke. **CÒN LẠI.**
- **CAO — manifest thiếu ⇒ tự chế 38 mẫu giả** [src/data/plantvillage.py:83](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/data/plantvillage.py#L83)
  ⇒ "train hoàn tất, acc ≈ 1/38 = 2,6%" và vẫn ghi file. **CÒN LẠI.**
- **CAO — `FedAvgPartition` không nạp được partition nào đang ship:** cần `fedavg_meta.json`, thứ
  **không tồn tại** trong cả 8 điều kiện `data/partitions_stage2_scratch_v3/*`. Cả lớp
  `readiness_report` là code không chạy được trên dữ liệu thật. **CÒN LẠI.**
- **CAO — hàm "chống rò rỉ" rỗng:** [src/data/split.py:13](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/data/split.py#L13)
  docstring hứa "kiểm tra tính bảo toàn" nhưng thân chỉ trả `len()`. Trong khi
  [src/data/metrics.py:245](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/data/metrics.py#L245)
  đã có audit 5 chiều — chỉ chưa được gọi. **CÒN LẠI.**
- **CAO ✓ — chọn model trên test nếu thiếu `global_val.csv`** [src/federated/server_app.py:107](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L107):
  fallback về `global_test.csv` rồi in ra dưới nhãn "Held-out Test Evaluation - No Leakage". 8 điều kiện
  ship đều có val ⇒ latent, nhưng hội đồng chỉ cần đọc. **CÒN LẠI.**
- **CAO — 4 quy tắc chọn checkpoint khác nhau** (accuracy ở `src`, val_loss ở run 99,09%, macro-F1
  PlantDoc ở published-demo) trong khi báo cáo bằng macro-F1 ⇒ không thể nói "các mô hình chọn cùng
  tiêu chí". **CÒN LẠI.**
- **CAO — `src/` không tái tạo được run đã báo cáo:** run lịch sử có AMP + ReduceLROnPlateau + early
  stopping + `use_deterministic_algorithms`, còn `src/centralized/train.py` **không có**
  scheduler/early-stop/AMP/determinism (grep: `legacy` có, `src` 0). Checkpoint centralized không lưu
  optimizer/scheduler/epoch ⇒ không resume được. **CÒN LẠI.**
- **CAO — không có class-imbalance handling:** `CrossEntropyLoss()` không weight, không
  `WeightedRandomSampler` (grep 0 kết quả) dù lệch 36× (152 ↔ 5.507 ảnh/lớp) mà báo cáo lại lấy
  macro-F1. **CÒN LẠI.**
- **TRUNG — biến thể chết / trùng tên:** hai `split_global_test` trùng tên trong `partitioner.py` (bản
  đầu cụt, thiếu `return`); `partition.py` gọi `np.random.seed` toàn cục, hứa min-size mà không
  enforce, chia theo ảnh bỏ qua nhóm lá; `PlantVillageDataset` và `get_default_transform` **tồn tại 2
  bản khác nhau** và mọi consumer thật import bản `plantvillage.py` ⇒ cơ chế `sample_rng_seed` chuẩn
  mực trong `dataset.py` là mã chết; `configs/partition_config.yaml` trôi (10 client vs 5, val_ratio
  0.0 vs 0.08, `base_dir: partitions_v3` không tồn tại ⇒ `partition_dataset.py`/`sweep_alpha.py` chưa
  từng chạy vào repo); `src/common/metrics.py` không nơi nào dùng;
  [scripts/deep_data_audit.py:11](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/deep_data_audit.py#L11)
  trỏ thư mục không tồn tại và **in cứng** "✅ TẤT CẢ … ĐẠT 100%". **CÒN LẠI.**

---

## 3. Client (Flower + học cộng đồng)

Ngoài P0 #1–2 ở trên:

- **CAO — Flower port bind `0.0.0.0`, không TLS, không xác thực, `client_id` tự khai.** Người lạ tới
  được port là được giao model cha và nộp cập nhật ⇒ vừa đầu độc vừa DoS; trùng `client_id` → `raise`
  → **abort cả vòng**. **ĐÃ SỬA một nửa**: mặc định giờ là `127.0.0.1` (xem §0.2); mTLS per-client và
  namespace `client_id` **còn lại**.
- **P1 — `torch.load(weights_only=False)` ở 5 chỗ** (bản đầu xếp P0 "RCE nặng nhất" là **phóng đại**,
  xem §9.3): [web/services/model_service.py:295](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/model_service.py#L295)
  và 4 script — `scripts/build_registry.py:96`, `scripts/promote_verified_run.py:33`,
  `scripts/validate_artifact_provenance.py:54`, `scripts/validate_network_artifact.py:53`.
  **Vì sao chỉ là P1:** đường web không nạp tùy tiện — `_trusted_checkpoint()`
  ([model_service.py:265](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/model_service.py#L265))
  đòi registry xác nhận trạng thái tin cậy **và** SHA-256 trên đĩa khớp entry trước khi trả đường dẫn,
  nên kẻ tấn công qua HTTP không đưa được file pickle vào. Rủi ro còn lại là **chuỗi cung ứng /
  bất kỳ ai ghi được lên đĩa** (khi đó họ cũng ghi được `registry.json` và `file_sha256`). Vẫn nên sửa
  vì state_dict thuần không cần pickle, và repo đã có 3 chỗ làm đúng:
  [web/demo_evidence.py:209](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/demo_evidence.py#L209),
  `scripts/evaluate_published_models.py:252`, `scripts/train_leaf_detector.py:159`. Chỗ nào thật sự cần
  object phức tạp thì dùng `torch.serialization.add_safe_globals` thay vì tắt hẳn kiểm tra.
  (2 chỗ nữa trong `tests/` là test tự nạp artifact do chính test tạo — không tính.)
  **ĐÃ SỬA (19/10, xem §0.3):** gom về một helper `src/common/checkpoint_io.py` với allowlist numpy
  hẹp (11 mục, đo từ 55 checkpoint), 5 nơi gọi chuyển sang helper, có 4 test khóa (gồm cả regression `safe_globals` gỡ
  allowlist của module khác). Chỗ nào còn `weights_only=False` và vì sao: xem cuối §0.3.
- **CAO — Denial-of-Training chỉ có cách ly *bằng tay*:** kẻ xấu gửi cập nhật "sát ngưỡng" làm candidate
  fail gate không thoái lui ở **mọi** vòng ⇒ `rounds.status = 'rejected'`, đầu cập nhật bị null hóa
  ([web/services/community_store.py:516](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/community_store.py#L516)),
  phiên bản mới không sinh ra ⇒ tiến trình học cộng đồng đứng lại cho tới khi quản trị can thiệp.
  Đường can thiệp **có tồn tại** (`admin_revoke_participant` ở [web/services/community_store.py:801](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/community_store.py#L801)
  và bộ lọc `participant_id NOT IN (SELECT … WHERE revoked = 1)` khi gộp), nhưng **không có**
  reputation/tự động cách ly/thử-lại-sau-khi-loại-outlier. Ngoài ra `admin_close_round` gộp khi chỉ cần
  `count >= 1` ([web/services/community_store.py:813](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/community_store.py#L813)),
  phá bất biến `k_min = 2` do chính [src/federated/community.py:82](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/community.py#L82)
  tuyên bố ("một người một mình không được trở thành mô hình chung"). **CÒN LẠI.**
- **TRUNG — Không robust aggregator** (Krum / trimmed-mean / median) — chỉ clip L2 (chặn *độ lớn*,
  không chặn *hướng*) + gate không thoái lui. Ngưỡng `5.0` hardcode không dẫn chứng; chưa đo FP/FN.
  Attack matrix thiếu label-flip / backdoor / ALIE. **CÒN LẠI.**
- **TRUNG — `weight_inspector` gần như không nối pipeline** ⇒ nói "thanh tra trọng số chống đầu độc"
  là over-claim. `compute_model_hash` chỉ hash "tên key + bytes", thiếu dtype/shape — mà hash này là
  xương sống của `parent_hash_ok`. **CÒN LẠI.**
- **TRUNG — Telemetry:** `bytes_downloaded` được **giả định** bằng `bytes_uploaded`;
  `network_rtt_ms` luôn 0 nhưng vẫn summarize mean/median/p95; `estimated_comm_overhead_ms` trừ theo
  `local_train_time_s` **do client khai** ⇒ client tự ép về 0. Log không append-only, không hash-chain. **CÒN LẠI.**
- **TRUNG — RAM guard fail-open trên Windows/macOS** (`available_mb()` trả `None`) ⇒ tính năng "bảo vệ
  RAM" im lặng tắt ngay trên máy dev/demo chính. `--workers 1` chỉ là quy ước, không enforce trong
  code. **CÒN LẠI.**

**Điểm rất đáng khen:** đường **cộng đồng** server-side trưởng thành hơn đường Flower nhiều —
fail-closed gate, clip, quorum, idempotency 1 update/mã/vòng, chain `parent_sha`, chặn body trước khi
parse, `np.load(allow_pickle=False)`, `X-Forwarded-For` set ở cấp `server` kèm comment giải thích vì
sao location không kế thừa.

---

## 4. Web backend & bảo mật

- **CAO ✓ — API huấn luyện fail-OPEN khi thiếu `COMMUNITY_ADMIN_TOKEN`.** Mã cũ ở
  [web/services/admin_auth.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/admin_auth.py):
  `if expected is None: return` ⇒ ai cũng gọi được `/api/centralized/train`, `/api/federated/train`,
  `/api/train/stop`. `.env.example` **không liệt kê** biến này, systemd dùng `EnvironmentFile=-…`
  (dấu `-` = thiếu vẫn boot), README chạy `uvicorn --host 0.0.0.0` không env. Ngược đời:
  `community_store.py` lại fail-CLOSED (503) cho cùng một secret. **ĐÃ SỬA** (§0.2) — riêng việc
  "token yếu/absent trên prod nên chết hẳn" vẫn nên làm ở tầng cấu hình deployment.
- **CAO — endpoint nặng nhất hoàn toàn công khai, không rate-limit:** `/api/inference/all`
  (7 mô hình + 7 Grad-CAM nếu `?explain=true`), `/api/evidence/{summary,details}`,
  `/api/weights/inspect` — trong khi `/api/community/predict` *có* 60/phút/IP ⇒ chứng tỏ bộ lọc tồn
  tại nhưng sót cả nhóm. Đây là kênh làm sập trang nhanh nhất (trùng ảnh 504 đã chụp trong
  `docs/ui-review-2026-10-14/`). **CÒN LẠI.**
- **CAO — khuếch đại chi phí:** `load_evidence()` chạy lại **7 lần mỗi request** (đọc + băm lại mỗi
  file + parse 4 JSON), `registry_models()` parse `registry.json` 116 KB **mỗi lần gọi**,
  `current_head()` re-hash NPZ mỗi predict. **CÒN LẠI.**
- **TRUNG — `async def` chặn event loop** tại `/api/inference/plantvillage` (torch forward ngay trên
  loop). Nhóm community đã dùng `run_in_threadpool` đúng ⇒ đây là bug thật, không phải chủ ý.
  **ĐÃ SỬA (19/10, xem §0.3).** Đã quét lại mọi `async def` trong `web/`: không còn route nào khác
  chạy việc chặn trên loop.
- **TRUNG — lockout admin là self-DoS:** 60 lần đoán sai (không cần đúng) → mọi admin toàn cầu bị chặn
  10 phút; sau nginx mà chưa ghi đè XFF thì 1 người sai khoá cả lớp. Không `Retry-After`. **CÒN LẠI.**
- **TRUNG — mật khẩu quản trị nằm trong `localStorage` dưới dạng đọc được, và trang *khuyên* đưa nó
  vào URL.** [web/static/js/core/admin-auth.js:34-41](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/core/admin-auth.js#L34)
  ghi `{u, p}` (mật khẩu thô, chỉ base64 khi gửi header) vào `sessionStorage` **luôn luôn**, và ghi thêm
  một bản vào **`localStorage`** khi "Ghi nhớ trên máy này" được chọn — hộp này **tick sẵn mặc định**
  ([web/static/admin/admin.html:67](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.html#L67))
  ⇒ mật khẩu sống qua các lần mở trình duyệt trên máy chung. Nặng hơn một bậc: đường link
  `…/quan-tri#token=<mật khẩu>` mà [web/static/admin/admin.html:59](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.html#L59)
  hướng dẫn dùng sẽ được [web/static/admin/admin.js:634](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.js#L634)
  lưu lại với `remember = true`, tức **dán mật khẩu vào URL một lần là nó vào `localStorage` vĩnh viễn**
  (dù phần sau dấu `#` đúng là không gửi lên server). Không logout server-side, không rotation. Trang
  công khai **không CSP** trong khi front-end có ~85 điểm `innerHTML` — hai thứ này cộng hưởng: một lỗ
  XSS ở trang công khai là đọc được mật khẩu quản trị. **ĐÃ SỬA MỘT PHẦN (19/10, xem §0.3) — nửa
  "bền vững" đã đóng, nửa "đọc được" thì chưa:**
  - Đã gỡ: hộp "Ghi nhớ trên máy này" **không còn tick sẵn** ⇒ mặc định mật khẩu chỉ vào
    `sessionStorage` và mất khi đóng tab; đường link `#token=<mật khẩu>` **không còn được nhận** ⇒
    không còn cảnh "dán mật khẩu vào URL một lần là nó vào `localStorage` vĩnh viễn".
  - Vẫn còn: mật khẩu vẫn nằm ở dạng **đọc được** trong storage của tab (chỉ base64 khi gửi header);
    không logout server-side, không rotation; trang công khai **vẫn không CSP** trong khi front-end có
    ~85 điểm `innerHTML` ⇒ một lỗ XSS ở trang công khai vẫn đọc được mật khẩu **của phiên đó**, và đọc
    được bản `localStorage` nếu người dùng đã chủ động bật "Ghi nhớ". **CÒN LẠI.**
- **TRUNG — lộ đường dẫn nội bộ qua `detail` lỗi** ở `monitoring.py:113`, `training.py:73`, và
  `run_dir` trong payload `/api/federated/network-metrics` — ngược lại nhóm community đã làm đúng
  (thông điệp cố định + log riêng). **CÒN LẠI.**
- **TRUNG — HTTPS:** systemd đặt `--proxy-headers` nhưng README/AWS guide hướng dẫn demo **HTTP** trên
  `0.0.0.0`, tức Basic auth bay plaintext. **CÒN LẠI.**
- **API design:** 100% endpoint khai `Dict[str, Any]` (OpenAPI rỗng), không versioning, động từ trong
  URL, 200 cho "không có gì để dừng", 400 cho lỗi nên là 422/202. **CÒN LẠI.**
- **Không có bí mật trong git** ✓ — đã soi `git ls-files` + `git log --diff-filter=D`. SQL tham số hoá
  toàn bộ, subprocess không shell, chống path traversal có `is_relative_to` sau `resolve()`. Không tìm
  thấy lỗ hổng. **KHỎE.**

---

## 5. UI/UX

- **CAO → TRUNG — kéo-thả: lỗi thật là "thả ra ngoài vùng làm dỡ cả SPA", không phải "kéo-thả không
  hoạt động"** (bản đầu ghi sai, xem §9.4). [web/static/index.html:178](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/index.html#L178)
  có `<input type="file">` là con trực tiếp của vùng thả, và [web/static/css/style.css:965](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/css/style.css#L965)
  cho nó `position:absolute; inset:0; opacity:0; z-index:10` ⇒ **thả ảnh vào trong vùng vẫn chạy** nhờ
  hành vi gốc của input. Cái hỏng là mã JS ở
  [web/static/js/pages/diagnosis.js:820](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/diagnosis.js#L820)
  chỉ thêm/bớt class mà **không `preventDefault()`** và handler `drop` không đọc `dataTransfer.files`,
  nên (a) trạng thái tô sáng `is-dragover` không phản ánh đúng việc vùng có nhận file hay không, và
  (b) **thả lệch ra ngoài vùng** thì trình duyệt điều hướng tới `file:///…` và dỡ ứng dụng giữa buổi
  bảo vệ. Rủi ro trình bày vẫn thật (một cú thả lệch tay là mất cả phiên), chỉ là không phải "tính năng
  hứa suông". **ĐÃ SỬA** (§0.2).
- **CAO — số "Kiểm tra cuối trên tập ảnh giữ riêng" thực chất là smoke 1 ảnh/nhóm trên dữ liệu tổng
  hợp.** Chuỗi bằng chứng: `index.html` hidden `smoke=true` → `training.js` →
  [web/routers/training.py:261](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/routers/training.py#L261)
  **bắt buộc** `smoke=True` → `network_demo.py` gắn `scientific_valid: False`. Nhưng nhãn hiển thị
  không kèm qualifier ⇒ hội đồng đối chiếu với bảng hash-verified ở bước 3 rồi thấy lệch ⇒ mất niềm
  tin vào **toàn bộ** cách trình bày. **ĐÃ SỬA** (§0.2).
- **CAO — router không có `unmount`, token chống race khai mà không dùng:**
  [web/static/js/core/router.js:116](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/core/router.js#L116)
  `const mine = ++token;` không được đọc lại; nút "Thử lại" sau lỗi (`:164`) mount **lần nữa trong khi
  listener cũ còn** ⇒ một click bị xử lý 2–3 lần, inference chạy đúp, RAM tụt dần trong buổi trình
  bày. **CÒN LẠI.**
- **TRUNG — [web/static/js/pages/overview.js:161](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/overview.js#L161):**
  `getEvidence()` không `.catch()` trong `Promise.all` ⇒ một lỗi mạng nhỏ làm sập cả bước 1 vào
  `.pane-error`, bỏ luôn nhánh `unavailable()` lịch sự đã viết sẵn. `:176` chuỗi không phòng vệ
  (`m[id].metrics` có thể null). **CÒN LẠI.**
- **TRUNG — hàng chờ của bảng tiến trình dùng `colspan="8"` cho **cả hai** chế độ**
  ([web/static/js/pages/training.js:87](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/training.js#L87)
  và [:532](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/training.js#L532)), trong khi bảng FedAvg chỉ 7 cột
  (`HEAD.federated`, thấy ở đầu file; `EMPTY_ROW.federated` đã đúng là 7) ⇒ ô "Đang chạy — chờ kết quả
  vòng đầu tiên…" tràn bảng đúng lúc người xem nhìn nhiều nhất (vừa bấm "Bắt đầu"). **CÒN LẠI.**
- **A11y:** thiếu `scope="col"` ở 3 bảng (`index.html:596/606/623`); skip-link focus vào `main` nhưng
  `pages.css:20` xóa outline; `#task-badge role="status"` đổi mỗi 1,5 s ⇒ screen reader đọc lặp;
  `.cm-table` 11 px (ma trận nhầm lẫn — không đọc được trên máy chiếu); `.chart-container` 280 px cố
  định với `--page-max: 1760px` ⇒ méo tỉ lệ ở 1920px; `min-height: 100vh` thay `dvh`; không có
  `<noscript>`; `overflow-x: clip` trên `body` che dấu hiệu tràn. **CÒN LẠI.**
- **TRUNG — LCP mobile chưa đạt** (2,74–4,74 s > 2,5 s) và nguyên nhân trực tiếp vẫn còn:
  [web/static/index.html:20](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/index.html#L20)
  preload **8 font**. **CÒN LẠI.**
- **Các cam kết review cũ CHƯA làm** (đối chiếu code): "Esc đóng toast" (grep `Escape` = 0 kết quả),
  phát hiện ảnh vỡ sớm ở bước cộng đồng, debounce/hủy inference khi đổi bước, 1 câu "câu hỏi nghiên
  cứu" ở bước 1, mức sàn 12 px, 2 cột tab lỗi trên mobile, mục "Hướng phát triển", và **chưa từng
  re-verify bằng Safari / điện thoại thật / screen reader**. **CÒN LẠI.**
- **XSS: không tìm thấy lỗ hổng** ✓ — `esc()` dùng nhất quán cho tên file, tên người tham gia, run id,
  hash, command; `toast.js` từ chối `innerHTML` cho nội dung truyền vào. Điểm cộng lớn. **KHỎE.**

---

## 6. Test, docs, deploy, vệ sinh repo

- **CAO — "gate số liệu" thực chất là deny-list bốn chuỗi số.** [scripts/update_status.py:59](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/update_status.py#L59)
  khai một bộ `FORBIDDEN_NUMBERS` gồm hai giá trị accuracy cũ (viết theo cả kiểu dấu phẩy lẫn dấu chấm)
  và chỉ grep `README.md` + `docs/*.md`, **không so với artifact**. Tệ hơn: một trong bốn chuỗi đó
  **là giá trị đúng** của `demo_fedavg_v7` ⇒ gate *cấm một số đúng* và *cho qua mọi số sai khác*
  (viết 90,45% hay 95% đều lọt). Fix: đổi sang đối chiếu `artifacts/registry.json`. **CÒN LẠI.**
- **CAO ✓ — [PROJECT_STATUS.md:53](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/PROJECT_STATUS.md#L53)
  ghi "repo không có remote nên không thể `git lfs pull`" là sai sự thật:** remote `origin` tồn tại
  (`github.com/khanh51024/Federated-Learning-DoAn-TotNghiep`, 8 branch). Phát ngôn sai về môi trường của
  chính mình là loại lỗi làm sụp cả khung "máy chấm sự thật trên đĩa". **ĐÃ SỬA (19/10, xem §0.3) ở
  generator** [scripts/update_status.py:360](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/scripts/update_status.py#L360)
  — sửa tay `PROJECT_STATUS.md` sẽ bị ghi đè vì dòng đó nằm ngoài khối Nhật ký.
  Bản đầu còn nói quá một nhịp ("blocker #12 mở được bằng `git lfs fetch --all origin`") — xem **§9.8**.
- **CAO — `final_acceptance.json` ghi `git.dirty = true` và commit … cha** ⇒ không tồn tại commit nào
  tái lập đúng bộ mã đã test (fingerprint `d66bc1adb…`). Fix: validator **từ chối** khi dirty, chạy 2
  bước (chạy → commit → `--attest-only`). **CÒN LẠI.**
- **CAO ✓ — README mô tả file không tồn tại:** [README.md:11](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/README.md#L11)
  hứa `centralized.yaml` / `federated.yaml` / `non_iid.yaml` (`configs/` chỉ có `partition_config.yaml`);
  README:124 hứa `src/centralized/inference.py` (không có, và **không mã nào import** nó — đã grep);
  README:168 hứa `centralized_best.pt, fedavg_best.pt` (không có trên đĩa); README:107 gọi
  `requirements.txt` là "Pinned dependencies" trong khi **toàn bộ là `>=`**. **ĐÃ SỬA (19/10, xem §0.3).**
  Bản đầu còn nói quá một nhịp ("người chấm đi theo quickstart sẽ **fail ngay Bước 3**") — xem **§9.7**;
  tác động thật của bốn chỗ "ma" là **điều hướng sai cây thư mục**, và nó tự mâu thuẫn với nguyên tắc
  của chính repo ("máy chấm sự thật trên đĩa").
- **CAO — repo `git` quá nặng để *nộp*, nhưng **push nhánh hiện tại thì rẻ** (hai việc khác nhau, bản
  đầu gộp làm một):** `.git` **4,3 GB** (pack 3,71 GiB + LFS 411 MB, 2 file `tmp_obj_*` rác), worktree
  3,1 GB, `origin/dataset-sources` **179.726 file (176.958 ảnh)**, 18 file `.pt` đang track dù
  `.gitignore` nói không track. ⇒ *nộp* cần một repo sạch chỉ chứa code + docs + JSON artifact (ảnh để
  Kaggle / Zenodo / tarball). Còn *push* 12 commit đang thiếu lên `origin/integration-gd1-gd2` chỉ là
  339 object, blob lớn nhất 11,2 MB ⇒ **~22 MB**, không dính gì tới 4,3 GB kia. **CÒN LẠI.**
- **TRUNG — test frontend mỏng hơn con số:** [tests/test_web_static_frontend.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_web_static_frontend.py)
  có **64/107 assert là so khớp chuỗi trong mã nguồn**, 32/36 test **không chạy runtime** nào (ví dụ
  `assert "reverse = false" in flnet`, hay so `index()` trong file để suy "thứ tự thực thi").
  140/976 assert của cả repo là grep. pytest không có coverage, không đăng ký marker
  `network/slow/requires_dataset`, **13 skip site** nhưng không tài liệu nào nêu. Fix: Playwright smoke
  6 route (khai báo deps) — hiện `playwright` không có trong `pyproject.toml`. **CÒN LẠI.**
  *Lưu ý khi sửa kéo-thả:* chính file test này có deny-list ngữ ("Test giữ riêng", "Kiểm tra độc lập",
  "tập ảnh độc lập") nhưng **không** khóa chuỗi "Kiểm tra cuối" — nên phần gắn nhãn smoke đã đổi an toàn.
- **TRUNG — gate khác cũng tự tham chiếu / quản trị hóa cái sai:** `check_docs_clean` chỉ glob
  `docs/*.md` nên **59 file binary (6,6 MB)** screenshots + 1 `.docx` lọt, dù `PROJECT_STATUS.md:32`
  tuyên bố "docs/ chỉ gồm tài liệu hiện hành"; `check_deps` **yêu cầu**
  `requirements.txt == dependencies + optional test deps` ⇒ thể chế hóa việc nhét `pytest`/`onnx` vào
  runtime prod; `scripts/audit_release_leakage.py:180` luôn `return 0` bất kể `passed` ⇒ CI không chặn
  được rò rỉ. **CÒN LẠI.**
- **TRUNG — deploy không dựng lại được bản live:** `cropfed-demo.service` trỏ `fl-demo-20261013` trong
  khi máy chạy `fl-demo-20261014-s3`; AWS guide mô tả EC2 + port 8080 + tmux +
  `X-Forwarded-For $proxy_add_x_forwarded_for` (spoof được) **mâu thuẫn trực tiếp** với
  `deploy/nginx-community.conf` đã đúng; `.env.example` **đã bổ sung** đủ biến, nhưng `FL_SERVER_PORT`
  vẫn lệch default 8089 ở docs deploy; systemd thiếu `MemoryMax` (máy 2 GB ⇒ OOM-kill cả host),
  `ProtectSystem`, `Restart=always`, không healthcheck; nginx thiếu `proxy_read_timeout` (nguồn 504)
  và `limit_req` cho `/api/inference*`, IP allowlist admin đang bị **comment**. **CÒN LẠI (một phần).**

---

## 7. Điểm mạnh — nên đưa hẳn vào slide

Phần này là lý do đồ án có giá trị, và nó **không** tự tham chiếu:

1. **Toán FedAvg đúng và có test chứng minh:** [tests/test_fedavg.py:12](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_fedavg.py#L12)
   kiểm tay `(100·1 + 300·3)/400 = 2.5`; dùng `n_k` thật của `label100`;
   `num_batches_tracked = parent + Σ delta` và **từ chối** khi counter đi lùi — đa số code FL nghiệp dư
   làm sai chỗ này. Tích luỹ float64 rồi cast, validate key/shape/dtype.
2. **Dirichlet non-IID cài đúng bản chuẩn:** rút `p_k ~ Dir(α·1_C)` *theo từng lớp*, quantity skew bằng
   một vector `q ~ Dir(β)` dùng chung, làm tròn Hamilton — không phải "shuffle rồi cắt".
3. **Non-IID đã được ĐO chứ không tuyên bố:** mỗi `partition_config.json` kèm `partition_metrics`
   (TVD/JS/KL, entropy chuẩn hoá, classes/client, CV, Gini, cosine-to-global, fairness spread,
   `rare_class_recall_mean`) và `split_diagnostics`; TVD tăng đơn điệu đúng hướng 0,0039 → 0,6955.
4. **Leaf-atomic split sạch, và đã kiểm chứng độc lập:** giao `group_id`/`leaf_group_id` giữa
   train↔val↔test và giữa các client = **0**; 21/21 cặp trùng byte nằm cùng nhóm, 0 cặp chéo split;
   test phủ 38/38 lớp. Chuỗi bằng chứng `audit_release_leakage.py` (so `byte_sha256` /
   `parent_scene_id`, 46.076 ảnh, `hash_mismatch: 0`).
5. **Thống kê nghiêm túc khi cần:** bootstrap CI **theo cụm lá/cảnh** (B=2000, seed ghi rõ) +
   **McNemar chính xác** (`binomtest`, không xấp xỉ χ²) + `selection_effect()` lượng hoá thiên lệch chọn
   checkpoint + temperature scaling đúng Guo et al. (ECE 0,295→0,261, fit trên 4.225 ảnh calibration
   tách rời).
6. **Tương đương chéo ngôn ngữ byte-exact Python ↔ JS** ([tests/test_community_js.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_community_js.py))
   và JS tái tạo đúng preprocessing PyTorch (`community-math.js`, resize tamangular 22-bit +
   `Math.fround`) — giá trị học thuật cao nhất mảng web.
7. **Network là thật, không diễn:** gRPC giữa 6 tiến trình riêng, pre-check port, re-verify "5 client
   accepted" ở **mọi** vòng audit, `tests/test_flower_live_socket.py` + attack matrix chạy qua socket
   thật và assert *không* có checkpoint khi reject.
8. **Trung thực trong báo cáo — điểm sáng số 1:** `EXPERIMENT_PROTOCOL.md:116` tự liệt kê 5 hạn chế
   (chọn checkpoint trên val, PlantDoc 172 ảnh/96 cảnh ⇒ CI ±9–10 điểm, 24% singleton, AdamW-vs-SGD,
   một seed một α); `pre_demo_check` trả `technical_demo_ready` chứ **không** phải `ready`; readiness
   ghi thẳng "CHƯA nghiệm thu khoa học"; `COMMUNITY_FL.md:92` có cột "Còn lại" thừa nhận không có
   secure aggregation; `label_screening` tự nhận người kiểm là AI; `summarize_multiseed.py` trả
   `std=0.0` khi n=1 thay vì giả vờ có phương sai. Hội đồng thường cho điểm cao hơn vì tự nhận hạn chế
   hơn là che.
9. **Kỷ luật UI:** tương phản WCAG đạt ở **cả hai** theme (4,5–15,5:1), màu biểu đồ ≥3:1 và an toàn mù
   màu (phân biệt bằng hình dạng + số trong ô, không chỉ màu), validate form tiếng Việt tại trường +
   `aria-invalid` + focus vào trường lỗi, `Retry-After` ở các 429, mọi bảng cuộn được bằng phím, toast
   có dedupe/pause/`role=alert` vs `status`, in-page limitation tự tính từ số liệu.

---

## 8. Thứ tự việc còn lại (theo rủi ro × chi phí)

**Ngay, không trade-off được:**
1. **Làm cho link repo hội đồng mở ra không trống** (P0 #3) — **CÒN LẠI, và là việc duy nhất trong nhóm
   này chưa làm.** Ba việc tách bạch, theo thứ tự rẻ dần:
   (a) `git push -u origin ui-upgrade-2026-10-14` — chỉ ~22 MB, không liên quan `.git` 4,3 GB;
   (b) đổi **nhánh mặc định** trên GitHub sang nhánh có code (đây mới là việc quyết định hội đồng thấy gì);
   (c) *nếu* cần nộp repo sạch thì dựng repo giao nộp riêng — việc này **không** phải điều kiện của (a)/(b).
   Cả ba đều là thao tác ra ngoài ⇒ **cần chủ repo quyết định**, không tự làm.
2. ~~4 mục README ma + `PROJECT_STATUS.md:53` + `930K` → `1.556.806`~~ — **XONG 19/10 (§0.3)**, gồm cả
   ghi chú `930K` là số backbone 927.008 và sửa `PROJECT_STATUS` ở **generator** chứ không phải ở file.
3. ~~Bỏ hướng dẫn `#token=` và tắt tick sẵn "Ghi nhớ trên máy này"~~ — **XONG 19/10 (§0.3)**, kèm cập
   nhật `scripts/e2e_community.py` (3 check `#token=` → 1 check khóa việc đã gỡ) và
   `docs/COMMUNITY_FL.md:127`. Nửa **chưa** xong của cùng phát hiện (mật khẩu vẫn đọc được trong
   storage của phiên; trang công khai vẫn không CSP) ghi ở §4.

**P1 (nửa ngày):** ~~`weights_only=True` cho 5 chỗ `torch.load`~~ và ~~`run_in_threadpool` cho
`/api/inference/plantvillage`~~ — **XONG 19/10 (§0.3)**. Còn lại: rate-limit + cache `load_evidence()` /
`registry_models()`; `colspan` và `getEvidence().catch()`; `unmount` / `{ signal }` trong router.

**2–3 ngày:** `round_timeout` + `--fraction-fit` + một bảng thực nghiệm C ∈ {1.0, 0.5, 0.2}; nếu muốn
`accepted` / `quarantined` / `rejected` ba mức thì sửa **đồng thời** test + attack matrix + docs rồi chạy
lại run nghiệm thu (xem §0.1).

**1 tuần (quyết định chất lượng khoa học):** chạy `stage2_scratch_clean_v3` cho 5 client ×
{label100, label1, label01} × 3 seed **cùng một optimizer / tiêu chí chọn checkpoint / BN policy**; thêm
**FedProx** (~10 dòng) làm baseline; vẽ `noniid_vs_alpha.png` + đường hội tụ centralized-vs-FL (dữ liệu
đã nằm sẵn trong `partition_config.json` và `rounds.jsonl`, chỉ thiếu hình); đổi `check_docs_numbers`
sang đối chiếu registry; `final_acceptance` fail khi dirty; viết chương Related Work + Limitations.

Nếu không kịp multi-seed, kết luận phải viết: *"một seed, CI đổi dấu ⇒ chưa bác bỏ được giả thuyết
không khác biệt"* — và chủ động đưa vào slide Limitations, đừng để slide nào nói "FedAvg tốt/kém hơn".

**Bước 4 (multi-seed) đã kiểm 19/10 — rơi vào nhánh Limitations, và Limitations đã có sẵn.** Không chạy
được trên máy này, nhưng **không phải vì thiếu dữ liệu**: `data/dataset_sources/` có
`PlantVillage-Dataset` + `mixed`, `data/dataset_manifest.json` ghi 54.305 ảnh. Nút thắt là máy chỉ có CPU
(`torch.cuda.is_available()` = `False`, 12 luồng) trong khi chính
[docs/EXPERIMENT_PROTOCOL.md:142](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/EXPERIMENT_PROTOCOL.md#L142)
ước tính ~10 GPU-giờ mỗi seed (~30 GPU-giờ cho ba seed), và run canonical còn thiếu W0/checkpoint (mục 12
BLOCKED trong `PROJECT_STATUS.md`). Câu Limitations **không cần viết thêm**:
[docs/EXPERIMENT_PROTOCOL.md:136](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/EXPERIMENT_PROTOCOL.md#L136)
("Một seed, một mức α. Chưa có độ lệch giữa các lần chạy"),
[:140-143](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/EXPERIMENT_PROTOCOL.md#L140)
(điều kiện để được khẳng định "FedAvg gần Centralized"), và ba chỗ ở Tab 6 — nơi hội đồng đọc thật:
[evidence.js:67](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/evidence.js#L67)
("Cần huấn luyện lại cùng điều kiện, nhiều seed trước khi kết luận hai cách tương đương"),
[:92](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/evidence.js#L92),
[:268](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/evidence.js#L268).
Ngay Tab 1 cũng đã in "Số lần chạy: 1 lần (seed …)"
([overview.js:281](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/pages/overview.js#L281)).
Việc còn lại của bước 4 là **chữ trên slide**, không phải mã.

---

## 9. Đính chính (2026-10-18, bổ sung 19/10) — tám chỗ bản đầu sai hoặc phóng đại

Bản 16/10 được thẩm định lại bằng cách mở từng file, chạy `git`, và *chạy thật* trường hợp nghi ngờ.
Tám chỗ dưới đây đã sửa ngay trong văn bản; ghi lại ở đây để người đọc bản cũ không dùng số sai.
9.1–9.6 là kết quả thẩm định ngày 18/10; **9.7–9.8 là hai chỗ chính bản đính chính tự phát hiện ngày
19/10** khi làm nhóm P1 — cùng một lỗi cũ: nói quá một nhịp so với bằng chứng vừa đo.

| # | Bản đầu đã ghi | Sự thật đã kiểm chứng |
|---|---|---|
| 9.1 | "Toàn bộ code hiện tại chưa được push" | `origin/integration-gd1-gd2` (`df491b3b5`) **có đủ** `src/ web/ tests/ docs/ scripts/ artifacts/`. Chưa lên chỉ là **12 commit** (~22 MB, blob lớn nhất 11,2 MB) + thay đổi chưa commit. Vấn đề thật: **nhánh mặc định** `origin/main` chỉ có `README.md` và `git merge-base origin/main HEAD` **rỗng** ⇒ "39 commit" là phép trừ vô nghĩa. |
| 9.2 | (đính chính 16/10) "token chỉ vào `sessionStorage`, tử tế hơn `localStorage`" | **Sai, và là sửa làm báo cáo tệ hơn.** [admin-auth.js:34-41](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/js/core/admin-auth.js#L34): mật khẩu thô vào `sessionStorage` luôn, **và vào `localStorage` khi "Ghi nhớ" được chọn — hộp này tick sẵn**. Link `#token=` còn bị lưu với `remember = true` ([admin.js:634](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/admin/admin.js#L634)). |
| 9.3 | "`weights_only=False` = RCE, P0 bảo mật nặng nhất" | Hạ xuống **P1**: [model_service.py:265-273](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/model_service.py#L265) kiểm trạng thái registry **và SHA-256 trên đĩa** trước khi trả đường dẫn ⇒ không có đường từ HTTP vào `torch.load`. Còn 5 chỗ (1 web + **4 script**, bản đầu đếm thiếu `validate_artifact_provenance.py`). Rủi ro thật là chuỗi cung ứng / ghi được lên đĩa. |
| 9.4 | "Kéo-thả là hứa suông; theo chuẩn HTML thì `drop` không fire trên zone" | **Sai một nửa.** `<input type=file>` phủ kín zone ([style.css:965-973](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/static/css/style.css#L965): `inset:0; opacity:0; z-index:10`) nên **thả trong vùng vẫn chạy** nhờ hành vi gốc. Lỗi thật: không `preventDefault` ⇒ thả **lệch ra ngoài** làm trình duyệt mở `file:///…` và dỡ SPA. Bản sửa vẫn đúng và vô hại (`cancelDrag` không trạng thái, mount lại cũng không sao). |
| 9.5 | "Không kiểm tra finiteness sau aggregate" xếp P0 chí mạng | Hạ mức độ **hai lần**. (a) Từng client **đã** được kiểm ở [strategy.py:85](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/strategy.py#L85) ⇒ đây là lớp phòng thủ thứ hai. (b) Bản đính chính 18/10 lại nói quá: "đường tràn thật… đã dựng lại được bằng lệnh thật" — **chỉ đúng ở mức hàm**, vì `aggregate_fedavg` được gọi thẳng từ test. Không đường nào từ ngoài tới được: Flower khóa `n_k` đúng bằng số dòng manifest ([server_app.py:216](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L216), [:521](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/server_app.py#L521)), cộng đồng từ chối `num_examples > MAX_EXAMPLES_ACCEPTED` ([community_store.py:646](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/community_store.py#L646)) rồi kẹp `n_cap` ([community.py:215](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/federated/community.py#L215)). Gate vẫn giữ (phòng thủ chiều sâu cho người gọi sau này) và vẫn có test ([test_fedavg.py:119](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_fedavg.py#L119)), nhưng test đó **không** chứng minh một lỗ đang khai thác được. |
| 9.6 | "`930K` sai ~40%, dùng tính băng thông FL ⇒ sai dây chuyền" | `930K` = **số backbone** (đo: `features` 927.008, `classifier` 629.798, toàn mô hình 1.556.806). Xuất hiện **một lần** ở [AWS_DEPLOYMENT_GUIDE.md:243](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/docs/AWS_DEPLOYMENT_GUIDE.md#L243); **không** phép tính băng thông nào trong repo dùng nó; web cũng không hiển thị tổng tham số ⇒ câu "Tab 1 cho thấy 930K" sai thêm một nhịp. Mức: vệ sinh tài liệu. |
| 9.7 *(thêm 19/10)* | "README ma ⇒ người chấm đi theo quickstart sẽ **fail ngay Bước 3**, ấn tượng đầu tiên rất tốn điểm" | **Chưa kiểm chứng và nhiều khả năng sai.** Không lệnh nào trong quickstart dùng ba file YAML bị hứa sai, và Bước 3 (`src.centralized.train --smoke`) chạy với `dataset_root=None` ([train.py:58](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/src/centralized/train.py#L58)). Không chạy thử để xác nhận vì training ghi vào `artifacts/`. Tác động thật chỉ là **điều hướng sai cây thư mục** — vẫn đáng sửa, nhưng là vệ sinh tài liệu chứ không phải "hỏng quickstart". |
| 9.8 *(thêm 19/10)* | "`PROJECT_STATUS.md` nói sai ⇒ blocker #12 **có thể mở được** bằng `git lfs fetch --all origin`" | **Nửa sau chưa kiểm chứng.** Đúng là remote `origin` tồn tại (8 branch) và câu "repo không có remote" là sai. Nhưng object LFS của run canonical (`oid bdfe01879a…`, 6.358.090 byte) **không có** trong `.git/lfs/objects`, và remote còn giữ nó hay không thì **chưa gọi mạng để biết** ⇒ không khẳng định được blocker mở được. Đã ghi vào generator đúng mức: "thử `git lfs pull`, không được thì tải lại từ Kaggle". |

**Ba rủi ro thật của chính bản sửa P0 (đã xử lý trong đợt 18/10):**
- *Chưa có test nào khóa hành vi mới* — 72 + 26 test đạt trước đó chỉ chứng minh "không phá cái cũ".
  ⇒ Đã thêm **12 test**: [tests/test_deployment_policy.py](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/tests/test_deployment_policy.py)
  (7, gồm 1 subprocess kiểm tra `__main__` có nối guard), 3 test auth IP xa/`X-Forwarded-For` giả mạo,
  2 test tràn `num_examples` (**mức hàm**, không phải đường vào thật — xem §9.5).
- *Lối tắt "từ máy này" tin `request.client.host`* — an toàn trên production hiện tại vì
  [deploy/cropfed-demo.service:18](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/deploy/cropfed-demo.service#L18)
  chạy `--proxy-headers --forwarded-allow-ips 127.0.0.1` và bind `127.0.0.1` sau nginx; nhưng nếu ai đó
  đặt app sau proxy **thiếu** `--proxy-headers` thì mọi request hiện ra là IP proxy và được coi là cục bộ.
  ⇒ Đã ghi cảnh báo vào docstring [admin_auth.py:9](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/admin_auth.py#L9) và `.env.example`.
- *Docstring mâu thuẫn với code* — [admin_auth.py:9](file:///D:/Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old/web/services/admin_auth.py#L9)
  vẫn nói "KHÔNG bị khóa, đúng hành vi cũ"; `.env.example` mất dấu tiếng Việt so bản gốc, có lỗi
  "mo kong" và một dòng thiếu khoảng trắng sau `#`. ⇒ Đã sửa cả hai; `.env.example` nay ghi rõ khối
  `FL_*` **chưa mã nào đọc** (kiểm bằng grep toàn repo) để khỏi ai tưởng nó có tác dụng.

---

## Phụ lục — giới hạn của chính báo cáo này

- Báo cáo dựa trên **đọc mã và chạy test trên máy dev CPU**, không chạy lại run huấn luyện thật; các
  nhận định về số liệu khoa học lấy từ artifact/protocol đang ship.
- Vài mục TRUNG về cấu hình deploy (nginx, systemd, EC2) được đối chiếu từ file trong repo, **chưa**
  kiểm tra trạng thái máy chủ đang chạy.
- Bản 16/10 có 6 chỗ sai/phóng đại (§9). Bài học phương pháp: mỗi phát hiện mức CAO phải kèm **một
  bằng chứng phủ định đã thử** (cổng kiểm nằm ngay phía trên? hành vi gốc của trình duyệt? nhánh nào
  thật sự chạy được?) trước khi xếp hạng.
- Thay đổi 16/10 chạm 11 file. Đợt 18/10 chạm thêm 7 file: `src/federated/strategy.py` (comment),
  `src/federated/server_app.py` (tách `check_deployment_policy`), `web/services/admin_auth.py`
  (docstring), `.env.example`, `tests/test_fedavg.py` + `tests/test_training_auth.py` +
  `tests/test_deployment_policy.py` (12 test mới). Cộng 3 file tài liệu: tài liệu này,
  `docs/README.md` (index) và `scripts/update_status.py` (`ALLOWED_DOCS`, nếu không gate
  `check_docs_clean` sẽ báo tài liệu lạ). Tất cả **không** đổi nghĩa thuật toán FedAvg.
  Bảng §0.2 là danh sách đầy đủ. Chưa commit, chưa push.

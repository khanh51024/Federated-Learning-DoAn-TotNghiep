# Học phân tán cộng đồng: mỗi người vào web là một client

Trang **Demo · Học phân tán** của web có khối **“Tham gia từ máy của bạn”**. Người có mã tham gia mở trang, thêm ảnh lá của mình, chọn nhóm cho từng ảnh,
huấn luyện **ngay trong trình duyệt** và gửi trọng số về máy chủ. Máy chủ gộp theo vòng (FedAvg) khi đủ K người. Ảnh không rời máy.

Tài liệu này mô tả phạm vi, giao thức, phòng thủ, cách vận hành và bằng chứng nghiên cứu đã đo. Đây là một trong ba cách nhận diện chính của đồ án (cùng Local only và PlantVillage): trang Chẩn đoán luôn dùng phiên bản mới nhất đã qua cổng.
Mô hình cộng đồng tách khỏi bảy mô hình đã kiểm hash; số đo mức báo cáo là thí nghiệm mô phỏng bên dưới, còn “độ chính xác giám sát” của từng vòng chỉ để theo dõi.

## Phạm vi: học gì, không học gì

- Backbone MobileNetV3-Small và hai lớp `classifier[0..1]` của **FedAvg v7 được đóng băng** (SHA-256 của checkpoint đã ghim).
  Phần này chạy trong trình duyệt bằng onnxruntime-web (wasm, một luồng) để ra vector đặc trưng 1024 chiều.
- Chỉ **`classifier.3`** (38 × 1024 + 38 = 38.950 tham số, 155.800 byte float32) được huấn luyện bằng softmax regression viết thuần JS
  trong Web Worker. Đây là *federated fine-tuning lớp cuối*, không phải huấn luyện lại cả mạng.
- Khởi điểm vòng 0 là head của FedAvg v7. Mỗi vòng gộp tạo một phiên bản head mới (v1, v2, …) lưu theo SHA-256.
- Không có: secure aggregation, huấn luyện lại backbone, nhãn do chuyên gia kiểm, phép đo khoa học trên ảnh ngoài đồng.

## Kiến trúc và luồng

```
Trình duyệt (mỗi người)                          Máy chủ (một tiến trình uvicorn, SQLite + file head)
 ảnh ─ decode+EXIF ─ resize kiểu PIL ─ ONNX ─►    GET  /api/community/feature-model   (ONNX tĩnh, SHA-256, cache dài)
 vector 1024 ─ softmax regression (Worker)         POST /api/community/join {code} → token HMAC
 ‖Δ‖, nhãn, số ảnh ─ head float32 (~208 KB b64) ─► POST /api/community/update  → kiểm → gộp khi đủ K → cổng → phiên bản mới
```

| Thành phần | Tệp |
|---|---|
| Lõi số học (mã hóa head, kiểm, cắt chuẩn, FedAvg, cổng) | `src/federated/community.py` (dùng lại `aggregate_fedavg`, `validate_client_update`, `compute_model_hash`) |
| Trạng thái, mã/token, vòng, lịch sử, khôi phục | `web/services/community_store.py` |
| HTTP, hạn mức, quản trị | `web/routers/community_fl.py` |
| Dự đoán bằng mô hình cộng đồng | `web/services/community_inference.py`, `POST /api/community/predict` |
| Giao diện (trang Demo · Học phân tán, ô ở trang Chẩn đoán) | `web/static/js/pages/community.js`, `workers/community-worker.js`, `core/community-math.js` |
| Xuất ONNX, dựng đặc trưng giám sát | `scripts/export_community_feature_model.py`, `scripts/build_community_eval_features.py` |
| Kiểm thử | `tests/test_community_fl.py`, `test_community_js.py`, `test_community_assets.py`, `scripts/e2e_community.py` |

Dữ liệu cộng đồng nằm ở `artifacts/community_fl/` (gitignored; đường dẫn khai báo trong `src/registry.py`):
`feature_model.onnx`, `base_head.npz`, `eval_features.npz`, `community_manifest.json` (SHA-256 mọi tệp) và thư mục `state/`
(SQLite WAL + `heads/vNNNNNN.npz`) bền qua khởi động lại. Máy chủ **fail-closed**: thiếu hoặc sai SHA thì tính năng tắt kèm lý do,
`/api/community/status` vẫn trả `enabled:false` để giao diện báo rõ.

## Giao thức

| Điểm cuối | Xác thực | Việc |
|---|---|---|
| `GET /api/community/status` | không | Phiên mở/đóng, vòng hiện tại, phiên bản, cấu hình client, 38 tên lớp, thông tin file ONNX |
| `POST /api/community/join` `{code}` | mã | Kiểm mã (lưu dạng HMAC, có hạn và số lượt) → token `v1.<payload>.<chữ ký>` hết hạn sau 6 giờ |
| `GET /api/community/feature-model` | không | File ONNX, `Cache-Control: public, immutable`, `X-Content-SHA256` |
| `GET /api/community/round?have=<sha>` | token | Vòng, hạn, số cập nhật, head cha (bỏ qua nếu `have` đã khớp) |
| `POST /api/community/update` | token | `{round_id, parent_sha, head, num_examples, label_counts[38], local, dp_applied}` |
| `GET /api/community/history?limit=` | không | Các vòng đã gộp/bị từ chối, các phiên bản và chỉ số giám sát (không có định danh người) |
| `POST /api/community/predict` | không | Ảnh → dự đoán của mô hình cộng đồng so với FedAvg v7 (qua cổng registry của FedAvg v7) |
| `/api/community/admin/*` | tài khoản `COMMUNITY_ADMIN_USER` (mặc định `admin`) + mật khẩu `COMMUNITY_ADMIN_TOKEN`; `Basic` hoặc `Bearer <mật khẩu>` | Cấu hình, phát/thu hồi mã, đóng vòng, khôi phục phiên bản |

Lỗi trả `{"detail": {"code", "message"}}` với `message` tiếng Việt; giao diện hiện `message`. Mã chính: `bad_code` 403,
`bad_token` 401, `closed`/`stale_round`/`already_submitted`/`stale_parent` 409, `too_few`/`bad_labels`/`non_finite`/`no_change` 422,
`disabled`/`admin_disabled` 503, quá hạn mức 429.

### Vòng

Mỗi vòng mở cho mọi token hợp lệ; **mỗi MÃ tham gia gửi tối đa một cập nhật mỗi vòng** (chống một người vào nhiều lần bằng một mã để
chiếm cả quorum), nên K đếm số mã khác nhau. Vì vậy hãy **phát mã theo từng người** (mỗi mã vài lượt vào lại, mặc định 3), không
dùng chung một mã cho cả lớp. Đủ `k_target` cập nhật hợp lệ thì gộp
(`round_ttl_seconds`) mà có ≥ `k_min` thì gộp, chưa đủ thì gia hạn. Vòng được đóng theo yêu cầu đến (không cần bộ hẹn giờ nền) và
khôi phục đúng sau khi khởi động lại. Phiên **mặc định đóng** cho tới khi quản trị mở.

### Kiểm một cập nhật (phía máy chủ, không tin số client tự báo)

1. Độ dài đúng 155.800 byte, base64 hợp lệ, hữu hạn, đúng khóa/shape/dtype, hash head cha khớp (`validate_client_update`).
2. `num_examples` ≥ `n_min` (5); tổng phân bố nhãn khớp số ảnh; trọng số FedAvg bị chặn ở `n_cap` (200) để một người không áp đảo.
3. Chuẩn L2 của delta tính lại từ payload và **cắt ở `clip_norm`** (mặc định 1,0). Cập nhật không đổi gì bị từ chối.
4. Thân yêu cầu ≤ 320 KB (nginx ≤ 1 MB), hạn mức theo người tham gia và theo IP.

### Cổng chấp nhận mô hình

Máy chủ không có ảnh. `build_community_eval_features.py` trích sẵn đặc trưng của toàn bộ ảnh validation của release đã ghim
(float16, ~9,5 MB) bằng đúng file ONNX mà trình duyệt dùng; đánh giá một head chỉ là một phép nhân ma trận. Head ứng viên bị **từ
chối** (giữ head cũ, ghi lý do) nếu độ chính xác tổng giảm quá `epsilon` (0,5 điểm), macro-recall giảm quá `epsilon_macro`, hoặc độ
chính xác ảnh ngoài đồng PlantDoc giảm quá `epsilon_plantdoc` (2 điểm; chỉ 269 ảnh nên nhiễu hơn).

Ngưỡng `epsilon` chỉ so với head cha nên nhiều vòng nhỏ có thể trôi dần xuống; vì thế cổng còn kiểm **ngân sách tích lũy**: ứng
viên không được thấp hơn mức tốt nhất từng đạt (kể cả head gốc) quá `budget` (1 điểm; macro-recall 1 điểm; PlantDoc 3 điểm). Một chỉ số
mà ứng viên không tính được bị coi là thoái lui. Mô hình cộng đồng khởi động lại vẫn giữ nguyên vòng; trạng thái đã lưu **gắn với bộ
artifact** đã tạo ra nó (SHA ONNX, đặc trưng giám sát, head gốc): đổi artifact mà giữ `state/` thì máy chủ từ chối chạy và yêu cầu
thư mục trạng thái mới.

Lưu ý khoa học: tập này đã được dùng để chọn mô hình gốc, sai số chuẩn khoảng 0,44 điểm và cổng chỉ chặn *tụt* chứ không chứng
minh *tốt lên*. Con số chỉ là tín hiệu giám sát của hệ thống.

## Phòng thủ và riêng tư

| Mối đe dọa | Biện pháp | Còn lại |
|---|---|---|
| Đoán/dò mã | Mã 10 ký tự (~49 bit) lưu HMAC, có hạn và số lượt; khóa tạm khi nhập sai nhiều | Mã bị lộ vẫn dùng được tới khi thu hồi |
| Giả/sửa token | HMAC-SHA256, hết hạn, thu hồi theo mã hoặc theo người | Lộ `COMMUNITY_SECRET` là mất toàn bộ (xoay khóa → token cũ vô hiệu) |
| Một người nhiều token / nhiều mã | Mỗi mã một cập nhật mỗi vòng; mã có số lượt vào lại hữu hạn; hạn mức theo IP | Người có nhiều mã vẫn chiếm nhiều chỗ: phát mã theo từng người |
| Cập nhật độc hại | Cắt chuẩn, trần trọng số, FedAvg, cổng không-thoái-lui | Cập nhật nằm trong ngưỡng và không làm tụt chỉ số vẫn lọt (ví dụ cấy hành vi lệch nhẹ) |
| Nhãn sai/ác ý | Nhãn do người dùng tự gán; cổng đo trên tập có nhãn kiểm | Không có chuyên gia kiểm nhãn |
| Lộ dữ liệu qua trọng số | Chỉ gửi lớp cuối; tùy chọn client cắt chuẩn + nhiễu Gaussian (`dp_enabled`) | **Chưa có secure aggregation**; phân bố nhãn gửi kèm; DP không có bộ kế toán ε |
| Quản trị | Token bằng biến môi trường; sai nhiều bị khóa; nên giới hạn IP ở nginx | — |

Ảnh không bao giờ được gửi: `scripts/e2e_community.py` kiểm không có yêu cầu `multipart`/`image/*` nào và mọi POST < 400 KB.

## Triển khai (EC2 `datn.tranngochung.id.vn`)

1. **Sinh artifact** (máy có checkpoint và dataset; cần gói `onnx`, `onnxruntime`):
   ```powershell
   python scripts/export_community_feature_model.py        # ONNX + head khởi điểm + kiểm khớp PyTorch
   python scripts/build_community_eval_features.py         # đặc trưng giám sát; dừng nếu không tái hiện số đã công bố
   ```
2. **Chép** thư mục `artifacts/community_fl/` (trừ `state/`) lên `~/fl-demo-.../artifacts/community_fl/` trên máy chủ. Máy chủ
   chỉ cần `numpy` (đã có), không cần `onnx`/`onnxruntime`.
3. **Biến môi trường** trong `/etc/cropfed/community.env` (chmod 600; tệp ngoài repo): `COMMUNITY_SECRET` (chuỗi ngẫu nhiên, ≥ 16
   ký tự; ngắn hơn thì không ai tham gia được), `COMMUNITY_ADMIN_TOKEN` (mật khẩu quản trị, ≥ 12 ký tự; ngắn hơn thì quản trị bị tắt; nên là một cụm từ dài), tùy chọn `COMMUNITY_ADMIN_USER` (tài khoản, mặc định `admin`), ví dụ
   `openssl rand -base64 36`;
   tùy chọn `COMMUNITY_RATE_SCALE` (nhân hạn mức theo IP cho sự kiện đông người dùng chung một NAT),
   `CORS_ALLOW_ORIGINS` (mặc định không mở CORS vì giao diện và API cùng origin). Đặt `COMMUNITY_STATE_DIR` (ví dụ
   `/var/lib/cropfed-community`, thuộc `ubuntu`) để trạng thái nằm ngoài thư mục phát hành và sống qua các lần đổi bản. Tệp môi trường
   do root sở hữu, `chmod 600`; xem quản trị token bằng `sudo grep COMMUNITY_ADMIN_TOKEN /etc/cropfed/community.env`.
4. **systemd**: dùng `deploy/cropfed-demo.service` (đúng một worker, `--proxy-headers`). **nginx**: áp
   `deploy/nginx-community.conf` (trần 1 MB và hạn mức cho `/update`, gzip cho wasm, ghi đè `X-Forwarded-For`). Rồi
   `sudo systemctl daemon-reload && sudo systemctl restart cropfed-demo && sudo nginx -t && sudo systemctl reload nginx`.
5. **Kiểm tra**: `curl -fsS https://datn.tranngochung.id.vn/api/community/status | jq '{enabled, open, version}'`.

### Trang quản trị `/quan-tri`

Cách dễ nhất để vận hành: mở `https://<máy chủ>/quan-tri` và đăng nhập bằng tài khoản `admin` + mật khẩu `COMMUNITY_ADMIN_TOKEN`. Trang không nằm trong menu của trang chính và không có
liên kết từ menu công khai; quyền thật nằm ở token mà các API quản trị kiểm. Mặc định token chỉ được giữ trong `sessionStorage` của tab (mất khi
đóng tab); ô "Ghi nhớ trên máy này" — **không** tick sẵn — mới ghi thêm vào `localStorage`, và "Đăng xuất" xóa cả hai. Link dạng
`/quan-tri#token=<mật khẩu>` **đã bị gỡ** (mật khẩu trong URL lộ khi chia sẻ màn hình và lọt vào bookmark); trang vẫn xóa fragment đó
để link cũ không để mật khẩu nằm lại trên thanh địa chỉ. Trang trả `Content-Security-Policy` không cho phép script/style inline và các tiêu đề `X-Frame-Options: DENY`,
`X-Robots-Tag: noindex`, `Referrer-Policy: no-referrer`.

Trang làm được: mở/đóng phiên (đóng khi vòng đang có người gửi thì phải bấm hai lần); chỉnh K, K tối thiểu, hạn vòng, ngưỡng cắt và các
cài đặt nâng cao (có kiểm giới hạn ngay trên biểu mẫu); tạo mã (hiện đúng một lần, kèm link tham gia `/?code=…#hoc-lien-ket`, sao chép hoặc
in danh sách); thu hồi mã; xem tiến độ vòng, các phiên bản và vòng đã gộp; gộp vòng ngay; khôi phục phiên bản; xem nhật ký. Người nhận link
tham gia vào thẳng trang Học phân tán với mã đã điền sẵn (mã được xóa khỏi thanh địa chỉ sau khi đọc).

Lưu ý: link chứa mã nên chỉ gửi riêng cho từng người; chuỗi truy vấn có thể nằm lại trong nhật ký truy cập của nginx. Nginx đã giới hạn
`/api/community/admin/` ở 20 yêu cầu/phút, trang tự làm mới mỗi 10 giây (6 yêu cầu/phút). Muốn chắc hơn, giới hạn `/quan-tri` và
`/api/community/admin/` theo địa chỉ IP tin cậy (mẫu trong `deploy/nginx-community.conf`).

### Vận hành bằng curl (tương đương)

```bash
A='Authorization: Bearer '$COMMUNITY_ADMIN_TOKEN; U=https://datn.tranngochung.id.vn/api/community
# mở phiên, đặt K và hạn
curl -sS -H "$A" -H 'Content-Type: application/json' -d '{"open":true,"changes":{"k_target":10,"k_min":5,"round_ttl_seconds":1800}}' $U/admin/settings
# phát 30 mã, mỗi mã dùng 1 lần, sống 6 giờ (mã chỉ hiện một lần)
curl -sS -H "$A" -H 'Content-Type: application/json' -d '{"count":30,"max_uses":1,"ttl_hours":6,"label":"buoi-bao-ve"}' $U/admin/codes
curl -sS -H "$A" $U/admin/state                                  # phiên, cấu hình, mã, nhật ký 100 sự kiện gần nhất
curl -sS -H "$A" -X POST $U/admin/round/close                    # gộp ngay với những gì đã nhận
curl -sS -H "$A" -H 'Content-Type: application/json' -d '{"version":0}' $U/admin/rollback   # khôi phục (tạo phiên bản mới)
curl -sS -H "$A" -H 'Content-Type: application/json' -d '{"open":false,"changes":{}}' $U/admin/settings   # đóng phiên
```

Chính sách chỉnh được (có giới hạn kiểm ở `src/federated/community.py: POLICY_BOUNDS`): `k_target`, `k_min`, `round_ttl_seconds`,
`clip_norm`, `n_min`, `n_cap`, `epsilon*`, và cấu hình huấn luyện cục bộ gửi cho client (`local_epochs` 3, `local_lr` 0,01,
`local_batch` 16, `holdout_fraction` 0,2, `dp_*`).

## Bằng chứng nghiên cứu đã đo

Đo trước khi viết mã (đặc trưng thật của release đã ghim; kho “người dùng” là split calibration, không giao với train/validation):

| Hạng mục | Kết quả |
|---|---|
| ONNX phần đóng băng | 6,08 MB; cosine ≥ 0,9999999 và 49/49 dự đoán khớp PyTorch |
| Trích đặc trưng một ảnh (Chromium, wasm một luồng) | 8–21 ms desktop; 65 ms ở CPU ×4; 185 ms ở CPU ×6 |
| Tải lần đầu | wasm ~11,2 MB (gzip ~3,0 MB) + ONNX 6,1 MB; sau đó được trình duyệt ghi nhớ |
| Resize có sẵn của canvas (`resizeQuality:'high'`) | **Không khớp** `canonical_v1`: cosine tối thiểu 0,948, 2/49 dự đoán đổi → thay bằng cài đặt kiểu PIL (khớp từng byte, có test) |
| Head khởi điểm trên tập giám sát | Tái hiện đúng số bundle đã công bố (PlantVillage và PlantDoc, tối đa lệch 1 ảnh) |
| 30 cập nhật đồng thời | tất cả 200 trong 0,7 s, một lần gộp; RSS máy chủ tăng ~2 MB |

Số mô phỏng cũ (6 vòng, 5 seed, không có mã trong repo) đã được thay bằng thí nghiệm có mã và tệp kết quả ở mục dưới.

## Thí nghiệm mô phỏng (số đo mức báo cáo)

Chạy `python scripts/build_community_sim_features.py` rồi `python scripts/simulate_community.py` (mã: `src/federated/community_sim.py`, bản port numpy của `trainSoftmax`/`splitHoldout`/`makeRng` trong `community-math.js`, đã kiểm khớp JS bằng `tests/test_community_js.py`). Báo cáo: `artifacts/community_fl/simulation_report.json`; web hiện ở chương 4, mục 4.5 và ở trang Học phân tán.

- Kho người tham gia: 4225 ảnh của split calibration (37 nhóm; không giao train và validation). Mỗi người 30–100 ảnh, học 3 epoch, lr 0.01, batch 16, giữ lại 20% để kiểm.
- Máy chủ mô phỏng gọi đúng `check_update`, `aggregate_heads` (trần `n_cap`), `gate_decision` của `src/federated/community.py`; cổng chạy trên tập giám sát validation như production.
- Số báo cáo trên **test giữ riêng 11015 ảnh** (chưa dùng để chọn mô hình). Mô hình gốc FedAvg v7: chính xác 92.66%, macro-recall 89.86%, PlantDoc 50.0% (172 ảnh).
- 20 vòng, 5 seed. Bảng: thay đổi độ chính xác (điểm phần trăm) so với mô hình gốc ở vòng cuối, trung bình các seed.

| Người tham gia | K=3 | K=10 | K=25 |
|---|---|---|---|
| Ảnh bất kỳ (IID) | +1,66 | +1,83 | +1,97 |
| Mỗi người 3 nhóm | +0,39 | +1,47 | +1,50 |
| Mỗi người 1 nhóm | −0,67 | +0,60 | +1,05 |
| Chỉ ảnh ngoài đồng (PlantDoc) | +0,19 | −0,21 | −0,01 |

Đối chứng không có cổng giám sát (K=10):

- Ảnh bất kỳ (IID), K=10: chính xác +1,92 (không cổng) so với +1,83 (có cổng); PlantDoc +0,47 so với +0,12.
- Mỗi người 3 nhóm, K=10: chính xác +1,55 (không cổng) so với +1,47 (có cổng); PlantDoc −0,47 so với 0,00.
- Chỉ ảnh ngoài đồng (PlantDoc), K=10: chính xác −0,78 (không cổng) so với −0,21 (có cổng); PlantDoc −1,40 so với −0,23.

Đọc kết quả: dữ liệu người tham gia đủ đa dạng thì mô hình chung nhích lên khoảng 1–2 điểm; mỗi người một nhóm bệnh thì cần K lớn; ảnh ngoài đồng gần như không tăng (đặc trưng đóng băng học chủ yếu từ ảnh trong phòng) và cổng giúp tránh tụt. PlantDoc test chỉ 172 ảnh nên số PlantDoc dao động mạnh giữa các seed. Không thay thế thử nghiệm với người thật.

## Trọng số và độ trễ của từng vòng

Mỗi vòng ghi (cột `rounds.detail`, JSON; cơ sở dữ liệu cũ tự thêm cột khi khởi động): trọng số gộp đúng công thức FedAvg `min(n, n_cap) / tổng` của từng đóng góp (ẩn danh, không có mã người tham gia), độ lớn cập nhật trước/sau khi cắt, đã bị cắt hay chưa, `train_ms`, `extract_ms`, `rtt_ms` (do client báo, chỉ để theo dõi), thời gian từ lúc mở vòng đến lúc nhận, `delta_norm` (độ lớn thay đổi của mô hình chung), `agg_ms`, `eval_ms`, thời lượng vòng.
`GET /api/community/history` trả các trường này; trang Học phân tán và trang quản trị hiện biểu đồ thay đổi độ chính xác theo phiên bản, bảng trọng số gộp từng người và nút tải trọng số.
`GET /api/community/heads/{version}` trả tệp `.npz` (`weight` 38×1024, `bias` 38) của một phiên bản còn lưu, kèm `X-Head-SHA`; công khai, giới hạn tần suất như `/history`.

## Giới hạn đã biết

- Mới đo trên Chromium; Firefox và WebKit chưa kiểm. Ảnh rất lớn (> 16 triệu điểm ảnh, giới hạn canvas của iOS) được thu nhỏ trước
  bằng bộ lọc của trình duyệt nên lệch nhẹ so với `canonical_v1`.
- Một tiến trình duy nhất giữ khóa vòng; không mở rộng ngang được nếu không chuyển trạng thái sang CSDL dùng chung.
- Vòng chỉ đóng khi có yêu cầu đến (không có bộ hẹn giờ nền) và cần người trong phiên để đủ K; demo công khai ít người thì dùng
  `k_min` thấp hoặc tổ chức theo buổi.
- Chưa có giới hạn tổng số người/ngày ngoài số lượt của mã.
- Dữ liệu máy chủ lưu: mã (chỉ dạng HMAC), mã định danh ngẫu nhiên của người tham gia, số ảnh, phân bố nhóm, số đo cục bộ do client tự
  báo, độ lớn cập nhật, và nhật ký sự kiện (kèm địa chỉ IP của lượt nhập mã). Không có ảnh, không có tên. File head chỉ giữ 200 phiên bản
  gần nhất (và phiên bản gốc); hàng chỉ số của mọi phiên bản vẫn còn.
- **Các điểm cuối huấn luyện sẵn có của demo** (`/api/federated/train`, `/api/centralized/train`, `/api/train/stop`) không có xác thực
  và chạy việc nặng trong cùng tiến trình giữ khóa vòng. Trên máy chủ công khai nên chặn chúng ở nginx hoặc thêm xác thực; đây là hành vi
  có từ trước của demo nên chưa bị thay đổi ở đây.

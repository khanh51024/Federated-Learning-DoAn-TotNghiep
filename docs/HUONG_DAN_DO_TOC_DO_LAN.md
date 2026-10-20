# Đo tốc độ FedAvg giữa hai máy trong mạng LAN

Mục này dành cho lúc bạn có máy thứ hai cùng mạng. Phép đo cho biết: mạng giữa hai máy nhanh cỡ nào, một vòng FedAvg mất bao lâu,
bao nhiêu phần là truyền trọng số và bao nhiêu phần là máy tính toán, và trọng số đi qua mạng có giữ nguyên từng byte không.
Kết quả chỉ là số đo tốc độ, không phải độ chính xác của mô hình.

## Chuẩn bị (làm một lần, trên máy hiện tại)

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10
```

Lệnh tạo `tmp/speedtest/bundle_client_01.tar.gz` (khoảng 6 MB: mã `src/`, hợp đồng dữ liệu và 380 ảnh PlantVillage của client 01).
Chép tệp này sang máy thứ hai (USB, chia sẻ thư mục, hoặc `scp`), rồi giải nén vào một thư mục trống, ví dụ `D:\speedtest`.

## Trên máy thứ hai

Cần Python 3.11 và các gói sau, cùng phiên bản Flower với máy chủ (`flwr==1.36.0`):

```powershell
cd D:\speedtest
python -m venv venv
.\venv\Scripts\pip install flwr==1.36.0 torch torchvision numpy pillow pyyaml scikit-learn scipy pandas matplotlib
```

## Trên máy chủ (máy hiện tại)

1. Tìm địa chỉ IP trong LAN của máy chủ: `ipconfig` (dòng IPv4, ví dụ `192.168.1.20`).
2. Mở cổng cho máy thứ hai, bằng PowerShell quyền Administrator (hai cổng: Flower và cổng đo mạng):

   ```powershell
   New-NetFirewallRule -DisplayName "FL speedtest" -Direction Inbound -Protocol TCP -LocalPort 18089,18090 -Action Allow -Profile Private
   ```

3. Chạy máy chủ Flower. Dùng `--host 0.0.0.0` để nhận kết nối từ máy khác, và chính sách `--reject-suspicious-updates`
   (chế độ quan sát chỉ được chạy trên loopback). Đặt ngưỡng chuẩn đủ lớn vì thống kê BatchNorm làm chuẩn cập nhật lên tới 60:

   ```powershell
   .\.venv\Scripts\python.exe -m src.federated.server_app --mode deployment --host 0.0.0.0 --port 18089 `
     --rounds 3 --num-clients 2 --local-epochs 2 --seed 42 --require-real-clients `
     --reject-suspicious-updates --max-update-norm 500 `
     --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 `
     --metrics-file tmp/speedtest/out/lan_history.json --checkpoint-file tmp/speedtest/out/lan_global.pt
   ```

4. Chạy client 00 ngay trên máy chủ (cửa sổ PowerShell khác), trỏ tới chính nó:

   ```powershell
   .\.venv\Scripts\python.exe -m src.federated.client_app --client-id 0 --server 127.0.0.1:18089 `
     --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 `
     --require-real-data --local-epochs 2 --seed 42
   ```

## Trên máy thứ hai: chạy client 01

Thay `192.168.1.20` bằng IP máy chủ:

```powershell
cd D:\speedtest
.\venv\Scripts\python.exe -m src.federated.client_app --client-id 1 --server 192.168.1.20:18089 `
  --partition-dir partition --dataset-root images_client_01 --require-real-data --local-epochs 2 --seed 42
```

Máy chủ tự dừng sau 3 vòng và ghi `tmp/speedtest/out/lan_history.json`.

## Tóm tắt kết quả

```powershell
.\.venv\Scripts\python.exe -m scripts.summarize_network_speedtest `
  --run lan=tmp/speedtest/out/lan_history.json --machines 0=may_chu,1=may_thu_hai --label lan `
  --notes "Hai máy cùng mạng LAN, cáp/Wi-Fi: ghi rõ"
```

Kết quả nằm ở `artifacts/network_speedtest/<thời điểm>_lan/summary.json`. Các số cần đọc:

| Số | Ý nghĩa |
|---|---|
| `probe_rtt_persistent_ms` | độ trễ khứ hồi, không tính bắt tay TCP |
| `probe_up_mb_per_s`, `probe_down_mb_per_s` | băng thông mỗi chiều đo bằng khối byte cùng cỡ mô hình |
| `local_train_time_s`, `train_images_per_s` | tốc độ tính của từng máy |
| `transfer_estimate_ms` so với `predicted_transfer_ms` | thời gian truyền suy ra từ thời gian vòng, đối chiếu với thời gian tính từ băng thông |
| `upload_hash_ok`, `hash_chain_ok` | trọng số đi qua mạng giữ nguyên byte |

Nên ghi thêm: loại kết nối (Ethernet hay Wi-Fi, băng tần 2,4 hay 5 GHz) và khoảng cách tới router, vì chúng quyết định băng thông.
Sau khi đo xong, xóa luật tường lửa:

```powershell
Remove-NetFirewallRule -DisplayName "FL speedtest"
```

## Lưu ý

- Máy chủ Flower trong repo không có TLS hay xác thực; chỉ mở cổng trong mạng riêng tin cậy và chỉ trong lúc đo.
- Hai máy phải dùng cùng phiên bản Flower, cùng `--seed` và cùng `--local-epochs`, nếu không máy chủ từ chối vòng.
- Không chạy phép đo khi máy đang làm việc nặng khác: thời gian tính của từng máy sẽ bị lẫn.

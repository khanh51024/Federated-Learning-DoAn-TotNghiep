# Triển khai web demo trên EC2 — 05/10/2026

## Trạng thái

- Trang: `https://datn.tranngochung.id.vn/`, nginx chuyển về `127.0.0.1:8088`.
- **Cập nhật 2026-10-10:** dịch vụ đã chuyển sang `/home/ubuntu/fl-demo-20261010` (registry v2 + cổng web_visible, mã commit `fe8bdd3`); bản 20261005 bên dưới giữ để quay lui. Unit cũ sao lưu ở `/home/ubuntu/cropfed-demo.service.bak-20261010`.
- **Cập nhật 2026-10-13:** dịch vụ chạy từ `/home/ubuntu/fl-demo-20261013b` (giao diện sáu bước mới + học liên kết cộng đồng + trang quản trị `/quan-tri`, xem [COMMUNITY_FL.md](COMMUNITY_FL.md)); `20261013` là bản ngay trước (chưa có trang quản trị), khôi phục bằng `/home/ubuntu/cropfed-demo.service.bak-20261013b`. Bản `20261010` giữ nguyên để quay lui: chép lại `/home/ubuntu/cropfed-demo.service.bak-20261013` về `/etc/systemd/system/cropfed-demo.service`, `daemon-reload`, `restart`; nginx khôi phục từ `/home/ubuntu/nginx-site.bak-20261013` và xóa `/etc/nginx/conf.d/cropfed-community.conf`. Biến môi trường (bí mật, chmod 600) ở `/etc/cropfed/community.env`; trạng thái vòng học ở `/var/lib/cropfed-community/` (nằm ngoài thư mục phát hành nên không mất khi đổi bản). Phiên cộng đồng **mặc định đóng** cho tới khi quản trị mở. Cách phát hành: tạo thư mục mới cạnh bản đang chạy (`cp -a`), thay `src web scripts configs docs`, chạy `scripts/build_registry.py` trong thư mục mới, thử ở cổng phụ, rồi mới đổi `WorkingDirectory`/`ExecStart` của unit.
- Release: `/home/ubuntu/fl-demo-20261005`; bản cũ `/home/ubuntu/fl-project` còn giữ nguyên để khôi phục khi cần.
- Dịch vụ: `cropfed-demo.service` chạy user `ubuntu`, tự khởi động khi reboot và tự chạy lại khi tiến trình lỗi. Unit nguồn: `deploy/cropfed-demo.service`.
- Web đọc bảy checkpoint của ba nhánh đã ghim và hiển thị 14 dòng đối chứng cùng validation. Số liệu này là artifact công bố đã kiểm hash, chưa tái đánh giá bằng ảnh gốc trên EC2.
- EC2 có 2 GiB RAM, hai swapfile 4 GiB. `/swapfile-fl-demo` được thêm vào `/etc/fstab` để 5 client smoke có thể chạy sau reboot.

## Nghiệm thu trên server

Phiên `artifacts/web_runs/network_smoke/20261005T015850Z_267db2e9/` được chạy từ API HTTPS. `session.json` ghi `status=completed`, `client_processes=5`, `rounds=2`, `local_epochs=1`, `smoke_samples_per_present_class=1`. `history.json` có 5 update hợp lệ và 0 update bị loại trong **mỗi** vòng. `global.pt` SHA-256: `26321ac5e897891830d5ea42aeaef3b7b42f1a1e335f79b273abd2f80fcc02a5`.

Đây là phiên **synthetic smoke** của MobileNetV3 scratch trên partition PlantVillage cũ, không tái hiện FedAvg v7 pretrained/mixed-data và không là bằng chứng độ chính xác khoa học. Một lượt trước bị OOM và một lượt hết timeout 600 giây; log các lượt đó vẫn ở cùng thư mục `network_smoke/` với `status=failed`.

## Kiểm tra vận hành

```bash
sudo systemctl status cropfed-demo.service
curl -fsS https://datn.tranngochung.id.vn/api/demo/evidence | jq '{available, models: (.models|length)}'
curl -fsS https://datn.tranngochung.id.vn/api/federated/network-metrics | jq '{historical_run, scientific_valid, audit_rounds: [.audit_rounds[] | {round, accepted_clients}]}'
swapon --show
```

Web không cần Flower server thường trú. Khi bấm Start, nó tạo một server cổng 8089 và năm client localhost, rồi dừng các tiến trình sau phiên. Cổng 8080 của Flower cũ đã được dừng vì log ghi 0 client; bản cũ vẫn còn trên đĩa.

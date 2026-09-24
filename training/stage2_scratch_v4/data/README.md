# Hướng dẫn Quản lý và Tái tạo Dữ liệu Phân hoạch Stage 2 (`data/README.md`)

Thư mục này quản lý các tệp cấu hình, fixtures kiểm định tính nhất quán và hướng dẫn lấy bộ dữ liệu phân hoạch (partition suite) phục vụ huấn luyện Stage 2 Scratch (FedAvg-only).

## Các tệp tin nằm trực tiếp trong repository:
- `duplicate_review.json`: Bảng đối chiếu các cặp ảnh trùng lặp hoặc cận trùng lặp nội bộ lớp (SHA-256 đối chiếu).
- `visually_verified_pairs.json`: 6 cặp ảnh tương đồng thị giác đã xác minh kiểm định rò rỉ cross-split.
- `four_visually_verified_pairs.json`: Tập con fixture kiểm định nhanh.

## 1. Bộ phân hoạch đầy đủ (Externalized Suite)

Để giữ repository gọn nhẹ khi clone và phát hành, toàn bộ dữ liệu phân hoạch 8 điều kiện non-IID (`partitions_stage2_scratch_v3`) được lưu trữ bên ngoài repo tại:
```text
training-data/stage2/partitions_stage2_scratch_v3/
```

### Thông số kỹ thuật bộ phân hoạch (Suite Descriptor):
- **Đường dẫn chuẩn:** `training-data/stage2/partitions_stage2_scratch_v3`
- **Tổng số tệp:** 84 files
- **Tổng dung lượng:** 264.037.275 bytes (~251.8 MB)
- **Giao thức suite:** `stage2_scratch_clean_v3` (trong `suite.json` và preflight runtime; lưu ý `stage2_scratch_fedavg_v4` là giao thức riêng của runner/release package, không phải giao thức suite)
- **Số client:** 5 clients (`client_00.csv` .. `client_04.csv`)
- **Random seed:** 42
- **Tỷ lệ phân chia mục tiêu:** Train 0.72, Val 0.08, Test 0.20
- **Tổng số ảnh:** 54.305 ảnh (Train: 38.984, Val: 4.400, Test: 10.921)
- **Số lớp:** 38 classes (PlantVillage)
- **Số điều kiện non-IID (8 điều kiện):**
  1. `feature01`: Feature Dirichlet skew mạnh ($\alpha=0.1$)
  2. `feature100`: Feature Dirichlet skew nhẹ ($\alpha=100.0$)
  3. `label01`: Dirichlet label skew mạnh ($\alpha=0.1$)
  4. `label1`: Dirichlet label skew vừa ($\alpha=1.0$)
  5. `label100`: Dirichlet label skew gần IID ($\alpha=100.0$)
  6. `label_quantity01`: Mixed label & quantity skew ($\alpha=0.1$, $\alpha_{qty}=0.1$)
  7. `quantity01`: Quantity/volume Dirichlet skew mạnh ($\alpha_{qty}=0.1$)
  8. `quantity100`: Quantity/volume Dirichlet skew nhẹ ($\alpha_{qty}=100.0$)

### Bảng Checksums đã Pin (Pinned Hashes):
- **Tree Fingerprint (84 files sorted POSIX):** `eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15`
- **suite.json SHA-256:** `109d8b4ddc97c2f31d7c07a5a9328266226908b0ccedb63820689a11873f9350`
- **duplicate_review.json SHA-256:** `53bbb477f71e0512535f58d17835e77cd727c65f4354e034ec1761120f0c2f74`
- **visually_verified_pairs.json SHA-256:** `257cbcc1b8a77af1c59ab26eff71e191c0763db0cb6c1b5050e2019500a626f1`
- **Condition Manifest Hashes & Phân bổ Client (n_k):**
  - `feature01`: manifest=`3255e9cade00da36f2f61f7f54a071d7197bac3bdc8a29855d35fdf2517ecf66`, n_k=[5133, 18545, 6779, 614, 7913]
  - `feature100`: manifest=`47497ca66a1b9e039d6a50b2ede5b07ae25df269a3c741be8e327cfa8802fe2b`, n_k=[8136, 8050, 7434, 7936, 7428]
  - `label01`: manifest=`96b6538185969974b6d4f2ba0b7565b19cf61f96a8a9ab99c53141e9ace7b889`, n_k=[10851, 7205, 6069, 8453, 6406]
  - `label1`: manifest=`951387f8fd3f45712a57908161daefe1a87a9a769cc8eccbc3f720050906bf6a`, n_k=[9399, 6566, 8419, 7985, 6615]
  - `label100`: manifest=`bf7df208557b2fdfda5edfb62a5b0e167b84b2e82e53575ca5500d2ffa860417`, n_k=[7568, 7699, 7817, 8032, 7868]
  - `label_quantity01`: manifest=`f2b4c43a6d7777630beba652fa1666c9b08b593fd18c6c03a67331dd4e831157`, n_k=[10, 14119, 12163, 12574, 118]
  - `quantity01`: manifest=`9fd2cacf5f3317b69e5327e0360651d92593f0c677de59fa94cf6ac7299dc0d8`, n_k=[10, 6288, 31362, 1314, 10]
  - `quantity100`: manifest=`f1e876649d508c34f108dac817f1cbcaa756116db8739bc9072b2f06f87232a9`, n_k=[7638, 8075, 7366, 7134, 8771]

Mỗi thư mục điều kiện gồm:
- `partition_config.json`: Cấu hình tham số phân hoạch, mapping client profiles
- `image_content.json`: Bảng SHA-256 hash của từng ảnh trong điều kiện (54.305 mục)
- `centralized_train.csv`: Tập train union 38.984 mẫu (13 cột chuẩn)
- `global_val.csv`, `global_test.csv`: Tập holdout val (4.400 mẫu) và test (10.921 mẫu)
- `clients/client_00.csv` đến `client_04.csv`: Danh sách mẫu phân bổ cho 5 client

---

## 2. Quy trình Lấy Dữ liệu Nội bộ và Tái tạo (Internal Acquisition & Reproduction)

### A. Nguồn Artifact Nội bộ (Authorized Internal Source)
Bộ phân hoạch bất biến được lưu trữ nội bộ tại:
- **Đường dẫn thư mục chuẩn:** `D:\university\do-an-tot-nghiep\training-data\stage2\partitions_stage2_scratch_v3`
- **Gói nén phát hành tham chiếu (ZIP r5):** `D:\university\do-an-tot-nghiep\training-artifacts\stage2\candidate-r5\package_build\fl_package_v4.zip` (nguồn lưu trữ tham khảo, chứa đầy đủ `data/partitions_stage2_scratch_v3/` với 84 tệp; không dùng làm source code candidate r7). Quy trình vận hành chuẩn ưu tiên sao chép trực tiếp từ thư mục nội bộ.

### B. Quy trình Acquisition Không Ghi đè (Non-destructive Acquisition Workflow)
> [!NOTE]
> Tiện ích `scripts/acquire_internal_suite.py` là công cụ chạy từ mã nguồn checkout (`source checkout tooling`). Bản build wheel chỉ phân phối các gói thư viện runtime (`stage2_scratch`, `stage1_compat`, `stage2_matched`, `src`, `fl_training`) và không đóng gói namespace `scripts/`.

Để lấy dữ liệu vào một vị trí làm việc mới (ví dụ trong checkout hoặc thư mục artifact riêng), thực hiện lệnh PowerShell sau:

```powershell
# Thực thi từ thư mục gốc candidate hoặc checkout nguồn
python scripts/acquire_internal_suite.py `
  --source "D:\university\do-an-tot-nghiep\training-data\stage2\partitions_stage2_scratch_v3" `
  --target-dir "path/to/target/partitions_stage2_scratch_v3" `
  --json-out "path/to/outside/acquire_report.json"
```

Tiện ích sẽ tự động thực hiện:
1. Kiểm tra quyền đọc của nguồn (`read permission`).
2. Kiểm tra đích an toàn: từ chối nếu thư mục đích đã tồn tại (kể cả thư mục rỗng) để đảm bảo không ghi đè; tuyệt đối không chấp nhận cờ `--allow-overwrite`.
3. Kiểm tra an toàn báo cáo `--json-out`: từ chối xuất report bên trong cây nguồn hoặc đích, từ chối ghi đè lên file report đã có sẵn.
4. Sao chép cây thư mục đầy đủ 84 tệp mà không sử dụng `dirs_exist_ok=True`.
5. Kiểm định toàn vẹn nghiêm ngặt:
   - Tổng số tệp = 84.
   - Dung lượng uncompressed = 264.037.275 bytes.
   - Mã băm toàn cây = `eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15`.
   - Khớp 100% mã băm của `suite.json`, `duplicate_review.json`, `visually_verified_pairs.json` và 8 condition manifests.
6. Chỉ in thông báo `[Acquire] SUCCESS` sau khi đã xác minh thành công và xuất report an toàn.

### C. Chạy Đóng gói và Offline Preflight từ Dữ liệu Vừa Nhận
Sau khi acquisition thành công, chạy lệnh đóng gói và kiểm định preflight từ export bằng cách truyền đường dẫn tường minh:

```powershell
python -m scripts.package_release_v4 `
  --suite "path/to/target/partitions_stage2_scratch_v3" `
  --output-dir "path/to/target/package_output"
```

Thiết lập biến môi trường để chạy integration test trong PowerShell:
```powershell
$env:FEDAVG_SUITE_DIR = "path/to/target/partitions_stage2_scratch_v3"
pytest tests/test_package_and_notebook_v4.py -k test_validate_suite_real_dataset_integration
```

Hoặc trong Bash:
```bash
export FEDAVG_SUITE_DIR="path/to/target/partitions_stage2_scratch_v3"
pytest tests/test_package_and_notebook_v4.py -k test_validate_suite_real_dataset_integration
```

---

## 3. Trạng thái Tái tạo và Blocker Công khai (Reproduction Status & Blocker B01)

### A. Tái tạo Nội bộ (`reproducibility_internal`):
- **Trạng thái:** **`VERIFIED (INTERNAL)`**.
- **Điều kiện:** Đã hoàn thành phép thử rehearsal thực tế từ export biệt lập sang thư mục nhận mới, xác minh 84 files, 264.037.275 bytes, mã băm `eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15`, và build package release thành công 100%.

### B. Tái tạo Ngoài Clone / Môi trường Độc lập (`reproducibility_external`):
- **Trạng thái:** **`BLOCKED / NOT_AVAILABLE`**.
- **Lý do (Blocker B01):** Hiện chưa có public release bucket hoặc URL công khai tải tự động bộ phân hoạch và dataset ảnh PlantVillage ngoài phạm vi storage nội bộ của máy trạm.
- **Cam kết:** Tuyệt đối không tạo URL giả mạo, không tự ý tải dữ liệu lên dịch vụ công cộng trái phép để giả đóng blocker. Blocker B01 tiếp tục được giữ công khai cho đến khi có hạ tầng lưu trữ chính thức.

### C. Công cụ chia phân hoạch thử nghiệm (`scripts/partition_dataset.py`):
`scripts/partition_dataset.py` là tiện ích phân hoạch thử nghiệm đơn lẻ cho một điều kiện riêng biệt (legacy utility), KHÔNG phải là pipeline generator tạo toàn bộ bộ 8 điều kiện `partitions_stage2_scratch_v3`.
Cú pháp cờ lệnh chuẩn (PowerShell):
```powershell
python scripts/partition_dataset.py `
  --dataset-path path/to/PlantVillage-Dataset/raw/color `
  --output-path data/custom_partition `
  --scenario label_skew `
  --alpha 0.1 `
  --num-clients 5 `
  --seed 42
```

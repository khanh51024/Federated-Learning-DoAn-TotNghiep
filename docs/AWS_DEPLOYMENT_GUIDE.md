# Hướng Dẫn Triển Khai Hệ Thống Federated Learning Trên AWS EC2 (End-to-End)

Tài liệu này hướng dẫn chi tiết từng bước (End-to-End) cách triển khai **Hệ thống Máy chủ Federated Learning (Flower Server)** và **Web Dashboard (FastAPI)** lên VPS **Amazon Web Services (AWS EC2)**, đồng thời kết nối các **Client (PC cá nhân / Laptop / Máy thuê)** từ xa qua Internet.

---

## 1. Kiến Trúc Triển Khai (Deployment Architecture)

```
       +----------------------------------------------------------------+
       |                       AWS CLOUD (EC2 VPS)                      |
       |  Public IP: 13.250.xx.xx (Ubuntu 22.04 LTS / 2 vCPU, 2-4GB)    |
       |                                                                |
       |   +-----------------------+        +-----------------------+   |
       |   |   Web Dashboard UI    |        |   Flower FedAvg       |   |
       |   |   (FastAPI/Uvicorn)   |        |   Central Server      |   |
       |   |   Cổng: 8088 (HTTP)   |        |   Cổng: 8080 (gRPC)   |   |
       |   +-----------^-----------+        +-----------^-----------+   |
       |               |                                |               |
       +---------------+--------------------------------+---------------+
                       |                                |
       Internet        | HTTP                           | TCP / gRPC
       Requests        | (Xem Dashboard)                | (Gửi/Nhận Weights)
                       |                                |
   +-------------------+---+            +---------------+---------------+
   | Trình duyệt Web       |            |   Các Client Phân Tán         |
   | Thầy cô / Hội đồng    |            |   (PC cá nhân / Laptop / Thuê)|
   | (Laptop / Điện thoại) |            |                               |
   | http://13.250.xx:8088 |            | Client 0: python client_app.py|
   +-----------------------+            | Client 1: python client_app.py|
                                        | Client 2: python client_app.py|
                                        | Client 3: python client_app.py|
                                        | Client 4: python client_app.py|
                                        +-------------------------------+
```

### Tại sao kiến trúc này tối ưu chi phí và hiệu năng?
1. **Server trên VPS**: Chỉ thực hiện thuật toán FedAvg (tính trung bình cộng tensor số học) và phục vụ giao diện Web Dashboard, không cần huấn luyện backpropagation nặng nên **KHÔNG CẦN GPU ĐẮT TIỀN** (tiết kiệm 95% chi phí thuê cloud).
2. **Client ở máy cá nhân**: Tận dụng GPU RTX của máy tính sinh viên/laptop để huấn luyện local trên dữ liệu lá cây PlantVillage. Trọng số cập nhật sau mỗi epoch được đẩy qua Internet lên VPS bằng giao thức gRPC bảo mật của Flower.

---

## 2. Giai Đoạn 1: Tạo và Khởi Động EC2 Instance Trên AWS Console

### Bước 1.1: Đăng nhập AWS & Chọn Region
- Truy cập [AWS Management Console](https://console.aws.amazon.com/).
- Chọn Region gần Việt Nam nhất để giảm độ trễ mạng: **Singapore (`ap-southeast-1`)** hoặc **Tokyo (`ap-northeast-1`)**.

### Bước 1.2: Khởi tạo Instance (Launch Instance)
- Vào dịch vụ **EC2** -> Click **Launch Instance**.
- **Name**: `FL-PlantVillage-Server`
- **Application and OS Images (AMI)**: Chọn **Ubuntu Server 24.04 LTS** hoặc **Ubuntu Server 22.04 LTS** (x86_64).
- **Instance Type**:
  - Khuyến nghị tốt nhất: `t3.small` (2 vCPU, 2 GiB RAM, giá ~$0.0208/giờ ~ 500đ/giờ) hoặc `t3.medium` (2 vCPU, 4 GiB RAM).
  - Nếu dùng tài khoản Free Tier mới: Có thể chọn `t2.micro` hoặc `t3.micro` (nhưng bắt buộc phải tạo Swap 2GB-4GB ở bước sau để tránh tràn RAM).
- **Key pair (login)**:
  - Chọn **Create new key pair**.
  - Key pair name: `fl-aws-key`
  - Key pair type: `RSA`
  - Private key file format: `.pem` (cho OpenSSH)
  - Nhấn **Create key pair** và lưu file `fl-aws-key.pem` vào thư mục an toàn trên máy bạn (ví dụ: `C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem`).
- **Configure Storage**:
  - Thay đổi dung lượng lên **25 GiB** đến **30 GiB** (loại `gp3`). Dung lượng này nằm trong hạn mức 30GB miễn phí của AWS và đủ chỗ cho hệ điều hành, Python packages và lưu checkpoint.

---

## 3. Giai Đoạn 2: Cấu Hình Tường Lửa (Security Group) - RẤT QUAN TRỌNG

Hệ thống cần mở 3 cổng mạng để hoạt động:

| Cổng (Port) | Giao thức | Tên gọi | Nguồn (Source) | Mục đích sử dụng |
| :--- | :--- | :--- | :--- | :--- |
| **22** | TCP | SSH | `My IP` hoặc `0.0.0.0/0` | Đăng nhập dòng lệnh từ máy tính cá nhân |
| **8088** | TCP | Custom TCP | `0.0.0.0/0` (Anywhere) | Mở Web Dashboard cho thầy cô / hội đồng truy cập |
| **8080** | TCP | Custom TCP | `0.0.0.0/0` (Anywhere) | Cổng Flower Server lắng nghe Client đẩy weights |

### Thao tác trên AWS:
1. Trong phần **Network settings**, chọn **Create security group**.
2. Đặt tên: `fl-server-sg`.
3. Nhấp **Add security group rule** để thêm:
   - Rule 1: Type `SSH`, Port `22`, Source: `0.0.0.0/0` (hoặc `My IP`).
   - Rule 2: Type `Custom TCP`, Port Range: `8088`, Source: `0.0.0.0/0` (Description: `FastAPI Web UI`).
   - Rule 3: Type `Custom TCP`, Port Range: `8080`, Source: `0.0.0.0/0` (Description: `Flower gRPC Server`).
4. Nhấn **Launch instance** để tạo máy chủ.

*Ghi lại **Public IPv4 address** của máy sau khi khởi động (ví dụ: `13.250.45.88`).*

---

## 4. Giai Đoạn 3: Kết Nối SSH & Cài Đặt Môi Trường VPS

Mở **PowerShell** hoặc **Windows Terminal** trên máy tính của bạn:

### Bước 3.1: Phân quyền key .pem trên Windows (tránh lỗi Unprotected Private Key)
Nếu gặp lỗi `Permissions for ... are too open`, chạy các lệnh sau trong PowerShell:
```powershell
icacls "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" /inheritance:r
icacls "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" /grant:r "$($env:USERNAME):(R)"
```

### Bước 3.2: Đăng nhập SSH vào VPS
```bash
ssh -i "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" ubuntu@<PUBLIC_IPV4_AWS>
```
*(Thay `<PUBLIC_IPV4_AWS>` bằng IP thực tế của bạn, ví dụ: `ubuntu@13.250.45.88`)*

---

### Bước 3.3: Tạo RAM ảo (Swapfile 4GB) trên VPS
Giúp VPS không bao giờ bị tắt tiến trình do Out-Of-Memory (OOM):
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

### Bước 3.4: Cập nhật hệ thống & Cài đặt gói cần thiết
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git tmux htop ufw
```

---

## 5. Giai Đoạn 4: Đưa Mã Nguồn Dự Án Lên VPS & Cài Thư Viện

### Cách 1: Đẩy code qua Git (Khuyên dùng)
Trên VPS, clone trực tiếp repository của bạn:
```bash
git clone https://github.com/<your-username>/<your-repo>.git fl-project
cd fl-project
```

### Cách 2: Copy trực tiếp từ máy cá nhân lên VPS qua SCP (Nếu chưa đưa lên GitHub)
Trên máy tính của bạn (mở PowerShell mới):
```powershell
scp -i "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" -r "d:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old\src" ubuntu@<PUBLIC_IPV4_AWS>:~/fl-project/
scp -i "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" -r "d:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old\web" ubuntu@<PUBLIC_IPV4_AWS>:~/fl-project/
scp -i "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" -r "d:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old\data" ubuntu@<PUBLIC_IPV4_AWS>:~/fl-project/
scp -i "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" -r "d:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old\artifacts" ubuntu@<PUBLIC_IPV4_AWS>:~/fl-project/
scp -i "C:\Users\<Tên_Bạn>\.ssh\fl-aws-key.pem" "d:\Federated-Learning-DoAn-TotNghiep-code-train-fedavg-old\requirements.txt" ubuntu@<PUBLIC_IPV4_AWS>:~/fl-project/
```

### Bước 4.1: Tạo môi trường ảo Python & Cài đặt PyTorch CPU
Tại thư mục `~/fl-project` trên VPS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# QUAN TRỌNG: Cài PyTorch CPU-only (Chỉ ~180MB thay vì bản CUDA 2.5GB ngốn đĩa)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện dự án
pip install -r requirements.txt
```

---

## 6. Giai Đoạn 5: Khởi Chạy Dịch Vụ Trên VPS Bằng `tmux`

Để máy chủ Web và Flower Server tiếp tục chạy ngay cả khi bạn tắt máy tính cá nhân hoặc ngắt kết nối SSH, ta sử dụng công cụ **`tmux`**.

### 5.1. Khởi chạy Web Dashboard (Cửa sổ tmux 1)
```bash
tmux new -s web_dashboard
source .venv/bin/activate
python3 -m uvicorn web.app:app --host 0.0.0.0 --port 8088
```
*Nhấn tổ hợp phím **`Ctrl + B`** rồi bấm phím **`D`** để thoát ra ngoài (tiến trình vẫn chạy ngầm).*

### 5.2. Khởi chạy Flower FedAvg Server (Cửa sổ tmux 2)
```bash
tmux new -s flower_server
source .venv/bin/activate
python3 -m src.federated.server_app --mode deployment --port 8080 --num-clients 5 --rounds 5
```
*Nhấn tổ hợp phím **`Ctrl + B`** rồi bấm phím **`D`** để thoát ra ngoài.*

> **Mẹo quản lý `tmux`:**
> - Xem danh sách các phiên đang chạy: `tmux ls`
> - Quay lại xem log của web: `tmux attach -t web_dashboard`
> - Quay lại xem log của server: `tmux attach -t flower_server`
> - Dừng phiên: Vào phiên đó và nhấn `Ctrl + C` rồi gõ `exit`.

---

## 7. Giai Đoạn 6: Kết Nối Client Từ Máy Cá Nhân (Laptop / PC)

Trên máy tính cá nhân của bạn (nơi lưu trữ dataset PlantVillage):

### Chạy 5 Client phân tán kết nối lên VPS AWS:

Mở 5 cửa sổ PowerShell riêng biệt (hoặc viết file `.bat` chạy song song):

```powershell
# Client 0
python -m src.federated.client_app --client-id 0 --server <PUBLIC_IPV4_AWS>:8080

# Client 1
python -m src.federated.client_app --client-id 1 --server <PUBLIC_IPV4_AWS>:8080

# Client 2
python -m src.federated.client_app --client-id 2 --server <PUBLIC_IPV4_AWS>:8080

# Client 3
python -m src.federated.client_app --client-id 3 --server <PUBLIC_IPV4_AWS>:8080

# Client 4
python -m src.federated.client_app --client-id 4 --server <PUBLIC_IPV4_AWS>:8080
```

#### File kịch bản chạy 1-click trên Windows (`run_5_clients.bat`):
Tạo file `run_5_clients.bat` trong thư mục dự án trên máy cá nhân:
```bat
@echo off
set AWS_SERVER=<PUBLIC_IPV4_AWS>:8080
echo Dang khoi chay 5 Client FL ket noi den %AWS_SERVER%...

start cmd /k "python -m src.federated.client_app --client-id 0 --server %AWS_SERVER%"
start cmd /k "python -m src.federated.client_app --client-id 1 --server %AWS_SERVER%"
start cmd /k "python -m src.federated.client_app --client-id 2 --server %AWS_SERVER%"
start cmd /k "python -m src.federated.client_app --client-id 3 --server %AWS_SERVER%"
start cmd /k "python -m src.federated.client_app --client-id 4 --server %AWS_SERVER%"

echo Tat ca 5 client da duoc kich hoat!
```

---

## 8. Giai Đoạn 7: Kiểm Thử Thực Tế & Demo Trước Hội Đồng

1. **Truy cập Web Dashboard**:
   Mở trình duyệt bất kỳ (Chrome/Edge/Safari trên laptop hoặc smartphone):
   ```
   http://<PUBLIC_IPV4_AWS>:8088
   ```
2. **Kịch bản Demo ấn tượng trước Hội đồng Thầy Cô**:
   - **Gửi link trực tiếp**: Thầy cô trong hội đồng có thể dùng điện thoại/laptop cá nhân mở link `http://<PUBLIC_IPV4_AWS>:8088` để xem trực tiếp hệ thống hoạt động thực tế trên Cloud.
   - **Tab 1 (Dashboard Overview)**: Cho thấy hệ thống đang online, số round, cấu trúc mô hình MobileNetV3-Small (930K tham số).
   - **Tab 3 (Federated Training)**: Khi các Client hoàn thành từng round, biểu đồ Accuracy/Loss cập nhật tức thì.
   - **Tab 5 (Weight Inspector)**: Minh chứng tính toàn vẹn trọng số với SHA-256 hash và L2 update norm theo thời gian thực.
   - **Tab 6 (Inference Test)**: Thầy cô chụp một bức ảnh lá cây bị bệnh (táo, cà chua, khoai tây...) và upload lên web để xem model dự đoán lớp bệnh và độ tin cậy confidence trong vòng vài mili-giây.

---

## 9. Giai Đoạn 8: Quản Lý Chi Phí & Tắt Máy Khi Không Dùng

- **Khi không làm việc / không demo**:
  1. Vào AWS Console -> EC2 -> Instances.
  2. Chọn máy chủ -> Nhấp **Instance state** -> Chọn **Stop instance** (KHÔNG chọn *Terminate* vì Terminate sẽ xóa vĩnh viễn ổ đĩa).
  3. Khi ở trạng thái *Stopped*, bạn chỉ phải trả tiền lưu trữ ổ cứng EBS (~0.08$/tháng cho 25GB ~ 2.000 VNĐ), **hoàn toàn không tốn tiền CPU/RAM**.
- **Khi cần demo lại**:
  1. Nhấp **Start instance**.
  2. Lưu ý: Public IP có thể thay đổi sau khi Start lại (trừ khi bạn gán **Elastic IP** cố định).
  3. SSH vào máy và chạy lại lệnh trong `tmux`.

---

## 10. Giai Đoạn 9: Cấu Hình Tên Miền Riêng (Custom Domain) & SSL HTTPS

Hệ thống đã được thiết lập gắn tên miền riêng **`datn.tranngochung.id.vn`** kèm chứng chỉ bảo mật SSL HTTPS miễn phí trọn đời (Let's Encrypt):

### 10.1. Bản ghi DNS
- **Loại (Type)**: `A`
- **Host / Name**: `datn`
- **Giá trị (Value)**: `<PUBLIC_IPV4_AWS>` (ví dụ: `54.253.69.179`)

### 10.2. Cấu hình Nginx Reverse Proxy
File cấu hình tại `/etc/nginx/sites-available/datn.tranngochung.id.vn`:
```nginx
server {
    server_name datn.tranngochung.id.vn;
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8088;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 10.3. Kích hoạt HTTPS với Certbot
```bash
sudo certbot --nginx -d datn.tranngochung.id.vn --non-interactive --agree-tos --redirect
```

Bây giờ bất kỳ ai cũng có thể truy cập hệ thống an toàn và chuyên nghiệp tại:
👉 **`https://datn.tranngochung.id.vn`**

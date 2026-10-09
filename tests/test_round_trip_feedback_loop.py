"""
tests.test_round_trip_feedback_loop:
Kiểm định trực tiếp vòng lặp phản hồi 2 chiều (Bidirectional Feedback Loop) giữa Server và các Client:
1. Server phát tán Global Model w_t xuống các Client.
2. Client nhận w_t, đối soát mã băm SHA-256 (Server -> Client).
3. Client huấn luyện cục bộ bằng dữ liệu riêng, sinh ra w_(k, t+1) khác với w_t.
4. Client đẩy bản cập nhật w_(k, t+1) + số mẫu n_k lên Server (Client -> Server).
5. Server đối soát toàn vẹn mã băm, tổng hợp FedAvg: w_(t+1) = sum(n_k * w_k) / sum(n_k).
6. [ĐIỂM THEN CHỐT THEO LỜI THẦY]: Server phát tán mô hình ĐÃ ĐƯỢC TỔNG HỢP w_(t+1) TRỞ LẠI XUỐNG CÁC CLIENT ở vòng t+1.
7. Client nhận w_(t+1) ở đầu vòng t+1, chứng minh:
   - Trọng số mới nhận KHÔNG CÒN LÀ trọng số cục bộ riêng của nó ở cuối vòng t.
   - Trọng số mới nhận KHỚP 100% với trọng số Server vừa tổng hợp từ tất cả các client.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.federated.client_app import PlantVillageClient
from src.federated.strategy import aggregate_fedavg
from src.model.mobilenetv3 import (
    compute_model_hash,
    create_model,
    get_parameters,
    set_parameters,
)


def test_bidirectional_server_client_feedback_loop():
    """
    Kiểm chứng thực nghiệm chặt chẽ quy trình 2 vòng (2 rounds) trên 2 clients:
    Chứng minh trọng số đẩy lên server, server tổng hợp, rồi đẩy ngược xuống client chính xác 100%.
    """
    device = torch.device("cpu")

    # 1. SERVER KHỞI TẠO GLOBAL MODEL (w_0)
    torch.manual_seed(42)
    np.random.seed(42)
    global_model = create_model(num_classes=38, pretrained=False).to(device)
    w_0 = get_parameters(global_model)
    hash_w0 = compute_model_hash(w_0)

    # 2. KHỞI TẠO 2 CLIENTS ĐỘC LẬP VỚI 2 PHÂN HOẠCH DỮ LIỆU KHÁC NHAU
    client_0 = PlantVillageClient(client_id=0, smoke=True, device=device)
    client_1 = PlantVillageClient(client_id=1, smoke=True, device=device)

    # =========================================================================
    # ROUND 1: SERVER -> CLIENTS -> TRAIN -> CLIENTS -> SERVER -> AGGREGATE
    # =========================================================================
    print("\n--- BẮT ĐẦU ROUND 1 ---")

    # BƯỚC 1: Server đẩy w_0 xuống Client 0 và Client 1
    # BƯỚC 2: Clients nhận w_0, kiểm tra mã băm
    reply_0 = client_0.fit(w_0, round_num=1)
    reply_1 = client_1.fit(w_0, round_num=1)

    assert reply_0["recv_hash"] == hash_w0, "Client 0 phải nhận đúng chính xác w_0 từ server"
    assert reply_1["recv_hash"] == hash_w0, "Client 1 phải nhận đúng chính xác w_0 từ server"

    # BƯỚC 3: Client huấn luyện cục bộ -> trọng số sau train w_(k,1) phải thay đổi so với w_0
    w_c0_r1 = reply_0["weights"]
    w_c1_r1 = reply_1["weights"]
    hash_c0_r1 = reply_0["hash"]
    hash_c1_r1 = reply_1["hash"]

    assert hash_c0_r1 != hash_w0, "Trọng số Client 0 sau huấn luyện phải khác w_0 ban đầu"
    assert hash_c1_r1 != hash_w0, "Trọng số Client 1 sau huấn luyện phải khác w_0 ban đầu"
    assert hash_c0_r1 != hash_c1_r1, "Client 0 và Client 1 có dữ liệu khác nhau nên update phải khác nhau"

    # BƯỚC 4: Client gửi update + n_k lên Server. Server kiểm tra SHA-256 toàn vẹn
    assert compute_model_hash(w_c0_r1) == hash_c0_r1, "Mã băm truyền từ Client 0 lên Server phải khớp tuyệt đối"
    assert compute_model_hash(w_c1_r1) == hash_c1_r1, "Mã băm truyền từ Client 1 lên Server phải khớp tuyệt đối"

    # BƯỚC 5: Server thực hiện tổng hợp FedAvg theo số mẫu n_k
    n0 = reply_0["num_examples"]
    n1 = reply_1["num_examples"]
    client_replies_r1 = [reply_0, reply_1]

    w_1_server = aggregate_fedavg(client_replies_r1, parent_weights=w_0)
    hash_w1_server = compute_model_hash(w_1_server)

    # Kiểm tra toán học: w_1_server phải là kết hợp có trọng số của cả 2 client
    # w_1_server != w_c0_r1 và w_1_server != w_c1_r1
    assert hash_w1_server != hash_c0_r1, "Mô hình toàn cục sau tổng hợp phải khác mô hình riêng của Client 0"
    assert hash_w1_server != hash_c1_r1, "Mô hình toàn cục sau tổng hợp phải khác mô hình riêng của Client 1"

    # Cập nhật Server global model
    set_parameters(global_model, w_1_server)

    # =========================================================================
    # ROUND 2: [CHỨNG MINH ĐƯA XUỐNG LẠI CLIENT]
    # SERVER ĐẨY w_1_server XUỐNG LẠI CHO CLIENTS
    # =========================================================================
    print("\n--- BẮT ĐẦU ROUND 2: ĐƯA MÔ HÌNH TỔNG HỢP XUỐNG LẠI CLIENT ---")

    # Server gửi w_1_server xuống cho Client 0 và Client 1
    reply_0_r2 = client_0.fit(w_1_server, round_num=2)
    reply_1_r2 = client_1.fit(w_1_server, round_num=2)

    # CHỨNG MINH 1: Client 0 và Client 1 nhận được ĐÚNG CHÍNH XÁC mô hình mà Server vừa tổng hợp
    assert reply_0_r2["recv_hash"] == hash_w1_server, (
        f"LỖI: Client 0 không nhận đúng w_1 từ Server! Recv: {reply_0_r2['recv_hash'][:8]} != Server: {hash_w1_server[:8]}"
    )
    assert reply_1_r2["recv_hash"] == hash_w1_server, (
        f"LỖI: Client 1 không nhận đúng w_1 từ Server! Recv: {reply_1_r2['recv_hash'][:8]} != Server: {hash_w1_server[:8]}"
    )

    # CHỨNG MINH 2: Trọng số bắt đầu Round 2 tại Client 0 KHÔNG PHẢI là trọng số w_c0_r1 của chính nó ở cuối Round 1
    assert reply_0_r2["recv_hash"] != hash_c0_r1, (
        "Client 0 phải nhận mô hình toàn cục đã có sự đóng góp của Client 1, không được giữ lại trọng số cô lập cũ"
    )

    # CHỨNG MINH 3: Client đo đạc chính xác Update Norm Delta_k so với w_1_server vừa nhận:
    # Delta_k = w_k,2 - w_1_server
    assert reply_0_r2["update_norm"] > 0, "Client 0 phải đo được độ lệch cập nhật ||Delta_k|| > 0 so với w_1_server"
    assert reply_1_r2["update_norm"] > 0, "Client 1 phải đo được độ lệch cập nhật ||Delta_k|| > 0 so với w_1_server"

    # CHỨNG MINH 4: Trọng số sau train Round 2 của Client 0 đã tiến hóa từ w_1_server
    hash_c0_r2 = reply_0_r2["hash"]
    hash_c1_r2 = reply_1_r2["hash"]
    assert hash_c0_r2 != hash_w1_server, "Client 0 train round 2 phải tiếp tục sinh ra trọng số mới"
    assert hash_c1_r2 != hash_w1_server, "Client 1 train round 2 phải tiếp tục sinh ra trọng số mới"

    # BƯỚC CUỐI: Server nhận cả 2 updates Round 2 và tổng hợp ra w_2_server
    w_2_server = aggregate_fedavg([reply_0_r2, reply_1_r2], parent_weights=w_1_server)
    hash_w2_server = compute_model_hash(w_2_server)

    assert hash_w2_server != hash_w1_server, "Mô hình toàn cục Round 2 phải tiếp tục tiến hóa"
    assert hash_w2_server != hash_c0_r2, "Mô hình toàn cục Round 2 phải là sự dung hòa của cả 2 client"

    print("\n[+] XÁC NHẬN HOÀN TOÀN: Vòng lặp phản hồi 2 chiều Client <-> Server hoạt động chính xác 100%!")


if __name__ == "__main__":
    test_bidirectional_server_client_feedback_loop()
    print("Test passed successfully!")

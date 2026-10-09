"""
scripts.run_network_clients: Tiện ích khởi chạy hàng loạt client Flower độc lập kết nối đến Server từ xa.
Hỗ trợ cả Windows và Linux.

Cách dùng:
    python scripts/run_network_clients.py --server 13.250.xx.xx:8080 --num-clients 5
    python scripts/run_network_clients.py --server 127.0.0.1:8080 --num-clients 5 --smoke
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Multi-Client Network Launcher")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Địa chỉ IP:Port của Flower Server trên AWS/VPS")
    parser.add_argument("--num-clients", type=int, default=5, help="Số lượng client cần khởi chạy (mặc định: 5)")
    parser.add_argument("--partition-dir", type=str, default="data/partitions_stage2_scratch_v3/label100")
    parser.add_argument("--smoke", action="store_true", help="Chạy chế độ thử nghiệm nhanh với tập dữ liệu nhỏ")
    args = parser.parse_args()

    print(f"[*] =================================================================")
    print(f"[*] KHOI CHAY {args.num_clients} CLIENT FLOWER KET NOI DEN SERVER AWS:")
    print(f"[*] Server Target : {args.server}")
    print(f"[*] Partition Dir : {args.partition_dir}")
    print(f"[*] Mode Smoke    : {args.smoke}")
    print(f"[*] =================================================================")

    processes: List[subprocess.Popen] = []

    # Ưu tiên sử dụng python trong .venv nếu có
    venv_py = Path(".venv/Scripts/python.exe")
    if not venv_py.exists():
        venv_py = Path(".venv/bin/python")
    py_exec = str(venv_py.resolve()) if venv_py.exists() else sys.executable

    print(f"[*] Python Executable: {py_exec}")

    try:
        for cid in range(args.num_clients):
            cmd = [
                py_exec,
                "-m",
                "src.federated.client_app",
                "--client-id",
                str(cid),
                "--server",
                args.server,
                "--partition-dir",
                args.partition_dir,
            ]
            if args.smoke:
                cmd.append("--smoke")

            print(f"[+] Dang bat Client {cid}...")
            p = subprocess.Popen(cmd)
            processes.append(p)
            time.sleep(1.0)  # Giãn cách 1s tránh xung đột khởi động cùng lúc

        print(f"\n[*] Tat ca {len(processes)} client da chay. Dang giam sat tien trinh (Nhan Ctrl+C de dung)...")

        # Cho tat ca tien trinh hoan thanh
        for p in processes:
            p.wait()

        print("\n[+] Tat ca Client da hoan thanh phien huan luyen voi Server thanh cong!")

    except KeyboardInterrupt:
        print("\n[!] Nhan lenh dung tu nguoi dung, dang ngat cac client...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("[+] Da dong toan bo tien trinh client.")


if __name__ == "__main__":
    main()

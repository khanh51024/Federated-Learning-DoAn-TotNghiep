"""Script tự động đóng gói, cập nhật dataset và kích hoạt huấn luyện trên Kaggle.

Quy trình:
1. Đóng gói mã nguồn sạch vào output/kaggle-matched-v5-deploy/20260919/kaggle_dataset
2. Cập nhật version mới cho Kaggle Dataset: nguynhongnamkhnh/stage2-matched-fedavg-v5
3. Chuẩn bị notebook với ACTION = 'run', toàn bộ 8 kịch bản cho SEEDS = [42]
4. Đẩy kernel lên Kaggle: nguynhongnamkhnh/fedavg-stage2-matched-v5
5. Kiểm tra trạng thái kernel trên Kaggle
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path("D:/university/do-an-tot-nghiep")
PACKAGE_DIR = WORKSPACE / "train-gd-2" / "gd2_federated_learning"
DEPLOY_DIR = WORKSPACE / "train-gd-2" / "output" / "kaggle-matched-v5-deploy" / "20260919"
DATASET_DIR = DEPLOY_DIR / "kaggle_dataset"
KERNEL_DIR = DEPLOY_DIR / "train_kernel"
PYTHON_EXE = PACKAGE_DIR / ".venv" / "Scripts" / "python.exe"


def step1_prepare_dataset_staging():
    print("\n=== BƯỚC 1: Chuẩn bị staging dataset ===")
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    folders = [
        "stage2_matched",
        "stage1_compat",
        "src",
        "fl_training",
        "data/partitions_stage2_matched_v5",
        "scripts",
    ]

    for rel_folder in folders:
        src = PACKAGE_DIR / rel_folder
        dst = DATASET_DIR / rel_folder
        if not src.is_dir():
            print(f"Bỏ qua (không tồn tại): {src}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if item.is_file():
                if "__pycache__" in item.parts or item.suffix in (".pyc", ".pyo"):
                    continue
                rel_path = item.relative_to(src)
                target_file = dst / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_file)

    standalone_files = [
        "requirements-stage1.txt",
        "requirements-stage2-matched.txt",
        "pyproject.toml",
    ]
    for filename in standalone_files:
        src = PACKAGE_DIR / filename
        dst = DATASET_DIR / filename
        if src.is_file():
            shutil.copy2(src, dst)

    # dataset-metadata.json
    dataset_metadata = {
        "title": "Stage2 Matched FedAvg v5",
        "id": "nguynhongnamkhnh/stage2-matched-fedavg-v5",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(DATASET_DIR / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    print(f"Staging dataset đã sẵn sàng tại: {DATASET_DIR}")
    file_count = len(list(DATASET_DIR.rglob("*")))
    print(f"Tổng số tệp/thư mục: {file_count}")


def step2_upload_dataset_version():
    print("\n=== BƯỚC 2: Cập nhật version dataset lên Kaggle ===")
    cmd = [
        str(PYTHON_EXE), "-m", "kaggle", "datasets", "version",
        "-p", str(DATASET_DIR),
        "-m", "Update Stage2 Matched v5 with baseline support and robust paths",
        "--dir-mode", "zip"
    ]
    print("Thực thi lệnh:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    assert res.returncode == 0, f"Lỗi upload dataset: {res.stderr}"
    print("Cập nhật dataset thành công!")


def step3_prepare_kernel():
    print("\n=== BƯỚC 3: Chuẩn bị kernel với ACTION='run' và toàn bộ 8 kịch bản cho Seed 42 ===")
    if KERNEL_DIR.exists():
        shutil.rmtree(KERNEL_DIR)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    nb_source = PACKAGE_DIR / "kaggle_stage2_matched_v5.ipynb"
    nb = json.loads(nb_source.read_text(encoding="utf-8"))

    # Cập nhật cell setup-paths
    for cell in nb["cells"]:
        if cell.get("id") == "setup-paths":
            new_source = []
            for line in cell["source"]:
                if line.startswith("ACTION = "):
                    new_source.append("ACTION = 'run'  # preflight / smoke / run / baseline / collect\n")
                elif line.startswith("CONDITIONS = "):
                    new_source.append("CONDITIONS = ['label100', 'label1', 'label01', 'quantity100', 'quantity01', 'label_quantity01', 'feature100', 'feature01']\n")
                elif line.startswith("SEEDS = "):
                    new_source.append("SEEDS = [42]  # Chạy toàn bộ kịch bản cho seed 42\n")
                elif line.startswith("SESSION_MINUTES = "):
                    new_source.append("SESSION_MINUTES = 420  # Bounded session limit (7 hours)\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

    train_nb_path = KERNEL_DIR / "kaggle_stage2_matched_v5.ipynb"
    train_nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook đã được chuẩn bị tại: {train_nb_path}")

    # kernel-metadata.json
    train_meta = {
        "id": "nguynhongnamkhnh/fedavg-stage2-matched-v5",
        "title": "FedAvg Stage2 Matched v5",
        "code_file": "kaggle_stage2_matched_v5.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "nguynhongnamkhnh/plantvillage-raw-images",
            "nguynhongnamkhnh/stage2-matched-fedavg-v5"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    train_meta_path = KERNEL_DIR / "kernel-metadata.json"
    train_meta_path.write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
    print(f"Kernel metadata đã sẵn sàng tại: {train_meta_path}")


def step4_push_kernel():
    print("\n=== BƯỚC 4: Đẩy kernel lên Kaggle để bắt đầu huấn luyện GPU ===")
    cmd = [
        str(PYTHON_EXE), "-m", "kaggle", "kernels", "push",
        "-p", str(KERNEL_DIR)
    ]
    print("Thực thi lệnh:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    assert res.returncode == 0, f"Lỗi push kernel: {res.stderr}"
    print("Kernel đã được đẩy lên Kaggle thành công!")


def step5_check_status():
    print("\n=== BƯỚC 5: Kiểm tra trạng thái kernel trên Kaggle ===")
    time.sleep(5)
    cmd = [
        str(PYTHON_EXE), "-m", "kaggle", "kernels", "status",
        "nguynhongnamkhnh/fedavg-stage2-matched-v5"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Trạng thái kernel:")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)


def main():
    step1_prepare_dataset_staging()
    step2_upload_dataset_version()
    # Chờ 15s để Kaggle xử lý dataset version mới
    print("Đang chờ 15 giây để Kaggle hoàn tất xử lý dataset...")
    time.sleep(15)
    step3_prepare_kernel()
    step4_push_kernel()
    step5_check_status()
    print("\n✅ TOÀN BỘ QUÁ TRÌNH TẢI LÊN VÀ BẮT ĐẦU HUẤN LUYỆN TRÊN KAGGLE ĐÃ HOÀN TẤT!")


if __name__ == "__main__":
    main()

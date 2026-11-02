#!/usr/bin/env bash
# Cài môi trường huấn luyện trên máy GPU Linux (chạy một lần). Dùng: bash scripts/remote/setup_env.sh [thư_mục_làm_việc]
# Thư mục làm việc mặc định ~/cropfed: code/ (giải nén train_code.tar), data/ (giải nén train_data_v4.tar), venv/, runs/.
set -euo pipefail
WORK="${1:-$HOME/cropfed}"
CUDA_TAG="${CUDA_TAG:-cu128}"   # bánh xe PyTorch dựng cho CUDA 12.8 chạy được trên driver mới hơn (driver CUDA 13.x tương thích ngược)
mkdir -p "$WORK"
cd "$WORK"
# Ảnh "CUDA ... Ubuntu" của nơi thuê không có sẵn python3/tmux (gặp ở máy thuê đầu tiên): tự cài khi chạy bằng root
if ! command -v python3 >/dev/null || ! python3 -m venv --help >/dev/null 2>&1 || ! command -v tmux >/dev/null; then
  if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv python3-pip tmux
  fi
fi
command -v python3 >/dev/null || { echo "thiếu python3"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install -r code/requirements-train.txt
# FLOWER=1: thêm Flower và Ray cho `train_v4.py --runtime flower`. flwr 1.36 cần Python >= 3.11; ảnh máy thuê từng gặp là 3.10, khi đó tạo venv bằng python3.11 trước khi chạy script này.
if [ "${FLOWER:-0}" = "1" ]; then
  python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Flower 1.36 cần Python >= 3.11, venv đang là " + sys.version.split()[0])'
  python -m pip install -r code/requirements-flower.txt
fi
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "PyTorch không thấy GPU: thử CUDA_TAG khác (cu126, cu130)"
PY

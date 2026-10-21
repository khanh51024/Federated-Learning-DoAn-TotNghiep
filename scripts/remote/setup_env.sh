#!/usr/bin/env bash
# Cài môi trường huấn luyện trên máy GPU Linux (chạy một lần). Dùng: bash scripts/remote/setup_env.sh [thư_mục_làm_việc]
# Thư mục làm việc mặc định ~/cropfed: code/ (giải nén train_code.tar), data/ (giải nén train_data_v4.tar), venv/, runs/.
set -euo pipefail
WORK="${1:-$HOME/cropfed}"
CUDA_TAG="${CUDA_TAG:-cu128}"   # bánh xe PyTorch dựng cho CUDA 12.8 chạy được trên driver mới hơn (driver CUDA 13.x tương thích ngược)
mkdir -p "$WORK"
cd "$WORK"
command -v python3 >/dev/null || { echo "thiếu python3"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install -r code/requirements-train.txt
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "PyTorch không thấy GPU: thử CUDA_TAG khác (cu126, cu130)"
PY

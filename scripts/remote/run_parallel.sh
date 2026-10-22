#!/usr/bin/env bash
# Chạy song song nhiều hạt giống trên một máy GPU: mỗi hạt giống một phiên tmux chạy scripts/remote/run_all.sh (7 run nối tiếp).
# Vì sao: mô hình nhẹ nên một run không dùng hết GPU; nghẽn là CPU (giải mã, tăng cường ảnh). Máy nhiều luồng CPU chạy song song được.
# Dùng: RELEASE_SHA=<sha> RECIPE="..." bash scripts/remote/run_parallel.sh [thư_mục_làm_việc]
# Biến tùy chọn: SEEDS (mặc định "42 123 2026"), WORKERS (mỗi phiên; mặc định nproc / số hạt giống - 1, tối thiểu 4),
#   CODE_DIR, OUT_DIR, BUDGET như run_all.sh. Theo dõi: tmux ls; tail <làm_việc>/run_all_s<hạt giống>.log
set -uo pipefail
WORK="${1:-$HOME/cropfed}"
: "${RELEASE_SHA:?cần RELEASE_SHA = SHA-256 của release_manifest.json (xem bundle.json)}"
SEEDS="${SEEDS:-42 123 2026}"
CODE="${CODE_DIR:-$WORK/code}"
read -r -a seed_list <<< "$SEEDS"
n=${#seed_list[@]}
per=$(( $(nproc) / n - 1 ))
WORKERS="${WORKERS:-$(( per < 4 ? 4 : per ))}"
command -v tmux >/dev/null || { echo "thiếu tmux (chạy scripts/remote/setup_env.sh)"; exit 1; }
for seed in "${seed_list[@]}"; do
  session="train_s$seed"
  if tmux has-session -t "$session" 2>/dev/null; then echo "phiên $session đang chạy, bỏ qua"; continue; fi
  tmux new-session -d -s "$session" \
    "env RELEASE_SHA='$RELEASE_SHA' RECIPE='${RECIPE:-}' SEEDS='$seed' WORKERS='$WORKERS' CODE_DIR='$CODE' OUT_DIR='${OUT_DIR:-}' BUDGET='${BUDGET:-}' \
     bash '$CODE/scripts/remote/run_all.sh' '$WORK' >> '$WORK/run_all_s$seed.log' 2>&1"
  echo "$(date '+%F %T') mở $session (worker $WORKERS)"
done
tmux ls

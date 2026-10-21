#!/usr/bin/env bash
# Chạy toàn bộ đợt huấn luyện v4 trên máy GPU: 3 hạt giống x (Centralized, FedAvg, 5 Local-only), lần lượt từng run.
# Dùng: RELEASE_SHA=<sha> bash scripts/remote/run_all.sh [thư_mục_làm_việc]
# Chạy lại lệnh này sau khi bị ngắt: run đã xong (có DONE) được bỏ qua, run dở chạy tiếp từ checkpoint_last.pt.
set -uo pipefail
WORK="${1:-$HOME/cropfed}"
: "${RELEASE_SHA:?cần RELEASE_SHA = SHA-256 của release_manifest.json (xem bundle.json)}"
SEEDS="${SEEDS:-42 123 2026}"
WORKERS="${WORKERS:-10}"
cd "$WORK/code"
. "$WORK/venv/bin/activate"
mkdir -p "$WORK/runs/v4/logs"
COMMON=(--dataset-root "$WORK/data/dataset_sources" --expected-release-sha "$RELEASE_SHA" --out "$WORK/runs/v4" --workers "$WORKERS")
status=0
for seed in $SEEDS; do
  for spec in "centralized" "fedavg" "local 0" "local 1" "local 2" "local 3" "local 4"; do
    set -- $spec
    method="$1"; extra=()
    name="${method}_s${seed}"
    if [ "$method" = "local" ]; then extra=(--client "$2"); name="local_c${2}_s${seed}"; fi
    echo "$(date '+%F %T') bắt đầu $name"
    if ! python scripts/train_v4.py --method "$method" --seed "$seed" "${extra[@]}" "${COMMON[@]}" >> "$WORK/runs/v4/logs/$name.log" 2>&1; then
      echo "$(date '+%F %T') LỖI $name (xem runs/v4/logs/$name.log)"; status=1
    fi
  done
done
echo "$(date '+%F %T') hết đợt, mã thoát $status"
touch "$WORK/runs/v4/ALL_FINISHED"
exit $status

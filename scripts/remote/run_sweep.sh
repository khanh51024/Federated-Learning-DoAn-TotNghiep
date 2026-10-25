#!/usr/bin/env bash
# Đợt GĐ2/GĐ3 trên máy GPU: cho mỗi hạt giống, mỗi cách chia client (SCHEMES) và mỗi thuật toán (ALGOS) chạy một run học liên kết;
# LOCAL=1 chạy thêm 5 run Local-only cho mỗi cách chia. Centralized không phụ thuộc cách chia nên dùng lại run của đợt v4b.
# Dùng: RELEASE_SHA=<sha> SCHEMES="..." ALGOS="..." bash scripts/remote/run_sweep.sh [thư_mục_làm_việc]
#   SCHEMES  mặc định "iid label_alpha_1_0 label_alpha_0_5 label_alpha_10_0" (xem scripts/build_extra_partitions.py; label_alpha_0_1 đã có từ v4b)
#   ALGOS    mặc định "fedavg"; mỗi phần tử là `tên[:tham_số]`: fedprox:<μ>, moon:<μ>, fedavgm:<η_s>, còn lại không tham số
#            ví dụ "fedavg fedprox:0.01 fedprox:0.1 fedbn scaffold fedavgm:0.1 moon:1.0"
#   LOCAL    1 = thêm Local-only mỗi cách chia (mặc định 0)
#   SEEDS, WORKERS, CODE_DIR, OUT_DIR (mặc định <làm_việc>/runs/sweep), RECIPE (mặc định công thức v4b), BUDGET như run_all.sh.
# Tên run: fedavg_s<seed>_<cách chia>[_<thuật toán>[_<tham số>]], local_c<k>_s<seed>_<cách chia>. Chạy lại lệnh sau khi bị ngắt: run đã xong được bỏ qua.
set -uo pipefail
WORK="${1:-$HOME/cropfed}"
: "${RELEASE_SHA:?cần RELEASE_SHA = SHA-256 của release_manifest.json (xem bundle.json)}"
SEEDS="${SEEDS:-42 123 2026}"
SCHEMES="${SCHEMES:-iid label_alpha_1_0 label_alpha_0_5 label_alpha_10_0}"
ALGOS="${ALGOS:-fedavg}"
LOCAL="${LOCAL:-0}"
WORKERS="${WORKERS:-10}"
CODE="${CODE_DIR:-$WORK/code}"
OUT="${OUT_DIR:-$WORK/runs/sweep}"
RECIPE="${RECIPE:---aug gpu_v2 --channels-last --batch-size 128 --lr 0.02 --amp bf16}"
cd "$CODE"
. "$WORK/venv/bin/activate"
mkdir -p "$OUT/logs"
COMMON=(--dataset-root "$WORK/data/dataset_sources" --expected-release-sha "$RELEASE_SHA" --out "$OUT" --workers "$WORKERS")
if [ -n "${BUDGET:-}" ]; then COMMON+=(--budget "$BUDGET"); fi
read -r -a RECIPE_ARGS <<< "$RECIPE"
echo "$(date '+%F %T') công thức: $RECIPE; cách chia: $SCHEMES; thuật toán: $ALGOS; local: $LOCAL; ra: $OUT"
status=0
run() {   # run <tên log> <đối số train_v4...>
  local name="$1"; shift
  echo "$(date '+%F %T') bắt đầu $name"
  if ! python scripts/train_v4.py "$@" "${COMMON[@]}" "${RECIPE_ARGS[@]}" >> "$OUT/logs/$name.log" 2>&1; then
    echo "$(date '+%F %T') LỖI $name (xem $OUT/logs/$name.log)"; status=1
  fi
}
for seed in $SEEDS; do
  for scheme in $SCHEMES; do
    for spec in $ALGOS; do
      algo="${spec%%:*}"; param=""; [ "$spec" != "$algo" ] && param="${spec#*:}"
      extra=()
      case "$algo" in
        fedprox) [ -n "$param" ] && extra=(--prox-mu "$param");;
        moon)    [ -n "$param" ] && extra=(--moon-mu "$param");;
        fedavgm) [ -n "$param" ] && extra=(--server-lr "$param");;
      esac
      tag="$scheme"; [ "$algo" != "fedavg" ] && tag="${scheme}_${algo}${param:+_$param}"
      run "fedavg_s${seed}_${tag}" --method fedavg --algo "$algo" --scheme "$scheme" --seed "$seed" --tag "$tag" "${extra[@]}"
    done
    if [ "$LOCAL" = "1" ]; then
      for c in 0 1 2 3 4; do
        run "local_c${c}_s${seed}_${scheme}" --method local --client "$c" --scheme "$scheme" --seed "$seed" --tag "$scheme"
      done
    fi
  done
done
echo "$(date '+%F %T') hết đợt, mã thoát $status"
touch "$OUT/FINISHED_seeds_${SEEDS// /_}"
exit $status

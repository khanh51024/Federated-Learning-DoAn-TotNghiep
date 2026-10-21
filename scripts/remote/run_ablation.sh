#!/usr/bin/env bash
# Ablation ngắn để chọn công thức trước đợt chính: Centralized, một hạt giống, mỗi biến thể một run; so trên val (history.jsonl, final.json).
# Không dùng tập thử hay PlantWild để chọn. So bằng: python scripts/summarize_ablation.py <OUT_DIR>
# Dùng: RELEASE_SHA=<sha> VARIANTS="A40 A B C D" bash scripts/remote/run_ablation.sh [thư_mục_làm_việc]
# Biến thể mới không có sẵn: đặt FLAGS_<tên>="..." (ví dụ FLAGS_E="--aug gpu_v2 --channels-last --train-size 192 ...").
set -uo pipefail
WORK="${1:-$HOME/cropfed}"
: "${RELEASE_SHA:?cần RELEASE_SHA = SHA-256 của release_manifest.json (xem bundle.json)}"
CODE="${CODE_DIR:-$WORK/code}"
OUT="${OUT_DIR:-$WORK/runs/ablation}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-14}"
VARIANTS="${VARIANTS:-A40 A B C D}"
GPU="--aug gpu_v2 --channels-last"
declare -A FLAGS=(
  [A40]="$GPU --budget 40"                                                     # cùng ngân sách với run mốc: kiểm đường GPU không làm giảm độ chính xác
  [A]="$GPU --budget 20"
  [B]="$GPU --budget 20 --sampling balanced --class-power 1"
  [C]="$GPU --budget 20 --sampling balanced --class-power 0.5"
  [D]="$GPU --budget 20 --sampling balanced --class-power 0.5 --logit-adjust 1"
  [A128]="$GPU --budget 20 --batch-size 128 --lr 0.02"                         # batch gấp đôi, lr theo quy tắc tuyến tính: ít bước hơn, nhanh hơn
)
cd "$CODE"
. "$WORK/venv/bin/activate"
mkdir -p "$OUT/logs"
status=0
for v in $VARIANTS; do
  override="FLAGS_$v"
  flags="${!override:-${FLAGS[$v]:-}}"
  if [ -z "$flags" ]; then echo "$(date '+%F %T') không biết biến thể $v"; status=1; continue; fi
  read -r -a args <<< "$flags"
  echo "$(date '+%F %T') bắt đầu $v: $flags"
  if ! python scripts/train_v4.py --method centralized --seed "$SEED" --tag "$v" --dataset-root "$WORK/data/dataset_sources" \
      --expected-release-sha "$RELEASE_SHA" --out "$OUT" --workers "$WORKERS" "${args[@]}" >> "$OUT/logs/$v.log" 2>&1; then
    echo "$(date '+%F %T') LỖI $v (xem $OUT/logs/$v.log)"; status=1
  fi
done
echo "$(date '+%F %T') hết ablation, mã thoát $status"
exit $status

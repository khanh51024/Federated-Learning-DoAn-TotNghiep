#!/usr/bin/env bash
# Verified 18-run batch. Run inside tmux; never uploads or evaluates PlantWild.
# Usage: bash scripts/remote/run_verification_batch.sh preflight|train [~/cropfed]
# train also requires network_speedtest_passed.json, written after the two-machine audit.
set -euo pipefail
MODE="${1:?preflight or train}"
WORK="${2:-$HOME/cropfed}"
CODE="$WORK/code"
OUT="$WORK/runs/v4_sweep"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export RELEASE_SHA=68e574519a1a9a53f4e71060d6d6ef70122ff946d16e41981b5930d3da92d51b
export WORKERS=8 BUDGET=40 ALGOS=fedavg LOCAL=0 ADAMW=0
export CODE_DIR="$CODE" OUT_DIR="$OUT"
export RECIPE="--aug gpu_v2 --channels-last --batch-size 128 --lr 0.02 --amp bf16"
cd "$CODE"
. "$WORK/venv/bin/activate"
mkdir -p "$WORK/tmp" "$OUT/logs" "$WORK/verification"

case "$MODE" in
  preflight)
    python -m pytest -q tests/test_eval_sweep_v4.py tests/test_flower_app.py \
      tests/test_fed_algorithms.py tests/test_fedsgd_equivalence.py tests/test_verify_distributed_run.py \
      > "$WORK/verification/tests.log" 2>&1
    command=(python scripts/train_v4.py --method fedavg --algo fedavg --runtime flower
      --scheme label_alpha_0_1 --seed 42 --device cuda
      --dataset-root "$WORK/data/dataset_sources" --expected-release-sha "$RELEASE_SHA"
      --aug gpu_v2 --channels-last --batch-size 128 --lr 0.02 --amp bf16 --workers 8
      --limit 3000 --budget 3 --out "$WORK/tmp/flower_preflight" --tag gpu_workers8)
    printf '%q ' "${command[@]}" > "$WORK/verification/preflight_command.txt"
    printf '\n' >> "$WORK/verification/preflight_command.txt"
    "${command[@]}" > "$WORK/verification/flower_preflight.log" 2>&1
    python scripts/verify_distributed_run.py --run "$WORK/tmp/flower_preflight/fedavg_s42_gpu_workers8" \
      --preflight --out "$WORK/verification/flower_preflight_audit.json"
    python scripts/verify_fedsgd_equivalence.py --device cuda --dataset-root "$WORK/data/dataset_sources" \
      --out "$WORK/verification/fedsgd_equivalence.json" > "$WORK/verification/fedsgd_equivalence.log" 2>&1
    python scripts/verify_distributed_run.py --run "$WORK/tmp/flower_preflight/fedavg_s42_gpu_workers8" \
      --preflight --equivalence "$WORK/verification/fedsgd_equivalence.json" --out "$WORK/preflight_passed.json"
    ;;
  train)
    python - "$WORK/preflight_passed.json" "$WORK/network_speedtest_passed.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    with open(path) as stream:
        report = json.load(stream)
    if report.get("passed") is not True:
        raise SystemExit(f"gate failed: {path}")
PY
    # A then B, a single Flower/Ray runtime at a time on the one GPU.
    export EXTRA="--runtime flower" TAG_SUFFIX=flower
    for scheme in label_alpha_0_1 iid; do
      for seed in 42 123 2026; do
        echo "$(date -u +%FT%TZ) start Flower $scheme seed=$seed"
        SEEDS="$seed" SCHEMES="$scheme" bash scripts/remote/run_sweep.sh "$WORK"
        python scripts/verify_distributed_run.py --run "$OUT/fedavg_s${seed}_${scheme}_flower" \
          --out "$WORK/verification/fedavg_s${seed}_${scheme}_flower.json"
      done
    done
    # C: three seeds in parallel, four schemes sequentially within each seed.
    export EXTRA="" TAG_SUFFIX=""
    export SCHEMES="label_alpha_100_0 label_alpha_10_0 label_alpha_1_0 label_alpha_0_5"
    pids=()
    for seed in 42 123 2026; do
      (
        SEEDS="$seed" bash scripts/remote/run_sweep.sh "$WORK"
        for scheme in $SCHEMES; do
          python scripts/verify_distributed_run.py --run "$OUT/fedavg_s${seed}_${scheme}" \
            --out "$WORK/verification/fedavg_s${seed}_${scheme}.json"
        done
      ) > "$WORK/verification/alpha_seed_${seed}.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for job in "${pids[@]}"; do wait "$job" || failed=1; done
    [ "$failed" -eq 0 ] || exit 1
    date -u +%FT%TZ > "$WORK/verification/BATCH_PASSED"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

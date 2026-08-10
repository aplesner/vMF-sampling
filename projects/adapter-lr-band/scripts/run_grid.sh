#!/usr/bin/env bash
# Two-worker driver for the vision-grid LR-band sweep on the 2x3090 node.
#
# ViT-S/VTAB fits comfortably on one 3090, so we run two arms in parallel,
# one per card (two independent workers beat 2x3090-PCIe DDP for this
# many-small-runs design — see 00-shared-constraints.md).
#
# Usage: bash run_grid.sh <dataset> <epochs> [ranks...]
# Example: bash run_grid.sh vtab 10 1 2 4 8 16 32
set -euo pipefail

DATASET=${1:?dataset}
EPOCHS=${2:?epochs}
shift 2
RANKS=${@:-"1 2 4 8 16 32"}
ARMS=(lora lora-scalar lora-diag delora delora-pc)
OUTDIR=results/$(date -u +%Y%m%dT%H%M%SZ)-${DATASET}
mkdir -p "$OUTDIR"

echo "grid: logspace(1e-5,1e-1,13) | ranks: $RANKS | seeds: 0 1 2 | $OUTDIR"

i=0
for ARM in "${ARMS[@]}"; do
  CARD=$((i % 2))   # worker i -> cuda:$CARD
  i=$((i + 1))
  for R in $RANKS; do
    CUDA_VISIBLE_DEVICES=$CARD python src/train.py \
      --arm "$ARM" --rank "$R" \
      --arch vit --dataset "$DATASET" --epochs "$EPOCHS" \
      --device cuda --seeds 0 1 2 \
      --out "$OUTDIR/sweeps.csv" \
      --steps-out "$OUTDIR/steps_${ARM}_r${R}.jsonl" \
      > "$OUTDIR/log_${ARM}_r${R}.txt" 2>&1 &
  done
  # cap in-flight jobs at 2 (one per card)
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 30; done
done
wait
echo "done -> $OUTDIR/sweeps.csv"

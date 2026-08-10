#!/usr/bin/env bash
# Project 1 Tier 0+1: full sweep suite on 2x RTX 3090 (one worker per card).
#SBATCH --job-name=p1-tier01
#SBATCH --partition=disco.med
#SBATCH --gres=gpu:geforce_rtx_3090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/itet-stor/aplesner/net_scratch/vmf-sampling/slurm/p1_tier01_%j.log

set -euo pipefail
ROOT=/itet-stor/aplesner/net_scratch/vmf-sampling/repo
cd "${ROOT}/projects/continual/src"
mkdir -p ../results/cluster

# node-local copy of CIFAR-10 (guarded, atomic; net_scratch is the source)
LOCAL=/scratch/aplesner/p1-continual/cifar10
if [ ! -f "${LOCAL}/cifar-10-batches-py/batches.meta" ]; then
  TMP="${LOCAL}.tmp.$$"
  rm -rf "${TMP}"; mkdir -p "${TMP}"
  cp -r ../data/cifar-10-batches-py "${TMP}/"
  mkdir -p "$(dirname "${LOCAL}")"
  mv "${TMP}" "${LOCAL}"
fi

uv sync --project "${ROOT}" --quiet
# cluster driver is CUDA 12.2: override the venv's torch with cu121 builds
uv pip install --python "${ROOT}/.venv/bin/python" --quiet \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

for tier in tier0 lars mlp-lowlr; do
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync --project "${ROOT}" python run_sweep.py \
      --tier "${tier}" --shard $i --nshards 2 --device cuda \
      --data-dir "${LOCAL}" \
      --out "../results/cluster/${tier}_shard$i.csv" \
      > "../results/cluster/${tier}_log$i.txt" 2>&1 &
  done
  wait
done
echo "ALL TIERS DONE"

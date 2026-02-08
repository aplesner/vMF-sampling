#!/bin/bash
#SBATCH --job-name=vmf_benchmark
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/vmf-sampling/benchmarks/benchmark_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu09
#SBATCH --time=6:00:00
#SBATCH --array=0-38%1

set -euo pipefail

DIMS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
BACKENDS=(numpy scipy torch)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
DIM_INDEX=$((TASK_ID % ${#DIMS[@]}))
BACKEND_INDEX=$((TASK_ID / ${#DIMS[@]}))

DIM="${DIMS[$DIM_INDEX]}"
BACKEND="${BACKENDS[$BACKEND_INDEX]}"
OUT_DIR="/itet-stor/${USER}/net_scratch/jobs/vmf-sampling/benchmarks"
OUT_DB="${OUT_DIR}/benchmark_${BACKEND}_dim${DIM}.db"

# Activate your environment here if needed, e.g.:
# source /path/to/venv/bin/activate

python /home/aplesner/code/vMF-sampling/benchmark.py \
  --dim "${DIM}" \
  --backends "${BACKEND}" \
  --seeds 0,1,2,3,4 \
  --kappa 500 \
  --db "${OUT_DB}"

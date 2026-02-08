#!/bin/bash
#SBATCH --job-name=vmf_benchmark
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/vmf-sampling/benchmarks/benchmark_%A_%a.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu09
#SBATCH --time=6:00:00
#SBATCH --array=0-26%4

set -euo pipefail

DIMS=(2 4 8 16 32 64 128 256 512)
BACKENDS=(numpy scipy torch)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
DIM_INDEX=$((TASK_ID % ${#DIMS[@]}))
BACKEND_INDEX=$((TASK_ID / ${#DIMS[@]}))

DIM="${DIMS[$DIM_INDEX]}"
BACKEND="${BACKENDS[$BACKEND_INDEX]}"
OUT_DIR="/itet-stor/${USER}/net_scratch/jobs/vmf-sampling/benchmarks"
OUT_DB="${OUT_DIR}/benchmark_shared.db"
OUT_CSV="${OUT_DIR}/benchmark_${BACKEND}_dim${DIM}.csv"

# Activate your environment here if needed, e.g.:
# source /path/to/venv/bin/activate

python /home/aplesner/code/vMF-sampling/benchmark.py \
  --dim "${DIM}" \
  --backends "${BACKEND}" \
  --seeds 0,1,2,3,4 \
  --kappa 50 \
  --db "${OUT_DB}" \
  --output "${OUT_CSV}"

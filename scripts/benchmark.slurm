#!/bin/bash
#SBATCH --job-name=vmf_benchmark
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/vmf-sampling/benchmarks/benchmark_%A_%a.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu09
#SBATCH --time=6:00:00
#SBATCH --array=0-38%12

set -euo pipefail

VENV_DIR="/scratch/$USER/venvs/vmf-sampling"
LOCK="/scratch/$USER/venvs/vmf-sampling.lock"

# Use uv cache in scratch (per node)
export UV_CACHE_DIR="/scratch/$USER/uv-cache-vmf-sampling"
mkdir -p "${VENV_DIR}"

# Compute requirements hash from uv.lock if present, else pyproject
if [ -f /home/aplesner/code/vMF-sampling/uv.lock ]; then
  REQ_HASH="$(sha256sum /home/aplesner/code/vMF-sampling/uv.lock | awk '{print $1}')"
else
  REQ_HASH="$(sha256sum /home/aplesner/code/vMF-sampling/pyproject.toml | awk '{print $1}')"
fi

(
  flock -n 9 || { echo "Waiting for venv lock..."; flock 9; }

  NEED_REBUILD=0
  if [ ! -d "${VENV_DIR}" ]; then
    NEED_REBUILD=1
  elif [ ! -f "${VENV_DIR}/.req_hash" ]; then
    NEED_REBUILD=1
  elif [ "$(cat "${VENV_DIR}/.req_hash")" != "${REQ_HASH}" ]; then
    NEED_REBUILD=1
  fi

  if [ "${NEED_REBUILD}" -eq 1 ]; then
    rm -rf "${VENV_DIR}"
    uv venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv pip install numpy scipy pandas tqdm
    uv pip install torch \
      --index-url https://download.pytorch.org/whl/cu122 \
      --extra-index-url https://pypi.org/simple
    echo "${REQ_HASH}" > "${VENV_DIR}/.req_hash"
  fi
) 9>"${LOCK}"

source "${VENV_DIR}/bin/activate"



DIMS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
BACKENDS=(numpy scipy torch)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
DIM_INDEX=$((TASK_ID % ${#DIMS[@]}))
BACKEND_INDEX=$((TASK_ID / ${#DIMS[@]}))

DIM="${DIMS[$DIM_INDEX]}"
BACKEND="${BACKENDS[$BACKEND_INDEX]}"
OUT_DIR="/itet-stor/${USER}/net_scratch/jobs/vmf-sampling/benchmarks"
OUT_CSV="${OUT_DIR}/benchmark_${BACKEND}_dim${DIM}.csv"


python3 /home/aplesner/code/vMF-sampling/benchmark.py \
  --dim "${DIM}" \
  --backends "${BACKEND}" \
  --seeds 0,1,2,3,4 \
  --kappa 50 \
  --output "${OUT_CSV}"

#!/usr/bin/env bash
# Shared body for CPU-only Slurm benchmark arrays.
set -euo pipefail

REMOTE_ROOT="${VMF_REMOTE_ROOT:-/itet-stor/${USER}/net_scratch/vmf-sampling}"
REPO_DIR="${VMF_REPO_DIR:-${REMOTE_ROOT}/repo}"
RUN_ID="${VMF_RUN_ID:?VMF_RUN_ID must be exported by scripts/cluster_submit.sh}"
OUT_DIR="${REMOTE_ROOT}/runs/${RUN_ID}"
VENV_DIR="${REMOTE_ROOT}/venvs/cpu"
LOCK_FILE="${REMOTE_ROOT}/venvs/cpu.lock"

mkdir -p "${OUT_DIR}" "${REMOTE_ROOT}/slurm" "$(dirname "${VENV_DIR}")"

# Keep dependency downloads and environments out of the small backed-up home.
export UV_CACHE_DIR="${REMOTE_ROOT}/uv-cache"
REQ_HASH="$(cat "${REPO_DIR}/pyproject.toml" "${REPO_DIR}/uv.lock" | sha256sum | awk '{print $1}')"
(
  flock 9
  if [ ! -x "${VENV_DIR}/bin/python" ] \
     || [ ! -f "${VENV_DIR}/.req_hash" ] \
     || [ "$(cat "${VENV_DIR}/.req_hash" 2>/dev/null || true)" != "${REQ_HASH}" ]; then
    rm -rf "${VENV_DIR}"
    uv venv --python 3.12 "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    uv pip install --requirement "${REPO_DIR}/pyproject.toml"
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    printf '%s\n' "${REQ_HASH}" > "${VENV_DIR}/.req_hash"
  fi
) 9>"${LOCK_FILE}"
source "${VENV_DIR}/bin/activate"

# Avoid nested BLAS/OpenMP oversubscription and make each task use its allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0

DIMS=(2 3 4 8 16 32 64 128 256 512 1024 2048 4096)
BACKENDS=(numpy scipy torch)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
DIM_INDEX=$((TASK_ID % ${#DIMS[@]}))
BACKEND_INDEX=$((TASK_ID / ${#DIMS[@]}))
DIM="${DIMS[$DIM_INDEX]}"
BACKEND="${BACKENDS[$BACKEND_INDEX]}"
OUT_CSV="${OUT_DIR}/benchmark_${BACKEND}_dim${DIM}.csv"
META="${OUT_DIR}/environment_${BACKEND}_dim${DIM}.txt"

{
  date --iso-8601=seconds
  echo "run_id=${RUN_ID}"
  echo "job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "host=$(hostname)"
  echo "allocated_cpus=${SLURM_CPUS_PER_TASK}"
  echo "allocated_gpus=${SLURM_JOB_GPUS:-none}"
  echo "repo=${REPO_DIR}"
  echo "python=$(python --version 2>&1)"
  python - <<'PY'
import numpy, scipy, torch
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"torch={torch.__version__}")
print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")
print(f"torch_num_threads={torch.get_num_threads()}")
print(f"torch_num_interop_threads={torch.get_num_interop_threads()}")
PY
  lscpu
} > "${META}"

rm -f "${OUT_CSV}"
cd "${REPO_DIR}"
python benchmark.py \
  --dim "${DIM}" \
  --backends "${BACKEND}" \
  --dtypes float64 \
  --seeds 0,1,2 \
  --target-time 2 \
  --target-call-time 1 \
  --mem-budget-gb 1.2 \
  --output "${OUT_CSV}"

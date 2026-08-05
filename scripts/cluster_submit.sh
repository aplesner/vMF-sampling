#!/usr/bin/env bash
# Sync and submit a reproducible CPU-only benchmark array.
set -euo pipefail

HOST="${VMF_CLUSTER_HOST:-tik42x}"
REMOTE_USER="${VMF_CLUSTER_USER:-aplesner}"
REMOTE_ROOT="${VMF_REMOTE_ROOT:-/itet-stor/${REMOTE_USER}/net_scratch/vmf-sampling}"
REMOTE_REPO="${REMOTE_ROOT}/repo"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${VMF_BENCH_TARGET:-arton10}"
DO_SYNC=1

usage() {
  echo "Usage: $0 [--target arton10|tikgpu10] [--no-sync]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      TARGET="$2"
      shift 2
      ;;
    --target=*)
      TARGET="${1#*=}"
      shift
      ;;
    --no-sync)
      DO_SYNC=0
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "${TARGET}" in
  arton10) SLURM_SCRIPT="scripts/benchmark.slurm" ;;
  tikgpu10) SLURM_SCRIPT="scripts/benchmark_tikgpu10.slurm" ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    usage
    exit 2
    ;;
esac

RUN_ID="${VMF_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${TARGET}}"
if [[ "${DO_SYNC}" -eq 1 ]]; then
  "${ROOT}/scripts/cluster_sync.sh"
fi

JOB_ID="$(ssh "${HOST}" "mkdir -p '${REMOTE_ROOT}/slurm' '${REMOTE_ROOT}/runs/${RUN_ID}' && cd '${REMOTE_REPO}' && VMF_RUN_ID='${RUN_ID}' VMF_REMOTE_ROOT='${REMOTE_ROOT}' VMF_REPO_DIR='${REMOTE_REPO}' sbatch --parsable '${SLURM_SCRIPT}'")"
printf 'TARGET=%s\nRUN_ID=%s\nJOB_ID=%s\nREMOTE_RESULTS=%s:%s/runs/%s\n' \
  "${TARGET}" "${RUN_ID}" "${JOB_ID}" "${HOST}" "${REMOTE_ROOT}" "${RUN_ID}"

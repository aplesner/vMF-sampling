#!/usr/bin/env bash
# Fetch one benchmark run into measurements/runs/<run-id>.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ID" >&2
  exit 2
fi
HOST="${VMF_CLUSTER_HOST:-tik42x}"
REMOTE_USER="${VMF_CLUSTER_USER:-aplesner}"
REMOTE_ROOT="${VMF_REMOTE_ROOT:-/itet-stor/${REMOTE_USER}/net_scratch/vmf-sampling}"
RUN_ID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="${ROOT}/measurements/runs/${RUN_ID}"

mkdir -p "${LOCAL_DIR}"
rsync -az --delete "${HOST}:${REMOTE_ROOT}/runs/${RUN_ID}/" "${LOCAL_DIR}/"
printf 'Fetched %s:%s/runs/%s -> %s\n' "${HOST}" "${REMOTE_ROOT}" "${RUN_ID}" "${LOCAL_DIR}"

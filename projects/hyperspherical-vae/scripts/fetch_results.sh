#!/usr/bin/env bash
# Fetch T1/T2 result rows + logs from the cluster into local results/.
set -euo pipefail
HOST="${VMF_CLUSTER_HOST:-tik42x}"
REMOTE=/itet-stor/aplesner/net_scratch/vmf-sampling/repo/results
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results"
mkdir -p "$LOCAL"
rsync -az --include='*/' --include='*.json' --include='*.jsonl' --exclude='*' \
  "${HOST}:${REMOTE}/" "${LOCAL}/cluster/"
echo "fetched -> ${LOCAL}/cluster"
find "${LOCAL}/cluster" -name '*.json' | grep -v jsonl | wc -l

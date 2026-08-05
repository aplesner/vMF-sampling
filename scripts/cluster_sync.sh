#!/usr/bin/env bash
# Sync the working tree to net_scratch on tik42x without using backed-up home.
set -euo pipefail

HOST="${VMF_CLUSTER_HOST:-tik42x}"
REMOTE_USER="${VMF_CLUSTER_USER:-aplesner}"
REMOTE_ROOT="${VMF_REMOTE_ROOT:-/itet-stor/${REMOTE_USER}/net_scratch/vmf-sampling}"
REMOTE_REPO="${REMOTE_ROOT}/repo"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_ARGS=(-az --delete)
if [[ "${1:-}" == "--dry-run" ]]; then
  RSYNC_ARGS+=(--dry-run)
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

ssh "${HOST}" "mkdir -p '${REMOTE_REPO}' '${REMOTE_ROOT}/slurm' '${REMOTE_ROOT}/runs'"
rsync "${RSYNC_ARGS[@]}" \
  --exclude-from="${ROOT}/.ignore_for_code_sync" \
  --exclude='.venv/' \
  --exclude='measurements/' \
  --exclude='docs/assets/' \
  --exclude='.DS_Store' \
  "${ROOT}/" "${HOST}:${REMOTE_REPO}/"
printf 'Synced %s -> %s:%s\n' "${ROOT}" "${HOST}" "${REMOTE_REPO}"

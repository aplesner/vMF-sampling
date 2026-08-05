#!/usr/bin/env bash
# Show active and recent vMF Slurm jobs on tik42x.
set -euo pipefail
HOST="${VMF_CLUSTER_HOST:-tik42x}"
JOB_ID="${1:-}"

if [[ -n "${JOB_ID}" ]]; then
  ssh "${HOST}" "squeue -j '${JOB_ID}' -o '%.18i %.12P %.24j %.2t %.10M %.10l %.4C %R'; echo; sacct -j '${JOB_ID}' --starttime now-7days -o JobID,State,Elapsed,MaxRSS,ExitCode,NodeList -X"
else
  ssh "${HOST}" "squeue -u \$USER -o '%.18i %.12P %.24j %.2t %.10M %.10l %.4C %R'; echo; sacct -u \$USER --name vmf-cpu-benchmark --starttime now-7days -o JobID,State,Elapsed,ExitCode,NodeList -X | tail -25"
fi

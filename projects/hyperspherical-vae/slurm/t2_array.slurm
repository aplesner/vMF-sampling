#!/bin/bash
#SBATCH --job-name=p4-t2
#SBATCH --partition=disco.all.phd
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/itet-stor/aplesner/net_scratch/vmf-sampling/slurm/p4-t2_%A_%a.log
# Two 3090s as independent workers: cap the array at 2 concurrent tasks.
# Submit with:  sbatch --array=0-N%2 projects/hyperspherical-vae/slurm/t2_array.slurm

set -euo pipefail

REPO=/itet-stor/aplesner/net_scratch/vmf-sampling/repo
TASKS_FILE=${TASKS_FILE:-$REPO/projects/hyperspherical-vae/slurm/t2_tasks.txt}
NODE_SCRATCH=/scratch/aplesner/p4
mkdir -p "$NODE_SCRATCH"

# Node-local data (guard: no re-transfer across jobs on the same node).
export VMF_DATA_ROOT="$NODE_SCRATCH/data"

cd "$REPO"
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_FILE")
echo "task $SLURM_ARRAY_TASK_ID on $(hostname) gpu $CUDA_VISIBLE_DEVICES: $CMD"
uv run python projects/hyperspherical-vae/scripts/run_t2.py $CMD

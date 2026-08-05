#!/bin/bash
#SBATCH --job-name=vmf-cpu-tikgpu10
#SBATCH --output=/itet-stor/%u/net_scratch/vmf-sampling/slurm/benchmark_%A_%a.log
#SBATCH --partition=disco.all.phd
#SBATCH --nodelist=tikgpu10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-38%1

# CPU-only by design: do not add --gres or use a CUDA PyTorch wheel.
set -euo pipefail
REPO_DIR="${VMF_REPO_DIR:-/itet-stor/${USER}/net_scratch/vmf-sampling/repo}"
exec bash "${REPO_DIR}/scripts/benchmark_task.sh"

#!/bin/bash
#SBATCH --job-name=vmf_table2
#SBATCH --output=/itet-stor/%u/net_scratch/jobs/vmf-sampling/table2/table2_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --nodelist=tikgpu06,tikgpu07,tikgpu09
#SBATCH --time=2:00:00

set -euo pipefail

OUT_DIR="/itet-stor/${USER}/net_scratch/jobs/vmf-sampling/table2"
SAMPLES_FILE="${OUT_DIR}/table2_samples.csv"
ERRORS_FILE="${OUT_DIR}/table2_errors.csv"
SUMMARY_FILE="${OUT_DIR}/table2_errors_summary.csv"

# Activate your environment here if needed, e.g.:
# source /path/to/venv/bin/activate

python /home/aplesner/code/vMF-sampling/table2_sample.py \
  --output "${SAMPLES_FILE}" \
  --num-samples 10000 \
  --seeds 0,1,2,3,4

python /home/aplesner/code/vMF-sampling/table2_estimate.py \
  --input "${SAMPLES_FILE}" \
  --output "${ERRORS_FILE}" \
  --summary-output "${SUMMARY_FILE}"

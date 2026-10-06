#!/usr/bin/env bash
#SBATCH --job-name=thermogpu-cpu
#SBATCH --output=case-studies/thermogpu/raw-results/slurm/%x-%A_%a.out
#SBATCH --error=case-studies/thermogpu/raw-results/slurm/%x-%A_%a.err
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

set -euo pipefail
BATCHES=(1 10 100 1000 10000 100000 1000000)
BATCH="${BATCHES[$SLURM_ARRAY_TASK_ID]}"
: "${THERMOGPU_ROOT:?submit with THERMOGPU_ROOT exported}"
: "${CASE_STUDY_ROOT:?submit with CASE_STUDY_ROOT exported}"

exec python "$CASE_STUDY_ROOT/case-studies/thermogpu/scripts/collect_direct.py" \
    --thermogpu-root "$THERMOGPU_ROOT" --backend cpu --batch "$BATCH"

#!/usr/bin/env bash
#SBATCH --job-name=thermogpu-tdar
#SBATCH --output=case-studies/thermogpu/raw-results/slurm/%x-%A_%a.out
#SBATCH --error=case-studies/thermogpu/raw-results/slurm/%x-%A_%a.err
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

set -euo pipefail
BUDGETS=(16 32 64 128 256)
BUDGET="${BUDGETS[$SLURM_ARRAY_TASK_ID]}"
: "${CASE_STUDY_ROOT:?submit with CASE_STUDY_ROOT exported}"
: "${THERMOGPU_ROOT:?submit with THERMOGPU_ROOT exported}"
: "${TDAR_ROOT:?submit with TDAR_ROOT exported}"

exec python "$CASE_STUDY_ROOT/case-studies/thermogpu/scripts/build_surrogates.py" --budget "$BUDGET" \
    --tdar-root "$TDAR_ROOT" --thermogpu-root "$THERMOGPU_ROOT"

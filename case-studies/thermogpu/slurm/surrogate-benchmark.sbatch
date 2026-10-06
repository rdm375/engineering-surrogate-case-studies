#!/usr/bin/env bash
#SBATCH --job-name=thermogpu-surrogate
#SBATCH --output=case-studies/thermogpu/raw-results/slurm/%x-%A_%a.out
#SBATCH --error=case-studies/thermogpu/raw-results/slurm/%x-%A_%a.err
#SBATCH --array=0-34
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail
BUDGETS=(16 32 64 128 256)
BATCHES=(1 10 100 1000 10000 100000 1000000)
BUDGET_INDEX=$((SLURM_ARRAY_TASK_ID / ${#BATCHES[@]}))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % ${#BATCHES[@]}))
BUDGET="${BUDGETS[$BUDGET_INDEX]}"
BATCH="${BATCHES[$BATCH_INDEX]}"
: "${CASE_STUDY_ROOT:?submit with CASE_STUDY_ROOT exported}"
: "${CPWA_RELU_ROOT:?submit with CPWA_RELU_ROOT exported}"
: "${CPWA_RELU_BENCHMARK_COMMAND:?submit with CPWA_RELU_BENCHMARK_COMMAND exported}"

exec python "$CASE_STUDY_ROOT/case-studies/thermogpu/scripts/benchmark_surrogates.py" \
    --budget "$BUDGET" --batch "$BATCH" --cpwa-relu-root "$CPWA_RELU_ROOT"

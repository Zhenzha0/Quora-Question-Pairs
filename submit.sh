#!/bin/bash
# Wrapper to submit GPU or CPU Slurm jobs with automatic log naming.
#
# Usage: ./submit.sh <gpu|cpu> <script.py> [args...]
# Example:
#   ./submit.sh gpu embed_quora.py
#   ./submit.sh cpu catboost_thresh.py --threshold 0.5

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu|cpu> <script.py> [args...]"
    exit 1
fi

MODE="$1"
SCRIPT="$2"
shift 2
EXTRA_ARGS="$@"

# Derive a clean job name from the script filename (strip path and .py)
JOB_NAME="$(basename "$SCRIPT" .py)"

# Ensure the logs directory exists
mkdir -p logs

case "$MODE" in
    gpu)
        SLURM_SCRIPT="slurm_gpu.sh"
        ;;
    cpu)
        SLURM_SCRIPT="slurm_cpu.sh"
        ;;
    *)
        echo "Error: mode must be 'gpu' or 'cpu', got '$MODE'"
        exit 1
        ;;
esac

sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    "$SLURM_SCRIPT" "$SCRIPT" $EXTRA_ARGS \
    | {
        read -r SBATCH_OUTPUT
        JOB_ID="${SBATCH_OUTPUT%%;*}"

        echo "Submitted batch job ${JOB_ID}"
        echo "Watch logs with:"
        echo "  tail -f logs/${JOB_NAME}_${JOB_ID}.log logs/${JOB_NAME}_${JOB_ID}.err"
    }

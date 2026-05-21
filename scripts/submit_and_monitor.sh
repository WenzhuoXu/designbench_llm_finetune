#!/bin/bash
# Submit grpo_qwen3_posterior job and start the health monitor once it enters RUNNING.
#
# Usage:
#   bash scripts/submit_and_monitor.sh
#   CHECKPOINT=checkpoints/sft/qwen3_14b/final bash scripts/submit_and_monitor.sh

set -euo pipefail

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"

# ── Submit ──────────────────────────────────────────────────────────────────
SUBMIT_OUT=$(sbatch slurm/grpo_qwen3_posterior.sbatch)
echo "$SUBMIT_OUT"
JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')

if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job ID from sbatch output: $SUBMIT_OUT"
    exit 1
fi

RUN_NAME="qwen3_14b_grpo_posterior_${JOB_ID}"
echo "Job ID : $JOB_ID"
echo "Run name: $RUN_NAME"
echo "SLURM log: logs/slurm/qwen3_grpo_posterior_${JOB_ID}.out"

# ── Wait for RUNNING ────────────────────────────────────────────────────────
echo "Waiting for job $JOB_ID to enter RUNNING state..."
while true; do
    STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "GONE")
    if [[ "$STATE" == "GONE" || -z "$STATE" ]]; then
        echo "ERROR: job $JOB_ID not found in queue immediately after submission."
        exit 1
    fi
    if [[ "$STATE" == "RUNNING" ]]; then
        echo "Job $JOB_ID is RUNNING."
        break
    fi
    echo "  state=$STATE — checking again in 30s..."
    sleep 30
done

# ── Launch monitor ──────────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"
MONITOR_LOG="$PROJECT_DIR/logs/monitor_${JOB_ID}.log"

module load anaconda3/2024.10-1 2>/dev/null || true
nohup conda run -n my_env python scripts/monitor_grpo_job.py \
    "$JOB_ID" \
    --run-name "$RUN_NAME" \
    --wandb-project designbench-training \
    --poll-interval 60 \
    > "$MONITOR_LOG" 2>&1 &

MONITOR_PID=$!
echo "Monitor PID : $MONITOR_PID"
echo "Monitor log : $MONITOR_LOG"
echo ""
echo "To tail monitor: tail -f $MONITOR_LOG"
echo "To stop monitor: kill $MONITOR_PID"
echo "To cancel job  : scancel $JOB_ID"

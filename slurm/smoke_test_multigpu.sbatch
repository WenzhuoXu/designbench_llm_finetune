#!/bin/bash
#SBATCH --job-name=smoke_multigpu
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:l40s-48:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

# ============================================================
# Multi-GPU Smoke Test — 2×L40S-48
#
# Verifies distributed-training correctness of the warmstart SFT
# pipeline AFTER the rank-gating / packing / device_map fixes.
#
# Uses qwen25_0p5b (tiny model) on 2 GPUs to catch multi-rank bugs
# without needing full H100 compute. L40S supports bf16 natively,
# matching the H100 production code path.
#
# Expected behaviors to verify from the output logs:
#   - Exactly ONE "DesignBench SFT Training" header
#   - Exactly ONE "W&B run initialized" line (or wandb disabled)
#   - Exactly ONE wandb/run-*/ directory
#   - No "padding-free" ValueError
#   - logs/{run_name}/metrics.jsonl has clean non-interleaved rows
#   - Training reaches max_steps
#
# Submit:
#   sbatch slurm/smoke_test_multigpu.sbatch
# ============================================================

set -euo pipefail

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"
echo "============================================================"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# HF cache (shared with H100 jobs)
export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo
export CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN_NAME="smoke_multigpu_${SLURM_JOB_ID}"
mkdir -p logs/slurm logs/"$RUN_NAME"

echo ""
echo "Launching 2-GPU torchrun smoke test (qwen25_0p5b, 20 steps) ..."
echo ""

# Override:
#   - model: qwen25_0p5b (tiny, fits easily on L40S-48)
#   - model.attn_implementation: sdpa (L40S supports FA2 but sdpa is simpler for smoke)
#   - sft=pretrain: uses packing=false (Fix 2) + DeepSpeed-compatible grad clip
#   - data=sft_traces: DesignBench SFT JSONL
#   - parallelism=deepspeed_zero2: tests the actual distributed code path
#   - sft.max_steps=20: short run, catches any rank-gating bugs quickly
torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  --master_addr=localhost \
  scripts/train_sft.py \
    model=qwen25_0p5b \
    model.attn_implementation=sdpa \
    sft=pretrain \
    sft.max_steps=20 \
    sft.save_steps=999 \
    sft.per_device_train_batch_size=1 \
    sft.gradient_accumulation_steps=2 \
    sft.max_seq_len=2048 \
    sft.eval_steps=10 \
    data=sft_traces \
    parallelism=deepspeed_zero2 \
    run_name="$RUN_NAME" \
    logging.wandb_project=designbench-training \
    logging.wandb_tags="[smoke,multigpu]"

echo ""
echo "============================================================"
echo "Smoke test completed: $(date)"
echo ""
echo "Verification checks (run after job completes):"
echo "  1. Single wandb run:"
echo "     ls wandb/ | grep $RUN_NAME | wc -l    # should be 1"
echo "  2. Single training header:"
echo "     grep -c 'DesignBench SFT Training' logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "     # should be 1 (not 2)"
echo "  3. No padding-free error:"
echo "     grep 'padding-free' logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
echo "     # should be empty"
echo "  4. Metrics file has clean rows:"
echo "     head logs/$RUN_NAME/metrics.jsonl"
echo "============================================================"

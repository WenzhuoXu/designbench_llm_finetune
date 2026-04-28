#!/bin/bash
# ============================================================
# Pipeline Smoke Test — 1 GPU, ~10 min
#
# Uses Qwen2.5-0.5B-Instruct (same arch as 14B targets) to
# verify the full pipeline: data loading → model loading →
# SFT training → logging — without needing H100 or flash-attn.
#
# Target GPU:  GPU-shared partition (1× V100-32 or V100-16)
# Runtime:     < 15 minutes
#
# Submit:
#   sbatch slurm/smoke_test.sbatch
# Or to prefer L40S if a slot opens:
#   sbatch --gres=gpu:l40s-48:1 slurm/smoke_test.sbatch
# ============================================================

#SBATCH --job-name=db_smoke_test
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --qos=gpu

# Request 1 GPU — scheduler will pick the best available slot
# (V100-32 preferred over V100-16 for 32 GB headroom)
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:20:00

#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err
#SBATCH --mail-type=END,FAIL

# ─── Environment ──────────────────────────────────────────────
module load cuda/12.6.1
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
conda activate my_env

export HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models
export TRANSFORMERS_CACHE=/ocean/projects/mch250030p/wxu7/hf_models
export TOKENIZERS_PARALLELISM=false

# Single-GPU: no torchrun needed
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "DesignBench SFT Smoke Test"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(python --version)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "============================================================"

# ─── Pre-flight checks ────────────────────────────────────────
echo ""
echo "[1/5] Running unit tests (no GPU) ..."
cd /ocean/projects/mch250030p/wxu7/llm_finetune
python -m pytest tests/test_rewards.py tests/test_targets.py -q --tb=short 2>&1 | tail -5
echo "Unit tests done."

# ─── Data pipeline check ─────────────────────────────────────
echo ""
echo "[2/5] Verifying data pipeline (dry run) ..."
python scripts/prepare_data.py \
    --model qwen25_0p5b \
    --dry-run \
    --num-examples 10 \
    2>&1 | tail -20
echo "Data pipeline check done."

# ─── SFT smoke run ───────────────────────────────────────────
echo ""
echo "[3/5] Running SFT smoke test (30 steps, 1 GPU, no W&B) ..."
python scripts/train_sft.py \
    model=qwen25_0p5b \
    sft=base \
    parallelism=none \
    data=sft_traces \
    "sft.max_steps=30" \
    "sft.per_device_train_batch_size=2" \
    "sft.gradient_accumulation_steps=2" \
    "sft.max_seq_len=1024" \
    "sft.packing=false" \
    "sft.dataloader_num_workers=2" \
    "sft.save_steps=999" \
    "sft.optimizer=adamw_torch" \
    "data.target_fn=action_only" \
    "logging.wandb_enabled=false" \
    "logging.log_every_n_steps=5" \
    "run_name=smoke_test_${SLURM_JOB_ID}" \
    2>&1 | tee logs/smoke_sft_${SLURM_JOB_ID}.log | tail -60

echo ""
echo "[4/5] SFT smoke run finished. Checking output ..."
# Check that a checkpoint was produced
if ls checkpoints/sft/smoke_test_*/checkpoint-* 1>/dev/null 2>&1; then
    echo "✓ Checkpoint directory created"
    ls -la checkpoints/sft/smoke_test_*/
else
    echo "⚠ No checkpoint found (expected with save_steps=999 for 30 steps)"
fi

# ─── Verify logged metrics ────────────────────────────────────
echo ""
echo "[5/5] Checking JSONL metrics log ..."
METRICS_FILE=$(ls logs/smoke_test_${SLURM_JOB_ID}/metrics.jsonl 2>/dev/null | head -1)
if [ -n "$METRICS_FILE" ]; then
    echo "Metrics file: $METRICS_FILE"
    echo "Last 3 entries:"
    tail -3 "$METRICS_FILE"
else
    echo "Metrics file not found at logs/smoke_test_${SLURM_JOB_ID}/metrics.jsonl"
    echo "Available log dirs:"
    ls logs/ 2>/dev/null || echo "(no logs dir)"
fi

echo ""
echo "============================================================"
echo "Smoke test complete!"
echo "Review full log: logs/smoke_sft_${SLURM_JOB_ID}.log"
echo "============================================================"

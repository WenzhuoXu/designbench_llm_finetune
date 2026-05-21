#!/bin/bash
# Submit all §6.3 ablation training jobs to SLURM.
#
# Each job uses grpo_h100.sbatch with a different RL_CONFIG and RUN_NAME.
# Jobs run independently — no dependency chain — so they compete for nodes.
# Wall time is 24h per job; resubmit individually if a job times out.
#
# Usage:
#   bash slurm/submit_ablations.sh                        # all jobs, qwen3_14b
#   bash slurm/submit_ablations.sh deepseek_r1_14b        # different model
#   bash slurm/submit_ablations.sh qwen3_14b /path/ckpt   # from SFT checkpoint
#
# §6.3 ablation map:
#   Step 1a  grpo_abl_01a_baseline_alpha2    r_env only, α=2
#   Step 1b  grpo_abl_01b_baseline_alpha5    r_env only, α=5 (primary baseline)
#   Step 1c  grpo_abl_01c_baseline_alpha10   r_env only, α=10
#   Step 2   grpo_abl_02_tree_expansion      +tree advantage (use_tree_expansion=true)
#   Step 3   (grpo_abl_01b_baseline_alpha5)  re-run step 1b from SFT checkpoint
#   Step 4   grpo_abl_04_pattern_signals     +macro + dead-end
#   Step 5   grpo_abl_05_llm_capability      +stagnation escape + difficulty weight
#   Step 6   grpo_abl_06_reasoning_grounding +forward prediction
#   Step 7   grpo_posterior                  full posterior reward (already exists)

set -euo pipefail

MODEL="${1:-qwen3_14b}"
CHECKPOINT="${2:-}"

SBATCH="slurm/grpo_qwen3_lora_2gpu.sbatch"
TIMESTAMP=$(date +%Y%m%d_%H%M)

submit() {
    local config="$1"
    local label="$2"
    local run_name="${MODEL}_${label}_${TIMESTAMP}"
    local ckpt_args=""
    if [ -n "$CHECKPOINT" ]; then
        ckpt_args="CHECKPOINT=$CHECKPOINT"
    fi
    echo "Submitting: $run_name  (rl=$config)"
    env MODEL="$MODEL" \
        RL_CONFIG="$config" \
        RUN_NAME="$run_name" \
        $ckpt_args \
        sbatch "$SBATCH"
}

# ── Step 1: Baseline α sweep ──────────────────────────────────────────────────
submit grpo_abl_01a_baseline_alpha2  "abl01a_base_a2"
submit grpo_abl_01b_baseline_alpha5  "abl01b_base_a5"
submit grpo_abl_01c_baseline_alpha10 "abl01c_base_a10"

# ── Step 2: +Tree expansion ───────────────────────────────────────────────────
submit grpo_abl_02_tree_expansion    "abl02_tree"

# ── Step 3: +SFT priors (re-run step 1b from SFT checkpoint) ─────────────────
# Requires a pre-built SFT checkpoint. Submit manually after warmstart SFT:
#   MODEL=qwen3_14b CHECKPOINT=checkpoints/sft/qwen3_sft/final \
#   RL_CONFIG=grpo_abl_01b_baseline_alpha5 \
#   RUN_NAME=qwen3_abl03_sft_prior_<date> sbatch slurm/grpo_h100.sbatch
echo ""
echo "Step 3 (SFT prior): submit manually after warmstart SFT completes."
echo "  MODEL=$MODEL CHECKPOINT=<sft_ckpt> RL_CONFIG=grpo_abl_01b_baseline_alpha5 \\"
echo "  RUN_NAME=${MODEL}_abl03_sft_prior_${TIMESTAMP} sbatch $SBATCH"
echo ""

# ── Step 4: +Pattern signals ──────────────────────────────────────────────────
submit grpo_abl_04_pattern_signals   "abl04_patterns"

# ── Step 5: +LLM-capability terms ────────────────────────────────────────────
submit grpo_abl_05_llm_capability    "abl05_llm_cap"

# ── Step 6: +Reasoning grounding ─────────────────────────────────────────────
submit grpo_abl_06_reasoning_grounding "abl06_reasoning"

# ── Step 7: Full posterior reward ────────────────────────────────────────────
submit grpo_posterior                "abl07_full_posterior"

echo ""
echo "All ablation jobs submitted at $TIMESTAMP for model=$MODEL"
echo "Monitor with: squeue -u \$USER"
echo "W&B project: designbench-training  (filter by run_name prefix: ${MODEL}_abl)"

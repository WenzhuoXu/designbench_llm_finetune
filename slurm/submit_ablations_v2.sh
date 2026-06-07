#!/bin/bash
# §6.3 Ablation Study v2 — phased submission (fixes LR bug + proper metrics)
#
# Changes vs. v1 (May 19, 2026):
#   - All jobs start from GOLD warmstart (100% format compliance, May 21)
#   - max_steps_train=200, num_epochs=100 → cosine T_max=200 (not 50)
#   - Tracks eval/rho_tree_agreement (ρ(t)) and eval/feasibility_rate
#   - Phased: submit v2_01b first, verify stability, then submit remaining 5
#
# Phase 1: verify LR fix + stability (30 min)
# Phase 2: submit remaining 5 jobs after verification
#
# Usage:
#   bash slurm/submit_ablations_v2.sh phase1    # primary baseline only
#   bash slurm/submit_ablations_v2.sh phase2    # remaining 5 jobs

set -euo pipefail

GOLD_CKPT="checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final"
MODEL="qwen3_14b"
SBATCH="slurm/grpo_qwen3_lora_2gpu.sbatch"
TIMESTAMP=$(date +%Y%m%d_%H%M)
PHASE="${1:-phase1}"

submit() {
    local config="$1" label="$2" ckpt="$3"
    local run_name="${MODEL}_${label}_${TIMESTAMP}"
    echo "  Submitting: $run_name  (rl=$config)"
    # Use --export to explicitly propagate vars; env-prefix alone unreliable on Bridges-2.
    sbatch --export="ALL,MODEL=${MODEL},RL_CONFIG=${config},RUN_NAME=${run_name},CHECKPOINT=${ckpt}" \
        "$SBATCH"
}

if [ "$PHASE" = "phase1" ]; then
    echo "========================================================"
    echo "PHASE 1: Primary baseline — verify LR fix + stability"
    echo "========================================================"
    submit grpo_abl_v2_01b "v2_01b_alpha5" "$GOLD_CKPT"
    echo ""
    echo "Job submitted. After ~30 min check stability:"
    echo "  tail logs/\$(ls -t logs/ | grep v2_01b | head -1)/metrics.jsonl"
    echo ""
    echo "Stability criteria (ALL must pass before Phase 2):"
    echo "  Step 10 LR ≈ 4.61e-7  (vs 4.80e-7 = still buggy T_max=50)"
    echo "  Step 20 KL > 0.003    (policy is actually moving)"
    echo "  Step 30 clip_ratio/region_mean > 0  (gradient clipping firing)"
    echo "  No OOM or NCCL errors in logs/slurm/*.err"
    echo ""
    echo "If verified, run: bash slurm/submit_ablations_v2.sh phase2"
    echo ""
    echo "If LR is still ~4.80e-7 at step 10:"
    echo "  → T_max bug persists. Edit grpo_trainer.py to pass"
    echo "    warmup_steps=10 and lr_scheduler_kwargs={'num_cycles': 0.5}"
    echo "    explicitly, bypassing HF's num_training_steps computation."

elif [ "$PHASE" = "phase2" ]; then
    echo "========================================================"
    echo "PHASE 2: Remaining 5 ablation jobs"
    echo "========================================================"

    # α sweep (H1: feasibility monotone in α)
    submit grpo_abl_v2_01a  "v2_01a_alpha2"  "$GOLD_CKPT"
    submit grpo_abl_v2_01c  "v2_01c_alpha10" "$GOLD_CKPT"

    # Tree expansion (H2: ρ(t) → ≥0.85, feasibility improves vs baseline)
    submit grpo_abl_v2_02_tree    "v2_02_tree_d1"  "$GOLD_CKPT"
    submit grpo_abl_v2_04_tree_d2 "v2_04_tree_d2"  "$GOLD_CKPT"

    # SFT prior control: same config as v2_01b but no warmstart (H3: gold ckpt matters)
    base_run="${MODEL}_v2_03_base_${TIMESTAMP}"
    echo "  Submitting: ${base_run}  (BASE MODEL, no warmstart)"
    sbatch --export="ALL,MODEL=${MODEL},RL_CONFIG=grpo_abl_v2_01b,RUN_NAME=${base_run},CHECKPOINT=,ALLOW_BASE_GRPO=1" \
        "$SBATCH"

    echo ""
    echo "5 jobs submitted at $TIMESTAMP"
    echo "W&B project: designbench-training"
    echo "Run name prefix: ${MODEL}_v2_  (filter in W&B to isolate this study)"
    echo ""
    echo "Key metrics to watch per job:"
    echo "  eval/rho_tree_agreement  — rises to ≥0.85 for v2_02_tree, v2_04_tree_d2"
    echo "  rewards/compute_rewards/mean  — increases with training (within-condition)"
    echo "  train/kl  — should be > 0.01 by step 100"
    echo "  clip_ratio/region_mean  — > 0 confirms policy is changing"

else
    echo "Unknown phase: '$PHASE'. Use 'phase1' or 'phase2'."
    exit 1
fi

#!/bin/bash
# Mechanics test of the con1 full-set chain (scripts/con1/orchard_full.sh) on one preempt GPU: a fresh r8 LoRA on the
# 42 smoke rows, 2 rows per step, a checkpoint every step. SIGUSR1 by hand mid-run (scancel --signal=USR1 --batch: the
# batch shell only, as --signal=B:USR1 does at the time limit) -> wait for a checkpoint, stop, export; the successor
# (CHAIN_MAX 2) resumes from the export (optimizer restarted, lr re-warmed) and finishes the epoch. Not a training
# run: it checks resume from an export, export, chain hand-over and DONE.
#SBATCH -J con1_full_test
#SBATCH -A ccm
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -t 00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/full/logs/con1_full_test_%j.out
S=/project/community/wxu2/con1/full/smoke
export REPO="${REPO:-/project/community/wxu2/con1/repo2m}"
export INIT=none M=$S/chain_test NTRAIN=1 ROWS_FILE=$S/train_q12.jsonl DEV=$S/dev4.jsonl PER=2 Q=12 \
  SAVE_EVERY=1 DEV_EVERY=3 REWARM=2 EXPORT_EVERY=240 CHAIN_MAX=${CHAIN_MAX:-2} \
  NEXT_ARGS="-p preempt --qos=preempt_qos" SBATCH_FILE=$REPO/slurm/con1_full_test.sbatch \
  TEST_OVR="policy.lora.r=8 policy.lora.lora_alpha=16 policy.lora.target_modules=model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
exec bash "$REPO/scripts/con1/orchard_full.sh"

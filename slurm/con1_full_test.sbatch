#!/bin/bash
# Mechanics test of the con1 full-set chain (scripts/con1/orchard_full.sh) on one GPU: an r8 LoRA on small rows, 2 rows
# per step, a checkpoint every step, the slot guard (scripts/con1/guard_v3.sh) writing into test slots ($RES/slots:
# small pre-made A.bin / B.bin / F1-F3.bin). INIT=none (a fresh LoRA) or INIT=F (an adapter from test F slots).
# SIGUSR1 by hand mid-run (scancel --signal=USR1 --batch: the batch shell only, as --signal=B:USR1 does at the time
# limit) -> wait for a checkpoint, stop, FORCE slot copy, exit; the successor (CHAIN_MAX 2) restores the slot, resumes
# (fresh optimizer, re-warm) and finishes the epoch. Not a training run.
#   sbatch -p general --qos=general_qos [--export=ALL,M=...,ROWS_FILE=...] slurm/con1_full_test.sbatch
#SBATCH -J con1_full_test
#SBATCH -A ccm
#SBATCH -t 00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/full/logs/con1_full_test_%j.out
S=/project/community/wxu2/con1/full/smoke
export REPO="${REPO:-/project/community/wxu2/con1/repo2m}"
export INIT=${INIT:-none} M=${M:-$S/chain_test} NTRAIN=1 ROWS_FILE=${ROWS_FILE:-$S/train_q12.jsonl} DEV=$S/dev4.jsonl PER=2 Q=12 \
  SAVE_EVERY=1 DEV_EVERY=3 GUARD_EVERY=120 CHAIN_MAX=${CHAIN_MAX:-2} RES=${RES:-$S/reserve_test} \
  NEXT_ARGS="${NEXT_ARGS:--p general --qos=general_qos}" SBATCH_FILE=$REPO/slurm/con1_full_test.sbatch \
  TEST_OVR="policy.lora.r=8 policy.lora.lora_alpha=16 policy.lora.target_modules=model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
exec bash "$REPO/scripts/con1/orchard_full.sh"

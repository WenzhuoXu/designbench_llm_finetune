#!/bin/bash
# con1 training (docs/consequence_round_con1.md) on Orchard: 4 GPUs, one trainer rank each, from the con1 repo copy
# (never the live ~/rl/repo; /home/wxu2 is over quota, so repo and logs live under /project/community/wxu2/con1).
# Restart-safe: the trainer resumes from the newest complete checkpoint (every 10 steps by default; on preempt add
# OVERRIDES="sft.save_every=1" and --requeue).
#   advanced: sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,RUN_DIR=...,ROWS=...,DEV=... slurm/con1_sft.sbatch
#   preempt:  sbatch -p preempt --qos=preempt_qos --requeue --export=ALL,... slurm/con1_sft.sbatch
#SBATCH -J con1_sft
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_sft_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo}"
export CONFIG="${CONFIG:-configs/rl/con1_sft.yaml}"
bash "$REPO/scripts/rl/orchard_sft_rows.sh"

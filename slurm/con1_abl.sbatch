#!/bin/bash
# con1 ablation round (scripts/con1/orchard_abl.sh): 4 GPUs = 4 parallel 1-GPU LoRA arms, then their skill questions.
#   sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,ARMS=...,OUT=...,QUERIES=...,VARIANTS=... slurm/con1_abl.sbatch
#SBATCH -J con1_abl
#SBATCH -A ccm
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_abl_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo}"
bash "$REPO/scripts/con1/orchard_abl.sh"

#!/bin/bash
# con1 stage 2 + control (scripts/con1/orchard_stage2.sh) on Orchard's advanced queue: 4 GPUs.
#   sbatch -p advanced --qos=adv_4gpu_qos --dependency=afterany:<round-1 job> slurm/con1_stage2.sbatch
#SBATCH -J con1_s2
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_s2_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo}"
bash "$REPO/scripts/con1/orchard_stage2.sh"

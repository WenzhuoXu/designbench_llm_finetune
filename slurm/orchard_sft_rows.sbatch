#!/bin/bash
# A supervised round on rows (scripts/rl/orchard_sft_rows.sh) on Orchard's advanced queue: 4 GPUs,
# one trainer rank each. A resubmitted job resumes from the newest complete checkpoint.
#   sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/<name>,CONFIG=configs/rl/<cfg>.yaml,ROWS=<train.jsonl>,DEV=<dev.jsonl> \
#          slurm/orchard_sft_rows.sbatch
#SBATCH -J dx_sft
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_sft_%j.out
bash "${REPO:-$HOME/rl/repo}/scripts/rl/orchard_sft_rows.sh"

#!/bin/bash
# A preference round on pairs (scripts/rl/orchard_pref_rows.sh) on Orchard's advanced queue: 4 GPUs, one trainer
# rank each. A resubmitted job resumes from the newest complete checkpoint.
#   sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/<name>,CONFIG=configs/rl/<cfg>.yaml,PAIRS=<pairs.jsonl>,\
#          REPLAY=<replay.jsonl>,DEV_PAIRS=<dev pairs>,DEV=<dev rows> slurm/orchard_pref_rows.sbatch
#SBATCH -J dx_pref
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_pref_%j.out
bash "${REPO:-$HOME/rl/repo}/scripts/rl/orchard_pref_rows.sh"

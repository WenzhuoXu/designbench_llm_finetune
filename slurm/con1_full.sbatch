#!/bin/bash
# con1 stage-1 continuation on the full consequence row set: one 24 h link of a chain (scripts/con1/orchard_full.sh).
#   sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,INIT=<stage-1 final adapter>,M=<run dir> slurm/con1_full.sbatch
# SIGUSR1 30 min before the limit: the job waits for the next checkpoint, exports it and hands over to its successor.
#SBATCH -J con1_full
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --signal=B:USR1@1800
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/full/logs/con1_full_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo2m}"
exec bash "$REPO/scripts/con1/orchard_full.sh"

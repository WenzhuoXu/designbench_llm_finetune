#!/bin/bash
# con1 main run (scripts/con1/orchard_main.sh): stage 1 consequence rows, skill questions, stage 2 policy rows, again.
#   sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,FMT=...,PER_FAMILY=1000,M=/project/community/wxu2/con1/main/run1 slurm/con1_main.sbatch
#SBATCH -J con1_main
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_main_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo}"
bash "$REPO/scripts/con1/orchard_main.sh"

#!/bin/bash
# con1 skill test (scripts/con1/orchard_skill.sh): vLLM servers with the adapters, then the fixed questions.
# Restart-safe (the client resumes by question), so it may run on preempt with --requeue.
#   sbatch -p preempt --qos=preempt_qos --requeue --gres=gpu:1 --export=ALL,NSERVERS=1,OUT=...,QUERIES=...,\
#          ADAPTERS="pref1=/project/community/wxu2/rl/runs/pref1_dx38mtR_20261123/final",MAXTOK_pref1=6144 slurm/con1_skill.sbatch
#SBATCH -J con1_skill
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_skill_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo}"
bash "$REPO/scripts/con1/orchard_skill.sh"

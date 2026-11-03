#!/bin/bash
# Uncapped dataxp evaluation (scripts/rl/orchard_dataxp_eval.sh) on Orchard's preempt queue: one node,
# 4 GPUs = 4 vLLM servers. Preemption is safe: --requeue restarts the job and the episodes resume by key.
#   sbatch --export=ALL,OUT=/home/wxu2/rl/eval/<label>,LABEL=base,PROBLEMS="<files>" slurm/orchard_dataxp_eval.sbatch
#SBATCH -J dx_eval
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A ccm
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_eval_%j.out
bash "${REPO:-$HOME/rl/repo}/scripts/rl/orchard_dataxp_eval.sh"

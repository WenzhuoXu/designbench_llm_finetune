#!/bin/bash
# Multi-turn GRPO on dataxp (llm_finetune/rl_dataxp), one Orchard node, 4 GPUs:
# 2 vLLM servers + 2 training ranks. `advanced` is not preempted (24 h, one node per user);
# --requeue covers node failures. A job that hits the time limit is simply resubmitted: the
# trainer resumes from its newest complete checkpoint.
#   sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/<name>,CONFIG=configs/rl/dataxp_grpo_p2_datav4.yaml \
#          [SMOKE=1] slurm/orchard_dataxp_grpo.sbatch
#SBATCH -J dx_grpo
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/rl/logs/dx_grpo_%j.out
bash "${REPO:-$HOME/rl/repo}/scripts/rl/orchard_dataxp_grpo.sh"

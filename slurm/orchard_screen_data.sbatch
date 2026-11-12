#!/bin/bash
# Data for a screened supervised round (scripts/rl/orchard_screen_data.sh) on Orchard's advanced queue:
# one node, 4 GPUs = 4 vLLM servers, the CPUs for the simulator stages. A restarted job goes on where
# it stopped (stage markers; sampling and scoring resume by state).
#   sbatch --export=ALL,OUT=/home/wxu2/rl/screen/<name>,LABEL=dx38mtR,ADAPTER=<peft dir>,ROLLOUTS=<run>/rollouts,POOL=<problems.jsonl> \
#          slurm/orchard_screen_data.sbatch
#SBATCH -J dx_screen
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=600G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_screen_%j.out
bash "${REPO:-$HOME/rl/repo}/scripts/rl/orchard_screen_data.sh"

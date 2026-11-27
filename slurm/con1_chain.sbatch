#!/bin/bash
#SBATCH -J con1_chain
#SBATCH -A ccm
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_chain_%j.out
bash /project/community/wxu2/con1/repo/scripts/con1/chain_truth.sh

#!/bin/bash
#SBATCH -J con1_main_prep
#SBATCH -A ccm
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/logs/con1_main_prep_%j.out
cd /project/community/wxu2/con1/gen
bash /project/community/wxu2/con1/repo/scripts/con1/main_prep.sh

#!/bin/bash
# con1 full row set, CPU prep (scripts/con1/full_prep.sh): off-menu moves for the remaining native states, the builder
# at Q = 12 and 36 (timed, exact tokens, checked, rebuilt once to prove the bytes repeat).
#SBATCH -J con1_full_prep
#SBATCH -A ccm
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /project/community/wxu2/con1/full/logs/con1_full_prep_%j.out
export REPO="${REPO:-/project/community/wxu2/con1/repo2m}"
cd /project/community/wxu2/con1/full
bash "$REPO/scripts/con1/full_prep.sh"

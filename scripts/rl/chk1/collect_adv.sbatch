#!/bin/bash
# collect_adv.sbatch (Orchard, advanced, 4 GPUs): the chk1 collection's third collector, submitted afterany on pref1's
# second benchmark job (the user's one advanced node runs one job at a time).
#  1. stops the general-queue collector OLD_JOB (it walks pool_rev into collect_rev) and waits until it has left the queue
#  2. pool_adv = the problems of pool_rev that collect_rev has no episode of, same order (built once; a restart reuses it)
#  3. collects them into collect_adv: a new file, so no node appends to a file another node wrote (Orchard NFS)
# drc walks the same pool front to back; the collectors are stopped where they meet. Episodes resume by key.
#   sbatch --dependency=afterany:<job> --export=ALL,OLD_JOB=<general collector> scripts/rl/chk1/collect_adv.sbatch
#SBATCH -J dx_chk1_adv
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_chk1_adv_%j.out
set -u
C=/project/community/wxu2/rl/chk1
REPO="${REPO:-$HOME/rl/repo}"
OLD="${OLD_JOB:?the general-queue collector job id}"
echo "=== $(date '+%F %T') job ${SLURM_JOB_ID:-x} on $(hostname): stopping $OLD"
squeue -h -j "$OLD" 2>/dev/null | grep -q . && scancel "$OLD"
for i in $(seq 90); do squeue -h -j "$OLD" 2>/dev/null | grep -q . || break; sleep 10; done
squeue -h -j "$OLD" 2>/dev/null | grep -q . && { echo "ABORT: $OLD is still in the queue"; exit 3; }
sleep 30                                    # NFS: let the old node's last writes reach the server
mkdir -p $C/collect_adv
if [ ! -s $C/collect_adv/pool_adv.jsonl ]; then
  /project/community/wxu2/rl/envs/sim/bin/python "$REPO/scripts/rl/chk1/remaining_pool.py" \
      $HOME/rl/logs/chk1/pool_rev.jsonl $C/collect_adv/pool_adv.jsonl $C/collect_rev/dx38mtR.episodes.jsonl \
      || { echo "ABORT: pool_adv not built"; exit 4; }
fi
export OUT=$C/collect_adv LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final THINK=1 \
       HARNESS_SCRIPT=dataxp_policy_eval_v3rc.py HARNESS_ARGS="--open-rules --record-prompts --checks 4 --check-actions" \
       PROBLEMS=$C/collect_adv/pool_adv.jsonl VARIANTS="photonics grid2op engibench" RUNS=1 NSERVERS=4 \
       WORKERS=${WORKERS:-80}
exec bash "$REPO/scripts/rl/orchard_dataxp_eval.sh"

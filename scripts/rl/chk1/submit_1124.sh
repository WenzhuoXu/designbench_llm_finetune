#!/bin/bash
# submit_1124.sh (Orchard login node, 2026-11-24 02:40):
#  1. pref1's benchmark as two advanced jobs of all 11 families each (pair a -> ${E}_all, pair b -> ${E}_all_b),
#     afterok on the training 175023; replaces 175024/175026 (general, weak) and 175025/175027 (advanced, rest):
#     general_qos runs one job at a time and another session's job holds it, so the weak families (the ones the
#     pairs train) would have come last.
#  2. chk1 collection on general (1 GPU): dx38mtR, harness v3rc --checks 4 --check-actions on the reversed pool
#     (drc collects the same pool front to back) -> /project/community/wxu2/rl/chk1/collect_rev.
set -u
cd ~/rl/repo || exit 2
NAME=pref1_dx38mtR_20261123
RUN_DIR=/project/community/wxu2/rl/runs/$NAME
E=/home/wxu2/rl/eval/${NAME}_open
for o in ${E}_all ${E}_all_b /project/community/wxu2/rl/chk1/collect_rev; do [ -e "$o" ] && { echo "ABORT: $o exists"; exit 2; }; done
for f in ~/rl/harness/scripts/dataxp_policy_eval_v3rc.py ~/rl/logs/chk1/pool_rev.jsonl /project/community/wxu2/rl/inputs/dx38mtR_final/adapter_model.safetensors; do
  [ -s "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
P="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
for sfx in "" _b; do
  A=$(OUT=${E}_all$sfx LABEL=pref1 ADAPTER=$RUN_DIR/final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules \
      PROBLEMS="$P" NSERVERS=4 WORKERS=128 \
      sbatch --parsable --dependency=afterok:175023 -p advanced --qos=adv_4gpu_qos -t 24:00:00 -J dx_eval_p1A$sfx \
      --export=ALL slurm/orchard_dataxp_eval.sbatch) || { echo "!!! all$sfx not submitted"; exit 6; }
  echo "pref1 benchmark$sfx (all 11 families): advanced $A -> ${E}_all$sfx"
done
scancel 175024 175025 175026 175027 && echo "cancelled 175024 175025 175026 175027"
mkdir -p /project/community/wxu2/rl/chk1
C=$(OUT=/project/community/wxu2/rl/chk1/collect_rev LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final THINK=1 \
    HARNESS_SCRIPT=dataxp_policy_eval_v3rc.py HARNESS_ARGS="--open-rules --record-prompts --checks 4 --check-actions" \
    PROBLEMS=/home/wxu2/rl/logs/chk1/pool_rev.jsonl VARIANTS="photonics grid2op engibench" RUNS=1 NSERVERS=1 WORKERS=16 \
    sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t 12:00:00 -J dx_chk1_col \
    --export=ALL slurm/orchard_dataxp_eval.sbatch) || { echo "!!! collection not submitted"; exit 6; }
echo "chk1 collection: general $C -> /project/community/wxu2/rl/chk1/collect_rev"
sleep 5
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.10M %R'

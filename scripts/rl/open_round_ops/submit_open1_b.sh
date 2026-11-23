#!/bin/bash
# submit_open1_b.sh (Orchard login node): the second pair of benchmark runs of the open1 adapter (2026-11-23), queued
# afterok on its training so it starts the moment GPUs allow. dx38's T=0 runs disagree between pairs (21 of 60
# engibench problems across four runs), so both sides of the A/B get at least two pairs.
#   advanced (4 GPUs; adv_4gpu_qos serialises it with 174790): the eight other families -> ${E}_rest_b
#   general (1 GPU; general_qos serialises it with 174789): engibench -> ${E}_weak_b
#   drc GPU 4 runs photonics + grid2op of this pair (drc_eval_open.sh, LABEL=open1) once the adapter is copied there.
set -u
T=${TRAIN_JOB:-174788}
RUN_DIR=/project/community/wxu2/rl/runs/open1_dx38mtR_20261123
export LABEL=open1 ADAPTER=$RUN_DIR/final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
E=/home/wxu2/rl/eval/open1_dx38mtR_20261123_open
cd ~/rl/repo || exit 2
for o in ${E}_rest_b ${E}_weak_b; do [ -e "$o" ] && { echo "ABORT: $o exists"; exit 2; }; done
A=$(OUT=${E}_rest_b VARIANTS="circuit2 frame3d2 grid2 seir2 traffic2 inventory2 molopt thermal2" NSERVERS=4 WORKERS=128 \
    sbatch --parsable --dependency=afterok:$T -p advanced --qos=adv_4gpu_qos -t 24:00:00 -J dx_eval_o1Rb \
    --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! rest pair b not submitted"
echo "advanced job $A -> ${E}_rest_b"
G=$(OUT=${E}_weak_b VARIANTS="engibench" NSERVERS=1 WORKERS=48 \
    sbatch --parsable --dependency=afterok:$T -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G \
    -t 8:00:00 -J dx_eval_o1Wb --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! engibench pair b not submitted"
echo "general job $G -> ${E}_weak_b"
sleep 8
squeue -u "$USER" -o '%.9i %.15j %.10P %.9T %.10M %R'

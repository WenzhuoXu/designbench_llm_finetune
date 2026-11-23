#!/bin/bash
# submit_dx38_rest_b.sh (Orchard login node): a second pair of dx38 runs under the open rules on the eight families
# outside the weak three, while the round trains on advanced (2026-11-23). T=0 runs disagree on a third of the
# engibench problems between pairs, so the families read as "replay" and "held out" get a second pair as well.
#   general (1 GPU, -t 5:15 so it is off before the round's own benchmark needs general): inventory2 molopt thermal2
#     frame3d2 -> /home/wxu2/rl/eval/dx38mtR_open_20261123_rest_b1
#   preempt (4 GPUs, --requeue; episodes resume by key): circuit2 grid2 seir2 traffic2
#     -> /home/wxu2/rl/eval/dx38mtR_open_20261123_rest_b2
# Same settings as dx38's open-rules baseline (split_open_eval.sh); keys are r1/r2 again: relabel when pooling.
set -u
export LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
cd ~/rl/repo || exit 2
export OUT=/home/wxu2/rl/eval/dx38mtR_open_20261123_rest_b1 VARIANTS="inventory2 molopt thermal2 frame3d2" NSERVERS=1 WORKERS=48
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
G=$(sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t 5:15:00 -J dx_eval_openRb1 --export=ALL slurm/orchard_dataxp_eval.sbatch) || exit 3
echo "general job $G -> $OUT"
export OUT=/home/wxu2/rl/eval/dx38mtR_open_20261123_rest_b2 VARIANTS="circuit2 grid2 seir2 traffic2" NSERVERS=4 WORKERS=128
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
P=$(sbatch --parsable -p preempt --qos=preempt_qos -t 8:00:00 -J dx_eval_openRb2 --export=ALL slurm/orchard_dataxp_eval.sbatch) || exit 3
echo "preempt job $P -> $OUT"
sleep 10
squeue -u "$USER" -o '%.9i %.15j %.10P %.9T %.10M %R'

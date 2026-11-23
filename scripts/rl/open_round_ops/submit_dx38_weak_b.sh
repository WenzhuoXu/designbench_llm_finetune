#!/bin/bash
# submit_dx38_weak_b.sh (Orchard login node): a second pair of dx38 runs under the open rules on photonics, grid2op,
# engibench (the families the open-rules rows are drawn from) on the general GPU while the round trains on advanced:
# twice the baseline runs where the A/B is read first. Same settings as dx38's open-rules baseline (split_open_eval.sh);
# separate OUT (its keys are r1/r2 again: relabel r3/r4 when pooling). Time limit 7 h so the general GPU is free before
# the round's own benchmark (afterok on the training) needs it.
set -u
export LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
export OUT=/home/wxu2/rl/eval/dx38mtR_open_20261123_weak_b VARIANTS="photonics grid2op engibench" NSERVERS=1 WORKERS=48
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
cd ~/rl/repo || exit 2
G=$(sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t 7:00:00 -J dx_eval_openWb --export=ALL slurm/orchard_dataxp_eval.sbatch) || exit 3
echo "general job $G -> $OUT"
sleep 8
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.10M %R'

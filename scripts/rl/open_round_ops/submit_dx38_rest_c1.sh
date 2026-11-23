#!/bin/bash
# submit_dx38_rest_c1.sh (Orchard login node): a third pair of dx38 runs under the open rules on inventory2 molopt
# thermal2 frame3d2, in the general GPU's gaps (2026-11-23). TLIMIT keeps each piece off general before the round's
# own benchmark (174789) needs it; rerun the script to resume the same OUT by episode key.
set -u
export LABEL=dx38mtR ADAPTER=/project/community/wxu2/rl/inputs/dx38mtR_final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
export OUT=/home/wxu2/rl/eval/dx38mtR_open_20261123_rest_c1 VARIANTS="inventory2 molopt thermal2 frame3d2" NSERVERS=1 WORKERS=48
cd ~/rl/repo || exit 2
G=$(sbatch --parsable -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G -t "${TLIMIT:?TLIMIT}" -J dx_eval_openRc1 --export=ALL slurm/orchard_dataxp_eval.sbatch) || exit 3
echo "general job $G -> $OUT (limit $TLIMIT)"
sleep 8
squeue -u "$USER" -o '%.9i %.15j %.10P %.9T %.10M %R'

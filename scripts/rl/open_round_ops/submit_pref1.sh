#!/bin/bash
# submit_pref1.sh (Orchard login node, 2026-11-23): preference round 1 from dx38mtR (llm_finetune/rl_dataxp/dpo_rows.py,
# configs/rl/dataxp_pref1.yaml) on the open-rules pairs (open_round/pairs: build_pairs.py per share, merge_pairs.py),
# then two benchmark pairs under the open rules, afterok on the training (the open1 round's settings):
#   general (1 GPU): photonics grid2op engibench -> ${E}_weak, then ${E}_weak_b (general_qos runs one at a time)
#   advanced (4 GPUs): the other eight families -> ${E}_rest, then ${E}_rest_b (adv_4gpu_qos: one at a time)
# The training queues behind the open1 benchmark's last advanced job (the QOS allows one 4-GPU job).
set -u
NAME=pref1_dx38mtR_20261123
R=/project/community/wxu2/rl/screen/open_round
RUN_DIR=/project/community/wxu2/rl/runs/$NAME
E=/home/wxu2/rl/eval/${NAME}_open
cd ~/rl/repo || exit 2
for f in $R/pairs/pairs_train.jsonl $R/pairs/pairs_dev.jsonl $R/pairs/replay_pool.jsonl $R/dev_open1.jsonl \
         configs/rl/dataxp_pref1.yaml slurm/orchard_pref_rows.sbatch scripts/rl/orchard_pref_rows.sh llm_finetune/rl_dataxp/dpo_rows.py; do
  [ -s "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
[ -e "$RUN_DIR" ] && { echo "ABORT: $RUN_DIR exists"; exit 2; }
for o in ${E}_weak ${E}_rest ${E}_weak_b ${E}_rest_b; do [ -e "$o" ] && { echo "ABORT: $o exists"; exit 2; }; done
T=$(sbatch --parsable -J dx_pref1 \
    --export=ALL,RUN_DIR=$RUN_DIR,CONFIG=configs/rl/dataxp_pref1.yaml,PAIRS=$R/pairs/pairs_train.jsonl,REPLAY=$R/pairs/replay_pool.jsonl,DEV_PAIRS=$R/pairs/pairs_dev.jsonl,DEV=$R/dev_open1.jsonl \
    slurm/orchard_pref_rows.sbatch) || { echo "ABORT: training not submitted"; exit 6; }
echo "training: $T (run $RUN_DIR)"
export LABEL=pref1 ADAPTER=$RUN_DIR/final THINK=1 HARNESS_SCRIPT=dataxp_policy_eval_v3r.py HARNESS_ARGS=--open-rules
export PROBLEMS="/home/wxu2/rl/harness/problems_v2.jsonl /home/wxu2/rl/eval/problems_v3_native.jsonl"
for sfx in "" _b; do
  G=$(OUT=${E}_weak$sfx VARIANTS="photonics grid2op engibench" NSERVERS=1 WORKERS=48 \
      sbatch --parsable --dependency=afterok:$T -p general --qos=general_qos --gres=gpu:1 --cpus-per-task=48 --mem=300G \
      -t 12:00:00 -J dx_eval_p1W$sfx --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! weak$sfx not submitted"
  A=$(OUT=${E}_rest$sfx VARIANTS="circuit2 frame3d2 grid2 seir2 traffic2 inventory2 molopt thermal2" NSERVERS=4 WORKERS=128 \
      sbatch --parsable --dependency=afterok:$T -p advanced --qos=adv_4gpu_qos -t 24:00:00 -J dx_eval_p1R$sfx \
      --export=ALL slurm/orchard_dataxp_eval.sbatch) || echo "!!! rest$sfx not submitted"
  echo "benchmark$sfx: general $G -> ${E}_weak$sfx, advanced $A -> ${E}_rest$sfx"
done
sleep 8
squeue -u "$USER" -o '%.9i %.14j %.10P %.9T %.10M %R'

#!/bin/bash
# The gradient probe (llm_finetune/rl_dataxp/grad_probe.py) on SFT rows instead of rollouts: the
# reference for what the probe reads on a signal known to train the model. scripts/rl_probe/
# gold_rows_to_run.py turns PER_FAMILY rows per family into a run directory per pass, then
# slurm/orchard_grad_probe.sbatch runs on it (one process per GPU, temperature 1 as in SFT).
#   PASSES: base = the base model (a fresh LoRA, B = 0), where SFT started
#           init = the config's policy.init_adapter (the SFT result)
#   sbatch --export=ALL,ROWS=<train.jsonl>,CONFIG=configs/rl/<cfg>.yaml,NAME=<name>,[PER_FAMILY=64],[PASSES="base init"] \
#          slurm/orchard_gold_probe.sbatch
# Output: $BASE/rl/probe/<NAME>_<pass>/ ; run directories: $BASE/rl/probe/gold_runs/<NAME>/<pass>.
# A rerun skips what is already complete.
#SBATCH -J dx_gold
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_gold_%j.out
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
ROWS="${ROWS:?SFT rows jsonl}"; CONFIG="${CONFIG:?config yaml}"; NAME="${NAME:?name}"
[ "${CONFIG:0:1}" = / ] || CONFIG="$REPO/$CONFIG"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
read -r MODEL INIT < <("$TRAIN_PY" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c["model"]["path"], c["policy"]["init_adapter"])' "$CONFIG")
[ -n "${MODEL:-}" ] && [ -n "${INIT:-}" ] && [ "$INIT" != None ] || { echo "model.path / policy.init_adapter missing in $CONFIG"; exit 2; }
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} rows=$ROWS name=$NAME per_family=${PER_FAMILY:-64} passes=${PASSES:-base init}"
rc=0
for P in ${PASSES:-base init}; do
  RUN="$B/rl/probe/gold_runs/$NAME/$P"
  case "$P" in
    base) AD=(--fresh-lora --like "$INIT") ;;
    init) AD=(--adapter "$INIT") ;;
    *) echo "unknown pass $P"; exit 2 ;;
  esac
  "$TRAIN_PY" "$REPO/scripts/rl_probe/gold_rows_to_run.py" --rows "$ROWS" --tokenizer "$MODEL" \
      --per-family "${PER_FAMILY:-64}" --run-dir "$RUN" "${AD[@]}" || { echo "prepare $P failed"; exit 3; }
  RUN_DIR="$RUN" OUT="$B/rl/probe/${NAME}_$P" STEPS=0 CONFIG="$CONFIG" \
      OVERRIDES="rollout.temperature=1.0 ${OVERRIDES:-}" LOGD="$HOME/rl/logs/dx_gold_${SLURM_JOB_ID:-x}_$P" \
      bash "$REPO/slurm/orchard_grad_probe.sbatch" || rc=$?
done
echo "DX_GOLD_DONE $(date '+%F %T %Z') rc=$rc"
exit $rc

#!/bin/bash
# The gradient probe on screened first replies of a run's stored rollouts: per problem, the applied
# first reply with the highest potential after it and another, unscreened first reply
# (scripts/rl_probe/first_move_run.py), scored under the config's policy.init_adapter at
# temperature 1 as a supervised step would, then scripts/rl_probe/first_move_analysis.py.
#   sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/<run>,CONFIG=configs/rl/<cfg>.yaml,NAME=<name> \
#          slurm/orchard_first_move_probe.sbatch
# Output: $BASE/rl/probe/<NAME>/ (sketches, first_move_analysis.txt); the probe's run directory:
# $BASE/rl/probe/first_move_runs/<NAME>. A rerun skips what is already complete.
#SBATCH -J dx_fmove
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_fmove_%j.out
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
RUN_DIR="${RUN_DIR:?the RL run whose rollouts are read}"; CONFIG="${CONFIG:?config yaml}"; NAME="${NAME:?name}"
[ "${CONFIG:0:1}" = / ] || CONFIG="$REPO/$CONFIG"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
SIM_PY="${SIM_PY:-$B/rl/envs/sim/bin/python}"
H="${HARNESS:-$HOME/rl/harness}"
C="${CACHE_DIR:-$B/rl/cache}"
PARTS="${PARTS:-8}"
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg
mkdir -p "$TMPDIR" "$C/xdg"
read -r MODEL INIT < <("$TRAIN_PY" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c["model"]["path"], c["policy"]["init_adapter"])' "$CONFIG")
[ -n "${MODEL:-}" ] && [ -n "${INIT:-}" ] && [ "$INIT" != None ] || { echo "model.path / policy.init_adapter missing in $CONFIG"; exit 2; }
RUN="$B/rl/probe/first_move_runs/$NAME"
OUT="$B/rl/probe/$NAME"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN_DIR name=$NAME adapter=$INIT parts=$PARTS"
if [ ! -s "$RUN/prompts.jsonl" ]; then
  # as the rollouts ran: one thread per simulator, PYTHONHASHSEED=0, the run's own harness and spelling
  PYTHONPATH="$REPO" PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  DX_ORIG_SCRIPTS="$H/scripts" DATAXP_SPELLING=multitool \
    "$SIM_PY" "$REPO/scripts/rl_probe/first_move_run.py" prompts --run-dir "$RUN_DIR" \
      --script "$H/scripts/dataxp_policy_eval_v3.py" --out "$RUN" --workers "${WORKERS:-24}" || { echo "prompts failed"; exit 3; }
fi
if [ ! -s "$RUN/labels.jsonl" ]; then
  CUDA_VISIBLE_DEVICES= "$TRAIN_PY" "$REPO/scripts/rl_probe/first_move_run.py" build --run-dir "$RUN_DIR" \
      --tokenizer "$MODEL" --adapter "$INIT" --out "$RUN" --parts "$PARTS" || { echo "build failed"; exit 3; }
fi
STEPS=$(seq -s, 0 $((PARTS-1)))
RUN_DIR="$RUN" OUT="$OUT" STEPS="$STEPS" CONFIG="$CONFIG" ARGS="" \
    OVERRIDES="rollout.temperature=1.0 ${OVERRIDES:-}" LOGD="$HOME/rl/logs/dx_fmove_${SLURM_JOB_ID:-x}" \
    bash "$REPO/slurm/orchard_grad_probe.sbatch"
rc=$?
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 "$TRAIN_PY" -W ignore "$REPO/scripts/rl_probe/first_move_analysis.py" "$OUT" "$RUN/labels.jsonl" \
    > "$HOME/rl/logs/dx_fmove_${SLURM_JOB_ID:-x}_analysis.txt" 2>&1 || rc=$?
cp "$HOME/rl/logs/dx_fmove_${SLURM_JOB_ID:-x}_analysis.txt" "$OUT/first_move_analysis.txt" 2>/dev/null
cat "$HOME/rl/logs/dx_fmove_${SLURM_JOB_ID:-x}_analysis.txt"
echo "DX_FMOVE_DONE $(date '+%F %T %Z') rc=$rc"
exit $rc

#!/bin/bash
# Per-turn gradient sketches of a run's own rollouts (llm_finetune/rl_dataxp/grad_probe.py): one process
# per GPU, each taking its token-balanced share of every step. Reads the run; writes only OUT.
# A rerun skips the (step, shard) files that are already complete.
#   export STEPS=100,80,60,40,20,0      # in the environment: sbatch splits --export values at commas
#   export ARGS="--segments 8 --no-action --hidden-points 33"   # optional extra grad_probe arguments
#   sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/<name>,CONFIG=configs/rl/<cfg>.yaml,\
#          OUT=/project/community/wxu2/rl/probe/<name>,[DIM=65536],[LIMIT=0],[OVERRIDES=...] \
#          slurm/orchard_grad_probe.sbatch
# Each kind of run (whole rationale / segments / hidden states only) gets its own OUT.
#SBATCH -J dx_gprobe
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/dx_gprobe_%j.out
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
RUN_DIR="${RUN_DIR:?run directory}"; CONFIG="${CONFIG:?config yaml}"; STEPS="${STEPS:?steps}"; OUT="${OUT:?output dir}"
[ "${CONFIG:0:1}" = / ] || CONFIG="$REPO/$CONFIG"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
C="${CACHE_DIR:-$B/rl/cache}"
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$C/xdg" "$OUT/logs"
cd "$REPO"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else NG=$(nvidia-smi -L | wc -l); DEVS=(); for i in $(seq 0 $((NG-1))); do DEVS+=("$i"); done; fi
N=${#DEVS[@]}
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$REPO config=$CONFIG run=$RUN_DIR steps=$STEPS out=$OUT gpus=$N args=${ARGS:-}"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; exit 143' TERM INT
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[k]} PYTHONPATH="$REPO" "$TRAIN_PY" -m llm_finetune.rl_dataxp.grad_probe --config "$CONFIG" \
    --run-dir "$RUN_DIR" --steps "$STEPS" --out "$OUT" --shard "$k" --shards "$N" --dim "${DIM:-65536}" \
    --limit "${LIMIT:-0}" ${ARGS:-} ${OVERRIDES:-} > "$OUT/logs/shard_${k}_job${SLURM_JOB_ID:-x}.log" 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
tail -n 3 "$OUT"/logs/shard_*_job${SLURM_JOB_ID:-x}.log
echo "DX_GPROBE_DONE $(date '+%F %T %Z') rc=$rc"
exit $rc

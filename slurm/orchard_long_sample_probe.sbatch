#!/bin/bash
# One-GPU measurement of the trainer's long-sequence path on a run's own step (scripts/rl/long_sample_probe.py):
# peak memory on the step's longest sample, on synthetic longer copies up to the context length, and the
# segmented-vs-whole-layer gradient difference. Reads the run; writes only OUT.
#   sbatch --export=ALL,RUN_DIR=/project/community/wxu2/rl/runs/<name>,CONFIG=configs/rl/<cfg>.yaml,STEP=43,\
#          [REPO=$HOME/rl/repo],[LENGTHS=65000,80000,...],[OVERRIDES=grpo.max_turns_per_episode=3] \
#          slurm/orchard_long_sample_probe.sbatch
#SBATCH -J dx_probe
#SBATCH -p advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH -A ccm
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH -o /home/wxu2/rl/logs/dx_probe_%j.out
set -uo pipefail
B="${BASE:-/project/community/wxu2}"
REPO="${REPO:-$HOME/rl/repo}"
RUN_DIR="${RUN_DIR:?run directory}"; CONFIG="${CONFIG:?config yaml}"; STEP="${STEP:?step}"
[ "${CONFIG:0:1}" = / ] || CONFIG="$REPO/$CONFIG"
TRAIN_PY="${TRAIN_PY:-$B/rl/envs/train/bin/python}"
C="${CACHE_DIR:-$B/rl/cache}"
export HF_HOME="${HF_HOME:-$B/rl/hf}"; export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$C/xdg"
cd "$REPO"
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} repo=$REPO config=$CONFIG run=$RUN_DIR step=$STEP"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
PYTHONPATH="$REPO" "$TRAIN_PY" scripts/rl/long_sample_probe.py --config "$CONFIG" --run-dir "$RUN_DIR" --step "$STEP" \
  --lengths "${LENGTHS:-65000,80000,100000,130000,165000,200000,262144}" --compare-max "${COMPARE_MAX:-14000}" \
  --out "${OUT:-$HOME/rl/logs/dx_probe_${SLURM_JOB_ID:-x}.json}" ${OVERRIDES:-}
echo "DX_PROBE_DONE $(date '+%F %T %Z') rc=$?"

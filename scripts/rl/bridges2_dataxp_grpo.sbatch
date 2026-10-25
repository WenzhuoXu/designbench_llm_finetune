#!/bin/bash
# Multi-turn GRPO on dataxp on one Bridges-2 GPU node (8 x H100-80; the GPU partition only gives
# whole nodes): NVLLM vLLM servers + NTRAIN training ranks, through the same launcher as Orchard.
# Persistent files live on /ocean (repo, harness copy, problems); runs under rl/runs.
#   sbatch [--export=ALL,SMOKE=1,RUN_DIR=...,CONFIG=...] scripts/rl/bridges2_dataxp_grpo.sbatch
#SBATCH -J dx_grpo
#SBATCH -p GPU
#SBATCH --gres=gpu:h100-80:8
#SBATCH -A mch250030p
#SBATCH -t 3:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/rl/logs/dx_grpo_%j.out
O=/ocean/projects/mch250030p/wxu7
export BASE=$O REPO=$O/rl/repo VLLM_VENV=$O/envs/vllm_env TRAIN_PY=$O/envs/d27_env/bin/python
export MODEL_PATH=$O/hf_models/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export CACHE_DIR=$O/rl/cache HF_HOME=$O/rl/cache/hf
export CONFIG=${CONFIG:-configs/rl/dataxp_grpo_p2_bridges2.yaml} RUN_DIR=${RUN_DIR:-$O/rl/runs/b2_canary}
export NVLLM=${NVLLM:-2} NTRAIN=${NTRAIN:-2}
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} commit=$(cat "$REPO/GIT_COMMIT" 2>/dev/null)"
# Pre-check on this node's CPU before any GPU work: the trainer and harness tests under Bridges-2's
# own envs (transformers 5.5 / peft 0.18 / numpy 2.4 here, 5.17 / 0.21 / 1.26 on Orchard). A failure
# ends the job within minutes instead of after the servers and the 27B trainer have loaded.
if [ -z "${SKIP_PRECHECK:-}" ]; then
  PRE="$O/rl/logs/precheck_${SLURM_JOB_ID:-x}.log"
  ( cd "$REPO" && PYTHONPATH="$REPO" DX_HARNESS_DIR="$O/rl/harness" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
      "$TRAIN_PY" -m pytest tests/test_rl_dataxp_core.py tests/test_rl_dataxp_trainer.py \
      tests/test_rl_dataxp_harness.py -q -x -p no:cacheprovider ) > "$PRE" 2>&1
  rc=$?
  grep -v "falling back" "$PRE" | tail -12
  [ $rc -eq 0 ] || { echo "ABORT: pre-check tests failed (rc=$rc), see $PRE"; exit 3; }
fi
bash "$REPO/scripts/rl/orchard_dataxp_grpo.sh"

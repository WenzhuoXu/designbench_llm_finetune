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
bash "$REPO/scripts/rl/orchard_dataxp_grpo.sh"

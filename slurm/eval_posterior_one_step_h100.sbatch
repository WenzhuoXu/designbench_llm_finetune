#!/bin/bash
#SBATCH --job-name=posterior_eval
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --qos=gpu

set -euo pipefail

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs/task: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "============================================================"

module load cuda/12.6.1
module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

export HF_HOME="/ocean/projects/mch250030p/wxu7/hf_models"
export TRANSFORMERS_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_CACHE="/ocean/projects/mch250030p/wxu7/hf_models"
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

mkdir -p logs/slurm logs/posterior

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/qwen3_14b.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/sft/warmstart_qwen3_14b_39083858/final}"
PROBLEMS_DIR="${PROBLEMS_DIR:-/ocean/projects/mch250030p/wxu7/DesignBench/data/problems}"
N_STATES="${N_STATES:-8}"
SAMPLING_MODE="${SAMPLING_MODE:-rollin_policy}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
DEPTH="${DEPTH:-1}"
REFERENCE_DEPTH="${REFERENCE_DEPTH:-2}"
LEVEL1_BRANCHING="${LEVEL1_BRANCHING:-5}"
DEEPER_BRANCHING="${DEEPER_BRANCHING:-3}"
SEED="${SEED:-42}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-15}"
OUTPUT_PATH="${OUTPUT_PATH:-logs/posterior/one_step_eval_${SLURM_JOB_ID}.jsonl}"

echo "Model config: $MODEL_CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Sampling mode: $SAMPLING_MODE"
echo "Output: $OUTPUT_PATH"
echo "Heartbeat seconds: $HEARTBEAT_SECONDS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

python -u scripts/eval_posterior_one_step.py \
  --model-config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --problems-dir "$PROBLEMS_DIR" \
  --n-states "$N_STATES" \
  --sampling-mode "$SAMPLING_MODE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --candidate-count "$CANDIDATE_COUNT" \
  --depth "$DEPTH" \
  --reference-depth "$REFERENCE_DEPTH" \
  --level1-branching "$LEVEL1_BRANCHING" \
  --deeper-branching "$DEEPER_BRANCHING" \
  --output "$OUTPUT_PATH" \
  --seed "$SEED" \
  --heartbeat-seconds "$HEARTBEAT_SECONDS"

echo "============================================================"
echo "Posterior evaluation completed: $(date)"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=mcts_gpt55
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs/task: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "============================================================"

module load anaconda3/2024.10-1
conda activate my_env

PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p logs/slurm logs/posterior

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export your OpenAI API token before submitting this job."
  echo "Example: export OPENAI_API_KEY='sk-...'"
  exit 1
fi

MODEL_CONFIG="${MODEL_CONFIG:-configs/model/openai_gpt55.yaml}"
MODEL_TAG="${MODEL_TAG:-openai_gpt55}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"
OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}"
OPENAI_TIMEOUT_S="${OPENAI_TIMEOUT_S:-180}"
OPENAI_CONCURRENCY="${OPENAI_CONCURRENCY:-8}"
LLM_CALL_BUDGET="${LLM_CALL_BUDGET:-0}"
PROBLEMS_DIR="${PROBLEMS_DIR:-/ocean/projects/mch250030p/wxu7/DesignBench/data/problems}"
PROBLEM_IDS="${PROBLEM_IDS:-auto_problem_000 auto_problem_001}"
PILOT_STEPS="${PILOT_STEPS:-3}"
RELIABILITY_TARGET="${RELIABILITY_TARGET:-0.90}"
MAX_DEPTH="${MAX_DEPTH:-3}"
LEVEL1_BRANCHING="${LEVEL1_BRANCHING:-5}"
DEEPER_BRANCHING="${DEEPER_BRANCHING:-3}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
MAX_STEPS="${MAX_STEPS:-9}"
ALPHA="${ALPHA:-5.0}"
GAMMA="${GAMMA:-0.99}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.8}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-design-bench}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mcts_ablation_${MODEL_TAG}_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-logs/posterior/mcts_inference_${MODEL_TAG}_${SLURM_JOB_ID}.jsonl}"

echo "API provider    : openai"
echo "Model config    : $MODEL_CONFIG"
echo "OpenAI model    : $OPENAI_MODEL"
echo "Reasoning effort: $OPENAI_REASONING_EFFORT"
echo "OpenAI concurrency: $OPENAI_CONCURRENCY"
echo "LLM call budget : ${LLM_CALL_BUDGET:-auto-estimate}"
echo "Problem IDs     : $PROBLEM_IDS"
echo "Pilot steps     : $PILOT_STEPS"
echo "Max depth       : $MAX_DEPTH"
echo "Candidate count : $CANDIDATE_COUNT"
echo "Max steps       : $MAX_STEPS"
echo "Output          : $OUTPUT_PATH"
echo "W&B project     : $WANDB_PROJECT / $WANDB_RUN_NAME"

# shellcheck disable=SC2086
python -u scripts/eval_mcts_inference.py \
  --api-provider openai \
  --model-config "$MODEL_CONFIG" \
  --openai-model "$OPENAI_MODEL" \
  --openai-reasoning-effort "$OPENAI_REASONING_EFFORT" \
  --openai-timeout-s "$OPENAI_TIMEOUT_S" \
  --openai-concurrency "$OPENAI_CONCURRENCY" \
  --llm-call-budget "$LLM_CALL_BUDGET" \
  --problems-dir "$PROBLEMS_DIR" \
  --problem-ids $PROBLEM_IDS \
  --pilot-steps "$PILOT_STEPS" \
  --reliability-target "$RELIABILITY_TARGET" \
  --max-depth "$MAX_DEPTH" \
  --level1-branching "$LEVEL1_BRANCHING" \
  --deeper-branching "$DEEPER_BRANCHING" \
  --candidate-count "$CANDIDATE_COUNT" \
  --max-steps "$MAX_STEPS" \
  --alpha "$ALPHA" \
  --gamma "$GAMMA" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --output "$OUTPUT_PATH"

echo "============================================================"
echo "OpenAI GPT-5.5 MCTS inference ablation completed: $(date)"
echo "Output: $OUTPUT_PATH"
echo "============================================================"

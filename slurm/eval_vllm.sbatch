#!/bin/bash
#SBATCH -J eval_vllm
#SBATCH -A mch250030p
#SBATCH -p GPU
#SBATCH -q gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=h100-80:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=13
#SBATCH -t 4:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/logs/eval_vllm_%j.out

# Serve a checkpoint (or the un-finetuned base) with vLLM and run the evaluator
# against it, with the CPU controls in the same script and on the same problems.
#
# A whole node: the GPU partition here refuses partial-node allocations, so the
# weights shard eight ways, which also makes serving fast enough to sweep domains.
#
# vLLM lives in its own environment -- my_env pins torch 2.4.0 and vLLM 0.29
# needs torch 2.13. The evaluator itself runs from my_env, which is where the
# simulator and the domain code live; the two talk over HTTP, which is the whole
# reason the evaluator speaks the OpenAI protocol.
set -uo pipefail
module load cuda/12.6.1
module load anaconda3
cd /ocean/projects/mch250030p/wxu7/llm_finetune

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.8-27B}
ARM_NAME=${ARM_NAME:-base}
ARMS=${ARMS:-base,fsd}
NPROB=${NPROB:-60}
CALLS=${CALLS:-8}
DOMAINS=${DOMAINS:-truss,synth,pipe,catalogue,cases}
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))

export HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models
export HF_HUB_CACHE=/ocean/projects/mch250030p/wxu7/hf_models
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore
export VLLM_LOGGING_LEVEL=WARNING

echo "serving $MODEL_PATH on port $PORT"
source activate /ocean/projects/mch250030p/wxu7/envs/vllm_env
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_PATH" \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --disable-log-requests \
  --port "$PORT" > logs/vllm_server_${SLURM_JOB_ID}.log 2>&1 &
SERVER=$!

# Wait for the server rather than sleeping a guessed interval: a 53 GB load off
# Lustre is not a predictable duration.
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "server up after ${i}0s"; break
  fi
  if ! kill -0 $SERVER 2>/dev/null; then
    echo "SERVER DIED — last 40 lines:"; tail -40 logs/vllm_server_${SLURM_JOB_ID}.log; exit 1
  fi
  sleep 10
done

conda deactivate 2>/dev/null || true
source activate my_env
export PYTHONWARNINGS=ignore

for D in ${DOMAINS//,/ }; do
  echo "################ $D"
  python scripts/eval_distilled.py \
    --domain "$D" \
    --n "$NPROB" \
    --calls "$CALLS" \
    --arms "$ARMS" \
    --model "$MODEL_PATH" \
    --base-model "$MODEL_PATH" \
    --base-url "http://localhost:${PORT}/v1" \
    --workers 8
done

kill $SERVER 2>/dev/null
echo "done"

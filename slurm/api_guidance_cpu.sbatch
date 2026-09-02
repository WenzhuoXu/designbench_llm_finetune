#!/bin/bash
#SBATCH --job-name=api_guide
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/slurm/%x_%j.err
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
export OMP_NUM_THREADS=1
export AWS_REGION=us-west-2
# credential read from a 600-perm file in $HOME, never written into the repo or logs
export AWS_BEARER_TOKEN_BEDROCK="$(cat "$HOME/.bedrock_token")"

echo "--- connectivity check from compute node ---"
python -u -c "
import os,requests
r=requests.post(f'https://bedrock-runtime.{os.environ[\"AWS_REGION\"]}.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse',
  headers={'Authorization':'Bearer '+os.environ['AWS_BEARER_TOKEN_BEDROCK'],'Content-Type':'application/json'},
  json={'messages':[{'role':'user','content':[{'text':'Reply OK'}]}],'inferenceConfig':{'maxTokens':8}},timeout=60)
print('bedrock status', r.status_code)
r.raise_for_status()
"
echo "--- experiment ---"
python -u scripts/api_potential_guidance.py \
  --n "${N:-30}" --max-steps "${MAX_STEPS:-20}" --workers "${WORKERS:-8}" \
  --out "${OUT:-results/api_guidance/hard_eval.jsonl}"

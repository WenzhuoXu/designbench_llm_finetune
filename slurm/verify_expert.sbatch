#!/bin/bash
source ~/.bashrc
module load anaconda3
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/verify_expert.py \
  /ocean/projects/mch250030p/wxu7/llm_finetune/data/expert/expert_traces_0_150.jsonl 44 2>/dev/null

echo
echo "--- token / size profile of the emitted file ---"
python scripts/profile_expert.py \
  /ocean/projects/mch250030p/wxu7/llm_finetune/data/expert/expert_traces_0_150.jsonl 2>/dev/null
echo DONE

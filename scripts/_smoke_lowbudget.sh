#!/bin/bash
source ~/.bashrc >/dev/null 2>&1
module load anaconda3 >/dev/null 2>&1
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
python scripts/da_lowbudget.py --n 6 --budgets 40 --workers 6 \
  --out /ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/da_lowbudget_smoke.jsonl 2>&1

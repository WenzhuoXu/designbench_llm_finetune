#!/bin/bash
source ~/.bashrc >/dev/null 2>&1
module load anaconda3 >/dev/null 2>&1
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
exec python -u scripts/da_lowbudget.py --n 60 --start 0 --workers 14 \
  --budgets 5,10,20,40,80 \
  --out /ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/da_lowbudget60.jsonl

#!/bin/bash
source ~/.bashrc
module load anaconda3
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
python -u scripts/da_planner_final.py \
    --start 150 --n 60 --workers 16 --max-steps 8 --resume \
    --out /ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/da_planner_final_smoke.jsonl

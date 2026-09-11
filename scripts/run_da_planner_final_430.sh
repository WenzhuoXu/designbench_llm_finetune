#!/bin/bash
# Full held-out run: DesignBench problems 150-580 (430 problems), 3 arms, 1290 episodes.
# Bedrock is used, so this must run on the LOGIN node (compute nodes have no internet).
# Launch detached:
#   cd /ocean/projects/mch250030p/wxu7/llm_finetune && \
#   setsid nohup bash scripts/run_da_planner_final_430.sh \
#       > logs/da_planner_final_430.log 2>&1 < /dev/null &
# --resume makes it restartable: rerun the same line and it skips finished (problem_id, arm).
# Expected wall time ~70 min at 16 workers (smoke: 180 episodes in 581 s).
source ~/.bashrc
module load anaconda3
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
python -u scripts/da_planner_final.py \
    --start 150 --n 430 --workers 16 --max-steps 8 --resume \
    --out /ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/da_planner_final_430.jsonl

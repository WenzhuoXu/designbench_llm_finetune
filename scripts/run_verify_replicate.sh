#!/bin/bash
# Adversarial verification of da_planner_final smoke.
#  A) independent replication of the 3-arm smoke in a fresh process (temperature 0.7, unseeded)
#  B) plain_llm at horizon 4, same 60 problems, to test the "horizon 8 is charitable" claim
source ~/.bashrc
module load anaconda3
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
R=/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance

echo "=================== A: replication, 3 arms, horizon 8 ==================="
python -u scripts/da_planner_final.py \
    --start 150 --n 60 --workers 16 --max-steps 8 \
    --out $R/verify_replicate_h8.jsonl

echo "=================== B: plain_llm only, horizon 4 ==================="
python -u scripts/da_planner_final.py \
    --start 150 --n 60 --workers 16 --max-steps 4 --arms plain_llm \
    --out $R/verify_plain_llm_h4.jsonl
echo "=================== DONE ==================="

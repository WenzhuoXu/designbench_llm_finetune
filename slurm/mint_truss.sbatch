#!/bin/bash
#SBATCH -J mint_truss
#SBATCH -A mch250030p
#SBATCH -p RM-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 04:00:00
#SBATCH -o /ocean/projects/mch250030p/wxu7/llm_finetune/results/mint_truss_%j.log
# 10,000 hard truss problems into a CLEAN directory. The first attempt wrote to problems_gen,
# candidates tried: the id is f"{prefix}_{accepted:04d}", so one pre-existing file deadlocks
# the run -- the counter never advances past a collision. The output dir must be UNTOUCHED.
# Even strata mix so the corpus spans the full difficulty range.
cd /ocean/projects/mch250030p/wxu7/llm_finetune
source ~/.bashrc
module load anaconda3
conda activate my_env
python scripts/generate_truss_problems.py \
  --n 10000 \
  --mode hard \
  --strata s1:1,s2:1,s3:1,s4:1,s5:1 \
  --require-policy-solve any \
  --workers 46 \
  --seed 20260911 \
  --prefix genc_problem \
  --out /ocean/projects/mch250030p/wxu7/DesignBench/data/problems_gen_c

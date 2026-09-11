#!/bin/bash
source ~/.bashrc
module load anaconda3
conda activate my_env
cd /ocean/projects/mch250030p/wxu7/llm_finetune
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "cores available: $(nproc)  SLURM_NTASKS=${SLURM_NTASKS:-unset}  SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"
PROCS=${PROCS:-44}

echo "=============================================================="
echo "SMOKE: problems 0-60"
echo "=============================================================="
python scripts/gen_expert.py --start 0 --n 60 --procs $PROCS --steps 12 --nprop 32 \
  --out /ocean/projects/mch250030p/wxu7/llm_finetune/data/expert/smoke60.jsonl 2>/dev/null

echo
echo "=============================================================="
echo "FULL: problems 0-150 (discovery split)"
echo "=============================================================="
python scripts/gen_expert.py --start 0 --n 150 --procs $PROCS --steps 12 --nprop 32 \
  --out /ocean/projects/mch250030p/wxu7/llm_finetune/data/expert/expert_traces_0_150.jsonl 2>/dev/null

echo
echo "DONE"
ls -la /ocean/projects/mch250030p/wxu7/llm_finetune/data/expert/

#!/bin/bash
#SBATCH --job-name=bat_oracle
#SBATCH --partition=RM-shared
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/oracle_%j.out
#SBATCH --error=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder/oracle_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=/ocean/projects/mch250030p/wxu7/llm_finetune:/ocean/projects/mch250030p/wxu7/DesignBench
BP=/ocean/projects/mch250030p/wxu7/envs/battery/bin/python
S=/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/battery_oracle_d2.py
D=/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder
$BP $S 12 std    12 > $D/oracle_std_12.log 2>&1 &
$BP $S 16 std    12 > $D/oracle_std_16.log 2>&1 &
$BP $S 8  std    12 > $D/oracle_std_8.log 2>&1 &
$BP $S 8  dense  12 > $D/oracle_dense_8.log 2>&1 &
$BP $S 12 dense  12 > $D/oracle_dense_12.log 2>&1 &
$BP $S 12 thin   12 > $D/oracle_thin_12.log 2>&1 &
$BP $S 8  thin   12 > $D/oracle_thin_8.log 2>&1 &
$BP $S 8  coarse 12 > $D/oracle_coarse_8.log 2>&1 &
$BP $S 12 coarse 12 > $D/oracle_coarse_12.log 2>&1 &
$BP $S 16 coarse 12 > $D/oracle_coarse_16.log 2>&1 &
wait
echo ALLDONE

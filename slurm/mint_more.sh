#!/bin/bash
# Four more clean mints (40,000 truss problems). Each needs its OWN untouched
# output dir and prefix: the generator derives the problem id from the accepted
# counter, so a single pre-existing file deadlocks the run.
cd /ocean/projects/mch250030p/wxu7/llm_finetune
DATA=/ocean/projects/mch250030p/wxu7/DesignBench/data
i=0
for k in d e f g; do
  i=$((i + 1))
  seed=$((20260920 + i))
  dst=$DATA/problems_gen_$k
  if [ -e "$dst" ]; then
    echo "DIR EXISTS $dst -- skipping"
    continue
  fi
  sed -e "s|problems_gen_c|problems_gen_$k|" \
      -e "s|genc_problem|gen${k}_problem|" \
      -e "s|--seed 20260911|--seed $seed|" \
      -e "s|^#SBATCH -p RM-small\$|#SBATCH -p RM-small\n#SBATCH -q low|" \
      -e "s|-J mint_truss|-J mint_$k|" \
      slurm/mint_truss.sbatch > slurm/mint_$k.sbatch
  sbatch slurm/mint_$k.sbatch
done

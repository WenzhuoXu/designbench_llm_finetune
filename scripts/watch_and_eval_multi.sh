#!/bin/bash
# Submit held-out evaluations as checkpoints appear, for several arms and steps.
set -u
SPLIT_FILE=/ocean/projects/mch250030p/wxu7/llm_finetune/data/splits/truss_v1_auto.json
cd /ocean/projects/mch250030p/wxu7/llm_finetune
ARMS="t5a_phi1_0823f t5b_phi2_0823f t5c_lookahead_0823f t5ao_phionly_v1_0823f t5bo_phionly_v2_0823f t5co_lookahead_only_0823 t5bw_informative_0823 t7_dpo_init_0823"
STEPS="${STEPS:-75 100}"
declare -A DONE
deadline=$(( $(date +%s) + 30*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  for a in $ARMS; do
    for ck in $STEPS; do
      key="${a}_${ck}"
      [ -n "${DONE[$key]:-}" ] && continue
      d="checkpoints/grpo/${a}/checkpoint-${ck}"
      if [ -d "$d" ] && [ -f "$d/adapter_config.json" ]; then
        name="${a%_0823*}_ckpt${ck}_eval"
        sbatch --export="ALL,CHECKPOINT=$d,RUN_NAME=$name,SPLIT_FILE=$SPLIT_FILE,SPLIT=eval" \
               slurm/eval_ablation_v2.sbatch >/dev/null 2>&1 \
          && echo "$(date +%H:%M) submitted $name"
        DONE[$key]=1
      fi
    done
  done
  sleep 300
done

#!/bin/bash
# Submit a held-out evaluation as soon as each arm writes a given checkpoint.
# Waiting for a run to finish before evaluating wastes hours per arm; the
# champion converged by step 50, so ckpt50 is the first fair comparison point.
set -u
CKPT_STEP="${1:-50}"
SPLIT_FILE=/ocean/projects/mch250030p/wxu7/llm_finetune/data/splits/truss_v1_auto.json
cd /ocean/projects/mch250030p/wxu7/llm_finetune
ARMS="t5a_phi1_0823f t5b_phi2_0823f t5c_lookahead_0823f t5ao_phionly_v1_0823f t5bo_phionly_v2_0823f t5co_lookahead_only_0823"
declare -A DONE
deadline=$(( $(date +%s) + 20*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  pending=0
  for a in $ARMS; do
    [ -n "${DONE[$a]:-}" ] && continue
    d="checkpoints/grpo/${a}/checkpoint-${CKPT_STEP}"
    if [ -d "$d" ] && [ -f "$d/adapter_config.json" ]; then
      name="${a%_0823*}_ckpt${CKPT_STEP}_eval"
      sbatch --export="ALL,CHECKPOINT=$d,RUN_NAME=$name,SPLIT_FILE=$SPLIT_FILE,SPLIT=eval" \
             slurm/eval_ablation_v2.sbatch >/dev/null 2>&1 \
        && echo "$(date +%H:%M) submitted eval for $a ckpt${CKPT_STEP} -> $name"
      DONE[$a]=1
    else
      pending=$((pending+1))
    fi
  done
  [ "$pending" -eq 0 ] && { echo "all arms evaluated at ckpt${CKPT_STEP}"; break; }
  sleep 300
done

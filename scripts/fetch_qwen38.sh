#!/bin/bash
# Qwen3.8-27B: dense 26.4B, ~53 GB bf16. Compute nodes have no internet, so this
# runs on the login node.
#
# Progress bars are OFF. When the project hit its storage quota, the failure did
# not surface as a download error -- it surfaced as tqdm raising OSError while
# writing the progress bar into this log, which left the process alive but
# wedged at zero bytes per minute. No progress bar, no such failure mode.
#
# No --exclude either: `hf download REPO [FILES...]` treats anything after the
# flag's first value as an explicit filename list, which silently turned the
# first attempt into "fetch these two glob-named files" and downloaded nothing.
set -euo pipefail
module load anaconda3
source activate my_env
export HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models
export HF_HUB_CACHE=/ocean/projects/mch250030p/wxu7/hf_models
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /ocean/projects/mch250030p/wxu7/llm_finetune
hf download Qwen/Qwen3.8-27B >> logs/fetch_qwen38.log 2>&1
echo "DONE $(date) $(du -sh $HF_HOME/models--Qwen--Qwen3.8-27B | cut -f1)" >> logs/fetch_qwen38.log

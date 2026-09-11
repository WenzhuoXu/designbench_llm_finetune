#!/bin/bash
# A serving environment, separate from my_env on purpose.
#
# my_env pins torch 2.4.0, and any vLLM new enough to know the qwen3_5
# architecture needs a much newer torch. Upgrading in place would put the
# training runs at risk for the sake of evaluation, so serving gets its own
# prefix. It lives in project space because the home quota is far too small for
# a vLLM install plus its CUDA wheels.
set -euo pipefail
module load anaconda3
module load cuda/12.6.1

ENV=/ocean/projects/mch250030p/wxu7/envs/vllm_env
LOG=/ocean/projects/mch250030p/wxu7/llm_finetune/logs/make_vllm_env.log
mkdir -p "$(dirname "$ENV")"

{
  echo "=== $(date) creating $ENV"
  conda create -y -p "$ENV" python=3.12
  source activate "$ENV"
  python -m pip install --upgrade pip
  python -m pip install vllm
  echo "=== installed, checking architecture support"
  python - <<'PY'
import vllm
print("vllm", vllm.__version__)
import torch
print("torch", torch.__version__)
from vllm.model_executor.models import registry as R
names = set()
for attr in ("_VLLM_MODELS", "_TEXT_GENERATION_MODELS", "_MULTIMODAL_MODELS"):
    d = getattr(R, attr, None)
    if isinstance(d, dict):
        names |= set(d)
print("architectures registered:", len(names))
print("qwen3 family:", sorted(n for n in names if "qwen3" in n.lower()))
print("Qwen3_5ForConditionalGeneration supported:",
      "Qwen3_5ForConditionalGeneration" in names)
PY
  echo "=== DONE $(date)"
} >> "$LOG" 2>&1

#!/bin/bash
# One vLLM server for Qwen3.8-27B on the GPU named by CUDA_VISIBLE_DEVICES (set by the caller),
# listening on 127.0.0.1:$1. VLLM_ENV picks the venv (default vllm029: vllm 0.29.0, torch 2.13.0+cu130);
# VLLM_EXTRA appends flags. --limit-mm-per-prompt and --max-num-seqs 64 are the flags the drc
# server of this model runs with (rows/logs/serve3.log there): no vision encoder, 64 streams.
# The compute nodes have no ninja/nvcc, so FlashInfer must not JIT-compile anything: the sampler
# falls back to PyTorch (VLLM_USE_FLASHINFER_SAMPLER=0) and the GDN prefill kernel to Triton
# (--gdn-prefill-backend triton); the venv's bin (with pip's ninja) is put on PATH as a safety net.
port=${1:?port}
ENV=${VLLM_ENV:-/project/community/wxu2/envs/vllm029}
export PATH="$ENV/bin:$PATH"
export HF_HOME=/project/community/wxu2/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_USE_FLASHINFER_SAMPLER=0
SNAP=/project/community/wxu2/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
echo "serve_one: gpu=${CUDA_VISIBLE_DEVICES:-?} port=$port env=$ENV $(date -u +%FT%TZ)"
exec "$ENV/bin/vllm" serve "$SNAP" \
  --served-model-name Qwen/Qwen3.8-27B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --max-num-seqs 64 \
  --gdn-prefill-backend triton \
  --host 127.0.0.1 --port "$port" \
  ${VLLM_EXTRA:-}

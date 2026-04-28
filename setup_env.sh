#!/bin/bash
# ============================================================
# DesignBench LLM Fine-Tuning — Environment Setup
# Creates the 'my_env' conda environment with all dependencies.
# Run once on the login node or in an interactive GPU session.
#
# Usage:
#   bash setup_env.sh
#   bash setup_env.sh --with-flash-attn   # also install flash-attn (needs GPU node)
#   bash setup_env.sh --with-vllm         # also install vLLM (needs GPU node)
# ============================================================

set -euo pipefail

WITH_FLASH_ATTN=0
WITH_VLLM=0
for arg in "$@"; do
    case "$arg" in
        --with-flash-attn) WITH_FLASH_ATTN=1 ;;
        --with-vllm) WITH_VLLM=1 ;;
    esac
done

echo "============================================================"
echo "Setting up 'my_env' conda environment"
echo "HF model cache: /ocean/projects/mch250030p/wxu7/hf_models"
echo "============================================================"

# Load modules
module load cuda/12.6.1
module load anaconda3/2024.10-1

# ─── Create environment ────────────────────────────────────────
if conda env list | grep -q "^my_env "; then
    echo "Environment 'my_env' already exists — updating packages"
else
    echo "Creating 'my_env' environment (Python 3.11) ..."
    conda create -n my_env python=3.11 -y
fi

# ─── Activate and install ─────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate my_env

echo "Python: $(which python)"
echo "Installing PyTorch 2.4 with CUDA 12.1 ..."
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

echo "Installing HuggingFace stack ..."
pip install \
    transformers>=4.47.0 \
    datasets>=2.20.0 \
    tokenizers>=0.19.0 \
    accelerate>=1.0.0 \
    peft>=0.13.0 \
    huggingface_hub>=0.24.0 \
    --quiet

echo "Installing TRL >= 0.12 ..."
pip install "trl>=0.12.0" --quiet

echo "Installing DeepSpeed ..."
DS_BUILD_OPS=0 pip install "deepspeed>=0.15.0" --quiet

echo "Installing logging and config packages ..."
pip install \
    wandb>=0.18.0 \
    pynvml>=11.5.0 \
    hydra-core>=1.3.2 \
    omegaconf>=2.3.0 \
    rich>=13.7.0 \
    --quiet

echo "Installing scientific packages ..."
pip install numpy scipy pandas --quiet

echo "Installing dev packages ..."
pip install pytest pytest-cov black ruff ipython --quiet

# ─── Optional: Flash Attention 2 ──────────────────────────────
if [ "$WITH_FLASH_ATTN" -eq 1 ]; then
    echo "Installing Flash Attention 2 (requires CUDA build tools) ..."
    echo "NOTE: This must be run on a GPU node (not login node)"
    pip install flash-attn>=2.6.0 --no-build-isolation
    echo "Flash Attention 2 installed"
else
    echo ""
    echo "Flash Attention 2 not installed (use --with-flash-attn on a GPU node):"
    echo "  srun --partition=GPU --account=mch250030p --gpus-per-node=h100-80:1 --pty bash"
    echo "  module load cuda/12.6.1 anaconda3/2024.10-1 && conda activate my_env"
    echo "  pip install flash-attn --no-build-isolation"
fi

# ─── Optional: vLLM ───────────────────────────────────────────
if [ "$WITH_VLLM" -eq 1 ]; then
    echo "Installing vLLM (requires GPU node for CUDA wheel) ..."
    pip install vllm>=0.6.0
    echo "vLLM installed"
else
    echo ""
    echo "vLLM not installed (use --with-vllm on a GPU node):"
    echo "  pip install vllm>=0.6.0"
fi

# ─── Configure HuggingFace cache ─────────────────────────────
mkdir -p /ocean/projects/mch250030p/wxu7/hf_models
echo ""
echo "Configuring HuggingFace cache → /ocean/projects/mch250030p/wxu7/hf_models"
# Add to .bashrc if not already there
BASHRC="${HOME}/.bashrc"
if ! grep -q "HF_HOME.*hf_models" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# DesignBench LLM training — HuggingFace cache" >> "$BASHRC"
    echo "export HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models" >> "$BASHRC"
    echo "export TRANSFORMERS_CACHE=/ocean/projects/mch250030p/wxu7/hf_models" >> "$BASHRC"
    echo "HF_HOME added to $BASHRC"
fi

# ─── Install llm_finetune package ─────────────────────────────
echo ""
echo "Installing llm_finetune package (editable) ..."
cd "$(dirname "$0")"
pip install -e . --quiet
echo "llm_finetune installed"

# ─── Run quick smoke tests ─────────────────────────────────────
echo ""
echo "Running smoke tests (no GPU needed) ..."
python -m pytest tests/test_rewards.py tests/test_targets.py -v --tb=short -q 2>&1 | tail -20

echo ""
echo "============================================================"
echo "Environment setup complete!"
echo ""
echo "Activate with:  conda activate my_env"
echo "Run tests with: pytest tests/ -v"
echo "Train SFT:      sbatch slurm/sft_h100.sbatch"
echo "============================================================"

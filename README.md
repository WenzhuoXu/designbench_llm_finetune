# DesignBench LLM Fine-Tuning Infrastructure

Research-grade SFT + GRPO RL training for 10B–20B LLMs on the [DesignBench](../DesignBench) benchmark:
iterative structural truss design using Tree-of-Thought reasoning.

## Features

- **SFT warmstart** on DesignBench gold traces (5,000 examples, multi-turn chat)
- **GRPO** (Group Relative Policy Optimization) with FEA-based rewards
- **vLLM** fast rollout generation (5–10× faster than HF generate, tensor_parallel=8)
- **Flash Attention 2** + `torch.compile` for H100 throughput
- **DeepSpeed ZeRO-2/3** for 14B+ model training across 8×H100
- **Research hooks**: pluggable SFT targets, reward/cost functions, MCTS data structures
- **Advanced logging**: W&B (metrics, rollout tables, GPU stats) + Rich console + JSONL

## Quick Start

```bash
cd /ocean/projects/mch250030p/wxu7/llm_finetune

# 1. Verify data pipeline
python scripts/prepare_data.py --model qwen3_14b --dry-run --num-examples 5

# 2. SFT training (local smoke test)
python scripts/train_sft.py model=qwen3_14b sft.max_steps=10 logging.wandb_enabled=false

# 3. Submit SFT job to H100 cluster
sbatch slurm/sft_h100.sbatch

# 4. Submit GRPO job (starting from SFT checkpoint)
CHECKPOINT=checkpoints/sft/qwen3_14b_sft_001/final sbatch slurm/grpo_h100.sbatch

# 5. Evaluate checkpoint
python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/sft/qwen3_14b_sft_001/final \
  --model-config configs/model/qwen3_14b.yaml
```

## Research Hooks

| Hook | File | Config |
|---|---|---|
| SFT supervision target | `llm_finetune/training/sft/targets.py` | `data.target_fn: action_only` |
| RL reward function | `llm_finetune/training/rl/rewards.py` | `rl.reward_fn: composite` |
| RL cost function | `llm_finetune/training/rl/costs.py` | `rl.cost_fn: constraint_violation` |
| Learned PRM/ORM | `llm_finetune/training/rl/reward_models.py` | (manual instantiation) |
| MCTS data structure | `llm_finetune/data/datasets/mcts_dataset.py` | `data=mcts_trees` |
| Chat template | `llm_finetune/data/processors/chat_formatter.py` | (per-model automatic) |

## Project Structure

```
llm_finetune/
├── .claude/CLAUDE.md              # Agent memory (read this for full context)
├── configs/                       # Hydra YAML configs (composable)
│   ├── model/{qwen3_14b,...}.yaml
│   ├── sft/{base,warmstart}.yaml
│   ├── rl/{grpo_base,grpo_truss}.yaml
│   └── data/{sft_traces,mcts_trees}.yaml
├── llm_finetune/
│   ├── data/                      # Datasets, processors, collators
│   ├── models/                    # Model loader (FA2, compile, LoRA) + vLLM server
│   ├── training/sft/              # SFTTrainer + SFTTarget hooks
│   ├── training/rl/               # GRPOTrainer + reward/cost/reward_model hooks
│   ├── envs/                      # TrussRolloutEnv (FEA execution)
│   └── logging/                   # LocalLogger + WandbLogger
├── scripts/                       # Entrypoints: prepare_data, train_sft, train_grpo, eval_checkpoint
├── slurm/                         # SLURM job scripts for H100 cluster
├── deepspeed/                     # ZeRO-2 and ZeRO-3 configs
└── tests/                         # pytest tests (no model download required for most)
```

## Configuration

The config system uses Hydra. Compose configs from components:

```bash
# SFT with thinking_and_action supervision, LoRA, no W&B
python scripts/train_sft.py \
  model=deepseek_r1_14b \
  sft=warmstart \
  data=sft_traces \
  data.target_fn=thinking_and_action \
  model.use_lora=true \
  logging.wandb_enabled=false

# GRPO with custom reward weights and large group size
python scripts/train_grpo.py \
  model=qwen3_14b \
  rl=grpo_truss \
  "rl.reward_weights.feasibility=2.0" \
  "rl.reward_weights.mass_reduction=0.5" \
  rl.group_size=16 \
  rl.kl_coef=0.02
```

## Adding New Reward Functions

```python
# In llm_finetune/training/rl/rewards.py:

class MyCustomReward(RewardFunction):
    def compute(self, rollout: RolloutResult, problem_spec: dict) -> float:
        # Your research idea here
        ...
    def name(self) -> str:
        return "my_reward"

# Register it:
REWARD_REGISTRY["my_reward"] = MyCustomReward

# Use via config:
# rl.reward_fn: my_reward
# Or as component in composite: rl.reward_weights.my_reward: 0.3
```

## Adding New SFT Targets

```python
# In llm_finetune/training/sft/targets.py:

class MyTarget(SFTTarget):
    def get_loss_mask(self, input_ids, labels, tokenizer) -> torch.Tensor:
        # 1 = compute loss here, 0 = ignore
        ...
    def transform_example(self, example, tokenizer) -> dict:
        # Optionally transform examples before tokenization
        return example

# Register:
TARGET_REGISTRY["my_target"] = MyTarget
# Use: data.target_fn: my_target
```

## Hardware Requirements

- 8× NVIDIA H100 80GB (per node) — required for 14B models with sequence packing
- CUDA 12.6.1
- ~512 GB system RAM (for DeepSpeed CPU offload)
- Fast storage for HF model cache: `/ocean/projects/mch250030p/wxu7/hf_models`

## Dependencies

```bash
# Core (via requirements.txt)
pip install torch>=2.4 transformers>=4.47 trl>=0.12 accelerate>=1.0 deepspeed>=0.15

# Flash Attention 2 (required for H100)
pip install flash-attn>=2.6 --no-build-isolation

# vLLM for fast rollouts (GRPO)
pip install vllm>=0.6

# Full install
pip install -e ".[all]"
```

## Running Tests

```bash
# All tests (reward/target tests don't need model downloads)
pytest tests/ -v

# Fast tests only (no FEA/model):
pytest tests/test_rewards.py tests/test_targets.py tests/test_data.py -v

# With FEA tests (requires DesignBench environment):
pytest tests/test_env.py -v
```

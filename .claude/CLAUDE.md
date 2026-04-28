# llm_finetune — Agent Context

## Project Purpose

This is a research-grade SFT + GRPO RL training framework for fine-tuning
10B–20B open-source LLMs on the **DesignBench** benchmark: iterative truss
structural engineering design using Tree-of-Thought reasoning.

The framework is built for research experimentation. The key design principle:
all research-variable components are exposed as **pluggable hooks** — you can
swap reward functions, SFT targets, cost functions, data loaders, and MCTS
sampling strategies without touching training infrastructure.

## Repository Location

```
/ocean/projects/mch250030p/wxu7/llm_finetune/
```

## Related Repositories

- **DesignBench**: `/ocean/projects/mch250030p/wxu7/DesignBench/`
  - `data/sft/train.jsonl` — 5,000 SFT training examples (multi-turn with <think> + FEA feedback)
  - `data/sft/dev.jsonl` — 5,000 SFT dev examples
  - `data/modification_trees/` — MCTS-like tree structures
  - `data/problems/` — 20 truss design problem specs
  - `validation/truss_executor.py` — FEA execution (used by TrussRolloutEnv)
  - `truss_tot/data_structures.py` — ReasoningTree, SolutionTrace classes

## HPC Configuration (Bridges-2 Cluster)

- **Account**: `mch250030p`
- **Partition**: `GPU`
- **QOS**: `gpu`
- **Hardware**: 8×H100-80GB per node, 104 CPU cores per node
- **CUDA**: 12.6.1
- **Conda env**: `my_env`
- **HF cache**: `/ocean/projects/mch250030p/wxu7/hf_models`
- **SLURM jobs**: `slurm/sft_h100.sbatch`, `slurm/grpo_h100.sbatch`

## Target Models (from DesignBench evaluation)

| Name (config) | HF ID | Size | Thinking |
|---|---|---|---|
| `qwen3_14b` | Qwen/Qwen3-14B | 14.8B | Yes |
| `deepseek_r1_14b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 14.0B | Yes |
| `phi4_reasoning` | microsoft/Phi-4-reasoning | 14.0B | Yes |
| `llama4_scout` | meta-llama/Llama-4-Scout-17B-16E-Instruct | 17.0B MoE | No |
| `phi4` | microsoft/phi-4 | 14.0B | No |
| `gemma3_12b` | google/gemma-3-12b-it | 12.0B | No |
| `qwen25_14b` | Qwen/Qwen2.5-14B-Instruct | 14.0B | No |

## Key Technology Stack

- **Training**: TRL ≥ 0.12 (SFTTrainer, GRPOTrainer)
- **Parallelism**: DeepSpeed ZeRO-2/3 + Accelerate
- **Fast rollouts**: vLLM (TRL VLLMClient, tensor_parallel=8)
- **Attention**: Flash Attention 2 (mandatory for H100)
- **Compilation**: torch.compile (inductor backend, ~20% speedup)
- **Logging**: W&B (project: `designbench-training`) + Rich console + JSONL
- **GPU monitoring**: pynvml (per-GPU util, memory, temp, power)
- **Config**: Hydra + OmegaConf YAML composition

## Research Hooks (Primary Research Variables)

These are the files to modify for research experiments:

### SFT Supervision Target
**File**: `llm_finetune/training/sft/targets.py`

```python
class SFTTarget(ABC):
    def get_loss_mask(input_ids, labels, tokenizer) → Tensor
    def transform_example(example, tokenizer) → dict
```

Built-ins: `full_sequence`, `action_only`, `thinking_and_action`, `final_answer`
Config: `data.target_fn: action_only` in YAML or CLI override

### RL Reward Functions
**File**: `llm_finetune/training/rl/rewards.py`

```python
class RewardFunction(ABC):
    def compute(rollout: RolloutResult, problem_spec: dict) → float
```

Built-ins: `feasibility`, `fos_improvement`, `mass_reduction`, `grammar_compliance`,
          `step_efficiency`, `progress`, `composite`
Config: `rl.reward_fn: composite` + `rl.reward_weights`

### RL Cost Functions
**File**: `llm_finetune/training/rl/costs.py`

```python
class CostFunction(ABC):
    def compute(rollout: RolloutResult, problem_spec: dict) → float
```

Built-ins: `token_budget`, `constraint_violation`, `computational`, `repetition`
Config: `rl.cost_fn: composite` + `rl.cost_weights`

### Learned Reward Models (PRM/ORM)
**File**: `llm_finetune/training/rl/reward_models.py`

Skeletons: `ProcessRewardModel`, `OutcomeRewardModel`, `CostModel`
Training data: DesignBench `scripts/build_prm_labels.py`

### MCTS Data Structures
**File**: `llm_finetune/data/datasets/mcts_dataset.py`

Key methods to override:
- `MCTSDataset.build_sample_text(sample)` — state text representation
- `MCTSDataset.compute_target_value(sample)` — value function definition
- `MCTSDataset.sample_curriculum(n)` — curriculum ordering

Sampling strategies: `flat`, `path`, `subtree`, `curriculum`

## Quick Commands

```bash
cd /ocean/projects/mch250030p/wxu7/llm_finetune

# Verify data pipeline (dry run, no model download needed... well tokenizer needed)
python scripts/prepare_data.py --model qwen3_14b --dry-run --num-examples 5

# SFT training (local, 1 GPU)
python scripts/train_sft.py model=qwen3_14b sft=warmstart sft.max_steps=10

# SFT training (8×H100)
sbatch slurm/sft_h100.sbatch

# GRPO training (8×H100, from SFT checkpoint)
CHECKPOINT=checkpoints/sft/qwen3_sft/final sbatch slurm/grpo_h100.sbatch

# Evaluation
python scripts/eval_checkpoint.py --checkpoint checkpoints/sft/final --model-config configs/model/qwen3_14b.yaml

# Run tests
pytest tests/ -v

# Run reward tests only (no model needed)
pytest tests/test_rewards.py tests/test_targets.py -v
```

## Config System (Hydra YAML)

```bash
# Override any config field:
python scripts/train_sft.py \
  model=qwen3_14b \                    # model config
  sft=warmstart \                      # SFT hyperparams
  data=sft_traces \                    # data config
  data.target_fn=action_only \         # research hook: SFT target
  sft.max_steps=100 \                  # quick test
  model.use_compile=true \             # torch.compile
  logging.wandb_enabled=false          # disable W&B

# GRPO overrides:
python scripts/train_grpo.py \
  model=qwen3_14b \
  rl=grpo_truss \
  rl.reward_fn=composite \
  "rl.reward_weights.feasibility=2.0" \  # boost feasibility weight
  rl.group_size=16 \                   # more rollouts per prompt
  rl.kl_coef=0.01 \                    # looser KL
  rl.use_vllm=false                    # disable vLLM (fallback to HF generate)
```

## W&B Dashboard

- **Training project**: `designbench-training`
- **Evaluation project**: `design-bench` (same as DesignBench validation)
- Key metrics to watch:
  - `train/loss`, `train/perplexity` (SFT)
  - `train/reward`, `train/policy_loss`, `train/kl` (GRPO)
  - `eval/feasibility_rate` (both)
  - `system/gpu_mean_util_pct`, `system/gpu_mean_mem_pct` (GPU utilization)
  - `rollouts/table` (per-rollout breakdown)

## Data Format (DesignBench SFT JSONL)

`DesignBench/data/sft/{train,dev}.jsonl` — pre-built multi-turn conversations
with `<think>` tags and step-level FEA feedback. This is the ONLY supported
data source.

```json
{
  "problem_id": "auto_problem_079",
  "trace_id": "auto_problem_079_trace_287",
  "trace_quality": 1.0,
  "messages": [
    {"role": "system", "content": "PROBLEM: ...\nINITIAL STRUCTURE: ..."},
    {"role": "system", "content": "INITIAL STATE ANALYSIS:\nMass: 213.21..."},
    {"role": "assistant", "content": "<think>Modification: ...\nAction: SCALE_PARAM(...)</think>"},
    {"role": "system", "content": "STRUCTURAL ANALYSIS RESULT:\nMass: 229.89..."},
    ...
    {"role": "assistant", "content": "<answer>Optimal feasible solution found via LP</answer>"}
  ]
}
```

The `WarmstartReasoningTarget` reformats this into the inference-time format:
- Merges system messages [0]+[1] into one system prompt
- Remaps subsequent system (FEA) messages → user role with `[Simulation Result]` prefix
- Extracts grammar actions from inside `<think>` into `<action>...</action>` tags after `</think>`

## Grammar Actions

| Action | Example |
|---|---|
| `SCALE_PARAM` | `SCALE_PARAM(3, radius, 1.15)` |
| `SCALE_MULTI_PARAM` | `SCALE_MULTI_PARAM([1,2], [radius:1.1, thickness:0.9])` |
| `ADD_MEMBER` | `ADD_MEMBER(0, 5, 6061_T6_Aluminum, Pipe, 0.03, 0.004)` |
| `MODIFY_PARAM` | `MODIFY_PARAM(3, radius, 0.030, 0.035)` |
| `REMOVE_MEMBER` | `REMOVE_MEMBER(4)` |
| `MOVE_JOINT` | `MOVE_JOINT(3, [0.0, 2.0], [0.5, 2.0])` |

## Design Constraints (DesignBench Truss)

- FOS_buckling ≥ 1.5 (Factor of Safety against buckling)
- FOS_yielding ≥ 1.5 (Factor of Safety against yielding)
- mass ≤ maximum_mass (structural mass constraint)
- Material: 6061 T6 Aluminum, Shape: Pipe (r, t parameters)

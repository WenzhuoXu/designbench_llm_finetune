"""
llm_finetune: SFT + GRPO training infrastructure for DesignBench LLMs.

Quick start:
    python scripts/train_sft.py model=qwen3_14b sft=warmstart
    python scripts/train_grpo.py model=qwen3_14b rl=grpo_truss

Research hooks (override these for your experiments):
    llm_finetune.training.sft.targets    — SFT supervision targets
    llm_finetune.training.rl.rewards     — RL reward functions
    llm_finetune.training.rl.costs       — RL cost functions
    llm_finetune.training.rl.reward_models — learned PRM/ORM models
    llm_finetune.data.datasets.mcts_dataset — MCTS tree data structures
"""

__version__ = "0.1.0"

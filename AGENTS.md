# Project Memory

## Mission

This is a research codebase for DesignBench LLM fine-tuning: SFT warmstarts, GRPO/RL training, MCTS-style evals, posterior rewards, and truss-design environments. Favor reproducibility, explicit experiment records, and small, reviewable changes over clever rewrites.

## Non-Negotiable Safety Rules

- Never run model loading, generation, vLLM servers, training, or GPU-heavy inference on a login node. Use Slurm batch jobs or an allocated compute node.
- Log every LLM interaction during eval/debug runs: full message list, raw outputs including parse failures, parsed action, source, and rank.
- Treat zero-candidate or zero-step eval runs as invalid failures, not successful results.
- Do not download large models, launch distributed jobs, or start long-running evals without an explicit user request.
- Avoid destructive cleanup of logs, checkpoints, figures, or datasets unless the user specifically asks for it.
- Do not write test outputs, smoke-test artifacts, plots, or temporary project files to `/tmp`; this controlled HPC environment requires artifacts to stay under project-managed paths such as `logs/`, `figures/`, or another explicitly requested project directory.

## Repo Map

- `llm_finetune/data/`: datasets, processors, collators, chat formatting.
- `llm_finetune/models/`: model loading and vLLM server integration. Treat as HPC-sensitive.
- `llm_finetune/training/sft/`: SFT trainer, warmstart callback, target/loss-mask logic.
- `llm_finetune/training/rl/`: GRPO trainer, rewards, costs, reward models, posterior tree logic.
- `llm_finetune/envs/`: DesignBench/truss environment integration.
- `scripts/`: command-line entrypoints for data prep, training, eval, and verification.
- `configs/`: Hydra config groups for models, data, SFT, RL, logging, PEFT, and parallelism.
- `slurm/`: batch scripts for GPU/H100 work.
- `tests/`: pytest coverage; most unit tests should not require model downloads or GPUs.
- `figures/` and `logs/`: generated research artifacts. Preserve provenance.

## Development Workflow

- Start by checking `git status --short`; assume unrelated dirty files are user work and leave them alone.
- Use `rg`/`rg --files` for search. Read nearby code before editing so new code matches local patterns.
- Keep changes scoped to the requested behavior. Do not refactor research plumbing unless needed for correctness.
- Prefer structured config changes through Hydra YAML files instead of hard-coded experiment constants.
- When adding a new research hook, update the relevant registry and include a minimal test or smoke check where practical.
- For data/eval changes, preserve enough metadata to reproduce the run: config path/overrides, source file, model id, checkpoint, seed, and output path.

## Commands

- Install editable dev package: `pip install -e ".[dev]"`
- Run fast unit tests: `pytest tests/test_rewards.py tests/test_targets.py tests/test_data.py tests/posterior -v`
- Run all local tests: `pytest tests/ -v`
- Run formatting/lint checks when available: `ruff check .` and `black --check .`
- Submit GPU jobs through Slurm, for example: `sbatch slurm/sft_h100.sbatch`
- Inspect Slurm jobs with `squeue`, `sacct`, `sstat`, and `scontrol`; do not replace batch execution with direct login-node GPU commands.

## Testing Expectations

- Add or update tests for changed reward functions, target transforms, dataset processors, parsing, and posterior tree behavior.
- Tests should be deterministic and small by default. Mock or fixture heavy model/env behavior rather than loading real checkpoints.
- A passing zero-work eval is not acceptable: assert candidate counts, step counts, and parse success/failure accounting explicitly.
- If a change can only be validated on compute nodes, document the Slurm script/command and expected output artifact.

## Logging And Artifacts

- Eval/debug logs must include raw model outputs even when parsing fails.
- Preserve parsed action, source, rank, candidate index, and any filtering/rejection reason.
- Include enough context in JSONL/metrics artifacts to connect figures back to run logs and configs.
- Do not overwrite existing logs or figures silently; use distinct run IDs or output directories.
- Keep all generated artifacts and smoke-test outputs inside the project tree. Do not use `/tmp` as a staging area for this project.

## Style

- Python target is 3.10; follow existing type and dataclass patterns.
- Ruff uses line length 100 and ignores `E501`; keep imports sorted.
- Keep comments focused on research assumptions, non-obvious invariants, or tricky parsing/eval behavior.
- Use clear names for experiment variants, reward components, and config overrides.

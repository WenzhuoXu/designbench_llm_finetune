# Multi-turn GRPO via TRL `rollout_func` — Design & Contract

> Why: Turn-0 found GRPO (and eval) drove a multi-turn-trained model **single-turn**, so it
> never practiced the real task (see `docs/ablation_ledger.md`). This wires a per-step
> generate→FEA→generate rollout into TRL's `GRPOTrainer` via the `rollout_func` hook, matching
> the master doc §3 (horizon H≈9) and the model's trained distribution. Approved approach
> (2026-06-13): `rollout_func` (keeps `<action>` grammar + existing warmstart; config-flagged).

## Verified TRL 1.0.0 contract (from source, not docstring)

`trl/trainer/grpo_trainer.py`, transformers 5.5.0:

- The train sampler is `RepeatSampler(mini_repeat_count=num_generations)` **unconditionally**
  (L902) — so the batch (`inputs`) handed to generation is already `N = B*G`, with each unique
  problem prompt appearing as **G consecutive identical entries** (one GRPO group).
- `rollout_func(prompts, trainer)` is called with `prompts = [x["prompt"] for x in inputs]`
  (L1606,1710) → length **N = B*G, with G-consecutive duplicates**. (The docstring's "no
  duplication / return G per prompt" does NOT match this build — verified against the code.)
- It must return a dict with `prompt_ids`, `completion_ids`, `logprobs` (each a length-N list,
  1:1 with the input prompts; `logprobs`/`completion_ids` per-token aligned). Optional:
  - `env_mask` (length-N, per-token; 1 = model-generated, 0 = environment/FEA tokens) — TRL
    pops it as the `tool_mask` so masked tokens get no gradient (L1649).
  - any other key → merged into `inputs` by index `i` (L1951-1954) and forwarded to reward
    funcs as a kwarg list (L1157,1196). **This 1:1-by-index merge is how we thread the
    per-rollout `rollout_result` to the reward safely.**
- Advantages group consecutive G: `rewards.view(-1, num_generations)` (L1967). So returning
  one rollout per input entry (1:1) yields correct groups; temperature gives in-group diversity.

## Our rollout_func (one independent multi-turn rollout per input entry)

For each prompt string (look up its spec via a `prompt_text → problem_spec` registry built
from the RL dataset):

1. `truss, goals = _load_truss_and_goals(spec)`; `state = _analyze_truss(...)` (initial FEA).
2. Loop ≤ max_steps:
   - Build messages = `formatter.build_messages(problem_text, action_history, initial_fea_result=initial_state)`.
   - `ctx_ids = apply_template(messages, add_generation_prompt=True)`. **Turn 0's `ctx_ids` is
     the returned `prompt_ids`**; everything after is `completion`.
   - Generate continuation `gen_ids` (+per-token logprobs) with the unwrapped policy model
     (HF generate, `output_scores=True`; vLLM disabled for multi-turn). Append to completion
     with `env_mask=1`, real logprobs.
   - Parse `<action>` from gen text; `_apply_action`+`_analyze_truss` → new `state`; append to
     `action_history`. Stop if `is_feasible`.
   - Re-template the conversation (incl. the new assistant turn + new user FEA turn) with
     `add_generation_prompt=True` → `next_ids`. The **delta** `next_ids[len(running):]` is the
     inter-turn glue (assistant close + FEA user turn + next gen prefix): append to completion
     with `env_mask=0`, `logprob=0.0`. Requires `next_ids` to extend `running` as a prefix
     (true for Qwen3's append-only `<|im_start|>/<|im_end|>` template; assert + warn otherwise).
3. Build a `RolloutResult` (state_history, action_sequence, parse_success, reaches_solution,
   final/initial_state) from the actual trajectory → return as extra field `rollout_result`.

Return `{prompt_ids, completion_ids, logprobs, env_mask, rollout_result}` (all length N).

## Reward integration

`_build_reward_callable.compute_rewards` already pulls `problem_spec` from kwargs and calls
`env.run_completion(spec, completion)`. Add: **if `rollout_result` is in kwargs (multi-turn),
use it directly** (the exact experienced trajectory) instead of re-running `run_completion`
on the FEA-interleaved transcript (whose `[Simulation Result]` markers the `\n\n` splitter
mis-handles). Single-turn path unchanged.

## Wiring & flags (additive, reversible)

- New config knob `rl.multi_turn: true` (+ `rl.max_turns`, reuse `rl.max_new_tokens` per turn).
- In `build_grpo_trainer`: when `rl.multi_turn` and not `use_vllm`, construct the rollout_func,
  register prompt→spec from `train_dataset`, and pass `rollout_func=` to `GRPOTrainer`.
- vLLM must be OFF for multi-turn v1 (per-turn HF generate). Revisit batched/vLLM later for
  throughput.

## Canary gate (before any full sweep)

Short run: warmstart ckpt, `group_size=4`, `max_steps_train≈5`, few problems, `use_vllm=false`,
2×H100. Pass = rollouts execute multi-turn (action_history grows, FEA stepping works), kl
becomes **>0** and grows, `clip_ratio/region_mean > 0`, `grad_norm` healthy, reward finite +
sane, `rho_tree_agreement` present in metrics.jsonl, no OOM/NCCL. Only then scale to S1/S2.

# Ablation Experiment Ledger

> Single source of truth for the DesignBench SFT+GRPO ablation study.
> Schema per `docs/ablation_orchestration_plan.md` §5.2. Orchestrator-owned.
> Target: `eval/feasibility_rate > 0.90` on held-out truss problems (§6).
> Theory anchor: `docs/plan_posterior_reward_walkthrough.md` ("master doc").

## Cell table

| turn | date | stage | config | model | run_name | job_id | status | feasibility_rate | rho_t | kl_final | reward_trend | failure_mode | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| turn | stage | rollout | cell | job | status | feas | grammar | mean_fos_b | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | S0 baseline | single | v2_01a_alpha2 | 41379673 | DONE | 0.00 | 0.041 | 0.55 | all 6 cluster ≈0; harness artifact (see narrative) |
| 0 | S0 baseline | single | v2_01b_alpha5 | 41379674 | DONE | 0.01 | 0.043 | 0.59 | α makes no difference |
| 0 | S0 baseline | single | v2_01c_alpha10 | 41379675 | DONE | 0.01 | 0.048 | 0.58 | |
| 0 | S0 baseline | single | v2_02_tree_d1 | 41379527 | DONE | 0.00 | 0.047 | 0.55 | |
| 0 | S0 baseline | single | v2_03_base | 41379676 | DONE | 0.00 | 0.045 | 0.57 | |
| 0 | S0 baseline | single | v2_04_tree_d2 | 41379677 | DONE | 0.01 | 0.046 | 0.57 | |
| 0 | MT validation | multi(25) | sft_warmstart | 41380436 | DONE | 0.00 | **1.00** | 0.41 | harness fix: grammar 0.04→1.00; warmstart = format-only (no policy) |
| 0 | MT validation | multi(25) | v2_02_tree_d1 | 41380721 | DONE | 0.04 | 0.40 | 0.41 | v2 RL ≈ no gain + **degraded grammar** (1.00→0.40) vs warmstart |
| 1 | MT-GRPO canary | — | mt_canary (warmstart) | 41380792 | DONE (mechanism OK) | — | — | — | rollout_func runs end-to-end, no crash, reward_std>0; but 680s/step + turns hit 512-tok cap (transcript 2486>2048 truncated). Needs termination + throughput fixes before sweep |
| — | MT baseline | multi(25) | warmstart (reference) | 41380436 | DONE | 0.00 | 1.00 | 0.41 | floor; format-only, no policy |
| 1 | S1 r_env α=5 | multi(25) | mt_s1_01b_alpha5 | 41384385/41394097 | DONE+EVAL | **0.12** | 0.44 | **1.42** | **FIRST feasibility gain.** FOS 0.41→1.42; but grammar 1.0→0.44 (r_env doesn't protect format) |
| 1 | S1 r_env α=2 | multi(25) | mt_s1_01a_alpha2 | 41387391/41398994 | DONE+EVAL | 0.08 | 0.49 | 1.09 | α-sweep |
| 1 | S1 r_env α=10 | multi(25) | mt_s1_01c_alpha10 | 41387392/41398995 | DONE+EVAL | 0.04 | 0.46 | 0.70 | **α=10 HURTS (H1 refuted)** |
| 1 | S2 tree D1 α=5 | multi(25) | mt_s2_02_tree_d1 | 41387393/41398996 | DONE+EVAL | 0.12 | 0.46 | **2.12** | tie on feas, best FOS, ρ=1.0 |

Checkpoint paths (all LoRA r=32, α=64 on base `Qwen/Qwen3-14B`):
- v2_01a_alpha2  → `checkpoints/grpo/qwen3_14b_v2_01a_alpha2_20260523_2324/final`
- v2_01b_alpha5  → `checkpoints/grpo/qwen3_14b_v2_01b_alpha5_20260522_2152/final`
- v2_01c_alpha10 → `checkpoints/grpo/qwen3_14b_v2_01c_alpha10_20260525_1228/final`
- v2_02_tree_d1  → `checkpoints/grpo/qwen3_14b_v2_02_tree_d1_20260525_1228/final`
- v2_03_base     → `checkpoints/grpo/qwen3_14b_v2_03_base_20260525_1228/final`
- v2_04_tree_d2  → `checkpoints/grpo/qwen3_14b_v2_04_tree_d2_20260525_1228/final`

---

## Narrative

### Turn 0 RESULT — S0 baseline + a blocking protocol-mismatch finding (2026-06-13)

**Baseline numbers (fixed harness, 100 problems each, greedy, max_steps=20):**

| cell | feasibility_rate | grammar_success | mean_fos_buckling | mean_steps |
|---|---|---|---|---|
| v2_01a_alpha2  | 0.00 | 0.041 | 0.548 | 15.3 |
| v2_01b_alpha5  | 0.01 | 0.043 | 0.589 | 14.6 |
| v2_01c_alpha10 | 0.01 | 0.048 | 0.581 | 14.0 |
| v2_02_tree_d1  | 0.00 | 0.047 | 0.548 | 15.2 |
| v2_03_base     | 0.00 | 0.045 | 0.574 | 14.9 |
| v2_04_tree_d2  | 0.01 | 0.046 | 0.568 | 14.7 |

All six cluster at **~0% feasibility** (1 feasible / 100 at best; union of solved across all
cells = 2 problems, intersection = 0). As predicted from the barely-moved policy, α and tree
depth make no difference. **But the dominant cause is NOT the weak policy — it is a rollout
protocol mismatch that makes the feasibility metric an artifact:**

**FAILURE MODE: single-turn eval/GRPO vs. a multi-turn-trained model.**
- The gold warmstart model emits **exactly one action per generation**, then `<|im_end|>`
  (compliance sample: `<think>…</think>\n<action>ADD_MEMBER(…)</action><|im_end|>`, 56 tokens,
  `single_action=True`, `n_actions=1`). It was SFT'd on multi-turn conversations (one action
  per assistant turn, FEA feedback between turns). This is *correct* warmstart behaviour.
- Both `eval_checkpoint.py` AND the GRPO trainer (`grpo_trainer.py:368
  run_completion(spec, completion)`) drive the model **single-turn**: one prompt (initial
  problem, no FEA feedback loop) → one completion → `TrussRolloutEnv.run_completion` tries to
  recover a whole 20-step trajectory by splitting the completion on `"\n\n"`
  (`_split_completion_into_steps`).
- Given no feedback loop, the model goes OOD: it rambles ~1000–1190 tokens (vs 56 in-dist),
  and the `"\n\n"` splitter shreds the `<think>` block into ~15 fragments of which ~1 parses
  as a grammar action → **grammar_success ≈ 0.04**, ≈1 action executed, FOS stuck ~0.55 « 1.5
  → feasibility ≈ 0 for *every* checkpoint. This is why all cells look identical.
- Multi-turn machinery EXISTS (`posterior/state_context.py StateContext.build_messages` with
  `action_history`+`fea_result`, used by the posterior evaluator) but is **not wired into the
  GRPO trainer or eval**. `TrussRolloutEnv` also has a per-action `step()` (verify) for a
  proper generate→apply→feedback loop.

**Implication:** the §6 acceptance metric is currently meaningless, and the diagnosed Turn-1
lever (higher LoRA LR) would NOT fix it — you cannot RL your way to feasibility when the
rollout only ever executes ~1 action. **The real fix is a multi-turn rollout harness.** Per
§6 criterion #4 this is an infra blocker; per the orchestrator's Turn-0 mandate ("eval
wiring … must be reproducible") fixing the eval to be multi-turn is in scope. Plan: (1) check
master doc for the intended rollout protocol; (2) add a multi-turn generate→FEA→generate loop
to eval and re-measure the TRUE baseline (the model may already be competent in its trained
distribution); (3) report the GRPO-trainer single-turn mismatch to the human, since making
GRPO rollouts multi-turn is a larger change with theory implications.

**Action taken (2026-06-13):** implemented a **multi-turn rollout path** in
`scripts/eval_checkpoint.py` (`_eval_problem_multiturn`, selected by `--rollout multi`, now
the default; legacy single-turn kept under `--rollout single`). It generates one action,
applies it via FEA (`_apply_action`/`_analyze_truss`), feeds the result back through
`ChatFormatter.build_messages(action_history, initial_fea_result)`, and repeats to
feasibility or max_steps — matching the model's trained distribution and master-doc §3.
`slurm/eval_ablation_v2.sbatch` now passes `ROLLOUT`/`MAX_PROBLEMS`/`MAX_NEW_TOKENS` env vars.
Multi-turn canary (gold warmstart, 25 problems) submitted as **job 41380436**
(`results/eval/sft_warmstart_mt25`). If the warmstart shows materially >0 feasibility
multi-turn, it confirms the metric was a harness artifact and reframes the whole study;
then re-baseline all checkpoints multi-turn and report the GRPO-trainer single-turn issue.

**Multi-turn canary RESULT (gold warmstart, 25 problems) — job 41380436:**
`grammar_success_rate = 1.00` (vs 0.04 single-turn) — **the multi-turn harness fully fixes
the format/parsing artifact.** But `feasibility_rate = 0.00` (0/25), `mean_steps = 20` (never
solves early), `mean_final_fos_buckling = 0.41`. Per-problem: most start infeasible
(init fosB ≈ 0.57 « 1.5); after 20 valid actions mean fosB *drops* to 0.41 and **mass
explodes** (init 444 → final 1132 kg avg; e.g. p000 mass 102→2648, p003 fosB 1.05→0.01).
Only 6/25 improved FOS.

**Conclusion (decisive, reframes the study):** with the harness corrected, the warmstart
model emits perfectly valid grammar but makes **incoherent/destructive** edits — it has **no
optimization policy**. This is the expected format-only warmstart behaviour ([[feedback_sft_is_warmstart_only]]).
Optimization competence is exactly what GRPO must instil — and the v2 GRPO ran on the broken
**single-turn** protocol (`grpo_trainer.py:368`), so that RL was spent off-distribution. The
earlier "raise LoRA LR" lever is moot: you cannot RL to feasibility when (a) rollouts execute
~1 action and (b) the training-time generation doesn't match the multi-turn task.

**Architecture scope:** GRPO uses TRL's `GRPOTrainer` (TRL 1.0.0), inherently single-turn —
`reward_fn(completions, prompts)`, one completion per prompt from internal (vLLM/HF) generate.
Making it multi-turn (generate→FEA→generate, per-step Lagrangian γΦ(s')−Φ(s), tree expansion
§3 at each step) is a **core-training-loop redesign with theory implications**, not a
config/hook change — i.e. beyond the orchestrator's "swap configs/hooks + submit" mandate and
the §0 rule "prefer config/hook changes over hacks in shared code." **→ REPORT to human with
a scoped proposal (§6 #4).** Confirmatory GRPO-tree_d1 multi-turn eval (job 41380721) running
to verify RL added nothing over warmstart (expected ≈0, since kl≈0.003).

**Human direction (2026-06-13):** "SFT only serves as grammar compliance check; it's fine if
the policy doesn't succeed." → warmstart 0%-feasible/100%-grammar is the intended state, not a
defect ([[feedback_sft_is_warmstart_only]]). Proceed to the real lever: make GRPO learn the
policy on the correct (multi-turn) protocol. Not redirected off-goal.

**Multi-turn GRPO feasibility scoping (no-GPU, TRL source inspection):** the installed TRL's
`GRPOTrainer` supports a custom **`rollout_func(prompts, trainer) → {prompt_ids,
completion_ids, logprobs}`** hook (experimental) and a higher-level `environment_factory`+
`tools` agentic path (needs transformers ≥5.2.0). **Chosen approach: `rollout_func`** — run
the generate→FEA→generate loop inside it (reusing the now-validated multi-turn eval logic),
return flattened ids/logprobs with a mask so only model-generated tokens get gradient, and
make the reward use the rollout's per-step state sequence (Lagrangian γΦ(s')−Φ(s)) rather than
the decoded FEA-interleaved text. Additive + config-flagged (reversible). Full sweep gated on
a validation canary (policy moves: kl↑, clip>0; multi-turn rollouts execute; reward trends).

**Multi-turn GRPO IMPLEMENTED (approved approach, 2026-06-13).** Full design +
verified TRL contract in `docs/multiturn_grpo_design.md`. Changes (additive, config-flagged
`rl.multi_turn=true`; single-turn path untouched):
- NEW `llm_finetune/training/rl/multiturn_rollout.py`: a TRL `rollout_func` running the
  per-step generate→FEA→generate loop; returns prompt_ids/completion_ids/logprobs + `env_mask`
  (1=model,0=FEA/template tokens → masked in the loss) + a per-rollout `RolloutResult`
  (threaded to the reward via TRL's 1:1 extra-fields merge). Prompt→spec registry keyed by
  prompt string.
- `grpo_trainer.py`: when `rl.multi_turn` (+ `use_vllm=false`), reconstruct the dataset's
  formatter, register specs, pass `rollout_func=` to GRPOTrainer; reward callable uses the
  threaded `rollout_result` when present.
- NEW knobs: `rl.multi_turn`, `rl.max_turns`, `rl.max_turn_tokens` (per-turn budget, distinct
  from `rl.max_new_tokens`=TRL max_completion_length=full transcript).
- All 3 modules compile under py3.11 (login `python3` is pre-3.7 — use the conda interpreter
  for `py_compile`).

**Confirmatory GRPO-tree_d1 multi-turn eval (job 41380721, interim):** grammar_success ≈
**0.30** per problem (vs warmstart **1.00**), feasible=False, n_steps=20. So the broken
single-turn RL appears to have **degraded** multi-turn grammar compliance vs the warmstart —
further evidence the v2 RL was counterproductive, not merely unhelpful. (Full result pending;
multi-turn eval is ~7 min/problem.)

**Canary submitted (job 41380792, `mt_canary_20260613_0216`):** warmstart ckpt, r_env only,
group_size=8, max_turns=5, max_steps_train=4, use_vllm=false, 2×H100. Gate per §canary in the
design doc (kl>0 & rising, clip_region>0, grad_norm healthy, reward finite, rho logged, no OOM)
before any S1/S2 multi-turn sweep.

**MT-GRPO canary RESULT (job 41380792, 4 steps, warmstart, 2×H100):** the rollout_func
**works end-to-end** — "Multi-turn GRPO enabled … 100 prompt specs", rollouts executed,
`rewards/compute_rewards/mean=6.66`, `reward_std=0.42`, `frac_reward_zero_std=0`, loss
computed, checkpoint saved, **no crash, no prefix-mismatch warnings**. Architecture validated.
Two blockers for a real sweep:
1. **Throughput:** `step_time≈680s` (sequential per-prompt-per-turn generation, no batching)
   → ~37h per 200-step cell. Needs batched generation across the group.
2. **Turn termination/length:** `completions/mean_length=2486`, `clipped_ratio=0.89` — each
   turn hits `max_turn_tokens=512` instead of ~56 (eval), and the transcript exceeds
   `max_completion_length=2048` → truncation. Root cause: training builds the formatter from
   `cfg.model.model_name_or_path` = the **checkpoint path** → thinking_mode=NONE, whereas the
   (well-behaved) eval used the base id → QWEN3. Fix: resolve thinking mode by model family in
   the rollout formatter; raise max_completion_length; set logging_steps=1 for the next canary.
(kl/clip not meaningful at 4 steps w/ cosine LR→0; metrics.jsonl empty since logging_steps=10.)

**Canary v2 (job 41381529, thinking-mode=QWEN3 fix, max_completion_length=4096, LOG_EVERY=1):**
step 1 → `reward=24.8, reward_std=0.77` (positive, valid actions parsed), but
`comp_len=2479, clip_frac=0.94` — **turns STILL hit the 512-tok cap**; the thinking-mode fix
did not shorten them. So the model rambles at temp 0.8 in the multi-turn context (vs concise
greedy eval). It's a throughput/verbosity issue (684s/step), not correctness.

**kl trend (6 steps) — POLICY MOVES under the correct protocol (the key validation):**
step4 `kl=0.0137` (>0.01 bar), `grad_norm=0.66` — vs the single-turn v2 runs which were FLAT
at kl≈0.003 / grad_norm≈0.07 across all 200 steps. So the multi-turn protocol fix is the real
unlock: the policy is learnable. (Signal intermittent only because the canary's cosine LR
collapses by step5–6 over T_max=6; a real T_max sustains it. reward not comparable across
steps — different problem batches.) **Conclusion: multi-turn GRPO is validated and worth a
real sweep.**

Remaining blocker = **throughput**: 684s/step (sequential per-rollout, per-turn generation;
turns ~2480 tok at temp 0.8) → 38h/200-steps > 24h wall. Options: (A) batched per-turn
generation across the group (proper, ~Nx faster, complex/risky refactor); (B) config
compromise now — ~100-step cells (19h<wall), smaller group/horizon — launch immediately with
validated code. Decision pending (user).

**ROOT CAUSE of rollout rambling (and likely poor action quality) — OOD prompt format.**
The compliance check (where the warmstart emits a concise ~50 tokens) builds prompts with
`WarmstartTransform` over dev.jsonl = DesignBench's **native rich format**: problem as a joint
listing + `INITIAL STATE ANALYSIS:` merged into the first user turn; FEA as
`[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\nMass: …\nFactor of Safety (buckling): …\n
Status: INFEASIBLE ✗\nConstraint violations:\n  - …`. But the RL rollout AND eval use
`chat_formatter.build_messages` (`_spec_to_problem_text` lists *members* + "Begin your
iterative optimization"; `_format_fea_feedback` uses `  Mass: X kg / FOS_buckling` + a
"Continue optimizing…" instruction). **Completely different wording/structure → the model is
out-of-distribution → it rambles to 512 tok/turn** (the 10× vs SFT's ~50). This very likely
also degraded the warmstart's multi-turn eval (0% feasible): OOD prompts → worse actions, not
just verbose ones.

**Fix (clean — reuse, don't replicate):** DesignBench exposes the canonical formatters —
`validation/model_interface.py:format_problem_prompt` and `model_adapters.py:_format_state`
(+ the `truss_tot/` generator that built the SFT data). Wire the rollout (`multiturn_rollout.py`)
and `eval_checkpoint.py` to build prompts/FEA via these, so RL/eval prompts match SFT exactly.
Expected: turns drop to ~50–150 tok (~5–10× throughput → sweep feasible w/o batched gen) AND
in-distribution actions (better feasibility). Re-canary, then launch the S1/S2 multi-turn
sweep. **Decision/awareness checkpoint with user before this next build phase.**

**Format-alignment IMPLEMENTED (2026-06-13):** new `llm_finetune/data/processors/designbench_prompt.py`
reproduces the exact SFT format (`format_eval_result` ≡ trace_extractor `_format_evaluation_result`;
`build_problem_text` ≡ `_generate_comprehensive_problem_text`; `build_messages` = system +
merged `<problem>+INITIAL STATE ANALYSIS` user + `<think>/<action>` assistant turns +
`[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:` user turns). Verified byte-for-byte vs a
dev.jsonl sample. Wired into both `multiturn_rollout.py` and `eval_checkpoint.py` (replacing
`chat_formatter.build_messages`); token bookkeeping unchanged (re-template+delta, content-agnostic,
prefix-checked). Canary v3 (job 41381810) submitted to confirm concise turns (comp_len↓ from
~2480) + step_time↓ + policy still moves. If confirmed → launch S1/S2 multi-turn sweep at a
proper horizon (now feasible within the 24h wall).

**Canary v3 (DesignBench-format prompts): format fix did NOT shorten turns** —
`comp_len=2560, clip_frac=1.00` (every turn hits the cap), same ~2500 as v1/v2. So the
rambling is NOT from OOD prompt format. eos is configured correctly (`eos_token=<|im_end|>`
=151645, in gen eos_ids), so **the model simply never emits its turn-terminator** — it keeps
generating (probably hallucinating the next FEA/turn) for the full budget. Likely temp-0.8 +
Qwen3-thinking driven non-termination. (NB the format fix is still correct/in-distribution and
worth keeping; it just isn't the length lever.) Pre-existing oddity also noted: GRPO loads the
warmstart via `from_pretrained(adapter_dir)` THEN `_apply_lora` → PEFT "modify a second time"
warning = a base+warmstart+fresh double-adapter stack (works, but messy; same as v2 runs).

**Diagnostic canary v4 (job 41382077):** added sample-completion logging to the rollout
(reveals what the model actually generates), temp 0.8→0.6, max_turn_tokens 384 (bounds
throughput regardless of root cause). Goal: SEE the generation to fix termination, then launch
the S1/S2 multi-turn sweep (`configs/rl/grpo_mt_01b.yaml`, `grpo_mt_02_tree.yaml` staged).

**Canary v4 SAMPLE — the long turns are GENUINE REASONING, not a bug (key reframing).**
The logged generations are coherent engineering CoT ("buckling FOS 1.07 < 1.5… members too
slender… increase cross-section… deflection too high → structure too flexible…"), cut off
mid-`<think>` at the 384 cap (`ends_eos=False`). The model is Qwen3 doing real extended
reasoning; the warmstart's brief-CoT SFT (LoRA r=32) is too light to suppress the base model's
verbose reasoning at temp 0.6. **Because the `<action>` comes AFTER `</think>`, capping turns
truncates reasoning before the action → breaks the rollout.** So long turns are inherent to
reasoning-based multi-turn RL (which the master-doc framework explicitly wants: r_pred/CoT
grounding). Throughput is therefore a genuine fork, not a bug to squash:
  - **A. Reasoning-faithful:** generous turn budget (~1.5–2k tok) + **batched per-turn
    generation** (the proper ~Nx throughput fix; complex/risky refactor) → full-horizon sweep.
  - **B. Action-only probe:** `enable_thinking=False` → ~20–50 tok/turn → fast full-horizon
    sweep NOW; tests whether multi-turn GRPO improves feasibility, but drops CoT (contradicts
    the reasoning thesis for S6 r_pred; OK for S1 r_env / S2 tree whose reward needs only the
    action). Re-introduce CoT + batched gen later if the probe shows promise.
  - **C. Reduced-scope reasoning sweep:** keep CoT, accept ~50–60 steps × 5 turns within the
    24h wall un-batched (checkpointed); limited but a real signal.
Decision = user's (research tradeoff: CoT-in-RL fidelity vs compute). Core science already
validated (multi-turn → policy learnable, kl 0.0137). Configs staged: grpo_mt_01b/02_tree.

**BATCHED GENERATION built + validated (canary v5, job 41384192) — reasoning infra works.**
Rewrote `multiturn_rollout.py` to a turn-major batched loop (`_generate_batch`: left-pad all
active rollouts, one `generate` per turn, per-seq logprob extraction + early-stop at eos,
`gen_batch_chunk` memory cap). Canary v5 (full-reasoning, max_turn_tokens=1536,
max_completion_length=8192): turns now **terminate** (sample `ntok=724/1102, ends_eos=True`;
~25% hit the 1536 cap), `step_time=511s` (→ 100 steps ≈ 14h < 24h wall), **no OOM** at
group_size=8 / comp_len=5885, `reward=25.1, reward_std=2.24`. Reasoning-faithful multi-turn
GRPO is now tractable.

### Turn 1 — first reasoning-based multi-turn GRPO sweep (2026-06-13)

**S1 PRIMARY baseline LAUNCHED** (job 41384385, `mt_s1_01b_alpha5`): grpo_mt_01b — r_env only,
α=5, multi_turn, max_turns=5, max_turn_tokens=1536, group_size=8, lr 5e-6, 100 steps, temp 0.6.
This is the first real reasoning-based multi-turn ablation. Validating the full run (kl rising,
reward trend, ρ logged, no OOM/divergence over ~25 steps) before fanning out the α-sweep
(01a/01c) + S2 tree. Hypothesis: feasibility/reward trends UP over training (vs the flat,
single-turn v2). Watch: assess trend at ~step 25 (~3.5h).

**S1 trend (01b, 25 steps): healthy, learning slow.** kl 0.0012→0.0022 (rising, still <0.01),
grad_norm ~0.07, `reward` drifting up (-4.4 → 5.6 → 7.2 → 8.9), reward_std large (real
advantages), step_time ~500s stable, comp_len ~5800 (full reasoning), **ρ logged to
metrics.jsonl ✓**. Well-directed but slow at lr 5e-6 → likely Turn-2 lever = LR bump (clean
next axis) if the full sweep's feasibility stays flat.

**Sweep fanned out (job→cell):** `41384385`=mt_s1_01b_alpha5 (running), `41387391`=mt_s1_01a_alpha2,
`41387392`=mt_s1_01c_alpha10, `41387393`=mt_s2_02_tree_d1 (queued). All: multi_turn, 5 turns,
1536 tok/turn, group 8, lr 5e-6, 100 steps. **02_tree exercises the untested tree-expansion ×
multi-turn integration — watching for early crashes.** Next: eval each finished checkpoint
(multi-turn) for feasibility_rate; compare α∈{2,5,10} (H1: monotone in α) and tree vs baseline
(H2: ρ→≥0.85, Φ↑); re-align with master doc §7.1; decide Turn 2.

**All 4 Turn-1 cells RUNNING; tree path + ρ(t) VALIDATED.** 02_tree step 5:
`reward=8.72, reward_std=5.25, rho=1.0` — the tree-expansion × multi-turn integration works and
**ρ(t) (master-doc §3.6 headline, previously uninstrumented) is now live in metrics.jsonl.**
No crashes; step_time ~500s across cells.

**OPEN ISSUE — GRPO checkpoint eval loading (double-adapter).** Training loads the warmstart
via `from_pretrained(adapter_dir)` (→ base+warmstart PeftModel) then `_apply_lora` adds a fresh
GRPO adapter (PEFT "second adapter" warnings). Saved checkpoint = the fresh adapter
(`base_model_name_or_path: Qwen/Qwen3-14B`). To reproduce the *training-time* model, eval must
load base+warmstart+grpo; `eval_checkpoint.py` currently loads base+grpo (missing warmstart) —
UNLESS warmstart was inactive during training (PEFT active-adapter semantics, needs runtime
check). Resolve before scoring Turn-1 checkpoints: eval the full stacked-adapter model (load
warmstart adapter, then the GRPO adapter) to match training; sanity-check coherent reasoning.
(Pre-existing — the v2 evals had the same loading.) Not blocking the running sweep.

**01b DONE (100 steps) — the policy LEARNED.** kl rose **monotonically 0.0012→0.0049 (4×)**
over training (vs single-turn v2's flat ~0.003) — clean evidence the policy moved progressively
under the correct protocol. Modest magnitude (lr 5e-6 cosine→0 by step 100); grad_norm
0.065–0.123. reward noisy (different problem batches) but positive. Checkpoints at 50/75/100/
final. **01b/final multi-turn eval launched (job 41394097, 25 problems, 20 steps — matched to
the warmstart 0%-feasible baseline)** → the key Turn-1 number: does multi-turn GRPO improve
feasibility? grammar_success will also sanity-check the checkpoint loads coherently
(double-adapter question). Other 3 cells finish ~4h.

**TURN 1 KEY RESULT (01b α=5): multi-turn GRPO works.** feasibility **0% → 12%**,
mean FOS_buckling **0.41 → 1.42** (warmstart → after 100 GRPO steps), on the SAME 25-problem
multi-turn eval. First feasibility gain in the project. Confirms the central thesis: the v2
failure was the single-turn protocol, not the reward/LR — with the correct protocol the policy
learns to drive FOS toward feasibility.
- **Named failure mode: grammar degradation (reward-hacking format).** grammar_success
  1.00 → 0.44. The S1 r_env-only reward (lagrangian_potential alone) gives no credit for
  valid format, so GRPO sacrifices grammar for FOS gains → 56% of actions fail to parse →
  caps feasibility. (Eval-loading sanity-confirmed: warmstart=1.00, 01b=0.44 same loader.)
- **Re-align w/ master doc:** §1 r_env shaping works (FOS↑↑); the missing piece is a
  format/grammar protector. The base grpo_truss/posterior configs DO include
  `grammar_compliance: 0.1` — the v2-derived S1 configs dropped it.
- **Turn-2 proposal:** add `grammar_compliance` (and/or `format`/KL-to-warmstart) to the
  reward to preserve parseability while keeping the FOS gains → expect higher feasibility.
  Secondary lever: higher LR (learning was slow, kl peaked ~0.005). Pending the α-sweep +
  tree results (01a/01c/02_tree, ~1h) to pick the single Turn-2 change (plan §4 discipline).

**TURN 1 FULL ANALYSIS (α-sweep + tree):**
- **H1 REFUTED:** feasibility non-monotone in α — α2=0.08, α5=**0.12**, α10=0.04; FOS α2=1.09,
  α5=1.42, α10=0.70. The master-doc §7.1 H1b predicted highest feasibility at α=10; instead
  **α=10 hurts** (over-weighting the Lagrange multiplier destabilises the FOS landscape). → α=5
  is the sweet spot. **Open problem (§8 candidate): re-examine the α scaling in the potential.**
- **H2 partial:** tree(D1,α5) ties baseline feasibility (0.12) but best FOS (2.12 vs 1.42),
  ρ=1.0 in training. Tree optimises harder; extra FOS headroom didn't convert to more feasible
  problems at this scale/horizon. Worth keeping for the FOS gain.
- **Universal failure mode:** grammar 1.0 → 0.44–0.49 across ALL trained cells → general
  r_env-reward-hacking of format; the targeted Turn-2 fix.

**TURN 2 (launched): grammar-protected reward, α=5, on both r_env and tree bases.**
One change (add `grammar_compliance` w=5, scaled to the ~10–25 r_env reward) vs the Turn-1
winners. Hypothesis: protecting parseable format lets the FOS gains convert to more feasible
problems (fewer wasted/unparsed turns) → feasibility > 12%.

**TURN 2 trained (both cells, 100 steps).** Adding `grammar_compliance` (w=5) drove
**stronger** policy movement: kl 0.0012→**0.0114** (grammar) and →**0.0097** (grammar+tree),
both crossing the 0.01 bar (vs Turn-1's 0.0049) — the grammar term adds learnable group-variance.
Evals launched (jobs 41439385 grammar, 41439386 grammar+tree; multi-turn, 25 problems). Checking:
(a) grammar_success recovers from 0.44, (b) feasibility > 12%.

**TURN 2 RESULT — grammar-reward hypothesis REFUTED.** grammar w=5 → grammar_success
0.44→**0.42–0.47 (FLAT, no recovery)**; feasibility 0.12→0.04/0.00 (within 25-problem noise);
FOS flat-to-down. Conclusions:
- **Grammar loss is drift/fragility-induced, not reward-addressable.** kl is only ~0.01 yet
  grammar collapses 1.0→0.44 — the thin r=32 warmstart's format is destroyed by tiny policy
  movement; a grammar reward can't restore it (and kl_coef×kl penalty is negligible at kl~0.01,
  so KL doesn't constrain it either).
- **Grammar may NOT be the bottleneck** — 12% feasibility was already reached at grammar 0.44.
- **Eval-noise caveat:** 25 problems → ±4% resolution (±1 problem); turn-to-turn feasibility
  deltas of a few points are unreliable. FOS (continuous) is the better within-turn signal,
  and it plateaus ~1.4–1.5 (right at the 1.5 threshold) → the policy pushes FOS to the line
  but can't reliably clear all constraints (buckling ∧ yielding ∧ mass).
- **Plateau:** Turn 2 yielded no improvement over Turn 1 → approaching §6 "diminishing returns".

**STRATEGIC INFLECTION (reporting to user, §6).** Validated: protocol fix → MT-GRPO learns
(0→12%). But we've plateaued ~12% feasibility / FOS~1.5; the grammar lever failed; and the road
to >90% is long (master-doc S3–S7 reward stack + likely more steps + a more robust warmstart +
less-noisy eval). Next-step options pending user direction (see report).

### Turn 3 — STRATEGY PIVOT: CoT capability probe → distillation (2026-06-14, user-directed)

User redirected after the Turn-2 plateau ([[project_cot_distillation_strategy]]):
**(1)** investigate CoT control + whether an *advanced* model can solve the truss task with
proper CoT (alone or with φ/full composite reward); **(2)** distill the advanced (teacher)
model's CoT policy into qwen3_14b (student). gemma4 = teacher only (unstable to finetune).

**Step 1 launched:** eval `qwen3_30b_a3b_thinking_2507` (base, no FT) multi-turn on the same
25 problems (job 41446462, 2×H100, downloads ~60GB first). Added CoT-sample logging to
`eval_checkpoint.py` + parameterized `eval_ablation_v2.sbatch` with `MODEL_CONFIG`. Questions:
does the advanced model reach >12% feasibility with its CoT, and is the CoT well-controlled
(reasoning → good actions)? If yes → it's the distillation teacher. gemma4_26b_a4b is cached as
a fallback teacher.

**Teacher CoT probe — KEY FINDING (answers "is CoT well controlled?"):** qwen3_30b's reasoning
is **excellent and well-controlled** (identifies buckling as the failure, reasons about
compression members / slenderness / moment-of-inertia via r,t — correct structural engineering).
BUT at max_new_tokens=1024 it's **truncated before reaching `</think><action>`** → `parsed=None`
every turn → state never changes → re-reasons identically → would score a misleading 0%. So the
advanced model HAS the reasoning; the bottleneck is **CoT length vs token budget** (30B reasons
very long) and possibly the action FORMAT (base model, no warmstart). Re-running with 6144
tok/turn on 6 problems (job 41446559) to measure true capability + capture COMPLETE parseable
trajectories. Insight for distillation: teacher has reasoning-not-format; qwen3_14b warmstart has
format-not-reasoning → distill teacher reasoning into the format-capable student.

**Teacher probe COMPLETE (qwen3_30b, 6144 tok/turn).** Even with ample budget the 30B reasons
up to ~1942 words/turn and **still `parsed=None` on most turns** (terminates naturally, not
truncated) — it does NOT commit to the final `<action>` block, despite the system prompt
explicitly instructing it ("exactly one grammar action per turn… `<action>…</action>`… no prose
outside `<think>`/`<action>`"). The few parser-extracted actions (gram 0.25) are exploratory
musings, not decisive moves → FOS unchanged (0.442) → ~0% feasible.
**Answers to the user's part-1 questions:**
- *Is the CoT well-controlled?* **Yes** — the advanced model's structural reasoning is excellent.
- *Can it accomplish the task with proper CoT?* **Not as a base model** — it reasons correctly but
  won't follow the action OUTPUT FORMAT from instructions alone (classic base-reasoning-model
  behaviour). It needs **few-shot exemplars or a format warmstart** to emit decisive, parseable
  actions and thus produce *solving* trajectories.
**→ Distillation path requires format-enabling the teacher first** (few-shot the 30B, or SFT-
warmstart it on the format data), THEN generate solving (reasoning→action) trajectories, THEN
SFT-distill into the format-capable qwen3_14b. Reporting to user for the approach decision.

**Few-shot teacher (qwen3_30b) FAILED.** With the worked example the 30B reasoned even MORE
(~3308 words/turn) and emitted **zero `<action>`** anywhere (grep confirms) → gram 0.00 (worse
than 0.25 without few-shot). The **A3B-*Thinking* model is fundamentally mismatched** to the
brief-structured-action task — it over-reasons and never commits to the `</think><action>`
output, and prompting (few-shot) can't fix it. Conclusion: a heavy thinking model is the wrong
teacher *form*; need a model that emits the `<think>+<action>` structure (instruct model, or a
warmstarted teacher).
**Next (autonomous, within user's named candidates): test `gemma4_26b_a4b`** (cached,
non-thinking instruct → should FOLLOW the format) with few-shot as the teacher. If it emits
actions + reasons + solves → it's the distillation teacher (gemma4 = teacher only, never
finetuned, per [[project_cot_distillation_strategy]]).

**COURSE-CORRECTION (user, 2026-06-16):** switching to a non-thinking teacher (gemma4) was
wrong — the *thinking process IS the CoT* we want to distill; qwen3_30b not emitting an
`<action>` tag doesn't make it a bad teacher; **parse the action a different way** instead.
Cancelled gemma4. Kept qwen3_30b-thinking. Implemented **action extraction** (`--extract-action`,
`_extract_action_followup` in eval): keep the model's FULL CoT, and when no `<action>` is
inline, ask the SAME model once more to commit its reasoning to a single grammar action →
parse that. This preserves the entire thinking process and parses the action separately. Re-run
job 41447690 (qwen3_30b, extraction, 6 problems). If it now SOLVES with its CoT → it's the
distillation teacher; trajectory = `<think>{CoT}</think><action>{extracted}</action>`.

**Two-model distillation pipeline BUILT (user-approved: 14B as action head).** Same-model
extraction failed (the Thinking variant always thinks in prose; grep found ZERO grammar syntax
anywhere). New `scripts/distill_gen.py` + `slurm/distill_gen.sbatch`: loads teacher
(qwen3_30b-Thinking) + student (qwen3_14b-warmstart) on 2×H100. Per turn: teacher reasons (full
CoT) → student is **prefilled with `<think>{CoT}</think>\n<action>` and continues** → emits the
grammar action (the assistant turn = the rollout step AND the SFT distillation target). Applies
via FEA → next state. Launched job 41448228 (6 problems). Tests: does 30B-reasoning + 14B-
grounding SOLVE (FOS↑, feasible)? If yes → generate trajectories on the train problems →
SFT-distill into qwen3_14b. Outputs trajectories JSONL for the distillation SFT.

**Distill pipeline BREAKTHROUGH + bugfix.** Diagnostic showed the prefill mechanism WORKS: the
14B student, prefilled with the teacher's CoT conclusion, emits **grounded** actions —
`SCALE_PARAM(M6,r,2.0)`, `SCALE_PARAM(M7,r,2.0)` (M6/M7 = exactly the compression members the
30B identified!). It only failed to PARSE: the teacher's prose uses member label "M6" (the
problem text renders members as M0,M1,…) but the grammar validator requires a numeric id
(`_is_int`). Fix: normalize `\bM(\d+)\b → \1` before parsing (`r` is already a valid PARAM_ALIAS).
Verified offline: `SCALE_PARAM(M6,r,2.0)` → parses. So teacher-reasoning → student-grounded-
action works. Re-running 6 problems (job in watch bvq383yfh) to confirm the pipeline now SOLVES
(actions apply → FOS↑ → feasible). If yes → generate trajectories at scale → SFT-distill into
qwen3_14b.

**DISTILL-GEN RESULT (30B reasons + 14B acts, zero-shot, 6 problems): mechanically perfect,
trajectories weak.** gram≈1.0, grounded actions, but **0/6 feasible, mean FOS 0.77**, mixed:
p001 0.51→1.32, p005 0.39→1.50 (good); p000/p003 small; p002 no action; **p004 0.84→0.00
(regressed)**. Worse than the trained 14B-GRPO (FOS 1.42 / 12%). Distilling this mixed/weak
data won't beat existing.

**KEY HYPOTHESIS (high-leverage):** the teacher GUESSES critical members because it sees only
the GLOBAL fos_buckling scalar, never **per-member FOS**. So it scales plausible but often-wrong
members (p004 regressed). This would explain why BOTH the 30B teacher AND the 14B-GRPO plateau
~FOS 1.5 — **the bottleneck is information/feedback, not reasoning quality.** `_analyze_truss`
returns only global mass/fos_b/fos_y/deflection. If DesignBench can expose per-member FOS /
which member is buckling, adding it to the FEA feedback could unlock BOTH GRPO and the teacher
(and make teacher trajectories worth distilling). Reporting to user: pursue per-member feedback
(addresses the real bottleneck) vs filter-distill the few good trajectories vs consolidate.

**ROOT-CAUSE CONFIRMED + FIX: the FEA feedback omitted the critical member.** DesignBench's
`analyze_truss` returns `min_fos_buckling_member_id` (and yielding) — exactly WHICH member is
failing — and DesignBench's own `model_adapters._format_state` shows it. But the SFT/trace
format (reproduced in `designbench_prompt.format_eval_result`) **dropped it**, so every model
(teacher AND 14B-GRPO) guessed. Verified on problem_000: the critical member is **M4**, but the
teacher was scaling **M6/M7** → FOS barely moved. Added the critical member to the feedback
("worst member(s): M4"). This is likely the universal plateau cause and a high-leverage unlock
for BOTH the teacher and GRPO. Testing on the teacher (distill-gen v3, 6 problems) — if it now
targets M4 and solves, trajectories become worth distilling AND a richer-feedback GRPO re-run is
warranted. (NB: omitting the critical member from the SFT format was effectively a data/format
bug affecting the whole study.)

**BREAKTHROUGH — critical-member feedback → FIRST FEASIBLE SOLVE in the project.** With the
"worst member(s): M4" feedback, the teacher immediately scaled member **4** (the actual critical
one) instead of guessing M6/M7 → **problem_000 feasible=True, FOS 0.442→1.694 in 7 steps**.
Nothing prior (warmstart, all GRPO cells, earlier teacher runs) had ever produced a feasible
design. **Confirms: the bottleneck was INFORMATION (which member is critical), not reasoning.**
Implications: (1) the teacher now generates genuinely-solving trajectories → worth distilling;
(2) **the same feedback should go into GRPO training** — likely breaks the 14B plateau directly.
Awaiting the 6-problem feasibility aggregate.

**CRITICAL-MEMBER FEEDBACK RESULT (teacher+student, 6 problems): feasibility 0% → 67% (4/6).**
p000→1.69(7 steps), p001→1.54(6), p003→1.52(4), p005→1.53(5) all FEASIBLE; p002 emitted no
action (edge case), p004 over-scaled→0. vs 0% without the critical member, vs 12% for the
trained 14B-GRPO. **Decisive: the bottleneck was information, not reasoning/RL.** The headline of
the whole study shifts: "RL can't learn" → "the prompt omitted the critical-member signal;
restore it and even zero-shot reasoning solves 67%."

**Direct next step LAUNCHED: GRPO-14B re-run with the enriched feedback** (job 41454559,
mt_t3_01b_critfeedback; the multiturn rollout auto-uses the enriched designbench_prompt). Tests
whether the feedback breaks the 14B plateau (12%/FOS1.42) directly — possibly to the >90% goal,
maybe without distillation. **Complementary (user's strategy):** the teacher now produces
67%-solving trajectories → viable to generate at scale + SFT-distill into qwen3_14b. Pending
user direction on whether to run distillation in parallel.

**GRPO-with-critical-member-feedback: learning MUCH stronger.** kl rises fast —
0.0014→0.0106(step35)→**0.0196(step55)**, vs the prior 01b run's 0.0049 at step *100* (≈4×
more movement). The richer feedback gives a clearer signal → policy moves more. Evaling
checkpoint-50 (multi-turn, enriched feedback, 25 problems; job 41469646) for an early read on
whether it clears the 12% plateau. Full run (job 41454559) → ~step 100 in ~5h.

**PLATEAU BROKEN — GRPO-with-critical-member-feedback @ checkpoint-50: feasibility 0.68 (68%)**,
mean FOS_buckling 1.63 (25 problems) — vs the previous GRPO ceiling of **12%**. Training learned
hard: kl 0.0014→**0.0346** at step 100 (~7× the prior 01b's 0.005). So the critical-member
feedback is THE unlock for GRPO too (mirrors the teacher's 0→67%). Full 100-step run done;
evaling final/checkpoint-100 (job 41473781) for the converged number (toward the >90% target).
Headline of the study is now clear: **the dominant lever was the omitted critical-member signal
in the FEA feedback; once restored, both zero-shot reasoning and GRPO jump from ~0-12% to
~68%.** (Plus the earlier prerequisite: the single→multi-turn protocol fix.)

**GRPO+critmember CONVERGED at 68% (ckpt50=ckpt100=0.68; more steps don't help).** Failure
analysis of the 8/25 unsolved: **3 = mass over-shoot** (FOS fixed but mass blew past the limit,
e.g. p002 mass 119→1441) — and the feedback flagged FOS/deflection violations but **NOT mass**,
so the model over-scaled blindly (same omission class as the critical member); **4 = degenerate**
(FEA→0, broken or huge-initial-mass p015/p017); **1 = under-improved**. grammar 0.67 (wasted
turns) persists.
**Next lever (one change): mass feedback.** Added the mass LIMIT to every feedback ("Mass X
(limit Y)") + an explicit mass-violation line ("Mass X > Y (EXCEEDS by Z%) — reduce material").
Data was already in the env state (`mass_constraint`). GRPO re-run launched with critical-member
+ mass feedback (job 41473933, mt_t4_01b_massfb) — targets the 3 mass-overshoot fails → expect
~80%. (Grammar fix + degenerate-problem handling are later levers.)

**MASS FEEDBACK REGRESSED (68%→52%) — reverted.** Adding an explicit mass-limit + "EXCEEDS,
reduce material" violation (mt_t4) made the policy **over-conservative**: mean FOS rose to 1.50
(hugging the threshold) but feasibility DROPPED to 0.52 — it traded 3 mass-overshoot fails for
more FOS-under-shoot fails. So a hard mass imperative over-corrects. Reverted designbench_prompt
to the champion config (**critical-member feedback only = 68%**). Lesson: the buckling-critical
signal is uniquely high-leverage; the mass constraint needs a *softer* treatment (reward-side
balance, not a hard prompt imperative) — deferred.
**Robust eval:** champion (mt_t3 GRPO+critmember final) on **50 problems** (job 41484986) to pin
the headline number (25-problem est. ±8%). CHAMPION STANDS: GRPO+critical-member ≈ 68%.

### Turn 0 — Instrumentation + S0 baseline (2026-06-12)

**State verification (plan §1) — corrections to the plan's premises:**

- **`results/eval/` is NOT empty as the plan claimed — but it is effectively empty.**
  Six dirs exist (`v2_01a_alpha2` … `v2_04_tree_d2`, created May 27) but every one is
  empty. A May-27 eval batch (jobs 41012798–41012803) was attempted and *all six crashed*
  before writing any result. So the conclusion stands: **no checkpoint has ever produced a
  feasibility number.**

- **Eval harness was broken with three distinct bugs** (root-caused from
  `logs/slurm/abl_v2_eval_41012798.err`), now fixed in `scripts/eval_checkpoint.py`:
  1. **Config not composed.** `OmegaConf.load(model_config)` does not process Hydra's
     `defaults: [base]`, so `torch_dtype` (and all of `base.yaml`) was missing →
     `ConfigAttributeError: Missing key torch_dtype` at `loader.py:71`. This is the crash
     that killed the May-27 batch. Fixed with a `_load_model_config()` helper that merges
     defaults relative to the config dir (general, recursive, not truss-specific).
  2. **Checkpoint never loaded.** The script computed `model_id = args.checkpoint or …`
     only for logging, then loaded `load_model_and_tokenizer(model_cfg)` whose
     `model_name_or_path` is the *base* `Qwen/Qwen3-14B`. The trained **LoRA adapter was
     never applied** → it would have silently scored raw Qwen3-14B. Fixed: detect
     `adapter_config.json`, `PeftModel.from_pretrained(...).merge_and_unload()`, and adopt
     the checkpoint's own tokenizer (training chat template).
  3. **Wrong device + slow generate.** `base.yaml device_map=null` (correct for DeepSpeed)
     would leave the eval model on CPU and mismatch the `.cuda()` inputs; and
     `gradient_checkpointing_enable()` forces `use_cache=False` → crawling generate. Fixed:
     eval-time overrides `device_map=auto`, `use_gradient_checkpointing=False`,
     `use_compile=False`. Also pass the *base HF id* (not the checkpoint path) to
     `ChatFormatter.from_model_id` so Qwen3 thinking mode resolves (a path defaults to NONE).

- **ρ(t) logging (§3.1) is ALREADY instrumented at HEAD — plan premise is stale.**
  `rho_tree_agreement` is computed every reward batch (`grpo_trainer.py:431
  _compute_rho_per_group`), stored in shared `_rho_state` (487), injected into the log dict
  in `GRPOCallback.on_log` (518), and written to `metrics.jsonl` via `local_logger.log_step`
  → `_write_jsonl` (104/246, no key filtering) *and* to W&B. The reason it is absent from
  the v2 `metrics.jsonl` is purely timing: the fix landed in commit **4bf470c (2026-06-07,
  "fix W&B rho logging")**, while the v2 runs ran **May 23–26**. **No new ρ(t) code needed.**
  Verification deferred to the first Turn-1 GRPO run (cheap gate: confirm `rho_tree_agreement`
  present in `metrics.jsonl` by step ~20).

- **Mail alerts (§5.1) added:** `--mail-type=END,FAIL` to `eval_ablation_v2.sbatch`,
  `--mail-type=BEGIN,END,FAIL` to `grpo_qwen3_lora_2gpu.sbatch`, both to wxu2@andrew.cmu.edu.

**GRPO failure signals quantified — refined root cause (the LR-too-low theory was wrong):**
- The v2 ablation configs **already set `learning_rate: 5.0e-6`** (10× the grpo_base 5e-7
  default), not 5e-7. So the policy-movement problem persisted *despite* the LR bump.
- **No T_max scheduler bug** (the submit script's feared failure did NOT occur): LR
  trajectory is a correct cosine over T_max=200 — peaks ~4.97e-6 at step 20, decays to
  ~3e-10 at step 200. (The old May-22 1245 run used the un-fixed 5e-7 peak and is a
  pre-fix artifact; the 2152 run and all 0525 runs use 5e-6.)
- The 5e-6 bump *did* help marginally: kl rose 0.0014 (5e-7 run) → 0.0035 (5e-6 run). But
  kl still tops out ~0.0035, far below the §5.3 >0.01 bar, and `clip_ratio/region_mean=0`
  every step (importance ratio ≈1, no clipping).
- **Decisive metrics:** `frac_reward_zero_std ≈ 0.08` (only 8% degenerate groups → healthy
  advantages), `reward_std` 17–32 (large), yet **`grad_norm ≈ 0.07–0.12` (tiny)**.
  Completions are ~1100 tokens (model thinks at length in RL; not the 50-token warmstart),
  clipped ratio ~3% (rarely truncated at max_new_tokens=2048). entropy ~0.30 stable.

**Root cause:** advantages are well-formed, but **parameter movement is negligible** —
peak LR 5e-6 × grad_norm ~0.1 ⇒ ~5e-7 update/step, cosine-decayed to zero by step 200. For
**LoRA** RL, 5e-6 is a *low* peak LR (LoRA typically wants 1e-5–1e-4); with grad_norm ~0.1,
kl ~0.003, and zero clipping there is large headroom before instability. The lever is a
**substantially higher peak LR** (and/or more steps / larger LoRA rank), NOT a scheduler or
zero-variance fix. kl_coef=0.04 is not the constraint (kl is tiny).

**Hypothesis (Turn 0):** the v2 sweep produced near-identical, near-base policies, so the
baseline eval numbers for all six cells will cluster together (and near base-model
performance), regardless of α or tree depth — because almost no learning occurred.

**Result:** S0 baseline eval batch is running on the fixed harness (all 6 cells). Canary
`v2_02_tree_d1` (job 41379527) cleared end-to-end — first problems evaluate at ~15–22 s
each. Fan-out submitted; job→cell map:
`41379527=v2_02_tree_d1`, `41379673=v2_01a_alpha2`, `41379674=v2_01b_alpha5`,
`41379675=v2_01c_alpha10`, `41379676=v2_03_base`, `41379677=v2_04_tree_d2`.

> **Heads-up — eval set is 100 problems, not 10.** `eval_checkpoint.py` globs
> `auto_problem_*.json` and found **100** specs (plan §1.3 said "10"). This is a larger,
> more robust baseline, but the GRPO runs trained on these same problem specs
> (`data=sft_traces`) — **so this is an in-distribution number, not strictly held-out.**
> Open item for Turn 1: carve a held-out split (or confirm DesignBench provides one) before
> using feasibility as the §6 acceptance metric. Early signal: first problems infeasible
> (fos_b ≈ 0.48–0.51 vs the 1.5 threshold), consistent with the barely-moved policy.

**Theory re-alignment:** §6.3 expects feasibility monotone in α (S1) and ρ(t)→≥0.85 with Φ
gains (S2). Neither is testable on the v2 checkpoints because (a) ρ(t) wasn't logged then
and (b) the policy barely moved. The v2 baseline therefore measures *warmstart + negligible
RL*, i.e. approximately the SFT-warmstart policy — useful as a floor, not as an α/tree test.

**Next proposal (Turn 1, pending baseline numbers):** the single structural change is the
**policy-movement fix = much higher peak LR**. Plan: a small LR-probe sweep on the PRIMARY
baseline cell (`grpo_abl_v2_01b`, α=5, r_env only) at peak LR ∈ {2e-5, 5e-5} (≈4×/10×
current), warmstart-initialised, max_steps_train≥200. **Gate before fanning the full sweep:**
confirm §3.2 sanity at the higher LR — kl>0.01 by step 100, `clip_ratio/region_mean>0`,
grad_norm up, reward trending within-condition, entropy not collapsing — AND that
`rho_tree_agreement` now appears in `metrics.jsonl` (verifies the §3.1 fix end-to-end). If a
LR moves the policy without KL-collapse, re-run S1/S2 at that LR. If even 5e-5 won't move it,
escalate LoRA rank (r=32→64) before concluding. Do NOT widen the model axis until the recipe
demonstrably moves the policy on qwen3_14b.

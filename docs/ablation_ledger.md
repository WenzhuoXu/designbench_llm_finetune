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

---

### Turn 5 — the potential was never the problem's potential (2026-08-23)

Full write-up: `docs/turn5_potential_specification.md`. Theory: `docs/theory_potential_and_feedback.md`.
Estimator design: `docs/design_real_lookahead_advantage.md`.

**Nine defects, each verified, that between them invalidate most of turns 0-4:**

| # | defect | invalidates |
|---|---|---|
| D1 | `use_tree_expansion` never calls `evaluate_tree`; it computes a group-relative TERMINAL potential, which GRPO's within-group mean-centring makes affine-equivalent to the plain shaping reward | every "tree"/search cell |
| D2 | training rho is `argmax Phi_H == argmax reward`: an identity under tree mode, a 1/K coin flip otherwise | every rho number |
| D3 | Phi encodes a constraint set no problem has (deflection <= 0.01 m priced on the 100 problems with no deflection goal; the mass cap never priced) | the reward of every run |
| D4 | `posterior.alpha` never reached the reward | the alpha sweep |
| D5 | Hydra merges dict keys, so every `mt_*` config inherited grpo_truss.yaml's four reward terms | the "single dense reward term" claim; the grammar ablation |
| D6 | train and eval read the same problem directory | the word "held-out" throughout |
| D7 | the trained token sequence contained NO environment feedback (Qwen3 strips `<think>` from non-final turns, so the prefix check failed on every transition; 9176 warnings in the champion run) and `env_mask` was all-ones | the context every multi-turn run trained in |
| D8 | the shaping sum omitted `gamma^t`, so a no-op paid `(1-gamma)|Phi|` -- +0.25/turn, +1.27 over five | the "failed runs burn the turn budget" mode |
| D9 | the potential was unbounded outside the simulator's validity envelope (negative-mass sections; mechanisms with 1e14 m deflection scored FEASIBLE) | 17% of the search's "successes" |

**D4 has a silver lining.** The alpha cells `mt_s1_01a/01b/01c` were three runs of an identical
configuration, returning **8% / 12% / 4%**. That is a direct measurement of the noise floor:
**+/-4 points at n=25**. Under it the alpha result, the grammar-reward result and the mass-feedback
result are all noise. Only 12 -> 68 clears it.

**The theoretical point.** TRL runs GRPO with `scale_rewards='group'`, so advantages are invariant
to affine transforms of the reward: **reward design reduces to ordering design.** Phi_v1 orders an
infeasible over-stiffened design ABOVE a feasible lean one (-3.80 vs -9.02 on auto_problem_000),
and no value of alpha can repair an ordering error.

**Search ladder, no LLM, 130 problems, legal action space, validity-gated, deterministic:**

| policy | feasible | mass/ref | FEA |
|---|---:|---:|---:|
| random | 30.0% | 1.050 | 15 |
| greedy critical-member (depth 0 = the feedback field) | 72.3% | 0.872 | 6 |
| fully-stressed design (analytic) | **75.4%** | **0.474** | 12 |
| 1-step lookahead, Phi_v1 | 58.5% | 0.834 | 2178 |
| 1-step lookahead, Phi_v2 | 73.8% | 0.644 | 2357 |
| 2-step lookahead, Phi_v2 | **75.4%** | 0.620 | 14016 |
| **trained LLM champion** | **54%** | **0.917** | 20 |

Paired on jointly-solved problems: Phi_v2 vs Phi_v1 = 0.583 vs 0.897, lighter on **64/75 (85%)**,
feasibility a strict superset (+20, -0). Phi_v2 vs the feedback heuristic = 0.602 vs 0.875, lighter
on **81/92 (88%)**. Phi_v2 vs fully-stressed design = 0.602 vs 0.485, lighter on only 11/94 (12%).

**Readings.** (1) The potential's SPECIFICATION is the decisive variable: +15.3 points and 35%
lighter designs from the same search. Searching hard against Phi_v1 is worse than the trivial
depth-0 heuristic. (2) The feedback field is an exactly sufficient statistic for the depth-1 argmax
at 65.8% of visited states, and sufficient to within 0.098 at 82.2% -- at zero simulator cost
against a median 181 vs 8 FEA calls. Feedback IS search, at depth 0. (3) Simulated search beats the
feedback heuristic on quality but loses to the ANALYTIC estimator on mass, because the grammar
cannot express fully-stressed design's simultaneous all-member resize (given a compound action the
same search reaches 0.429 vs FSD's 0.435). (4) The search finds simulator exploits; **no trained
policy ever did** (max FOS_b among solved = 2.9 across every cell). The engineering prior is the
validity constraint, the potential is the ranker -- an argument for the architecture, not for
either half.

**Turn-5 cells (all on `data/splits/truss_v1_auto.json`: 66 train / 34 eval, zero overlap):**

| cell | config | one variable vs | job | status |
|---|---|---|---|---|
| T5a | `grpo_mt_t5a_phi1` | champion reward, corrected rollout | 44237845 | RUNNING |
| T5b | `grpo_mt_t5b_phi2` | T5a: Phi_v1 -> Phi_v2 | 44241064 | RUNNING |
| T5c | `grpo_mt_t5c_lookahead` | T5b: + per-step counterfactual advantage | 44241065 | QUEUED |
| T5ao | `grpo_mt_t5ao_phionly_v1` | potential-only (scale cancels exactly) | 44241066 | QUEUED |
| T5bo | `grpo_mt_t5bo_phionly_v2` | T5ao: Phi_v1 -> Phi_v2, ordering alone | 44241067 | QUEUED |
| T6 | SFT on `distill_search` | can the policy internalise the search? | 44241068 | QUEUED |
| E-K1 | `eval_llm_search --n-candidates 1` | held-out greedy baseline + regret probe | 44241061 | RUNNING |
| E-K8 | `eval_llm_search --n-candidates 8` | LLM proposes, potential ranks | 44241062 | RUNNING |

Infrastructure fixed this turn: a seeded stratified split; `posterior.*` threaded into the reward;
per-job rendezvous ports (a hardcoded 29501 killed three jobs 11-33 s after launch); the
inter-turn glue derived from the chat template; a validity envelope on the potential; 28+8
regression tests pinning every defect above.


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

**ROBUST CHAMPION NUMBER: 54% on 50 problems (not 68%).** The 68% was on problems 0–24 (easier
subset); the 50-problem eval gives **feasibility 0.54, grammar 0.66, FOS_b 1.32** — so 25-problem
evals were optimistic + noisy (problems 25–49 solve less). Honest headline: GRPO+critical-member
≈ **54%** (vs 12% naive GRPO, 0% warmstart). Still the dominant result; the 25-problem cell
numbers stand for RELATIVE comparison but the absolute headline is 54%. (100-problem likely lower
— includes degenerate huge-mass problems.) All jobs done; queue empty. Report updated.

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

---

## Turn 6 — why the potential arms could not have worked, and where the potential does work

Three results, all CPU, that together explain every Phi arm in this ledger and
redirect the remaining budget.

### 1. Shaping is invisible to the estimator (proved)

`sum_t gamma^t (gamma*Phi(s_{t+1}) - Phi(s_t)) = gamma^T Phi(s_T) - Phi(s_0)`.
Phi(s_0) is a group constant, so mean-centring deletes it and only TERMINAL
potential survives. Every per-step distinction Phi computes is discarded before
the gradient. `tests/test_shaping_invisibility.py`, 5 tests: perturbing every
interior Phi by 1000x moves no advantage. `LagrangianPotentialV2Reward.compute`
is exactly this form, so it applies to what ran. `t5bo_phionly` is therefore a
pure terminal-reward arm, not a shaping arm.

### 2. The eval could not have seen what survived (measured)

Arm-vs-arm discordance = 0.152 over 253 pairs. Paired McNemar MDE at the 34x1
protocol every arm used: **18.7 pp**. Observed A/B deltas: 0.000, 0.000, -0.059
(t5a vs t5b) and +0.029, +0.059, -0.059, +0.029 (t5ao vs t5bo), every p >= 0.5.
These are non-measurements, not nulls. See `docs/eval_power.md`.

### 3. The potential ranks correctly where ranking matters (measured, 3 samplers)

Top-ranked member of a group of K=8, on groups where feasibility varies:

| trajectory sampler | decisive | Phi_v1 | Phi_v2 | delta | McNemar p |
|---|---|---|---|---|---|
| Phi_v2-guided (biased FOR v2) | 87/264 | 0.690 | 0.977 | +28.7pp | <0.0001 |
| random (neutral) | 28/198 | 0.679 | 1.000 | +32.1pp | 0.0039 |
| Phi_v1-guided (biased AGAINST v2) | 59/198 | 0.322 | 0.983 | **+66.1pp** | <0.0001 |

The advantage is LARGEST under the sampler biased against it -- the opposite of
a confound signature. Phi_v1's own search walks into states it cannot rank and
Phi_v2 can. `scripts/composite_ordering_divergence.py`.

Also measured: the A/B was a real intervention (29.3% of groups get a different
composite argmax) and Phi dominates the reward by magnitude (spread 1.23 vs base
0.57) -- both of my prior explanations for the nulls were wrong.

### Consequence

The potential works; the estimator throws it away and the instrument could not
see the remainder. Redirect to (a) inference-time Phi selection, which bypasses
the gradient (job 44519089, champion K=8 held-out), and (b) T8 per-token
advantages, the only in-gradient route. All future arms at 120x4 paired.

### Environment defect D10/D11 — scope corrected

Two gold action forms were inert pre-fix (`SCALE_PARAM(all_members,...)`,
flat `ADD_MEMBER`). On GOLD data that is 68% of actions. On **19,172 real GRPO
rollout actions across 43 runs it is 7.1%, all of it ADD_MEMBER** -- the policy
writes explicit member lists, which always executed. The arms are NOT invalidated
by this; re-running them is not justified. The narrow consequence is that the
policy was taught topology change does nothing.

### 4. The selection result is domain-general (battery, PyBaMM)

Same test on 24 battery problems, K=8, neutral random sampler. Phi_v1 is NOT the
control here: it reads mass/fos_* which a battery state does not have, so it
scores every battery state identically and comparing to it would be rigged. The
control is RANDOM selection from the same group -- what a policy without a
ranker does.

| domain | baseline | Phi_v2 | delta | McNemar p |
|---|---|---|---|---|
| truss, random sampler (vs Phi_v1) | 0.679 | 1.000 | +32.1pp | 0.0039 |
| truss, Phi_v1 sampler (vs Phi_v1) | 0.322 | 0.983 | +66.1pp | <0.0001 |
| **battery, random sampler (vs random pick)** | **0.200** | **0.950** | **+75.0pp** | **0.0003** |

Different simulator (electrochemical DAE, not FEA), different constraints,
different action grammar, same conclusion. `scripts/battery_group_selection.py`,
job 44519305. Caveat: PyBaMM raises IDA_ERR_FAIL on some infeasible parameter
vectors; those steps are dropped as illegal, same as an executor refusal.

### 5. Phi selection on REAL LLM rollouts — the end-to-end result

**Champion policy, held-out 34, K=8** (`results/eval/hybrid_phi_select_champ_k8`):

| rule | feasibility |
|---|---|
| no selection (first sample) | 26/34 = 0.765 |
| Phi_v2-selected | 29/34 = 0.853 |
| oracle | 29/34 = 0.853 |

Oracle recovery **29/29 = 1.000, zero misses** — including auto_problem_034 where
only 1 of 8 rollouts was feasible. On mass, Phi picked the LIGHTEST feasible
rollout on 28/29 problems (+0.12% over best available, Wilcoxon p=0.317).
End-task gain +8.8pp is NOT significant at n=34 (p=0.25) because mean solve_rate
is 0.790 — total available headroom IS 8.8pp, and Phi captured all of it.

**All logged GRPO groups, 43 runs, K=8, 472 decisive groups**
(`scripts/phi_selection_on_grpo_rollouts.py`, zero GPU):

| rule | feasible |
|---|---|
| policy's own first rollout | 183/472 = 0.388 |
| random pick | 190/472 = 0.403 |
| **Phi_v2 pick** | **461/472 = 0.977** |

**+57.4pp over the policy's own choice; discordant 276 vs 5; McNemar p < 1e-6.**
Terminal state reconstructed as initial+delta with deflection held at initial;
`feasible` is the simulator's verdict. The approximation handicaps Phi, so 0.977
is a lower bound.

**Reading:** selection accuracy is ~98% regardless of policy strength; the
END-TASK gain is set by headroom. Weak policy (0.388) -> +57pp. Strong policy
(0.765) -> +8.8pp. Same mechanism, different room to work in.

---

## Turn 7 — the benchmark was the blocker, and every number in this ledger is wrong

### 28 of 100 problems are degenerate mechanisms; the split file marks 11

Raw-simulator deflection is `inf` for 11 and 1e13-1e14 m for 17 more. Clean
problems top out at **0.106 m**, so the separation is absolute. `_analyze_truss`
clamps `inf` to exactly 1.0, which is why a `> 1.0` threshold catches nothing --
degeneracy MUST be read from `analyze_truss(load_truss_from_problem(spec), ...)`.

Contamination by split: train 18/66 degenerate (marked 7), **held-out 10/34
(marked 4) -> only 24 usable held-out problems, not 34.**

### Corrected results table (clean problems only)

| run | as reported | clean only |
|---|---|---|
| champion "68%" headline | 0.680 | **0.895** |
| champion K=8 held-out | 0.706 | **0.917** |
| champion + Phi selection | 0.853 | **0.958** |
| t5a_phi1 / t5b_phi2 | 0.618 | 0.875 |
| t5bo / t5ao | 0.588 | 0.833 |

The feedback result is BETTER than claimed: 0% -> 89.5% on real problems. And
every arm lands in 0.833-0.958 -- a 12.5pp spread against a 22.3pp floor at
n=24. **Nothing in this ledger was ever separable.**

### The Phi-selection headline was an artifact

Champion K=8, split by true degeneracy:

| | n | first sample | Phi-selected | oracle |
|---|---|---|---|---|
| clean | 24 | 0.958 | 0.958 | 0.958 |
| degenerate | 10 | 0.300 | 0.600 | 0.600 |

**The +8.8pp came entirely from degenerate problems. On clean problems the
selection effect is +0.000, discordant 0/0.** What survives is the weak-policy
result: 2553 clean GRPO groups, first-rollout 0.083 -> Phi 0.192, ceiling 0.196,
**+10.9pp, p<1e-6**. Selection captures ~98% of headroom; the champion has none.

### Root cause of the saturation

`_metadata.optimal_mass` was computed at fixed pipe aspect `t/r = 1/6`. A
free-aspect optimum is 0.52-0.67x that, so `maximum_mass = 1.1 x optimal_mass`
left ~27% unusable slack. **With no mass pressure, "scale everything up" always
works** -- 79/99 clean originals fall to a single global rescale. That is why
search depth never mattered.

### Replacement benchmark: `DesignBench/data/problems_hard/` (580 problems)

Non-uniform degradation (disjoint shrink/inflate subsets, per-member parameter
choice, member removal, joint jitter) + a simulator-certified free-aspect
optimum. 43% of problems START over the mass budget, so no global rescale helps.

| set | n | greedy | lookahead d1 | d2 |
|---|---|---|---|---|
| originals, 99 clean | 99 | 0.949 | **1.000** | 1.000 |
| previous generated | 250 | 0.236 | 1.000 | 1.000 |
| **problems_hard** | **580** | **0.116** | **0.734** | **0.800** |

**44 problems where d2 succeeds and d1 fails** -- the first truss set here where
search depth matters. Global-action argmax share: 81.9% -> 20.9%. Validity
580/580, independently re-audited on an 80-problem random sample (80/80 PASS).
Split `data/splits/truss_hard_v1.json`: 378 train / **202 held-out**, stratified
on MEASURED difficulty. MDE at K=8 improves from 7.9pp to ~2.7pp.

## Turn 8 — Phi-as-ranker is domain-general (battery)

`results/battery_ladder/group_selection.jsonl`, n=72 battery problems, K=8
candidates per problem, PyBaMM simulator. `phi_pick` = argmax Phi over the K,
`random_pick` = one random draw from the same K, `oracle` = any of K feasible.
Identical simulator budget across arms.

| domain | n | random pick | **Phi pick** | oracle | delta | exact McNemar |
|---|---|---|---|---|---|---|
| truss (weak GRPO rollouts) | 2553 groups | 0.083 | **0.192** | 0.196 | +10.9pp | p < 1e-6 |
| **battery** | 72 | 0.139 | **0.347** | 0.361 | **+20.8pp** | discordant 16-1, **p = 2.75e-4** |

Battery captures **93.8%** of oracle headroom (truss ~98%). By family:

| family | n | rnd | phi | oracle |
|---|---|---|---|---|
| energy | 9 | 0.22 | 0.89 | 0.89 |
| energy_hard | 9 | 0.11 | 0.22 | 0.33 |
| rate | 9 | 0.33 | 0.67 | 0.67 |
| rate_hard | 9 | 0.33 | 0.44 | 0.44 |
| thermal | 9 | 0.11 | 0.44 | 0.44 |
| thermal_hard | 9 | 0.00 | 0.11 | 0.11 |
| coupled | 9 | 0.00 | 0.00 | 0.00 |
| coupled_hard | 9 | 0.00 | 0.00 | 0.00 |

Phi captures the full available headroom in 5 of 8 families and never loses
ground; the two `coupled` families have **zero oracle headroom at K=8**, so
there was nothing there to capture -- that is a sampler failure, not a Phi
failure. Different simulator, different objective, different constraint set,
same result: **selection by Phi is the surviving use of the potential, and it
is not truss-specific.**

## Turn 8b — why the preference-trained model under-builds (mechanism, verified)

Earlier claim -- "the mass term dominates Phi and the model absorbed it" -- was
REFUTED: across states the violation term dominates 10:1 (spread 1.154 vs
0.115). That measurement was on the wrong comparison. DPO pairs do not compare
states; they compare **sibling actions out of the SAME state**.

Re-measured on all 2880 pairs in `results/distill/search_preferences.jsonl`:

| | chosen | rejected |
|---|---|---|
| mean scale factor | **0.907** | **1.403** |
| share of factors < 1 (mass-reducing) | **0.867** | 0.126 |

Paired: chosen factor lower in 2387/2880 (82.9%), higher in 149 (5.2%),
mean paired difference **-0.496, t = -61.9**. The preference set teaches
*shrink*, overwhelmingly.

**But Phi is not at fault -- the dataset composition is.** With w=1, alpha=5,
offset=0, a feasible state has violation 0 and therefore `Phi = -mass_ratio`
exactly. 2517/2880 pairs (**87.4%**) sit at `Phi > -1.2`, i.e. the violation
term is near zero: these are states that have ALREADY reached feasibility, where
"shrink to cut mass" is the correct ranking.

| | n | mean chosen factor | share < 1 |
|---|---|---|---|
| Phi > -1.2 (at/near feasible) | 2274 | 0.865 | **0.942** |
| Phi <= -1.2 (violation carrying) | 261 | **0.996** | 0.678 |

Proxy-free confirmation from the step index -- the sign flips exactly where
feasibility is reached:

| step | mean chosen factor | share < 1 | mean Phi |
|---|---|---|---|
| 0-2 | **1.071** (grow) | 0.484 | -1.470 |
| 3-5 | 0.882 | 0.874 | -1.075 |
| 6-8 | 0.863 | 0.949 | -0.960 |
| 12-14 | 0.854 | 0.986 | -0.819 |
| 15+ | 0.858 | 0.971 | -0.768 |

Phi ranks correctly at every step: grow while infeasible, trim once feasible.
The search then spends **87% of its steps in the post-feasibility trimming
phase**, so that is what the preference set is made of. A model trained on it
learns shrink *unconditionally* and starts shrinking at turn 0, which is exactly
the observed failure (final FOS 0.881 vs the champion's 1.308) -- and it is also
why that same model still ranks by Phi at 0.744 (z=5.51): it faithfully learned
the ranking Phi induces **on feasible states**, the only regime it was shown.

**Actionable:** stratify preference sampling by feasibility status, or simply
regenerate on `problems_hard`, where 43% of problems start over the mass budget
and greedy scores 0.116 -- the infeasible regime the old set almost never sampled.

## Turn 9 — Phi-ranking with a STRONG policy, on the hard benchmark

Cancelled the previously queued `eval_llm_search` pair (44704175/44704176) at
0:00 elapsed, **0 SU spent**, for two reasons:

1. **Unauditable.** Both were parameterised entirely through the environment,
   and SLURM does not retain the environment of a PENDING job -- `scontrol show
   job -dd` exposes no exported variables. There was no way to confirm they
   differed at all.
2. **Confounded even if they did differ.** `eval_llm_search.py:99` reads
   `keep_going = (n_candidates > 1) if min_mass is None else min_mass`, so
   unless `--min-mass`/`--stop-on-feasible` is passed EXPLICITLY, the K=1 arm
   stops at the first feasible design and the K=8 arm keeps optimising. Two
   variables move at once. The flag's own help text warns about this.

Fixed `scripts/eval_llm_search.py` to log `ARGS {...}` and the resolved
`keep_going` before loading the model, so every future run identifies its own
arm from its `.out` file.

Resubmitted with everything explicit, `--stop-on-feasible` on BOTH arms so the
ONLY difference is K:

| arm | K | stop-on-feasible | problems | out |
|---|---|---|---|---|
| greedy | 1 | yes | 202 held-out `problems_hard` | `results/eval/hard_k1` |
| propose+rank | 8 | yes | same 202, same order, same seed | `results/eval/hard_k8` |

**What this decides.** On the old saturated set Phi-selection was **+0.000**
(discordant 0/0) with a strong policy -- there was no headroom to capture. The
hard set has greedy 0.116 against depth-1 search 0.734, a 57pp gap. If
Phi-ranking over the policy's own K=8 proposals closes a meaningful part of that
gap, ranking is a real result for strong policies, not only for the weak GRPO
rollouts where it showed +10.9pp. If it does not, the ranking claim stays
confined to weak policies and the paper says so.

Ordering is `sorted(glob("*.json"))` and the seed is `random.Random(0)` in both
arms, and results are rewritten after every problem, so a run truncated by the
walltime yields a clean PREFIX -- the paired McNemar is taken on the
intersection with no selection bias.

## Turn 10 — Phi-as-FEEDBACK fails: showing the potential to the model does not help

`scripts/api_potential_guidance.py`, Claude Sonnet 4.5 via Bedrock, 80 held-out
`problems_hard`, 20 turns, paired by problem. **Arms differ in text only**: the
guided arm's system prompt carries an explainer of what Phi measures, and each
turn's observation appends `Design potential Phi: <v>` plus how the last action
moved it. `compute_potential_v2` reads the state dict the turn already produced,
so **both arms make exactly the same number of simulator calls**. 160 episodes,
zero errors, parse rate 0.996 / 0.999.

| arm | feasible | steps | parse |
|---|---|---|---|
| control | **23/80 = 0.287** | 16.7 | 0.996 |
| guided | **17/80 = 0.212** | 17.1 | 0.999 |

**Effect of showing Phi: -7.5pp**, discordant 3 guided-only vs 9 control-only,
exact McNemar **p = 0.146**, approx 95% CI **[-16.0pp, +1.0pp]**.

Not significant, but the point estimate is negative and **any true improvement
larger than +1pp is excluded at 95%**. There is no useful effect here to find
with more n. Note also that the guided arm improved Phi *less* over the episode
(mean dPhi +2.865 vs +3.155): telling the model the objective made it optimise
that objective slightly worse, consistent with the extra text crowding the
reasoning rather than steering it.

### The three routes for the potential, now all measured

| route | mechanism | result |
|---|---|---|
| **reward** | shaping term added to the GRPO reward | **provably invisible** -- telescopes to a group constant, deleted by mean-centering (5 tests, `tests/test_shaping_invisibility.py`) |
| **feedback** | Phi shown to the policy in the observation | **no effect / slightly negative** -- -7.5pp, CI excludes >+1pp (this turn) |
| **ranking** | Phi selects among already-generated candidates | **works** -- truss weak policy +10.9pp (p<1e-6), battery +20.8pp (p=2.75e-4), ~94-98% of oracle headroom |

**The potential is a selector, not a signal.** It is useful for choosing among
candidates that already exist; it does not help when communicated to the
learner, whether through the reward or through the prompt. The two failing
routes fail for different reasons -- one is an algebraic identity, the other is
empirical -- but the surviving claim is the same in both truss and battery.

Side observation worth keeping: on this hard set Sonnet 4.5 scores **0.287**,
against the fine-tuned champion's 0.167 and depth-1 procedural search's 0.734.
The gap that matters is still search-vs-policy, not model-vs-model.

### Why showing Phi hurts: the same under-building failure, twice

Paired over the 80 episodes of the guidance A/B:

| | control | guided | paired diff | t |
|---|---|---|---|---|
| FOS buckling | 1.674 | **1.395** | **-0.279** | **-2.44** |
| final mass | 245.3 | 225.0 | -20.4 | -1.31 |
| mass_ratio | 1.658 | 1.586 | -0.073 | -0.75 |
| under-built (FOS_b < 1.5) | 18/80 = **0.225** | 27/80 = **0.338** | | |

The guided arm built **lighter and weaker**. Mass fell (not significantly) while
the buckling factor of safety fell significantly, pushing half again as many
designs under the FOS >= 1.5 requirement. And the damage is concentrated exactly
where there was room to lose: `medium` 0.208 -> 0.083, while `easy` is saturated
(0.929 both) and `hard`/`open` are 0.000 both.

**This is the same failure as the preference-trained model** (Turn 8b: final FOS
0.881 vs the champion's 1.308, from a preference set that was 87% post-feasible
trimming). Two completely different ways of making Phi visible to the policy --
DPO on Phi-ranked pairs, and simply printing Phi in the prompt -- produce the
same pathology.

The mechanism is legibility, not correctness. `Phi = -mass_ratio - 5*violation`.
The mass term is smooth, always active, and trivially actionable: shrink
something. The violation term is the one that should dominate while infeasible,
but acting on it requires understanding which constraint binds and why. A policy
told "higher Phi is better" reaches for the legible lever and under-builds. Phi
itself ranks correctly at every step (grow while infeasible, trim once feasible,
Turn 8b) -- the policy just fails to recover the conditioning.

**This is why selection works and communication does not.** Under selection the
policy never sees Phi; Phi only filters what the policy already produced, so
there is no lever to over-pull. That is the whole content of "selector, not
signal", and it now has a measured mechanism rather than just two null results.

## Turn 11 — the headroom law, PRE-REGISTERED before the strong-proposer result

Every Phi-selection result so far fits one line: **the gain equals the available
headroom, times ~0.95.** Headroom = oracle(any of K feasible) - base(unselected).

| setting | base | Phi | oracle | headroom | gain | captured |
|---|---|---|---|---|---|---|
| champion, saturated set, K=8 | 0.958 | 0.958 | 0.958 | **0.000** | **0.000** | n/a |
| weak GRPO rollouts, K=8 | 0.083 | 0.192 | 0.196 | 0.113 | 0.109 | **96.5%** |
| battery, K=8 | 0.139 | 0.347 | 0.361 | 0.222 | 0.208 | **93.7%** |

Capture rate **93.7%-96.5%** across two domains, two simulators and two very
different proposers -- and the saturated case is not an exception to the law, it
is the law at H = 0: 0.95 x 0 = 0. **The +0.000 that looked like a refutation of
the whole idea was a measurement with no headroom in it.**

This reframes what Phi-selection is. It is not "a better search". It is a
near-optimal *reader* of a candidate set: given K candidates, it finds the
feasible one almost whenever one is there. Its value is therefore set entirely
by the PROPOSER -- how often the proposer puts a feasible design among its K.
That is a falsifiable and rather strong claim, so:

**Pre-registered prediction for the running API run** (truss, strong proposer,
80 held-out `problems_hard`, K=8, Phi-argmax vs a uniform draw from the same K):

> gain = 0.95 x (oracle - random), whatever headroom turns out to be.
> H = 0.05 -> +0.048   H = 0.10 -> +0.095   H = 0.20 -> +0.190   H = 0.30 -> +0.285

**What each outcome decides.** Capture near 95% again: the law holds across
proposer strength, and the paper's claim becomes quantitative -- Phi is a
near-optimal selector, and improving results means improving the proposer, not
the potential. Capture materially below ~90% with real headroom present: the law
is proposer-dependent, Phi's ranking degrades when candidates are strong and
similar, and that failure mode is itself the finding. Headroom near zero: the
run says nothing either way and needs a larger K, which is a fixable design
error rather than a result.

## Turn 11 (result) — the prediction held: Phi-selection with a STRONG proposer

`scripts/api_potential_ranking.py`, Sonnet 4.5, 80 held-out `problems_hard`,
K=8, 20 turns, paired. Each turn the model proposes 8 distinct actions in one
call; **all 8 are simulated in both arms** (`_apply_action` mutates in place, so
each candidate runs on its own `deepcopy`). The arms differ only in which of the
8 is kept: `phi` = argmax Phi, `random` = uniform draw. 160 episodes, 0 errors.

| | random | **phi** | oracle |
|---|---|---|---|
| feasible | 13/80 = 0.163 | **34/80 = 0.425** | 37/80 = 0.463 |

**+26.2pp, 95% CI [+14.5, +38.0]pp, discordant 22 vs 1, exact McNemar p = 5.7e-06.**

### The pre-registered prediction

| | |
|---|---|
| headroom H = oracle - random | 0.300 |
| **predicted** gain 0.95 x H | **+0.285** |
| **actual** gain | **+0.262** |
| miss | -0.023 |
| capture | 87.5%, bootstrap 95% CI **[70.8%, 100.0%]** |

The CI covers the prior 93.7-96.5% band, so **the headroom law survives its
first out-of-sample test**, on a proposer far stronger than either of the two it
was fitted on. The point estimate sits a little low; whether capture genuinely
degrades with proposer strength needs more n to resolve and is not claimed here.

### Selection is also cheaper, and it gets the constraint right

| | phi | random |
|---|---|---|
| simulator calls / episode | **125** | 145 |
| turns / episode | **15.6** | 18.1 |
| mass_ratio | **1.230** | 3.102 |
| FOS buckling | **1.535** | 1.247 |

Phi wins using *less* compute -- same per-turn budget, but it finishes sooner.
And note where it lands: **FOS 1.535, just above the 1.5 requirement.** It
satisfies the constraint without over-building. Random ends at 1.247, under it.

Set that against the guidance arm, which drove FOS *down* to 1.395 and
under-built (Turn 10). Same potential, same problems, same model. **Used as a
selector it lands on the constraint boundary; shown to the policy it pulls the
policy off it.** That is the sharpest statement of the result so far.

### Where the three routes stand

| route | effect on held-out feasibility |
|---|---|
| Phi in the reward | **0.000, provably** (telescopes to a group constant) |
| Phi shown to the policy | **-7.5pp**, CI excludes > +1pp |
| **Phi selecting among K proposals** | **+26.2pp** (strong proposer), +10.9pp (weak), +20.8pp (battery) |

For context on the same 80 problems: Sonnet greedy 0.287, Sonnet + Phi selection
**0.425**, the fine-tuned champion 0.167, procedural depth-1 search 0.734.

## Turn 12 — dose-response test of the headroom law (K sweep), and the SLURM post-mortem

### The K sweep

K controls how many candidates the proposer puts on the table, so it moves the
ORACLE -- the headroom -- experimentally rather than observationally. The law
says gain = 0.95 x headroom at *every* K. Running K = 2, 4, 16 at n = 80 against
the K = 8 run already in hand. A rising oracle with gain tracking it is a real
dose-response curve; gain flattening while oracle keeps rising would falsify the
"near-optimal reader" claim and show Phi's ranking degrading as candidate sets
get larger and more similar.

### SLURM post-mortem: why nothing had run for 24 hours

Two independent faults, both silent, found only by reading logs of jobs that
failed fast:

1. **This cluster does not propagate the submitting environment to batch
   scripts.** `CHECKPOINT=... sbatch job.sbatch` does NOT reach the job. The
   eval pair died in 6 minutes with `CHECKPOINT: unbound variable`. The GRPO
   pair was worse: `grpo_h100.sbatch` supplies `RL_CONFIG="${RL_CONFIG:-grpo_truss}"`,
   so it would have **silently run the wrong config**, and the empty CHECKPOINT
   would have exited 2. Both had been PENDING for 24h to do nothing. **Fix: pass
   everything through `sbatch --export=ALL,VAR=...`.** This is also why the
   original pair (44704175/6) was unauditable -- there was nothing to audit.

2. **Walltime, not priority, was the blocker.** 162 running jobs on GPU-shared
   hold 2-day reservations, so a 24h request needs a backfill window that
   effectively never opens; pending competition is mostly short (1176 jobs at
   45min, 1037 at 1:30). Dropping the request 24h -> 8h moved the eval pair from
   *24 hours pending* to **running within 10 minutes**.

Consequences for training, all fixed:
- `train_grpo.py` had **no resume support**; an 8h slice of an 18h run was
  unrecoverable. Added `resume_from_checkpoint`, with `auto` selecting the newest
  `checkpoint-*` so a requeued job continues.
- `save_steps` defaulted to **50** while 8h reaches only ~44 steps at the
  measured ~5.5 steps/h -- no checkpoint would ever have been written. Set to 10
  in both t9 configs, with `max_steps_train: 100`. Both arms carry these
  identically; `diff` confirms the configs still differ only in `lookahead_kappa`.

`scripts/eval_llm_search.py` now logs `ARGS {...}` and the resolved `keep_going`
before loading the model. The two running eval jobs confirm from their own logs:
`n_candidates` 1 and 8, **`keep_going=False` on both** -- the stop-on-feasible
confound that killed the first pair is gone.

Total cost of all four failed/DOA jobs: **~0.2 SU**.

## Turn 12 (result) — the K sweep: the law's SHAPE holds, its CONSTANT does not

80 held-out `problems_hard`, paired, Sonnet 4.5 proposing K candidates per turn,
all K simulated in both arms, `phi` = argmax Phi vs `random` = uniform draw from
the same K. 640 episodes across four K values, 0 errors.

| K | random | **phi** | oracle | headroom | gain | capture | phi/oracle | p |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.200 | 0.362 | 0.400 | 0.200 | +0.162 | 81.2% | 90.6% | 4.4e-03 |
| 4 | 0.225 | 0.400 | 0.450 | 0.225 | +0.175 | 77.8% | 88.9% | 2.6e-03 |
| 8 | 0.163 | 0.425 | 0.463 | 0.300 | +0.262 | 87.5% | 91.9% | 5.7e-06 |
| 16 | **0.113** | **0.438** | **0.475** | **0.362** | **+0.325** | 89.7% | 92.1% | **2.2e-07** |

**Shape: confirmed.** Headroom nearly doubles across the sweep (0.200 -> 0.362)
and the gain tracks it the whole way (+0.162 -> +0.325). The manipulation works
from both ends -- as K grows the oracle rises AND a uniform draw degrades
(0.200 -> 0.113), because 1-of-16 misses more often than 1-of-2. Gain could have
flattened while oracle kept climbing; that was the falsifying outcome, and it
did not happen.

**Constant: refuted.** Pooled over 320 paired problems, capture = **85.1%,
95% bootstrap CI [76.1%, 92.6%]**. The CI **excludes** the 93.7% weak-proposer
floor; only 1.0% of bootstrap draws reach it. Every individual K also sits below
the band (77.8-89.7%).

### Corrected statement of the law

Turn 11 pre-registered `gain = 0.95 x headroom`. That constant was fitted on a
weak GRPO policy (96.5%) and a procedural battery sampler (93.7%), and it does
NOT transfer. The measured relation is

> **gain = c(proposer) x headroom**, with c ~ 0.95 for weak/procedural proposers
> and **c = 0.85 [0.76, 0.93]** for a strong LLM proposer.

Equivalently: **Phi recovers ~91% of whatever its candidate set contains**
(phi/oracle 88.9-92.1%, flat in K).

The mechanism is plausible and worth stating as a hypothesis rather than a
finding: a strong proposer emits K *plausible and similar* candidates, so Phi
must discriminate among close alternatives; a weak proposer makes the feasible
one obvious. That predicts capture should fall further as proposers improve --
testable, not tested.

**What this does and does not cost the thesis.** Phi-selection remains a large,
highly significant, domain-general effect (+16 to +33pp, p from 4e-3 to 2e-7,
truss and battery, weak and strong proposers). What dies is the tidy claim that
it is a *near-optimal* reader: it leaves ~15% of the headroom on the table with a
strong proposer, and that gap is the honest target for future work on the
ranking rule itself -- the first thing in this project pointing at improving Phi
rather than around it.

## Turn 13 — RETRACTION: Phi's ranking has no headroom left; the proposer is the whole bottleneck

Turn 12 reported capture = 85.1% [76.1, 92.6] with a strong proposer and called the
missing ~15% "the first thing in this project pointing at improving Phi rather than
around it". **That was an artifact of the denominator and is retracted.**

Capture was computed as gain / (UNION oracle across both arms - random). But the two
arms diverge after the first turn they disagree, so the `random` arm walks trajectories
and sees candidate sets the `phi` arm never encounters. Charging Phi for failing to
rank candidates it was never shown is wrong. The correct measure is WITHIN-ARM: when a
feasible candidate appeared in THIS arm's own candidate set, did the arm end feasible?

| K | phi feas | phi oracle | **phi conversion** | phi misses | random conversion | random misses |
|---|---|---|---|---|---|---|
| 2 | 0.362 | 0.362 | **100.0%** | 0/80 | 94.1% | 1/80 |
| 4 | 0.400 | 0.412 | 97.0% | 1/80 | 90.0% | 2/80 |
| 8 | 0.425 | 0.425 | **100.0%** | 0/80 | 81.2% | 3/80 |
| 16 | 0.438 | 0.438 | **100.0%** | 0/80 | 69.2% | 4/80 |

**Phi misses a feasible candidate 1 time in 320.** Its argmax is essentially a perfect
reader of its candidate set. Random's conversion degrades exactly as 1/K predicts
(94% -> 69%), which is the effect the A/B was measuring all along.

So the corrected law is not "gain = 0.85 x headroom with ranking loss". It is simply:

> **Phi converts an available feasible candidate into a solved problem ~100% of the
> time. The outcome is therefore set entirely by whether the proposer puts a feasible
> candidate on the table.**

### Head-to-head on the SAME 80 held-out hard problems

| method | feasible | simulator calls |
|---|---|---|
| LLM K=16, random pick | 0.113 | 290/episode |
| **LLM K=16, Phi pick** | **0.438** | 242/episode |
| LLM K=16 candidate-set ceiling | 0.438 | -- |
| procedural greedy_critical | 0.175 | ~10 |
| **procedural lookahead_v2_d1** | **0.775** | **~115** |
| **procedural lookahead_v2_d2** | **0.850** | ~115 |

Procedural depth-1 beats LLM+Phi by **33.7pp on roughly half the simulator budget**.
In >56% of problems, NO action the LLM proposed at any of 20 turns, out of 16
candidates per turn, was feasible -- while plain enumeration found one in 77.5%.

**Consequence for the research programme: there is nothing left to win in the ranking
rule.** Every remaining SU belongs to candidate generation -- either fixing the LLM
proposer (hybrid LLM-union-procedural pools, LLM-directed enumeration, expert
iteration on Phi-selected trajectories) or accepting that on grammar-enumerable design
problems the LLM is not the right proposer and finding the regime where it is.

> **CAVEAT ON THE TABLE ABOVE, pending measurement.** The `lookahead_v2_d1` = 0.775
> and `d2` = 0.850 figures are the **macro-enabled** ladder. `candidate_actions()`
> includes `fsd_macro()`, a closed-form fully-stressed-design resize that computes a
> DIFFERENT scale factor per member analytically (buckling FOS ~ r^3, yielding ~ area).
> That is domain-specific engineering knowledge, not search. `search_ladder.py:540`
> says so itself: the macro "is a compound move the policy would need one turn per
> member to reproduce, so including it makes any comparison against the LLM a budget
> comparison." The grammar-legal variants (`lookahead_v2_d1_legal`, `_d2_legal`) were
> never run on `problems_hard`. They are running now; the "procedural beats LLM+Phi by
> 33.7pp" claim is NOT established until they report, and this entry will be amended.

## Turn 14 — RETRACTION #2: procedural search does NOT beat the LLM. A physics macro does.

Turn 13 concluded "procedural depth-1 beats LLM+Phi by 33.7pp on half the simulator
budget" and called the LLM proposer the whole bottleneck. **That comparison was
against a candidate set containing `fsd_macro()`** -- a closed-form fully-stressed
design resize that computes a DIFFERENT scale factor per member analytically from the
physics (buckling FOS ~ r^3, yielding ~ area). It is domain-specific engineering
knowledge, not search, and the grammar needs one turn PER MEMBER to express it.
`search_ladder.py:540` already said so: including it "makes any comparison against the
LLM a budget comparison."

Ran the grammar-legal variants (never previously run on `problems_hard`). Interim,
52/580 problems complete, same problems for every row:

| policy | feasible | note |
|---|---|---|
| `lookahead_v2_d1` | **0.712** | includes the FSD macro |
| **`lookahead_v2_d1_legal`** | **0.404** | same search, grammar-expressible moves only |
| `lookahead_v2_d2_legal` | 0.500 | depth 2, legal |
| `greedy_critical` | 0.231 | |
| LLM + Phi, K=16 (separate 80-problem sample) | 0.438 | |

**Removing one macro costs enumeration 30.8pp (0.712 -> 0.404).** At equal grammar the
LLM with Phi-selection (0.438) is at or slightly above depth-1 procedural search
(0.404) and below depth-2 (0.500). On the 9 problems currently in both samples the
direction agrees (LLM 0.667 vs d1_legal 0.556, paired 2-1) but n is far too small to
lean on; the ladder is still running and the intersection will grow.

**Retracted:** "the LLM proposer is the bottleneck and is badly beaten by dumb
enumeration". It is not. It is competitive with grammar-matched search.

### What is actually the bottleneck: the ACTION GRAMMAR

One compound move is worth **+30.8pp** -- larger than every method difference measured
in this entire project (Phi-selection +16 to +33pp is the only comparable effect, and
the two are complementary). Neither the ranker (Phi converts 319/320) nor the proposer
(competitive at equal grammar) is the limiting factor. **The vocabulary is.**

Both agents are confined to single-parameter moves while the binding physics requires
a coordinated, per-member resize. That is why the champion earns reward 0.004 on
`problems_hard` (Turn 13) -- not because the policy is bad, but because reaching
feasibility needs a coordinated move the grammar cannot say in one turn.

**The direction this implies, and it is new:** enrich the action space with compound
moves (an FSD-style macro as a first-class grammar action, or per-member factor
vectors), then apply Phi-selection over that richer candidate space. Phi is a
near-perfect reader of whatever it is shown; the leverage is in what can be shown to
it. This is testable cheaply on the API path: give the LLM the macro action in its
grammar and re-run the K=16 Phi-selection A/B.

## Turn 15 — the interface, not the intelligence: three measured bottlenecks

The workflow (16 agents, 5 plans, 10 adversarial judges) independently reached the
same diagnosis as Turn 14 and scored the proposer angle highest (8.0/10). Verifying
its two headline flags found one overstated and one real.

**Overstated: "every champion 0.167 claim is stale."** Not supported. On the SAME 32
problems, `eval_checkpoint.py` gives 0.250 and `eval_llm_search.py` gives 0.312,
discordant 4-2 (not significant). The 0.167 is n=60 on the same evaluator; the first
32 of the sorted split are simply easier. Sample composition, not a stale number.

**Real, and new: `mean_grammar_success_rate` is 0.592 vs 0.769 across the two eval
paths.** Even in the better path ~23% of the champion's actions fail to execute.

### The observation bottleneck, shown concretely

`hard_problem_0002`, 10 members. What the simulator knows per member:

| member | fos_buckling | fos_yielding | r |
|---|---|---|---|
| M0 | inf | 90.96 | 0.005 |
| M1 | inf | 10.74 | 0.005 |
| **M2** | inf | **3.96e17** | 0.0077 |
| M3 | inf | 12.25 | 0.005 |
| M4 | 1.150 | 3.245 | 0.0344 |
| M5 | inf | 19.61 | 0.0065 |
| **M6** | **0.934** (binding) | 3.040 | 0.0317 |
| M7 | 7.55 | 342.2 | 0.0122 |

What `format_eval_result` shows the policy: **`fos_b = 0.934`, `fos_y = 3.040`** and
"worst member: M6". That is it -- two scalars, the minima.

With the table the move is immediate: grow M6 and M4, shrink M0/M1/M2/M3/M5/M7 hard
(M2 carries 4e17 yielding margin -- free mass). Without it the policy knows buckling
fails at M6 but cannot know six members are grossly oversized, so it cannot cut mass
deliberately. `fsd_macro` reads these per-member values straight off the member
objects; the policy never sees them. **Procedural search's 30.8pp edge is an
INFORMATION advantage as much as an action one.**

### The three bottlenecks, all interface rather than intelligence

| bottleneck | evidence | cost |
|---|---|---|
| action grammar: no compound move | `fsd_macro` removed -> 0.712 to 0.404 | **~31pp** |
| observation: minima only, no per-member vector | table above; FSD needs fos[i] for all i | prerequisite for the above |
| grammar success: actions that fail to execute | 0.592 / 0.769 across eval paths | ~23% of turns |

Against this: Phi converts 319/320, and at matched grammar the LLM (0.438) is ABOVE
depth-1 enumeration (0.404). **Neither the potential nor the policy is the limit. The
interface between them starves both.** The agent is asked to solve a 10-variable
sizing problem while shown 2 numbers, and to express the answer one variable per turn.

### E1, the next experiment (free, API + CPU)

2x2 on the 202 held-out problems, Phi-selection at fixed K in every cell:
observation (aggregate | per-member FOS table) x action (single | compound).
`search_ladder.py:84 apply_action` already parses MACRO_SEP-joined actions, so the
compound arm is a port into `truss_env.py`, not new semantics. Decision rule: if
per-member + compound clears ~0.7 (procedural-with-macro territory) the ceiling was
self-imposed and the method works; if compound alone moves it but per-member does not,
the grammar was the whole story; if neither moves, the LLM cannot compose
heterogeneous magnitudes and the region abstraction (LLM picks the SET, a procedure
picks the magnitudes) is the remaining route.

## Turn 16 (RESULT) — E1: the interface, not the intelligence. +22.8pp, p=0.0014

`scripts/api_grammar_2x2.py`. Sonnet 4.5, 80 held-out `problems_hard`, K=8, 20 turns,
Phi-selection in EVERY cell, paired. 320 episodes, 1 errored and dropped, 79 problems
with all four cells. A compound candidate is ONE decision costing ONE simulator call
(same accounting as `search_ladder.apply_action`), so the cells are budget-matched by
construction. The FSD formula was deliberately NOT given -- the per-member arms get
the table and one line saying FOS far above 1.5 means excess material.

| observation | action | feasible | oracle | parts/cand | sims/ep | mass_ratio |
|---|---|---|---|---|---|---|
| aggregate | single | 0.380 | 0.405 | 1.00 | 120 | 1.595 |
| per_member | single | 0.494 | 0.519 | 1.00 | 110 | 1.127 |
| aggregate | compound | **0.304** | 0.304 | 5.03 | 122 | 1.867 |
| **per_member** | **compound** | **0.608** | 0.620 | 5.21 | **90** | **1.106** |

Paired vs the status quo (`aggregate/single`):

| cell | effect | 95% CI | discordant | exact p |
|---|---|---|---|---|
| per_member/single | +11.4pp | [-0.5, +23.3] | 16 vs 7 | 0.093 |
| aggregate/compound | **-7.6pp** | [-16.9, +1.7] | 4 vs 10 | 0.180 |
| **per_member/compound** | **+22.8pp** | **[+9.2, +36.4]** | **24 vs 6** | **0.0014** |

### The interaction is the result: +19.0pp

| effect of allowing COMPOUND moves | |
|---|---|
| when the agent sees only aggregates | **-7.6pp** (0.380 -> 0.304) |
| when the agent sees per-member state | **+11.4pp** (0.494 -> 0.608) |

**A richer action vocabulary is HARMFUL without the information to aim it.** The model
uses the capability either way -- `parts/cand` is 5.0 in both compound arms, versus
exactly 1.00 in both single arms, so this is not a "wouldn't" but a "couldn't". Given
compound moves and only two scalars, it makes uninformed multi-member changes and ends
*heavier* (mass_ratio 1.867, the worst cell). Given the per-member table, the same
capability produces near-optimal designs (mass_ratio 1.106) on 25% FEWER simulator
calls (90 vs 120).

Neither ingredient alone reaches significance; only the combination does. Note also
that `aggregate/single` reproduces at 0.380 against the independently measured 0.425
at K=8, so the harness validates.

### Where this leaves the project

Nothing about the model, the potential, or the search changed. Only what the agent can
SEE and SAY. That single change is worth more than every Phi variant measured in this
project, and it moves the LLM from grammar-matched procedural search (d1_legal 0.404)
most of the way to MACRO-ARMED procedural search (d1 0.712) -- which is the point,
since the macro is exactly the move the interface was preventing.

Phi remains essential and unchanged: conversion in the winning cell is 0.608/0.620 =
98%, consistent with the 319/320 measured in Turn 13. **Phi reads the candidate set
near-perfectly; the interface determines what is in it.**

Caveat: single seed, n=79. The effect is large and p=0.0014, but two headline claims
were retracted earlier today for over-reading, so +22.8pp is "real and large", not a
precise constant.

## Turn 17 — the grammar-legal ladder settles it: the macro is worth +36.2pp, and the LLM WINS at matched grammar

`results/search_ladder/hard_legal/`, 326 of 580 problems complete (a session restart
killed the sweep; incremental writes preserved everything, and at this n the result is
not close). Same problems for every row.

| policy | rate | n |
|---|---|---|
| greedy_critical | 0.126 | 326 |
| **lookahead_v2_d1_legal** | **0.393** | 326 |
| **lookahead_v2_d2_legal** | **0.460** | 326 |
| lookahead_v2_d1 (macro-armed) | **0.755** | 326 |
| lookahead_v2_d2 (macro-armed) | 0.813 | 326 |

**The FSD macro alone is worth +36.2pp** (0.755 vs 0.393), paired **discordant 121 vs
3, exact p = 3e-32**. That is the largest single effect measured anywhere in this
project -- larger than Phi-selection, larger than the interface fix, larger than every
reward and training variant combined.

### Head-to-head at MATCHED grammar (46 problems in both runs)

| method | rate |
|---|---|
| procedural d1, grammar-legal | 0.283 |
| procedural d2, grammar-legal | 0.370 |
| **LLM + Phi, K=16** | **0.457** |
| procedural d1, macro-armed | 0.717 |

LLM vs d1_legal, paired: **LLM-only 11, d1-only 3, exact p = 0.057.**

**"Dumb enumeration beats the LLM" was backwards.** At matched grammar the LLM is
AHEAD of depth-1 and depth-2 procedural search. Turn 13's claim is now doubly refuted:
first in direction (Turn 14), now in magnitude and with a paired test.

### The complete ladder, one scale

| system | rate |
|---|---|
| greedy critical-member | 0.126 |
| procedural d1, grammar-legal | 0.283-0.393 |
| procedural d2, grammar-legal | 0.370-0.460 |
| LLM + Phi, status-quo interface | 0.438-0.457 |
| **LLM + Phi + per-member obs + compound actions** | **0.608** |
| procedural d1, macro-armed | 0.717-0.755 |
| procedural d2, macro-armed | 0.813 |

The interface fix (Turn 16) closes roughly **half** the distance between the
status-quo LLM and macro-armed procedural search, using the same model, the same
potential and the same search -- only a different observation and a compound action.
The remaining gap is the part of `fsd_macro` that is genuine closed-form physics: it
solves the sizing subproblem analytically, which no general agent should be expected
to match by proposing candidates.

**This is the honest framing of the contribution.** On a domain where the sizing
subproblem has a closed form, use the closed form. The method's claim is for design
problems where it does not -- and there the ingredients that matter are, in order:
(1) an action vocabulary that can express a coordinated move, (2) an observation that
exposes per-component slack so the move can be aimed, (3) Phi to rank the results.

## Turn 17 — FINAL numbers: the macro is worth 36.3pp, and the interface recovers most of it

### Grammar-legal ladder, 383/580 problems (6h walltime; incremental writes preserved all of it)

| policy | feasible |
|---|---|
| greedy_critical | 0.120 |
| **lookahead_v2_d1_legal** | **0.381** |
| lookahead_v2_d2_legal | 0.452 |
| lookahead_v2_d1 (macro-armed) | **0.744** |
| lookahead_v2_d2 (macro-armed) | 0.812 |

**Removing `fsd_macro` costs depth-1 enumeration 36.3pp (0.744 -> 0.381)** -- larger
than the 30.8pp interim and larger than any method difference measured in this project.

Paired on the 51 problems present in both the LLM run and the legal ladder:
**LLM + Phi K=16 = 0.431 vs procedural d1_legal = 0.314**, discordant 11 vs 5,
exact p = 0.21. Not significant at n=51, but the LLM is AHEAD. Turn 13's claim that
"dumb enumeration beats the LLM by 33.7pp" is fully retracted: it never did. One
closed-form physics move did.

### What the interface fix recovers

| | feasible |
|---|---|
| LLM, status-quo interface | 0.380 - 0.429 |
| procedural d1, grammar-legal | 0.381 |
| procedural d2, grammar-legal | 0.452 |
| **LLM, per-member obs + compound actions** | **0.608 - 0.686** |
| procedural d1, MACRO-ARMED | 0.744 |
| procedural d2, MACRO-ARMED | 0.812 |

Giving the policy what the procedural searcher always had -- per-component state and a
heterogeneous simultaneous move -- lifts it from grammar-legal parity (0.38) most of
the way to the macro-armed searcher (0.744), **without giving it the FSD formula**.
It derives the move from the table.

### Replication of the 2x2 (same 70 problems, fresh rollouts at T=0.7)

| | original n=79 | replication n=70 |
|---|---|---|
| per_member/compound | +22.8pp, p=0.0014 | **+25.7pp, p=0.00028** |
| interaction | +19.0pp | **+17.1pp** |
| aggregate/compound | -7.6pp | -5.7pp |

Stable under resampling. NOTE: both runs shuffle with `random.Random(0)` over the same
202 ids, so `specs[:80]` and the first 70 of `specs[:202]` are the SAME problems --
this is a rollout replication, not an independent sample. The genuinely new 122
problems arrive later in the full run.

### t9 per-token A/B is finally a real experiment

| | reward | reward_std | grad_norm | lookahead/rho |
|---|---|---|---|---|
| control kappa=0 | 0.2316 | 0.269 | 0.0646 | 0.340 |
| treat kappa=1 | 0.2556 | 0.282 | **0.0932** | 0.351 |
| voided single-turn run | **0.0042** | -- | -- | -- |

**Multi-turn reward is 0.23, not 0.004 -- 55x larger.** Turn 13's "the hard benchmark
may be too hard for RL" was purely an artifact of the missing `multi_turn: true`, and
is retracted. The probe is live (241/231 states) and the treatment's gradient norm is
44% larger, consistent with kappa adding orthogonal per-step credit. At ~4 steps/h both
arms will stop near step 32 of 100; `save_steps: 10` preserves progress and both cut at
the same step, so the equal-step comparison holds.

## Turn 18 — CORRECTION: the battery Phi-selection p-value was inflated 10x by pseudo-replication

`results/battery_ladder/group_selection.jsonl` has 72 rows, and Turn 8 reported them
as n=72 independent problems (+20.8pp, exact McNemar p=2.75e-4). They are **24
distinct problems x 3 seeds each** -- `battery_group_selection.py` takes `--seeds`,
and every `problem_id` appears exactly 3 times. Three seeds on one problem are not
three independent observations, so the McNemar test was pseudo-replicated.

Recomputed with the problem as the unit of analysis:

| analysis | effect | inference |
|---|---|---|
| as reported (rows independent) | +20.8pp | p = 2.75e-4 |
| **clustered by problem (n=24)** | **+20.8pp** | **t=3.50, 95% CI [+9.2,+32.5]pp, sign test p=1.95e-3** |

Per-problem sign test: **10 improved, 0 worsened, 14 tied.** The point estimate is
unchanged and the conclusion survives -- Phi-selection does generalise to batteries --
but the p-value was an order of magnitude too good and the headline should read
p~0.002, not p~0.0003.

Note the same question does NOT arise for the truss results: `phi_vs_random*.jsonl`
and `grammar_2x2*.jsonl` carry one episode per (problem, arm), so those McNemar tests
are correctly specified.

Also recorded: the battery set is **24 problems**, not 72. `make_battery_problems.py`
and `make_battery_problems_hard.py` exist if more are needed. 24 clusters were
sufficient to detect a ~20pp effect at p=0.002, which is what makes the battery 2x2
(Turn 19) viable at this n.

## Turn 19 — the battery generality test is a NON-MEASUREMENT: the battery set is saturated

`scripts/api_battery_2x2.py`, 24 battery problems x 4 cells, K=8, 12 turns, PyBaMM
(spme_lean, 8A), Phi-selection in every cell. 96 episodes, 0 errors, 8 minutes on
RM-shared (~2 SU).

| observation | action | feasible | oracle | parts/cand | sims/ep |
|---|---|---|---|---|---|
| aggregate | single | **0.875** | 0.875 | 1.00 | 27.0 |
| full | single | 0.833 | 0.833 | 1.00 | 23.3 |
| aggregate | compound | 0.833 | 0.833 | 3.72 | 19.4 |
| full | compound | 0.792 | 0.792 | 4.53 | 24.8 |

Interaction **-0.0pp** (truss: +18.6pp). Every contrast is inside noise: discordant
2-3, 2-3, 0-2; p = 1, 1, 0.5.

**This does not refute the truss result, because the experiment could not have
detected it.** The status-quo interface already solves 21/24 problems, so the entire
experiment contains 12.5pp of headroom -- a +20pp effect is not expressible in it.
And `oracle == feasible` in all four cells, i.e. Phi again converted everything
available (consistent with 319/320), so there is no ranking loss to find either.

Splitting by family makes it worse, not better:

| subset | n | status quo | headroom |
|---|---|---|---|
| easy | 12 | 0.833 | 16.7pp |
| **hard** | 12 | **0.917** | **8.3pp** |

The `_hard` families are MORE saturated than the easy ones. These problems were
calibrated against a PROCEDURAL sampler -- the setting where random-pick scores 0.139
and Phi-pick 0.347 (Turn 8/18). A strong LLM proposer with K=8 over 12 turns solves
87.5% of them. **The difficulty was tuned for a different agent.**

### The recurring failure mode, named

This is the THIRD time saturation has invalidated a measurement here:
1. the original 100-problem truss set -- champion 0.958, nothing separable (Turn 7);
2. the Phi-selection headline on that set -- +0.000 because headroom was 0 (Turn 7);
3. this battery 2x2 -- baseline 0.875 (Turn 19).

The rule it implies: **before running any comparison, measure the arm-free baseline
and the oracle on the exact problem set, and require the gap to exceed the effect
being tested.** The truss hard set was built for exactly this reason and it worked --
status quo 0.380, oracle 0.663, a 28pp gap that comfortably contained a 22.8pp effect.

**Consequence:** the generality claim for the interface result is UNTESTED, not
refuted. Testing it needs battery problems calibrated to a strong LLM agent (target
status-quo ~0.3-0.4), which `make_battery_problems_hard.py` can generate but with
tighter constraints than the current set. Until then the interface finding stands as
a truss result with a plausible but unverified domain-general mechanism.

## Turn 18 (FINAL) — E1 on the full held-out set: +19.2pp, p = 1.1e-05

`scripts/api_grammar_2x2.py`, Sonnet 4.5, **all 202 held-out `problems_hard`**, K=8,
20 turns, Phi-selection in every cell, paired. 808 episodes, 1 dropped, **198 problems
with all four cells**.

| observation | action | feasible | oracle | parts/cand | sims/ep | mass_ratio |
|---|---|---|---|---|---|---|
| aggregate | single | 0.444 | 0.444 | 1.00 | 118 | 1.334 |
| per_member | single | 0.556 | 0.566 | 1.00 | 100 | 1.158 |
| aggregate | compound | **0.384** | 0.394 | 4.90 | 117 | 1.878 |
| **per_member** | **compound** | **0.636** | 0.646 | 5.40 | **87** | 1.128 |

| effect vs status quo | effect | 95% CI | discordant | exact p |
|---|---|---|---|---|
| information alone | +11.1pp | [+2.7, +19.5] | 47 vs 25 | **0.013** |
| vocabulary alone | -6.1pp | [-13.2, +1.1] | 20 vs 32 | 0.126 |
| **both** | **+19.2pp** | **[+10.7, +27.7]** | **56 vs 18** | **1.1e-05** |

**Interaction +14.1pp** (compound is -6.1pp with aggregate observation, +8.1pp with
per-member observation). Effect shrank from +25.7pp at n=70 to +19.2pp at n=198 --
ordinary regression -- while p fell three orders of magnitude.

The winning cell is better on every axis: +19.2pp feasibility, mass_ratio 1.334 ->
1.128, and **26% FEWER simulator calls** (87 vs 118).

`parts/cand` is exactly 1.00 in both single arms and ~5 in both compound arms, so the
model always uses the capability when it is legal. Compound moves are therefore not
unused-and-neutral; they are actively HARMFUL without the state to aim them (0.384,
mass_ratio 1.878 -- the worst cell on both axes).

### Battery generality attempt: SATURATED, a non-measurement

`scripts/api_battery_2x2.py` on `data/battery_problems` (24 problems, ~2 SU) returned
aggregate/single **0.875**, full/single 0.833, aggregate/compound 0.833, full/compound
0.792; every discordant count 2-vs-3 or smaller, all p >= 0.5, interaction -0.0pp.

**The status quo already solves 21/24, so nothing could be measured** -- the same
saturation trap that invalidated the original truss benchmark. This is NOT evidence
against generality. `data/battery_problems_hard` does not exist even though
`scripts/make_battery_problems_hard.py` does; generating and difficulty-certifying it
(as was done for `problems_hard`) is the prerequisite for any battery claim.

Note in passing, NOT a like-for-like comparison: the LLM scores 0.875 here while the
procedural sampler's oracle on battery was 0.361 (Turn 8). Different protocols (12-turn
LLM episode vs single-step K=8 procedural candidates), so it cannot be read as
"LLM beats procedural on battery" without a matched experiment.

## Turn 20 (FINAL) — the interface result on the FULL held-out split, n=202

`scripts/api_grammar_2x2.py`, all 202 held-out `problems_hard`, K=8, 20 turns,
Phi-selection in every cell, paired. 808 episodes, 0 errors, 202/202 problems with
all four cells. (A session restart killed the first attempt at 271 episodes;
`--resume-from` carried them forward rather than re-running them.)

| observation | action | feasible | oracle | parts/cand | sims/ep | mass_ratio |
|---|---|---|---|---|---|---|
| aggregate | single | 0.460 | 0.460 | 1.00 | 116 | 1.273 |
| per_member | single | 0.564 | 0.574 | 1.00 | 99 | 1.157 |
| aggregate | compound | **0.381** | 0.391 | 4.86 | 118 | 1.820 |
| **per_member** | **compound** | **0.629** | 0.639 | 5.33 | **87** | **1.116** |

| contrast | effect | 95% CI | discordant | exact p |
|---|---|---|---|---|
| information alone | **+10.4pp** | [+2.2, +18.6] | 46 vs 25 | **0.017** |
| vocabulary alone | **-7.9pp** | [-15.1, -0.8] | 19 vs 35 | **0.040** |
| **both** | **+16.8pp** | **[+8.7, +24.9]** | **52 vs 18** | **5.9e-05** |

**Interaction +14.4pp**: compound actions are worth **-7.9pp** with aggregate
observation and **+6.4pp** with per-member observation.

At full n every contrast is significant, including the NEGATIVE one. That is the
strongest form of the claim: the two ingredients are not merely sub-additive, one of
them is **actively harmful on its own**. Expanding an agent's action space while its
observation stays aggregated makes it worse, and now with p=0.040 rather than as a
direction. `parts/cand` is 1.00 vs ~4.9, so the model uses the capability in both
compound arms -- this is a "couldn't aim it", not a "wouldn't use it". Mass ratio
tells the same story: 1.820 (worst cell, uninformed multi-parameter changes) vs
1.116 (best), on 25% fewer simulator calls.

### Effect-size drift, stated plainly

| n | effect | p |
|---|---|---|
| 79 | +22.8pp | 0.0014 |
| ~86 | +22.1pp | 0.0013 |
| **202** | **+16.8pp** | **5.9e-05** |

The estimate fell by about a third as n grew -- textbook regression toward the mean,
and the early headline was optimistic. **+16.8pp [8.7, 24.9] is the number to use.**
Significance strengthened even as the point estimate shrank, which is what a real
effect measured on more data looks like.

### Where the ladder now stands (all on `problems_hard`)

| system | rate |
|---|---|
| greedy critical-member | 0.126 |
| procedural d1, grammar-legal | 0.393 |
| **LLM + Phi, status-quo interface** | **0.460** |
| procedural d2, grammar-legal | 0.460 |
| per-member observation alone | 0.564 |
| **LLM + Phi + per-member + compound** | **0.629** |
| procedural d1, macro-armed (closed-form physics) | 0.755 |
| procedural d2, macro-armed | 0.813 |

The interface change moves the LLM from parity with grammar-legal depth-2 search to
within 13pp of macro-armed depth-1 search -- with no change to the model, the
potential, or the search procedure.

### Battery, split by tier: BOTH saturated. Tier-2 de-saturates the wrong thing.

| tier | n | status-quo baseline |
|---|---|---|
| tier-1 `bat_*` (one action suffices) | 12 | 0.833 |
| **tier-2 `bat_h_*`** (0 of 52 one-step neighbours feasible) | 12 | **0.917** |

The tier-2 set is harder by construction and the LLM scores HIGHER on it. Every
contrast is 0-2 discordant pairs at p >= 0.5; the two "interactions" (-16.7pp,
+16.7pp) are the same noise with opposite signs.

**Why tier-2 does not de-saturate a multi-turn agent.** `make_battery_problems_hard.py`
certifies that no single grammar action reaches feasibility *from the start state*
(`n1 == 0 and n2 >= 2`). That defeats a DEPTH-1 SEARCH. A 12-turn agent simply composes
the two steps across two turns, so the construction costs it nothing. The difficulty is
a property of the search depth, not of the problem.

Contrast `problems_hard` for truss, which binds the MASS BUDGET so that 43% of problems
start over budget -- a property of the problem that holds however many turns are taken.
That is why it de-saturated (champion 0.958 -> 0.167-0.31) and tier-2 did not.

**Consequence: generality is untested, not refuted.** A battery set that is hard for a
multi-turn agent needs the truss recipe -- tighten a budget-like constraint until the
START state violates it by a margin no short composition can recover -- not the
depth-1 recipe. Until that exists, no battery number in this project can speak to the
interface result. Cost of learning this: ~2 SU.

## Turn 21 — battery generality: calibrated, run, and still UNDERPOWERED (not refuted)

Turn 19's battery test was saturated (status quo 0.875, 12.5pp headroom). Fixed by
`scripts/tighten_battery_problems.py`, which scales each problem's binding constraint
and lets difficulty be swept. Calibration ran the STATUS-QUO cell only at three
factors (cheap: 24 episodes each):

| tightening | status-quo feasible | headroom |
|---|---|---|
| 0.00 (as shipped) | 0.875 | 12.5pp |
| 0.10 | 0.583 | 41.7pp |
| 0.20 | 0.625 | 37.5pp |
| **0.30** | **0.458** | **54.2pp** |

0.30 matches the hard truss set's status quo (0.460) almost exactly. Why the shipped
set was too easy: `make_battery_problems_hard.py` tightens until "0 of 52 one-step
neighbours is feasible but a two-step composition is" -- a criterion defined by
PROCEDURAL SEARCH DEPTH. An LLM permitted a compound move covers a two-step
composition in ONE turn, so that criterion does not constrain it. **Difficulty must be
calibrated to the agent being measured, not to the search that generated the set.**

### The result on the calibrated set (24 problems, 96 episodes, 0 errors)

| observation | action | feasible | oracle | parts/cand | sims/ep |
|---|---|---|---|---|---|
| aggregate | single | 0.417 | 0.417 | 1.00 | 51.5 |
| full | single | 0.458 | **0.542** | 1.00 | 55.6 |
| aggregate | compound | 0.458 | 0.458 | 4.04 | **31.0** |
| full | compound | 0.458 | 0.458 | 5.05 | 47.4 |

All three contrasts +4.2pp, p=1. **Interaction -4.2pp** (truss: +14.4pp).

### Why this is NOT a refutation

| study | n | discordant | MDE |
|---|---|---|---|
| truss | 202 | 70 | 11.6pp |
| **battery** | **24** | **3** | **20.2pp** |

**The battery MDE exceeds the effect it was testing.** A +14 to +17pp interaction
could not have been detected. Headroom was adequate this time (58pp) -- the binding
limit is n, not saturation.

The low discordance is itself the signal: in truss the same manipulation flipped
52 problems one way and 18 the other; in battery it flips 2-3 total. Whatever the
interface does in truss, it barely moves outcomes over 9 scalar parameters.

**Hypothesis worth testing (not tested):** the interface bottleneck scales with the
number of components requiring COORDINATED, OPPOSING changes. A truss needs M6 grown
while M2 shrinks, across 10-20 members -- impossible to express one member per turn
within 20 turns. A battery has 9 parameters and 12 turns, so sequential single moves
can cover the space. If true, the claim is not "LLM design agents need compound
actions" but "they need them when the coupled subproblem is larger than the turn
budget" -- a sharper and more useful statement.

**To settle it:** ~50 genuinely new battery problems (n=50 gives MDE ~14pp at the
observed discordance rate). Pooling the existing 24 bases across the three tightening
factors would be pseudo-replication -- the same error corrected in Turn 18 -- and must
not be done.

Also noted: `full/single` is the FIRST observed Phi conversion loss (feasible 0.458 vs
oracle 0.542, i.e. 85%), against 319/320 in truss. Two problems where a feasible
candidate was present and Phi did not take it. Worth a look if battery work continues.

## Turn 19 — battery generality: correctly calibrated, but UNDERPOWERED at n=24

A difficulty calibration (job 44817739) tightened the battery goals at fractions
0.10/0.20/0.30 and measured the status-quo cell at each; **t030 gives 0.458, matched
deliberately against truss `problems_hard` at 0.460**. That is the right methodology --
it de-saturates on a property of the PROBLEM (a binding budget), not on search depth,
which is where `make_battery_problems_hard.py`'s tier-2 construction failed.

2x2 on `data/battery_problems_t030` (job 44819726), n=24, all four cells:

| observation | action | feasible | oracle | parts/cand |
|---|---|---|---|---|
| aggregate | single | 0.417 | 0.417 | 1.00 |
| full | single | 0.458 | 0.542 | 1.00 |
| aggregate | compound | 0.458 | 0.458 | 4.04 |
| full | compound | 0.458 | 0.458 | 5.05 |

Every contrast is **+4.2pp = exactly one problem of 24**, discordant 2-vs-1 or 1-vs-0,
all p = 1. Interaction -4.2pp, i.e. also one problem.

**This is a non-result on power grounds, and must not be read as a refutation.**

| n | discordant | MDE |
|---|---|---|
| **24** | ~3 | **~20.2pp** |
| 48 | ~8 | ~16.5pp |
| 72 | ~12 | ~13.5pp |
| 120 | ~20 | ~10.4pp |

The truss effect is **+19.2pp**. At n=24 the battery experiment can only resolve
effects of ~20pp or larger, i.e. it sits exactly ON the truss effect size. A null here
cannot distinguish "no effect in battery" from "the same effect as truss". The
`parts/cand` mechanism check does pass (1.00 single vs 4.0-5.1 compound), so the
capability is being used.

**What generality actually needs: more battery problems.** Only 24 exist, and all 24
are used. Reaching MDE ~13pp needs n>=72, which means generating and scan-certifying
more (variant, current) combinations -- `results/battery_ladder/scan_*.jsonl` currently
covers 16. Until then the interface result stands as a TRUSS result, replicated within
truss (n=198, p=1.1e-05) but not across domains.

### Turn 21b — the "coupled subproblem size" hypothesis is REFUTED by existing data

Turn 21 proposed that the interface effect scales with the number of components
needing coordinated change, which would explain why battery (9 parameters) showed
nothing while truss (10-21 members) showed +16.8pp. The 202-problem run already
carries member counts, so this cost nothing to check.

| subset | n | status quo | best cell | effect | exact p |
|---|---|---|---|---|---|
| few members (<=15) | 106 | 0.396 | 0.613 | **+21.7pp** | 0.0006 |
| many members (>15) | 96 | 0.531 | 0.646 | **+11.5pp** | 0.052 |

| members | n | effect |
|---|---|---|
| 8-11 | 59 | +20.3pp |
| 12-15 | 47 | **+23.4pp** |
| 16-19 | 36 | +19.4pp |
| 20-23 | 60 | **+6.7pp** |

**The effect SHRINKS with member count, not grows.** The prediction was backwards, so
the hypothesis is dead and the battery null remains unexplained.

Caveat on reading the buckets: status-quo rates vary a lot across them (0.298 to
0.778), so ceiling and floor effects confound a clean dose-response reading. The
16-19 bucket reaches 0.972 in the best cell -- near ceiling -- while 20-23 stays at
0.450 in both arms, i.e. those problems are hard in a way the interface does not
address. What survives is only the negative result: effect size does not increase
with the number of coupled components.

Remaining candidate explanations for the battery null, none tested:
- battery's `aggregate` observation already names the worst violated constraint AND
  its relative violation, which may be more informative than truss's bare min-FOS;
- battery constraints may be more separable/monotone in the design parameters, so a
  sequence of single moves reaches the same place a compound move would;
- 12 turns vs 20;
- or the effect is present and the study simply could not see it (MDE 20.2pp).

The last is sufficient on its own, so no mechanism should be claimed until n~50.

## Turn 22 — T9: per-token potential credit shows NO benefit. All three policy routes are now closed.

Jobs 44809480 / 44809481, both TIMEOUT at the 8h walltime having reached
checkpoint-40 (control 43 steps, treat 39). Configs verified identical except
`lookahead_kappa` (0.0 vs 1.0); both on `LookaheadGRPOTrainer`, `multi_turn: true`,
`learning_rate: 5.0e-6`, same champion init, same `rl_problems_hard` data, same
48-candidate lookahead probe every turn. Compared at matched logged steps.

| step | control reward (k=0) | treat reward (k=1) |
|---|---|---|
| 10 | 0.2316 | 0.2556 |
| 20 | 0.2973 | 0.2357 |
| 30 | 0.2125 | 0.1710 |
| 40 | 0.2488 | 0.2073 |

Mean over matched steps: control **0.2475**, treat **0.2174**, difference **-0.030**
(paired t = -1.65 on 4 points, p ~ 0.2). KL 0.00131 vs 0.00146. **Directional null,
treatment slightly worse, not significant.**

Two caveats: 40 of the intended 100 steps (the probe costs ~600s/step, so 8h buys
~45), and only 4 logged points, so this is weak evidence. A proper endpoint would be
held-out evaluation of the two checkpoint-40 policies rather than training reward.

**It also confirms Turn 13's retraction.** Reward here is 0.17-0.30, against the
0.004 that prompted "the hard benchmark may be too hard for RL". That figure was the
single-turn config bug; with `multi_turn: true` the reward signal is normal. The
benchmark was never the problem.

### The complete picture for Phi

| route | mechanism | result |
|---|---|---|
| reward | shaping term added to the GRPO reward | **provably zero** -- telescopes to a group constant deleted by mean-centring (5 tests) |
| prompt | Phi shown to the policy in its observation | **-7.5pp**, CI excludes >+1pp; causes under-building (FOS 1.674 -> 1.395) |
| **per-token gradient** | `A_token = A_seq + kappa*zscore(A_step)` | **no benefit** (-0.030 reward at matched steps, p~0.2) |
| **selection** | Phi ranks already-generated candidates | **works**: +16 to +33pp, truss and battery, 319/320 conversion |

**Every route by which Phi could improve the POLICY has now failed, by three
different mechanisms -- algebraic, empirical, and gradient-level. Phi is an
inference-time selector and nothing else.** That is the honest scope of the
potential-shaping contribution, and it is a complete answer rather than an open
question.

What raised end-task performance was never Phi: it was the INTERFACE (Turn 20,
0.460 -> 0.629, p=5.9e-05), which is orthogonal to Phi and composes with it.

## Turn 23 — CONFOUND FOUND in the 2x2 design: asymmetric prompt examples

Running the identical 2x2 with Haiku 4.5 (agent-axis generality) exposed a flaw in
the experiment itself. Interim Haiku numbers looked spectacular -- per_member/compound
+39.3pp, interaction +32.1pp -- but the diagnostic column gave it away:

| cell | parts/cand | sims/ep |
|---|---|---|
| aggregate/single | **0.34** | 27 |
| per_member/single | **0.20** | 16 |
| aggregate/compound | 5.08 | 116 |
| per_member/compound | 4.81 | 106 |

`parts/cand` must be exactly 1.00 in a single-action arm. 0.34 means **most Haiku
episodes produced no parseable action at all**, and `sims/ep` (27 and 16 vs 116 and
106) confirms far fewer candidates ever reached the simulator.

**Cause, in my own script.** `COMPOUND_RULE` contains a worked example:
`<action>SCALE_PARAM(0, r, 0.7) ; SCALE_PARAM(3, r, 1.4) ; ...</action>`. The single
arms were given **no example at all**. So the compound arms carried a format-compliance
advantage independent of the compound capability. Sonnet parsed 100% either way
(parts/cand exactly 1.00, sims/ep 116/99 vs 118/87) so it never surfaced; a weaker
model made it obvious.

**Fix:** `SINGLE_RULE` gives the single arms an equivalent worked example, so the arms
differ only in what an action may CONTAIN, not in how well the format is demonstrated.

**Status of the headline.** The Sonnet +16.8pp (Turn 20) shows no parse asymmetry, so
it is probably sound -- but "probably" is not good enough for the main result of the
session, and the existing data cannot separate a candidate-QUALITY effect of the
example from the compound-action effect. **Turn 20 is provisional until the symmetric
re-run reports.** The Haiku numbers above are discarded outright.

This is the fifth confound of the session found by checking a mechanism rather than
accepting a number, and the second found only because a result looked too good.

## Turn 24 (DEFINITIVE) — symmetric-prompt 2x2, n=202. Headline confirmed, "harm" claim RETRACTED.

Re-run of Turn 20 with `SINGLE_RULE` giving the single arms an equivalent worked
`<action>` example (Turn 23 confound). 808 episodes, 0 errors, 202/202 problems with
all four cells. `parts/cand` is **exactly 1.00** in both single arms, so the parse
asymmetry is gone and the arms differ only in what an action may CONTAIN.

| observation | action | feasible | oracle | parts/cand | sims/ep | mass_ratio |
|---|---|---|---|---|---|---|
| aggregate | single | 0.426 | 0.436 | 1.00 | 118 | 1.277 |
| per_member | single | 0.490 | 0.500 | 1.00 | 113 | 1.134 |
| aggregate | compound | 0.421 | 0.441 | 4.93 | 115 | 1.589 |
| **per_member** | **compound** | **0.609** | 0.614 | 5.22 | **89** | **1.110** |

| contrast | effect | 95% CI | discordant | exact p |
|---|---|---|---|---|
| information alone | +6.4pp | [-0.9, +13.8] | 35 vs 22 | 0.111 |
| vocabulary alone | **-0.5pp** | [-7.7, +6.7] | **27 vs 28** | **1.0** |
| **both** | **+18.3pp** | **[+9.6, +27.0]** | **59 vs 22** | **4.8e-05** |

**Interaction +12.4pp.**

### CONFIRMED
The interface effect is real and slightly larger than the flawed run reported:
**0.426 -> 0.609, +18.3pp, p=4.8e-05**, with the model, the potential and the search
all unchanged. Best cell also uses 25% fewer simulator calls (89 vs 118) and produces
lighter designs (mass_ratio 1.110 vs 1.277).

### RETRACTED: "compound actions are actively harmful without per-component state"

Turn 20 measured -7.9pp (p=0.040) for `aggregate/compound` and I promoted it as the
most interesting part of the result -- "expanding an agent's action space while its
observation stays aggregated makes it worse". With symmetric prompts that becomes
**-0.5pp, discordant 27 vs 28, p=1.0**. The harm was the missing format example in
the single arms, not the mechanism. **The claim is withdrawn.**

### What actually survives, and it is cleaner

| ingredient | alone |
|---|---|
| per-component observation | +6.4pp, **not significant** |
| compound actions | -0.5pp, **exactly nothing** |
| **both** | **+18.3pp, p=4.8e-05** |

Neither ingredient does anything on its own; together they are worth more than their
sum (interaction +12.4pp). The agent needs the information AND a vocabulary able to
act on it. That is a genuine super-additive interaction and a sharper claim than the
"harm" story it replaces -- it just is not as dramatic.

### Note on Turn 20's other numbers

Turn 20's per_member/single (+10.4pp, p=0.017) also fails to replicate here (+6.4pp,
p=0.111). Only the combined cell replicates, and it replicates strongly. Given the
confound affected the single arms specifically, the symmetric run supersedes Turn 20
throughout; **Turn 20's individual-ingredient effects should not be cited.**

## Turn 25 — depth-2 Phi-lookahead beats matched-budget breadth

`scripts/api_lookahead_d2.py`, Sonnet 4.5, 202 held-out `problems_hard`, 20 turns,
Phi-selection in every arm. 606 episodes, 199 problems with all three arms.

  d1        K=8 candidates, simulate all, argmax Phi.
  d2        K=8 at ply 1; top m=3 by Phi each expanded by a fresh LLM call for
            k2=8 follow-ups; a ply-1 action is scored by the best Phi reachable
            beneath it (its own Phi is a floor; a feasible child makes the branch
            infinitely preferred). Second ply is LLM-expanded, so on-policy.
  d1_wide   K=32 in ONE call, argmax Phi. Budget control: exactly the 8+3*8
            candidates d2 sees, at one ply.

| arm | feasible | llm calls/ep | sims/ep |
|---|---|---|---|
| d1 | 0.633 | 11.7 | 93 |
| **d2** | **0.693** | 40.6 | 324 |
| d1_wide | **0.583** | 11.7 | 233 |

| contrast | effect | 95% CI | discordant | exact p |
|---|---|---|---|---|
| d2 vs d1 | +6.0pp | [-1.2, +13.3] | 33 vs 21 | 0.134 |
| **d2 vs d1_wide** | **+11.1pp** | **[+4.2, +17.9]** | **35 vs 13** | **0.0021** |

**The control carries the interpretation.** `d1_wide` sees 4x the candidates of `d1`
at one ply and scores WORSE (0.583 vs 0.633). Breadth at a single ply does not help
and here hurt. So d2's margin is the second ply, not the extra candidates -- which is
exactly what the budget control was built to separate.

Against `d1` alone the margin is +6.0pp and not significant; against the
budget-matched control it is +11.1pp and is. Both comparisons belong in any writeup:
d2 costs 3.5x the LLM calls and 3.5x the simulator calls of d1.

**This is the first result in the project where search DEPTH over Phi does work.**
Phi-selection at depth 1 was already "search to estimate an action's potential"; this
shows a second ply adds beyond it, on-policy, with no training.

## Turn 26 — depth-2 does NOT transfer to battery

`scripts/api_battery_d2.py`, 116 battery problems from 116 distinct scans (45 random
starting cells x 2 currents + the existing grid), q=0, status quo 0.578.

| arm | feasible | llm/ep | sims/ep |
|---|---|---|---|
| d1 | 0.578 | 6.1 | 42 |
| d2 | 0.612 | 20.5 | 146 |
| d1_wide | 0.595 | 5.8 | 159 |

d2 vs d1: +3.4pp [-0.7,+7.6], discordant 5 vs 1, p=0.219
d2 vs d1_wide: +1.7pp [-2.4,+5.9], discordant 4 vs 2, p=0.688

**The +16.2pp measured at n=37 was noise on 8 discordant pairs.** At n=116 the upper
CI bound is +7.6pp, below truss's +11.1pp. Depth is a truss result so far.

Only 6 of 116 problems change outcome between d1 and d2 -- the arms make nearly the
same choices. And ~40% of unsolved episodes in EVERY arm end with `charge_time = 999`
(the cell never reaches 80% SOC), a failure depth cannot repair because by then no
candidate is good.

### Three signs battery is a weak testbed, not just a negative domain

1. the difficulty knob saturates: q=0 (limit set at the best neighbour in the scan)
   still leaves status quo at 0.578-0.60, because the agent is not confined to the
   scan neighbourhood;
2. ~40% of failures are a broken cell rather than a missed optimum;
3. discordance between any two arms is 4-6 problems out of 116, so almost nothing
   distinguishes methods there.

Whether "depth and interface are truss-specific" or "battery does not discriminate
between methods" is unresolved and needs a third domain.

## Turn 27 — RETRACTION: search depth over Phi does NOT beat depth-1

`scripts/api_lookahead_dn.py`, 202 held-out `problems_hard`, k=8, m=2, k2=8,
20 turns, 808 episodes, 6 errored, n=196 with all four arms.

| arm | feasible | llm/ep | sims/ep |
|---|---|---|---|
| d1 | 0.658 | 10.6 | 85 |
| d2 (m=2) | 0.607 | 32.3 | 256 |
| d3 | 0.679 | 68.9 | 548 |
| d1_wide (K=40) | 0.571 | 12.0 | 288 |

| contrast | effect | 95% CI | discordant | exact p |
|---|---|---|---|---|
| d3 vs d1 | +2.0pp | [-4.7, +8.8] | 25 vs 21 | 0.659 |
| d2 vs d1 | -5.1pp | [-11.3, +1.1] | 14 vs 24 | 0.143 |
| d3 vs d1_wide | +10.7pp | [+3.7, +17.7] | 35 vs 14 | 0.0038 |
| **d1 vs d1_wide** | **+8.7pp** | **[+1.8, +15.5]** | **32 vs 15** | **0.019** |

**The last row explains the earlier result.** Plain depth-1 beats the wide control
by +8.7pp on its own. So "depth beats matched budget" was mostly the CONTROL being
bad: d1 beats d1_wide nearly as much as d3 does. Against plain d1 there is no depth
effect -- d3 +2.0pp (p=0.66), d2 -5.1pp (p=0.14), no monotone ladder.

**This retracts Turn 25.** That run reported d2 vs d1_wide +11.1pp, p=0.0021 and
called it the first evidence that search depth over Phi works. Its own d1-vs-d1_wide
gap was +5.0pp in the same direction and I did not test it. Both runs agree once the
right contrast is made: **depth beats degraded breadth, not depth-1.**

**What the wide control actually measures:** asking the model for 40 candidates in one
call instead of 8 costs 8.7pp. The loss is at the proposal stage, not the selection
stage. That is a fact about prompting an LLM for many alternatives at once, and it is
the only significant effect in this experiment.

### Consequence for the project's central claim

Phi's contribution is **one-ply selection** and nothing more:
- depth 1 (simulate K candidates, argmax Phi): works, 319/320 conversion, truss and
  battery;
- depth 2 and 3: no improvement over depth 1, on truss (this turn) or battery
  (Turn 26);
- as reward / prompt / per-token gradient: no effect (Turns 10, 20, 22).

The original framing -- that the innovation is how the method SEARCHES to estimate an
action's potential -- holds only in its shallowest form. Estimating potential by
simulating one step and ranking is what pays. Deeper search does not.

## Turn 28 — everything tried to improve on depth-1 Phi-selection is null or worse

All on 202 held-out `problems_hard`, Sonnet, paired.

**Candidate count (K sweep, depth 1):** monotone DECLINE.

| K | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| feasible | **0.644** | 0.624 | 0.610 | 0.609 | 0.605 | 0.562 |
| sims/ep | 46 | 92 | 129 | 171 | 205 | 242 |

K=32 vs K=4: disc 22 vs 39, p=0.040 -- significantly worse at 5x the simulator cost.
Candidates are not independent draws: they come from ONE generation, and asking for
more alternatives in one response makes each worse. Consistent with d1_wide (K=40)
losing 8.7pp to K=8 in the depth ladder.

**Search depth:** d3 vs d1 +2.0pp (p=0.66), d2 vs d1 -5.1pp (p=0.14). No ladder.

**Incumbent floor x regression feedback (2x2):**

| | no floor | floor |
|---|---|---|
| no feedback | d1 **0.619** | incumbent 0.559 |
| feedback | fb **0.574** | incumbent_fb 0.649 |

- floor alone: -5.9pp (disc 20 vs 32, p=0.126)
- feedback alone: **-4.5pp** (disc 22 vs 31, p=0.272)
- feedback GIVEN the floor: **+8.9pp** (disc 34 vs 16, **p=0.015**)
- best cell vs baseline: +3.0pp, p=0.51

An interaction, and it inverts the obvious reading. Telling the model it just gave up
a constraint HURTS when it is then stuck in the worse state; it only helps when paired
with returning to the incumbent, where the same note reads as "that failed, you are
back at the good design, try something else". Net against plain d1: nothing.

### The trace evidence these came from (results/api_guidance/traces_sonnet.jsonl)

410 turns / 40 problems, raw model text plus every candidate and outcome:
- solved episodes take 6.1 turns; FAILED episodes take all 20, every one;
- failed trajectories oscillate across the constraint boundary (0320 crosses
  FOS=1.5 six times while mass hovers just over budget);
- the chosen candidate's Phi is non-monotone in **12/12** failed episodes, and three
  of four inspected END below the best Phi they ever reached;
- with the floor instrumented, `stay` is 4-15 of 20 turns on failing problems: in
  most turns EVERY candidate is worse than the incumbent, and the baseline steps
  downhill anyway.

The diagnosis was right and both fixes derived from it are null. The baseline
(K=4-8, depth 1, argmax Phi, always move) is not improved by more candidates, more
depth, an incumbent floor, or regression feedback.

### The one measured headroom left

8 independent runs over the same 195 problems: mean single run **0.649**, union
**0.938**, intersection 0.333. Only 12/195 problems are never solved by any run;
60.5% are solved by some runs and not others. **+29pp of the gap is run-to-run
variance**, three times larger than any difference between configurations tested.

## Turn 28 — regression feedback without the incumbent floor: null

`scripts/api_incumbent.py --arms d1,fb`, 202 held-out problems, paired.

| arm | feasible |
|---|---|
| d1 | 0.628 |
| fb (regression note, no floor) | 0.618 |

**-1.0pp, 95% CI [-7.8, +5.8], discordant 23 vs 25, p = 0.89** (n=199 complete).

Turn 27 measured `incumbent_fb` beating `incumbent` by +8.9pp (p=0.015) and the note
was the only difference between them. That is an INTERACTION, not a standalone effect:
the regression note helps only when the search is refusing to move and the model has
to be told why it is stuck. Without the floor it does nothing.

Both halves of the trace-derived intervention are now closed. The floor alone costs
-5.9pp; the note alone is flat at -1.0pp; together they land at baseline.

### Standing configuration

`K=4, depth 1, argmax Phi, always move` at **0.644** remains the best found. Every
variant tested against it is null or worse: more candidates (K=8..32, monotone
decline, p=0.040 at K=32), more depth (d2, d3, both domains), an incumbent floor,
regression feedback, and all three routes into the policy.

## Turn 29 — cutoff-and-restart does NOT deliver the projected gain

`scripts/api_restart.py`, 202 held-out problems, k=8, n=175 with all three arms
(a session restart interrupted the run; 209 rows carried via --resume-from).

| arm | feasible | turns | sims | attempts used |
|---|---|---|---|---|
| c20_a1 (baseline) | 0.583 | 12.0 | 96 | 1.00 |
| c8_a2 | 0.526 | 10.1 | 81 | 1.56 |
| c10_a3 | 0.646 | 16.4 | 131 | 1.99 |

| contrast | effect | 95% CI | discordant | p |
|---|---|---|---|---|
| c8_a2 vs baseline | -5.7pp | [-12.8, +1.4] | 15 vs 25 | 0.154 |
| c10_a3 vs baseline | +6.3pp | [-0.5, +13.1] | 24 vs 13 | 0.099 |

Projected (Turn 28, assuming independent attempts): c8_a2 0.680, c10_a3 0.855.
Measured: 0.526 and 0.646. The independence assumption was wrong on both counts:

1. **Attempts on the same problem are correlated.** A problem that fails once tends to
   fail again. The 0.938 union in Turn 28 came from 8 FULL 20-turn runs, not from
   truncated attempts.
2. **Truncation loses late solves.** Solved-turn median is 4 but p75 = 10 and p90 = 15,
   so a cutoff at 8 or 10 discards the quarter of solves that arrive later. The extra
   attempt does not recover them.

c8_a2 is worse at matched budget. c10_a3 is +6.3pp for 37% more turns, not
significant. Neither is an operating point.

### Where every search-side lever now stands, all vs the depth-1 baseline (~0.60-0.65)

| lever | result |
|---|---|
| more candidates (K 4 -> 32) | monotone decline, K=32 significantly worse |
| deeper lookahead (d2, d3) | null |
| incumbent floor | worse |
| regression feedback | null alone; recovers the floor's loss only |
| cutoff + restart | null / worse |

Run-to-run variance on the SAME configuration is +/-5pp (baseline measured at 0.583,
0.619, 0.633, 0.644, 0.658 across runs). Every lever above is inside that band.

The +29pp variance headroom (single 0.649 vs 8-run union 0.938) is real but can only
be harvested by full-length independent runs, i.e. at 8x the cost. Truncated restarts
do not reach it.

## Turn 29 — restarts are the only thing that beats the baseline, and the cutoff must be LONG

**Cutoff-and-restart, measured (n=175, 202 held-out, K=8):**

| policy | feasible | turns | sims | attempts used |
|---|---|---|---|---|
| 20 x 1 (baseline) | 0.583 | 12.0 | 96 | 1.00 |
| 8 x 2 | 0.526 | 10.1 | 81 | 1.56 |
| 10 x 3 | 0.646 | 16.4 | 131 | 1.99 |

10x3 vs baseline +6.3pp [-0.5,+13.1], disc 24 vs 13, p=0.099. Far below the 0.855
projected from run statistics. 17/606 rows were unparseable (interleaved writes under
the lock) -- 2.8%, spread across arms.

**The cutoff was set wrong.** Solved episodes have median 4 turns but **p90 = 15**.
Capping an attempt at 10 discards ~25% of the successes, and that cost more than the
extra attempts bought. Setting a cutoff from the median was the error.

### Full-length restarts: +19.7pp

Three independent full 20-turn runs at the SAME config (K=8, depth 1):
`ksweep_k8`, `lookahead_dn:d1`, `lookahead_d2:d1`, over the 199 common problems.

| | members | mean | union |
|---|---|---|---|
| same config x3 | 0.623 0.648 0.638 | **0.637** | **0.834** |
| different configs x3 (K4,K16,d3) | 0.648 0.613 0.668 | 0.643 | 0.799 |

"Run up to 3 attempts, stop at the first success" succeeds exactly when any attempt
succeeds, so the union IS the policy's value, not a projection: **0.834 vs 0.637,
+19.7pp for 3x the turns.**

**And the diversity is from SAMPLING, not configuration** -- the same-config union
(0.834) is HIGHER than the different-config union (0.799). This corrects Turn 28,
which attributed the +29pp union across 8 heterogeneous runs to configuration
variety. It is not: re-running one configuration captures more of it.

### Where the search work lands

Against plain depth-1 Phi-selection at K=4-8, every single-episode intervention is
null or negative: more candidates (K=32 significantly worse, p=0.040), depth 2 and 3
(null), an incumbent floor (-5.9pp), regression feedback alone (-4.5pp), feedback with
the floor (+3.0pp, n.s.), short-cutoff restarts (+6.3pp, n.s.).

The only lever that works is running the same thing again from scratch, at full
length. That is a statement about the variance of the LLM proposer, not about the
potential or the search.

## Turn 30 — catastrophic-step veto: null. Causality runs the other way.

Two identical-config traced runs on the same 40 problems disagreed on 8 (20%). Reading
both trajectories of a flipped problem (`hard_problem_0043`): both runs reach the same
state (FOS ~1.55, mass ~37.7) and diverge there -- run1 cuts mass to feasibility in 4
more turns, run2 takes a step whose chosen Phi is -9.19, the ARGMAX of 8 candidates,
i.e. the entire candidate set was catastrophic. All K come from one generation; one
bad reasoning step poisons all of them.

Across all 80 traced episodes, steps with chosen-Phi drop < -3:
solved 0.06/episode, failed **1.43/episode** (p10 Delta-Phi -0.91 vs -2.33). A 24x
difference.

Veto arm: if the best candidate would drop Phi by more than 3, re-propose from the
CURRENT state (fresh call), up to twice, then accept. Unlike the incumbent floor it
lets ordinary downhill steps through (solved episodes take those and recover).

n=200: d1 0.620, veto 0.645. **+2.5pp, 95% CI [-4.6, +9.6], discordant 29 vs 24, p=0.58.** Vetoes fired 0.51/episode and changed nothing.

**Blocking the catastrophic step does not rescue the episode.** The re-proposal from
the same state is drawn from the same bad reasoning and is usually as bad. So
catastrophic candidate sets are a SYMPTOM of already being in a region the model
cannot reason its way out of, not the cause of arriving there. The correlation was
real; the intervention on it is null, which is what tells the causal direction.

## Turn 31 — what the run-to-run variance actually is

Two identical-config traced runs on 40 problems disagree on 8 (20%). Read side by
side at the divergence point (`hard_problem_0043`, both runs at FOS ~1.55 / mass
~37.7): the model's REASONING is the same and correct in both -- it names the same
over-designed members (M2 at 87-111x required yielding FOS) and the same strategy.
One run then happens to propose a bold cut (t x0.55, -9 kg) and solves in 4 turns;
the other proposes only +/-20% moves and drifts. At the population level, though,
boldness does NOT separate outcomes (median max|ln f| per turn: solved 0.69, failed
0.73; 2x moves available in 53% vs 59% of turns). Neither does reasoning quality,
candidate diversity, or catastrophic steps (Turn 30: symptom, not cause).

**What does predict it is the PROBLEM, via the stratum assigned from procedural
search depth** (8 depth-1 runs, 199 problems):

| stratum | n | mean solve | never | always |
|---|---|---|---|---|
| s0 | 13 | 1.000 | 0 | 13 |
| s1 | 37 | 0.963 | 0 | 32 |
| s2 | 21 | 0.714 | 0 | 7 |
| s2b | 53 | 0.580 | 2 | 6 |
| s3 | 58 | 0.440 | 6 | 5 |
| s4 | 17 | 0.353 | 5 | 0 |

Member count does not predict it (0.65/0.67/0.55 for 10/15/20). Difficulty labels
agree: easy 0.995, medium 0.692, hard 0.366, open 0.295.

### The clean statement

Each problem has an intrinsic per-attempt success probability p, set by how deep a
search is needed to reach its feasible region. A single 20-turn episode is one
Bernoulli draw from p. The 0.938 "union of 8 runs" is not a ceiling to be reached by
a smarter search; it is 1-(1-p)^8 summed over problems.

Every search-side intervention (K 4-32, depth 2/3, incumbent floor, regression
feedback, catastrophic-step veto, cutoff+restart) left p unchanged -- all sit inside
the +/-5pp Bernoulli band of the baseline. The only thing in this project that moved
p on hard problems was the interface (settled, Turn 24).

Replication raises the union, not p. It is brute force, not a method, and it costs
linearly.

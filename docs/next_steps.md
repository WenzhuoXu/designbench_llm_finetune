# Next steps

Written 2026-08-23. Champion = `checkpoints/grpo/mt_t3_01b_critfeedback_20260616_0900/final`
(Qwen3-14B + LoRA r32, GRPO multi-turn, α = 5, critical-member feedback).
Headline: **54% feasible on 50 held-out problems** (68% on the easier first 25).

Ordered by ratio of what it settles to what it costs.

---

## P0. Fix the ρ instrumentation before quoting ρ anywhere

**The ρ currently logged is not the ρ from the design doc, and under the champion
config it cannot be anything but a constant.**

`_compute_rho_per_group` (`grpo_trainer.py:345`) computes, per problem group:

```
rho = 1  if  argmax_i Φ_H(i)  ==  argmax_i reward(i)
```

Two consequences:

1. **With `use_tree_expansion: true`** the reward is `lagrangian_potential`, i.e.
   `r = γΦ(s′) − Φ(s)`, which telescopes to `γ^H Φ(s_H) − Φ(s_0)`. Within a group the
   problem is fixed, so `Φ(s_0)` is constant and `argmax(reward) ≡ argmax(Φ_H)` by
   construction. **ρ = 1.0 is an algebraic identity, not a measurement.** That is why it
   reads 1.0 from step 0 rather than rising.
2. **With `use_tree_expansion: false`** (the champion) `phi_H_per_rollout` is never
   filled — it stays `[0.0] * n` (`grpo_trainer.py:432`) — so `max()` returns index 0 every
   time and ρ fires only when the best-reward rollout happens to be rollout 0. That is
   ~1/8 at group size 8, which matches the noisy 0/1 values in the champion's
   `metrics.jsonl`.

Neither number means what §3.6 defines: `ρ(t) = Pr[argmax_k π_θ(a_k|s) = argmax_k V̂(s,a_k)]`,
an agreement rate between the **policy's** top action and the **tree's** top action at a
state. `PosteriorEvalRecord.rho_tree_agreement` in `posterior/evaluator.py:633` does compute
the right thing, but it runs only in `scripts/eval_posterior_one_step.py`, not in training.

**Do:** either wire the evaluator's per-context definition into the training loop, or stop
logging the current quantity. Until then, do not claim ρ as evidence that the policy
internalised the lookahead. **Cost: hours. Blocks: a claim currently on slide 3 of the talk.**

---

## P1. Three cheap experiments that answer the questions an audience will ask first

### P1.1 Greedy sizing baseline (no LLM)

Every successful trajectory in the deck is `SCALE_PARAM(named_member, r, ~1.3)` repeated.
That is fully-stressed design, the standard sizing heuristic. Until its number is on a slide,
the result reads as "an LLM was never needed here."

**Do:** loop `analyze_truss` → read `min_fos_buckling_member_id` → `execute_grammar_action`
with `SCALE_PARAM(id, r, 1.3)` → repeat to feasible or 20 turns, over the same 50 problems.
Both functions are in `DesignBench/validation/truss_executor.py`.
**Cost: ~30 lines, CPU only, seconds to run. No GPU, no job submission.**
**Settles:** whether the contribution is "environment feedback design for LLM agents" or a
restatement of a classical heuristic. Either answer is publishable; only one is defensible
unprepared.

### P1.2 Isolate what the training bought

Slide 8 shows the **untrained** 30B+14B cascade reaching 4/6 with the critical-member field.
Slide 9 shows the **trained** 14B at 54%/50 and 68%/25. These are different systems on
different problem sets and are never compared.

**Do:** run four cells on one problem set, one turn cap, one decoding config:
untrained 14B ± field, trained 14B ± field. `scripts/eval_checkpoint.py` already supports
`--max-problems`; the field is toggled in `designbench_prompt.py:73-82`.
**Cost: 4 eval jobs, no training.**
**Settles:** the marginal value of GRPO over the prompt change. This is the single most
likely question from the floor.

### P1.3 Report mass, not only feasibility

Feasibility is a pass/fail constraint check. Three of the 50 runs pushed mass past 3× its
starting value and still counted as failures only because they missed the limit; nothing
reports how heavy the 27 successes are.

**Do:** add mean and median `final_mass / initial_mass` for the feasible set, and ideally a
ratio against an LP or fully-stressed sizing optimum on the same problems.
**Cost: post-processing of existing `eval_results.json`, plus P1.1 for the reference point.**
**Settles:** whether the policy produces designs an engineer would build.

---

## P2. Close the three known failure modes

From the 50-problem breakdown: 13 never reached FOS 1.5, 10 met FOS but failed mass or
deflection, and the mean parse-and-execute rate is 0.663.

### P2.1 Price mass and FOS jointly in Φ
The hard prompt rule regressed 68% → 52% by making the policy hug FOS 1.5. The reward-side
lever is untried: raise the violation threshold `target_fos` (`potential.py:27,52,78`) from
1.5 to ~1.7, or add an explicit mass term, and sweep. Softening where the hard rule
over-corrected is the natural follow-up.

### P2.2 Format-protecting KL
Solved runs parse 0.78 of actions, failed runs 0.53. Failed runs burn all 20 turns. A
grammar-compliance *reward* already failed (12% → 4%); the untried lever is the KL anchor to
the warmstart, which had perfect grammar (1.00). Sweep `kl_coef` above 0.04, or apply KL only
on format spans.

### P2.3 Degenerate solver states
The 13 hard failures start at mean initial FOS 0.32 against 0.70 for the solved set. Some are
structurally broken intermediates. Detect and either skip or repair them rather than spending
the turn budget on them.

---

## P3. Statistical power

Almost every ablation conclusion sits inside the ±8 point band at n = 25: the α sweep
(8/12/4), the grammar reward (12→4), the mass rule (68→52). Only 12 → 68 clears it.

`DesignBench/data/problems/` holds **131 problems**; evaluation currently uses 25 or 50.
**Do:** re-evaluate the champion and the two or three ablations that matter on the full 131,
and run more than one seed for any cell whose conclusion the paper depends on.
**Cost: eval only, no retraining.** This is what converts most of the ledger from suggestive
to reportable.

---

## P4. Larger moves

- **Distillation.** `results/distill/traj_v3_critmember.jsonl` already holds 4/6 solving
  trajectories from the 30B+14B cascade. Distilling them into the 14B is the standing reserve
  if P2 stalls.
- **Guided search, decided properly.** After P0, either the tree advantage measurably improves
  ranking and earns its 6× solver cost, or it does not and can be cut. Right now the D=1 run
  ties on feasibility (12% → 12%) while producing the best mean FOS in the study (2.12), and
  the diagnostic that was supposed to adjudicate this is broken.
- **Second design domain.** The framework is meant to generalise past trusses. The action
  grammar and the potential Φ are the two truss-specific pieces; everything else is domain
  agnostic. Porting to one non-FEA domain would test that claim.

---

## Suggested order

1. P0 (hours) — stop quoting a broken diagnostic.
2. P1.1 and P1.2 (a day) — these two decide how the work is framed.
3. P3 on the champion (eval only) — get one number you can defend.
4. P2.1 and P2.2 — the two untried levers on the largest failure buckets.
5. P4 as results dictate.

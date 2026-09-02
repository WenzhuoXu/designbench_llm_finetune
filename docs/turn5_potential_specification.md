# Turn 5 — The potential was never the problem's potential

**Date:** 2026-08-23 · **Status:** in flight · **Ledger turn:** 5
**Question this turn answers:** the framework's stated innovation is estimating an action's
*potential* on a design state by search. Turns 0–4 concluded that state feedback matters and
search does not. This turn asks whether that conclusion was ever tested.

It was not. What follows is what the code actually did, what it cost the study, and the
replacement.

---

## 1. Six defects, each verified

| # | Defect | Evidence | What it invalidates |
|---|---|---|---|
| D1 | `use_tree_expansion: true` never runs a search | `grpo_trainer.py:278` `_compute_group_tree_advantages` computes `γΦ(s_H^k) − mean_j Φ(s_H^j)`; `evaluate_tree` is called only from `scripts/eval_mcts_inference.py` | every "tree" cell (`v2_02`, `v2_04`, `mt_s2_02_tree_d1`, `mt_t2_grammar_tree`) |
| D2 | Training-time ρ is not §3.6's ρ | `grpo_trainer.py:345` — `argmax_i Φ_H(i) == argmax_i reward(i)`: an identity under tree mode, a 1/K coin flip otherwise (`phi_H_per_rollout` stays `[0.0]*n`, `:432`) | every ρ number quoted, incl. "ρ(t)=1.0" |
| D3 | Φ encodes a constraint set no problem has | 130 problems: 100 constrain `{fos_b, fos_y, mass ≤ m*}` with **no** deflection goal, 30 constrain `{fos_b, fos_y, deflection ≤ d*}` with **no** mass goal (23 of those 30 have `d* = inf`, i.e. FOS-only). `potential.py` hardcodes `deflection ≤ 0.01` for all and prices no mass cap | the reward of every GRPO run |
| D4 | `posterior.alpha` never reached the reward | `build_reward_from_config` called `CompositeReward(weights=...)` with no component kwargs, so `alpha=5.0` always | the α-sweep: `mt_s1_01a/01b/01c` (α=2/5/10) were **the same configuration three times** |
| D5 | Reward weights were silently inherited | Hydra merges dict keys, so every `mt_*` config added its term *on top of* `grpo_truss.yaml`'s four. The champion's reward was `feasibility 1.0 + fos_improvement 0.5 + grammar_compliance 0.1 + step_efficiency 0.1 + lagrangian_potential 1.0`, not the "single dense reward term" of the status report | the grammar-reward ablation (it "added" a term already present at 0.1) |
| D6 | Train and eval read the same directory | `configs/data/rl_problems.yaml` → all 131; `eval_checkpoint.py:270` → the same 131, first 50 | the word "held-out" everywhere in the study |
| D7 | **The trained token sequence contained no environment feedback** | Qwen3's template strips `<think>` from non-final assistant turns, so the re-rendered conversation is never a prefix extension of the running ids and `multiturn_rollout.py:236-238` fell through to `delta = []`. The champion's log has **9176** `prefix mismatch` warnings, ~128 per problem on all 100 | the context every multi-turn run trained in |
| D9 | The potential was unbounded on states where the simulator stops meaning anything | a min-mass lookahead drove pipes to `t > 2r` (area `π·t·(2r−t) < 0` ⇒ negative mass, reported ratio −508) and thinned members until the truss became a mechanism (forces vanish, FOS explodes, deflection 1.5e14 m, scored FEASIBLE because the mass family states no deflection limit) — **17% of its "successes"** | every ladder number before the envelope was added |
| D8 | A no-op step was rewarded | the shaping sum omitted `γ^t` (`rewards.py:409-417`), so the return is a path integral; where `Φ<0`, a no-op pays `(1−γ)|Φ|` — measured **+0.25 per wasted turn, +1.27 over five** on a typical infeasible state | the "failed runs burn all 20 turns" failure mode |

Plus two smaller ones: the system prompt's worked example, `SCALE_PARAM(all_members, thickness, 1.224)`,
is rejected by the executor's `(\d+)` member-id regex on every use; and one problem id
(`auto_problem_000`) appears in two files.

### D4 has a silver lining: a free triplicate

Because α never reached the reward, `mt_s1_01a/01b/01c` are three seeds of one configuration.
They returned **8% / 12% / 4%** feasibility at n=25. That is a direct measurement of the noise
floor: **±4 points at n=25**, a ±50% relative spread. Under it, the α result ("non-monotone,
peaks at 5"), the grammar-reward result (12→4) and the mass-feedback result (68→52) are all
inside noise. Only 12 → 68 clears it.

### D7 in full: the multi-turn fix was only half-landed

The rollout rebuilt the whole conversation each turn and accepted the delta only if it
extended the running sequence. Qwen3's chat template renders a *final* assistant message as
`<think>\n\n</think>\n\nA<|im_end|>` but a *non-final* one as plain `A<|im_end|>` — the reasoning
block is dropped. So the re-render is shorter than the running sequence, never a prefix
extension, and the guard silently produced an empty delta on every transition:

- the trained completion was `gen(turn0) ++ gen(turn1) ++ …`, with **no FEA feedback and no chat
  glue** — not even well-formed chat;
- `env_mask` only ever received `1`s, so TRL's environment-token masking (`trl:1649`, the entire
  stated reason for using `rollout_func`) was dead code;
- TRL recomputes per-token logprobs over `prompt_ids ++ completion_ids`, so every token of turns
  ≥ 1 was scored under a context that never existed at sampling time — roughly two thirds of all
  trained tokens at the observed ~4.1k mean completion length. With `num_iterations=1` the PPO
  ratio is identically 1 (`clip_ratio/region_mean = 0.0` in every logged step), so there is no
  importance-sampling correction to absorb it.

The FEA stepping, and therefore the *reward*, were computed on the true trajectory — which is why
12 → 68% still happened. It is the *context* that was corrupted. **Fixed** by deriving the glue
once from the template itself (render a probe conversation whose assistant turn is followed by a
user turn, read off everything after the assistant's own content) rather than reconstruct-and-diff.
Template-agnostic; 36 tokens of glue where there were 0; four regression tests in
`tests/test_multiturn_probe.py`.

One residual, documented rather than hidden: the flat training sequence necessarily retains the
model's own `<think>` from earlier turns (a sequence TRL can score must be append-only), while
evaluation re-renders through the template and drops them. Aligning the two is a separate,
cheap experiment.

---

## 2. Why D1 produced a null result rather than a wrong one

TRL 1.0.0 runs GRPO with `scale_rewards='group'`: advantages are mean-centred **and** divided by
the within-group standard deviation. So GRPO is invariant to affine transforms of the reward, and

> under group-normalised GRPO, reward design reduces to **ordering** design.
> Only the order Φ induces over a group's K rollouts reaches the gradient.

`lagrangian_potential` sums `γΦ(s_{t+1}) − Φ(s_t)`, which telescopes to
`γΦ(s_H) − Φ(s_0) − (1−γ)Σ_t Φ(s_t)`; within a group `Φ(s_0)` is constant. The "tree advantage"
adds `γΦ(s_H) − mean_j Φ(s_H^j)`. Both are the same monotone function of `Φ(s_H)` up to a group
constant, so enabling tree expansion **re-weights a term already in the reward and adds no
lookahead information whatsoever**. A tie was the only possible outcome.

*(Correction to an earlier reading: because of D5 the champion's return is not a pure Φ surrogate —
a terminal `feasibility` reward is present at weight 1.0, so Φ genuinely acts as shaping on top of
a task reward. The ordering argument above is unaffected: the two cells differ only in the weight
on the Φ-dependent part.)*

---

## 3. Why the potential's argmax is not the feasible set

`Φ_v1(s) = log(m₀/m) − α·[softplus(1.5−fos_b) + softplus(1.5−fos_y) + softplus(defl/0.01 − 1)]`

Three independent errors:

1. **A phantom constraint at weight α.** 100/130 problems have no deflection goal. Reducing
   deflection is rewarded anyway, and the only way to reduce deflection is to add material.
2. **The real constraint is unpriced.** No term fires when `mass > maximum_mass`. Mass appears
   only as `log(m₀/m)`, weight 1 against the violations' weight α=5.
3. **Nothing saturates.** `softplus(1.5 − 6.0) = 0.011` versus `softplus(1.5 − 1.55) = 0.668`:
   raising FOS from 1.55 to 6.0 still buys `α·0.657 ≈ 3.3` of potential. The mass it costs is
   charged at `log`.

The sign of the objective term is also wrong for this benchmark. `maximum_mass = 1.1 ×
optimal_mass` on all 100 mass-family problems and the initial design is a *degraded*, under-built
version of that optimum, so **mass must rise** to reach feasibility — exactly the direction
`log(m₀/m)` penalises.

Concretely, on `auto_problem_000` (mass cap 265 kg):

| design | mass | fos_b/fos_y | feasible | **Φ_v1** | **Φ_v2** |
|---|---:|---:|:--:|---:|---:|
| lean | 250 | 1.55 / 1.60 | **yes** | −9.02 | **−1.27** |
| over-stiffened | 900 | 6.00 / 8.00 | no | **−3.80** | −15.73 |

**Φ_v1 ranks the infeasible design 5.22 points above the feasible one**, from the FOS and mass
terms alone with deflection held equal. Let the two designs deflect as they really would (a
900 kg structure deflects far less than a 250 kg one) and the phantom deflection term compounds
it to 13.7. Either way the ordering is inverted, and any search that maximises Φ_v1 faithfully
walks away from feasibility. The D=1 tree cell producing the best mean FOS in the
study (2.12) with no feasibility gain is not a failure of search — it is search succeeding at the
objective it was given.

---

## 4. The replacement: a design program, not a truss potential

`Φ(s) = −w·f̂(s) − α·Σ_c hinge(ĝ_c(s))` — the negated Lagrangian of the program stated by the
problem. `f̂` is the objective over a problem-supplied reference; `ĝ_c` is the *relative* violation
of constraint c, hinged so a satisfied constraint stops paying. Both dimensionless, so α means the
same thing in every domain, and a problem contributes no term for a constraint it does not have.

Constraints are recovered from the goals dict **by naming convention** — `minimum_x`/`min_x` ⇒
`x ≥ limit`, `maximum_x`/`max_x` ⇒ `x ≤ limit` — which is the convention DesignBench already uses
in its truss goals (`minimum_fos_buckling`, `maximum_mass`, `maximum_deflection`) *and* its battery
goals (`max_temperature`, `max_plating`, `max_charge_time`, `min_neg_potential`). Non-finite limits
are dropped: an unbounded constraint is not a constraint.

**A new domain is onboarded by naming its goals, not by writing a potential.**
`posterior/potential.py`: `DesignProgram`, `program_from_goals`, `compute_potential_v2`,
`program_from_truss_spec`, `program_from_battery_goals`.

---

## 5. What the potential is worth, measured without an LLM

`scripts/search_ladder.py` runs fixed non-LLM policies over all 130 problems on CPU.
Deterministic, no seed noise. Two accounting rules make the numbers comparable to a policy's:

* **Legal action space.** Only actions the environment's grammar can express — one `SCALE_PARAM`,
  or one `SCALE_MULTI_PARAM` with a single shared factor. An earlier run let the search take a
  *compound* move (a different factor per member between two simulator calls); it is strictly
  outside the grammar and made every comparison against the LLM a budget comparison.
* **Validity envelope.** A state is scored infeasible if it leaves the simulator's domain of
  validity, whatever the goals say (D9). Without it 17% of the search's "successes" were
  mechanisms.

| policy | what it estimates the potential with | feasible | mass/ref | FEA calls |
|---|---|---:|---:|---:|
| random | nothing | 30.0% | 1.050 | 15 |
| greedy critical-member | the simulator's argmax hint (**depth 0**) | 72.3% | 0.872 | 6 |
| fully-stressed design | an **analytic** first-order model, all members at once | **75.4%** | **0.474** | 12 |
| 1-step lookahead, **Φ_v1** | simulation, misspecified potential | 58.5% | 0.834 | 2178 |
| 1-step lookahead, **Φ_v2** | simulation, aligned potential | 73.8% | 0.644 | 2357 |
| 2-step lookahead, **Φ_v2** | deeper simulation | **75.4%** | 0.620 | 14016 |

Paired on the problems each pair jointly solves:

| comparison | median mass/ref | lighter on |
|---|---|---|
| lookahead **Φ_v2** vs lookahead **Φ_v1** | **0.583** vs 0.897 | **64/75 (85%)** |
| lookahead **Φ_v2** vs critical-member feedback | **0.602** vs 0.875 | **81/92 (88%)** |
| lookahead **Φ_v2** vs fully-stressed design | 0.602 vs **0.485** | 11/94 (12%) |

Feasibility sets nest strictly: Φ_v2 ⊃ Φ_v1 (**+20, −0**), D=2 ⊃ D=1 (+2, −0).

1. **The potential's specification is the decisive variable.** Identical search, identical action
   space, identical budget: **+15.3 points of feasibility and 35% lighter designs**, a strict
   superset of solved problems, an 85% pairwise win rate. Under the stricter accounting the effect
   is *larger* than it first appeared. And searching hard against Φ_v1 (58.5%) is far **worse than
   the trivial depth-0 heuristic** (72.3%): a misspecified potential makes search actively harmful.
2. **Simulated search beats the feedback heuristic on quality** — 88% win rate, 0.602 against 0.875
   — but **loses to the analytic estimator on mass** (12% win rate). With legal single actions and
   20 turns it cannot express what fully-stressed design does in one move. Given the compound
   action, the same search reaches 0.429 against FSD's 0.435; the action grammar, not the
   estimator, is what binds.
3. **Which is why the second domain decides the contribution.** Fully-stressed design encodes
   seventy years of structural mechanics, and it would be dishonest to claim generic search beats
   it on trusses. But FSD does not exist for batteries, linkages, or any new domain, while Φ_v2
   needs only a simulator and a goals dict in the `minimum_`/`maximum_` convention. **The
   domain-general estimator transfers; the analytic one does not.** That is the value proposition,
   and it is an empirical claim the battery domain tests rather than an argument.

An independent measurement of the feedback-search relationship, over the 1051 states the greedy
policy visits: the feedback field is an exactly sufficient statistic for the depth-1 argmax at
**65.8%** of states and sufficient to within 0.098 potential units at **82.2%** — at zero simulator
cost, against a median **181 vs 8** FEA calls per problem for the oracle. It breaks where the
theory says it should: when two constraints bind at once (agreement 64.8%), when the best move is
non-local, and when no constraint binds and the objective takes over (agreement **13.6%**).

### Where the LLM earns its place

The search finds simulator exploits; **the policy does not**. Across every evaluation cell in the
study the maximum FOS_buckling among solved designs is **2.9** — no trained policy ever produced a
mechanism, while an unguarded potential-maximising search produced them in 17% of its successes.
The engineering prior keeps the policy inside the physical envelope; the potential ranks within it.
That is an argument for the architecture rather than for either half of it:

| | optimises well | stays physical | transfers to a new domain |
|---|:--:|:--:|:--:|
| analytic heuristic (FSD) | **yes** | yes | **no** |
| potential-guided search | yes | **no**, without an envelope | **yes** |
| LLM policy alone | **no** (0.917) | **yes** | yes |
| LLM proposes, potential ranks, envelope bounds | — | — | — |

The last row is the method, and the measured gap between the policy (54% / 0.917) and the search
it should have internalised (75.6% / 0.620) is the objective, tracked by `lookahead/regret`.

### The 11 problems nothing solves
9 are structurally degenerate: the FEA returns `mass=inf`, `fos=0` at the initial state and still
does after every parameter is driven to its bound. They are mechanisms, fixable only by
`ADD_MEMBER`. Φ is *constant* on them, so potential-guided search cannot rank actions there at all
— a real gap in the theory, and the honest reason ~8% of this benchmark is out of reach.

---

## 6. What is now running

Held-out discipline first: `data/splits/truss_v1.json` (`scripts/make_splits.py`) is a seeded split
stratified on family × degeneracy × initial-violation tercile — 86 train / 43 eval, zero overlap,
matched medians (0.507 vs 0.526). `configs/data/rl_problems_split.yaml` restricts training to the
train side; `eval_checkpoint.py --split-file ... --split eval` restricts evaluation to the other.

The A/B, one variable:

- **T5a** `grpo_mt_t5a_phi1` — champion reward, retrained on the train split with the corrected
  rollout. Note this is *not* a replication of the published champion, which trained on the
  evaluation problems (D6) with no feedback in its context (D7) and a stall bonus (D8).
- **T5b** `grpo_mt_t5b_phi2` — identical except `lagrangian_potential` → `lagrangian_potential_v2`.
- **T5c** `grpo_mt_t5c_lookahead` — identical to T5b plus `lookahead_advantage`: the per-step
  counterfactual estimator `Σ_t [Φ(s_{t+1}) − E_c Φ(f(s_t, c))]`. Its baseline varies with the
  rollout *and* the step, so unlike the old "tree advantage" it is not an affine image of any
  per-trajectory quantity and cannot be normalised away by `scale_rewards='group'`. Candidates
  come from the simulator's action space rather than from extra LLM samples, which is what makes
  a dense every-step lookahead affordable: ~0.13 s per state against a ~500 s training step.

Both reward blends are now written out explicitly so the inherited terms (D5) are documented and
identical across arms. Prediction, from §5: T5b should move feasibility, and move mass at
feasibility more sharply.

The ρ instrumentation (D2) is replaced rather than repaired. `posterior/lookahead_probe.py` ranks
the candidate actions at a visited state and locates the action the policy actually took, logging
`lookahead/{rho, regret, rank_frac, margin, sigma, reliable_frac, advantage}`. Binary ρ is the
wrong statistic — it reads 0 both for an action in the top 3% and for the worst action available;
**regret** is what keeps reporting how far the policy sits below potential-optimal. `reliable`
implements the master doc's ranking-reliability condition as an online gate: where the top two
candidates differ by less than the noise band, a search budget cannot buy a decision. Measured at
the initial state of `auto_problem_000`: margin 0.0059 against a 0.249 threshold — the lookahead
*cannot* rank there, which independently predicts the observed tie.

Still open: the second domain (battery/PyBaMM, in progress), and aligning the training context
with the evaluation context (the D7 residual).

---

## 7. The second domain: batteries

`scripts/battery_ladder.py` runs the SAME ladder on 24 battery-design problems simulated with
PyBaMM (DFN + lumped thermal + partially-reversible lithium plating + solvent-diffusion SEI),
across eight families. The potential is the same `DesignProgram`: constraints recovered from the
battery goals by the same `minimum_`/`maximum_` convention parser
(`max_temperature`, `max_plating`, `max_charge_time`, `min_neg_potential`), objective = charge
time. **No battery-specific potential code was written.**

| policy | feasible | obj/ref | sims |
|---|---:|---:|---:|
| random | 8.3% | 0.935 | 9 |
| greedy critical-constraint (**depth 0**) | 25.0% | 0.826 | 8 |
| enumerate + **blind** potential | 62.5% | 0.811 | 187 |
| enumerate + **goal-derived Phi_v2** | **91.7%** | **0.796** | 119 |
| Phi_v2, D=2 | **95.8%** | 0.807 | 257 |

A caveat stated precisely: on batteries the shipped Phi_v1 reads `fos_buckling` / `mass` /
`deflection`, which a battery simulation never produces, so it scores every state identically.
That arm is **enumeration with no ranking signal**, not a misspecified potential. The genuine
misspecified-vs-correct comparison remains the truss one (58.5% -> 73.8%).

**The inversion.** The depth-0 feedback heuristic scores 72.3% on trusses and **25.0%** on
batteries. On trusses the critical-member field is a near-sufficient statistic for the depth-1
argmax -- FOS is a `min` over members and `analyze_truss` hands over the argmin, so the simulator
performs the inner maximisation for free. On batteries every constraint is coupled: electrode
thickness moves charge time, cell temperature and plating simultaneously, so no single critical
element exists and depth-0 has nothing to grip. Ranking those same candidates by the potential is
worth **+29.2 points** and *fewer* simulator calls (119 vs 187), because it reaches feasibility
sooner.

By family, the difference concentrates exactly where the theory says it should -- the hard
variants, where more than one constraint binds:

| family | greedy | blind | Phi_v2 | Phi_v2 D=2 |
|---|---|---|---|---|
| coupled / energy / thermal | 2/3 | 3/3 | 3/3 | 3/3 |
| rate | 0/3 | 2/3 | 3/3 | 3/3 |
| **coupled_hard** | 0/3 | 0/3 | **3/3** | 3/3 |
| **energy_hard** | 0/3 | 0/3 | **3/3** | 3/3 |
| **rate_hard** | 0/3 | 1/3 | **3/3** | 3/3 |
| **thermal_hard** | 0/3 | **3/3** | **1/3** | 2/3 |

Across the hard variants: greedy **0/12**, blind enumeration 4/12, Phi_v2 **10/12**. Phi_v2 solves
a strict superset of greedy's set (+16, -0).

### The counterexample, closed

Adding the lexicographic feasibility offset (M = 5, tau = 0.005) takes the battery domain to
**100% feasible on all 24 problems, every family 3/3**, using **33% fewer simulator calls**
(80 against 119) because the search stops walking toward infeasible-but-fast designs. The
objective cost is 1.3% (obj/ref 0.806 against 0.796) -- the expected lexicographic trade.

| policy | feasible | obj/ref | sims |
|---|---:|---:|---:|
| Phi_v2, plain Lagrangian | 91.7% | 0.796 | 119 |
| **Phi_v2 + feasibility offset** | **100.0%** | 0.806 | **80** |

The original diagnosis follows.

### The counterexample, and what it says

`thermal_hard` is the one cell where the potential is worse than no potential (1/3 against 3/3).
It is the objective-competes-with-violation break of Proposition 2: Phi_v2's objective term rewards
faster charging, which raises cell temperature, so the greedy potential step walks *away* from the
binding thermal constraint while blind enumeration simply halts at the first feasible design.
Depth-2 partially recovers it (2/3). The indicated fix is a two-phase constraint price -- restore
feasibility under a large alpha, then optimise inside the feasible set -- not more depth. Until
that is run, the honest statement is that a single-alpha Lagrangian potential can be dominated by
undirected enumeration when the objective and the binding constraint pull in opposite directions.

### What this settles

The original experiments were run in the one regime where the innovation cannot show value.
Trusses are weakly coupled and expose per-element diagnostics, so the cheapest estimator of an
action's potential already captures most of it: feedback beat search, and every search ablation
read as a null. That is a property of the benchmark, not of the method. **The value of explicit
search scales with constraint coupling**, and with simulator cost -- PyBaMM's DFN runs at 3.73 s
against the truss FEA's 4 ms, 900x, which is what turns the ranking-reliability gate from a logged
diagnostic into a scheduling rule.

---

## 8. What transferring the potential actually cost

Onboarding batteries needed no potential code — the convention parser recovered the constraint set
from the goals dict, and the same lookahead ran. But two *scale* assumptions baked into the truss
setting did not transfer, and both were found by the `thermal_hard` counterexample rather than by
inspection.

**1. The hinge softness `tau` is in relative units, and must be small against the domain's typical
violation.** A truss starting at FOS 0.44 against a 1.5 target has a relative violation of 0.71; a
battery cell 2% over its temperature limit has 0.021 — a **34x** gap. At `tau = 0.05` the battery's
violations sit entirely inside the hinge's smooth region, so a *satisfied* constraint still charges
~0.03 and the potential cannot cleanly separate feasible from infeasible.

**2. A single constraint price `alpha` is not portable, and lowering `tau` does not rescue it.**
Even with a perfect hinge, the battery pair ranks wrongly: `alpha*V = 5 x 0.021 = 0.10` against an
objective gap of 0.33, so a cell that is 2% too hot but charges twice as fast outranks a feasible
one. `alpha` would have to exceed ~16 here and ~5 on trusses.

Two mechanisms, doing different jobs:

* `DesignProgram.with_violation_scale(initial_state)` divides violations by the problem's own
  initial violation, so `alpha` becomes dimensionless — "how far this problem started from
  feasible". Measured: the 34x cross-domain gap in effective `alpha` closes to ~2x. **This makes a
  single alpha sensible across domains but does NOT guarantee the ordering** (the battery pair is
  still mis-ranked at alpha=5).
* `feasibility_offset` (M) adds `-M*tanh(V/tau)`, ranking any violating design below any satisfying
  one — the lexicographic "restore feasibility, then optimise" construction, smoothed so the
  landscape stays differentiable. **This is what actually fixes the counterexample**, at M=5, and it
  leaves the truss ordering untouched.

A false start worth recording: the first attempt shrank the objective term while infeasible
(`w_eff = w/(1+kappa*V)`). That makes infeasibility *cheaper* — a violating design stops paying for
the objective it is failing to earn — and inverts the ranking further. Gating the objective is the
wrong direction; offsetting the violation is the right one.

The general lesson, which is the one that matters for a framework claiming to span domains: **the
constraint set transfers by convention, but the scales do not.** A potential that is merely
"correctly specified" in its constraint set can still be dominated by undirected enumeration if its
two terms are not commensurable in the new domain. Both mechanisms above are per-problem and
automatic; neither requires a human to tune a domain constant.

### The offset is conditional, and the condition is computable

Running the same offset on trusses (tau = 0.02, M = 5) changes feasibility **not at all** and costs
mass:

| domain | without offset | with offset |
|---|---|---|
| battery | 91.7% feasible, 119 sims | **100.0%**, **80 sims**, +1.3% objective |
| truss (with compound action) | 76.2%, mass 0.437 | 76.2%, mass 0.457 (+4.6%) |
| truss (legal grammar only) | 73.8%, mass 0.644 | 73.8%, mass 0.682 (+5.9%) |

Which is the predicted behaviour. The offset buys a guarantee trusses do not need: their relative
violations are O(0.7), so the objective term cannot outbid `alpha * V` however it is weighted.
Batteries sit at O(0.02), where it can and does.

So the rule is not "always offset" but a condition evaluable per problem, from the initial state
alone and with no human input: **enable the offset when the objective's dynamic range can exceed
`alpha` x the typical violation.** The same initial-state statistic that makes `alpha` dimensionless
(`with_violation_scale`) is what decides it.

This is the third instance of one lesson. The constraint SET transfers across domains by naming
convention; the SCALES -- the hinge width, the constraint price, and whether feasibility needs a
lexicographic guarantee at all -- must be derived per problem. A framework that ships fixed values
for them is a framework that works in one domain.

---

## 9. Distilling the search into the policy (T6), and what it costs

`scripts/distill_search_traces.py` runs the lookahead on the training split and writes each decision
as a conversation turn whose reasoning is the search's own justification -- which constraint binds,
which element the simulator blames, how many candidates were compared, what the chosen action did
to the potential and by what margin. Every figure is read off the search that produced the action.
Restricted to actions the environment grammar can execute, and to alpha in {2, 5, 10} for path
diversity: **141 unique traces, 123 train / 18 dev with dev problems held out whole, 2563
supervised turns, all feasible, median mass/ref 0.649** -- against the policy's 0.917.

The SFT itself is cheap (155 steps, 17 minutes on 2 GPUs). Two things went wrong first and are
worth recording because both are easy to repeat:

* The first run took **6 optimizer steps**. `warmstart_reasoning` supervises every assistant turn
  inside ONE conversation-shaped example, so a trace is one example; at the inherited
  `gradient_accumulation_steps=16` on 2 GPUs, 59 traces is 1.8 steps per epoch. Coverage was never
  the problem, step count was.
* TRL 1.0's `packing=True` becomes padding-free batching, which rejects a custom data collator --
  and the collator is what applies the target's loss mask. `configs/sft/pretrain.yaml` already
  documented this; `warmstart` inherits `packing: true` from `base`.

### The distilled policy is temperature-brittle

Evaluated on the 34 held-out problems at **temperature 0.8**, the distilled policy produced **zero
parseable actions on 15 of 34 problems** -- and `n_simulator_calls = 1` on every one of them, so it
failed at turn 0 and never recovered. The other 19 averaged 0.868 grammar. The failure does not
track structure size (median 15 members among failures against 16 among successes, identical
ranges) and none of the failing problems was outside the training range.

At **greedy decoding the same checkpoint is fine** -- grammar 0.40 to 1.00, no zeros, solving
problems the champion also solves. So this is sampling brittleness, not a format regression: the
traces are short (median assistant turn 451 characters against the champion's ~1400 tokens) and
low-entropy, so the policy's output distribution is sharp and derails when sampled hot.

That matters beyond the evaluation. **GRPO samples rollouts at temperature 0.6**, so a distilled
checkpoint carries this brittleness into RL, where it would read as a collapse in grammar rate.
Mitigations, in order of expected value: more phrasing diversity in the generated reasoning; mixing
the original warmstart traces back in so the format prior is refreshed; and lowering the rollout
temperature. Any RL run started from a distilled checkpoint should check grammar rate in the first
few steps before spending a full run.

### A harness defect this exposed

When no action parses, both `eval_llm_search.py` and `eval_checkpoint._eval_problem_multiturn`
append the raw unparsed text as the turn's *action*, so it is rendered back into the next prompt as
an assistant turn full of garbage. One parse failure therefore poisons the remaining context. It is
not what caused the turn-0 failures above, but it is what prevents recovery from any of them, and
it applies to every evaluation in the study. The correct behaviour is to append a corrective user
turn, or to drop the failed turn, rather than to record malformed text as an action.

### Did distillation transfer the search's quality? Not as measured so far.

Head to head on the 34 held-out problems, greedy decoding, identical harness:

| cell | feasible | median mass/ref | grammar |
|---|---:|---:|---:|
| champion (GRPO) | **0.618** | 0.864 | 0.713 |
| T6 distilled (SFT on search traces) | 0.559 | 0.871 | 0.743 |

Paired: champion solves 21, T6 solves 19; T6 gets 3 the champion misses and misses 5 it solves;
on the 16 both solve the mass is indistinguishable (0.867 against 0.871, T6 lighter on 2/16).
The search that produced the traces reaches **0.620** on these problems. None of that transferred.

One caveat makes this an unfair test of the mass claim specifically, and it is the harness's fault
rather than the policy's: `eval_checkpoint._eval_problem_multiturn` **halts at the first feasible
design**, while every distillation trace is a 20-turn trajectory that keeps shedding mass *after*
reaching feasibility. The distilled policy is therefore never given a turn in which to exhibit the
behaviour it was trained on -- and its mean step count, 13.7, shows it stopping early. A greedy
run that continues past feasibility is the comparison that can see it; it is in flight.

Two further reasons the transfer may be weak regardless, worth separating before concluding:
155 optimizer steps on 123 traces is a small intervention for a 14B policy; and the reasoning in
the traces is generated from templates, so the model can fit the surface form without acquiring
the decision rule underneath. The second is testable by measuring the distilled policy's
`regret_vs_procedural` against the champion's -- if the decision rule transferred, regret should
fall even where the wording is copied.

### Verdict on SFT distillation: the wrong instrument, not merely too little data

Run in the harness the traces actually assume -- greedy decoding, continuing past the first
feasible design so the mass behaviour can show -- the comparison is decisive:

| cell | feasible | median mass/ref | grammar | **regret vs procedural candidates** |
|---|---:|---:|---:|---:|
| champion (GRPO) | **0.727** | 0.851 | 0.718 | **1.606** |
| T6 distilled (SFT on 2563 search turns) | 0.382 | 0.819 | 0.593 | **3.427** |
| teacher (the search itself) | -- | 0.620 | -- | 0 |

Paired on the 22 problems both finished: feasible **0.727 -> 0.364**; on the 8 both solve the
distilled policy is not lighter (0.861 against 0.828, lighter on 2/8).

The regret column is the evidence that matters. It measures the one thing distillation was meant
to transfer -- how far the policy's chosen action sits below the best available at that same state
-- and it **more than doubled**. A policy trained on 2563 of the search's own decisions came out
worse at deciding than the model that never saw them.

**Diagnosis: a shape mismatch, not a volume shortfall.** The search's competence is an argmax over
counterfactuals: "of the ~150 actions available here, this one raises Phi most". A demonstration
shows the model only which action won -- never the alternatives, the margin, or the ranking.
Behavioural cloning has to reverse-engineer the ranking function from single positive examples.
More traces buy more of the same signal.

Volume is a real secondary problem: 123 traces span only **41 distinct problems**, three alpha
variants each, against 34 held-out problems. But it is secondary, because the diagnostic says the
decision rule moved backwards rather than failing to arrive.

**The instruments that carry the comparison**, in order of cost:

1. `lookahead_advantage` (T5c, already training). Scores each action against the mean of what was
   achievable at the same state, delivered through GRPO's existing gradient path. No new trainer.
2. Preference pairs -- `scripts/distill_preference_pairs.py` extracts them from the search that was
   already being run: **2880 pairs, 48 distinct problems, 960 states, median Phi gap 0.049, drawn
   from 150 ranked candidates per state**. Both sides of a pair carry the SAME reasoning (the state
   facts the simulator reports) and differ only in the action, so the preference cannot be won by
   prose. Note this reads the same problems ~23x harder but does NOT increase problem diversity:
   48 distinct situations against SFT's 41. It therefore separates the two hypotheses cleanly --
   if the failure was learning a ranking function, preferences fix it; if it was generalising
   across structures, they will not.

A useful property of preferences here: they are defined at states where the trajectory FAILS, so
the problems the search never solved still produce data. That is why the pair set covers 48
problems where the trace set covered 41.

---

## 10. How much CAN the potential change GRPO's gradient?

TRL runs GRPO with `scale_rewards='group'`, so a reward function contributes only the ORDER it
induces over the K rollouts of a group. Two potentials that order every group identically train
identically, whatever their formulas look like. That bound is measurable, and it should be measured
before a null result is interpreted.

`scripts/group_ordering_divergence.py` samples 264 groups of 8 stochastic trajectories on the
training split and compares the two orderings:

| | |
|---|---|
| Phi_v1 and Phi_v2 pick the same best rollout | 61.7% |
| Kendall tau between the orderings | 0.786 median, 0.763 mean |
| groups where the orderings differ substantially (tau < 0.5) | 10.2% |

Restricted to the 87 groups containing BOTH feasible and infeasible rollouts -- the only groups
where the ordering can matter for feasibility:

| | top-ranked rollout is feasible |
|---|---:|
| **Phi_v1** | **69.0%** |
| **Phi_v2** | **97.7%** |

**The contrast is real and large.** Phi_v2 puts a feasible trajectory at the top of the group
97.7% of the time against Phi_v1's 69%: a 28.7-point difference in the quality of the signal
GRPO actually consumes. A null result in the T5a/T5b arms cannot be explained by the two
potentials being equivalent in practice, because they are not.

**But the same measurement explains why a null is plausible anyway.** Only 33% of groups are
mixed. At B = 2 groups per step over 100 steps, that is roughly **67 informative gradient events**
for an entire run -- a very thin budget for a 14B policy to acquire a ranking distinction. The
other two thirds of groups are uniformly feasible or uniformly infeasible, where the feasibility
ordering carries nothing (the mass ordering still differs, which is a weaker signal).

The prescription that follows is not a different formula. It is **more informative groups**:
longer training, more problems per batch, or a sampler that preferentially draws problems whose
groups come out mixed. The last is the most efficient and is a genuine method contribution --
curriculum by *signal informativeness* rather than by difficulty, computable from the same
potential.

---

## 11. A measurement-reliability warning

Evaluation is the throughput bottleneck (~2.5 h per 34-problem cell, sequential), so a turn-major
batched path was added -- the same computation in a different order, identical greedy decoding,
identical FEA, identical stopping rule. It ran 2.4x faster (4361 s against ~9060 s) and **did not
reproduce the sequential result**:

| | sequential | batched |
|---|---:|---:|
| feasibility | 0.618 | 0.500 |
| median mass/ref | 0.864 | 0.884 |
| per-problem feasibility agreement | -- | 28/34 |
| identical step counts | -- | 13/34 |

Two candidate causes, not yet separated:

* the batched path truncates generation at the first token of its own eos set, while the
  sequential path lets `generate` stop on the model's configured eos -- these can differ and yield
  a different parsed action;
* batching changes kernel reduction order, perturbing logits slightly, and greedy decoding over
  ~14 turns of 1536 tokens amplifies one flipped token into a different trajectory.

**The study keeps the sequential path.** Comparability across arms matters more than eval
throughput, and an unverified 12-point shift disqualifies the faster route.

The second explanation, if it is the right one, matters well beyond this optimisation: it would
mean a multi-turn greedy rollout is chaotically sensitive to numerically irrelevant perturbations,
and the true uncertainty on a 34-problem cell would be much wider than the +/-3.4 point seed-noise
estimate inherited from the alpha triplicate. That would be sufficient on its own to explain why
the T5 arms show no separation, and it would apply retroactively to every 25- and 50-problem cell
in this study.

**The test that separates them** is cheap and should be run before any arm comparison is called
final: evaluate one checkpoint twice through the batched path at batch sizes 1 and 8. Batch size 1
is arithmetically the sequential path, so agreement there isolates the eos-handling difference,
while disagreement between batch 1 and batch 8 isolates the numerical-cascade effect.

---

## 12. Why potential shaping could not work inside GRPO, and the one channel left

Eight arms found no benefit from the corrected potential in training. That is not a measurement
artefact alone -- the greedy metric is too noisy to rank arms (section 11), but the in-training
probe is not, and it says the same thing with 790 states per arm and 20 paired points:

| contrast (matched probe sampling) | n | mean d rank_frac | t |
|---|---:|---:|---:|
| Phi_v1 -> Phi_v2, composite | 20 | -0.0029 | -0.40 |
| **Phi_v1 -> Phi_v2, potential only** | 20 | **+0.0236** | **+2.16** |
| Phi_v2 only -> informative sampling | 11 | +0.0123 | +0.35 |
| Phi_v2 only -> DPO initialisation | 10 | -0.0143 | -1.39 |
| lookahead advantage: in composite -> alone | 14 | +0.0060 | +0.67 |

Higher rank_frac is worse. Nothing survives Bonferroni over five contrasts (|t| > 2.6), and the
one nominally-significant row points AWAY from the theory. Measured at identical initial states,
scored by the very potential the arm was trained on.

### The structural reason

GRPO computes ONE advantage per sequence and broadcasts it to every token. Decompose a per-token
advantage vector as

    A^k  =  mean_t(A^k_t) * 1   +   A~^k ,        A~^k orthogonal to 1

The signal a per-sequence reward can deliver is confined to the first term. Everything the
potential knows that is *specific to a step* -- which action at THIS state beat which alternatives
-- lives in the second, and **no group-level affine transform of any per-trajectory scalar can
produce it**. Under `scale_rewards='group'` the reward is used only up to an affine map, so the
gradient sees the ORDER of whole trajectories and nothing finer.

That explains three observations at once that otherwise look unrelated:

* summing the per-step counterfactual advantage into a scalar reward (T5c, T5co) moved nothing --
  the sum is precisely the projection onto `span{1}`;
* Phi_v2 orders MIXED groups far better than Phi_v1 (97.7% against 69.0%) yet changes nothing in
  training, because only 33% of groups are mixed and the per-trajectory ordering is all that
  reaches the gradient -- roughly 66 informative events in a 100-step run;
* the only intervention that demonstrably moved the policy's ranking was DPO on same-move
  preference pairs (0.724 against a 0.50 surface ceiling), which scores individual ACTIONS
  directly and never passes through a trajectory scalar.

### The remaining channel: per-token delivery (T8)

TRL 1.0 already accepts a (B, T) advantage tensor from a subclass (`grpo_trainer.py:2291`) and
merges a rollout's extra fields into `inputs` in place (`:1949`), so the per-token vector computed
during the rollout is available after `super()` returns. `llm_finetune/training/rl/
lookahead_grpo_trainer.py` supplies

    A_token  =  A_sequence  +  kappa * zscore(A_step)

with `A_step = Phi(s_{t+1}) - E_c Phi(f(s_t, c))` written onto exactly the tokens the policy
generated for turn t, and environment tokens carrying none. `kappa = 0` is bit-identical to stock
GRPO, so **T5bo is T8's exact control** and the pair isolates the delivery channel alone.

This is the theory's last untested form inside GRPO. If it moves nothing, the supportable claim is
the one the evidence already carries: the potential's specification is decisive for SEARCH
(+15.3 points of feasibility and 35% lighter designs on 130 truss problems; 25% -> 100% across a
second, unrelated domain), and it transfers to a policy through per-action PREFERENCES, but
trajectory-level RL is the wrong vehicle for a per-action quantity.

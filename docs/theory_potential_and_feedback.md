# Potential, Feedback, and Search: the theory that unifies the two results

> **Scope.** Two empirical results have to be explained by one theory:
> (R1) replacing the shipped potential Φ_v1 with the goal-aligned Φ_v2 moves an identical
> one-step search from 67% → 88% feasible on the 100-problem mass family, a strict superset
> (+21, −0); (R2) adding one line of simulator feedback — the buckling-critical member id — to
> the observation moved trained multi-turn GRPO from 12% → 68%/54%, while every search cell
> tied its non-search control. This document derives why both are true, why they are the *same*
> statement, and what survives the derivation as a real method.
>
> Companion documents: `docs/plan_posterior_reward_walkthrough.md` (the design that is being
> audited here), `docs/reward_rho_connection.md` (the config→ρ mapping), and
> `docs/turn5_potential_specification.md` (the Φ_v1 → Φ_v2 specification change).
> Every number below is reproducible from
> `results/search_ladder/full/{runs,lookahead_diagnostics}.jsonl` (130 problems, 1051 decision
> states, CPU only, no LLM) or from the arithmetic printed inline.

---

## 0. Notation and standing assumptions

Deterministic finite-horizon MDP `M = (S, A, T, r, γ, H)`, `T : S × A → S`, `γ = 0.99`,
`H ≤ 5` at training time (`configs/rl/grpo_mt_01b.yaml:18`, `max_turns: 5`).

| Symbol | Meaning |
|---|---|
| `μ : S → R^m` | **metric map** — the simulator readout (`mass, fos_buckling, fos_yielding, deflection, …`) |
| `G` | **goals dict**, the problem's own constraint statement (`minimum_x` / `maximum_x` convention) |
| `P = (f, {g_c}, ref)` | **design program** recovered from `G` (`potential.py:191-245`) |
| `f̂(s) = f(s)/ref` | normalised objective, oriented so lower is better (`potential.py:229-241`) |
| `ĝ_c(s)` | **relative** violation of constraint `c`, hinged (`potential.py:171-187`) |
| `Φ` | state potential; `Φ_v1` = `compute_potential` (`potential.py:47-68`), `Φ_v2` = `compute_potential_v2` (`potential.py:337-349`) |
| `c(s)` | **indicator / feedback field** — simulator-reported identity of the binding element or constraint |
| `A_c(s) ⊆ A` | actions whose support contains `c(s)` |
| `V̂^(D,b)(s,a)` | depth-`D`, branching-`b` lookahead value (`plan_…:§3.1`) |
| `F(s) = {s : ĝ_c(s) ≤ 0 ∀c}` | the feasible set |

Two potentials, written once:

```
Φ_v1(s) = log(m0/m) − α·[ sp(1.5−fos_b) + sp(1.5−fos_y) + sp(δ/0.01 − 1) ],   sp = softplus
Φ_v2(s) = −w·f̂(s)  − α·Σ_c hinge(ĝ_c(s)),   hinge(x) = τ·sp(x/τ)
```

`α = 5`, `w = 1`, `τ = 0.05` (`configs/rl/grpo_mt_01b.yaml:31-33`, `potential.py:251-259`).

**Champion configuration under analysis.** `mt_t3` = `configs/rl/grpo_mt_01b.yaml` (reward
`composite{lagrangian_potential: 1.0}`, `cost_fn: null`, `use_tree_expansion: false`,
`group_size: 8`) plus the enriched observation of `designbench_prompt.py:71-81`.

---

## 1. (T1) The champion's reward is a surrogate objective, not a shaping term

### 1.1 Definition: what Ng–Harada–Russell actually guarantees

**Definition 1 (NHR shaping).** Given base MDP `M = (S,A,T,r,γ)` and `Φ : S → R`, the shaped MDP
is `M' = (S,A,T,r+F,γ)` with `F(s,a,s') = γΦ(s') − Φ(s)`.

**Theorem (NHR 1999).** Every optimal policy of `M'` is optimal in `M` and conversely; the
guarantee is *relative to a base reward `r`*, and the mechanism is that the shaped return
differs from the base return by the constant `−Φ(s_0)` plus a terminal `γ^HΦ(s_H)` which is
absorbed by the convention `Φ(s_terminal) = 0`.

**Two conditions are load-bearing and both are violated here:**

* **(C1) A base reward must exist.** Invariance is a statement about which policies are optimal
  for `r`. With `r ≡ 0` every policy is optimal in `M`, so the theorem is vacuously true and
  says nothing about `M'`. Under the champion config the reward dict is exactly
  `{lagrangian_potential: 1.0}` and nothing else (`configs/rl/grpo_mt_01b.yaml:9-14`;
  `CompositeReward.compute`, `rewards.py:354-359`, sums only the declared components).
  So `r_base ≡ 0`.
* **(C2) The terminal potential must be zeroed.** NHR requires `Φ(s_H) = 0` at absorbing states
  (or an infinite horizon). The implementation applies `Φ` at `s_H` like any other state.

**Claim T1.** Under the champion configuration, `Φ_v1` is not a shaping term. It is the entire
objective: the policy optimises `γ^H Φ_v1(s_H) − Φ_v1(s_0)` (or, as implemented, the variant in
§1.2). Consequently the policy-invariance theorem is inapplicable, and *every* misspecification
of `Φ` is a misspecification of the task — not a harmless bias that a base reward would correct.

**Exact condition under which shaping would be invariant here.** Add a terminal task reward and
zero the potential at the horizon:

```
r_total(s,a,s') = R_task(s') + γΦ(s') − Φ(s),        Φ(s_terminal) := 0
R_task(s) = 1[s ∈ F(s)] · (1 + w·(1 − f̂(s))⁺)       (feasibility indicator + normalised quality)
```

Then the argmax over policies is determined by `R_task` alone, `Φ` only reshapes the credit path,
and *any* bounded `Φ` — including `Φ_v1` — is safe. This is the minimum change that converts the
present setup into the one the master doc claims it already is (`plan_…:§1.2` "No separate
terminal bonus is needed"; `§1.3` "Principled … Aligned with the literal objective"). Without it,
`§3.7`'s NHR-invariance check is checking a property the system does not have.

### 1.2 The implemented return does not even telescope

`plan_…:§1.2` asserts `Σ_t γ^t r_env,t = γ^H Φ(s_H) − Φ(s_0)`, and `§1.3` lists
"Telescopes by construction. Trajectory return depends only on `Φ(s_0), Φ(s_H)`" as a property.

The implementation accumulates the per-step shaping term **without the `γ^t` factor**
(`rewards.py:406-417`, and identically in the v2 reward at `rewards.py:757-767`):

```python
for t in range(len(history) - 1):
    total += compute_step_reward(history[t], history[t+1], gamma=self.gamma, alpha=self.alpha)
```

Therefore the trajectory reward actually handed to GRPO is

```
R_impl = Σ_{t=0}^{H-1} [ γΦ(s_{t+1}) − Φ(s_t) ]
       = γΦ(s_H) − Φ(s_0)  +  (γ−1)·Σ_{t=1}^{H-1} Φ(s_t)              (†)
```

The residual `(γ−1)ΣΦ_t` is a **path integral**, not a boundary term. Two consequences:

* **A stationary state pays a positive reward.** At any `s` with `Φ(s) < 0`, a no-op transition
  earns `γΦ − Φ = (1−γ)|Φ| > 0`. Measured on the 1051 decision states the greedy policy visits,
  the violation term alone gives `α·V` with median 9.0 and mean 10.6, so **a wasted turn at a
  median infeasible state is worth `+0.09` of reward**, and a 5-turn stall is worth `+0.45`.
  This is a reward for lingering in violation, proportional to how bad the violation is.
* **The lead's finding #1 is *almost* right and worth sharpening.** Under `use_tree_expansion:true`
  the trainer sets the reward to `γΦ_H^k − mean_j Φ_H^j` (`grpo_trainer.py:331-333`), which uses
  *only* terminal potentials. Under the champion (`use_tree_expansion:false`) the reward is (†).
  After GRPO's within-group mean-centring the two differ by exactly

  ```
  (γ−1)·[ Σ_t Φ(s_t^k) − mean_j Σ_t Φ(s_t^j) ]
  ```

  — the group-centred path integral. The two cells are affine-equivalent **up to this term**, not
  exactly. Since the term is bounded by `(1−γ)(H−1)·max|Φ| ≈ 0.01·4·18.6 ≈ 0.74` at the p90 of
  observed `α·V`, against a terminal spread of order 10, the equivalence is a ~7% approximation.
  The 12% vs 12% tie is still the expected outcome; the residual is a nuisance term that
  rewards stalling, not a second mechanism.

**Fix (one line).** Multiply the accumulated term by `γ**t` in both loops, or — better, since the
horizon is short and the discount is cosmetic — set `γ = 1` and add the terminal task reward of
§1.1. With `γ = 1` the residual in (†) vanishes identically and the stall bonus goes to zero.

### 1.3 Consequence: `argmax Φ_v1 ⊄ F`

Because `Φ_v1` *is* the objective (§1.1), the only thing that matters about it is whether its
argmax is the feasible set. It is not. Reproduced exactly on `auto_problem_000`
(goals `{minimum_fos_buckling: 1.5, minimum_fos_yielding: 1.5, maximum_mass: 264.82}`,
`_metadata.optimal_mass = 240.75`, initial mass 102.55):

| design | mass | fos_b | fos_y | feasible | `Φ_v1` | `Φ_v2` |
|---|---:|---:|---:|:--:|---:|---:|
| lean | 250 | 1.55 | 1.60 | **yes** | −9.02 | **−1.27** |
| over-stiffened | 900 | 6.00 | 8.00 | no (mass 900 > 265) | **−3.80** | −15.73 |

`Φ_v1` ranks the infeasible design **5.22 above** the feasible one; `Φ_v2` ranks the feasible one
14.46 above. The gap is **independent of `m0`**, which is what makes it a specification error
rather than a calibration error:

```
Φ_v1(stiff) − Φ_v1(lean) = [log(m0/900) − log(m0/250)] − α[V(stiff) − V(lean)]
                         = −log(3.6) − 5·(0.325812 − 1.626160)
                         = −1.2809 + 6.5017 = +5.2208
```

The three independent defects, each visible in that arithmetic:

1. **A phantom constraint priced at `α`.** `violation_score` (`potential.py:40-44`) always adds
   `sp(δ/0.01 − 1)`. 100 of 130 problems have no deflection goal. Both rows above carry the same
   `sp(−1) = 0.3133` term, so it cancels in *this* comparison, but along a trajectory it is a
   standing gradient toward stiffness, and stiffness costs mass.
2. **The real constraint is unpriced.** `mass > maximum_mass` produces no term at all. Mass
   enters only as `log(m0/m)`, at weight 1 against the violations' weight 5, and with the wrong
   sign for this benchmark: `maximum_mass = 1.1 × optimal_mass` and the initial design is a
   degraded, under-built version of the optimum, so mass *must rise* to reach feasibility.
3. **Nothing saturates.** `sp(1.5−1.55) = 0.6685` versus `sp(1.5−6.0) = 0.0110`: pushing FOS from
   1.55 to 6.0 still buys `α·0.657 ≈ 3.29`. `hinge` (`potential.py:251-259`) fixes this up to a
   bounded residual: a satisfied constraint pays at most `α·τ·log 2 = 5·0.05·0.693 = 0.173`, and
   that residual decays exponentially in the margin (in the lean row above, all three satisfied
   constraints together contribute only 0.233).

**Note on quoted numbers.** `docs/turn5_potential_specification.md:82-83` reports this pair as
−18.09 / −4.42 (gap 13.67). That table assumed a nonzero deflection for the lean design; the
`m0`-invariant gap recomputed here with `deflection` absent is 5.22. The *direction and the
conclusion are unchanged* — `Φ_v1` strictly prefers the infeasible design — but the magnitude in
that document should be restated as 5.22 or its deflection assumption made explicit.

### 1.4 The theorem this section replaces

`plan_…:§1.3` claims `Φ_v1`-maximisation "= mass-minimisation subject to feasibility, smoothed".
That is true only if `Φ` is the Lagrangian of *the problem's own program*. It is:

**Proposition 1.** Let `P` have objective `f` and constraints `{g_c ≤ 0}` and let
`Φ(s) = −w f̂(s) − α Σ_c hinge(ĝ_c(s))`. If `α > w · L / min_c s_c` where `L` is a Lipschitz
constant of `f̂` over the reachable set and `s_c` the normaliser of constraint `c`, then
`argmax_s Φ(s) ⊆ F` whenever `F ≠ ∅` up to the hinge residual `α τ log 2` per constraint.
*Proof sketch.* Off `F`, moving any violated `ĝ_c` down by `ε` gains `≥ α(ε − τ log 2)` while the
objective can lose at most `w L ε / s_c`; by the assumption on `α` the net is positive, so no
maximiser lies strictly outside `F` beyond the residual. ∎

`Φ_v2` satisfies the hypothesis by construction (`ĝ_c` is relative, so `s_c = |limit|`, and `f̂`
is `O(1)` by the `ref` normalisation, `potential.py:401-406`). `Φ_v1` does not, because the
`mass ≤ m*` constraint is absent from its violation sum entirely — `α` cannot dominate a term
that is not there. **This is the whole of (R1).** Search does not fail; search succeeds at the
program it was given, which is the wrong program.

---

## 2. (T2) State feedback is a depth-limited search oracle

### 2.1 Setup

Assume the simulator exposes, alongside `μ(s)`, an **indicator** `c(s)`: the identity of the
element or constraint that attains the worst margin. This is not a modelling assumption — it is
already emitted:

* truss: `analyze_truss` returns `min_fos_buckling_member_id` / `min_fos_yielding_member_id`
  as the `argmin` over members (`DesignBench/validation/truss_executor.py:118-132`), and these
  are injected verbatim into the observation at
  `llm_finetune/data/processors/designbench_prompt.py:80-81`
  (`"— worst member(s): M3"`). That single line is the entire (R2) intervention.
* battery: `_binding` returns the constraint key with the largest relative violation
  (`scripts/battery_ladder.py:251-260`).

Let `supp(a)` be the set of design variables an action writes, and
`A_c(s) := { a ∈ A : c(s) ∈ supp(a) }`.

**Depth-1 with truncated continuation is one-step greedy on Φ.** With
`Ṽ^(0) = Φ` (`plan_…:§3.1`, "Truncated" continuation),

```
V̂^(1)(s,a) = [γΦ(T(s,a)) − Φ(s)] + γ·Φ(T(s,a)) ∝_a Φ(T(s,a))
```

so `argmax_a V̂^(1)(s,a) = argmax_a Φ(T(s,a))`. This is exactly what `run_lookahead`
computes (`scripts/search_ladder.py:399-470`), and what `_lookahead_diagnostic` scores at every
state the greedy policy visits (`scripts/search_ladder.py:268-340`).

### 2.2 Definition: ε-sufficiency of the feedback field

**Definition 2.** `c` is an **ε-sufficient statistic for the depth-1 argmax at `s`** iff

```
ε_c(s) := max_{a ∈ A} Φ(T(s,a))  −  max_{a ∈ A_c(s)} Φ(T(s,a))   ≤   ε.
```

`ε_c(s) = 0` means restricting the search to the feedback-indicated action set costs *nothing*:
the depth-1 oracle can be replaced, at zero simulator cost, by reading one field of the
observation. This is the precise sense in which "state feedback is a search oracle".

### 2.3 The sufficiency theorem

**Proposition 2.** Fix `s` and suppose:

* **(H1) Single active constraint, no one-step coupling.** Exactly one constraint `c*` has
  `ĝ_{c*}(s) > 0`; every other constraint satisfies `ĝ_c(s) ≤ −τ`, and no `a ∈ A` moves any
  satisfied constraint above its limit in one step.
* **(H2) Violation dominance.** For all `a, a′`: if `ĝ_{c*}(T(s,a)) < ĝ_{c*}(T(s,a′))` then
  `α[ĝ_{c*}(T(s,a′)) − ĝ_{c*}(T(s,a))] > w[f̂(T(s,a)) − f̂(T(s,a′))]`. (The objective can never
  overturn a strict violation improvement.)
* **(H3) Critical-element monotonicity.** The binding metric is a min over elements,
  `g_{c*}(s) = min_i g_{c*,i}(s)`, attained at `i = c(s)`; and for `a ∉ A_c(s)`,
  `g_{c*,c(s)}(T(s,a)) = g_{c*,c(s)}(s)`.

Then `ε_c(s) = 0`, i.e. `argmax_a Φ(T(s,a)) ∈ A_c(s)`.

*Proof.* Under (H1) all satisfied constraints have `hinge(ĝ_c) ≤ τ log 2` before and after any
one-step transition, so `Φ(T(s,a)) = −w f̂(T(s,a)) − α·hinge(ĝ_{c*}(T(s,a))) + O(ατ)`. Under (H3),
an action outside `A_c(s)` leaves the argmin element's margin unchanged, so the min — hence
`ĝ_{c*}` — cannot decrease, so `hinge(ĝ_{c*})` cannot decrease. Any `a ∈ A_c(s)` that strictly
decreases it therefore weakly dominates on the violation term, and by (H2) the objective term
cannot reverse the comparison. Hence the maximiser lies in `A_c(s)`. ∎

The theorem is exactly why the FOS feedback is powerful: `fos_buckling` is defined as a `min`
over members and the simulator hands over the `argmin`. **`c(s)` is a pointer to the argmin of
the term that dominates `Φ`.** Reading it is a free evaluation of the inner maximisation.

### 2.4 The three failure modes, each with its measured rate

Each hypothesis is falsifiable and each break was measured over the 1051 greedy-visited states
of `results/search_ladder/full/lookahead_diagnostics.jsonl` (median 160 candidate actions
scored per state, one FEA call each).

| Break | Hypothesis violated | Mechanism | Measured |
|---|---|---|---|
| **B1 Constraint coupling** | H1 | relieving buckling on member `i` redistributes load and pushes yielding on member `j` past its limit | at the 176 states where **both** FOS constraints bind, the member-level argmax lies in `A_c` only **64.8%** (`Φ_v2`) / **56.8%** (`Φ_v1`) of the time, versus **69.5% / 72.5%** at the 636 states where exactly one binds |
| **B2 Non-local actions** | H3 | an all-member `SCALE_MULTI_PARAM` trivially contains `c(s)` but is not "targeted"; and load redistribution means acting on `j ≠ c(s)` *can* raise the min | the unrestricted depth-1 argmax is a **global** all-member action at **57.1%** (`Φ_v2`) / **75.5%** (`Φ_v1`) of states |
| **B3 Objective competes** | H2 | when no constraint is violated the hinge is saturated and `Φ` is governed by `f̂` alone, so `c(s)` is undefined or uninformative | at the 44 states with **neither** FOS binding, agreement collapses to **13.6%** (both potentials) and the hint costs a strictly positive median `Φ` (0.026 `v2`, 0.031 `v1`) versus a median of **exactly 0.000** everywhere else |

**Aggregate ε.** Over all 856 states where a critical-member action exists:

```
ε_c(s) = 0        at 65.8% of states  (Φ_v2)   |  66.2%  (Φ_v1)
ε_c(s) ≤ 0.098    at 82.2%            (Φ_v2)   |  74.6%  (Φ_v1)
```

(0.098 is the master doc's own ranking-resolution threshold `2γ^{D+1}σ_G`, `plan_…:§3.3`.)

### 2.5 The formal statement of "why feedback beat search"

**Corollary.** On this benchmark, `c` is a 0-sufficient statistic for the depth-1 argmax at
about two thirds of decision states and an 0.098-sufficient statistic at about five sixths.
Therefore:

* Publishing `c(s)` in the observation delivers ≈82% of the depth-1 oracle's value at **zero**
  additional simulator calls and zero change to the training objective.
* Running the depth-1 oracle instead costs a median **181 FEA calls per problem versus 8** for
  the feedback-guided greedy policy (`results/search_ladder/full/runs.jsonl`) — **22.6×** — to
  buy the remaining ≈18%.

The 12%→68% jump is therefore not evidence against the potential-of-an-action thesis. It is
evidence that **the cheapest possible estimator of that potential was already sitting unused in
the simulator's output**, and that the expensive estimator was being spent uniformly over states
where the cheap one is exactly optimal.

**Where search retains value — precisely the complement.** The set `{s : ε_c(s) > 0.098}` is
17.8% of states under `Φ_v2`, and it is *enriched* exactly in the B1 and B3 regimes predicted by
Proposition 2 (§3.2 quantifies this). At the problem level this shows up as a genuine, small,
one-directional gain: `lookahead_v2_d1` solves 5 problems `greedy_critical` misses and misses 3
it solves (88.5% vs 86.2% overall), and `lookahead_v2_d2` is a **strict superset** of
`lookahead_v2_d1` (+3, −0), 90.8%.

**A caution that qualified the "search wins on quality" hypothesis — since resolved.** On the
first ladder (stop-at-first-feasible, no compound action) `lookahead_v2_d1` ended **heavier** than
`greedy_critical` on the problems both solved: median `+8.6%`, greedy lighter on 75 of 109. The
mechanism diagnosed here was correct: when a constraint is violated the depth-1 argmax buys the
most violation relief per step, which is a *global* scale-up (57.1% of states), and a search that
halts the moment it becomes feasible never spends a turn taking the mass back off. Depth-1 search
under a violation-dominated `Φ` is mass-blind while infeasible.

Two changes, both implied by that diagnosis, reversed it, and the later ladders supersede the
numbers above:

1. **Run the budget out instead of halting at feasibility.** Once feasible the violation term is
   saturated, so the objective term is the only gradient left and the same search spends its
   remaining turns shedding mass.
2. **Give the search an action the grammar can express in one move.** The per-member argmax cannot
   reproduce what fully-stressed design does in a single step; with a compound "resize each member
   by its own utilisation" action available, it can.

Measured on the validity-gated 130-problem ladder, paired on jointly-solved problems:
`lookahead_v2_d1` 0.602 against `greedy_critical` 0.875 — **lighter on 81/92 (88%)** — and with the
compound action 0.422 against fully-stressed design's 0.493, **lighter on 90/97 (93%)**. So search
does win on quality, once it is allowed to keep optimising and to express the move. The two-phase
`α` prescription below remains correct and is separately confirmed: on the battery `thermal_hard`
family, where the objective can outbid a small violation, a lexicographic feasibility offset took
the domain from 91.7% to **100%** feasible with 33% fewer simulator calls.

## 3. (T3) The decision rule: when to spend a search budget

### 3.1 The master doc's condition, made measurable

`plan_…:§3.3` (Theorem 2) states: ranking is preserved when
`Δ_rank > 2 γ^{D+1} σ_G(s)`, with `σ_G(s)` the standard deviation of greedy-myopia gaps across
candidate continuation states. Two problems with using it as written:

1. **`σ_G` is not computable at decision time.** `Δ_G(s) = V*(s) − V^G(s)` requires `V*`. The doc
   substitutes an assumed `σ_G ≈ 0.05` (`plan_…:§3.3`, "empirical estimate") and every numeric
   row in that section inherits the assumption.
2. **`Δ_rank` is defined on `V*`, not on the estimator.** What a runtime gate can see is the
   estimator's own top-1/top-2 gap.

**Replace both with the observable resolution gap.** Define

```
Δ̂_rank(s) := V̂^(D)(s, a_(1)) − V̂^(D)(s, a_(2))     (top two candidates under the estimator)
```

and read Theorem 2 contrapositively: **if `Δ̂_rank(s) ≤ 2γ^{D+1}σ̂_G` the ranking is below the
estimator's resolution, and the two candidates are `Φ`-equivalent to within the bias — so no
budget can buy a better decision at this state.** Note the direction: a small margin is not a
reason to search harder, it is a reason to stop, because the alternatives are interchangeable in
the objective.

Measured (`Φ_v2`, 856 states, threshold 0.098):

| candidate granularity | median `Δ̂_rank` | fraction below resolution |
|---|---:|---:|
| all candidates (incl. macros / global) | 0.0967 | 50.2% |
| member-targeted actions only | 0.0033 | **87.7%** |

This *confirms* `plan_…:§3.3`'s qualitative split ("D=0 sufficient for action-class ranking;
D≥1 needed for within-class parameter ranking") and **inverts its prescription**: within-class
parameter ranking is precisely the granularity at which the estimator has no resolution at
7/8 of states. Depth spent there buys unresolvable distinctions. Depth is worth spending only at
the class/scope granularity, and only where the class choice is contested.

### 3.2 The gate

**Decision rule.** Expand a lookahead tree at `s` iff

```
Expand(s)  :=  [ ε̂_c(s) > θ_ε ]  ∧  [ Δ̂_rank(s) > 2 γ^{D+1} σ̂_G ]
                 hint insufficient        decision resolvable
```

Both terms must be estimable *without* the tree, or the gate costs what it saves. Cheap proxies,
computable from `μ(s)` alone (zero extra simulator calls):

* `ε̂_c(s)` high ⟸ **more than one constraint violated** (B1) **or none violated** (B3). Both are
  a scan of the metric vector against the program's constraints —
  `DesignProgram.violations(state)` (`potential.py:223-224`).
* `Δ̂_rank(s)` high ⟸ the LLM's own `K`-sample action-**class** entropy, which `plan_…:§3.5`
  already proposes as `θ_expand`; or, cheaper, the spread of `ĝ_c` across constraints.

**Measured budget.** Applying the gate literally (`θ_ε = 0.098`, resolution 0.098) to the 856
diagnostic states under `Φ_v2`:

| gate term | fires at |
|---|---:|
| hint insufficient (`ε_c > 0.098`) | 17.8% |
| decision resolvable (`Δ̂_rank > 0.098`) | 58.4% |
| **both — search-worthy** | **11.6%** |

and the search-worthy set is enriched exactly where Proposition 2 predicts:

| regime | share of all states | share that are search-worthy |
|---|---:|---:|
| both FOS constraints bind (B1) | 176 | **21.0%** |
| neither binds — objective-governed (B3) | 44 | **31.8%** |
| exactly one binds (Proposition 2 holds) | 636 | 7.5% |

So the gate cuts the tree budget by **8.6×** while concentrating it on the two regimes where the
feedback field provably loses information. Under `Φ_v1` the same gate fires at 23.2% — twice as
often — which is another way of saying a misspecified potential manufactures spurious
decision points.

### 3.3 The correct ρ(t) estimator, and why the current one cannot exist

`plan_…:§3.6` defines the headline diagnostic

```
ρ(t) := Pr[ argmax_k π_θ(a_k | s) = argmax_k V̂^(D,b)(s, a_k) ]
```

with the falsifiability target `ρ ≥ 0.85`. Note the two argmaxes are over **the same candidate
set `{a_k}` at the same state `s`**. The correct estimator is therefore:

```
ρ̂(t) = (1/|B|) Σ_{s ∈ B}  1[ argmax_{a ∈ K(s)} log π_θ(a | s)  ==  argmax_{a ∈ K(s)} V̂^(D)(s, a) ]
```

where, per sampled state `s` on the current on-policy state distribution:
1. `K(s)` = `K` distinct parsed actions sampled from `π_θ(·|s)` **at that one state**;
2. each is scored by one simulator call, `V̂^(1)(s,a) ∝ Φ(T(s,a))` (truncated continuation);
3. the policy's argmax is the candidate with the highest sequence log-probability under `π_θ`
   (the generation logprobs are already produced by the rollout, so this is free).

Cost: `K` simulator calls per gated state — the same `K` the group already spends, but at a
*shared* state instead of on divergent trajectories.

**Why `_compute_rho_per_group` (`grpo_trainer.py:345-373`) cannot measure this.** Four
independent reasons, any one of which is fatal:

1. **It compares trajectories, not actions at a state.** `groups[problem_id]` collects the `K`
   *rollouts* of a GRPO group (`grpo_trainer.py:358-360`). They share `s_0` and diverge
   immediately. `argmax` over them is a comparison across different terminal states, so the
   conditional `| s` in ρ's definition has no referent.
2. **Under tree mode it is an algebraic identity.** The reward is set to
   `γΦ_H^k − mean_j Φ_H^j` (`grpo_trainer.py:331-333`), a strictly increasing affine function of
   `Φ_H^k` within the group (`γ > 0`, the subtrahend is constant per group). Hence
   `argmax_k reward_k ≡ argmax_k Φ_H^k` and `ρ ≡ 1` by construction, at
   `grpo_trainer.py:369-371`.
3. **Under the champion (non-tree) config it is a near-coin-flip with no lookahead content.**
   The reward is (†) of §1.2; it differs from `γΦ_H` only by `−Φ_0` (constant per group) and the
   group-centred path integral. So `ρ` measures whether the trajectory with the best terminal
   potential also had the best `(1−γ)`-weighted path integral — a nuisance comparison, ≈`1/K`
   under exchangeability.
4. **No `V̂^(D,b)` is ever formed.** `use_tree_expansion: true` routes to
   `_compute_group_tree_advantages` (`grpo_trainer.py:434-436`), which computes only terminal
   potentials and never expands a tree or calls an evaluator. The second operand of ρ's argmax
   does not exist anywhere in the training loop.

`docs/reward_rho_connection.md` ("Where ρ(t) is measured") correctly points at
`evaluator.py:633`, `rho_tree_agreement=int(top_model_action == tree_result.tree_best_action)`,
which *is* the right per-context comparison — but it runs offline in
`scripts/eval_posterior_one_step.py`, over that script's own candidate generation, not on the
training state distribution. The claim in that document that `tree_advantage` is the "primary
ρ(t) driver" and that "GRPO directly pulls π_θ toward the tree-best ranking" is not supported by
the code path it cites: the flywheel diagram's middle box (`D=1, b=5` lookahead, ~24 FEA
calls/step) is never executed.

**Minimal instrumentation to make the thesis falsifiable.** Log, per gated state, the triple
`(ε̂_c(s), Δ̂_rank(s), 1[policy-argmax == Φ-argmax])`. Then `ρ(t)` is measurable, and — more
useful — it is decomposable into `ρ` restricted to the search-worthy 11.6% versus the
hint-sufficient 82.2%. The thesis "the LLM internalises lookahead" predicts the *first* rises;
the thesis "the LLM internalises the feedback field" predicts only the second does. The current
logging cannot distinguish these, which is why the study could not adjudicate its own claim.

---

## 4. (T4) Domain generality

### 4.1 The interface a domain must expose

Everything above is stated in terms of `(f, {g_c}, A, T)` and never in terms of trusses. A
domain plugs in by supplying five things, of which only the third contains any domain code:

| # | Object | Truss | Battery |
|---|---|---|---|
| 1 | **metric map** `μ : S → R^m` | `analyze_truss` → `mass, fos_buckling, fos_yielding, deflection` (`truss_executor.py:95-141`) | `BatterySimulator._metrics` → `charge_time, max_temperature, max_plating, min_neg_potential, energy_density` (`scripts/battery_env.py:248-290`) |
| 2 | **goals dict** in the `minimum_`/`maximum_` convention | `{minimum_fos_buckling: 1.5, minimum_fos_yielding: 1.5, maximum_mass: 264.8}` | `{max_temperature, max_plating, max_charge_time, min_neg_potential}` |
| 3 | **objective declaration** `(key, sense, ref)` | `program_from_truss_spec`: `mass`, `min`, `ref = _metadata.optimal_mass` (`potential.py:379-414`) | `program_from_battery_goals`: `charge_time`, `min`, `ref = max_charge_time` (`potential.py:423-448`) |
| 4 | **action enumerator** `A(s)` with `supp(a)` | `SCALE_PARAM(mid, param, f)` etc. (`search_ladder.py:133-166`) | `SCALE_PARAM(neg_thickness, 0.9)` etc. (`battery_env.py:298-345`) |
| 5 | **transition** `T = execute ∘ simulate` | `execute_grammar_action` + `analyze_truss` | `execute_action` + `BatterySimulator.evaluate` |
| 6* | **indicator map** `c(s)` (optional; required for T2) | `min_fos_*_member_id`, element-level (`truss_executor.py:118-132`) | `_binding` → constraint key, constraint-level (`battery_ladder.py:251-260`) |

Given 1–3, `program_from_goals` (`potential.py:280-334`) recovers the constraints by naming
convention and `compute_potential_v2` is the potential — no per-domain potential code is
written. **A new domain is onboarded by naming its goals.** `Φ_v1` is not portable at all: it
hardcodes `fos_buckling`, `fos_yielding`, `deflection ≤ 0.01`, and `log(m0/m)`
(`potential.py:24-68`).

Both domains' Lagrangians are commensurate because both `f̂` and `ĝ_c` are dimensionless: `ĝ_c` is
a *relative* violation `(value − limit)/scale` and `f̂` is `value/ref`. That is what makes `α = 5`
mean the same thing for a 265 kg mass cap and a 900 s charge-time cap.

### 4.2 What is honestly **not** general

1. **The naming convention is syntax, not semantics.** If a goal's stripped name does not match a
   metric key, `_resolve_metric_key` (`potential.py:262-277`) falls through to the stripped
   string, `Constraint.violation` finds nothing, and returns `missing = 0.0`
   (`potential.py:173-175`, returning `missing=0.0`) — a **silently unpriced constraint**, i.e. the exact `Φ_v1` failure
   mode reintroduced through a typo. *Required hardening:* assert at program construction that
   every `Constraint.key` is present in the metric map, and fail loudly otherwise.
2. **`α` is comparable across domains only when no limit is near zero.** The default normaliser
   is `max(|limit|, 1e-9)` (`potential.py:324`). A constraint like `min_neg_potential ≥ 0`
   degenerates: the relative violation explodes and is only saved by the hard cap at 100
   (`potential.py:187`). Such constraints need an explicit `scales` entry, and until they get one
   the cross-domain `α` claim is false for them.
3. **`Φ_v2` is not exactly flat on the feasible set.** The smooth hinge leaves `≤ α τ log 2 =
   0.173` per satisfied constraint. Bounded and controllable via `τ`, unlike `Φ_v1`'s unbounded
   softplus — but it does mean a tiny residual incentive to over-satisfy remains.
4. **The indicator map is the least portable piece, and T2 lives on it.** The truss indicator is
   *element-level* (`M3`), so `A_c` is a small, sharp subset of `A`. The battery indicator is
   *constraint-level* (`max_plating`), so `A_c` is whatever relief heuristic maps that constraint
   to knobs (`_RELIEF`, `battery_ladder.py:265-290`) — a hand-authored table, not a simulator
   output. Expect `ε_c` to be materially worse there, and (H3)'s "min over elements" structure to
   be absent. **This is untested:** `scripts/battery_ladder.py` mirrors `search_ladder.py`
   line-for-line but `results/battery_ladder/` currently contains only probe and parameter-scan
   output — no `runs.jsonl`. Replicating the §2.4 table on batteries is the single experiment
   that decides whether T2 is a theorem about design or a theorem about trusses.
5. **Determinism.** Proposition 2 and `plan_…:§3.2` assume deterministic `T`. `trussme` is
   deterministic; PyBaMM is deterministic but *discontinuous*: a failed solve snaps every metric
   to a sentinel (`charge_time: 999, max_temperature: 1000, …`, `battery_env.py:85-94`). Those
   sentinels are in the constraints' own units, so `hinge` saturates at the cap and `Φ` is
   **flat over the entire failure region** — no ranking signal, so neither search nor feedback
   can escape it by argmax. This is the battery analogue of the 9 structurally degenerate truss
   problems (FEA returns `mass = inf` / `fos = 0` regardless of sizing) among the 11 that no
   procedural policy solves. Domains need an escape gradient defined *inside* their failure
   region, or those instances are unreachable by any potential-guided method.
6. **The action grammar and its locality partition** (`plan_…:§0.4`, Local/Mid/Non-local) is
   truss-specific, and `§4.3`'s stagnation-escape term is defined in terms of it. `§8.5` of the
   master doc already flags this as open; it remains open.
7. **The `§2` tree-mined signals** (`r_macro`, `r_dead`, `ρ_nn`) require a corpus of historical
   search records per domain. None exists for batteries or linkages. They are not part of the
   portable core.

### 4.3 A measurement caveat that affects the proposed headline metric

The natural quality metric is mass at feasibility versus `_metadata.optimal_mass`. **That
reference is not currently a valid lower bound.** Among the 86 feasible `greedy_critical` runs on
the mass family, **97.7% end below `optimal_mass`** (median ratio 0.834, min 0.441), while
`is_feasible` genuinely enforces `mass ≤ maximum_mass` (`truss_executor.py:142-156`). Either the
ground-structure LP that produced `optimal_mass` uses a different admissibility model than
`trussme`, or the field means something other than "minimum feasible mass". Validate that
reference before any quality claim is built on it; otherwise report mass against
`maximum_mass` (the constraint that is actually enforced).

---

## 5. Ledger: master doc vs implementation vs measurement

| # | Master-doc claim | Status | Evidence |
|---|---|---|---|
| 1 | `§1.2/§1.3` "telescopes by construction; return depends only on `Φ(s_0), Φ(s_H)`" | **contradicted by implementation** | `rewards.py:406-417` and `:757-767` omit `γ^t`; return carries `(γ−1)Σ_{t=1}^{H-1}Φ_t` |
| 2 | `§1.2` "the optimal policy under `r_env` alone maximises `Φ(s_H)`… No separate terminal bonus is needed" | **true, and that is the defect** | with `r_base ≡ 0` NHR is vacuous; `Φ` is the objective (§1.1), so `Φ`'s misspecification is the task's |
| 3 | `§1.3` "Aligned with the literal objective" | **false for `Φ_v1`** | infeasible 900 kg design scores +5.22 above a feasible 250 kg design; mass cap never priced (`potential.py:40-44`) |
| 4 | `§3.7` NHR-invariance check | **inapplicable** | invariance is asserted relative to a base reward that does not exist; also `Φ(s_H) ≠ 0` |
| 5 | `§3.1` `V̂^(D,b)` is the advantage signal; `reward_rho_connection.md` "GRPO directly pulls π_θ toward the tree-best ranking" | **not executed** | `use_tree_expansion:true` → `grpo_trainer.py:278-342`, terminal potentials only, no tree |
| 6 | `§3.3` `σ_G ≈ 0.05`, all numeric rows | **assumed, not measured** | replace with observable `Δ̂_rank`: median 0.0967 (all candidates) / 0.0033 (member actions) |
| 7 | `§3.3` "D=0 sufficient for class ranking, D≥1 for within-class" | **confirmed, prescription inverted** | within-class margins are below resolution at 87.7% of states — depth there is unbuyable |
| 8 | `§3.5` `D=1,b=5` default at 6× FEA cost | **cost understated in practice** | measured 181 vs 8 median FEA calls/problem (22.6×) for +2.3 pp feasibility |
| 9 | `§3.6` ρ(t) ≥ 0.85 falsifiability target | **unmeasurable as implemented** | `grpo_trainer.py:345-373`; identity under tree mode, coin-flip otherwise, no `V̂` operand |
| 10 | `§3.5` adaptive expansion at `θ_expand` | **right idea, wrong trigger** | LLM entropy alone gates on resolvability; the sufficiency term `ε̂_c` is the one that matters (11.6% vs 58.4%) |
| 11 | `plan_…:§0.2` feasibility `f(s)` includes `δ ≤ 0.01` for all problems | **false** | 100/130 problems have no deflection goal; 30 have no mass goal |
| 12 | `docs/turn5_potential_specification.md:82-83` Φ_v1 = −18.09 / −4.42 | **direction right, magnitude unreproducible** | `m0`-invariant gap is +5.22 with `deflection` absent; state the deflection assumption |
| 13 | — (not in the doc) | **format bug still live** | the prompt's worked example `SCALE_PARAM(all_members, thickness, 1.224)` (`designbench_prompt.py:44`) is rejected by `SCALE_PARAM\s*\(\s*(\d+)` (`truss_executor.py:211`); every imitation of the example fails |

---

## 6. What the theory says to do

1. **Fix the objective before fixing the estimator.** Add `R_task(s_H)` and set `Φ(s_terminal)=0`
   (§1.1), switch to `Φ_v2`, and either restore `γ^t` or set `γ=1` (§1.2). Until then no search
   result is interpretable, because search is optimising a program the benchmark did not state.
2. **Treat `c(s)` as the depth-0 rung of the search ladder**, not as a competing idea. The
   ladder is: `c(s)` free → depth-1 at `K` sim calls → depth-`D` at `K b^{D-1}`. (R2) says rung 0
   is nearly free and nearly optimal; (R1) says every rung is worthless under a misspecified `Φ`.
3. **Gate the budget on `Expand(s)` (§3.2)** — 11.6% of states, 8.6× cheaper, concentrated on the
   coupled and objective-governed regimes where Proposition 2 provably fails.
4. **Log `ρ̂(t)` correctly (§3.3), split by the gate.** That single instrument decides whether the
   claimed innovation exists: `ρ̂` rising on the search-worthy subset is the thesis; `ρ̂` rising
   only on the hint-sufficient subset means the model learned to read the feedback line.
5. **Run `scripts/battery_ladder.py` to completion** and reproduce the §2.4 table. The
   element-level indicator is what makes `ε_c ≈ 0` on trusses; if the constraint-level battery
   indicator gives `ε_c ≫ 0`, then the search budget is worth *more* in that domain, not less —
   and the domain-general claim becomes a claim with two data points instead of one.

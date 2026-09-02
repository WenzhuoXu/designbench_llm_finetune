# How much measurement does Φ_v1 vs Φ_v2 need? A power analysis of four instruments

> **Decision this document is for.** ~800 GPU-hours remain of ~1500 (632 GPU-h spent since
> 2026-06-01 by `sacct`). Of the eight T5 arms (`configs/rl/grpo_mt_t5*.yaml`), five have
> held-out evals and show no separation on feasibility; T5bw, T5co and T7 are still training.
> The question is whether to spend the remainder on *more of the same measurement* (bigger or
> repeated multi-turn evals) or on a *different* measurement (fixed-state potential-regret
> probing), and at what n either one becomes conclusive.
>
> **Answer, up front — the headline is that the study has been reading the wrong endpoint.**
>
> 1. **The held-out feasibility rate cannot win this.** It needs **≈ 42 GPU-h per arm** to resolve
>    5 points and **≈ 117 GPU-h per arm** to resolve 3; eight arms at 3 points is **≈ 940 GPU-h**,
>    more than the whole remaining budget, for one scalar. And the effect is almost certainly ≤ 5
>    points: the collected data already bound it at **|Δ feasibility| < 0.087**.
> 2. **The objective endpoint is 8.7× cheaper and is already collected.** Mass ratio over all 34
>    problems, with a pre-registered penalty for failures, has a paired sd of 0.142 and a cost law
>    of `0.0121/Δ²` GPU-h against the rate's `0.1056/Δ²`. Applied to the 15 existing cells it says
>    **|Δ mass ratio(Φ_v2 − Φ_v1)| < 0.043 (95 %), i.e. less than 19 % of the 0.228 the same
>    specification change buys offline** — a quantitative negative result, for **zero** extra GPU.
>    This is where the offline effect lives (median mass ratio 0.894 → 0.602, and 0.707 → 0.422
>    with compound actions) while feasibility moves only 0.738 → 0.762.
> 3. **The fixed-state rank-percentile probe is 45× cheaper than the rate** (band 22–92× on
>    measured generation throughput, constant in the target): 3 feasibility-equivalent points for
>    **2.6 GPU-h per arm**, with a ceiling — set by having only 129 problems, not by GPU — of
>    **1.7–3.6 points for ≈ 4.3 GPU-h per arm, ≈ 34 GPU-h for all eight arms**. It is the only
>    instrument that removes the confound that regret measured along each policy's own trajectory
>    is not comparable across policies.
>
> So the T5 *ordinal* question ("does Φ_v2 make the policy rank its own actions better at matched
> states?") is winnable for **under 35 GPU-h**; the *objective* question is winnable and largely
> answered; the *feasibility* question is not winnable at any effect size below ~5 points. §5 says
> what to claim. The two sampled-reliability runs in flight **cannot** deliver the feasibility
> answer (MDE 0.101 at n = 34 × 4, against arm means 0.049 apart) but the **same GPU-hours read on
> the mass endpoint give MDE 0.034 = 15 % of the offline effect** — so complete them, change the
> metric, and fix the one line at `scripts/eval_checkpoint.py:371` that collapses the objective to
> `min` over samples (§4.6).
>
> Three things to do immediately, all cheap: (1) recompute every existing cell on the imputed
> mass-ratio endpoint (§4.5) — zero GPU, and it is the strongest claim available; (2) turn on the
> in-training probe (§4.4) — **zero GPU**, ~26 CPU-minutes per training run, and it should have
> been on since turn 0; (3) run the ε-greedy link calibration (§7) before committing budget,
> because the probe's cost advantage is `45/s²` in the link slope s and that is the only number
> that could overturn §4.3.

---

## 1. What the current instrument can resolve, measured on its own output

Fifteen evaluation cells exist: five arms × checkpoints 25/50/75, 34 held-out problems each
(`results/eval/t5{a,b,c,ao,bo}*_ckpt{25,50,75}_eval/eval_results.json`). T5bw, T5co and T7 have
no eval cells yet (`logs/t5{bw,co}_*`, `logs/t7_dpo_init_0823`), so the variance decomposition
below is over five arms; adding three more arms at the same n does not change it, because the
limit is per-arm noise, not the number of arms.

| arm | ck25 | ck50 | ck75 | mean | sd across ckpts |
|---|---|---|---|---|---|
| T5a `grpo_mt_t5a_phi1` (Φ_v1 composite) | 0.529 | 0.618 | 0.529 | **0.559** | 0.051 |
| T5b `grpo_mt_t5b_phi2` (Φ_v2 composite) | 0.529 | 0.618 | 0.471 | **0.539** | 0.074 |
| T5c `grpo_mt_t5c_lookahead` | 0.441 | 0.559 | 0.559 | **0.520** | 0.068 |
| T5ao `grpo_mt_t5ao_phionly_v1` | 0.559 | 0.529 | 0.588 | **0.559** | 0.029 |
| T5bo `grpo_mt_t5bo_phionly_v2` | 0.588 | 0.588 | 0.529 | **0.569** | 0.034 |

Pooled within-arm sd across checkpoints **0.0542**; pooled between-arm sd at a fixed checkpoint
**0.0462**. The two contrasts that isolate the specification change disagree in sign:
Φ_v1 → Φ_v2 is −0.020 on the composite reward (T5a → T5b) and +0.010 on the potential-only
reward (T5ao → T5bo).

The decisive statistic is not the rate but the **variance decomposition**. sd of the five arm
means = 0.0175; se of a single arm mean = 0.0542/√3 = 0.0313. The between-arm component is
therefore estimated at **≤ 0** — the arm-to-arm spread is smaller than the noise in one arm's
own mean. se of any pairwise arm difference = 0.0443, so the study's actual claim is

> **|Δ feasibility(Φ_v2 − Φ_v1)| < 0.087 (95%)** — nothing more.

### 1.1 Pairing by problem does not rescue it

All arms see the same 34 problems, so McNemar-style pairing is available. Over the 30
same-checkpoint arm-vs-arm comparisons the mean number of discordant problems is **4.67 / 34**
(p_disc = 0.137). Over the 15 *within-arm* checkpoint-vs-checkpoint comparisons it is
**5.07 / 34** (p_disc = 0.149). **The noise channel is wider than the signal channel**: two
checkpoints of the same run disagree on more problems than two different arms do. Pairing
removes the problem main effect, which was never the difficulty; it does nothing about the
knife-edge divergence.

The clean measurement of that divergence is `results/eval/champ_heldout_greedy` vs
`results/eval/champ_heldout_greedy_batched` — **identical checkpoint, identical problems**, one
sequential and one batched: 0.618 → 0.500, with **6 of 34 problems flipping** (5 lost, 1 gained).
Two runs that should be bit-identical disagree on 17.6 % of problems. Reading that as two draws
of a per-problem Bernoulli, `2·v = 0.176`, so the per-problem outcome variance is

```
v_noise = 0.088          (vs 0.2475 for a fair coin at p = 0.55)
```

i.e. most problems are near-deterministic and a minority are genuine coin flips. This single
number drives every cost in §4 for the outcome-based instruments.

---

## 2. (a) The effect size to look for

### 2.1 What the offline data establishes, and what it does not

The offline ladder is deterministic and the Φ_v1 → Φ_v2 effect in it is enormous
(`results/search_ladder/v5_valid/runs.jsonl`, 130 problems, identical action set, identical
budget, validity-gated):

| policy | feasible | median mass ratio |
|---|---|---|
| `random` | 0.300 (39/130) | 1.023 |
| `lookahead_v1_d1_minmass_legal` | 0.585 (76/130) | 0.894 |
| `greedy_critical` (depth-0 feedback heuristic) | 0.723 (94/130) | 0.875 |
| `lookahead_v2_d1_minmass_legal` | 0.738 (96/130) | 0.602 |
| `greedy_fsd` (engineering baseline) | 0.754 (98/130) | 0.489 |

Paired by problem, Φ_v1 → Φ_v2 is **+20 / −0** discordant (exact two-sided McNemar
**p = 1.9 × 10⁻⁶**), and on the 75 problems both solve, the mass-ratio improvement is
mean −0.228, sd 0.206, **d_z = −1.11**. Widening the action space to compound moves
(`results/search_ladder/v6_macro_valid/runs.jsonl`) preserves the ordering and moves the effect
into the objective: feasibility 0.738 → 0.762 (a 2.3-point step, already below the eval's noise
floor) while median mass ratio goes **0.707 → 0.422**. That is the shape to keep in mind — **the
potential's benefit shows up mostly as objective quality, and feasibility is the least sensitive
channel to read it in.** It is a further argument against making a feasibility rate the primary
endpoint for the T5 comparison. In the group-ordering study
(`results/search_ladder/group_ordering.jsonl`, 264 sampled GRPO groups, 87 of them mixed
feasible/infeasible) Φ_v2 puts a feasible rollout top of a mixed group **85/87 = 0.977** against
Φ_v1's **60/87 = 0.690**, discordance **27 vs 2**.

Those are effect sizes for *the search*, where the potential IS the decision rule. Inside GRPO
the potential is one term in a reward, filtered through a policy gradient, a KL penalty and a
LoRA parameterisation. **Nothing in the offline data licenses a prediction of how much of a
0.15-feasibility search effect survives that filter.** So the effect size has to be built from
the policy side, and it is built in three steps below, with the assumption at each step named.

### 2.2 The per-state distributions, measured at a fixed state set

`results/search_ladder/full/lookahead_diagnostics.jsonl` (1051 states on 127 problems) records
per-state ranking summaries under both potentials, but it does **not** record what a *policy's own
action* would rank — it was collected along the `greedy_critical` trajectory and stores
`crit_best_rank`, the rank of the **best** critical-member action among member candidates
(`scripts/search_ladder.py:331`). Its distribution, over the 856 states where a critical-member
action exists (mean 144 member candidates per state):

| quantity | mean | sd | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| rank_frac of best crit action, Φ_v1 | 0.020 | 0.053 | 0 | 0 | 0.068 | 0.775 |
| rank_frac of best crit action, Φ_v2 | 0.023 | 0.059 | 0 | 0 | 0.075 | 0.819 |
| regret `crit_value_gap`, Φ_v1 | 0.370 | 1.139 | 0 | 0 | 1.332 | 13.65 |
| regret `crit_value_gap`, Φ_v2 | 0.198 | 0.698 | 0 | 0 | 0.411 | 9.04 |
| top-two margin, all candidates, Φ_v1 | 1.201 | 1.450 | 0 | 0.781 | 3.282 | 21.6 |
| top-two margin, all candidates, Φ_v2 | 2.1e4 | 6.7e5 | 0 | 0.097 | 0.726 | **2.2e7** |

Two things to take from it. First, it is an **oracle** quantity — the *best* critical action, not a
policy's action — so mean rank_frac 0.02 with median 0 is far too optimistic to stand in for a
policy, and it cannot be used as the effect-size baseline. Second, look at the last row: Φ_v2's
margin is unbounded above (the objective term is `−f(s)/ref` and `ref` can be small), which is the
first hard argument for a **bounded** statistic — the same argument the eval-side regret tails make
below.

So the distribution was measured directly, with a fixed-state probe built from the shipped
components (`llm_finetune/training/rl/posterior/lookahead_probe.py:75` `probe_state`,
`:188` `truss_candidate_actions`, `:221` `truss_param_bounds`, and
`llm_finetune/training/rl/posterior/potential.py:387` `compute_potential_v2`).
**72 problems × 20 states = 1440 states**, states generated by a seeded policy-independent
random walk over legal in-bounds sizing actions, candidate set = the same 48-action subsample at
every state for every arm (`rng = Random(1000 + depth)`), all 48 simulated one step and scored
with Φ_v2 (α = 5, τ = 0.05). Reproduction script and raw rows:
`…/scratchpad/fixed_state_probe.py`, `…/scratchpad/probe_snapshot.jsonl`.

Five *surrogate policies* were scored at those identical states. They are not LLMs; they are
cheap deterministic rules that bracket the behavioural range an LLM policy occupies, and their
purpose is to supply the dispersion terms that the power calculation needs.

| surrogate action at state s | mean rank_frac | sd | median | mean regret (Φ_v2 units) | sd |
|---|---|---|---|---|---|
| `crit13` — scale the binding-critical member by 1.3 (the depth-0 feedback rule) | **0.047** | 0.125 | 0.021 | 0.144 | 0.327 |
| `crit17` — same member, factor 1.7 | 0.076 | 0.185 | 0.021 | 0.098 | 0.244 |
| `noncrit13` — right parameter, **wrong member** | 0.577 | 0.208 | 0.604 | 0.770 | 0.665 |
| `rand0…3` — uniform legal action | 0.459 (mean of 4) | 0.272 | 0.44 | 0.748 | 0.666 |
| `crit085` — critical member, **wrong direction** | 0.979 | 0.070 | 0.979 | 1.564 | 0.711 |

Three properties matter for what follows.

1. **Dynamic range.** rank_frac spans 0.047 → 0.979 across policies that are all "plausible" —
   a span of 0.93. Held-out feasibility spans 0.44 → 0.62 across the five evaluated T5 cells — a
   span of 0.18. The percentile statistic has roughly **five times the usable range** of the rate
   on the same underlying behaviour, and unlike the rate it is defined at every turn rather than
   once per episode.
2. **Raw regret is the wrong statistic.** In potential units regret is heavy-tailed —
   `mean_regret_vs_procedural` per problem in
   `results/eval/{champ_greedy_minmass,t6_greedy_minmass}/eval_results.json` runs
   median 0.14–0.22 against maxima of 22.2 and 42.9, and Φ_v2's own top-two margin reaches
   2.2 × 10⁷ at one state in the offline diagnostics. A paired-by-problem t-test on mean regret
   between two policies that differ by **35 feasibility points** (champ 0.735 vs t6 0.382 at
   identical settings) gets only **t = 2.09, d_z = 0.42 at n = 25**. Mean regret is dominated by
   a handful of states and is barely more powerful than the binary rate. Use **rank_frac**
   (bounded in [0,1] by construction) or **log1p(regret)**; both are reported below.
3. **Do not gate on `reliable`.** The section-3.3 reliability gate
   (`lookahead_probe.py:127`, `margin > 2γ^(D+1)σ`) passes only **24.9 %** of these states, and
   on that subset the `crit13`→`crit17` contrast collapses from d_z = +0.30 to **d_z = −0.11**.
   The gate selects states with a clear top-two winner — precisely the states where every
   sensible policy agrees. It is the right decision variable for *when to pay for search*; it is
   the wrong filter for *measuring policies*.

### 2.3 From "a policy difference" to "a Δ in mean rank_frac" — the chain, stated explicitly

Two arms trained from the same warmstart for ~100 GRPO steps take the *same* action at most
states. Model that directly with two parameters, both of which the probe measures:

* **q** — the fraction of fixed states at which the two arms emit materially different actions
  (their **disagreement rate**). Nothing else about the arms enters.
* **μ₁** — the mean rank_frac difference *at those disagreeing states* (the **asymmetry**: the
  extent to which one arm's divergent choice is systematically the better-ranked one).

Then the arm-level effect and its paired dispersion are exact:

```
Δ      = q · μ₁
σ_d²   = q · (μ₁² + s₁²) − (q·μ₁)²          s₁ = sd of the per-state difference where they differ
```

Measured on the 1440 fixed states, taking `crit13` as arm A and a uniform legal action as arm B's
divergent choice (the maximum-asymmetry case): **μ₁ = 0.4117, s₁ = 0.2991**. Simulating the
mixture directly reproduces the algebra to within 1 %:

| q (disagreement rate) | Δ = q·μ₁ | σ_d | d_z | n₈₀ (i.i.d. paired) |
|---|---|---|---|---|
| 0.05 | 0.021 | 0.112 | 0.184 | 232 |
| 0.10 | 0.041 | 0.156 | 0.265 | 112 |
| 0.20 | 0.082 | 0.212 | 0.388 | 52 |
| 0.30 | 0.124 | 0.250 | 0.494 | 32 |
| 0.50 | 0.206 | 0.295 | 0.698 | 16 |
| 1.00 | 0.412 | 0.299 | 1.377 | 4 |

Two closed forms fall out and are the load-bearing formulas of this document. With
`Z² = (1.95996 + 0.84162)² = 7.8489`:

```
maximum asymmetry (μ₁ = 0.412):   n₈₀ ≈ Z²(μ₁²+s₁²)/(q·μ₁²) = 12.0 / q            states
general, small asymmetry:          n₈₀ ≈ Z²·q·s₁² / Δ²        = 0.702 · q / Δ²      states
```

The second is the one to use, because it does not require guessing μ₁: it says *the cost of
resolving a rank_frac gap Δ scales with the arms' disagreement rate and the inverse square of
the gap*, and q is the first thing the probe prints.

**The assumption chain, and where it is weakest.**

1. *(measured)* The per-state rank_frac dispersion of plausible policies is s₁ ≈ 0.30, and a
   depth-0-feedback policy sits at 0.047 while a blind one sits at 0.46.
2. *(measured, §2.4)* Mean rank_frac maps to end-of-rollout feasibility with slope ≈ −1.
3. *(assumed)* The T5 arms' behavioural difference is describable by a disagreement rate q with
   a modest asymmetry — i.e. they are the *same* policy perturbed, not two different policies.
   Justified by construction (same warmstart, same 100 steps, same data) and by the ≤ 0
   between-arm variance component in §1, but it is an assumption. **If it is false because the
   arms are literally identical (q → 0), no instrument at any n can separate them, and that is
   itself the finding.**
4. *(assumed)* Fixed states drawn from a policy-independent walk have similar rank_frac
   dispersion to the states the arms actually visit. This is the **weakest link**. The walk
   drifts: median mass ratio 0.724 → 0.903 over 20 steps, and the fraction of states with any
   feasible successor falls to 0.056, against 0.62 on the `greedy_critical` trajectories in
   `lookahead_diagnostics.jsonl`. The direction of the bias is favourable — on-trajectory states
   have *tighter* rank_frac spread (oracle sd 0.059 there vs 0.125 here), so σ_d shrinks and
   power rises — but the fixed state set should be drawn from a reference policy's trajectories
   (the warmstart checkpoint's, frozen once) rather than from a random walk. See §6.

### 2.4 The link: mean rank_frac → feasibility

Ranking is only interesting if it predicts the outcome. Three points are already available on
identical problem sets, with mean rank_frac measured under Φ_v2 on the fixed-state probe and
feasibility from `results/search_ladder/v5_valid/runs.jsonl`:

| policy | mean rank_frac | feasibility |
|---|---|---|
| `lookahead_v2_d1` (the Φ_v2 argmax — rank_frac ≡ 0 by construction) | 0.000 | 0.738 |
| `greedy_critical` ≈ `crit13` | 0.047 | 0.723 |
| `random` | 0.459 | 0.300 |

Chord slope over the full range: **(0.300 − 0.738)/0.459 = −0.954 feasibility per unit
rank_frac** — one point of mean rank_frac ≈ one point of feasibility. The interior point sits
0.03 above the chord, so the true curve is mildly concave, i.e. the slope is *shallower* than −1
near the good end where the arms live.

**This chord is a splice and should be treated as such.** The rank_frac column is measured on the
probe's random-walk states against a 48-action subsample; the feasibility column comes from the
ladder's own rollouts against its ~144-action legal set (`scripts/search_ladder.py:133`
`candidate_actions`). rank_frac is a percentile, so it is comparable across set sizes only if the
sets are drawn the same way, and they are not. The clean version is an ε-greedy Φ_v2 policy run on
**one** action set — the probe's — sweeping ε and recording both mean rank_frac and feasibility
from the same rollouts (`…/scratchpad/eps_calibration.py`, 129 problems × ε ∈ {0, .05, .1, .2, .4,
1} × 3 seeds; a 4-point local variant restricted to ε ≤ 0.2 pins the slope in the region that
matters). Until that lands, the three points above are the best available and the slope should
carry a factor-of-3 uncertainty band in either direction.

**Working slope |s| = 1.0 feasibility per unit rank_frac.** It is the middle of the supported
range, and both directions matter because §4.3 shows the probe's cost advantage is `45/s²`:

* *Shallower — favours the probe.* The local slope between the two best policies is 0.32
  (−0.015 feasibility over +0.047 rank_frac), which would make the probe **~10× better still**
  than the headline (45/0.32² ≈ 435×). This is the region the T5 arms actually live in.
* *Steeper — hurts the probe.* A compounding link, where one bad turn in a 14-turn rollout costs
  the episode, amplifies rank_frac into feasibility by roughly the horizon: `P(solve) =
  0.739·(1−ε)^14` with `rank_frac = 0.047 + 0.412·ε` gives Δfeas 0.05 ⇒ **Δrank_frac ≈ 0.002**,
  i.e. s ≈ 25, which would erase the advantage entirely. That model is refuted at the far end —
  at ε = 1 it predicts 0 % feasible where the random policy actually scores 30 % — but it is not
  refuted near the good end, which is exactly what the ε-greedy sweep would settle.

**All rank_frac ↔ feasibility conversions below use |s| = 1 and should be read as
"feasibility-equivalent", not as feasibility.** The ordinal verdict of §5.2 does not use the link
at all.

**Target effect size.** With |s| = 1, the interesting Δ in mean rank_frac is whatever
corresponds to the feasibility difference worth detecting:

| feasibility difference | equivalent Δ mean rank_frac | comment |
|---|---|---|
| 0.15 | 0.15 | the offline Φ_v1→Φ_v2 search effect; already excluded by §1 (CI ±0.087) |
| 0.087 | 0.087 | the current instrument's detection floor |
| 0.05 | 0.05 | plausible upper end of the surviving in-training effect |
| 0.03 | 0.03 | the honest target: worth a paragraph in a paper |
| 0.01 | 0.01 | below the noise of anything this project can build |

---

## 3. (b) States required at 80 % power, paired by state, with clustering

### 3.1 The test

All arms are scored at an identical state set with an identical candidate set, so the estimator
is the mean over states of the **within-state paired difference** in rank_frac (or in
log1p regret). Two-sided α = 0.05, power 0.80, so `Z² = (z_.975 + z_.80)² = (1.95996 +
0.84162)² = 7.8489`. For an i.i.d. paired design:

```
n₈₀ = Z² · σ_d² / Δ²
```

### 3.2 Intra-problem correlation

States from one problem are correlated, so the effective sample size is
`n_eff = n / DEFF`, `DEFF = 1 + (m̄ − 1)·ρ` with `m̄` states per problem. ρ was estimated by
one-way random-effects ANOVA (variance components from MSB/MSW with the ANOVA average cluster
size) on both datasets:

| quantity | dataset | k problems | N states | m̄ | ρ |
|---|---|---|---|---|---|
| rank_frac, **level**, Φ_v2 oracle | `lookahead_diagnostics.jsonl` | 119 | 856 | 7.15 | **0.105** |
| rank_frac, **level**, Φ_v1 oracle | same | 119 | 856 | 7.15 | 0.134 |
| regret (raw), level | same | 119 | 856 | 7.15 | 0.339 |
| log1p regret, level | same | 119 | 856 | 7.15 | 0.435 |
| `any_candidate_feasible` | same | 127 | 1051 | 8.23 | 0.609 |
| rank_frac **difference** (Φ_v2 − Φ_v1) | same | 119 | 856 | 7.15 | **0.081** |
| rank_frac, level, `crit13` | fixed-state probe | 72 | 1440 | 20 | 0.230 |
| rank_frac, level, `rand0` | fixed-state probe | 72 | 1440 | 20 | 0.000 |
| log1p regret, level, `crit13` | fixed-state probe | 72 | 1440 | 20 | 0.504 |
| rank_frac **difference** `crit13`→`crit17` | fixed-state probe | 72 | 1440 | 20 | **0.270** |
| rank_frac **difference** `crit13`→`noncrit13` | fixed-state probe | 72 | 1440 | 20 | 0.105 |
| rank_frac **difference**, ε-divergence model, ε ∈ {0.1, 0.3, 1.0} | fixed-state probe | 72 | 1440 | 20 | 0.003 / 0.005 / 0.033 |

The pattern is the standard one and it is the reason to pair: **levels** are clustered
(ρ = 0.10–0.50, worst for regret, which inherits each problem's mass scale), but the **paired
difference** is not (ρ_d = 0.00–0.27, and ≈ 0.03 under the divergence model that actually
describes two arms of one training recipe). Working value **ρ_d = 0.10**, sensitivity band
[0.03, 0.30]. Note also that raw regret is clustered *four times* more than rank_frac — another
reason to use the percentile.

### 3.3 States needed

`n₈₀ = Z²·σ_d²/Δ² · (1 + (m−1)ρ_d)`, with σ_d from §2.3 at disagreement rate q, ρ_d = 0.10,
m = 20 states per problem (DEFF = 2.9).

Which σ_d is the honest one matters, so it is stated: the tables below use the **maximum-asymmetry**
σ_d (μ₁ = 0.412 retained), σ_d(q = 0.3) = 0.250. The small-asymmetry form —
σ_d = √q · s₁ = 0.164 at q = 0.3, which is what two arms whose divergent choices are *both*
policy-like would show — gives `n₈₀ = 0.702·q/Δ²`, i.e. **2.3× fewer states** (84 instead of 196
i.i.d. at Δ = 0.05). Every cost in §3.3 and §4 is therefore conservative by roughly that factor;
the verdicts do not depend on which is used.

| Δ (rank_frac) ≈ feas-equiv | q = 0.3 → σ_d = 0.250 | q = 1.0 → σ_d = 0.299 | problems at m = 20 (q = 0.3) |
|---|---|---|---|
| 0.100 | 142 states | 203 | 8 |
| 0.050 | 568 | 813 | 29 |
| 0.030 | 1 579 | 2 259 | 79 |
| 0.020 | 3 553 | 5 083 | 178 |
| 0.010 | 14 211 | 20 331 | 711 |

**Headline answer to (b).** For the honest target — Δ = 0.03 in mean rank_frac, ≈ 3
feasibility-equivalent points — the paired-by-state test needs **1 579 states, spread over 79
problems at 20 states each, i.e. an effective sample size of n_eff = 1 579 / 2.9 = 545**
(and 545 = Z²·0.250²/0.03² exactly, which is the consistency check). For Δ = 0.05 it is
**568 states over 29 problems, n_eff = 196**. Both are inside the benchmark's 129-problem
budget; Δ = 0.02 (3 553 states over 178 problems) is not, and §3.4 is why.

### 3.4 The hard ceiling: 129 problems

The benchmark has 129 usable problems. Adding states per problem buys progressively less because
DEFF grows with m; the ceiling is `n_eff → K/ρ_d = 129/0.10 = 1290`:

| ρ_d | m = 10 | m = 20 | m = 40 |
|---|---|---|---|
| 0.03 | n_eff 1016, Δ_min 0.022 | 1643, **0.017** | 2378, 0.014 |
| 0.10 | 679, 0.027 | 890, **0.024** | 1053, 0.022 |
| 0.20 | 461, 0.033 | 537, **0.030** | 586, 0.029 |
| 0.30 | 349, 0.038 | 385, **0.036** | 406, 0.035 |

(Δ_min at q = 0.3; multiply by 1.20 for q = 1.0. Total states = 129·m, so the m = 20 column is
2 580 states per arm — **4.3 GPU-h per arm** at §4.1's 6 s/state, i.e. the entire column costs less
per arm than two of the 34-problem greedy evals already run.)

**So the fixed-state probe's resolution floor on this benchmark is Δ ≈ 0.017–0.036 in mean
rank_frac — 1.7 to 3.6 feasibility-equivalent points — at 2 580 states per arm.** Below that the
answer is "buy more problems", not "buy more GPU".

---

## 4. (c) Cost per unit power

### 4.1 Measured GPU costs (from `sacct`, this project's own jobs)

| job kind | what it is | elapsed on 1×H100 | generations | s / generation |
|---|---|---|---|---|
| `abl_v2_eval` ×17 | 34-problem greedy multi-turn eval | 2.1–3.1 h, mean ≈ 2.75 h | 34 × 14.5 ≈ 490 | **19.1** |
| `eval_k1` 44242371 | `eval_llm_search` K=1 + procedural probe | 3.96 h | ≈ 440 | 32.4 |
| `eval_t6k1` 44248119 | same, other checkpoint | 1.74 h | ≈ 530 | 11.8 |
| `eval_t6k8` 44248120 | K=8, batched `num_return_sequences` | 4.41 h | 5 440 | **2.92** |
| `eval_k8` 44245749 | K=8, champion (longer completions) | 17.7 h | 5 440 | 11.7 |

One more datum is decisive for the probe's cost, and it is a **cross-prompt** batching
measurement rather than a same-prompt one: `abl_v2_eval` job **44280577**
(`results/eval/champ_heldout_greedy_batched`, `--batch-size 8` over 8 problems at once) took
**1 h 23 m** for 34 × 14.09 ≈ 479 generations = **10.4 s/generation**, against 20.3 s for the
sequential run of the same checkpoint. So batching 8 *different* prompts through the multi-turn
harness gives only **2.0×**, while 8 samples from *one* prompt gives 6.5× (2.92 s). The truth for a
fixed-state probe is between the two and closer to the better end, because a probe has **no
sequential dependency at all** — 2 580 independent prompts, no synchronised turns, no stragglers
draining the batch — so the batch can stay full and go wider than 8.

Working figures below: **0.0765 GPU-h per problem-rollout** (2.6 h / 34) for the outcome
instruments and **6 s = 1.67 × 10⁻³ GPU-h per state** for the probe — the midpoint of the measured
2.9–10.4 s band, with 3 s and 12 s carried as the sensitivity ends. Total spend to date, for
calibration: **632 GPU-h over 107 GPU jobs since 2026-06-01**.

The probe's simulator cost is **not** GPU cost. 48 candidates × ~8 ms = 0.4 s of CPU per state,
which is why the probe belongs in two pieces: generate the actions on the GPU node, score them on
RM-shared. Run inline it adds ~10 % to the GPU figure.

### 4.2 The two cost laws

Both instruments have MDE ∝ n^(−1/2) and cost ∝ n, so cost ∝ 1/Δ² for both and their **ratio is
a constant independent of the target**:

```
outcome rate, paired by problem:   C_arm = 0.0765 · Z²·2·v_noise / Δ²  = 0.1056 / Δ²   GPU-h
fixed-state rank_frac probe:       C_arm = 1.67e-3 · Z²·σ_d²·DEFF / Δ² = 2.37e-3 / Δ²  GPU-h
                                                                  (q = 0.3, ρ_d = 0.10, m = 20)
```

### 4.3 GPU-hours for 80 % power, per arm

| target (feasibility-equivalent) | outcome rate: GPU-h/arm | (problem-rollouts) | fixed-state probe: GPU-h/arm | (states) | ratio |
|---|---|---|---|---|---|
| 0.087 — today's floor | 14.0 | 183 | 0.31 | 188 | **45×** |
| 0.050 | 42.3 | 553 | 0.95 | 568 | 45× |
| 0.030 | 117.4 | 1 535 | 2.63 | 1 579 | 45× |
| 0.024 — probe's ceiling at ρ_d = 0.10 | 183.4 | 2 398 | 4.11 | 2 467 | 45× |
| 0.020 | 264.1 | 3 454 | 5.92 | 3 553 | 45× |
| 0.010 | 1 056 | 13 814 | 23.7 | 14 211 | 45× |

The ratio is **31× at q = 1.0**, **45× at q = 0.3**, **115× at q = 0.1** — the probe gains *more*
the more similar the arms are, which is the regime that matters here. Under the measured
generation-cost band it is **26× at 10.4 s/state** (the batched multi-turn harness as it stands)
to **92× at 2.9 s/state** (full-batch decode).

**The one thing that could kill the advantage is the link slope, and it must be stated.** The
table's left column is a feasibility target; the probe pays in rank_frac units, so with
`Δrf = Δfeas / |s|`:

```
ratio = 44.6 / s²          s = |d feasibility / d mean rank_frac|
```

At the measured chord slope s = 0.95 the ratio is 45×; at the local slope near the good end
implied by the two best offline policies (s ≈ 0.32) it is 435×; but at s = 3 it is only 5×, and at
s ≈ 24 — the amplification a strict "one bad turn loses the 14-turn episode" model gives (§2.4) —
the probe's advantage disappears entirely (0.08×). **The probe is 45× better because the empirical link is
close to linear, not because percentile statistics are magic.** §2.4 gives the three points that
support s ≈ 0.3–1.0 and the reason the compounding model is refuted at the far end; the ε-greedy
sweep in §7 is the measurement that pins s near the good end, and it is the single number most
worth having before committing budget. Note also that the *ordinal* claim of §5.2 needs no link
at all — the link is required only to translate the probe's result into feasibility language.

The 45× decomposes exactly, and the decomposition is worth seeing because it says the gain is
**cost per measurement**, not statistical cleverness:

```
ratio = (cost per problem-rollout / cost per state) × (2·v_noise / (σ_d²·DEFF))
      = (0.0765 / 0.001667)                          × (0.176 / (0.0625 × 2.9))
      =  45.9                                        ×  0.972
      =  44.6
```

The second factor is ≈ 1: measured in its own units the probe's per-observation noise
(σ_d²·DEFF/2 = 0.091) happens to be almost identical to the outcome instrument's
(v_noise = 0.088). **All of the gain is the first factor** — one measured decision costs one
batched generation (6 s) instead of a whole 14.5-turn sequential rollout (277 s): 14.5× fewer
generations per measurement × 3.2× cheaper per generation from batching. That is why the probe
wins by the same factor at every target, and why the factor degrades to 26× if the generations
cost the 10.4 s the current batched harness delivers rather than the 6 s a dependency-free probe
should reach.

Note what is **not** claimed: the generation is not shorter. The policy thinks ~1 100 tokens
before emitting an action and truncating that would change the action distribution being measured,
so the per-generation cost is the eval's cost divided only by batching. The 14.5× comes entirely
from the fact that a fixed-state probe measures a decision per generation whereas a rollout
measures an outcome per 14.5 generations.

### 4.4 A fourth instrument that costs no GPU at all

Worth stating because it is free and was available from turn 0. The GRPO loop already generates
every rollout; `probe_state()` at each *visited* state adds **48 FEA calls of CPU and zero
generation**. At group size 8 × 5 turns × 100 steps ≈ 4 000 states per run, that is
4 000 × 48 × 8 ms ≈ **26 CPU-minutes** spread over a multi-hour training job — free, and it
yields a per-step curve of mean rank_frac / regret with n = 4 000 rather than one number at the
end.

Its limitation is exactly the confound this document is about: those are the policy's *own*
states, so the series is a **within-run monitor** ("is this arm internalising the lookahead as it
trains?"), not a cross-arm comparator. That is still strictly more than `rho_tree_agreement`
currently provides — which is an algebraic identity when the reward is monotone in Φ_H and a 1/K
coin flip otherwise (`lookahead_probe.py:6-14`, on `grpo_trainer.py:392` `_compute_rho_per_group`, logged at `:636`), i.e. it cannot rise. Turning it on for the
remaining training runs costs nothing and should not wait for any budget decision.

### 4.5 The endpoint nobody costed: mass ratio, which is ~9× cheaper and already collected

§2.1 noted that the potential's offline benefit shows up mostly in the **objective**, not in the
feasibility rate. That has a direct measurement consequence, because a continuous objective with a
moderate coefficient of variation is a far better statistic than a binary rate — and every T5 cell
already stores `mass_ratio` per problem.

Define the endpoint on **all 34 problems** with a pre-registered penalty for failure
(`mass_ratio = 1.2` for any problem not solved, above every feasible value observed), so the
estimand does not depend on which problems an arm happens to solve. Measured on the 30
same-checkpoint arm-vs-arm comparisons and the 15 within-arm checkpoint comparisons:

| | paired sd of the per-problem difference | MDE at n = 34, R = 1 | cost law |
|---|---|---|---|
| feasibility rate | (binary, v_noise = 0.088) | 0.202 | 0.1056 / Δ² |
| **mass ratio, imputed at 1.2** | **0.142** (within-arm noise 0.143) | **0.068** | **0.0121 / Δ²** |
| mass ratio, imputed at 1.5 | 0.235 (noise 0.240) | 0.113 | 0.0331 / Δ² |
| mass ratio, jointly-feasible only | 0.112 over n̄ = 16.3 of 34 (48 %) | 0.078 | 0.0157 / Δ² |

**The imputed mass endpoint is 8.7× more powerful per GPU-hour than the feasibility rate**
(0.1056 / 0.0121). The jointly-feasible variant is 6.7× — worse than the imputed one *and*
selection-biased, because the conditioning set depends on the arms; use the imputation, and state
the penalty in advance. As with the binary rate, the paired sd (0.142) equals the within-arm
noise sd (0.143), so the dispersion is knife-edge divergence and repeats average it down.

Referenced to the offline effect this endpoint has real reach. The offline Φ_v1 → Φ_v2 mass-ratio
improvement is **−0.228** (paired, n = 75, d_z = −1.11, §2.1):

| fraction of the offline mass effect surviving into training | Δ | GPU-h/arm |
|---|---|---|
| 100 % | 0.228 | 0.5 |
| 50 % | 0.114 | 2.1 |
| 25 % | 0.057 | 8.4 |
| 10 % | 0.023 | 52.7 |

**And the existing 15 cells already answer it.** Computing the imputed endpoint on the data in
hand, Δ = (Φ_v2 − Φ_v1), negative meaning Φ_v2 produces lighter designs:

| contrast | ck25 | ck50 | ck75 | mean |
|---|---|---|---|---|
| composite, T5a → T5b | +0.0183 (t = +0.92) | +0.0115 (t = +0.46) | −0.0001 (t = −0.01) | **+0.0099** |
| potential-only, T5ao → T5bo | −0.0028 (t = −0.14) | −0.0408 (t = −1.84) | +0.0211 (t = +1.13) | **−0.0075** |

Every estimate is inside ±0.042 of zero and the two contrasts still disagree in sign. Taking the
single-checkpoint se of 0.022 as the conservative one (the three checkpoints are not independent),
the study can already state:

> **|Δ mass ratio(Φ_v2 − Φ_v1)| < 0.043 (95 %) — less than 19 % of the 0.228 the same
> specification change buys in the offline search.**

That is a *quantitative* negative result on the objective, an order of magnitude tighter in
effect-fraction terms than the feasibility CI of §1, and it costs zero additional GPU-hours. It is
the number §5 should lead with.

### 4.6 Where the in-flight runs land

| plan | n × R | MDE | GPU-h/arm |
|---|---|---|---|
| current 34-problem greedy, 1 rollout | 34 | **0.202** | 2.6 |
| 34-problem, 4 samples at T = 0.6 (**in flight**) | 136 | **0.101** | 10.4 |
| all 129 problems, 1 greedy rollout | 129 | 0.104 | 9.9 |
| all 129 problems, 4 rollouts | 516 | 0.052 | 39.5 |
| 34 problems, 16 rollouts | 544 | 0.050 | 41.6 |

**The two sampled reliability runs now in flight will not separate the arms.** At n = 34, R = 4
their MDE is 0.101 against a target of ≤ 0.05, and that is the *optimistic* figure: it assumes
v_noise = 0.088, measured from batching non-determinism on a greedy decode. Temperature-0.6
sampling has more per-draw variance, and at v = 0.15 the MDE is 0.132, at the fair-coin bound
v = 0.2475 it is 0.169 — **worse than the greedy eval they are meant to replace**. Sampling
changes the estimand to something better defined (expected solve rate at the training
temperature) but it does not buy resolution per GPU-hour; only n × R does.

**But do not cancel them — re-read them.** The same 10.4 GPU-h per arm, scored on the imputed
mass-ratio endpoint of §4.5, reaches `MDE = Z·√(2·(0.142²/2)/(34·4)) = 0.034` — **15 % of the
offline 0.228**, four times finer than anything the feasibility rate can deliver at that price. The
runs are worth completing; only the metric they were launched to produce is not.

One code change is needed and it is one line. `_eval_problems_sampled`
(`scripts/eval_checkpoint.py:308`) correctly averages the binary outcome into `solve_rate`
(`:361`) but collapses the objective to `mass_ratio = min(masses)` (`:371`, and `final_mass` the
same way at `:369`) — a **best@n** statistic, which (a) improves with n so it is not comparable
across cells run at different n, and (b) throws away the per-sample spread the mass estimator
needs. Recording the mean of the *imputed* per-sample mass ratios alongside it converts these runs
from underpowered to decisive at no extra GPU cost.

Also note that the outcome instrument is not free of the 129-problem ceiling either: reaching
MDE 0.03 needs 1 535 problem-rollouts, i.e. 12 independent rollouts of every problem in the
benchmark, per arm.

---

## 5. (d) Is the T5 comparison winnable?

### 5.0 Split the outcome question in two

"Does Φ_v2 make a better policy?" has two endpoints and they have completely different answers.

* **On the feasibility rate** — not winnable (§5.1).
* **On the objective (mass ratio, imputed) — winnable, and effectively already won** (§5.1a). The
  answer is negative and quantitative: less than 19 % of the offline objective gain survives, for
  zero additional GPU-hours (§4.5). This is the endpoint the offline result actually moves
  (0.894 → 0.602 median mass ratio, and 0.707 → 0.422 with compound actions) and it is the one the
  study should have been reading all along.

### 5.1 The feasibility rate: not winnable

"Does Φ_v2 raise held-out **feasibility** relative to Φ_v1 inside GRPO?" — **not winnable with
~800 GPU-h**, for three independent reasons.

1. **The effect is small or zero.** §1's variance decomposition puts the between-arm component
   at ≤ 0 and the 95 % interval on the contrast at ±0.087, and the two contrasts that isolate the
   specification change **disagree in sign** (−0.020 composite, +0.010 potential-only). There is
   no prior here favouring a positive effect; the honest prior is centred on zero.
2. **The cost is prohibitive at the plausible effect size.** Resolving 0.03 costs 117 GPU-h per
   arm. The minimum interesting comparison is not two arms but four (T5a/T5b for the composite
   contrast, T5ao/T5bo for the ordering-only contrast), i.e. **470 GPU-h** — and that buys an
   answer only if the true effect is ≥ 0.03. If it is 0.015, the requirement quadruples to
   ~1 900 GPU-h.
3. **A confound is not being paid for.** Even at 470 GPU-h the comparison is 4 arms × 1 seed. The
   within-arm sd across checkpoints of the *same* run is 0.054 — as large as the entire
   between-arm spread — so a single training seed per arm cannot attribute a difference to the
   potential rather than to the seed. Buying seeds multiplies the bill again.

Do not spend the remaining budget here.

### 5.1a The objective: winnable, and the answer is already in hand

The same three reasons do not apply to the imputed mass-ratio endpoint, because its cost law is
8.7× kinder (§4.5). At the 34 problems and 3 checkpoints already collected, every Φ_v1 → Φ_v2
estimate is inside ±0.042 of zero with the two contrasts disagreeing in sign, giving
**|Δ| < 0.043 (95 %) < 19 % of the offline 0.228**. If a tighter bound is wanted, the price is
modest and known: **8.4 GPU-h per arm** to resolve 25 % of the offline effect, **52.7 GPU-h per
arm** to resolve 10 %. Spending ~35 GPU-h to take the four key arms (T5a/T5b, T5ao/T5bo) to the
25 % bound is the only outcome-side expenditure this analysis recommends, and it should be
conditioned on the probe first showing that the arms differ at all (§5.2).

Two conditions on it: the penalty value must be fixed **before** the runs (1.2 here — sensitivity
to 1.5 is tabulated in §4.5 and costs 2.7× the GPU), and the jointly-feasible-only variant must
not be used, because its conditioning set depends on the arms being compared.

### 5.2 The ordinal question: yes, and cheaply

"At an identical, fixed set of states, does the Φ_v2-trained policy rank its own chosen action
higher in the Φ_v2 order than the Φ_v1-trained policy does?" — **winnable**, at

* **n = 2 580 states per arm** (129 problems × 20 fixed states), **4.3 GPU-h per arm**,
  **≈ 34 GPU-h for all eight arms**,
* resolving Δ mean rank_frac down to **0.024** at ρ_d = 0.10 (0.017 at ρ_d = 0.03, 0.036 at
  ρ_d = 0.30) — **1.7 to 3.6 feasibility-equivalent points**, i.e. 2.4–5× finer than the outcome
  instrument's 0.087 floor, for 1/40 of the money,
* with the confound the context flags **removed by construction**: every arm is scored at the
  same states with the same candidate enumeration and the same subsample seed, so "a policy that
  blunders into easy states shows low regret" cannot happen.

A cheaper pre-check comes free and should run first: **the pairwise action-agreement rate q at
those fixed states**, which needs the generations but none of the potential machinery. It decides
the whole question. If q < 0.05 the arms are the same policy and §5.3 is the write-up. If
q ≳ 0.2, then by §2.3 a 0.05 gap needs only ~570 states and the answer arrives in ~1 GPU-h/arm.

### 5.3 Two honesty conditions on the ordinal result

* **Circularity.** Scoring with Φ_v2 the arms that were *trained* on Φ_v2 is not a fair test of
  "Φ_v2 is the better specification"; it is a test of "training on Φ_v2 installs the Φ_v2 order",
  which is a real but weaker claim. The non-circular endpoint is available and costs only CPU:
  score the same fixed states and the same policy actions by **solve-from-here** — run the
  offline Φ_v2 ladder to completion from each candidate successor and record whether it reaches
  feasibility and at what mass. That criterion is neither arm's training signal. Report the
  Φ_v2-rank result as the mechanism and the solve-from-here result as the outcome.
* **Direction of the claim.** Mean rank_frac is a *feasibility-equivalent* under a linear link
  measured over three points (§2.4). Do not convert it to a feasibility number in a headline.
  Report the rank_frac difference with its CI, and report the calibration slope separately.

### 5.4 What is supportable with the evidence already in hand

Independently of any new run, these claims are already carried by collected data:

1. **The potential's specification determines search quality, decisively.** Identical action set,
   identical budget, deterministic: 0.585 → 0.738 feasible, +20/−0 paired
   (**McNemar p = 1.9 × 10⁻⁶**), median mass ratio 0.894 → 0.602, and on the 75 commonly-solved
   problems d_z = −1.11. (`results/search_ladder/v5_valid/runs.jsonl`)
2. **It determines the ranking GRPO actually consumes.** On 87 mixed feasible/infeasible sampled
   groups, Φ_v2 puts a feasible rollout first 97.7 % of the time against Φ_v1's 69.0 %,
   discordance 27 vs 2. (`results/search_ladder/group_ordering.jsonl`) This is the *credit
   assignment* claim, and it is measured on real rollout groups, not on search.
3. **It generalises across domains.** The battery/PyBaMM ladder, 24 problems, identical machinery
   (`results/battery_ladder/full_scales-none/runs.jsonl`, `…/gated_offset5/runs.jsonl`):
   `random` 0.083 → `greedy_critical` (depth-0 feedback heuristic) 0.250 → `lookahead_v1_d1`
   0.625 → `lookahead_v2_d1` 0.917 → **1.000** with the problem-scaled feasibility offset. Note
   the honest caveat for a power document: at n = 24 the Φ_v1 → Φ_v2 step is 9 vs 2 discordant,
   **exact p = 0.065** — suggestive, not significant, and it should be reported that way.
   Φ_v2 vs the feedback heuristic is 16 vs 0, **p = 3.1 × 10⁻⁵**.
4. **On the objective, the in-training null is quantified, not merely unresolved.**
   Recomputing the 15 existing cells on the imputed mass-ratio endpoint (§4.5) gives
   **|Δ mass ratio(Φ_v2 − Φ_v1)| < 0.043 (95 %), less than 19 % of the offline 0.228**, with the
   composite and potential-only contrasts disagreeing in sign (+0.010, −0.008). That is a real
   finding — *the offline objective gain does not transfer through GRPO at this scale* — and it is
   free. It is stronger and more interesting than "no separation on feasibility".
5. **The feasibility negative result is a measurement result, not a null result, and it is
   quantified.**
   Within-arm sd across checkpoints 0.054 vs between-arm sd 0.046; a same-checkpoint
   sequential-vs-batched rerun moves feasibility 0.618 → 0.500 with 6/34 problems flipping;
   within-arm checkpoint-to-checkpoint discordance (5.07/34) exceeds between-arm discordance
   (4.67/34). **The single-rollout held-out feasibility rate is not a usable ranking statistic at
   this n**, and §4.3 gives the price of making it one.

That is a complete and defensible story: the potential is established offline (1) and on the
group-ordering that GRPO actually consumes (2), it generalises across domains (3), the in-training
objective transfer is bounded at < 19 % of the offline effect (4), and the feasibility comparison is
reported as underpowered *with the power analysis that shows what it would have cost* (5). Claim 2
is the bridge that makes this more than a search paper, and claim 4 is the honest limit statement —
both are already paid for.

---

## 6. Instrument specification

What exists, and the three gaps.

| piece | status |
|---|---|
| `probe_state()` — enumerate, simulate, score, locate the policy action, emit rho / regret / rank_frac / margin / sigma / reliable | **exists**, `llm_finetune/training/rl/posterior/lookahead_probe.py:75` |
| `counterfactual_advantage()` — Φ(s′) − E_c[Φ(f(s,c))] | exists, `lookahead_probe.py:144` |
| candidate enumerator + bounds, subsample-stable by construction | exists, `lookahead_probe.py:188`, `:221` |
| regret along the policy's own rollout | exists, `scripts/eval_llm_search.py:135-172`, `--procedural-probe` |
| **fixed** state set shared across arms | **missing** — `eval_llm_search.py` probes states the policy itself reached, which is exactly the confound |
| `rank_frac` against the *procedural* candidate set | **missing** — `eval_llm_search.py:134` computes rank only among the policy's own K samples, so it is identically 0 at K = 1 (see the four `mean_rank_frac_of_first_sample = 0.0` cells in `results/eval/*_k1/eval_results.json`). `probe_state()` already computes the right one; it is simply not wired into the evaluator |
| per-state rows written out | **missing** — only per-problem means are stored, which discards the n that makes this instrument work |

Design that follows from the numbers above:

1. **State set.** Freeze one set: 129 problems × 20 states = 2 580, taken from the *warmstart*
   checkpoint's greedy trajectories (not a random walk — §2.3 assumption 4, and not any T5 arm's
   trajectories, or the confound returns). Serialise the truss state, the goals, and the
   candidate-subsample seed. m = 20 is the sweet spot: §3.4 shows doubling to m = 40 buys
   +18 % effective n at ρ_d = 0.10 (+5 % at ρ_d = 0.30, +45 % at ρ_d = 0.03) for 2× the GPU, so
   go to m = 40 only if the first run measures ρ_d below ~0.05.
2. **Candidate set.** `truss_candidate_actions(..., max_candidates=48, rng=Random(1000+depth))`
   — identical for every arm. 48 is enough: rank_frac is a percentile, and its resolution
   (1/48 ≈ 0.021) is already at the ceiling Δ_min of §3.4.
3. **One generation per state**, batched ≥ 8, at the training temperature; record the parsed
   action, whether it parsed, and the successor. An unparsed or invalid action scores at the worst
   candidate (`lookahead_probe.py:119-121` already does this) — that is the right convention, because
   forfeiting a turn is a real cost and grammar success differs across arms (0.49–0.88 across the
   cells inspected).
4. **Write one row per state**, not per problem: `problem_id, depth, arm, action, parsed,
   rank_frac, regret, log1p_regret, margin, sigma, reliable, solve_from_here`.
5. **Analysis.** Paired-by-state difference in rank_frac, with problem-clustered standard errors
   (cluster on `problem_id`, 129 clusters — enough for a cluster-robust CI). Report q, the
   agreement rate, alongside. Secondary endpoint log1p(regret), never raw regret (§2.2).
6. **Scoring is CPU.** Generate on GPU-shared; score the 48 candidates and the solve-from-here
   criterion on RM-shared with `--qos=low`. 2 580 states × 48 × 8 ms ≈ 17 core-minutes for the
   ranking, plus the ladder rollouts for solve-from-here.

**Suggested spend of the remaining ~800 GPU-h**, in priority order.

| item | GPU-h |
|---|---|
| Recompute all 15 existing cells on the imputed mass-ratio endpoint (§4.5) | **0** |
| Turn on the in-training probe (§4.4) for every future run | **0** |
| Fixed-state probe, 8 arms × 2 580 states, at the resolution ceiling (§5.2) | 34 |
| Same probe on the warmstart and the champion, as anchors on the rank_frac scale | 9 |
| ε-greedy link calibration (§2.4) — CPU only, pins the `45/s²` factor | 0 (≈ 1 core-hour) |
| Imputed mass ratio to the 25 %-of-offline bound, 4 key arms (§5.1a) | 34 |
| One additional training seed for T5a and T5b (attribution, not power) | ~2 × training cost |
| **total measurement** | **≈ 77 + seeds** |

Note what is *not* on the list: any further held-out **feasibility** evaluation. That leaves the
large majority of the budget for training changes rather than for measuring the ones already run —
which is the right allocation, because §5.1 says the arms as run do not contain a
feasibility-resolvable effect at any affordable n, and §5.1a says their objective effect is already
bounded below 19 % of the offline one.

---

## 7. Provenance — every number above

**Constants.** `z_.975 = 1.959964`, `z_.80 = 0.841621`, `Z² = 7.848858`. Paired continuous test:
`n₈₀ = Z²σ_d²/Δ²`. Clustered: `× (1 + (m̄−1)ρ)`. Paired binary (McNemar, small Δ):
`n₈₀ = Z²·p_disc/Δ²`, and with per-problem outcome variance v under repeats,
`Var(D) = 2v/(nR)` so `MDE = Z√(2v/(nR))`.

**Data read (all pre-existing, read-only).**

| file | used for |
|---|---|
| `results/eval/t5{a,b,c,ao,bo}*_ckpt{25,50,75}_eval/eval_results.json` | §1 arm table, within/between sd, McNemar discordance; §4.5 imputed mass-ratio endpoint (paired sd 0.142, contrasts +0.0099 / −0.0075) |
| `results/eval/champ_heldout_greedy{,_batched}/eval_results.json` | §1.1 `v_noise = 0.088` (6/34 flips, same checkpoint) |
| `results/eval/champ_batch{1,8}/eval_results.json` | 12-problem batching effect 0.667 → 0.500 |
| `results/eval/{champ_greedy_minmass,t6_greedy_minmass,champ_heldout_k1,t6_distill_k1}/eval_results.json` | §2.2 regret heavy tails; paired t = 2.09, d_z = 0.42 at n = 25 |
| `results/eval/{champ_heldout_k8,t6_distill_k8}/eval_results.json` | within-K rank_frac 0.216 / 0.202, sd of problem means 0.097 / 0.119 |
| `results/search_ladder/v5_valid/runs.jsonl` | §2.1 ladder table, McNemar 20/0, mass d_z = −1.11 |
| `results/search_ladder/v6_macro_valid/runs.jsonl` | compound-action arm (0.738 → 0.762, mass 0.707 → 0.422) |
| `results/search_ladder/full/lookahead_diagnostics.jsonl` | §2.2 oracle rank_frac, §3.2 ρ for levels and differences |
| `results/search_ladder/group_ordering.jsonl` | §5.4 claim 2: 60/87 vs 85/87 on mixed groups, 27 vs 2 |
| `results/battery_ladder/{full_scales-none,gated_offset5}/runs.jsonl` | §5.4 claim 3: 24 battery problems, 0.083 → 0.250 → 0.625 → 0.917 → 1.000, McNemar 9 vs 2 (p = 0.065) |
| `logs/slurm/abl_v2_eval_44280577.out` + `sacct` | §4.1 cross-prompt batch-8 throughput: 1 h 23 m for 34 problems = 10.4 s/generation |
| `sacct -X -u wxu7 -S 2026-06-01` | §4.1 elapsed times; 632 GPU-h to date over 107 GPU jobs |

**Data generated for this analysis** (CPU only, no model loaded — no GPU was used and nothing in
the repository was modified except this document). The two scripts live in an **ephemeral session
scratchpad**, so the reproducible artifact is the specification, not the path: §2.2 gives the state
generator, the candidate enumerator and its seeds, and the surrogate rules; §6 gives the same for
the real instrument. Both run in ~15 min on 24 cores against `DesignBench/data/problems` and
`llm_finetune/training/rl/posterior/{potential,lookahead_probe}.py`.

* `fixed_state_probe.py` → `probe_snapshot.jsonl`: 72 problems × 20 states = **1440** fixed states,
  48 candidates each simulated and scored under both Φ_v1 and Φ_v2, five surrogate policies
  located in that order. Source of §2.2's table, §2.3's `μ₁ = 0.4117 / s₁ = 0.2991`, §3.2's
  probe-side ρ, and the `reliable`-gate finding.
* `eps_calibration.py`: ε-greedy Φ_v2 policy over the identical 48-action candidate set,
  129 problems × ε ∈ {0, .05, .1, .2, .4, 1} × 3 seeds, to replace §2.4's three-point chord with a
  six-point measured curve of feasibility vs mean rank_frac, and to pin the **local** slope s in
  the good-policy region. This is the one number that can move the §4.3 verdict: the probe's
  advantage is `45/s²`, so s ≈ 1 gives 45×, s ≈ 3 gives 5×, s ≳ 7 gives none. The three points
  available now support s ≈ 0.3–1.0 (chord 0.954, local 0.32), and the compounding model that
  would give s ≈ 24 is refuted at the far end — it predicts 0 % feasible for the random policy,
  which actually scores 30 %. The ordinal verdict of §5.2 does not depend on this sweep; the
  feasibility-equivalent translations in §2.4 and §4.3 do.

**Known limitations.**

1. The surrogate policies are rules, not LLMs. They supply σ_d and s₁, i.e. *dispersion*, which
   is a property of the state and candidate set more than of the policy — but they cannot supply
   q, the arms' disagreement rate, and q is the one free parameter left in the answer. It costs
   one probe run to measure.
2. The fixed states come from a random walk, which drifts off the region policies occupy (§2.3
   assumption 4). The bias is favourable but the real instrument should use warmstart trajectories.
3. `v_noise = 0.088` is inferred from one same-checkpoint sequential-vs-batched pair (6/34). It is
   corroborated by within-arm checkpoint discordance (5.07/34 ⇒ v ≤ 0.075) but it is one
   measurement, and it may understate the variance of temperature-0.6 sampling — §4.6 gives the
   sensitivity, and the conclusion (the in-flight 4-sample runs are underpowered) is *strengthened*
   if v is larger.
4. ρ_d is estimated from surrogate contrasts, not from two real arms, and ranges 0.00–0.27 across
   contrasts. §3.4 tabulates the whole range; the verdict in §5.2 holds at every value in it.

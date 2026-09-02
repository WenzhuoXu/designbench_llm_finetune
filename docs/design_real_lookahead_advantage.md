# Real per-step lookahead advantages in multi-turn GRPO

**Date:** 2026-08-23 · **Status:** proposal (nothing applied) · **Scope:** the experiment that
actually tests the innovation, plus the patch plan to run it.

The framework's claim is that an action's value on a design state can be estimated by *search*
(`V̂^(D,b)`), and that training on that estimate beats training on outcomes. No run has ever
tested it. `use_tree_expansion: true` never calls `evaluate_tree`; it computes a group-relative
**terminal** potential (`grpo_trainer.py:278` `_compute_group_tree_advantages`) which TRL's
within-group mean-centring collapses onto the reward that was already there. This document
specifies the estimator that does not collapse, where its candidate actions come from, how it
reaches the gradient through TRL 1.0.0 without a fork, what it costs, and the exact edits.

Everything below is domain-general by construction: the only per-domain objects are a
`DesignProgram` (already recovered from a problem's own goals by
`posterior/potential.py:280 program_from_goals`) and a **candidate enumerator**. Truss and battery
adapters for both are named in §7.

---

## 1. Notation

For one GRPO group of `K = 8` rollouts on problem `p`:

| symbol | meaning |
|---|---|
| `s_t^k` | state visited by rollout `k` at turn `t`; `s_0` shared by the whole group |
| `a_t^k` | the action the policy emitted at `s_t^k` |
| `f(s,a)` | the simulator's successor state (deterministic; `envs/truss_env.py:375,383`) |
| `Φ(s)` | design-program potential, `potential.py:337 compute_potential_v2` = `−w·f̂(s) − α·Σ_c hinge(ĝ_c(s))` |
| `C(s)` | candidate action set at `s`, `\|C(s)\| = b` |
| `Φ_H^k` | `Φ(s_{H_k}^k)`, terminal potential |
| `R_k` | the scalar reward TRL receives for rollout `k` |

TRL 1.0.0 defaults that matter (verified in the installed source): `scale_rewards="group"`
(`grpo_config.py:697`), `loss_type="dapo"` (`:709`), `importance_sampling_level="token"`
(`:668`), `num_iterations=1` (`:597`), `steps_per_generation=gradient_accumulation_steps` (`:428`).
With `per_device_train_batch_size=1, gradient_accumulation_steps=8, group_size=8`
(`configs/rl/grpo_base.yaml`), the generation batch per rank is **exactly one GRPO group**, so
group statistics are rank-local.

---

## 2. (b) The estimator, and the proof that GRPO cannot cancel it

### 2.1 Depth-1

```
V̂^(1)(s, a)  = Φ(f(s, a))
b̂(s)         = (1/b) Σ_{c∈C(s)} Φ(f(s, c))          ← counterfactual baseline, a function of s ONLY
σ̂(s)         = std_{c∈C(s)} Φ(f(s, c))
Â_t^k        = [ γ·Φ(s_{t+1}^k) − b̂(s_t^k) ] / (σ̂(s_t^k) + ε)
```

`Â_t^k` reads: *how far did this action beat the mean of what was achievable from the state it was
taken in, measured in units of how much the choice at that state mattered at all.* The σ̂ division
is not cosmetic — it is what makes the quantity dimensionless and therefore comparable across
domains without retuning κ.

### 2.2 Depth D ≥ 2

Replace `Φ` by the depth-`d` search value, itself a pure state function:

```
M_0(s) = Φ(s)
M_d(s) = max_{c ∈ C(s)} γ · M_{d−1}( f(s, c) )
Â_t^k  = [ γ·M_{D−1}(s_{t+1}^k) − (1/b) Σ_c M_{D−1}(f(s_t^k, c)) ] / (σ̂_D(s_t^k) + ε)
```

`M_{D−1}` is exactly what `posterior/tree_expansion.py:127 evaluate_tree` computes when its
`continuation_fn` returns `Φ` at the leaf — the function training has never called. `D` is now a
real experimental knob rather than a config field with no effect (`multiturn_rollout.py:208`
currently hardcodes `depth=1`, and even that only feeds the reliability gate).

### 2.3 Why the previous estimator was guaranteed to tie

The shipped tree advantage is `T_k = γΦ_H^k − m`, `m = (1/K)Σ_j Φ_H^j`
(`grpo_trainer.py:328-340`). Suppose the reward is any positive affine function of it,
`R_k = c·T_k + d` with `c > 0` and `d` constant within the group. Then

```
mean_k R_k  = c(γm − m) + d
R_k − mean  = c·γ·(Φ_H^k − m)
A_k          = (R_k − mean) / (std_k + 1e-4)  =  (Φ_H^k − m) / (std_k(Φ_H)/γ + …)
```

which is **identically** the advantage produced by `R_k = Φ_H^k`. The `− m` subtraction is
annihilated by the mean-centring that follows it (`grpo_trainer.py:1967, 1987-1989` in TRL). The
"tree D=1" cell and the "α=5 baseline" cell optimised the same objective; 12% vs 12% was the only
possible outcome.

Measured, over 8 problems × K=8 rollouts (this doc's verification script, procedural policy, no LLM):

| quantity | R² against `Φ_H` |
|---|---|
| `γΦ_H^k − mean_j Φ_H^j` (shipped tree advantage) | **1.0000** |
| `Σ_t [γΦ(s_{t+1}) − Φ(s_t)]` (`lagrangian_potential_v2`) | **0.9998** |
| `Σ_t [Φ(s_{t+1}) − b̂(s_t)]/σ̂` (proposed) | **0.5512**  (range 0.005 – 0.981) |

The second row is the more damaging one: the *shaping reward itself* is, at γ=0.99 and H≤5,
numerically a function of `Φ_H` alone. So the tree arm did not merely fail to add search — it added
a second copy of the one signal already present. Any comparison built on that pair was a
weight sweep.

### 2.4 Why the new one survives

Take γ=1, σ̂≡1 for exposition and sum over the rollout:

```
S_k = Σ_t [ Φ(s_{t+1}^k) − b̂(s_t^k) ]
    = Σ_t [ Φ(s_{t+1}^k) − Φ(s_t^k) ]  −  Σ_t [ b̂(s_t^k) − Φ(s_t^k) ]
    = [ Φ_H^k − Φ(s_0) ]               −  G_k ,        G_k := Σ_t g(s_t^k),
      g(s) := b̂(s) − Φ(s) = mean_c [ Φ(f(s,c)) − Φ(s) ]
```

The first bracket is the telescoped terminal term GRPO already carries and centres away (`Φ(s_0)`
is a group constant). `G_k` is the **total one-step improvement that was on the table along the
path** — a functional of the whole trajectory, not of its endpoint. Two rollouts that finish in
the same state but traverse states with different available improvement receive different `S_k`.
Hence `S_k` is not measurable with respect to `σ(Φ_H^k)`: it is not an affine function of `Φ_H^k`,
not any function of it, and within-group centring leaves a non-degenerate residual. The measured
R² of 0.55 is that residual, empirically.

`−G_k` is also the *interpretation* the design doc always wanted: it divides out state difficulty.
A rollout stuck in a hard state is no longer punished for the state; it is judged on whether it
took the best of the moves that state offered.

### 2.5 The per-token version is outside the image of the map, not merely off its diagonal

GRPO's advantage is one scalar per sequence, broadcast to every token: it lives in
`span{1_[k]} ⊂ R^{K×T}`. Decompose a per-token injection `Â^k ∈ R^T` as

```
Â^k = ā^k · 1  +  Ã^k ,     ā^k = mean_{t ∈ model tokens} Â_t^k ,   Ã^k ⟂ 1
```

`§2.4` shows the coefficient `ā^k` already carries information beyond `Φ_H`. `Ã^k` is stronger:
**no** group-level affine transform of **any** per-sequence scalar reward can produce it. Per-token
delivery is therefore not a variance-reduction nicety — it is the only channel through which
per-step credit can reach the gradient at all.

### 2.6 The added term does not move the optimum

Write the unnormalised per-step term as `Â_t = γM(s_{t+1}) − b̂(s_t)` and split:

```
Â_t = [ γ·M(s_{t+1}) − M(s_t) ]   +   [ M(s_t) − b̂(s_t) ]
       ↑ potential-based shaping     ↑ a function of s_t alone
```

The first is exact potential-based shaping with potential `M` (Ng, Harada & Russell 1999): it
preserves the optimal policy of the underlying MDP. The second is an action-independent baseline:
valid for any policy-gradient estimator, changing variance but not the expected gradient
direction. **This holds for every choice of `C(s)`, including a policy-independent procedural one**,
because `b̂` depends only on `s`. That is the theoretical licence for §3's decision, and it is the
property that distinguishes this from arbitrary reward shaping.

---

## 3. (a) Where the candidate actions come from

Three sources, with the bias each injects into `V̂` and `b̂`.

### (i) Extra LLM samples at each visited state

`b̂(s) = E_{c ~ π_θ(·|s)}[Φ(f(s,c))]` is then the exact one-step on-policy state value under Φ, and
`Â_t` is the true one-step advantage. **Bias: none** beyond Monte-Carlo error at finite `b`; the
baseline tracks the policy as it improves, which is what an advantage is supposed to do.

Cost is the objection, and it is decisive. Measured from `logs/mt_t3_*`, `logs/mt_t4_*`:
`train_runtime` 44.5k–50.5k s / 100 steps → **445–505 s per step**, of which generation is
essentially all of it; `completions/mean_length` 3,430–5,890 **model** tokens per rollout across
≤5 turns → 700–1,200 tokens per turn. Full-reasoning candidate samples multiply decode steps by
`(1+b)` at batch `8(1+b)`; at these batch sizes a 14B LoRA decode is bandwidth-bound, so the step
grows ≈1.7× at b=3, ≈2.2× at b=5, ≈3.0× at b=8. That is 190 → 108 / 86 / 62 steps in a 24 h job.

A cheaper variant — prefill `<think></think>` and cap candidates at ~64 new tokens — costs only
+10–15% wall clock, but the candidates then come from the *non-thinking* policy `π̃ ≠ π_θ`. `b̂` is
still a valid state-only baseline (§2.6 survives), but it is no longer the on-policy value, so the
"true advantage" argument for (i) is lost. Keep this as an ablation arm, not the main estimator.

### (ii) Reuse the K=8 sibling rollouts at the same turn index

`b̃_k(t) = (1/(K−1)) Σ_{j≠k} Φ(s_{t+1}^j)`. Leave-one-out makes it independent of `a_t^k`, so it is
an **admissible baseline** — unbiased. It is nevertheless the wrong quantity, for a specific
reason: it is a function of the *group*, not of `s_t^k`. The siblings share the problem but their
states have diverged, so `Φ(s_{t+1}^j)` is a successor of a different state. The estimator then
conflates "this action was bad" with "this rollout is in a harder state" — precisely the
state-difficulty confound `b̂(s)` exists to remove. A rollout that wandered somewhere hard is
penalised at every subsequent step no matter what it does.

It is valid **exactly where the states coincide**: at `t = 0`, all `K` rollouts sit on `s_0`, so
`b̃_k(0)` *is* a within-state counterfactual baseline with candidates drawn from `π_θ` — option (i)
at `b = K−1` for zero extra generation. Use it there, as a free on-policy cross-check
(`lookahead/turn0_*`), and nowhere else. Note also that summing `b̃` over `t` and telescoping
reproduces the failed group-relative terminal estimator when `H = 1`: (ii) *is* the degenerate case
whose collapse §2.3 proves. That makes it the ideal negative control (arm A5).

### (iii) A procedural candidate set from the simulator — **selected**

`C(s)` enumerated from the action grammar and the problem's own parameter bounds, as
`scripts/search_ladder.py:133 candidate_actions` already does (per-member `SCALE_PARAM`, global
`SCALE_MULTI_PARAM`, and the fully-stressed-design macro at `:170 fsd_macro`), with the policy's
own action always included in the scored set.

Cost: one transition per candidate. Measured on this cluster, 6 problems × 60 transitions:
**2.2 ms median** for `deepcopy + apply + analyze` (1.4–3.7 ms over 9–21 members), and **6.4 ms**
for the full probe code path including `Φ` over the 3,752-call verification run. Against a 450 s
step this is free (§5).

**Bias, stated plainly.** `b̂` is now the mean over a *fixed, policy-independent* set. Two
consequences: (1) `Â_t` is the advantage of the policy's action against a **reference action
distribution**, not against the policy's own — which is exactly the "distil search into the policy"
claim, so it is the intended semantics rather than a defect; (2) the set is expressible only in
sizing moves, so actions the enumerator cannot mirror (`ADD_MEMBER`, `MOVE_JOINT`, topology change)
are scored against a sizing-only reference and will look systematically worse. That is a real bias
and must be reported. Two mitigations, both cheap: always include the policy's own action in the
scored set (so `rank_frac` and `regret` remain well-defined), and use
`tree_expansion.py:75 stratified_select_candidates` with `ensure_full_class_coverage=True` so each
action class is represented when `b ≥ |ACTION_CLASS_ORDER|`.

**Why (iii) wins.** The expensive resource here is LLM sampling; the cheap one is the simulator, by
five orders of magnitude on the truss domain. The entire point of a design benchmark is that a
high-fidelity transition model is available and language models do not have one — so the candidate
set should be generated where the cost is. And the offline evidence is already in: the CPU search
ladder shows this exact candidate set under Φ_v2 reaching 88.5% feasibility over all 130 problems
with **zero** LLM calls (`results/search_ladder/full/runs.jsonl`), versus 54% for the trained
champion. The reference set demonstrably contains near-optimal actions; the bias is toward a set
that works.

The three sources become arms, not a guess: **A2 = (iii)**, **A4 = (i)**, **A5 = (ii)**.

---

## 4. (c) Delivering a per-step signal through TRL 1.0.0

TRL's reward path is per-rollout scalar. Two ways out; both were checked against the installed
source at `/jet/home/wxu7/.conda/envs/my_env/lib/python3.11/site-packages/trl`.

### Option A — scalar sum (`R_k += κ Σ_t Â_t^k`)

Zero TRL involvement. `LookaheadAdvantageReward` (`rewards.py:773`) already does this, reading
`rollout.tree_metrics["lookahead_probes"]` threaded by the rollout_func. It survives group
centring by §2.4. But it delivers one number for the whole trajectory: every token of the rollout
gets the same coefficient, so per-step credit assignment — the actual claim — is lost. Keep it as
arm **A1**, the ablation that separates "does the search signal help at all" from "does localising
it to the step help".

### Option B — per-token advantage injection — **selected for the headline arm**

TRL 1.0.0 supports `(B, T)` advantages natively. The chain, verified end to end:

| step | source | fact |
|---|---|---|
| extras pass through | `trl/trainer/grpo_trainer.py:1612` | any key the rollout_func returns beyond `prompt_ids/completion_ids/logprobs` becomes an extra field |
| extras reach the batch | `:1949-1955` | `inp[key] = values[i]` — merged 1:1 into the same `inputs` list object the caller passed in, so an override can read them after `super()` returns |
| rewards are gathered | `:1239` (`_calculate_rewards`) | full-batch, then grouped by `view(-1, num_generations)` at `:1967` |
| scalar advantages | `:1987-1989` | `rewards − mean_grouped`, `/ (std + 1e-4)` under `scale_rewards="group"` |
| local slice | `:2014-2015` | `advantages[process_slice]`, length `len(prompts) == len(inputs)` — index-aligned with local `inputs` |
| returned | `:2091-2097` | `output["advantages"]` |
| buffering | `:1115-1117`, `utils.py:866 shuffle_sequence_dict`, `utils.py:831 split_tensor_dict` | both permute/slice along dim 0 only; a `(B, T)` tensor survives shuffling and micro-batch splitting intact, in unison with `completion_ids` |
| the loss | `:2290-2294` | *"To support subclasses that provide advantages with shape (B, T) … we **conditionally** unsqueeze"* — an explicit, documented contract |
| elementwise use | `:2353-2355` | `per_token_loss1 = coef_1 * advantages`, `coef_1` is `(B,T)` at `importance_sampling_level="token"` |
| env tokens excluded | `:2263` | `mask = completion_mask * tool_mask`; `env_mask` was popped into `tool_mask` at `:1649` |
| alignment | `:1769-1774` | `completion_ids` is **right**-padded, so a right-zero-padded `(B,T)` advantage matches column for column |

So: **subclass `GRPOTrainer`, override `_generate_and_score_completions`, no fork, no monkeypatch,
no vendored copy.**

```python
def _generate_and_score_completions(self, inputs):
    out   = super()._generate_and_score_completions(inputs)   # inputs[i] now carries the extras
    A     = out["advantages"]                                  # (B,) group-centred
    B, T  = out["completion_ids"].shape
    step  = torch.zeros(B, T, device=A.device, dtype=A.dtype)
    for i, inp in enumerate(inputs):
        for (lo, hi), a in zip(inp.get("step_spans") or [], inp.get("step_advantages") or []):
            step[i, lo:min(hi, T)] = a
    out["advantages"] = A.unsqueeze(1) + self.kappa * step.clamp(-3.0, 3.0)
    return out
```

Guards the subclass must assert, because each would silently reinterpret a `(B,T)` tensor:
`use_liger_kernel is False` (`:2179` → `compute_liger_loss` hands advantages to a kernel expecting
`(B,)`), `off_policy_mask_threshold is None` (`:2303`, `get_off_policy_mask` expects `(B,1)`),
`loss_type != "vespo"` (`:2361`). All three are already at their defaults in this project
(`grpo_trainer.py:156-203` sets none of them).

`Â` is clipped to ±3 because it is already σ̂-normalised, and the GRPO advantage it is added to is
z-scored — so `κ ∈ {0.3, 1.0}` is directly interpretable as the relative weight of per-step
lookahead against outcome credit.

**Run both.** A1 (scalar) and A2 (per-token) share every other setting, so their difference is a
clean measurement of whether per-step localisation matters. If A1 ≈ A2, the paper's claim narrows
honestly to "search-derived signal helps"; if A2 > A1, the credit-assignment claim is earned.

---

## 5. (d) Cost model

Baseline, measured: **445–505 s/step**; 100 steps in ~12.4 h; a 24 h `GPU-shared` job at
2×H100-80 fits ≈190 steps. Per rank the generation batch is one group: 8 rollouts × ≤5 turns =
**≤40 probed states per step**. Transition cost **6.4 ms** (probe code path, measured).

### Truss, procedural candidates (option iii)

| D | b | transitions/step | added s/step | % of a 450 s step | steps in 24 h |
|---|---|---|---|---|---|
| 1 | 3  | 120   | 0.8  | +0.2% | 190 |
| 1 | 5  | 200   | 1.3  | +0.3% | 190 |
| 1 | 8  | 320   | 2.0  | +0.5% | 189 |
| 1 | 12 | 480   | 3.1  | +0.7% | 188 |
| 1 | 48 *(current probe default)* | 1,920 | 12.3 | +2.7% | 185 |
| 2 | 3  | 480   | 3.1  | +0.7% | 188 |
| 2 | 5  | 1,200 | 7.7  | +1.7% | 187 |
| 2 | 8  | 2,880 | 18.4 | +4.1% | 182 |
| 2 | 12 | 6,240 | 40   | +8.9% | 174 |

Every cell fits a 24 h job. The simulator is not the constraint on this domain; `b = 12, D = 1`
costs 0.7% and `D = 2, b = 8` costs 4.1%. **Recommended: D=1, b=12 for the headline; D=2, b=8 as
the depth arm.** Single-threaded; if `D=2, b≥12` is ever wanted, pre-fork a worker pool at import
time in `train_grpo.py` *before* the model is loaded (forking after CUDA init is unsafe).

### LLM candidates (option i), same D=1

| b | mode | step time | steps in 24 h |
|---|---|---|---|
| 3 | full reasoning | ~800 s | 108 |
| 5 | full reasoning | ~1,000 s | 86 |
| 8 | full reasoning | ~1,400 s | 62 |
| 5 | `<think></think>` prefill, 64-token cap | ~510 s | 169 |

Only the last row is affordable at experiment scale; it is arm A4, with the distributional caveat
of §3(i).

### The battery domain inverts the cost model — a first-class design constraint

Measured from `results/battery_ladder/probe_dfn_full.jsonl` and `probe_spme_lean.jsonl`:
PyBaMM **DFN median 3.73 s/sim**, **SPMe-lean 1.63 s/sim** — 250–580× the truss FEA. Naïvely,
`D=1, b=8` over 40 states costs **20 minutes per training step**. Three mechanisms make the same
code affordable, and each is independently worth measuring:

1. **Transition memoisation** keyed by `(problem_id, tuple(action_history), action)`. The
   simulator is deterministic. At `t = 0` all `K = 8` siblings share `s_0`, so the candidate set is
   evaluated once per (problem, turn) instead of `K` times — an immediate ~K× saving at turn 0 and
   a large one wherever trajectories reconverge.
2. **A simulator budget with the reliability gate.** `lookahead_probe.py:124-127` already computes
   `margin = Φ(best) − Φ(2nd)`, `σ`, and `reliable = margin > 2γ^(D+1)σ`. Where the top two
   candidates are not separated beyond the noise band, the lookahead *cannot* rank them, so
   spending on it buys nothing. Spend a per-batch budget `B_sim` where `reliable` is likely; skip
   elsewhere. This turns the gate from a logged diagnostic into the scheduling rule.
3. **Fidelity split:** rank candidates with the cheap model (SPMe, 1.63 s), realise the chosen
   transition with the expensive one (DFN, 3.73 s). The lookahead needs an *ordering*, not a value.
   This yields a directly testable quantity — Spearman ρ between `Φ_SPMe` and `Φ_DFN` over the
   same candidate sets — which is arguably the most transferable result in the study: it says how
   coarse a surrogate a search-based potential tolerates.

With (1)+(2)+(3) at `b = 5`, `B_sim = 64/step`: 64 × 1.63 s ≈ 104 s of candidate scoring per step,
against a battery step already dominated by its own DFN transitions.

---

## 6. The experiment

**Data.** `data/splits/truss_v1.json` (86 train / 43 eval, stratified on family × degeneracy ×
initial violation, seed 20260823) — this is the first run in the study whose eval set is genuinely
held out. All arms: `group_size 8`, `max_turns 5`, 100 steps, LoRA r32 on the warmstart checkpoint,
**≥2 seeds**.

| arm | change from A0 | tests |
|---|---|---|
| **A0** control | `feasibility 1.0 + fos 0.5 + grammar 0.1 + step_eff 0.1 + lagrangian_potential_v2 1.0` | the T5b reference |
| **A1** scalar sum | + `lookahead_advantage 1.0`, D=1 b=12 procedural | does search signal help at all |
| **A2** per-token *(headline)* | per-token injection, κ=1.0, D=1 b=12 procedural | does per-step credit assignment help |
| **A3** depth | A2 with D=2, b=8 | does deeper `V̂` beat shallower — the `D` claim |
| **A4** candidate source | A2 with LLM candidates (no-think, b=5) | procedural vs on-policy candidate bias |
| **A5** sibling control | A2 with the group baseline (option ii) | **negative control** — predicted ≈ A0 |
| **A6** reference line | no training: the procedural Φ_v2 D=1 policy | 88.5% feasible, the bar |

**Headline metric changes.** Feasibility rate alone is disqualified: the 86%-feasible greedy
heuristic beats the 54% trained champion, so the metric cannot separate the hypotheses. Report
instead, on the eval split:
* `median mass_ratio = final_mass / _metadata.optimal_mass` among feasible rollouts,
* `frac(feasible AND mass_ratio ≤ 1.1)` — feasible *and* near-optimal, the joint success rate,
* `lookahead/regret` = `Φ(best candidate) − Φ(policy's successor)` — the internalisation metric.
  ρ is binary and saturates; regret keeps reporting after ρ stops moving.

**Pre-registered predictions** (write these down before submitting, so the null is informative):
1. A2 > A0 on mass-ratio and on joint success; A1 strictly between them.
2. **A5 ≈ A0 within the ±4-point noise floor at n=43** — the arm that demonstrates the mechanism is
   the *state-conditioned baseline*, not the reward magnitude. If A5 also improves, the conclusion
   is "any extra dense term helps" and the lookahead claim is not supported.
3. `lookahead/regret` falls monotonically in A2, flat in A0 and A5.
4. Per-batch diagnostic `R²(Σ_t Â_t, Φ_H)` logs ≈0.5 in A2 and ≈1.0 in A5 — a live, cheap check
   that the estimator has not silently degenerated back into the collapsed one. **This is the
   single most valuable guard rail in the design**; the previous attempt would have been caught on
   step 1 by it.

**Domain generality.** The battery arm (12 problems, `data/battery_problems/`, 4 families) is a
transfer and cost study, not an A/B — n=12 cannot support a training comparison. Report: (a) that
the estimator code path is byte-identical, with only `program_from_battery_goals`
(`potential.py:423`) and a battery candidate enumerator swapped; (b) the SPMe↔DFN rank correlation
of §5; (c) the search ladder under Φ_v2 vs Φ_v1 via the existing mirror `scripts/battery_ladder.py`.
The claim to make is *"the estimator is domain-general and its cost profile is not"* — which is
a stronger and more honest paper than a second feasibility table.

---

## 7. (e) Patch plan — exact edits, none applied

### 7.1 `llm_finetune/training/rl/posterior/lookahead_probe.py`

| line | edit |
|---|---|
| `:75` `probe_state(...)` | add params `value_fn: Callable[[dict], float] \| None = None`, `depth: int` (already present but unused for search), `memo: dict \| None = None` |
| `:104` | `scored.append((action, value_of(nxt)))` where `value_of = value_fn or (lambda s: compute_potential_v2(s, program, **phi_kwargs))` — this is the single line that turns D=1 into D≥1 |
| `:116-121` | same substitution for `policy_next_state`; keep the "forfeited step scores at the worst candidate" rule but set a new `policy_failed: bool` field so it is distinguishable in logs rather than silently indistinguishable from a genuinely bad action |
| `:129-141` `ProbeResult(...)` | add `advantage_z = (policy_phi − baseline_phi)/(sigma + 1e-6)`, `n_sim_calls: int`, `policy_failed: bool` |
| after `:141` | **new** `def search_value(state, program, transition_fn, candidates_fn, depth, gamma, branching, memo)` implementing `M_d` of §2.2, delegating candidate selection to `tree_expansion.py:75 stratified_select_candidates` for class coverage |
| `:188-218` `truss_candidate_actions` | replace body with an import from the new shared module below — the current set is `SCALE_PARAM` only, a strict subset of the set that produced the 88.5% offline result, so training and the offline evidence are measuring different searches |
| new, end of file | `CANDIDATE_ENUMERATORS: dict[str, Callable] = {"truss": ..., "battery": ...}` + a `CandidateEnumerator` Protocol — the second (and last) per-domain hook alongside `program_from_goals` |

### 7.2 **New** `llm_finetune/envs/truss_candidates.py`

Move `candidate_actions` (`scripts/search_ladder.py:133`), `fsd_macro` (`:170`) and the
`mass ≤ 0 ⇒ invalid` guard (`:201-215`) into an importable module; edit
`scripts/search_ladder.py:133,170,201` to import from it. Rationale: the training-time candidate
set must be *the same object* as the one the offline ladder result was measured with, or the two
numbers are not comparable.

### 7.3 `llm_finetune/training/rl/multiturn_rollout.py`

| line | edit |
|---|---|
| `:248-256` | extend the config block: `lookahead.{depth, b, candidate_source, sim_budget_per_batch, kappa, per_token, reliable_gate}` |
| `:296` `st.update(...)` | add `step_spans=[]`, `step_adv=[]` |
| `:331-334` | capture `start = len(st["comp_ids"])` **before** `st["comp_ids"] += gen_ids`, then `st["step_spans"].append((start, len(st["comp_ids"])))`. The env glue appended at `:392-395` stays outside the span — correct, those tokens carry `env_mask=0` and no gradient |
| `:344-349` (unparsed branch) | **bug fix, load-bearing:** emit a probe row here too, with `policy_failed=True` and the worst-candidate advantage. Today `_probe_here` is called only inside the `else` (parsed) branch, so `LookaheadAdvantageReward` (`rewards.py:811`) sums over a number of steps that equals the *parse-success count*. A policy that emits garbage scores 0 and looks neutral instead of bad; the reward silently depends on the parse rate rather than on the policy's choices |
| `:351-355` `do_probe` | drop the `probe_budget[0] > 0` states-cap on the training path; replace with a **simulator-call** budget (states are the wrong unit — `b` varies with member count, 72–168 candidates across the problem set) |
| `:356` | hoist `pre_truss = copy.deepcopy(st["truss"])` above the parse branch, since the unparsed path now needs it |
| `:367-372` `_probe_here(...)` | pass `depth`, the shared enumerator, and a batch-scoped `memo` dict keyed by `(problem_id, tuple(action_history), action)` — the K× saving at turn 0 and the mechanism that makes the battery domain viable |
| `:182-223` `_probe_here` | thread `depth`/`value_fn`/`memo`; return `advantage_z`, `sigma`, `n_sim_calls`, `policy_failed` |
| `:208` `depth=1` | → `depth=probe_depth` (currently hardcoded; `depth` reaches only the reliability gate) |
| `:401` `out = {...}` | add extra fields `"step_spans"` and `"step_advantages"` (one list per rollout). No TRL change needed: `trl/trainer/grpo_trainer.py:1612` passes unknown keys through and `:1949-1955` merges them 1:1 into `inputs` |
| `:419` | also stash `step_spans` / per-step z-advantages into `rr.tree_metrics` so the scalar-sum arm (A1) and the W&B rollout table can read them |

### 7.4 **New** `llm_finetune/training/rl/lookahead_grpo_trainer.py`

`class LookaheadGRPOTrainer(GRPOTrainer)` with the `_generate_and_score_completions` override of
§4 Option B; constructor takes `kappa` and asserts the three guards (`use_liger_kernel is False`,
`off_policy_mask_threshold is None`, `loss_type != "vespo"`). Logs, via
`self._log_metric` (`trl/trainer/grpo_trainer.py:1137`): `lookahead/kappa_effective`,
`lookahead/step_adv_abs_mean`, `lookahead/frac_clipped`, and the §6 guard rail
`lookahead/r2_step_sum_vs_phiH`.

### 7.5 `llm_finetune/training/rl/grpo_trainer.py`

| line | edit |
|---|---|
| `:97` | import `LookaheadGRPOTrainer` alongside `GRPOTrainer` |
| `:263` `trainer = GRPOTrainer(**trainer_kwargs)` | select `LookaheadGRPOTrainer(..., kappa=...)` when `rl.lookahead.per_token` is true |
| `:278-296` `_compute_group_tree_advantages` | **keep** (the ablation ledger cites it) but add a startup `log.warning` stating it is affine-equivalent to `lagrangian_potential` under `scale_rewards="group"`, so no future run is designed around it |
| `:345` `_aggregate_probe_metrics` | add `lookahead/advantage_z`, `lookahead/policy_failed_frac`, `lookahead/sim_calls`, `lookahead/turn0_*` (the option-(ii) free on-policy cross-check) |
| `:385` `_compute_rho_per_group` | rename its emitted key `rho_tree_agreement` → `rho_group_argmax_legacy` (`:585`, `:611`), so nobody reads the identity of finding 2 as §3.6's ρ |

### 7.6 `llm_finetune/training/rl/rewards.py`

`:811-825` `LookaheadAdvantageReward.compute` — read `advantage_z` rather than raw `advantage`
(dimensionless, so κ transfers across domains), and sum over **all** turns now that failed turns
also emit a probe row. Registry at `:829-839` unchanged.

### 7.7 Configs

New: `configs/rl/grpo_la_a0_control.yaml`, `…a1_scalar`, `…a2_pertoken`, `…a3_depth2`,
`…a4_llm_candidates`, `…a5_sibling_control` — each written out **in full**, not via `defaults:`
inheritance, because Hydra merges dict-valued keys and every previous `mt_*` config silently
inherited `grpo_truss.yaml`'s four reward terms on top of the one it declared.
New `configs/data/rl_problems_split.yaml` → `split_file: data/splits/truss_v1.json`, `split: train`.

### 7.8 `llm_finetune/data/datasets/rl_dataset.py`

`:78-117` `from_problems_dir` — accept `split_file` / `split` and filter `problem_files` by the id
list. Without this every arm still trains on its own eval set.

### 7.9 `scripts/eval_checkpoint.py`

`:158-168` result dict — add `optimal_mass` from `spec["_metadata"]["optimal_mass"]` and
`mass_ratio`. `:367-372` aggregate — add `median_mass_ratio_feasible` and
`frac_feasible_within_1_1x`, and read the eval half of the split.

### 7.10 `llm_finetune/data/processors/designbench_prompt.py:37,44`

The worked example `SCALE_PARAM(all_members, thickness, 1.224)` is rejected by the executor's
`SCALE_PARAM\s*\(\s*(\d+)` member-id regex on every use — the one in-context demonstration the
model gets is an action that always fails. Replace with a concrete member id. This is a confound
in every arm and should be fixed *before* the sweep, not during it, so all arms share it.

### 7.11 `tests/test_lookahead_advantage.py` (new)

1. **Collapse theorem:** group-centring `γΦ_H − mean(Φ_H)` equals group-centring `Φ_H` to
   floating-point — the regression test that would have caught the original defect.
2. **Non-collapse:** on a fixture group, `R²(Σ_t Â_t, Φ_H) < 0.95`.
3. **TRL contract:** a `(B,T)` advantage tensor round-trips through `shuffle_sequence_dict` and
   `split_tensor_dict` in unison with `completion_ids`, and `_compute_loss`'s conditional unsqueeze
   leaves it untouched.
4. **Failed-action accounting:** `probe_state` returns a row with `policy_failed=True` when the
   policy's action does not parse or does not apply.

---

## 8. Recommended configuration (one line)

> **A2:** per-token advantage injection via `LookaheadGRPOTrainer`, `κ = 1.0`, `D = 1`, `b = 12`
> procedural candidates from the shared truss enumerator with class coverage and the policy's own
> action always scored, `Φ = compute_potential_v2`, σ̂-normalised and clipped to ±3, dense at every
> turn with a per-batch simulator budget, on `data/splits/truss_v1.json`, 100 steps × 2 seeds,
> judged on median `mass/optimal_mass` among feasible and on `lookahead/regret`, against A0
> (no lookahead) and A5 (sibling baseline — the predicted null).
> Added cost: **+0.7% wall clock**. The previous attempt's failure was free to avoid.

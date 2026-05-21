# Reward Config → ρ Connection

> How `configs/rl/grpo_posterior.yaml` maps to the three ρ symbols in
> `docs/plan_posterior_reward_walkthrough.md`.

---

## The three ρ symbols

The methods doc uses ρ for three distinct quantities. They must not be confused:

| Symbol | Section | Meaning |
|---|---|---|
| **ρ(t)** | §3.6 | LLM-vs-tree-best agreement rate — the headline training diagnostic |
| **ρ_nn** | §2.4 | Dead-end avoidance radius in φ-space (5th-percentile of within-gold distances) |
| **ρ_adapt** | §4.4 | Per-batch cross-problem KL divergence — trajectory-level multiplier |

---

## ρ(t) — the headline metric

$$\rho(t) := \Pr\!\left[\underbrace{\arg\max_k \pi_\theta(a_k \mid s)}_{\text{LLM top action}} = \underbrace{\arg\max_k \hat{V}^{(D,b)}(s, a_k)}_{\text{tree-best action}}\right]$$

At GRPO iteration $t$, ρ(t) measures whether the LLM's greedy pick agrees with the
action the lookahead tree ranks highest. As training progresses, ρ(t) should rise
toward 1 — meaning the LLM has internalised the lookahead into its single-shot priors
and tree expansion is no longer needed at inference.

**Falsifiability target**: ρ(t) ≥ 0.85 on dev problems at convergence (§3.6).

---

## Config reward terms and their role in driving ρ(t)

### Step-level reward (§6.1)

$$r_\text{step}(s, a, s', t) =
  \underbrace{\gamma\Phi(s') - \Phi(s)}_{\texttt{lagrangian\_potential}}
  + \underbrace{\beta_2\, r_\text{macro}}_{\texttt{macro\_completion}}
  - \underbrace{\xi\, r_\text{dead}}_{\texttt{dead\_end\_avoidance}}
  + \underbrace{\mu\, r_\text{pred}}_{\texttt{forward\_prediction}}
  + \underbrace{\kappa\, r_\text{escape}}_{\texttt{stagnation\_escape}}$$

### Trajectory-level scaling (§6.1)

$$R(\tau) =
  \underbrace{\bigl(1 + \lambda\,d(s_0)\bigr)}_{\texttt{difficulty\_weighted}}
  \cdot
  \underbrace{\bigl(1 + \nu\,\rho_\text{adapt}\bigr)}_{\text{not yet wired}}
  \cdot \sum_{t=0}^{H-1} \gamma^t\, r_\text{step}(s_t, a_t, s_{t+1}, t)$$

### Full mapping

| Config key | Doc symbol | Default | Direct ρ(t) role |
|---|---|---|---|
| `lagrangian_potential: 1.0` | $r_\text{env} = \gamma\Phi(s')-\Phi(s)$ | §1 | Defines the Φ-landscape that $\hat{V}^{(D,b)}$ estimates; foundation for all ρ signals |
| `tree_advantage: 1.0` | $\hat{V}^{(D,b)}$ or $\gamma\Phi(s_H)-\Phi(s_0)$ | §3.1 | **Primary ρ(t) driver** — see §The tree_advantage flywheel below |
| `step_normalized_phi: 0.50` | $(\Phi_H - \Phi_0)/H$ | §3.5 | Efficiency shaping; rewards large-scope actions that also tend to rank highest in the tree |
| `difficulty_weighted: 0.30` | $1 + \lambda\,d(s_0)$, λ=0.5 | §4.2 | Upweights hard initial states; ensures ρ(t) is not trivially inflated by easy problems |
| `macro_completion: 0.05` | $\beta_2 = 0.05$ | §2.3 | Bonuses for gold-trace action-class patterns; biases policy toward tree-consistent sequences |
| `forward_prediction: 0.10` | $\mu = 0.10$ | §4.1 | Forces CoT to commit to FEA predictions; reduces post-hoc reasoning that cannot benefit from tree priors |
| `stagnation_escape: 0.30` | $\kappa = 0.30$ | §4.3 | Rewards topology-class switches at plateaus — the states where tree ranking is most informative |
| `feasibility: 0.50` | terminal indicator | — | Sparse terminal signal; tree expansion provides the dense credit-assignment that feasibility alone cannot |
| `grammar_compliance: 0.10` | format check | — | No direct ρ role; prevents parse failures from confounding candidate generation |
| `dead_end_avoidance: 0.20` (cost) | $\xi = 0.20$ | §2.4 | Uses **ρ_nn** radius; see §ρ_nn below |
| `constraint_violation: 1.0` (cost) | smooth FOS/mass penalty | §1.1 | Keeps policy in regions where Φ gradients are informative |

---

## The tree_advantage flywheel

`OnlineTreeAdvantageReward` ([rewards.py:556](../llm_finetune/training/rl/rewards.py#L556))
has two modes selected by `use_tree_expansion`:

```
use_tree_expansion: false  (previous default)
  └─→ fallback: γΦ(s_H) − Φ(s₀)
       = full-trajectory Φ gain, a D=∞ proxy
       GRPO group-normalises across K=8 rollouts
       → signals which trajectories ended at higher Φ
       → ρ(t) rises slowly as a side effect of Φ-shaping

use_tree_expansion: true   (current)
  └─→ reads rollout.tree_metrics["tree_advantage"]
       = A^(D,b)(s_t, a_t) from D=1, b=5 lookahead
       = V̂^(1,5)(s_t, a_t) − mean_j V̂^(1,5)(s_t, a_j)
       GRPO directly pulls π_θ toward the tree-best ranking
       → ρ(t) rises as an explicit training objective
```

### Causal chain with tree expansion enabled

```
GRPO step, state s
  │
  ├─ LLM samples K=8 candidates {a₁…a₈}
  │
  ├─ FEA + D=1, b=5 stratified lookahead
  │    └─ V̂^(1,5)(s, aₖ) for each k          ← ~24 FEA calls/step, ~2s/rollout
  │
  ├─ Advantage: A^(1,5)(s, aₖ) = V̂(s,aₖ) − mean_j V̂(s,aⱼ)
  │
  ├─ Policy gradient ∝ A^(1,5) · ∇log π_θ(aₖ|s)
  │    └─ raises prob of tree-best action, lowers prob of tree-worst
  │
  └─ ρ(t) = Pr[LLM top = tree best] rises over training iterations
       └─ at convergence ρ → 1: LLM picks tree-best single-shot
            → tree expansion unnecessary at inference time
```

### Bias bound reminder (Theorem 1)

With γ=0.99, D=1, greedy continuation, empirical $\bar\Delta_G \approx 0.2$:

$$\bigl|\hat{V}^{(1,5)}(s,a) - V^*(s,a)\bigr| \le \gamma^2 \cdot \bar\Delta_G \approx 0.196$$

The bound shrinks slowly with depth; the real benefit of D=1 over D=0 is **ranking
accuracy** (Theorem 2), not absolute bias. The condition for correct ranking is:

$$\Delta_\text{rank} > 2\gamma^2 \sigma_G(s) \approx 0.098$$

Action-class differences in $V^*$ are typically 0.2–0.4 on truss problems, well above
this threshold. Within-class parameter differences (~0.05–0.15) are at the boundary —
D=1 marginally helps there.

---

## Where ρ(t) is measured

`PosteriorEvalRecord.rho_tree_agreement` in
[evaluator.py:633](../llm_finetune/training/rl/posterior/evaluator.py#L633):

```python
rho_tree_agreement=int(top_model_action == tree_result.tree_best_action),
```

This is the per-context binary indicator. Averaged over all eval contexts it gives the
empirical ρ(t). It is computed by `scripts/eval_posterior_one_step.py`, not during the
GRPO training loop — it is a **diagnostic**, not a training loss term.

Track it in W&B as `eval/rho_tree_agreement_mean` to watch ρ(t) rise over training.

---

## ρ_nn — Dead-end avoidance radius (§2.4)

$$r_\text{dead}(s, a) = \mathbb{1}\!\bigl[
  \exists (s', a') \in \mathcal{D}_\text{dead-end} :
  \mathrm{class}(a) = \mathrm{class}(a'),\
  \|\varphi(s) - \varphi(s')\| < \rho_\text{nn}
\bigr]$$

**ρ_nn = 5th percentile of within-gold pairwise φ-distances.** This is a
data-derived constant set when building the dead-end index from the modification
trees, not a YAML hyperparameter.

Config connection: `dead_end_avoidance: 0.20` in `cost_weights` is the ξ
coefficient that scales $r_\text{dead}$ as a cost. The ρ_nn radius controls
*selectivity* (how conservatively the penalty fires); ξ controls *magnitude*.
The term is self-gating: when the current (s, a) is far from any known dead-end
in φ-space, it returns 0 and never penalises OOD states.

---

## ρ_adapt — Cross-problem strategy adaptation (§4.4)

$$\rho_\text{adapt} = \frac{1}{B}\sum_{b=1}^{B}
D_\text{KL}\bigl(p(\text{a-class} \mid b) \,\|\, p(\text{a-class})\bigr)$$

This measures how much the LLM's action-class distribution varies *across problems*
in a batch. A high ρ_adapt means the policy adapts its strategy to each problem
rather than applying a problem-agnostic action loop.

**Current status: not yet wired.** The trajectory multiplier $(1 + \nu\,\rho_\text{adapt})$
with ν=0.05 requires a batch-level view: aggregate action-class histograms across B
problems, compute the per-problem KL, then scale all trajectory rewards. This cannot
be done inside a single `RewardFunction.compute()` call (which is per-rollout). It
belongs in the GRPO update loop and is listed as step 7 in the §6.3 ablation
sequence — last, after the LLM-capability gap has already been confirmed at step 5.

---

## Full picture

```
grpo_posterior.yaml                                     ρ connection
────────────────────────────────────────────────────────────────────
lagrangian_potential ──→ Φ-landscape definition      │
tree_advantage       ──→ V̂^(D,b) / Φ_H−Φ_0 ────────→ ρ(t) primary driver
step_normalized_phi  ──→ efficiency shaping          │
difficulty_weighted  ──→ hard-init upweight          │
macro_completion     ──→ gold-pattern bonus          │  all push π_θ toward
forward_prediction   ──→ CoT grounding               │  tree-consistent
stagnation_escape    ──→ plateau-break bonus         │  behaviour
                                                     │
dead_end_avoidance   ──→ ξ cost, uses ρ_nn radius ──→ ρ_nn (data constant)
                                                     │
[not wired]          ──→ (1+ν·ρ_adapt) multiplier ──→ ρ_adapt (batch-level)
                                                     │
                    eval script measures             ↓
                    rho_tree_agreement ─────────→ empirical ρ(t) per context
```
# Synthetic CoT for run2 gold actions: own choice vs rationalization (2026-10-04)

**Question.** Gold trajectories (Φ-select on dataxp v2 / run2) contain actions only. Can we get reasoning for SFT in either of two ways?
1. **Own choice.** The model's own choices land on the gold move often and stably enough that its own (causal) reasoning can be kept.
2. **Rationalization.** The model is given the gold state, the gold action and what happened, and writes the reasoning. Is that reasoning true, stable and sensible for a good, verifiable fraction?

**Setup.**
- **Model:** Qwen3.8-27B, the model being tuned.
- **States:** 70 repair-phase states, 10 per v2 domain, sampled at random turns of run2 gold trajectories.
- **Observation:** the training-prompt observation (`render_cot.policy_observation`), median about 11k characters.
- **Code:** `dataxp/gold_probe.py` (prepare / run / score / verify / compress).
- **Raw outputs:** `shared/dataxp/goldprobe/`.

## Answer

**Option 1 does not hold.** The model's own choice matches the gold move in about a quarter of samples. Adding the previous trace, or the post trace (later gold steps), does not change that.

**Option 2 holds.** Rationalization from the gold action plus its outcome produces reasoning that is true and sensible in **83%** of traces. The 210 traces were judged by two independent Claude auditors, who agreed on 96% of them. The three samples per state are consistent in 66 of 70 states, and 65 of 70 states (93%) have at least one fully verified trace.

## Option 1: own choice (gold not shown)

"Move" means the same element and parameter, the same value for toggles and categoricals, and for continuous or additive parameters the same direction with a step within ×2 of gold.

| variant | samples | completed | same element | same move | states with ≥1 sample on gold | Φ percentile of picks |
|---|---|---|---|---|---|---|
| thinking on, 28k budget (K=2) | 140 | 47% (45% hit the budget, 8% overflowed the context) | 24% | 21% | 31% | 65–99 |
| thinking off, written reasoning (K=4) | 560 | 94% | 36% | **28%** | 42% | 79–95 |
| + previous trace (K=4, partial, 3 domains) | 180 | 93% | 31% | 22% | 42% | 79–94 |
| + previous and post traces (K=4) | 560 | 98% | 34% | **27%** | 48% | 73–95 |

Rate of samples landing on the gold move (thinking off, observation only), per domain:

| traffic2 | thermal2 | grid2 | frame3d2 | inventory2 | circuit2 | seir2 |
|---|---|---|---|---|---|---|
| 42% | 35% | 31% | 31% | 21% | 20% | 11% |

What this shows:
- The model's own picks are usually good moves (79–99th Φ percentile), but they are mostly not the gold move.
- Showing the previous trace or the post trace does not make it pick gold.
- Thinking on is worse in practice. Its thinking runs 7–28k tokens, and half the samples never finish.

## Option 2: rationalization (gold action plus outcome given)

The generator sees four things:
- the observation;
- the gold action;
- the earlier steps;
- for information only, the actual effect of the gold action and the later steps.

It writes 80–200 words of reasoning (thinking on), then the action and a claims block listing every effect on an aggregate metric that it states or implies.

| prompt | deterministic pass | good by both auditors | states with ≥1 good trace (K=3) | stable states |
|---|---|---|---|---|
| v1 (`rat`) | 94% | 77% | 67/70 | 67/70 |
| **v2 (`rat2`)** | 81%\* | **83%** | 65/70 | 66/70 |
| v1 without outcome or post trace (`rat_nopost`) | 84% (16 effect claims contradicted by the actual outcome) | not audited | – | – |

\* v2's deterministic failures are almost all "ungrounded" numbers. The auditors confirmed these are correct arithmetic on the observation, and none matches an outcome value, so nothing leaked. The grounding check is too strict for v2's number-heavy style.

**What the checks cover.**
- **Deterministic:**
  - the action equals gold;
  - every number is grounded in the observation;
  - every claimed aggregate effect agrees with the replayed outcome;
  - there is a claims block;
  - the length is within bounds.
- **Audit:** two independent Claude auditors per domain read every trace against the observation and the actual outcome. They judge:
  - state claims are true;
  - effect claims are consistent with the outcome;
  - nothing leaked from the outcome;
  - the explanation makes sense;
  - the 3 samples agree with each other.

Showing the outcome makes the reasoning more truthful. Without it, the same prompt produces effect claims that the real outcome contradicts (16 of 210, mostly inventory2). With it, there are none.

**v2 (good by both auditors), per domain:**

| circuit2 | traffic2 | thermal2 | seir2 | frame3d2 | grid2 | inventory2 |
|---|---|---|---|---|---|---|
| 97% | 93% | 87% | 83% | 80% | 72% | 67% |

v1 was better on grid2 (90%), so use v1 there.

**What fails (either auditor, v2):** of 209 traces,
- 19 have reasoning that does not hold up;
- 16 have effect statements not supported by the outcome;
- 11 make wrong statements about the current state;
- 2 leak information from the outcome.

No cheap signal predicts which traces fail. Agreement of effect claims across samples, number of claims and length all leave the good rate at 65–84%. So verification at scale needs the strong-model audit, not just the deterministic checks.

## Recipe for the SFT dataset (option 2)

For each repair-phase turn of run2 (8,440 turns):

1. Generate K=3 rationalizations with `rat2`, or `rat` for grid2, with thinking on and a 16k-token budget.
2. Apply the deterministic checks: the action must equal gold, no leaked outcome numbers, the claims must agree with the replayed outcome, and the claims block and length must pass. Relax grounding to allow arithmetic on observation numbers.
3. Have a different, stronger model audit the surviving traces. Keep the first trace judged good; drop the turn if none is.
4. Expected yield is about 93% of turns, and about 83% of individual traces pass.

**SFT target:** the rationale followed by the action. The claims block is only for verification and is not trained on.

**Cost:**
- Generation: about 4.5k tokens per trace, so 8,440 × 3 × 4.5k ≈ 115M tokens. That is about 60 hours on one Blackwell, or roughly a day on 4 H100s.
- Audit: about 5k tokens per trace.

**Caveat.** These traces state expected effects that are true in hindsight. The policy learns to state them without having seen the outcome. Whether that helps or hurts at inference is an SFT question. The collaborator's DX-20 found that state-only reasoning hurt through myopia, and these traces are not state-only.

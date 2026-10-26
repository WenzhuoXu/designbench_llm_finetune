# Rendering chain-of-thought for the dataxp multi-domain SFT set

**Status (2026-09-23):** data generated and gated. The CoT method is settled
and validated on 56 repair-phase turns across 7 domains: a **state-only**
trace (read the current state, say why this element; no predictions),
rendered by the tuned model **with thinking on**, filtered by deterministic
checks only. Not yet rendered at scale.
**Target model:** `Qwen/Qwen3.8-27B` — the model being tuned. Render with it.

This is a handoff: everything needed to continue, including the mistakes
already made and measured, so they are not repeated.

---

## 1. Where everything is

Shared, group-writable (bridges2): `/ocean/projects/mch250030p/shared/dataxp/`

| path | content |
|---|---|
| `run1/<variant>/accepted/{problems,trajectories,pairs}.jsonl` | the gated dataset: 13 variants, 1,586 problems, 1,085 trajectories (16,387 turns), 69,887 pairs |
| `run1/<variant>/report.json`, `run1/portfolio.json` | gate results, ladder table per variant |
| `code/dataxp/` | generator, domains (with `describe()`), `render_cot.py` |
| `cot_v4/<variant>.jsonl` | **current method**: 56 state-only traces (8 per domain), full prompts |
| `cot_v4_audit.json` | independent audit of those 56 (see §4) |
| `cot_obs/`, `cot_judged/` | superseded: effect-claiming traces and their outcome check (§2e) |

Model: bridges2 `HF_HOME=/ocean/projects/mch250030p/wxu7/hf_models`; drc
`/storage/wxu/huggingface` (same snapshot `1d4bf0f…`). Architecture class is
`qwen3_5` (`Qwen3_5ForConditionalGeneration`) — transformers' name for the
architecture, not the release; the weights are Qwen3.8-27B.

Design of the dataset itself: `docs/multi_domain_expansion.md`.

---

## 2. What failed before — read this first

Each of these was measured, not guessed.

**(a) Templated CoT justifying the winner does not transfer the decision rule.**
T6 (turn5 §9): regret vs procedural candidates went **1.606 → 3.427**.

**(b) Preference routes are closed.** DPO on Φ-ranked pairs taught
unconditional shrinking (ledger Turn 8b); printing Φ in the prompt did the
same (Turn 10). SFT is the live route.

**(c) 59% of trajectory turns are post-feasibility.** Render only repair-phase
turns (`binding_constraint != null`): ≈6,700 turns.

**(d) Wrong renderer inputs (2026-09-22):** pairs with "Candidate A/B"
(the policy never sees a rejected action's outcome); telling the renderer
which candidate won; a "state" of aggregate metrics only (the model invented
element roles); marking only the most severe constraint VIOLATED. Fixed by:
trajectory turns only, one action, `describe()` observation, every violation
marked.

**(e) Letting the trace predict the effect of the change.** With rules that
allowed directional claims ("lowering Cc raises GBW"), 27/56 traces
contradicted the measured outcome (replayed ground truth, independently
audited). Errors grew with how indirect the lever is (inventory 1/8, traffic
6/8): the model cannot predict second-order and network effects. → The trace
now makes **no predictions at all** (§3).

**(f) Telling the model "no predictions" is not enough without thinking.**
Thinking off, even with explicit allowed/not-allowed examples: an independent
audit found 4/56 traces clean; the model kept writing "increasing B raises I2"
and misread the observation (wrong bounds, wrong rankings). **Thinking on,
same prompt: 46/56 clean** (0 regex hits, 2 subtle predictions, 9 misreads).

**(g) Do not use the tuned model as its own judge.** A Qwen3.8-27B judge
(thinking on) agreed with an independent audit on 51/56 traces but passed 2
of its 3 "clean" traces wrongly and missed 19 state misreadings: it shares the
renderer's blind spots. It also cost ~40 min per 56 traces. Removed. Fix
errors at the source (the observation, §3) and measure with a different,
stronger model on a sample (§4).

---

## 3. The method

One trace per repair-phase trajectory turn:
**observation → reasoning → `<action>SET_PARAM(i, param, value)</action>`**

**Observation (self-contained).** The trajectory is replayed from its spec to
the turn (`render_cot.iter_states`); replay is exact (<1e-12 on the six
deterministic domains). `describe(st, spec)` prints PROBLEM (setup, tunables,
units, bounds, equations), DESIGN (every element, role, values), RESULTS
(per-element simulator outputs). The prompt adds CONSTRAINT STATUS (every
violation marked, most severe flagged), CHANGES ALREADY MADE, and the change to
reason toward with its direction and **the room in that direction** (read back
by clamping a copy, so per-element limits such as a generator's p_max are
right). `describe()` is what the policy sees at inference; keep them identical.

Observation wording was changed where the audit showed a recurring misread:
traffic prints capacity per approach, `Q 12.2x` (a multiple of storage, not
vehicles) and `blocked 84% of time by I1`; thermal lists the warmest nodes
under the limit; every prompt states the room in the move's direction.
`describe()` changes do not touch the dataset or replay.

**Renderer.** Same model, `enable_thinking=True` (only the answer is kept),
temperature 0.9, 8k-token budget, requests pooled across variants. System
prompt (in `render_cot.py`):
- cover: violated constraints and gaps; where the per-element results place
  the problem; the chosen element's role, setting, range, and how it compares
  with the others; a closing sentence naming the change and what it is
  *aimed at*;
- no predictions in any tense or form — dependencies of current results
  ("I2 = B·Ib") and naming the target ("aimed at the L23 overload") are fine;
  "raising B raises I2", "to relieve", "without affecting" are not;
- 60–150 words, rotated STYLE line (without it 43% opened with the same four
  words); end with the exact action line.

**Checks (deterministic, per trace):** every number appears in the
observation (or is a margin/percentage of key quantities); no predictive
surface forms (will/would/should/could/might/likely/"to reduce"/"so that"…,
with "room to increase" exempt); no search/candidate vocabulary; action line
present and equal to the gold action.

---

## 4. Results on the 56-turn validation set

| version | renderer | pass deterministic checks | clean by independent audit |
|---|---|---|---|
| effect-claiming (§2e) | thinking off | 50/56 | 29/56 outcome-consistent |
| state-only v2 | thinking off | 38/56 | 4/56 |
| state-only v3 | **thinking on** | 55/56 | **46/56** |
| state-only v4 | thinking on + observation fixes | 55/56 | **44/56** |

Independent audit = Claude subagents, one finder per domain and one skeptic
that tries to refute each finding, reading the full observation and doing the
arithmetic. It measures the method; it is not a per-trace filter. Two
independent runs of this audit on v4 agree on 46/48 traces.

What remains wrong in v4 (12 traces): one leaked draft (the thinking text
spilled into the answer; caught by the checks, now also by a length cap), one
implied trade-off ("tension between the objective and two violations",
prompted by a STYLE line that has since been replaced), and ten small state
misreadings: a violation attributed to the wrong group or to extra members
(seir peak in g0, not g1; frame3d 1.205 set by m18 alone), upward headroom
quoted for a decrease (grid, once, despite the new room-in-direction line),
"104.0%" for 104.2%, "still" at turn 0, a metric with no limit called "within
limits", the objective confused with a constraint (inventory), and thermal
grid-position/range wording. Apart from that trade-off, no trace states an
outcome number or a direction of effect. Traffic went from 6/8 to 8/8 after its wording fixes;
the other domains moved within sampling noise (temperature 0.9).

---

## 5. Next steps

1. Render all ≈6,700 repair-phase turns with the v4 settings (§6). At 16
   concurrent requests the Blackwell server does 56 thinking traces in about
   10 min (≈20 h for all ≈6,700 turns on one Blackwell; split across H100s to go faster).
2. Audit a random sample (≥50, stratified by domain) with a different, stronger
   model; fix recurring misreads in `describe()` wording rather than filtering.
3. Consider excluding turns whose gold move worsens the most-severe violation
   (circuit_0002 turns 0–1 lower Cc and drive PM 5.5° → −2.0° → −7.7°): a
   state-only trace can name the target but the state does not justify it.
4. Gate SFT on `regret_vs_procedural` (the metric that doubled in T6) and on
   parse rate at temperature 0.6–0.8, not on loss.

---

## 6. How to run

**Serving (drc).** `code/serve_qwen.sh` serves the snapshot on the Blackwell
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=4`; any other index is
another user's card) at `127.0.0.1:8766`, 32k context, 16 sequences. Loading
takes ~6 min. It uses the vLLM 0.29 env at `/storage/wenwenw/envs/vllm_drc`
(has `Qwen3_5ForConditionalGeneration` and sm_120 kernels). `/` is nearly
full: keep every cache under `/storage`. On bridges2 instead: `GPU-shared`,
`--gres=gpu:h100-80:1`, ≤12 CPUs; pass the **snapshot path**, not the repo id,
under `HF_HUB_OFFLINE`.

**Render** (CPU client on the same box; `--prompts-only` builds prompts
without the model):

```bash
cd <shared>/code
python -m dataxp.render_cot --data <shared>/run1 --out <out> \
    --variants frame3d circuit grid thermal traffic inventory seir \
    --endpoint http://127.0.0.1:8766/v1 --concurrency 16 --limit 0 \
    --temperature 0.9 --thinking
```

Keep flag per row: `accepted` (all deterministic checks pass).

**Row schema:** `problem_id, turn, variant, cot, action, element_label,
binding_constraint, state_metrics, n_prior_changes, accepted, reject_reasons,
ungrounded_numbers, prompt, prompt_chars, render_model, render_params`.

**Assembling SFT conversations** is a separate step: system = the problem
statement, user = the observation (the prompt up to, not including, the STYLE
line), assistant = `<think>{cot minus action line}</think>` + the action line.

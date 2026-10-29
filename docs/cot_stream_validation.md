# Validating the physics CoT stream before the full data and SFT run

Pre-registered 2026-10-29, before any stream is generated. Any change after results is logged in
§6 with its reason; results never change a rule silently.

## 1. What is being validated

**The stream.** One reasoning episode per design state:
1. Diagnosis with mechanism.
2. Candidate changes, each with predicted outcomes as 80% intervals `[low, expected, high]`.
3. Probes with `CHECK(action)`. The simulator returns what the design would show after that change,
   and nothing is applied.
4. After each probe, a review comparing prediction and result, a physical explanation of the
   error, and updated predictions.
5. One final action.

**The goal is physics reasoning:**
* predicting what an intervention does (accuracy);
* knowing when that prediction is unreliable (calibration);
* needing fewer probes as both improve.

The probes are training scaffolding, not search. Background: `docs/cot_writer_pilot.md` (§7.3 and
§7.4: the writers' wrong choices follow their own wrong predictions on coupled quantities).

**The three stages, cheapest first.** Each has a stop rule fixed here:
1. Stage 1 checks the stream itself, with no training.
2. Stage 2 is a small matched SFT probe.
3. Stage 3 compares learning-curve slopes before the full run.

## 2. Shared definitions

**Environment.**
* The state records, observations and spelling are the datav4 ones: b024-b027
  `rows/v2/states_phi.jsonl`, `DATAXP_SPELLING=multitool`, code `/storage/wxu/corpus/code_v4`.
  The observation is unchanged, including frame3d2's own-member tables (see §5, open decision 2).
* **Probe result.** The lines of the child state's observation that differ from the current
  observation, each with its section header, plus the child's full CONSTRAINT STATUS block. It is
  marked "simulated; the current design is unchanged". Unchanged lines are left out and the result
  says so. Nothing the next observation would show differently is hidden.
* Each probe counts as one simulator call.
* **Probe cap: 6 per state.** The pilot's GPT writers listed a median of 4 (sol) and 6 (terra)
  candidates. How often a stream hits the cap is reported. If over 20% of streams hit it, the cap
  binds; this is reported as a finding, and Stage 2 data uses a cap of 10.
* **Decision criterion stated to every writer and student:** every constraint satisfied first,
  then the lowest objective; on a near-limit constraint a small margin is worth a small objective
  cost. This is Φ's ordering in words. It is the default until the owner decides (§5, open
  decision 1).

**Held-out probe set H(s).** Four moves per state, fixed by rule from the state's enumerated legal
moves, not chosen by the writer:
1. the teacher's move;
2. the highest-Φ move on a different element than the teacher's;
3. the highest-Φ move on a third element; on frame3d2 it is the weakest member's highest-Φ move
   when that member is not already covered;
4. one uniformly random legal move (seeded by the state id), not already in H.

The H moves are asked in a side question that is never part of the stream. The writer is asked:
"predict the outcome of these changes". The reply is scored, not kept.

**Queried quantities per H move.**
* All CONSTRAINT STATUS metrics, which include the objective.
* On frame3d2 also `N_kN` and `fos_b` of the three lowest-FOS members of the current design, and of
  the member the move changes (from `frame3d2._nominal`).
* Element-level quantities of the other families are not queried. Their coupling shows only
  through the global metrics. This is a measurement choice, stated here.

**Score.** The interval score of the 80% interval (Gneiting & Raftery 2007), with α = 0.2:
`IS = (hi - lo) + 10 (lo - y)[y < lo] + 10 (y - hi)[y > hi]`.
* It is divided by the scale `s`, from `cot_pilot/scripts/verify.py`:
  * constraint metrics with a nonzero limit: `max(|y|, |limit|)`;
  * other metrics: `max(|y|, |current|)`;
  * member force: `max(|y|, |current N|)`;
  * FOS: `max(|y|, 1.5)`.
* Each normalised score is capped at 10, the score of a point miss by 100% of the scale. Uncapped
  values are reported too.
* A queried quantity that is missing or unparsed scores 10.
* Lower is better.

**Secondary measures.**
* Point error of the expected value, and skill over persistence:
  `1 - sum|expected - y| / sum|current - y|`.
* 80% coverage (target 0.80).

**Strata, reported separately and never used for gating:**
* family;
* frame3d2 own member vs other members (coupling);
* H moves on an element the stream probed vs not probed.

**Statistics.** The unit is the state. Per state, take the mean normalised score over its queried
(H move, quantity) pairs. Compare paired differences with a 10,000-resample state bootstrap and a
95% percentile CI.

**States for Stage 1: 250 train-split states from b024-b027.**
* The 200 pilot states (`/home/wxu/cot_pilot/states/selected.jsonl`): frame3d2 50 hard and
  50 easy, and 25 each from circuit2, grid2, seir2 and traffic2.
* 25 frame3d2 states whose best move is a joint move (`hard_top_joint`, excluded from the pilot).
* 25 frame3d2 states without a failing-members table (`no_table`, excluded from the pilot).
* The two new groups use the pilot's draw rule: seed 20261028, at most one state per problem.

The pilot's exclusions are not inherited.

**Writers in Stage 1.**
* `gpt-6.1-sol` and `gpt-6.1-terra`: Responses API, default reasoning, carrying their reasoning
  items between turns.
* `Qwen/Qwen3.8-27B`, thinking off: vLLM on Orchard, T = 0.6.
* K = 1 stream per state per writer.
* Qwen with thinking on is not run. In the pilot it gave no visible reply at a 16,000-token budget;
  a larger budget is a separate decision.

**Smoke.** First, 10 states (2 per family) per writer. One revision of the stream instruction or
the parser is allowed after the smoke and logged in §6. After the main run starts, no changes.

## 3. Stage 1: the stream itself (no training)

**G1, grounded and well-formed** (per writer). The writer passes if all three hold:
* (a) in at least 95% of `<review>` entries, the stated observed value equals the injected probe
  result within relative 5e-3. That tolerance is the rounding of the result's 3-significant-figure
  display. Entries whose key names no reported quantity are counted separately.
* (b) at least 90% of streams end in a legal action within the cap;
* (c) at least 90% of probes are legal.

A writer that fails G1 is dropped.

**G2, the corrections carry physics** (per writer). The H side question is asked twice in the same
stream:
* "pre": after the writer's first message, before any probe result;
* "post": after the stream ends.

Comparison:
* The paired difference pre − post in the normalised score, over H moves the writer did not probe.
* Streams with no probe have no post-probe answer and are counted, not scored.
* Pass: the 95% CI of the pooled mean difference lies above 0 (post better).
* Reported: per family, and the not-probed-element stratum. That stratum is transfer of a mechanism,
  not interpolation from a nearby probe.

A writer that fails G2 is dropped. If no writer passes, the stream is narration: stop and report.

**G3, the student can use the stream** (per writer that passes G1 and G2). The untuned
Qwen3.8-27B (thinking off, greedy) answers the same H side question in four contexts:
* (a) the state only;
* (b) the state plus the writer's whole stream (without its side questions);
* (c) as (b), but each probe result is replaced by the result of a different random legal move on
  the same state (seeded);
* (d) the state plus the probe actions and their results only, with no writer text.

H moves the stream probed are excluded.
* Pass: (b) beats (a), with the 95% CI of the paired mean above 0. If no writer passes, stop: the
  stream's content is not usable by our model family.
* Reported:
  * (b) vs (c): whether the specific results are used;
  * (b) vs (d): whether the writer's text adds usable physics beyond the raw probe data. If (b) does
    not beat (d), this is flagged; the predictions-only arm in Stage 2 tests it under training.

**Writer selection for Stage 2.** Among the writers that pass G1-G3, the one with the lowest G3 (b)
score. If two are within 2% of each other, the cheaper one.

## 4. Stage 2: matched SFT probe

**Data.**
* Streams from the selected writer on 1,050 new train-split states from b024-b027, 210 per family:
  1,000 for training and 50 for the dev loss (10 per family).
* Frame3d2 states are drawn uniformly, with no hard/easy filter.
* No Stage 1 state is used, and at most one state per problem.
* A stream becomes a row if it finished (status done) with a legal final action. There is no
  filter on decision quality (no Φ-percentile acceptance), because the target is the physics in
  the stream, not the teacher's decision. How many streams this rule drops is reported.
* Row format: the sftprobe tier format (system, user, then alternating assistant and probe-result
  turns), with the `assistant_after_think` target, so there is one training sample per assistant
  turn and every prefix is exactly as served. That target serves the student with thinking on and
  an empty think block, as in the sftprobe arms. The untuned reference is served with thinking off,
  as in Stage 1.

**Arms.** All arms use the same states, the same streams and the same recipe:
* **S, full stream:** loss on every assistant message; probe results are user turns and are not
  supervised.
* **C, shuffled control:** S with every probe result replaced by the result of a different random
  legal move on the same state. Everything the writer wrote is kept. If C matches S, the student
  learned the format, not the physics.
* **P, predictions-only:** S with the free text removed. Each assistant message keeps only its
  `<review>`, `<predictions>` and `<check>`/`<action>` blocks.

**Reference.** The untuned Qwen3.8-27B with the same instruction; no training.

**Training.** LoRA on Qwen3.8-27B with the sftprobe recipe: lr 1e-4, 2 epochs, thinking off.
Orchard preempt, checkpointing every few steps with resume.

**Evaluation.** 100 held-out states per trained family, from problems not used in any training
data, plus 100 each from thermal2 and inventory2 (transfer). K = 2, T = 0.6.
* **M1 physics:** the H side question after the student's first message, before any probe
  (normalised score; skill over persistence; frame3d2 coupling stratum).
* **M2 calibration:** 80% coverage and the interval score on the same answers.
* **M3 decisions** at probe budgets 0, 2 and 6:
  * Φ percentile;
  * accepted (at or above the 90th percentile, or the teacher's move);
  * frame3d2 right-element rate;
  * probes used.
* **M4 episodes:** 20 held-out problems per family (all 7), probe budget 2, the ladder's turn
  count. Recorded: final feasibility, objective over the `phi_select` reference, and simulator calls
  (probes plus steps).

**Gate to Stage 3.**
* S beats the untuned reference on M1 (95% CI above 0), and
* S beats C on M1 (95% CI above 0).

Otherwise, stop: the stream does not teach physics through SFT at this size, or it teaches format
only.

## 5. Stage 3: slope before scale

* 1,000 more streams on new train states (200 per family), for 2,000 in all.
* S, C and P are trained at 500, 1,000 (= Stage 2) and 2,000 streams. The 500 set is a fixed,
  family-stratified subset of the 1,000 set.
* The Stage 2 evaluation is used at every size.

**Recommendation rule for the full run:**
* **Full run with S data:** S's M1 at 2,000 is better than at 500 (95% CI above 0), and S − C at
  2,000 is at least S − C at 1,000.
* **Full run with P data instead:** P is at least as good as S at 2,000 on both M1 and M3.
* **Otherwise, no full run:** report the curves.

**Open decisions that stay with the owner.** The defaults above stand until the owner decides.
1. The decision criterion: Φ's ordering in words, as above, or a stated feasible-then-objective
   rule, or a final-design score.
2. Whether frame3d2's own-member tables are a priori hints. These are the failing members' radius
   to FOS 1.5 and the install-alone outcomes.
3. The probe budget counted at evaluation.

## 6. Change log

**2026-10-29, after the Qwen smoke** (10 states; the one allowed revision). The instruction text is
unchanged; three infrastructure and parser fixes:
* **Context.** One grid2 stream outgrew the vLLM context of 65,536 tokens at its 6th message (input
  50k tokens). The Qwen server now runs at 131,072 (the model's limit is 262,144). A stream that
  still outgrows it ends with status `context_length`, counted as not finished (G1 b) and not
  retried.
* **G1 (a) tolerance.** The results display some values with fewer significant figures than three
  (`max_deflection_m` 0.0116 is shown as "0.012"). A review value now also counts as grounded when it
  equals the displayed value exactly.
* **Smoke records** are kept in `out/smoke_qwen.jsonl` and are not part of the Stage 1 data.

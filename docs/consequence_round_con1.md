# Consequence round con1 — pre-registration, then results

Written 2026-11-25, before any con1 model, tree or skill-test result existed. Sections 1–7 are the
pre-registration and are not edited after results come in (corrections go in a dated addendum at the end).
Results go in section 8 and later.

**Yardstick (owner, 11-25):** "strong sign of actual physical reasoning, generalized across problem types, and if
we can show improvisation that would be even better." A feasibility gain alone does not meet it. The round has to
show that the model predicts what the simulator does, on problem types it was not trained on, and that it uses the
prediction.

## 1. What is trained

**Consequence rows.** One row = one training state, served exactly as the benchmark harness shows it to the policy
(the same system prompt and user text: `dataxp_policy_eval_v3rc.py --open-rules`, observation, open-rules lines,
refused-move note), followed by a question:

> QUESTION (no move is made now): if the move `MOVE` were played at the current design, what would the simulator
> return? Answer with the simulator's result message exactly as a check of that move returns it: either the line
> 'CHECK RESULT for <move>: refused by the simulator: <reason>. ...', or the CHECK RESULT line followed by the block
> CONSTRAINT STATUS (after this move), the block EFFECT OF THIS MOVE (before -> after) with one 'metric  before ->
> after' line per metric, and 'Feasible after this move: yes.' / 'no.'

(The format description was added on 11-25 before any result, so that pref1, which has never seen a check result,
is asked a question it can answer in the right form; later questions of a row repeat only the first two sentences.)

The target is the harness's own result message for that move, word for word: what a v3rc check returns
(`CHECK RESULT for MOVE ...`, `CONSTRAINT STATUS (after this move)`, `EFFECT OF THIS MOVE (before -> after)`, the
photonics fabrication line, `Feasible after this move: yes|no.`), or the refusal line with the simulator's reason.
The answer has an empty think block: no rationale in these rows. Good, bad and refused moves are all included.

**k-shot structure.** A row asks Q = 12 questions about the same state, one after the other, as a conversation:
question 1 is appended to the state prompt, each later question is a new user turn, and every answer is
supervised. So the answer at position j is a prediction made with j observed (move → result) pairs of the same
state in context. Every row therefore trains both the zero-shot forecast (position 0) and updating from probes
(positions 1–11). This is our analogue of Gen-1.5's one-shot "physical prompting" and replaces separate k-shot rows
(same objective, the state prompt is paid once instead of 12 times).

**Policy rows.** The open-rules restated dx38 replay rows (`ownrows_own`, 8 families, rationale then action; the
replay set the chk1 round used, local copy inside `analysis/remote/rl/chk1/orchard/train_chk1.jsonl.gz`). They
keep the policy writing a rationale before its action (the owner's standing rule). The consequence rows are an
extra training objective on the same weights, not an action-only arm: the policy is still served and judged with
rationale-before-action, and no policy reply in training is action-only.

**Training.** LoRA r64 from pref1's final adapter (`runs/pref1_dx38mtR_20261123/final`), dx38's recipe
(`dataxp_sft_scr1.yaml`: lr 1e-4, cosine, 3% warm-up, 64 rows per step, one epoch, loss temperature 1.0), every
row weighing the same. The trainer supervises every assistant turn of a consequence row (a masked mean over the
answer tokens); policy rows are supervised as before. Mix: about 3 consequence rows per policy row. The final
adapter is the primary one (no checkpoint picked on any test).

LoRA first. A full fine-tune is tried only if LoRA fails the skill test (section 4) and there is a specific reason
to expect more capacity to help (for example: training loss on consequence rows still falling at the end, or
dev error concentrated in long numeric blocks).

## 2. Data

**Families.** Consequence rows on the 8 training families: engibench, grid2op, photonics (native verbs) and
circuit2, frame3d2, grid2, seir2, traffic2 (verb surface). Held out entirely: inventory2, molopt, thermal2 (no
consequence row, no policy row). Benchmark problems are held out entirely (the training pool
`train_p2_datav4.jsonl` excludes them; checked again by canonical spec when the rows are built).

**States.**
- Native families: the states of the chk1 collection episodes on the training pool (pref1 and dx38 at T=0 under
  v3rc --open-rules --checks 4 --check-actions; recorded prompts), replayed with the harness's own bookkeeping.
- Verb-surface families: there is no collection of the policy's own states there, so states come from scripted
  episodes on the training pool: each move is, with probability 1/2, the best legal move by Phi (phi_select) and
  otherwise a uniform legal move, up to 6 moves; every turn is a state. No model is involved.
- About 2,000 states per family for training rows (fewer if the source has fewer), plus about 1% of problems as dev.

**Moves per state.** Every enumerated legal move is simulated (median legal set: engibench 48, grid2op 75,
photonics 32; verb-surface 50–160). A row's 12 questions are drawn from: the policy's own moves at that state (the
committed move and every move checked there in the chk1 collection, up to 4; native families only), the 2 best
legal moves by Phi, 2 off-grid moves on verb-surface families (a parameter scaled by a log-uniform factor in
[0.5, 2], outside the enumerated factors), and uniform legal moves for the rest. Uniform legal moves alone would
drown the good outcomes (a random legal policy reaches feasible@20 0.10), and legal-only moves would contain no
refusals; the own moves bring the refusals at the rate the policy actually makes them.

**Result message on verb-surface families.** The harness allows checks on native families only. On a verb-surface
family the message is check_text's construction on the harness's own verb-surface apply path (check_move, clamping
by domain.set, the history entry the harness appends), i.e. what v3rc would show if it allowed checks there.

**Platform.** All prompts and messages come from the harness's functions (`scripts/con1/conseq_lib.py`).
`verify_replay.py` replays recorded chk1 episodes and compares byte for byte. On Linux (the benchmark platform) the
target is exact; on Windows the solver output differs in the last printed digit of some values (measured: e.g.
`388.71` vs `388.709`, `flow +1.49e-14` vs `-5.44e-14`) and the order of exactly-tied cells. Data are generated on
Linux when Orchard or drc is available; if any part is generated on Windows the report says which, with the
measured share of messages that differ.

## 3. Skill test (before the benchmark)

**Test states.** From pref1's own benchmark episodes (pair a, `pref1_dx38mtR_20261123_open_all`), all 11
families: for each problem one turn drawn uniformly (seed 0) from the episode's turns, replayed with the harness's
bookkeeping. Two groups: **held-out problems** (benchmark problems of the 8 training families) and **held-out
families** (inventory2, molopt, thermal2). At most 60 states per family.

**Test moves.** 8 per state, by the same recipe as training (own = the move pref1 committed there in pair a, and in
pair b when pair b reached the same state; 2 best legal; off-grid on verb-surface families; uniform legal), in a
seeded random order.

**Two passes per model.** (1) zero-shot: each of the 8 questions alone (k = 0). (2) sequential: the 8 questions as
one conversation, the true result of each earlier question in context (teacher forcing), the prediction at
position j made with k = j observed pairs. Rendering is the training rendering (`/v1/completions` on token ids built
by the same function).

**Models / baselines.**
- con1 (final adapter), answering with an empty think block as trained.
- pref1 asked the same questions (thinking on, up to 4,096 tokens: its natural mode, the most favourable for it).
- Majority: the most common verdict of the family in con1's training transitions (held-out families: the test
  set's own majority, an oracle baseline, labelled so).
- Trees: per family, LightGBM on move features (verb, numeric arguments, element/parameter, value and ratio to the
  current value) plus state features (the current metric values, Phi, binding constraint, turn), trained on all
  simulated transitions of con1's training states of that family: a 3-class classifier for the verdict and a
  LambdaRank model (query = state) for ordering. Undefined on held-out families (their verbs are never seen).
- Nearest probe (k-shot control): predict the verdict of the most similar earlier probe in the same conversation
  (same verb, then the closest arguments); majority when there is none.

**Reading a predicted message.** Refusal if the answer contains `refused by the simulator`, or if no metric line can
be read and the answer says the move is refused / rejected / not applied (prose answers are read fairly; changed
11-25 before any result). Otherwise the predicted
value of each metric is the `after` value of its line in a section headed EFFECT (else its CONSTRAINT STATUS value, else unchanged), and
the predicted Phi is the problem's own potential of those values (`PhiEvaluator._phi_one`, all families are
deterministic); verdict better if predicted Phi > Phi of the state + 1e-9, else worse; photonics ranks by
(fewer predicted rule-breaking pixels, then Phi), the open-rules rule. An answer that is neither (no readable
message) counts as wrong.

**Metrics** (per family; group = mean over its families, se by bootstrap over states, 2,000 resamples):
- **V**: 3-way verdict accuracy (refused / worse / better) at k = 0. Primary.
- **D**: direction accuracy among moves that were applied and predicted applied (better vs worse). Separates
  physics from refusal prediction.
- **PO**: within-state pairwise ordering accuracy over truly applied moves with different Phi (predicted-refused
  moves rank last; ties count 1/2).
- **N**: per metric with a true change, sign of the predicted change right; and the share of metrics where the
  prediction is closer to the truth than "no change".
- Refusal precision and recall.
- **k-shot gain G(k)**: for each move at position k ≥ 1 of the sequential pass, correct(sequential) − correct(zero-
  shot) on the same move, averaged per k and pooled over k ≥ 1 (primary: pooled k = 4..7).

## 4. Decision rules (fixed now)

1. **Stop rule (surface features).** On held-out problems (8 training families, k = 0), con1 must beat the trees on
   V (mean over the 8 families) by at least 2 se. If it does not: stop, report "learns no more than surface
   features", no benchmark.
2. **Refusal predictor.** If con1 beats the trees on V but not on D (mean over the 8 families, 2 se) in any of the
   three native families or overall, the result is called a refusal predictor, not physics, whatever V says.
3. **Generalisation across problem types.** On the held-out families con1 must beat (a) pref1 asked the same
   question and (b) the oracle majority on V at k = 0, by 2 se each. Otherwise the report says the forecast does
   not transfer to unseen families.
4. **Benchmark.** If rule 1 passes: con1 policy (plain v3r --open-rules, one simulator call per turn) on all 11
   families, 2 pairs (r1/r2 each), against pref1's 2 pairs (`ab_pooled.py`, paired by problem). The bar: con1 −
   pref1 feasible@20 > 0 by 2 se pooled over all families (feasible@20 = feasible at 20 simulator calls for a plain
   policy). Groups reported: native 3, replay 5, held-out 3. Also: refused share, share of applied moves that raise
   Phi, turns per episode, and the share of FORECAST-style lines in its rationales that the simulator bears out
   (forecasts.py; reported only where such lines exist).
5. **Forecaster in the loop.** If rules 1–3 pass and rule 4 fails (the forecaster is good but the policy does not
   use it): no more policy retraining. Instead a loop that calls the forecaster explicitly: per turn the policy draws
   K = 8 replies at T = 0.8, the con1 forecaster predicts each distinct move's result, the move with the highest
   predicted Phi (refusals last) is committed; one simulator call per turn. Compared with pref1 greedy at equal
   calls and with best-of-8 with the simulator (8 calls per turn, the ceiling). A variant checks the top 2 predicted
   moves in the simulator first (3 calls per turn) and is compared on the calls curve.

## 5. Improvisation (fixed now)

- **I1, one-shot physics on unseen families.** On inventory2 / molopt / thermal2 test states: con1's pooled k-shot
  gain G(4..7) in V, against pref1's gain on the same moves (difference of gains, bootstrap over states).
  Improvisation in this sense is claimed only if con1's gain is > 0 by 2 se, larger than pref1's by 2 se, and con1
  at k = 4..7 beats the nearest-probe control by 2 se (more than copying the closest probe).
- **I2, recovery after a surprise.** In benchmark episodes, a surprise is a committed move that is refused or lowers
  Phi. Recovery = the next turn's move is applied and raises Phi. con1 vs pref1 recovery rate (bootstrap over
  problems), all families and held-out families separately; claimed if con1 is higher by 2 se.
- Not measured: "held-out-family solves with move patterns absent from every training row". It needs dx38's
  original SFT rows (Orchard only) and a similarity rule over numeric arguments that we would be choosing after the
  fact; I1 is the cleaner test of the Gen-1.5 sense and is fixed in advance.

## 6. Open points from the owner discussion, decided

- **Prediction target: the full result message.** It carries every metric's change (the physics), the verdict is
  derived from it, and it is literally what a check returns, so the same skill serves the in-loop forecaster (rule
  5). Cost: 150–350 target tokens per answer; every printed digit is penalised alike.
- **Fit with rationale-before-action:** consequence rows are an extra training objective on the same weights. The
  policy rows and the served policy keep rationale-before-action; the report says so in these words.
- **Refusal-only outcome:** decided by rule 2 above, measured by D.

## 7. Resources and placement

Data on CPU (local Windows / drc / Orchard nodes); training on Orchard (`advanced` only after coordinating with the
RL session, else `preempt` with a checkpoint every step and resume); skill test and benchmark on whichever GPUs are
free after checking Orchard, drc GPU 4 and trace. Code under `scripts/con1/`; outputs in `analysis/remote/rl/con1/`
(main checkout) and drc `/storage/wxu/con1/`.

## 8. Owner revisions of 11-25 (before any result) — these override sections 1, 3, 4 where they differ

Relayed by the parent session ("Zero-shot physics reasoning comparison"), the owner's instructions:

- Answer structure, how predictions are compared and penalised, and what the model is shown change accuracy a lot;
  each is an ablation axis chosen from evidence, with short, cheap runs before the main run (section 9).
- Exact digits are the wrong measure of physics, and "close" is not understanding either (predicting no change is
  often close). Scoring is by **physical facets**, each against a **no-change baseline**: regime (refused or
  applied; feasible after; which constraints are violated after), direction (sign of each metric's change), size
  (relative change, factor-scale bins), coupling (effects on quantities other than the element moved, where the
  message exposes them). Decision weighting only for errors that flip the regime or feasibility. A profile across
  facets is reported, not one number. Classes are skewed: balanced accuracy and per-class recall, not raw accuracy;
  missed refusals and false refusals reported apart.
- **Ranking is not a target and not a selection criterion** (choosing among moves is left to later steps: preference
  rows, RL, the check loop). No ranking objective, no ranking-weighted loss. PO (section 3) is dropped as a metric.
  One diagnostic is kept, reported but never used to select: **within-state discrimination** (do predictions
  differ across the moves of one state; e.g. the true rate of 'better' among moves predicted better vs predicted
  worse, the measure on which dx38 scored 0.47 vs 0.50 on engibench).
- The stop rule (section 4 rule 1) compares con1 with the boosted trees **on the same per-move facets**, not on V or
  within-state AUC. The trees get per-metric direction and size models.
- Held-out families are touched once, at the end. Ablations select on held-out problems of training families,
  split by problem.
- Any change to the observation (what the model is shown) is a scope decision: it is reported with its measured
  effect and needs the owner's yes before the main run; the policy's prompt must then change the same way.
- Coupling: the result message carries system-level metrics only (no per-line flows, per-member stresses, per-pixel
  fields), so coupling is **not measurable with the current message**; it needs a target or observation level that
  exposes per-element quantities (a round-2 candidate, section 9).

**Facet scorer (all formats parse into the same facets):**
- Regime class per move: refused / applied-infeasible / applied-feasible. Balanced accuracy over the classes present
  in a family, per-class recall, refusal precision and recall. No-change baseline: the state's own regime (applied,
  infeasible).
- Violated-after set (applied moves predicted applied): exact-set accuracy and per-constraint crossing accuracy
  (does a constraint cross its limit in either direction). Baseline: the current violated set.
- Direction per metric line (up / down / same, same = change within 1e-5 of the value: the verbatim message prints
  6 digits, so smaller changes are not expressible in it): balanced accuracy over the
  three classes per family. Baseline: all 'same'.
- Size per metric line with a true change: right direction and predicted relative change within the same or an
  adjacent half-decade bin (factor ~3). Baseline: 0.
- Decision-weighted: feasibility-after recall and precision for moves that make the design feasible.
- Diagnostics (not selection): Phi verdict (better / worse) where values can be reconstructed (bin midpoints for the
  facet format), within-state discrimination, exact-message match.
- Group value = mean over families; se by bootstrap over states.

## 9. Ablations (fixed before any ablation result)

**Procedure.** Short LoRA runs from pref1 on consequence rows only (no policy rows), one 1-GPU run per arm, 4 arms
in parallel on one 4-GPU job; identical states and moves for every arm (paired); then each arm answers the zero-shot
questions of the held-out-problem test states of the same families. One factor changes per arm relative to the
baseline B (a small fractional design over the axes).

**Round 1 — answer structure and loss** (families with data ready: engibench, grid2op, photonics, circuit2,
frame3d2; 150 training states per family, 12 questions per state, 16 rows per optimizer step, one epoch, lr 1e-4):
- B  `verbatim`: the harness message word for word, every answer token supervised.
- M  `masked`: same text, loss only on informative tokens (after values, VIOLATED markers, feasible word, refusal
  reason; not the header, metric names, limits, before values).
- F  `facet`: verdict first, FEASIBLE AFTER, VIOLATED AFTER, then per metric direction + half-decade size bin of the
  relative change (no digits).
- D  `delta`: verdict first, FEASIBLE AFTER, VIOLATED AFTER, then per metric the signed change and the relative change
  at 3 significant digits.
(scripts/con1/targets.py renders all four from the same simulated record.)

**Selection metric (round 1 and later rounds).** Per family, the mean of three facet scores on held-out problems,
zero-shot: regime balanced accuracy, direction balanced accuracy, size share; group = mean over the round's
families. The arm with the highest score wins; if the best two are within 1 se (bootstrap over states), the one with
fewer target tokens wins. The profile (every facet, baseline, trees) is reported for every arm.

**Round 2 — what the model is shown** (levels fixed after round 1 and before its results, written here then):
candidates are raw physical state that the message or observation leaves out, never ranked or influence tables
(memory no-a-priori-influence) — e.g. grid2op per-line flows in the target (also the only way to measure coupling),
the photonics pixel grid written as coordinates or run lengths instead of the current form. Any level that changes the
observation needs the owner's yes before the main run.

**Main run** after the rounds: the chosen format and loss, all 8 training families, ~2,000 states per family, policy
rows mixed in (section 1), on `advanced` (owner: con1 has priority there). Sections 3–5 then apply with the facet
scorer replacing V / D / PO as the primary skill measures.

## 10. Results

(none yet)

## 11. Addendum 2026-11-25, 14:45 — the owner's consolidated goal (before any con1 result)

Relayed by the parent session. This merges and supersedes the earlier owner messages; where it differs from
sections 1–9, it wins.

**Goal, in priority order.** (1) The model has learned physics: it predicts the consequences of moves on facets that
need physics (regime, direction, relative size, coupling), beyond surface features and beyond refusal rules. (2) This
holds on problem types it never trained on, at k = 0 and especially after a few probes. (3) Improvisation (I1, I2).
(4) Supporting evidence only: the policy uses the skill (benchmark at equal simulator calls); feasibility is not the
bar. Not goals: exact digits, ranking, raw accuracy.

**Target axis (replaces §1's fixed choice and §6's "full message" decision).** The verbatim message is one level, and
in any verbatim level the fixed wording is masked from the loss. Round 1 levels (one 1-GPU arm each, same states and
moves): **M** verbatim with fixed wording masked; **F** facet: verdict, feasibility, violated constraints, then per
metric direction and the relative change in 1-2-5 bins (0.1, 0.2, 0.5, 1, 2, 5, ... 1000 %, about factor-2 wide), no
digits; **D** structured delta: verdict, feasibility, violated, per metric the signed change and percent change at 3
significant digits; **S** structured absolute: verdict, feasibility, violated, per metric before -> after at 4
significant digits. The question text states each level's form. The unmasked verbatim arm of §9 is dropped.
*Decision (mine):* in every level the informative tokens weigh 1 and the fixed wording (labels, metric names, limits,
before values, header) weighs 0.05, not 0: the model has to write that scaffold itself at test time and a zero weight
gives it no gradient to learn it; 0.05 removes about 95% of the scaffold's share of the loss. The trainer reports the
loss on informative tokens alone (`nll_info_token`).

**Observation axis.** How the physics and simulator state is shown is a second axis (round 2: e.g. the photonics grid
written differently, unranked element-level state). Any change needs the owner's yes with its measured effect;
influence / fix tables stay banned. Noted for the owner: the photonics benchmark observation already carries an
adjoint sensitivity map and the six pixels "whose filling/emptying lowers target_loss most"
(`dataxp/domains/photonics.py`, ~l.1680–1708), an influence table of the banned kind, inherited, not changed here.

**Metrics (replace §3's V / D / PO and §8's selection score).** Primary is a profile of facets, each against the
no-change baseline (`scripts/con1/facets.py`):
- regime: balanced accuracy over refused / applied-infeasible / applied-feasible with per-class recall; refusal
  precision and recall; feasibility-flip recall and precision; constraint crossings (recall, false-crossing rate);
- **V_bal**: the 3-way verdict (refused / worse / better by the Phi rule) as balanced accuracy with per-class recall;
  the predicted Phi is the problem's potential of the predicted values (facet level: bin geometric midpoints);
- **direction (D / N)**: per-metric sign (up / down / same) balanced accuracy;
- **size**: per metric with a true change from a nonzero value, right sign and |log2(predicted change / actual
  change)| <= 1 (within factor 2); the mean |log2| is reported too;
- **coupling**: direction and size on the metrics not directly moved. *Decision (mine):* "directly moved" = the
  move's own bookkeeping, readable without solving physics: circuit2 area, power_mw; engibench volume,
  n_components; frame3d2 mass; grid2 commitment_cost, n_committed; grid2op step, redispatch_mw, redispatch_cost,
  curtailed_mw, storage_cost, lines_out, lines_unavailable; photonics fill_fraction, n_islands,
  min_feature_violations; seir2 doses_per_capita, cost; traffic2 lanes_used; molopt heavy_atoms, mol_weight, n_rings,
  max_ring_size, rot_bonds; thermal2 cost, link_cost, sink_cost; inventory2 none. Everything else is coupled.
- Diagnostics, never in a rule: within-state discrimination (P(true better | predicted better) − P(true better |
  predicted worse)), PO, exact-message match. LambdaRank trees are dropped; the surface-feature control is the facet
  trees (`trees_facets.py`: verdict, regime, violated per constraint, direction + size bin per metric).
- Format ceilings, measured by scoring the true answers rendered in each level (4 native+replay families): selection
  score masked 1.000, delta 1.000, facet 0.996, absolute 0.990; V_bal 0.992 / 0.998 / 0.965 / 0.996 (bin midpoints
  and 4-digit rounding lose a little).

**Selection (round 1 and later rounds).** Per family the mean of regime balanced accuracy, direction balanced
accuracy, coupling direction balanced accuracy and size; group = mean over the round's families, held-out problems,
zero-shot; highest wins, within 1 se the level with fewer target tokens wins. Held-out families are not looked at
before the final skill test; splits are by problem.

**Rules (replace §4).**
1. Stop (surface features): on held-out problems of the 8 training families at k = 0, con1 beats the facet trees on
   V_bal and on direction balanced accuracy, each by at least 2 se. Otherwise stop: "learns no more than surface
   features".
2. Refusal predictor: if con1 beats the trees (held-out problems) or pref1 and the oracle majority (held-out
   families) on V_bal but not on direction balanced accuracy, the result is called a refusal predictor, for the
   held-out families as well as the training families.
3. Generalisation: on the held-out families, con1 beats pref1 (asked the same question) and the oracle majority on
   direction balanced accuracy and on V_bal, each by 2 se; a pass on V_bal alone is not generalisation. Reported at
   k = 0 and after probes (I1).
4. Benchmark: reported as supporting evidence (feasible@20 at equal simulator calls, 2 pairs per side, refused share,
   Phi-up share, recovery I2), not as a bar.
5. Forecaster in the loop: unchanged (if rules 1–3 pass and the policy does not use the skill).
I1 is judged on the k-shot gain in direction balanced accuracy and V_bal (same moves, sequential vs zero-shot).

**Round 1 as submitted.** Families engibench, grid2op, photonics, circuit2, frame3d2 (the data generated by 13:47;
engibench and photonics states come from about 15 episodes each, so the round compares formats, not final skill),
150 training states per family, 12 questions per state, 16 rows per step, one epoch, lr 1e-4, LoRA from pref1, no
policy rows. Orchard job 175877 (arms re-pointed to M / F / D / S before it started).

**Note 2026-11-25, 15:10 — parent-session review of §11 (before any result).** Three scoring fixes; job 175877 is
unaffected (it only produces predictions).
1. Bookkeeping metrics (the "directly moved" table) are readable without physics and carry trivially to new families
   ("radius up -> mass up"). Rules 1–3, I1 and the selection score use the **coupled** metrics only: coupled
   direction balanced accuracy (`cpl_dir_BA`) and coupled size (`cpl_size`). Pooled and bookkeeping direction
   (`dir_BA`, `bk_dir_BA`) are reported as a sanity check only.
2. Refusal follows legality rules, not physics. The **physics regime** term is `regime_phys` = mean of (a) balanced
   accuracy of applied-infeasible vs applied-feasible on truly applied moves (`feas_BA`) and (b) constraint-crossing
   balanced accuracy (`cross_BA` = mean of crossing recall and 1 − false-crossing rate). The verdict in the rules is
   **`V_bal_applied`**: worse vs better balanced accuracy on truly applied moves (a refusal predicted for an applied
   move counts as wrong). Refusal precision / recall and the 3-way `V_bal` stay in the profile, out of selection and
   rules (rule 2 still compares the 3-way `V_bal` with `V_bal_applied` and `cpl_dir_BA` to name a refusal predictor).
3. Photonics' observation carries the adjoint map (first-order change of target_loss per pixel flip, block sums plus
   the top-6 pixels each way), so target_loss's direction and size can be read off the map. Every rule and the
   selection score are computed twice, over all families and without photonics (`facets.py` prints both);
   photonics target_loss facets are reported separately as `adj_dir_BA` / `adj_size` under the label "adjoint map in
   observation". The observation is not changed here (the owner decides on removing the map).
Selection = mean of `regime_phys`, `cpl_dir_BA`, `cpl_size` (each family, then mean over families).
Rule 1: con1 beats the facet trees on `V_bal_applied` and on `cpl_dir_BA` (held-out problems, 8 families, k = 0, 2 se
each), with and without photonics. Rule 3: on the held-out families, con1 beats pref1 and the oracle majority on
`cpl_dir_BA` and `V_bal_applied` (2 se each). Ceilings measured by scoring the true answers in each level:
`V_bal_applied` 0.992 (masked) vs 0.956 (facet: Phi from bin midpoints); `cpl_dir_BA` and `regime_phys` 1.000 for both.

**Note 2026-11-25, 15:00 — owner decision on the photonics map (before any result).** The adjoint map stays and the
observation is not changed: a map the simulator can compute deterministically is part of that simulator's
interface; a physically reasoning model works with whatever a simulator returns (without a map it needs one more
step). So:
1. Primary rules and selection include photonics. The without-photonics numbers are a sensitivity check, not
   co-primary. Photonics is described as "simulator interface includes an adjoint map" (target_loss facets still
   listed apart as `adj_dir_BA` / `adj_size`).
2. The surface-feature control gets the same information the observation shows (`scripts/con1/obs_features.py`,
   `trees_facets.py --obs-feats`): photonics: the map's first-order change of target_loss for the move,
   `PhotonicsDomain._gain(g, pat, new)` with g = `_sensitivity(fx, pat)` and new = the materialised pattern, plus the
   pixels added / removed; engibench: the per-cell field S its observation lists (strain-energy share, |grad T|^2 or
   optical intensity, top cells + coarse map), summed and averaged over the cells the move fills and clears;
   circuit2: the simulator's own what-if rows in its observation (one-tunable x1.1 sensitivities of DC gain /
   crossover / phase margin, scaled by log(new/old)/log 1.1 for the moved tunable; the other compensation schemes;
   the cascode and nulling-resistor toggles). Checked and not given (their observations show per-element state —
   member forces, line flows, queues, costs — but no per-move effect estimate): frame3d2, grid2, seir2, traffic2,
   grid2op. Held-out families have no trees.
3. How much the map alone predicts: sign agreement of the map's first-order target_loss change with the true change
   on the photonics questions is reported (`map_sign.py`; a property of the data, not a con1 result).
Not built (undecided by the owner): an eval-only photonics condition without the map.
   Measured 11-25 15:05 on the 47 photonics test states (354 applied questions with a nonzero true change): the map's
   first-order target_loss change has the right sign for **0.684** of moves (own moves 33/48 = 0.69, legal moves
   209/306 = 0.68) and is within a factor of 2 of the true change for **0.257**. So the map alone carries most of the
   direction of target_loss but little of its size (the moves flip many pixels at once, beyond first order).

**Note 2026-11-25, 15:40 — parent review of round-1 rows (before any result).** Job 175877 is left as it is (a
format comparison on the same 750 states and moves). Two changes for the main-run rows and for scoring:
1. **Out-of-menu moves on native families.** engibench, grid2op and photonics questions came only from the policy's
   own moves and the domain's ranked menu (photonics: 28 of 32 adjoint-ranked; 59% of applied photonics questions
   are 'better'), so arbitrary and bad moves were rare there. Each native state now also gets 2 out-of-menu calls,
   src `offmenu` (`transitions.offmenu_moves`; `add_offmenu.py` adds them to states generated earlier): a verb the
   menu uses, every argument drawn at random over that argument's range across the whole menu (numbers uniform over
   the menu's span, floats widened by a quarter of the span each way; names uniform over the names in the menu; json
   lists a random non-empty subset of the items in the menu's lists; rectangle corners sorted). Some are refused, by
   a rule or by the physics, as arbitrary moves are. Their verdict mix is reported next to the menu moves. The main
   run's rows draw from own / best legal / off-menu (native) or off-grid (verb surface) / uniform legal as before.
   Round 1's question set is unchanged; the final skill test uses a question set rebuilt the same way (`queries_v2`:
   the round-1 recipe plus the off-menu moves of the native test states).
2. **Refusal split by reason.** A refusal whose reason is a simulation outcome is physics and counts in the physics
   regime: `refused-physical` = the reason matches "would end the episode", "diverge", "no AC solution", "blackout",
   "isolated load", "non-connected grid" (grid2op: the move islands part of the grid and the power flow has no
   solution; 46% of grid2op questions). Every other refusal is a rule and stays out: `refused-rule` = bad or
   out-of-range arguments (e.g. "ERODE: px must be 1..2"), cooldown / one-action-per-step ("is illegal (cooldown,
   ...)"), forced outage ("line N is under forced outage"), ambiguous redispatch ("is ambiguous: ...
   InvalidRedispatching"), a move that changes nothing ("changes nothing"), unknown elements, the fabrication rule.
   Physics regime = balanced accuracy over refused-physical / applied-infeasible / applied-feasible (true classes;
   a predicted rule refusal counts as wrong) averaged with constraint-crossing balanced accuracy; the verdict in the
   rules becomes **`V_bal_phys`** = balanced accuracy over diverged (refused) / worse / better on moves that are not
   rule-refused. Refusal recall / precision (any reason) and per-kind recall stay in the profile.

**Note 2026-11-25, 16:05 — off-menu moves drawn within each verb's validity rules (parent, before any result).** The
first version drew grid2op / molopt / engibench arguments independently, so 63% / 46% / 21% of those questions were
rule refusals (legality, not physics). Now (`transitions.offmenu_moves`): the first argument takes a value the menu
uses; a later name / json argument half the time comes from the values the menu pairs with that first argument (an
element on the chosen substation) and otherwise from all values the menu uses; numbers stay uniform over the menu's
span; any proposal the state refuses by a RULE (bad combination, cooldown, forced outage, out-of-ramp redispatch,
changes nothing) or that is already a menu move is redrawn (up to 40 tries per move). Physical refusals are kept.
Measured on the 231 native test states (Orchard job 175904; better / worse / refused-physical / refused-rule):
engibench off-menu 0.35 / 0.65 / 0 / 0 (menu 0.42 / 0.58 / 0 / 0); grid2op 0.22 / 0.51 / 0.28 / 0 (menu 0.18 / 0.56 /
0.26 / 0); photonics 0.40 / 0.60 / 0 / 0 (menu 0.62 / 0.38 / 0 / 0); molopt 0.43 / 0.57 / 0 / 0 (menu 0.43 / 0.57). Yield
2 per state except photonics (88 of 120: states whose design breaks the fabrication rule have tiny menus).
The final skill test's question set (`queries_v2`) and the main-run states use these.

**Note 2026-11-25, 16:25 — round 2 = the owner's P rows (before any result).** Round 1 (175877) still decides how the
outcome is written. Round 2 contrasts the row *structure*, on the same training states, with the encoding round 1
picks (rows are pre-built in all four encodings; the round-1 winner's set is used):
- **P** (owner's format): the state and goal exactly as the policy prompt shows them; then the model's OWN reply:
  its reasoning (loss 0), its action (loss 0: its own pick, good or bad), then `EXPECTED RESULT:` and the true
  simulator result of that action (the header and the informative tokens weigh 1, fixed wording 0.05). Replies:
  pref1's first reply of each turn in the chk1 collection (pref1 at T = 0 on training-pool problems, the reply to the
  turn's prompt before any check result), labelled with the result `transitions.py` simulated for that own move.
  Native families only in round 2 (chk1 covers engibench, grid2op, photonics); for the other families the main run
  would use dx38's replay rows, which needs their states rebuilt from the rows (not done yet).
- **Qown**: the same states, ONE question about the same own move, the same result: identical supervised outcomes,
  only the structure differs.
- **Q12**: the round-1 question rows (12 questions: own / best legal / off-menu / uniform legal) on a twelfth of the
  same states: the same number of supervised outcomes as P (matched outcome tokens), more diverse moves.
- **PQ**: P rows + Q12 rows together (the main-run candidate if both structures help).
400 states per native family (selection seeded; the selected states first got their off-menu moves), rows per step
16, one epoch, lr 1e-4, LoRA from pref1, one 1-GPU arm each.
**Both test forms for every arm:** the question form (the skill test, `queries_v2`: the round-1 recipe plus the
native test states' off-menu moves; held-out problems only) and the **P form**: the test state's policy prompt, the
model's own reply sampled at T = 0, its own action simulated, its EXPECTED RESULT block scored with the same facets
(`p_client.py` on GPU, `p_score.py` + `facets.py` on CPU). The P form also reports the chosen actions' verdict mix.
Multi-turn P rows (each later turn showing the real result) are not built: a turn's prompt is a full observation of
4–12k tokens, so an episode row would run past what one H100 trains on.
The harness already ignores an `EXPECTED RESULT:` block after the action: it reads the last `<action>` line of the
answer (checked on Orchard with `dataxp.ownrows.common.parse_reply`); benchmark runs need no change.
Selection between the round-2 arms uses the same facet score (§11 notes), in the question form; the P form is
reported beside it. The structure (P, Q, or PQ) and the encoding chosen go to the main run.

**Note 2026-11-25, 16:40 — round 2 (P rows) WITHDRAWN; real-task check added (before any result).** The 16:25 note
above is withdrawn: the owner was asking why we chose the question format, not proposing theirs. No round-2 rows or
job remain (the row build, Orchard 175908, was cancelled and its outputs deleted; `build_prows.py`, `p_client.py`,
`p_score.py` stay in the repo unused). The owner's actual question: does a model trained on question rows still do
the real task (the policy prompt, a reply that gives a rationale then an action)? Round 1 alone does not measure this,
so round 1 now also:
1. keeps every arm's final adapter (adapter files only, ~1.9 GB each; copied while /project/community has more than
   4 GB free; the optimizer state stays on the node), and
2. runs each arm on the real task in the same job, on the servers already up: the benchmark harness and settings of
   pref1's baseline (`dataxp_policy_eval_v3r.py --open-rules`, thinking on, T = 0, 20 turns, one simulator call per
   turn, 2 runs per problem) on the round's 5 families (engibench, grid2op, photonics, circuit2, frame3d2; 60
   benchmark problems each), episodes resumable by key. Reported per arm against pref1's two pairs on the same
   families (`bench_format.py`, `ab_pooled.py`): whether the reply format holds (rationale before the action, action
   parse rate, result-message or answer-format text leaking into the reply, completion tokens) and feasible@20 at equal
   simulator calls. This tells whether training on consequence questions damaged or helped the policy before the
   main run commits to a row mix; it is supporting evidence, not a selection rule.

**Note 2026-11-25, 16:55 — two stages in the published order (owner-approved, before any result).** Following Early
Experience (arXiv 2510.08558: world-model rows first, then policy training; its comparison is imitation learning vs
implicit-world-modelling-then-imitation at an equal update budget) and Dyna-Think (arXiv 2506.00320, the same order),
as cited by the owner:
1. **Round 1** (175877): training and skill questions unchanged. Its real-task run is cut to a damage check: 1 run per
   problem on 20 benchmark problems per family (the first 20 by id; `data/bench_sub20.jsonl`), 5 families, 100 episodes
   per arm; reported: format holding (rationale before the action), action parse rate, leakage, feasible@20. These
   arms have no policy stage, so this only shows how much damage prediction-only training does; it is not the test.
2. **Stage 2**: the winning round-1 arm's adapter continued on policy rows only (the open-rules restated dx38 replay
   rows, tier `ownrows_own`, rationale then action, 8 training families: 8,753 rows, 60 more held out as dev), one
   epoch = 137 steps of 64 rows, lr 1e-4 (dx38's recipe; `configs/rl/con1_policy_drc.yaml`).
   **Control**: the same rows, steps and lr from pref1 with no consequence stage (the papers' comparison; without it a
   gain could come from the policy rows alone). The control does not depend on round 1 and started at once on drc GPU
   4 (run `/storage/wxu/con1/stage2/runs/control_pref1_policy`, 15:49).
   Both are tested on the real task with pref1's benchmark settings on held-out problems, plus the skill questions;
   held-out families stay untouched until the final test.
3. **Main run**: the joint mix of section 1 (consequence rows + 1/3 policy rows) is replaced by the two stages —
   consequence rows (chosen format, 8 families) then the same policy rows — against the same control (one control
   run serves both the pilot and the main run: same rows, steps and lr from pref1).

**Note 2026-11-25, 17:10 — placement corrections (no result yet).** (a) Round 1's first job (175877) waited two hours
and then failed in its first minutes: the repo copy uploaded from the Windows checkout had CRLF line endings, which
bash rejects (`set: pipefail\r`). Every text file of the Orchard copy is now LF (checked with `bash -n`), and round 1 is
resubmitted unchanged as **175928**. (b) The control could not run on drc: the box has 93 GB of RAM with about 18 GB
free, and the kernel killed the 27B trainer (SIGKILL) at its first step after loading. The control now runs on
Orchard in the same job as stage 2 (**175931**, `advanced`, after round 1): two 2-GPU trainings side by side on one
node (the same hardware for both arms, the same rows / steps / lr), then the skill questions and the real task for
both. The policy rows are rebuilt on Orchard from the same chk1 file with the same seed (175930).

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

## 8. Results

(none yet)

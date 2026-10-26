# Leave-families-out slope test — pre-registration (DRAFT, awaiting the owner's yes)

**Question.** Does held-out-domain performance rise with the number of training domain families
when the rows are the model's own simulator-accepted actions (the recipe that kept transfer in
the SFT probe: own_act 92 vs act 74 on thermal2)? This is the cheap test that decides whether
the FLAN-style multi-domain bet (tens of families) is worth building.

**Model.** Qwen3.8-27B, LoRA r64, lr 1e-4, 2 epochs, effective batch 64, max 16,384 tokens,
`data.target_fn=assistant_after_think`, the collaborator's recipe (`scripts/train_sft.py`), one
training seed (42) per arm. Trained on drc GPU 4 (or Orchard preempt with per-step checkpoints).

**Families.** Training pool (8): circuit2, frame3d2, grid2, seir2, traffic2 (v2) and grid2op,
photonics, engibench (native). Held out entirely (3): thermal2 and inventory2 (comparable with
every earlier result) and molopt (the structurally farthest family: molecules, not engineering).
Truss is excluded (different harness).

**Rows.** own_act form only (assistant turn = `<action>VERB(...)</action>`, empty thinking):
states from the corpora's repair phase, K=6 samples per state from the un-tuned model (thinking
off, native instruction + grammar), a sample kept if its pick is at or above the 90th Φ
percentile of the legal set or equals the teacher's move (`dataxp/ownrows/`). Problems used for
rows are disjoint from the evaluation problems (last 60 of each variant's 130 are reserved for
evaluation; v2 evaluation problems are `problems_v2.jsonl` as before).

**Arms (fixed total budget, 500 rows each, so family count is the only thing that varies).**
Row availability after sampling (multitool spelling, K=4, rule ≥ 90th percentile or teacher):
seir2 369, frame3d2 309, traffic2 164, circuit2 145, grid2 83, grid2op 71, engibench 39,
photonics 11 (1,191 in all; 38% of replies hit the 2,000-token cap, the main yield limiter).
Allocation rule for an arm with k families: each family contributes min(available,
ceil(500 / k)); any shortfall is filled from the arm's largest families; rows are drawn with a
fixed seed and listed in the arm's manifest.

| arm | families | subsets |
|---|---|---|
| k=2 | {seir2, frame3d2}, {seir2, circuit2} | the two pairs with ≥ 500 rows available (seed-fixed draw) |
| k=4 | {seir2, frame3d2, traffic2, circuit2}, {frame3d2, traffic2, circuit2, grid2op} | two subsets, one with a native family |
| k=8 | all 8 | one (natives contribute all 121 of their rows) |

Five training runs. Rows: K=4 samples per state (changed from 6 on 2026-10-23 for throughput:
~27 replies/min on the shared GPU), 2,000-token cap at sampling, `own_act` form. Native-family
yields are low under the acceptance rule (engibench 39 rows, photonics 11; grid2op pending), so a
native family contributes what it has (≤ 100 rows) and the fixed 800-row budget is filled from
the v2 families; per-family row counts are part of the report. Two spellings exist
(`datav4_multitool`, `datav4_set_parameter`); the slope test runs on multitool first, and the
spelling contrast (same states, same rule) is a separate pre-registered comparison once both
row sets exist. Reference arms: the un-tuned base (thinking off, V2/native instruction) and
act (teacher actions, 578 rows, from the probe) are already evaluated on thermal2/inventory2;
base is evaluated on molopt once.

**Evaluation.** Collaborator's harness (dataxp_policy_eval_v3, T=0, 20 turns, 2,048-token
cap): held-out thermal2 60 problems × 2 runs (120), inventory2 60, molopt 60 × 2 (120); plus the
in-domain check on the trained families (60 problems each, one run) for the k=8 arm and one k=2
arm.

**Readouts (pre-registered).**
- Primary: pooled held-out feasible@12 (thermal2 + inventory2 + molopt, 300 episodes per arm) as
  a function of k; the k=8 − k=2 difference (k=2 averaged over its two subsets), paired by
  problem, problem-clustered bootstrap 95% interval.
- Secondary: each held-out domain separately; held-out vs the un-tuned base (is any arm above the
  base?); in-domain feasible@12 for k=8 vs k=2 (what diversity costs in-domain).
- Behaviour: parse rate, A-B-A-B episodes, undo share, reply tokens per arm.

**Decision rule.** "Family count is a lever" if the k=8 − k=2 pooled held-out difference is
positive with the clustered interval excluding 0 and the k=4 arms lie between (monotone).
"Not a lever at this scale" if the interval includes 0 or the ordering is not monotone — then the
30-family build is not funded on this evidence. Any arm above the un-tuned base on pooled
held-out is reported separately as the first transfer gain of the project, whatever the slope.

**Cost.** Rows: ~2,000 states × 6 samples at 27B on the Blackwell, ~1 day. Training: 5 runs ×
~1.5 h. Evaluation: ~3,600 episodes, ~15 h on the Blackwell. Total about 3 days wall.

**Not decided by this test.** Whether rationales help (closed by the probe); whether RL helps;
whether evaluation budget matters (the own arms' truncation is avoided here by the own_act form).

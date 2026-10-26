# SFT probe: does training on outcome-informed rationales (rat2) help? (pre-registration, 2026-10-18)

Written before any rationale for this probe was rendered or any adapter trained. Changes after
results are logged at the end with the date.

## Question

`docs/synthetic_cot_gold_probe.md` found that rationales written from the gold action **and its
actual outcome** (prompt `rat2`, `rat` for grid2) are true and sensible in 83% of traces (double
Claude audit). Does training on them help the policy, compared with training on the same gold actions
without reasoning?

Context (collaborator, `shared/handoffs/dataxp_progress_20261012.en.md`):
- DX-28 trained no-prediction rationales written **without** the outcome on 192 v2 rows: thermal2
  26/120 vs 51/120 for action-only, circuit2 7/60 vs 23/60.
- On v1, a shuffled-rationale control attributed about half of the loss to the extra supervised text.
  This has not been separated on v2.
- Their open question Q4 asks whether rat2 helps; it is untested.

## Design

**Rows.**
- Source: the collaborator's DX-26 control tier `train_supp1teacher.jsonl`. It has 2,241 phi_select
  repair-phase rows from run2_supp problems 0130-0259 over 5 domains: circuit2, frame3d2, grid2,
  seir2, traffic2.
- Prompts: served-prompt identical to their eval (instruction V2, sha `3d4867180337`).
- Sampling, per domain:
  - **train:** whole problems at random (seed 0) until ≥120 rows;
  - **dev:** further problems until ≥12 rows, disjoint from train.

**Rationales.**
- `rat2` per `dataxp/gold_probe.py` (`rat` for grid2), with Qwen3.8-27B, thinking on. K=2 samples per
  row; up to 2 more samples only for rows with no good sample after the K=2 audit rounds.
- Train 623 rows, dev 70 rows. Rows were drawn and states replayed with the collaborator's
  `dataxp_code_v2` snapshot; the recorded and replayed metrics match on every row.
- The renderer sees:
  - the served observation;
  - the gold action;
  - the earlier steps with their effects;
  - for information only, the gold action's replayed outcome and the later gold steps.
- **Kept sample:** the first sample that passes the deterministic checks:
  - the action equals gold;
  - numbers are grounded in the observation;
  - claims agree with the replayed outcome;
  - the claims block and the length are within limits.

  It must also be judged good by one Claude auditor (same rubric as the gold probe). If it is not,
  the next passing sample is tried.
- Rows with no good sample are dropped from **every** arm.
- The claims block is stripped. The rationale goes in the assistant's `reasoning_content` (think
  block), as in DX-27/28.

**Arms.** Identical rows, prompts, recipe and step count; only the think block differs:

| arm | think block |
|---|---|
| `act` | empty (action only) |
| `rat` | this row's rationale |
| `shuf` | a rationale from another row of the same domain and a different problem (derangement, seed 0) |

**Training.** The collaborator's recipe (`repo_dx25/scripts/drc_train_dataxp.sh`):
- LoRA, lr 1e-4 cosine, warmup 0.03, 2 epochs, effective batch 64, max_seq_len 16,384;
- target `assistant_after_think`;
- one seed, DRC Blackwell.

**Evaluation.** The collaborator's DX-24 harness (`dx_eval.sh`, `dataxp_policy_eval.py episodes`):
- same 420 problems (`problems_v2.jsonl`);
- `dataxp_code_v2_w1fix`, instruction V2, `--param-line`;
- thinking-on serving, T=0, 20 turns.
- Domains:
  - the 5 trained domains, one run each (60 problems per domain);
  - held-out thermal2, two runs (120 problem-runs).

## Pre-registered readouts and decision

**Primary.** `rat − act` feasible@12, paired, with a bootstrap 95% CI and exact McNemar:
- **(P1)** pooled over the 5 trained domains (300 problems);
- **(P2)** thermal2 (120 problem-runs).

**Verdict.**
- **helps:** the CI is above 0 on P1 or P2, and the other is not significantly negative.
- **hurts:** the CI is below 0 on P1 or P2, and the other is not significantly positive.
- **inconclusive:** otherwise.

Following the collaborator's noise rule, a thermal2 difference under 15 episodes (/120) is not
interpreted.

**Secondary (reported, not gated):**
- `shuf − act` (cost of the extra supervised text) and `rat − shuf` (content);
- @20;
- share of moves undoing the previous move;
- thermal2 sink relocations per 100 steps, and episodes with any relocation;
- median reply tokens and parse rate;
- objective at first feasibility on problems both arms solve;
- share of replies opening by blaming the last change (the DX-28 routine).

**References (not re-run, from the collaborator's evaluations on the same problems):**
- reasoning base: thermal2 101/120;
- terse base: 71;
- DX-24 action-only on 2,380 rows: 73;
- DX-28 cot: 26; DX-28 act: 51.

## Results (2026-10-19)

**Verdict (pre-registered rule): hurts.** P1 is significantly negative and P2 is not
significantly positive.

### Primary: rat − act, feasible@12, paired

| | rat | act | diff | 95% CI | McNemar p |
|---|---|---|---|---|---|
| P1: 5 trained domains (/300) | 156 | 186 | −0.100 | [−0.153, −0.047] | 4.4e-4 |
| P2: thermal2, 2 runs (/120) | 76 | 74 | +0.017 | [−0.067, +0.100] | 0.85 |

The thermal2 difference (2/120) is under the noise rule and is not interpreted.

### Secondary (not gated)

| comparison | P1@12 | P2@12 | P1@20 | P2@20 |
|---|---|---|---|---|
| rat − act | −0.100 | +0.017 | −0.077 [−0.13, −0.02] | +0.050 |
| shuf − act | −0.333 [−0.39, −0.27] | −0.458 [−0.55, −0.38] | −0.357 | −0.492 |
| rat − shuf | +0.233 [+0.18, +0.29] | +0.475 [+0.38, +0.58] | +0.280 | +0.542 |

Feasible@12 by arm and domain:

| | circuit2 | frame3d2 | grid2 | seir2 | traffic2 | P1 /300 | thermal2 /120 |
|---|---|---|---|---|---|---|---|
| act | 29 | 33 | 28 | 42 | 54 | 186 | 74 |
| rat | 16 | 23 | 23 | 39 | 55 | 156 | 76 |
| shuf | 5 | 7 | 13 | 29 | 32 | 86 | 19 |

Behaviour (P1 / thermal2):

| | act | rat | shuf |
|---|---|---|---|
| undo share of moves | 0.14 / 0.10 | 0.21 / 0.13 | 0.19 / 0.36 |
| median reply tokens | 22 / 21 | 241 / 243 | 255 / 264 |
| parse rate | 1.00 / 1.00 | 1.00 / 0.935 | 0.994 / 0.998 |
| replies opening by blaming the last change | – | 0.4% / 0.6% | 0.1% / 0% |
| thermal2 episodes with a sink relocation (/120) | 12 | 43 | 13 |

Further readouts:
- **Objective at first feasibility**, on problems both arms solve: median ratio 1.00 for every
  pair.
- **thermal2 parse failures for rat** are all `unknown_parameter` (80 turns): parameter names
  thermal2 does not have.

### Reading

- **Content is used.** Matched rationales beat mismatched ones by +0.23 (P1) and +0.48 (thermal2)
  in every domain.
- **Supervising about 270 reasoning tokens per row costs a lot on its own.** With content that
  does not belong to the row, P1 falls by a third and thermal2 by almost half. True rationales win
  back about 70% of the P1 loss and all of the thermal2 loss, but on the trained domains they
  still end 0.10 below action-only.
- **No routine collapse as in DX-28.**
  - The opening "most severe / binding constraint" appears in 56% of training rationales and 58%
    of rat's eval replies; DX-28 went from 58% to 99.9%.
  - Blame-the-last-change openings are under 1%.
  - Thermal2 does not collapse: 76 vs 74, against DX-28's 26 vs 51.
  - rat relocates the heat sink in 43/120 episodes, closer to the reasoning base (31) than to
    action-only (12).
- **Where it loses on the trained domains: oscillation.** Episodes with an A-B-A-B flip on one
  knob: rat 84/360, act 53/360 (circuit2 30 vs 18, seir2 13 vs 0). A typical circuit2 failure
  alternates Cc 0.294 ↔ 0.588 for eight turns, fixing GBW and then phase margin in turn.
- **rat's own test-time predictions** ("I expect X to rise") match the outcome 83% of the time
  (heuristic parse; circuit2 88%, thermal2 67%). Wrong predictions are not the main failure.
- **Caveats:**
  - one seed, 578 rows;
  - kept rows are biased toward explainable gold moves (frame3d2 99/122);
  - act on 578 rows reaches 0.62 on P1, against 0.82 for DX-24 on 2,380 rows.

Files on drc are in `/storage/wxu/sftprobe/eval/` (`r1/`, `r2/`, `report_final.json`).
Adapters are in `/home/wxu/sftprobe_ckpt/sftprobe_{act,rat,shuf}/final`.

## Validation matrix v2 (pre-registered 2026-10-19, before any of these runs)

Question: does any synthetic-reasoning recipe beat action-only SFT on the same rows, and is the
2026-10-19 result stable across training seeds? Everything below uses the same 578 / 69 rows, the
same recipe, the same harness and the same readouts as above unless stated.

**Arms.**

| arm | think block | action | what it measures |
|---|---|---|---|
| act_s1, act_s2, rat_s1, rat_s2 | as act / rat | gold | training-seed replicates (seeds 1, 2; the originals are seed 42) |
| rat3 | search-shaped rationale (`SYSTEM_RAT3`): violations, 2–4 concrete options each with predicted effects derived from the observation's own numbers, the decision by net | gold | option 3: a rationale that carries the comparison, not just the justification |
| own | the model's own written reasoning (thinking off, `own_nothink`, ≤250 words, K=6 at T=0.6) | the model's own move | self-generated reasoning selected by Φ (the STaR / expert-iteration recipe, round 0) |
| own_act | empty | the same own move | the action-only control for `own` |
| base_think, base_nothink | – | untuned Qwen3.8-27B, thinking on / off | the reference the SFT arms are measured against on these problems |

**rat3 rows.** Rendered for the 578 + 69 kept rows only. A sample is kept if it passes the
deterministic checks and one Claude auditor with a 5-criterion rubric (the gold-probe four plus
`comparison_ok`: every option's predicted effect is derived from an observation number and the
made change wins the comparison as written). Deterministic checks: action = gold; claims agree with
the replayed outcome; **every listed alternative is replayed in the simulator and its predicted
directions must match**; a leak check replaces the grounding check (derived numbers allowed; any
number that matches an outcome or later-step value and nothing in the observation fails). Up to 4
samples per row; a row with no good sample keeps its rat2 rationale (count reported) so the rows
stay identical to act.

**own rows.** Among a row's parseable samples, the pick with the highest Φ percentile among the
enumerated legal moves at that state (gold_probe's PhiEvaluator and seeds). The row is kept if that
percentile is ≥ 90 or the pick is the gold move; otherwise it is dropped from both `own` and
`own_act`. The share kept, the mean percentile and the share equal to gold are reported.

**Readouts.**
- Seeds: `rat − act` @12 on P1 and P2 per seed, and the per-seed P1 scores of act and rat (range
  = seed noise). The 2026-10-19 verdict stands if the P1 difference has the same sign in ≥ 2 of
  3 seeds and the pooled paired CI (problems × seeds, bootstrap over problems) excludes 0.
- rat3: primary `rat3 − act` @12 on P1 and P2, same verdict rule; secondary `rat3 − rat`.
- own: primary `own − own_act` @12 on P1 and P2, same verdict rule; secondary `own − act` and
  `own_act − act` on the common rows' problems (all 420 eval problems are common; the rows differ).
- References: base_think and base_nothink on the same 420 problems (thermal2 twice), reported.
- inventory2 (second held-out domain, 60 problems, one run) for every arm, reported, not gated.
- The behaviour readouts above for every arm.

**Headline rule.** A recipe "works" if its primary comparison is `helps` by the original rule. If
none does, synthetic reasoning as SFT data is reported as not working at this scale, with the
seed range as the uncertainty.

**Compute.** Training on drc GPU 4 (and on Bridges-2 H100s if the environment there reproduces the
smoke run); evaluation on drc with the collaborator's harness. Renders on drc's Ada cards.

**Change logged 2026-10-19 15:10, before any rat3 sample was audited:** rat3 is rendered with
thinking off (written reasoning only, 4k-token budget). With thinking on, the option-comparison
prompt ran past 10k thinking tokens per sample on the Ada server (12 in flight, none finished in
12 min), which would have taken over 10 h for one pass. rat2 was rendered with thinking on, so the
rat3 vs rat comparison also differs in generation mode; the audit pass rate of rat3 is reported
next to rat2's so the reader can see whether thinking off cost quality.

**own_cands selection, change logged 2026-10-20 18:30 (before training).** The model did not keep to
the "under 250 words" instruction: the written reasoning has median 604 words (p10 336, p90 1,253).
The build script's default 320-word cap would have kept 140 of 578 rows although 645 of 647 rows
have a sample at or above the 90th Φ percentile. The cap is raised to 900 words (drops the
run-on tail only). Consequence: `own_cands` supervises about 1,000 reasoning tokens per row, about
4× `rat`; `own_cands − own_cands_act` stays the matched comparison, and the comparison with `rat`
is confounded by length. A compressed variant (gold_probe `compress`, 80–200 words, same action)
is added as a secondary arm `own_cands_short` if time allows.

**Compute change, 2026-10-20 evening / 2026-10-21 00:20.** Bridges-2 was dropped (queue slipped
to Oct 22, and the fla/triton build that drc needs would not have run on H100 unchanged). The
own_cands / own_cands_act and own / own_act arms are trained on Orchard (`preempt` partition,
H100 80 GB, one arm per GPU), same scripts, config and library versions as drc, then copied to drc
for evaluation; nothing is evaluated on Orchard. The own / own_act job was preempted at step
11/16 and, because the launcher saved no intermediate checkpoints (`save_steps=1000`, see 10-20
10:05) and deleted the run directory on start, it restarted from scratch. The launcher now saves a
checkpoint after every optimizer step (`sft.save_steps=1`, `save_total_limit=1`; adapter +
optimizer, 5.3 GB) and a relaunch resumes the newest checkpoint through `train_sft.py`'s
fingerprint-checked auto-resume; `sft.eval_steps=4` is passed as an integer (the same schedule as
the inherited 0.25 on a 16-step arm) so that `load_best_model_at_end` stays off and the final
adapter is still the last step's weights, as in every other arm. The resume path is rehearsed on a
4-step smoke run (kill after checkpoint-2, relaunch) before it is relied on. The own / own_act run
in flight (restarted 23:52) cannot checkpoint; if it is preempted once more it restarts under the
new launcher.

*Outcome (2026-10-21 03:24).* The rehearsal relaunch resumed at step 3 from checkpoint-2. The real
job was preempted twice more (00:39 at step 2, 01:04 at step 3) and resumed from checkpoint-2 and
checkpoint-3 each time, no step re-run. After the fourth preemption the submission wrapper held the
job (its restart cap) and a plain `--requeue` resubmission on `preempt` had an estimated start of
07:18, over 5 h away, so by the owner's rule the remainder ran on `advanced` (2 GPUs, 84 min,
resumed from checkpoint-3). own: 16 steps, final eval loss 0.227; own_act: 16 steps, 0.108.
Adapters copied to drc (`/storage/wxu/sftprobe_ckpt/sftprobe_own{,_act}/final`, sha256 verified
5520a5a7… and 3ec53169…). Orchard GPU-hours used so far ≈ 16 of the 100 allowed.

**Preliminary readout, 2026-10-21 10:31 (rat3 complete; own_cands r1 complete, its r2 and its
matched control own_cands_act still running; not the final table).** feasible@12, P1 n=300 paired
by problem; P2 thermal2 n=120 for rat3/act/rat, n=60 so far for own_cands.

| A − B | P1 A vs B | diff [95% CI] | P2 | verdict |
|---|---|---|---|---|
| rat3 − act | 151 vs 186 | −0.117 [−0.170, −0.063] | 73 vs 74, −0.008 | hurts |
| rat3 − rat | 151 vs 156 | −0.017 [−0.060, +0.027] | 73 vs 76 | inconclusive |
| own_cands − act | 119 vs 186 | −0.223 [−0.280, −0.170] | 17 vs 37 (n=60) | hurts (pending r2) |
| own_cands − rat | 119 vs 156 | −0.123 [−0.173, −0.073] | 17 vs 38 (n=60) | hurts (pending r2) |
| **own_cands − own_cands_act** (primary) | 119 vs 191 | −0.240 [−0.293, −0.190] | 33 vs 71 /120, −0.317 [−0.408, −0.225] | **hurts** (final, 14:13) |
| own_cands_act − act | 191 vs 186 | +0.017 [−0.027, +0.063] | 71 vs 74, −0.025 [−0.092, +0.042] | inconclusive (null; final) |

(14:13, final for the own_cands pair.) own_cands − act on thermal2 n=120: 33 vs 74, −0.342
[−0.433, −0.250]. Report: `eval/report_owncands_final.json`.

**Seed 1, r1 complete 2026-10-22 06:08 (`eval/report_prelim_seeds1.json`; thermal2 second run
pending).** rat_s1 − act_s1: P1 153 vs 190, −0.123 [−0.177, −0.073], p<0.001; thermal2 (n=60) 35
vs 36, −0.017; inventory2 4 vs 10, −0.100 [−0.200, 0.000] → hurts, same as seed 42 (rat − act
−0.100). Behaviour rat_s1: A-B-A-B 71, undo 0.20, median reply 249 tokens; act_s1: 57, 0.16, 23.
Per-domain rat_s1: circuit2 17, frame3d2 23, grid2 21, seir2 37, traffic2 55.

**Queue change, 2026-10-22 09:40 (no change to what is measured).** base_think at the harness's
20,000-token thinking budget completed one episode in 44 minutes with 8 streams (that turn alone
4,420 tokens), i.e. about 30 h for the 420 episodes, and it was ahead of the seeds' thermal2 second
runs and the 8,192-cap sensitivity runs. Order is now: rat_s1 / rat_s2 r2; own_cands and own at
8,192 tokens (r1 + r2); base_think r1 and r2 last, with 12 streams, full budget. The partial
base_think file is discarded.

**Seed 2, r1 complete 08:40 (`eval/report_prelim_seeds2.json`; thermal2 second run pending).**
rat_s2 − act_s2: P1 142 vs 176, −0.113 [−0.173, −0.057], p<0.001; thermal2 (n=60) 39 vs 39;
inventory2 11 vs 8 → hurts. Seed range of the rat − act gap: −0.100 (seed 42), −0.123 (seed 1),
−0.113 (seed 2). rat_s2 behaviour: A-B-A-B 71, undo 0.19, median reply 249; act_s2: 47, 0.15, 23.
Per-domain rat_s2: circuit2 15, frame3d2 20, grid2 19, seir2 36, traffic2 52.

**own pair final, 2026-10-22 02:48 (`eval/report_own_final.json`).** own − own_act (primary):
P1 140 vs 163, −0.077 [−0.130, −0.023], p=0.007; thermal2 (n=120) 91 vs 92, −0.008 [−0.108,
+0.084]; inventory2 19 vs 9, +0.167 [+0.050, +0.283], p=0.013 → **hurts** by the rule (P1
negative, P2 null); the inventory2 gain is the first held-out-positive signal in the matrix and is
not part of the rule. own_act − act: P1 163 vs 186, −0.077 [−0.127, −0.027], p=0.004; thermal2 92
vs 74, +0.150 [+0.083, +0.225], p<0.001. own − act: P1 −0.153 [−0.213, −0.097]; thermal2 91 vs 74,
+0.142 [+0.050, +0.233], p=0.005. own − rat: P1 −0.053 [−0.103, −0.003]; thermal2 +0.125 [+0.017,
+0.233]. own − own_cands: +0.070 / +0.483 / +0.117 (helps).
own_act behaviour: A-B-A-B 68 episodes, undo 0.22, median reply 21 tokens (the rat-like
oscillation, from action-only supervision on the model's own picks). Per-domain own_act: circuit2
22, frame3d2 23, grid2 25, seir2 45, traffic2 48. Report `eval/report_prelim_ownpair.json`.

**own r1 complete 23:23 (own_act and the thermal2 second runs still to come).** own vs act: P1
140 vs 186, −0.153 [−0.213, −0.097]; thermal2 (n=60) 45 vs 37, +0.133 [+0.017, +0.250], p=0.057;
inventory2 19/60. own vs rat: P1 −0.053 [−0.103, −0.003], thermal2 +0.117 (n.s.). own vs
own_cands: +0.070 P1, +0.467 thermal2, +0.117 inventory2. Per-domain own: circuit2 15, frame3d2
17, grid2 18, seir2 45, traffic2 45. Behaviour: parse rate 0.46 (54% of turns at the 2,048 cap,
no action), median reply at the cap, A-B-A-B 3 episodes, undo share 0.01. Reading: on the domains
where the base model is already strong (seir2 47, thermal2 51) own sits at base level; where
action-SFT gains (circuit2, frame3d2, grid2, traffic2) own does not, and half its turns are lost
to the cap. Report `eval/report_prelim_own.json`.

**Reference arm base_nothink (un-tuned Qwen3.8-27B, thinking off, 1,024-token cap), r1 complete
18:36.** feasible@12 per domain (of 60): circuit2 14, frame3d2 23, grid2 23, seir2 47, traffic2
46 (P1 153/300), thermal2 51, inventory2 16. Against the tuned arms: act 186/300 in-domain but
thermal2 37 (act_s1 36, act_s2 39), inventory2 act_s1 10 / act_s2 8; rat 156 and rat3 151 in-domain,
thermal2 38 / 37, rat3 inventory2 12; own_cands_act 191 / 35 / 8; own_cands 119 / 17 / 12. Reading:
action-SFT on the five training domains gains about +33/300 in-domain (circuit2 +15, frame3d2
+10, grid2 +5, traffic2 +8, seir2 −5) and loses about 14/60 on held-out thermal2 and 7/60 on
inventory2 relative to the un-tuned model; every tuned arm is below the base out of domain, and the
rationale arms are at base level in-domain. This is the known pattern from the full-corpus SFT
(artifact "Distilling the Design Search", 2026-09-11), replicated here at 578 rows; it is not a new
finding. No full-corpus model was retrained for this probe: `act` is the probe's matched 578-row
control (pre-registration of 2026-10-18), and `base_nothink` is the un-tuned model. base_think
(thinking on, 20,000-token cap) is still to run (chain3). The base arms were not part of the rat2
probe's verdict rule; they are reported as pre-registered references in the v2 matrix.

(12:02.) Held-out inventory2 (P3, n=60): own_cands 12, own_cands_act 8 (+0.067 [−0.03, +0.17],
p=0.34). own_cands_act − act null means the model's own picks (88% gold, mean Φ percentile 99.6)
supervise as well as the gold actions; the reasoning text is where the matched pair separates.

Behaviour (P1): own_cands median reply 1,826 tokens (rat 241, rat3 317, act 22); A-B-A-B
episodes own_cands 14, rat 73, rat3 71, act 44; undo share own_cands 0.04, rat 0.21, act 0.14.
Seeds so far: act_s1 190/300, act_s2 176/300 (act 186). Per-domain P1 for own_cands vs act:
circuit2 14/29, frame3d2 13/33, grid2 19/28, seir2 25/42, traffic2 48/54.

*Generation budget (found 2026-10-21 11:10).* The harness caps tuned thinking arms at 2,048
generated tokens per turn (`dx_eval.sh` MAXT; the un-tuned base-think reference gets 20,000).
own_cands hits the cap on 2,996 of 6,070 turns (49%; `finish=length`, no `<action>`, the turn is
a no-op), rising from 24% at turn 0 to 65% at turn 19; its uncapped replies have median 1,117
and p90 1,824 tokens, against a supervised median of about 1,000. rat3 caps on 4% of turns, rat on
0%. The stored reply text begins "Let's evaluate the candidates: 1. … 2. …", i.e. the model
reproduces the four-candidate hint list that was in the render prompt but is absent at inference,
inventing candidates each turn. The pre-registered readouts stay at the harness's budget; a
secondary, non-pre-registered budget-sensitivity run of own_cands at 8,192 tokens (r1 + r2,
`eval/r1_cap8k`, `eval/r2_cap8k`) is queued after everything else. `dx_eval.sh` now
reads the cap from `THINK_MAXT` (default unchanged, 2,048). 14:55: the `own` arm caps the same
way (first 11 episodes: 57 of 108 turns at the cap, supervised median about 700 tokens), so the
sensitivity run covers both own arms (own_cands then own, each r1 + r2). 19:05: the Ada job
finished (act_s2 and base_nothink, r1 + r2); it now runs act_s1 r2, and the Blackwell follow-up is
one chain (drc j1459): rat_s1, rat_s2, base_think r1; their r2; then the 8,192-cap runs.

**Evaluation schedule change, 2026-10-21 04:40 (no change to what is measured).** The Ada server
(TP4 on four RTX 4000) evaluates thinking arms at ~0.25 episodes/min (rat_s1 would have taken
~30 h), against ~0.9–1.5/min on the shared Blackwell. Its job was stopped at rat_s1 88/420 and
relaunched on the non-thinking arms only (act_s2, base_nothink; r1 then r2). The Blackwell queue
after the own pair is rat_s1 r1, rat_s2 r1, base_think r1, then r2 for rat_s1, rat_s2, act_s1,
base_think. The eval writes each arm's episodes file from scratch, so no arm is ever run by two
processes; the partial rat_s1 file is discarded. Expected completion: own_cands pair by midday,
own pair tonight, everything else by 2026-10-22 evening.

**rat3 closed at three samples, 2026-10-20 14:10 (change from "up to 6", before training).**
Round 3 accepted 68 of 148 (43 strict). Rows with an accepted rat3: **297 of 647 (46%)**;
circuit2 88/133, seir2 73/142, grid2 51/126, frame3d2 44/113, traffic2 41/133; 184 rows pass the
strict five-criterion bar. Yield per round fell 135 → 94 → 68, each round costs about 2 h of
rendering plus an audit, and the owner asked for less agent use, so the remaining 350 rows keep
their rat2 rationale. The `rat3` arm is therefore 46% verified comparison and 54% rat2; both the
pooled and the per-domain fallback shares are reported with the result.

**rat3 coverage, 2026-10-20 10:40, after two audited samples per row:** 229 of 647 rows have an
accepted rationale (circuit2 77/133, seir2 57/142, grid2 41/126, frame3d2 32/113, traffic2
22/133). Round 2 accepted 94 of 197 on the rat2 bar (53 strict). A third sample is rendering for
the 418 rows without one; at this yield the arm will still carry roughly 40% rat2 fallback rows,
so `rat3` tests "verified comparison where the model could write one, rat2 elsewhere", and no
domain will meet the ≤ 25% fallback subset rule. Both facts go in the results.

**Change logged 2026-10-20 10:05.** rat_s2 failed at step 4 with "no space left on device" while
writing an intermediate checkpoint (drc's system disk is full: 7.5 GB free, most of it another
project's data). rat_s2 and every later arm are trained with `sft.save_steps=1000`, i.e. no
intermediate checkpoints, only the final adapter. This does not change the trained weights
(`load_best_model_at_end` was already off, so the final adapter is the last step in every arm).
act_s1, rat_s1 and act_s2 completed with the original setting.

**Hint pilot, element-only variant (2026-10-20):** 74 rows × 4 samples, thinking off: the pick is
the gold move in 67% of samples (candidate list 59%, no hint 28%), mean Φ percentile 93. Not added
as an arm: the element hint gives away the choice of lever, so its trace is not a comparison.

**Addition logged 2026-10-19 19:10 (pre-registered before any own_cands row is used for training).**
Two more arms, `own_cands` / `own_cands_act`, the hinted own choice: the prompt lists four shuffled
candidate moves (gold, one of the five best other moves by Φ, two random legal moves) and the model
evaluates and picks, thinking off, written reasoning, K=4. Selection and matched control exactly as
for `own` (highest-Φ-percentile pick, kept if ≥ 90th percentile or the gold move; the training
prompt carries no hint). Pilot that motivated it (100 rows, 4 samples): the pick is the gold move in
59% of samples (28% without the hint), any of the 4 samples in 84% of rows, mean percentile 96.
Primary readout `own_cands − own_cands_act`; secondary vs `own` and `act`.

**Change logged 2026-10-19 19:30, after the first rat3 audit round and before any rat3 row is
trained on.** Round 1 (285 passing first samples): 88 (31%) pass all five criteria; 135 (47%) pass
the four rat2 criteria (`comparison_ok` fails on 144, almost always a magnitude error in a
rejected option, e.g. doubling treated as two 10% steps, with the conclusion surviving).
- The arm is built from samples that pass the **four rat2 criteria plus the simulator check on
  every alternative's direction**, i.e. the same audit bar rat2 was held to, with the comparison's
  directions verified mechanically. Holding rat3 to a stricter bar than rat2 would confound the
  arm comparison with a selection difference. The five-criterion (strict) rate is reported as
  well, and Stream-of-Search / the retry-data result suggest magnitude errors in rejected
  options are not what matters for training.
- Up to 6 samples per row instead of 4, since first-sample coverage is 135/647 rows.
- Fallback to the row's rat2 rationale remains for rows with no accepted sample; the fallback
  share is reported per domain, and the rat3 − act comparison is also reported on the subset of
  problems whose domains have ≤ 25% fallback.

**Observed before audit (2026-10-19 18:22), rat3 deterministic pass on the first sample: 44%**
(circuit2 71, seir2 58, grid2 33, frame3d2 29, traffic2 25). Nearly all failures are a wrong
predicted direction for an alternative in the simulator (e.g. traffic2 "max_queue_frac down" when
the move leaves it unchanged; frame3d2 "fos_yielding up" when the minimum falls through
redistribution). The rule stays; the fallback-to-rat2 count will be reported per domain.

**Change logged 2026-10-19 15:15, from the 20-row pilot, before any rat3 sample was audited:**
- The verifier read the chosen action as the last `SET_PARAM` in the text, which for rat3 is inside
  the candidates block; it now ignores that block. (All 20 pilot rows had failed `action` for this
  reason.)
- The deterministic leak flag is recorded but does not gate. In grid2 the observation states that
  effects are exact and linear, so a correct derivation reproduces the outcome value and was
  flagged as a leak (4 of 20 pilot rows, e.g. "12.2"). Leakage is judged by the auditor's
  `no_leak` criterion, as in the rat2 probe; the flag is shown to the auditor.
- Simulated-alternative check on the pilot: 8 of 20 rows had at least one wrong predicted
  direction for an alternative. This is the intended filter and stays.

### Cross-check with the collaborator's progress document of 2026-10-20

Source: `/ocean/projects/mch250030p/shared/handoffs/dataxp_progress_20261020.en.md` (wwang25; DX-24
to DX-35, DX-34 running). Same v2 domains, same V2 instruction, same 2,048-token reply cap, thermal2
two runs /120, trained domains /300 per run; their "reasoning base" = our `base_nothink`.

| Quantity | Theirs | Ours | Agreement |
|---|---|---|---|
| reasoning base, thermal2 /120 | 101 (DRC) / 105 (B2) | 51/60 r1 (r2 pending) | yes |
| reasoning base, trained domains /300 | 159 / 147 | 153 | yes |
| teacher-action SFT, thermal2 | DX-24 73 (2,380 rows) | act 74; act_s1 36/60, act_s2 39/60 | yes |
| teacher-action SFT, trained domains | 247 (+0.29) | 186 (+0.11, 578 rows) | scales with rows |
| rationales for teacher actions, thermal2 | DX-28 27 vs action-only 51 (template collapse) | rat 76 vs act 74 (no transfer loss) | differs: ours are audited non-template text |
| rationales for teacher actions, trained domains | below base | rat −0.10, rat3 −0.12, seeds −0.12/−0.11 vs act | both negative |
| own rationale vs own action-only, thermal2 | OWN-R 86 vs OWN-A 61 (114 rows); 96 vs 84 (1,353 rows, n.s. @12) | own 91 vs own_act 92 | **differs** |
| own action-only vs teacher action-only, thermal2 | not run as a pair | own_act 92 vs act 74 | — |
| inventory2 | every tuned model below base (40-seed judgment) | every arm ≤ base except own 19 vs 16 (single seed) | mostly |
| base thinking on vs reasoning base | DX-23 −0.038 [−0.116, +0.041] | base_think not run | — |

Reading. Their headline §0.4 ("base-style text in the thinking block keeps transfer") is not what
our matched pair shows: with the same own rows and own actions, reasoning present or absent gives
the same thermal2 result (91 vs 92). In our data the action source separates transfer: teacher
actions (act, 18 steps) 74, own picks (own_act, 16 steps) 92, both action-only. That fits their own
DX-25 diagnosis (the teacher's fixed step sizes) better. A reconciliation consistent with both
sets: transfer loss tracks how far training moves the policy; their OWN-A-full moved a lot (+0.238
in-domain at 66 steps) and lost transfer, rationale rows move the policy less per step (their
OWN-R +0.05, our own −0.04), and their A2 test rules out the loss-weight version of this but not
the policy-shift version. Caveats: own_act has one training seed; our 489 own rows are not the
578 gold rows; their own rationales are sampled short (60–150 words) while ours are a median 464
words and cap at 2,048 tokens on half the turns at inference (their base also hits the cap: 83 of
300 problems excluded in their format-gain check). Their DX-23 makes our `base_think` reference
redundant; it is dropped from the queue unless asked for.

## Data as built (2026-10-18, before training)

**Kept rows: 578 of 623 train (93%) and 69 of 70 dev.**

| domain | train kept | audited good |
|---|---|---|
| circuit2 | 118 / 121 | 86% |
| frame3d2 | 99 / 122 | 45% |
| grid2 | 114 / 125 | 76% |
| seir2 | 127 / 131 | 82% |
| traffic2 | 120 / 124 | 88% |

**Renders and audits.**
- 1,130 renders: 899 rat2 and 231 grid2 `rat`.
- Deterministic pass rates: 91% for rat2. grid2 is 72% over all samples; its first samples pass
  82% after the action-check fix.
- 897 audited by single Claude auditors, 647 judged good.
- Good rate by sample: first sample 82%, second 60%, third or fourth 39%. Rows that fail once tend
  to keep failing.
- Most frame3d2 rejections share one cause: the gold (Φ-select) move is not the one the
  observation supports. Typical cases:
  - a member that is not failing is upsized, or steel is chosen, where the table lists a nearly
    free aluminium fix;
  - the rationale then has to claim relief through "redistribution", which the observation gives
    no basis for.

**Selection effect.** The dropped rows are skewed toward moves the observation cannot explain. All
three arms drop the same rows, so the comparison stays matched, but the kept set is easier to
explain than the full tier.

**Tiers.**
- Train/serve parity holds on every row of all six files.
- The arms have identical prompts and action text; act's think block is empty; shuf differs from
  rat on every row.
- Rationale median: 146 words, about 270 supervised tokens per row. act supervises about 21 tokens.
- Files in `/storage/wxu/sftprobe/tiers` on drc, sha256 in `tools/final_build.sh` output:
  - `train_act` 2e437b73…
  - `train_rat` b5b9ac7b…
  - `train_shuf` 7217a8f9…

**Content difference from DX-28, noted for interpretation.** The rat2 rule allows the rationale
to state an expected effect qualitatively when the outcome confirms it. DX-28's rationales were
written without the outcome and with no predictions. So `rat` differs from DX-28 in two ways:
it saw the outcome, and it states expected directions.

## Changes after pre-registration

- **2026-10-18 (before any training), render order.** Samples are rendered lazily:
  - first sample 0 for every row;
  - sample k+1 only for rows whose samples 0..k all failed (deterministic checks or audit), up to 4.

  The kept-sample rule is unchanged: the first sample in index order that passes and is judged
  good. Samples are i.i.d. draws, so the kept rationales are distributed exactly as with K=2 up
  front. Reason: about 280 renders per hour on one GPU, and sample 1 is unused for every row whose
  sample 0 is kept. At this point 264 of 1,386 planned renders were done and 92 had been audited.
- **2026-10-18 (before any training), action check fix.**
  - The bug: the renderer's prompt shows the gold value at 6 significant digits (e.g. `139.266`
    for 139.26575). The model copies that, and the "action equals gold" check compared it with the
    full-precision value, so it failed.
  - Affected: 16 of 83 grid2 renders (grid2 values are continuous set-points).
  - The fix: the check now also accepts the displayed value. These samples were never audited or
    used; they now go through the audit like any other.
  - The trained action is the tier row's own action and is unaffected.
- **2026-10-18, second server.** grid2 rationales (`rat`) are rendered by a second instance of
  the same model (Qwen3.8-27B, bf16, same serving flags, tensor-parallel on 4 RTX 4000 Ada cards),
  in parallel with the main server. Samples are draws from the same model, so only throughput
  changes.
- **2026-10-18 (before any training), shuf dev set.**
  - The problem: the dev frame3d2 rows come from two problems, one holding 7 of the 13 kept rows,
    so no pairing across problems exists.
  - The fix: for the dev split only, a shuf row may take the rationale of another row of the same
    problem.
  - This does not affect what is trained. Dev loss is only logged: `load_best_model_at_end` is off
    because `save_steps` 0.1 is not a multiple of `eval_steps` 0.25, so the final adapter is the
    last step. The train split keeps the strict cross-problem rule.

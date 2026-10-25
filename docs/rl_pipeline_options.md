# RL phase for dataxp: options for the owner (2026-10-25)

Nothing here has been built or launched. Each option names what is trained, on what, how it is
judged and what it costs. The recommendation and the first step are at the end. Choose an option
before anything is built.

## 0. Sources

- **October 22 Meeting artifact** (claude.ai/artifact/VcraKEbD47pGeQkPvCVadC, the
  collaborator's; "DATAXP V2 · SFT THAT KEEPS THE MODEL'S OWN REASONING · AS OF OCT 22").
  - The Artifact tool's read could not show its approval prompt in this session. I read the
    artifact in the in-app browser instead: it is shared with anyone who has the link.
  - Its setup is the **v2 protocol**: five trained families (circuit2, frame3d2, grid2, seir2,
    traffic2); held-out thermal2 ("heat-sink") and inventory2. V2 instruction, thinking on,
    the collaborator's harness, feasible within 12 turns, 60 problems per family, two runs.
- **Local docs.** `docs/corpus_native_domains.md`, `docs/slope_test_prereg.md`,
  `docs/sft_probe_rat2.md`, `docs/multiturn_grpo_design.md`, `docs/datav4_append_plan.md`,
  `docs/dataxp_spellings.md`, `docs/synthetic_cot_literature.md`, `docs/ablation_ledger.md`.
- **Bridges-2.** The collaborator's `dataxp_progress_20261020.en.md` (DX-24..DX-35), still the
  newest handoff file.
- **Orchard.** The **DX-37 base readout** on the datav4 harness, both spellings, all 11
  families: `/project/community/wxu2/wenwenw/dx37/analysis/base_only.md`. DX-37's trained
  arms R and A exist as training files only; neither has an evaluation yet.

### The artifact's numbers (feasible within 12 turns; quoted as shown)

| test | untuned | with reasoning (OWN-R-full) | moves only (OWN-A-full) | + on-policy (DX-34 R1) | merged (β = 0.5) |
|---|---|---|---|---|---|
| trained domains, run 1 (/300) | 159 | 171 | 224 | 178 | **217** |
| trained domains, 2 runs (/600) | 306 | 336 | 449 | 357 | running |
| heat-sink, 60 × 2 (/120) | 105 | 96 | 84 | 97 | **105** |
| heat-sink, 120 new × 2 (/240) | 191 | 189 | – | 190 | running |
| inventory, 60 × 2 (/120) | 21 | 10 | 6 | 8 | 6 / 60 (run 1) |
| inventory, 120 new × 2 (/240) | 45 | 23 | – | 23 | running |

The artifact also states these results and plans:

- **On-policy self-distillation (DX-34 round 1):** "+0.038 [+0.003, +0.073] in-domain, just
  above the significance line; no change on the unseen domains".
- **Merged model.** It adds 0.5 × the moves-only weight change to the with-reasoning model.
  - In-domain against the untuned model: +0.193 [+0.139, +0.246].
  - "Preliminary: it was chosen on these problems; a re-test on new ones is running."
- **Larger held-out test.** Heat-sink is level (−0.004 [−0.067, +0.058]). "The inventory gap
  is real (23 vs 45 of 240)."
- **Diagnostics.** The first move picks the teacher's parameter 35% of the time (with
  reasoning), 47% (moves only) and 31% (untuned). On inventory, the fine-tuned models make
  structural changes in about 4% of moves, against 12% for the untuned model.
- **Next steps it lists:**
  - re-test the merged model on fresh problems;
  - β = 0.7 and 1.0;
  - train one model directly with a second seed;
  - for inventory, judge samples "by where they lead after a few moves" instead of by a
    single improving move;
  - longer term, "train with the simulator's score as a reward (RL)".

**On your summary.** Held-out performance is on par with the untuned model on **heat-sink**:
105 vs 105 of 120 for the merged model, and 190 (189) vs 191 of 240 on new problems. It is
**below** the untuned model on **inventory**: 23 vs 45 of 240 for both the with-reasoning
and the on-policy models, and merged 6/60 on run 1. Every fine-tuned model in the artifact is
below the untuned model on inventory.

### Base reference on the datav4 harness (DX-37, multitool, thinking off, two runs)

Nine problems are left out because they start feasible. That leaves trained8 = 476 problems
and heldout3 = 175 (thermal2 60, inventory2 60, molopt 55). Inventory2 is judged robustly over
40 seeds.

| feasible@12, run 1 / run 2 | circuit2 | frame3d2 | grid2 | seir2 | traffic2 | grid2op | photonics | engibench | thermal2 | inventory2 | molopt | trained8 | heldout3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base, multitool | 19 / 16 | 20 / 24 | 22 / 26 | 45 / 42 | 48 / 48 | 11 / 8 | 1 / 1 | 25 / 25 | 53 / 53 | 11 / 12 | 15 / 17 | **191 / 190** | **79 / 82** |
| base, set_parameter | 14 / 18 | 24 / 25 | 27 / 27 | 48 / 46 | 45 / 46 | 8 / 9 | 1 / 2 | 23 / 20 | 49 / 53 | 6 / 10 | 15 / 13 | 190 / 193 | 70 / 76 |

On heldout3 @12, multitool − set_parameter is +0.043 [+0.003, +0.083]. A clustered CI on a
175-problem, two-run difference is about ±0.04.

## 1. Established results the options build on (cited, not re-derived)

- **Φ is a selector, not a policy signal** (ledger Turns 10, 13, 22).
  - As a reward term it telescopes to a group constant under GRPO's mean-centring.
  - In the prompt it cost −7.5pp. Per-token Φ credit was null.
  - Selection by Φ converts an available feasible candidate about 100% of the time.
  - "The simulator's score as a reward" therefore has to mean a **terminal outcome**
    (feasible, turns, objective), not Φ or any per-step potential.
- **Truss GRPO learned once three things were fixed** (Turn 0 and the champion entries):
  - the single-turn protocol bug;
  - a too-low LoRA lr (5e-6: kl ~0.003, grad_norm ~0.1, zero clipping);
  - the observation missing the critical member (12% → 54% on 50 problems; 0.895 on clean
    problems after Turn 7).

  It then plateaued, converging by step 50. A grammar reward did not stop format drift
  (Turn 2).
- **SFT trade-off.** Teacher-action SFT gains in-domain and loses transfer. SFT on own
  accepted actions keeps more of it (probe thermal2: own_act 92 vs act 74 of 120). With
  reasoning, transfer is kept and the in-domain gain is small. The merge recovers the
  in-domain gain on heat-sink (preliminary). **Nothing so far fixes inventory.**
- **Expert iteration has been run once on v2.** DX-34 R1: +0.038 [+0.003, +0.073] in-domain,
  held-out unchanged. That is the known size of one round of self-distillation with one-step
  acceptance.
- **Inventory mechanism (artifact).** Fine-tuned models stop making structural moves (4% vs
  12%). The acceptance filter keeps a sample only if its single move already improves the
  design, so it discards exploratory and structural moves. Every SFT recipe so far shares this
  filter, and so do the datav4 rows (≥ 90th Φ percentile or the teacher's move).
- **Literature anchors.**
  - RL's Razor: forgetting tracks the KL to the base.
  - Chu et al. (ICML 2025): SFT memorises, RL generalises.
  - Self-distillation fine-tuning (SDFT); on-policy data as the transfer mechanism (Huan et al.).
  - Expert iteration: Anthony et al. 2017, STaR, ReST-EM (most of the gain in round 1).
  - FLAN-style task diversity.

## 2. Decision 0: which environment the first RL run uses

| | P1: v2 protocol (the artifact's) | P2: datav4 protocol (Orchard v3, DX-37) |
|---|---|---|
| families | 5 trained; thermal2 and inventory2 held out | 8 trained (v2 five + grid2op, photonics, engibench); thermal2, inventory2 and molopt held out |
| spelling / mode | SET_PARAM, V2 instruction, thinking on, 2,048-token cap | multitool verbs (or set_parameter), thinking off |
| start checkpoint | **S = merged β = 0.5** (or OWN-R-full); exists, but the weights are with the collaborator | no evaluated datav4 SFT yet (DX-37 R / A trained data only); base otherwise |
| comparable to | the artifact directly | DX-37 base; the artifact only indirectly |
| simulators | v2: milliseconds per step | natives up to 0.4 s per step; photonics base 0.02 (almost no reward) |
| target the evidence points to | inventory (23 vs 45 / 240), which SFT has not moved | broad held-out transfer incl. molopt |

The Orchard v3 harness serves both: its SET_PARAM path is byte-identical to the original on
the old variants. So one rollout worker covers P1 and P2. **My recommendation is P1 first.**
S exists and is strong, the comparison to the artifact is direct, the simulators are cheap, and
inventory is a concrete failure that a terminal reward can address and a one-step filter
cannot. Extend to P2 once the pipeline is proven.

## 3. Evaluation block shared by every option (pre-registered)

- **Harness.** Orchard v3 in the chosen protocol, T = 0, 20 turns, two runs; inventory2 judged
  robustly over 40 seeds.
  - **P1:** trained 300 × 2, heat-sink 60 × 2 plus the 120 new × 2, inventory 60 × 2 plus the
    120 new × 2. The new-problem sets come from the collaborator.
  - **P2:** all 11 families as in DX-37.
- **References.**
  - **untuned:** the artifact's column (P1) or DX-37 base (P2). Not re-run.
  - **S:** the artifact's merged model, after its fresh-problem re-test. Under P1 it is
    re-evaluated only if the Orchard harness numerics differ from the collaborator's. It is
    never retrained.
- **Statistic.** Paired by problem, problem-clustered bootstrap 95% CI.
- **Criteria.**
  1. **Primary (held-out):** arm − S on heat-sink + inventory pooled, new problems (480
     episodes), lower CI bound above 0. Inventory is reported separately. The RL phase's job
     is the side SFT has not fixed.
  2. **Guard (in-domain):** arm − S on the trained domains (600 episodes), lower CI bound
     above −0.05.
  3. **Headline:** arm − untuned with a lower CI bound above 0 on both the trained domains
     and the held-out pool, with inventory not significantly below untuned. This would be the
     first both-sides result.
  4. **Stop:** arm − untuned on heat-sink has an upper CI bound below 0.

  Under P2 the same criteria apply with trained8 / heldout3 and DX-37 base.
- **Diagnostics, reported and not gated.**
  - Per-family @12 and @20.
  - Structural-move share on inventory (the artifact's 4% vs 12%).
  - Parse rate, illegal moves, cap share and reply tokens.
  - A-B-A-B episodes and undo share.
  - Per-token KL to the untuned model on held-out prompts (RL's Razor).
- **Evaluation cost.** About 1,300–1,400 episodes per arm, about 6–8 GPU-h on Orchard.

## 4. Options

### A. Expert iteration with multi-step acceptance

- **Change from DX-34.** One round with one-step acceptance gave +0.038 in-domain and no
  held-out change, so the same round again is not worth funding. The variant worth testing is
  the artifact's own next step 4: accept a sample by **where it leads after a few moves**.
  - Apply the sampled move, let S continue for m = 3 moves, and keep the sample if the design
    is feasible or has strictly fewer violated constraints than after S's own greedy move.
  - Rank the kept samples by turns-to-feasible.
  - This keeps structural moves that pay off later.
- **Trained.** A LoRA from the untuned model on S's rows plus the round's rows (retrain from
  base, as in DX-34).
- **Matched control.** The same states with the one-step filter, for the same steps.
- **Primary readout.** The inventory structural-move share and the primary held-out criterion.
- **Compute.**

  | item | GPU-h |
  |---|---|
  | harvest (500 episodes) | ~3 |
  | sampling (8 per state, ~2,400 states) | ~9 |
  | m-move continuations | ~10 |
  | two trainings (~2,300 rows, ~110 steps each) | ~25 |
  | evaluation, two arms | ~14 |
  | **total** | **~60** |

  Wall time is 3–4 days.
- **Risks.**
  - It overlaps the collaborator's planned next step; coordinate so it is not done twice.
  - The continuation policy is S, so it can only credit moves S knows how to follow up.
- **Infra missing.**
  - A continuation-based acceptance in place of DX-30a (the replay exists for v2).
  - Under P2, replay for natives and time-stepping domains.

### B. Multi-turn GRPO with a terminal reward and a KL leash to the untuned model (main candidate)

- **Trained.** S's LoRA, continued. The reference is the model with the adapter disabled, i.e.
  the untuned model, so the KL term is the KL to the base.
  - The merged S is a weight-space sum of two LoRA deltas. It has to be expressed as one
    adapter (sum the deltas, or SVD back to rank 64–128) or be merged into the base weights.
    In the second case the reference must be loaded separately as the untuned model.
  - A second arm from the untuned model is optional.
- **Environment.** The harness's own episode loop as the rollout worker, on training problems
  (never evaluation problems), in the chosen protocol. Horizon 12 turns, reply cap = the
  evaluation cap.
- **Reward.** No Φ and no format term.
  - R = 1[feasible by turn 12] + 0.2 · (12 − t_feasible)/12. R = 0 if never feasible.
  - grid2op game-over (P2) gives 0.
  - Environment tokens are masked from the loss.
  - Inventory is held out, so it is not trained on. Structural moves can only be learned on
    training families where they pay off later: frame3d2 member installs, circuit2 cascode and
    compensation, grid2 commit/decommit and line switching.
- **GRPO settings.**
  - G = 8 rollouts per problem at T = 0.8, 8 problems per step.
  - Zero-variance groups are dropped and replaced, and their share is reported.
  - LoRA lr 1e-5 (the canary also tries 3e-5), β_KL 0.02–0.05, clip 0.2, 150 steps.
  - Evaluation at steps 50 and 150.
- **Stop rules.**
  - The training parse rate falls more than 10 points.
  - kl stays under 0.005 for 30 steps (lr too low, as in Turn 0).
  - KL to the untuned model exceeds 0.1 nats per token on held-out prompts.
- **Compute.**

  | item | GPU-h |
  |---|---|
  | canary | ~8 |
  | main run (150 steps at ~5–6 min on 4 H100: TP-2 vLLM + 2 training GPUs) | ~55 |
  | evaluation | ~14 |
  | **one arm** | **~75** |

  Engineering takes 3–5 days. A run takes about 15 h on `advanced`, resumable across jobs.
- **Risks.**
  - Infrastructure for multi-turn 27B rollouts and weight sync.
  - Training families may reward structural moves too rarely to transfer that habit to
    inventory.
  - It may converge early (truss converged by step 50).
- **How the truss failures are avoided.**
  - Rollouts are the harness loop, checked by a byte-identical parity test at T = 0.
  - lr 1e-5 with kl/clip gates.
  - Φ is kept out of the reward.
  - The KL leash targets the untuned model.
  - The observation is frozen for the whole study.
- **Infra missing.**
  1. A rollout function returning token ids, logprobs and an env mask, built from
     `_run_episode`. The TRL contract in `docs/multiturn_grpo_design.md` was verified on
     **TRL 1.0.0**; Orchard has **1.13.0**, so re-verify. A thin custom policy-gradient loop is
     the fallback.
  2. vLLM server mode with per-step LoRA hot-loading.
  3. A trainer that checkpoints every step (adapter, optimizer, RNG, sampler position) and
     resumes.
  4. Training-problem files.
  5. S's weights and the new-problem evaluation sets from the collaborator.
  6. (P2 only) grid2op stopping at game-over.

### C. Single-decision GRPO on stored states, with a reward from a short continuation

- **Trained.** S's adapter, with KL to the untuned model.
- **Data.** States from S's own episodes, or the stored corpus states.
- **Reward.** Apply the sampled move, let a fixed finisher continue up to 8 turns, and score
  feasible within budget plus a speed bonus.
  - If the finisher is phi_select, Φ is used only to select inside it.
  - If the finisher is S at T = 0, it costs GPU.
- **Why consider it.** It is the standard `GRPOTrainer` with a reward function: no multi-turn
  trainer and per-decision credit. It is option A's acceptance rule turned into a graded
  reward.
- **Compute.** About 70 GPU-h. Engineering takes 1–2 days.
- **Risks.**
  - A teacher finisher rewards "the teacher can finish from here", a softer form of the
    teacher imitation that lost transfer.
  - One-step credit cannot teach when to stop or undo across turns.

### D. Train the proposer for small-K search (Φ selects at test time)

- **Scope decision for you.** The deployed system would be "the model proposes K moves, Φ
  picks the best" (K + 1 simulator calls per turn). Φ exists for the held-out families too.
- **Gate (no training, ~12–15 GPU-h).** S and the untuned model as proposers with K = 4 + Φ,
  against K random legal moves + Φ and full phi_select, on the evaluation problems.
  - Truss precedent: the LLM proposer never beat random proposals inside the same search.
  - D stops at the gate if that repeats here.
- **Training, if the gate passes (~65 GPU-h).** Single-decision GRPO. A sample's reward is its
  marginal contribution to the best of its group under Φ ranking: the use of Φ that works.
- **Infra missing.** An "LLM proposer + Φ select" policy in `dataxp_policy_eval_v3.py`.

### E. Leave-families-out slope test on datav4 (a gate, not RL)

- **What.** The `docs/slope_test_prereg.md` draft (never approved), re-sized to about 2,000
  own_act rows per arm, with k = 2, 4 and 8 families, multitool, held-out heldout3.
- **Compute.** About 45 GPU-h, 1.5–2 days on preempt with per-step checkpoints.
- **Decides** whether family breadth is a lever for P2 and for the append batches. It is
  independent of A–D.

## 5. Summary

| | trains (from) | signal | new infra | GPU-h | wall | main risk |
|---|---|---|---|---|---|---|
| A | LoRA from untuned, S's rows + new | own replies kept by where they lead after m moves | continuation acceptance | ~60 | 3–4 d | overlaps the collaborator's next step |
| **B** | S's adapter, KL to untuned | feasible within 12 turns | rollout worker, LoRA sync, resumable trainer | ~75 per arm | 3–5 d build + ~1 d per run | infra; structural moves may not transfer to inventory |
| C | S's adapter | continuation outcome per decision | reward service | ~70 | 2–3 d | finisher bias; no multi-turn credit |
| D | gate first | best-of-group under Φ ranking | proposer + Φ arm | 15 (+65) | 1 d gate | proposer may not beat random |
| E | 5 SFT arms | none (diagnostic) | none | ~45 | 1.5–2 d | underpowered |

## 6. Recommendation and first step

**Run B under P1, starting from S (the merged model once its fresh-problem re-test holds),
with the KL leash to the untuned model.**

- **Signal.** It is the only option whose signal is the evaluation outcome itself, reached
  through the model's own multi-turn play.
- **Inventory.** It credits moves that pay off several turns later, which is exactly what the
  one-step filters behind every SFT recipe discard (the inventory diagnosis).
- **Known size of the alternative.** One round of self-distillation with one-step acceptance
  is already measured at +0.038 in-domain with no held-out change.
- **Alongside.** Run D's gate (cheap, settles the policy-vs-search scope). Leave A to the
  collaborator's planned multi-step filter unless you want it here.

**First concrete step after your choice (no GPU).**

1. Build the rollout worker from the harness's `_run_episode`.
2. Pass the parity test: S at T = 0 on 3 training problems per family gives byte-identical
   transcripts to `dataxp_policy_eval_v3.py episodes`, with token ids, logprobs and the env
   mask returned.
3. Check TRL 1.13's `rollout_func` against the installed source.
4. Ask the collaborator for S's adapters and the new-problem evaluation sets.
5. Run a 2-hour canary on `advanced` (~8 GPU-h) with these pass gates:
   - kl > 0.01 and rising, with clip ratio > 0;
   - finite rewards and a zero-variance group share under 50%;
   - an unchanged parse rate;
   - resume tested by a kill after step 2.

**Decisions I need from you.**

- Which options.
- P1 or P2 for the first run.
- S = the merged model (preliminary) or OWN-R-full.
- Whether I may ask the collaborator for the weights and problem sets.

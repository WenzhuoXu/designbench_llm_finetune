# con1 — the full consequence row set and the stage-1 continuation job

Written 2026-11-26 by a side session of the con1 session (branch `claude/con1-2m-rows-e5f775`). Companion of
`docs/consequence_round_con1.md` §11 (stage 1 = Orchard job 176033 on 15,857 facet rows). Nothing here was trained;
the continuation job is ready and not submitted.

## 1. What the set is

Every simulated outcome of every training state (and, separately, of every dev state), minus the (state, move)
pairs stage 1 trained on (`main/rows/train_facet.jsonl`) or used as dev rows (`main/rows/dev_facet.jsonl`), written
as consequence rows in the facet format by stage 1's own row code (`build_rows.conseq_row`, `targets.py`,
unchanged: same system/user text, question wording, answer layout, informative spans, fixed-wording weight 0.05).

- **States.** `main/states_native_offmenu.jsonl.gz` (stage 1's 5,902 native states with their 2 off-menu moves),
  `full/states_native_offmenu_rest.jsonl.gz` (the other 5,822 native states, off-menu moves added by this session,
  `add_offmenu.py` k = 2), then `gen/out/states_native_orch.jsonl.gz` and `gen/out/states_scripted.jsonl.gz`; the
  first line of a sid wins. drc job j2614's output (`/storage/wxu/con1/gen/out/states_native_drc.jsonl.gz`) holds
  exactly the same 11,724 states as Orchard's (same 1,084 episode keys, same sid set by md5), so nothing was copied
  from drc. `data/states_native*.jsonl.gz` add no state either (all 11,724 come from the Orchard file).
- **Held out.** inventory2 / molopt / thermal2 states and any state on a problem of `data/queries_v2.jsonl` or
  `gen/bench_problems.jsonl` stop the build; splits other than train/dev stop it too.
- **Rows.** Per state, R = ceil(n / Q) rows of balanced size over disjoint moves: own moves, then off-menu /
  off-grid moves, dealt round-robin (one per row before any row gets a second), uniform moves fill the rest, each row
  shuffled; `random.Random` seeded by (seed 0, sid). A row longer than the cap (exact tokens) makes the state's rows
  smaller (R + 1) until all fit.
- **Exact tokens.** Lengths are conseq_format.encode's (the trainer's layout) with the model's tokenizer, counted as
  the sum of the separately tokenised pieces; the rows of 1 state in 50 are also tokenised whole and must agree.
- **Same bytes every build.** The trainer's fingerprint holds the rows file's sha1, and each 24 h job rebuilds the
  file on its own node; the build was run twice (30 and 64 workers) and compared byte for byte (§3).

## 2. Code

| file | what |
|---|---|
| `scripts/con1/build_full_rows.py` | the builder (stats JSON, per-row lengths) |
| `scripts/con1/full_check.py` | independent checks of a build (no pair twice, none of stage 1's, counts add up, row layout) |
| `scripts/con1/full_estimate.py` | step-time fit on stage 1's measured steps, epoch time per candidate set |
| `scripts/con1/full_prep.sh`, `slurm/con1_full_prep.sbatch` | CPU prep: off-menu moves, builds at Q = 12 and 36, checks |
| `scripts/con1/orchard_full.sh`, `slurm/con1_full.sbatch` | the continuation job (one 24 h link of a chain) |
| `scripts/con1/guard_cycle.sh` | the parent's `parent_check/guard_cycle.sh` with env `RES`, `STAGES`, `FORCE` (defaults: the original's behaviour) |
| `slurm/con1_full_test.sbatch` | 1-GPU mechanics test of the chain (fresh r8 LoRA, small rows) |
| `llm_finetune/rl_dataxp/sft_rows.py` | `sft.lazy_rows`, `sft.check_rows`, `sft.resume_dir` (a checkpoint copied off the node: exact resume), `sft.rewarm_steps` (all off by default) |

Reused, not rebuilt (owner rule): the monitor `parent_check/mon.py` and the checkpoint guard `parent_check/guard_cycle.sh`
(a copy with three env hooks, because the original hardcodes stage names `stage1 stage2`, which would have put this
run's copies on top of stage 1's in `con1_reserve`).

Deployed copy: Orchard `/project/community/wxu2/con1/repo2m` (job 176033 reads `con1/repo`, untouched).

## 3. Measurements

**Off-menu moves** (Orchard CPU job 176036, 90 workers, 965 s): the 5,822 native states outside stage 1's selection
(grid2op 1,430 train + 1 dev, photonics 4,349 train + 42 dev; every engibench state was already selected), 0 errors:
8,878 off-menu moves (grid2op 2,862, photonics 6,016; photonics states whose design breaks the fabrication rule have
tiny menus and yield fewer than 2). File `full/states_native_offmenu_rest.jsonl.gz` (73 MB).

**The set** (same job; identical for Q = 12 and Q = 36, only the grouping differs):

| family | train states | outcomes simulated | in stage 1 | **in the set** | dev outcomes in the set |
|---|---:|---:|---:|---:|---:|
| circuit2 | 3,327 | 204,495 | 24,000 | **180,495** | 1,443 |
| engibench | 1,857 | 103,552 | 22,284 | **81,268** | 50 |
| frame3d2 | 3,491 | 475,561 | 24,000 | **451,561** | 1,538 |
| grid2 | 2,977 | 288,647 | 24,000 | **264,647** | 3,739 |
| grid2op | 3,430 | 278,422 | 24,000 | **254,422** | 46 |
| photonics | 6,348 | 158,702 | 18,181 | **140,521** | 1,312 |
| seir2 | 3,054 | 295,879 | 24,000 | **271,879** | 2,599 |
| traffic2 | 2,678 | 367,508 | 24,000 | **343,508** | 4,998 |
| **total** | 27,162 | 2,172,766 | 184,465 | **1,988,301** | 15,725 |

`full_check.py`: 2,004,026 (state, move) pairs (train + dev), none twice, none of stage 1's 185,777 (train + dev)
pairs, no malformed row; simulated = stage 1 + set for every family and split; nothing dropped for length (CHECK_OK
for both builds). Off-menu / off-grid moves in the set: 19,783 (engibench none: stage 1 used both of every engibench
state's); own moves 20,884 (engibench, grid2op, photonics).

**Q = 12 vs Q = 36** (exact tokens, cap 24,000 per row):

| | Q = 12 | Q = 36 |
|---|---:|---:|
| rows | 181,372 | 76,558 |
| questions per row (mean) | 10.96 | 25.97 |
| tokens per row: mean / median / p95 / max | 11,759 / 10,939 / 19,228 / 24,000 | 18,517 / 18,685 / 23,669 / 24,000 |
| prompt tokens per row (mean) | 7,700 | 8,544 |
| training tokens | 2,132.8 M | 1,417.6 M |
| tokens per outcome | 1,073 | 713 |
| rows file | 5.84 GB | 4.16 GB |
| build on 30 CPUs (node /tmp) | 85 s | 74 s |

Stage 1 for comparison: 15,857 rows, 179.5 M tokens, 973 tokens per outcome, longest row 25,859 tokens. Memory: the
longest row of every candidate is 24,000 tokens; stage 1 (job 176033) peaked at 68.9 GiB on steps whose longest row
was 25.0–25.4k tokens, so no row of the set should need more.

**Same bytes:** the Q = 12 build was repeated with 64 workers: identical file (sha1
5d008b7bff76da9b07f94aef5ba9cc449047c4a2). Q = 36: sha1 955b00b135c6857572cbdd5908cc6f12eb782663. 3,797 (Q = 12) and
1,577 (Q = 36) rows tokenised whole agree with the piece sums.

**Time** (`full_estimate.py`, fit on job 176033's measured steps: stage 1's batches reconstructed exactly from
row_order and shard_samples, 0 mismatches against the logged longest row of each step): see the table in §4.

**Regime of the outcomes in the set** (train; as the facet answer states it): applied-infeasible 1,871,916 (94.1%),
refused 107,368 (5.4%; grid2op 107,013), applied-feasible 9,017 (0.45%). Feasibility flips are rare: most states are
infeasible designs and most moves leave them infeasible. Off-menu moves added here: grid2op 2,860 = 1,264 refused,
1,570 applied-infeasible, 26 applied-feasible; photonics 5,968, nearly all applied-infeasible.

## 4. The continuation job

**Choice: Q = 36, 28 rows per step** (mine, from the measurements). Q = 36 needs 713 tokens per outcome against
1,073 at Q = 12 (the state prompt is paid once per 26 questions instead of 11) with the same longest row (24,000,
the cap) and the same outcomes; 28 rows per step = 7 per GPU and ~727 outcomes per optimizer step, close to stage 1's
744 (and to Q = 12's 702 at 64 rows), so the number and size of updates stay as in stage 1's recipe (lr 1e-4, 3%
warm-up, cosine, one epoch). Rows of a state with few outcomes left hold fewer questions; as in stage 1, every row
weighs the same, so their outcomes weigh more.

| | steps | outcomes / step | training hours (2 fits) | dev passes | outcomes per GPU-second |
|---|---:|---:|---:|---:|---:|
| Q = 12, 64 rows/step | 2,834 | 702 | 131.2–131.5 | 0.9 h | 1.05 |
| Q = 36, 64 rows/step | 1,197 | 1,661 | 87.3–91.9 | 0.4 h | 1.50–1.58 |
| **Q = 36, 28 rows/step** | **2,735** | **727** | **87.8–92.5** | **0.9 h** | **1.49–1.57** |

Fits on job 176033's first 18 measured steps (161.5 s mean per step; residual sd 1.6 s): "tokens only" = time
proportional to the slowest rank's tokens (0.881 s per 1,000 tokens); the other adds an attention term per rank
(0.791 s/ktok + 0.0071 s/ktok²; the split between the two terms is weakly determined, so both are given). Dev pass
(stage 1's 120 dev rows): 59.6 s, every 50 steps.

**Wall-clock:** 89–93 h of training; each 24 h link loses ~10 min at start (rows built in ~75 s, model load, row
index) and up to 30 min at the hand-over, so ~23.2 h of training per link: **4 links (a 5th only if the attention
fit is right and the queue gaps add up)**, about 4–5 days on 4 H100s if the links start back to back.

**How a link runs** (`scripts/con1/orchard_full.sh`). The node layout is `orchard_main.sh`'s:
`/tmp/con1_main_<job>/full/` with `metrics.jsonl`, `ckpt/step_N/COMPLETE`, `final/` (stage name `full`), so the
parent's monitor and guard work on it.
1. submits its successor (`-p advanced --qos=adv_4gpu_qos --dependency=afterany:<this job>`) unless `$M/DONE`,
   `$M/STOP` or CHAIN_MAX (8) is reached; two failed links in a row at the same copied step stop the chain;
2. builds the Q = 36 rows on the node's /tmp and stops if the sha1 is not 955b00b1…;
3. trains with `orchard_sft_rows.sh` + `con1_sft.yaml` + overrides `policy.init_adapter=$INIT sft.rows_per_step=28
   sft.save_every=5 sft.keep_last=1 sft.dev_every=50 sft.lazy_rows=true sft.check_rows=false
   sft.resume_dir=<newest con1_reserve/full_ckpt_step_N>`; full checkpoints (adapter + optimizer) every 5 steps on node
   /tmp; dev = stage 1's 120 dev rows, so the dev curve continues stage 1's (dev_start = the stage-1 adapter);
4. runs the guard (`RES=con1_reserve STAGES=full`) every 10 min from inside the job (srun --overlap): the newest
   whole checkpoint (adapter, optimizer, scheduler, rng; ~5.5 GB) to `con1_reserve/full_ckpt_step_N` every ~2 h (one
   kept), the final adapter to `con1_reserve/full_final` as soon as it exists; space from the guard's placeholder
   chunks in `con1_reserve/chunks` (the pool the parent's guard keeps); metrics and the trainer log to `$M`;
5. at SIGUSR1 (30 min before the 24 h limit): waits for the next checkpoint (≤ 15 min), stops the trainer, runs the
   guard with `FORCE=1` (copies that checkpoint now), retries until 5 min before the limit, exits 0; the successor
   resumes **exactly** from the copy (optimizer, scheduler, rng restored; no re-warm needed);
6. at the end of the epoch: the guard copies the final adapter to `con1_reserve/full_final`; `$M/DONE`; successor
   cancelled. A run that has had a checkpoint copied refuses to restart from step 0 if the copy is gone.

**Mechanics tests** (Orchard, 1 GPU, fresh r8 LoRA on small rows):
- First version, adapter-only exports (preempt; 176038, 176043 → 176044): export, hand-over, resume with re-warm,
  DONE. The second link was preempted, requeued, and resumed from the step-30 export. Slurm then ended that requeued
  job seconds after its trainer exited, before its end steps; the copy remained. The end section now copies the
  final adapter before anything else.
- Final version, guard copies (general; 176056 → 176057): the in-job guard copied step 5; USR1 at step 12 → waited
  for checkpoint 16 → `FORCE` copy → exit; the successor resumed from `full_ckpt_step_00016` with optimizer,
  scheduler and rng restored (lr 9.378e-5 = the schedule's), 16 metric lines restored, dev curve continuing (0.184 →
  0.178 → 0.167); `parent_check/mon.py` read `/tmp/con1_main_<job>/full` unchanged.

**Shared storage (the one hard limit).** `/project/community` had 0.85–1.1 GB free during this work. A checkpoint
copy is ~5.5 GB (with the optimizer) and the final adapter 1.87 GB; both come out of the guard's placeholder pool in
`con1_reserve/chunks` (9 GB on 11-26 00:05, kept at up to 16 GB while the volume has > 2.5 GB free). If the pool and
the volume are both short, the copies fail (logged `guard: FAILED ...`, retried every 10 min); a link whose copy at
the hand-over fails leaves the successor to resume from the previous copy (up to ~2 h back). The volume is also
wiped every two weeks (Orchard wiki calendar); the chain reads `repo2m`, `full/`, `main/states_native_offmenu.jsonl.gz`,
`main/rows/*_facet.jsonl`, `gen/out/*.gz`, `data/queries_v2.jsonl`, `gen/bench_problems.jsonl`, `con1_reserve` for
~5 days. Off-Orchard copies of `con1_reserve/full_ckpt_step_*` / `full_final`: the RL session's sha1-checked
`pull_ckpt.sh` / `finish_ckpt.sh` (an operator step from the login side; Orchard reaches neither drc nor trace).

**Submit** (after stage 1's final adapter is in `con1_reserve/stage1_final`, where the parent's guard copies it):

    sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,INIT=/project/community/wxu2/con1_reserve/stage1_final,M=/project/community/wxu2/con1/full/run1 /project/community/wxu2/con1/repo2m/slurm/con1_full.sbatch

Stop the chain: `touch /project/community/wxu2/con1/full/run1/STOP` (the running link finishes; no successor trains).

**Monitor** (the parent's tool, unchanged; from the login node through the job's own allocation):

    srun --jobid=<id> --overlap -N1 -n1 python3 /project/community/wxu2/parent_check/mon.py /tmp/con1_main_<id>/full

loss (first / last 10 steps), informative-token nll, grad norm (median, last-10 max), peak memory, lr, s/step, NaN,
dev passes (stage 1's 120 dev rows) against dev_start (the stage-1 adapter), newest node checkpoint, final. The job's
log (`full/logs/con1_full_<id>.out`) carries the guard's SAVED / FAILED lines; the chain's state is in
`full/run1/chain/` (next_of_<job>, copied_step_N, fail_*). Plan: every 2–3 h; expected ~110–125 s per step at 28
rows (7 per GPU, ~130k tokens per GPU); act on a non-finite loss, a grad-norm spike, rising dev loss, peak memory
above ~76 GiB or FAILED guard lines.

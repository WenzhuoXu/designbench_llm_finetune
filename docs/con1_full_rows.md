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
| `scripts/con1/mon_full.py`, `scripts/con1/mon_full.sh` | live monitor of a running link (node /tmp via srun --overlap) |
| `slurm/con1_full_test.sbatch` | 1-GPU mechanics test of the chain (fresh r8 LoRA, 42 rows) |
| `llm_finetune/rl_dataxp/sft_rows.py` | `sft.lazy_rows`, `sft.check_rows`, `sft.resume_dir`, `sft.rewarm_steps` (all off by default) |

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

**How a link runs** (`scripts/con1/orchard_full.sh`):
1. submits its successor (`-p advanced --qos=adv_4gpu_qos --dependency=afterany:<this job>`) unless `$M/DONE`,
   `$M/STOP` or CHAIN_MAX (8) is reached; two failed links in a row at the same exported step stop the chain;
2. builds the Q = 36 rows on the node's /tmp and stops if the sha1 is not 955b00b1…;
3. trains with `orchard_sft_rows.sh` + `con1_sft.yaml` + overrides `policy.init_adapter=$INIT sft.rows_per_step=28
   sft.save_every=5 sft.keep_last=1 sft.dev_every=50 sft.lazy_rows=true sft.check_rows=false
   sft.resume_dir=$M/export sft.rewarm_steps=10`; full checkpoints (adapter + optimizer) every 5 steps on node /tmp;
   dev = stage 1's 120 dev rows, so the dev curve continues stage 1's (dev_start = the stage-1 adapter);
4. every 2 h copies the newest checkpoint's adapter (1.87 GB) + state.json + metrics to `$M/export`, replacing the
   previous one; after a failed copy it retries every 10 min;
5. at SIGUSR1 (30 min before the 24 h limit) waits for the next checkpoint (≤ 15 min), stops the trainer, exports,
   exits 0; the successor resumes from `$M/export`: weights and step from the export, optimizer restarted, lr
   re-warmed over 10 steps (Adam's first steps after a restart would move each weight by about the full lr);
6. at the end of the epoch: final adapter to `$M/final`, `$M/DONE`, successor cancelled.

**Mechanics test** (Orchard preempt, 1 GPU, fresh r8 LoRA on 42 / 168 small rows; jobs 176038, 176043 → 176044):
lazy rows and the skipped row check, periodic export (in place, 1.0 GB free), USR1 → wait for checkpoint 16 → stop →
export → exit, successor started at once, resumed at step 17 with the export's adapter (checksum checked), lr
3.13e-5 → 6.19e-5 → 9.18e-5 (re-warm over 2 steps, then the scheduled cosine), the 16 earlier metric lines restored,
the dev curve continuing (0.184 → 0.180 → 0.171), DONE, final adapter copied, successor cancelled.

**Shared storage (the one hard limit).** `/project/community` had 0.95–1.1 GB free during this work. An export is
1.87 GB:
- the first export needs 1.87 GB free; later exports replace the previous one in place (net zero; state.json is
  removed first and written last, so a half-written export is never resumed from, and a run that has exported
  before refuses to restart from step 0);
- with ≥ 2.5 GB free an export is written beside the old one and swapped in;
- the final adapter needs another 1.87 GB, or takes the export's place after 30 min of failed copies.
With 1 GB free the first export fails and is retried every 10 min; if it never succeeds, a link's 23 h are lost at
its end. **Free at least ~2 GB (better 4 GB) before submitting.** The volume is also wiped every two weeks (Orchard
wiki calendar); the chain reads `repo2m`, `full/`, `main/states_native_offmenu.jsonl.gz`, `main/rows/*_facet.jsonl`,
`gen/out/*.gz`, `data/queries_v2.jsonl`, `gen/bench_problems.jsonl` and the INIT adapter for ~5 days.

**Submit** (after stage 1's final adapter exists; job 176033 writes it to `main/run3/stage1/final`, its M; the brief
named `main/run2`, whichever holds `adapter_config.json`):

    sbatch -p advanced --qos=adv_4gpu_qos \
      --export=ALL,INIT=/project/community/wxu2/con1/main/run3/stage1/final,M=/project/community/wxu2/con1/full/run1 \
      /project/community/wxu2/con1/repo2m/slurm/con1_full.sbatch

Stop the chain: `touch /project/community/wxu2/con1/full/run1/STOP` (the running link finishes; no successor trains).

**Monitor** (live, from the login node; reads node /tmp through `srun --jobid=<id> --overlap`):

    bash /project/community/wxu2/con1/repo2m/scripts/con1/mon_full.sh <slurm job id>

prints step / total, loss and informative-token nll (last 20 vs the 20 before), dev loss at each pass against the
stage-1 adapter's (dev_start), grad norm (median, last-20 max), seconds per step and the epoch's ETA, peak memory
with the step's longest row, lr, the newest node checkpoint and export, and FLAGS: non-finite loss or grad norm,
grad norm > 5× median, training loss up > 10% over 20 steps, dev loss up at two passes in a row, peak memory > 76
GiB, steps > 30% slower, no step for 20 min. Plan: run it every 2–3 h; on a flag read the trainer log
(`srun --jobid=<id> --overlap -N1 -n1 tail -50 /tmp/con1_full_<id>/logs/train.log`). Expected per-step time
~110–125 s at 28 rows (7 per GPU, ~130k tokens per GPU). Tested on stage 1's running job 176033.

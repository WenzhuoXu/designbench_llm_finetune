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
| `scripts/con1/guard_v3.sh` | the parent's `parent_check/guard_v3.sh` (in-place slots) with env `RES`, `STAGES`, `FORCE`, `NO_FINAL`, `REUSE_SLOT` (unset: the original) |
| `slurm/con1_full_test.sbatch` | 1-GPU mechanics test of the chain (fresh r8 LoRA, small rows) |
| `llm_finetune/rl_dataxp/sft_rows.py` | `sft.lazy_rows`, `sft.check_rows`, `sft.resume_dir` (a checkpoint copied off the node: exact resume when it has the optimizer, else fresh optimizer + `sft.rewarm_steps`) (all off by default) |

Reused, not rebuilt (owner rule): the monitor `parent_check/mon.py` and the checkpoint guard `parent_check/guard_v3.sh`
(a copy with env hooks: the original hardcodes stage names `stage1 stage2` and writes finals into F, which holds the
stage-1 final the continuation starts from).

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

**Format: facetm** (parent's decision 11-26, from con1's drc arm on round-1's 750 rows: facetm vs facet on held-out
problems regime_phys 0.678 vs 0.601, +0.077 ± 0.012; coupled direction 0.683 vs 0.668, +0.015 ± 0.006). The target
code is targets.py's `facetm` through `build_rows.conseq_row`, unchanged (`build_full_rows.py --target facetm`). The
continuation starts from the facet-trained stage-1 final, so it switches format at step 0: facetm only adds the
MARGINS AFTER block before FEASIBLE; its first ~20 steps' loss (logged per step) show the switch; dev = stage 1's dev
states and moves in facetm (`main/rows/dev_facetm.jsonl`), dev_start = the stage-1 final on them. The margin block's
loss is not logged on its own. **Row cap 22,000 tokens** (the ~22k memory limit; stage 1 measured 68.9 GiB on steps
whose longest row was 25.0–25.7k).

**facetm, Q = 12 vs Q = 36** (Orchard CPU job 176126; same 1,988,301 outcomes; exact tokens; checks as in §3: no pair
twice, none of stage 1's, counts add up; the Q = 36 build repeated with 64 workers: identical bytes):

| | facetm Q = 12 | **facetm Q = 36** | facet Q = 36 (cap 24k, §3) |
|---|---:|---:|---:|
| rows | 198,873 | **121,670** | 76,558 |
| questions per row | 10.0 | **16.3** | 26.0 |
| tokens per row mean / p95 / max | 14,763 / 21,716 / 22,000 | **20,073 / 21,860 / 22,000** | 18,517 / 23,669 / 24,000 |
| prompt tokens per row | 8,949 | 10,300 | 8,544 |
| training tokens | 2,935.9 M | **2,442.2 M** | 1,417.6 M |
| tokens per outcome | 1,477 | **1,228** | 713 |
| rows file | 7.8 GB | 6.6 GB | 4.2 GB |
| build (30 CPUs) | 159 s | 98 s | 74 s |
| sha1 | 181dc4e7… | **5f0c9acb07e4433b5b7975dd698d9b2ff9a50649** | 955b00b1… |

facetm answers are longer (one margin line per constraint) and every question repeats the full facetm format text,
so the 22,000 cap binds for most Q = 36 rows (p95 21,860): 16 questions per row on average, grid2op only 7.5 (its
prompt is long). Per family (Q = 36): circuit2 18.1 questions/row, engibench 24.4, frame3d2 18.2, grid2 19.3, grid2op
7.5, photonics 13.4, seir2 20.5, traffic2 28.2; grid2op's 34,015 rows are 28% of all rows for 13% of the outcomes.

**Short later questions (parent's change, 11-26 08:50).** The full answer-format text (~200 tokens, masked from the
loss) is given once, in a row's first question; later questions read "QUESTION (no move is made now): if the move X
were played at the current design, what would the simulator return? Same answer form."
(`build_full_rows.py --follow short`; targets.py untouched, so the held-out test queries keep the full text in every
question and their scores stay comparable; the drc per-link eval asks single questions, k = 0, which match a row's
first question exactly). The dev rows (stage 1's dev states / moves in facetm) get the same later-question wording on
the node at job start (1,192 later questions; answers and first questions unchanged).

| facetm, cap 22,000, Q = 36 | full text in every question | **"Same answer form." after the first** |
|---|---:|---:|
| rows | 121,670 | **78,236** |
| questions per row | 16.3 | **25.4** |
| tokens per row mean / p95 / max | 20,073 / 21,860 / 22,000 | **15,309 / 21,830 / 22,000** |
| training tokens | 2,442.2 M | **1,197.7 M** |
| tokens per outcome | 1,228 | **602** |
| sha1 | 5f0c9acb… | **cb197f297f6fd5a249175d3844d431b9a7b159b5** |

(Orchard CPU job 176142: checks pass, the 64-worker rebuild is byte-identical.) Questions per row with short later
questions: circuit2 27.1, engibench 24.8, frame3d2 31.3, grid2 30.3, grid2op 15.0, photonics 17.8, seir2 30.4, traffic2
30.3.

**Think-closing piece not supervised (owner-approved, 11-26 09:00).** Every answer of the continuation's rows carries
`"prefix_weight": 0.0` (`build_full_rows.py --prefix-weight 0`; conseq_format._span_mask's existing key, as the P rows
used): the empty reasoning + `\n</think>\n\n` before each answer weighs 0; the answer content keeps 1 on informative
tokens and 0.05 (`fixed_weight`) on fixed wording, the end token 1. Reason: stage 1 trains the model to close its
think block at once, and answer-only world-model SFT hurt the agent in RWML (arXiv 2602.05842); RL with thinking comes
later. Stage 1 (176033) is unchanged. The dev rows get the same weight on the node. Checked on built row 0 (engibench,
27 questions) with the trainer's own `conseq_format.encode` (`scripts/con1/dump_weights.py`): `\n`, `</think>`, `\n\n`
weight 0; "VERDICT:", metric names, padding 0.05; " applied", " satisfied 20-50%", " no", " volume", " same" 1;
`<|im_end|>` 1; over the row 4,113 completion tokens = 1,425 at 0 (user turns + the 27 think-closing pieces), 1,512 at
0.05, 1,176 at 1; all 27 `</think>` tokens 0. Rows file sha1 47edac9b76e8ec40074aa25990e3d9726728120c (CPU job 176148;
checks pass; rebuild byte-identical); tokens and hours unchanged (weights only).

**Choice: Q = 36, short later questions, 28 rows per step** (mine): 7 rows per GPU, ~711 outcomes per optimizer step
(stage 1: 744), 2,795 steps, lr 1e-4, 3% warm-up, cosine, one epoch.

| facetm, Q = 36 | steps | outcomes / step | training hours (2 fits) | dev passes | outcomes per GPU-second |
|---|---:|---:|---:|---:|---:|
| full text, 44 rows/step | 2,766 | 719 | 150.6–158.6 | 0.9 h | 0.87–0.92 |
| **short, 28 rows/step** | **2,795** | **711** | **73.8–75.8** | **0.9 h** | **1.82–1.87** |

Fits on 176033's first 188 measured steps (tokens only: 0.881 s per 1,000 tokens on the slowest rank; with an
attention term 0.805 s/ktok + 0.0061 s/ktok²; dev pass 59 s).

**Wall-clock:** 75–77 h of training; each 24 h link loses ~12 min at start (rows built in ~70 s, slot restore,
model load, row index) and up to 30 min at the hand-over, so ~23.2 h of training per link: **4 links**, about
3.3–3.5 days on 4 H100s if the links start back to back (CHAIN_MAX 10).

**How a link runs** (`scripts/con1/orchard_full.sh`). Node layout as `orchard_main.sh`'s
(`/tmp/con1_main_<job>/full/` with `metrics.jsonl`, `ckpt/step_N/COMPLETE`, `final/`), so `parent_check/mon.py`
reads it. Copies off the node use the parent's in-place slot method (`parent_check/guard_v3.sh`, 11-26 05:4x: the
shared volume sits at ~170 MB free and any space released is taken by other users within seconds, so nothing is ever
released): a rolling copy is a tar of `<ckpt>/adapter` + `state.json` written in place into
`con1_reserve/slots/A.bin` or `B.bin` (alternating; `A.meta` / `B.meta` = "stage step length sha1 time", written
after a sha1 read-back); `F1–F3.bin` + `F.meta` hold a stage's final adapter. Optimizer state does not fit.
1. submits its successor (`-p advanced --qos=adv_4gpu_qos --dependency=afterany:<this job>`) unless `$M/DONE`,
   `$M/STOP` or CHAIN_MAX (8) is reached; two failed links in a row at the same copied step stop the chain;
2. start adapter, always at the fixed node path `/tmp/con1_init` (the trainer's fingerprint holds the path):
   a resuming link restores the newest `full` slot (A or B; sha1 checked against its meta) into `/tmp/con1_resume`
   and hard-links its adapter as `/tmp/con1_init` (only its LoRA config matters there: the resume overwrites every
   weight); the first link (`INIT=F`, default) restores the stage-1 final from F1–F3 per `F.meta`, which must read
   `stage1 final ...` (else the link stops); `INIT=<dir>` copies an adapter directory instead; a run that has
   copied a checkpoint before never restarts from step 0;
3. builds the facetm Q = 36 rows (short later questions, think-closing weight 0, cap 22,000) on the node's /tmp and
   stops if the sha1 is not 47edac9b…;
4. trains with `orchard_sft_rows.sh` + `con1_sft.yaml` + overrides `policy.init_adapter=/tmp/con1_init
   sft.rows_per_step=28 sft.save_every=5 sft.keep_last=1 sft.dev_every=50 sft.lazy_rows=true sft.check_rows=false
   sft.resume_dir=/tmp/con1_resume|null sft.rewarm_steps=10`: a resume restarts AdamW, re-warms the lr over 10 steps
   and takes the schedule position from state.json; dev = stage 1's 120 dev states / moves in facetm (dev_start =
   the start adapter);
5. runs the guard (`scripts/con1/guard_v3.sh` = the parent's 11-26 06:2x version + env `RES`, `STAGES`, `FORCE`,
   `NO_FINAL`, `REUSE_SLOT`; here `STAGES=full NO_FINAL=1 REUSE_SLOT=1`) every 10 min from inside the job: the newest
   checkpoint into the older slot every ~2 h; metrics and the trainer log to `$M`. The parent's guard keeps the
   stage-1 final in F1–F3 and puts stage 2's final into the older of A/B, which rolling writes never overwrite;
   alternating targets would then skip every write after the first, so `REUSE_SLOT=1` rewrites the one remaining
   slot (checked on fake checkpoints, CPU job 176095: original skips step 10, REUSE_SLOT writes it into A, B's final
   untouched, A restores to the step-10 adapter byte for byte). While stage 2's final holds a slot, the continuation
   has one rolling slot: its only copy is invalid during each ~1 min write;
6. at SIGUSR1 (30 min before the 24 h limit): waits for the next checkpoint (≤ 15 min), stops the trainer,
   FORCE-writes it into a slot (retrying until 5 min before the limit), exits 0; the successor resumes from it. At the
   end of the epoch the last checkpoint (its adapter = the final adapter) is FORCE-written the same way and `$M/DONE`
   names the slot, step, length and sha1; successor cancelled.

**Mechanics tests** (Orchard general, 1 GPU, r8 LoRA on 168 small rows, test slots of 8–16 MB):
- Slots, `INIT=none` (176088 → 176092): in-job guard wrote step 8 into A; USR1 → checkpoint 15 → FORCE into B →
  exit; the successor restored B (sha1 checked), resumed at step 16 (lr 8.6e-6 → 1.7e-5 → 2.5e-5, re-warm), trained
  to step 84; the last checkpoint went into A and DONE named it. (The first successor, 176089, failed on a bug in
  the restore, `cat` | `head -c` under pipefail; fixed and re-run.)
- Slots, `INIT=F` (176093 → 176094): the test F1–F3 were filled with phase 1's final the way the guard writes a
  final; link 1 restored it (dev loss before training 0.164, phase 1 ended at 0.166), wrote step 5 into A; USR1 →
  step 16 into B; the successor restored B, used its adapter as `/tmp/con1_init`, kept the fingerprint
  (eb3eb4ca43f41b33) and resumed with the re-warm.
- Earlier versions (adapter-only directories, whole checkpoints, preemption + requeue: 176038–176080) are in git
  history (commits 5dd665b–df508e8).

**Shared storage.** The volume sat at 0.1–0.2 GB free on 11-26 morning; the slot method writes only into files
already held. Two points for the parent: (a) `B.bin` is 1,867,797,544 bytes and an adapter tar 1,867,816,960, so
a write into B extends it by 19,416 bytes (needs that much free space); (b) `guard_v3.sh` wrote each stage's final
into the same F1–F3 (fixed by the parent on 11-26: F holds only the stage-1 final; stage 2's final goes into the
older of A/B, which leaves the continuation one rolling slot until that final is taken off Orchard and its slot
released). The continuation's own final ends in A or B (named in `$M/DONE`) and should be
taken off Orchard (RL session's sha1-checked `pull_ckpt.sh` / `finish_ckpt.sh`) before anything writes those slots
again. The chain also reads `repo2m`, `full/`, `main/states_native_offmenu.jsonl.gz`, `main/rows/*_facet.jsonl`,
`gen/out/*.gz`, `data/queries_v2.jsonl`, `gen/bench_problems.jsonl` for ~5 days; the volume is wiped every two weeks.

**Submit** (after stage 1 is scored, with the stage-1 final in F1–F3, F.meta reading `stage1 final ...`):

    sbatch -p advanced --qos=adv_4gpu_qos --export=ALL,M=/project/community/wxu2/con1/full/run1 /project/community/wxu2/con1/repo2m/slurm/con1_full.sbatch

Stop the chain: `touch /project/community/wxu2/con1/full/run1/STOP` (the running link finishes; no successor trains).

**Monitor** (the parent's tool, unchanged; from the login node through the job's own allocation):

    srun --jobid=<id> --overlap -N1 -n1 python3 /project/community/wxu2/parent_check/mon.py /tmp/con1_main_<id>/full

loss (first / last 10 steps), informative-token nll, grad norm (median, last-10 max), peak memory, lr, s/step, NaN,
dev passes (stage 1's 120 dev rows) against dev_start (the stage-1 adapter), newest node checkpoint, final. The job's
log (`full/logs/con1_full_<id>.out`) carries the guard's SAVED / FAILED lines; the chain's state is in
`full/run1/chain/` (next_of_<job>, copied_step_N, fail_*). Plan: every 2–3 h; expected ~90–100 s per step at 28
rows (7 per GPU, ~107k tokens per GPU); act on a non-finite loss, a grad-norm spike, rising dev loss, peak memory
above ~76 GiB or FAILED guard lines.

## 5. Accuracy per link (drc, Orchard keeps training)

At the end of each 24 h link the newest `full` slot (A or B) holds that link's checkpoint (its successor's log names
it: "resuming from slot X (full step_N ...)"). One pass per link, while the next link trains:

1. **Orchard → local** (owner's PC, bytes only): `LABEL=full_<step> bash scripts/con1/pull_slot.sh A|B` — the RL
   session's route (its `orr.sh` / `orch.sh` over the IAP forward, 1800 MiB byte-range chunks of the slot's valid
   [0, length) bytes, each chunk's sha1 checked against Orchard's, 3 tries; the meta must be unchanged after the pull
   and the joined parts must give the meta's sha1) → `D:/designbench_con1/full_xfer/con1_full__<label>/`.
2. **local → drc**: MCP upload of that directory to `/storage/wxu/rl_open/orchard_xfer/` and the `.remote.sha1` file
   beside it (≤ 2 GB per call; one part each), then on drc the RL session's
   `bash /storage/wxu/rl_open/orchard_xfer/join_ckpt.sh con1_full <label>` → "1 files DRC_SHA_MATCH",
   `/storage/wxu/rl_open/orchard_runs/con1_full/ckpt/<label>/slot.tar`. No MCP download from drc.
3. **drc eval** (con1's OK, 11-26 09:00): `bash /storage/wxu/con1/full_eval/eval_link.sh <label>`
   (`scripts/con1/eval_link.sh`): the adapter to `con1/arm/run_c2m_<label>/final`, slot.tar deleted; con1's
   generalised `drc_arm.sh` (`drc_arm_next.sh` until it replaces drc_arm.sh) with `LAB=c2m_<label> FMT=facetm
   ROWS=none WAIT=1 QUERIES=arm/main/queries_v2.jsonl FAM=all MAXTOK=4096 OUTDIR=full_eval PASSES=zero`: final exists,
   so no training; GPU 4 under con1's flock `arm/gpu4.lock`; the main-run test set (held-out problems, 8 families,
   3,687 questions with native off-menu moves); facets without the harness next to stage 1's answers
   (`full_eval/preds_stage1.jsonl`, from 176033's slot P) and the main-run trees (`arm/main/preds_trees.jsonl`);
   ~60–75 min of GPU 4 plus the server start; `full_eval/curve.txt` rewritten with one line per checkpoint and the
   comparison lines.
4. The curve's first point: the stage-1 final from the F slots (`pull_slot.sh F`, label stage1_final), asked the same
   facetm questions.

The adapters stay on drc (`con1/arm/run_c2m_*/final`, 1.9 GB each; /storage had 45 GB free on 11-26), so they are
also the off-Orchard copies of each link; slot.tar and the local parts go after the extraction / DRC_SHA_MATCH.
Transfer checked on 11-26 with the real slot A (stage-1 step 120, 1,867,816,960 bytes): pull 1 min, upload 54 s,
"1 files DRC_SHA_MATCH", tar = adapter/ + state.json with stage 1's fingerprint; test copies removed.

## 6. Link 1 and the move request (11-27)

**Link 1** (176234, 11-26 15:39 – 11-27 15:19): stopped for the time limit at step 850 of 2,795, slot B
(`full step_00850`, sha1 ef5f356…); loss (mean of 10 steps) 0.0704 → 0.0463, dev loss 0.0997 → 0.0504 (step 799),
grad norm flat (median 0.046), 96.5 s/step, peak 68.9 GiB. 176235 (link 2) is held by the parent (owner order: no GPU
hours on rows that carry the move request).

**Eval of step 850** (drc; all numbers also from `indep_score.py`, a second scorer without facets.py, which matches
facets to 3 decimals). Standard protocol (con1's skill_client: the prompt ends `<think>\n`, the answer is what follows
the model's own `</think>`): unreadable 27% (v2) / 57% (v3) on the trained families, 32% / 72% on the unseen ones —
the model ends its turn without writing `</think>` (370 of the v2 failures after one token). The think-closing piece
was trained at weight 0 (§4, owner 11-26), so the model no longer closes the think block itself. With the block closed
in the prompt (`closed_think.sh`: think_client `--budget 0`), 99.9–100% readable, and against the stage-1 final
(regime_phys / cpl_dir_BA / cpl_size, paired se over states):

| | v2 | v3 |
|---|---|---|
| trained families | −.021±.012 / +.026±.005 / +.070±.006 | −.015±.009 / +.016±.007 / +.070±.005 |
| unseen families | −.028±.012 / −.001±.007 / −.018±.007 | −.023±.012 / −.002±.007 / −.015±.006 |

Files: drc `full_eval/` (v2), `full_eval/v3/` (v3, `compare_full_00850.txt`), `full_eval/closed/report_full_00850.txt`.

**Rows without the move request** (owner order 11-27; con1 eff023a `conseq_lib.consequence_user`): `full_prep_nc.sh`
(Orchard 176464) rebuilt the set with the same settings in `repo2m_v3`: tag `facetm_short_pw0_nc`, train sha1
29b3a469cfcf1e9db7b32be1fd435805ac08db0d, 77,244 rows (running set 78,236: 628 states the cap split finer on the
longer prompt), the same 1,988,301 questions and 27,162 states per split and family, 1,165.9M tokens (−2.7%).
`full_noconf_check.py`: no move-request mark in any row, the OPERATIONS heading once per prompt; `full_pair_check.py`:
every row identical to the running set's except the cut prompt (70,338 rows; the 628 re-cut states: every answer
identical); a second build gave the same bytes. Stage 1's dev rows with the cut: `full/dev_stage1_facetm_nc.jsonl`.
Not trained on (the parent decides). The continuation's think-closing weight (0) is the cause of the broken protocol
above; any next link needs that decided too.

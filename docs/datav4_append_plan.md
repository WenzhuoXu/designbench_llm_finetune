# datav4 append plan: batches b002 to b005

Written 2026-10-24. b001 is the datav4 delivery of 2026-10-23 (`docs/corpus_native_domains.md`,
"datav4 delivered"): `datav4_multitool/rows/`, `datav4_set_parameter/rows/` and the shared
`datav4_corpus/` on bridges2 (`/ocean/projects/mch250030p/shared/dataxp/`). **The owner is working on
those files. Nothing in them is modified, moved or deleted.** New data is appended only as new
directories:

| what | where (bridges2, under `/ocean/projects/mch250030p/shared/dataxp/`) |
|---|---|
| rows of batch b00N | `datav4_<spelling>/batches/b00N/`: `rows/<family>/{train_own,train_own_act,dev_in_own,dev_in_own_act}.jsonl` + `manifest_own.json`, `rows/_groups/<group>/`, `manifest.json`, `BATCH.md` |
| new corpus problems of b00N | `datav4_corpus/<family>/accepted_b00N/`: `{problems,trajectories,pairs}.jsonl`, `report.json`, `sources.json` |

The top-level README, manifest, instruction and grammar files are not updated. Each batch is
self-describing (`BATCH.md` + `manifest.json`), and its served instruction and grammars are checked to
equal the spelling directory's `instruction.txt` and `grammar/<family>.txt`.

## Batches

| batch | content | families | states | replies (both spellings) | GPU-h (Orchard, estimate) |
|---|---|---|---|---|---|
| b002 | redraws for b001 states with no row: the K=4 replies of b001 are kept; the one redraw pass b001 never ran redraws the truncated (`finish == length`) keys | the 8 training families | 1,602 (no row in either spelling; 29 dev) | 3,391 multitool + 3,201 set_parameter truncated keys | ~3 |
| b003 | the unused run2 problems (0000-0129, none used so far) + the run2_supp problems b001 did not select (0130-0259) | circuit2, frame3d2, grid2, seir2, traffic2 | run2 3,087 + run2_supp 455 = 3,542 | ~39,000 (K=4 + redraw of ~38%) | ~17 |
| b004 | new native problems, indices 130-259 (`run_generation --start-index 130 --n 130`) | grid2op, engibench, photonics | ~830 (b001 rate: 2.4, 2.3, 1.7 states per problem) | ~8,300 | ~4 |
| b005 | new v2 problems, indices 260-389 (`--start-index 260 --n 130`) | circuit2, frame3d2, grid2, seir2, traffic2 | ~3,100 (run2's 3,087 for the same count) | ~34,000 | ~15 |

The b002 counts come from the b001 work files on drc (census 2026-10-24: 3,272 states, the same set in
both spellings; 2,062 without a multitool row, 1,970 without a set_parameter row, 1,602 without either).
Per family, the b002 states and truncated keys (multitool / set_parameter) are: circuit2 400 (917 / 963),
grid2 358 (945 / 888), frame3d2 229 (478 / 421), traffic2 212 (562 / 501), seir2 125 (90 / 80),
engibench 106 (170 / 131), photonics 99 (161 / 125), grid2op 73 (68 / 92). GPU estimates use b001's
rate (11.5 GPU-h for ~26,000 replies in both spellings, about 2,300 replies per GPU-hour).
Generation for b004 and b005 runs on drc CPUs and needs no GPU (b001 timings: engibench 4.8 h on 8
workers, photonics 6.7 h on 4, grid2op 1.6 h; the v2 five are minutes each).

## Rules (every batch, checked by the tools)

1. **Both spellings, same states.** Each batch is sampled once under `DATAXP_SPELLING=multitool` and once
   under `set_parameter`, over the same state set (same sids, same splits). `package_datav4 --append
   --twin-states <other spelling's rows root>` refuses a batch whose state set or splits differ.
2. **The b001 recipe.** Own-action rows; K=4 replies per state from the un-tuned Qwen3.8-27B, thinking
   off, 2,000-token cap, temperature 0.6, `--prompt probe`; acceptance = the best pick is at or above the
   90th Φ percentile of the legal set or equals the teacher's move; `--max-words 320`; plus **one** redraw
   pass of the truncated keys (`sample.py` re-run on the same output redraws exactly those). The packager
   checks what the files record: K per state, prompt `probe`, mode `own_nothink` (no dry-run replies),
   the model name, no thinking text, reply tokens <= 2,000, `min_pct` 90 / `max_words` 320 in the tier
   manifest, and no reply drawn in a pass above `--draw-passes`.
3. **Rows record their batch.** Every row gets `"batch": "b00N"`, `"draw_passes"` (the passes the batch
   ran: 2 = the K-reply draw + the redraw pass) and, when the sampler recorded it, `"draw_pass"` (the
   pass that drew the row's own reply).
4. **Held out, never given rows:** thermal2, inventory2, molopt. A batch containing any of their rows is
   refused.
5. **Evaluation problems are never used for rows.** Native: the last 60 by index of each native family's
   first 130 (`/storage/wxu/corpus/harness/problems_v3_native.jsonl` on drc). v2:
   `/storage/wxu/sftprobe/problems_v2.jsonl` on drc. Every new training problem is checked by canonical
   spec (`dataxp_policy_eval_v3`'s `canon()`: JSON of the spec without `problem_id` / `domain_variant` /
   `index`, sorted keys) against both files, and by id. `select_states --exclude-specs` leaves such
   problems out before sampling; `package_datav4 --append` drops any row of one (counted in the manifest
   and BATCH.md); the batch `--check` refuses one.
6. **No state twice.** No row whose sid (`domain|instance|turn`) or `(instance, turn)` appears in any
   prior batch. `--prior <spelling dir>` reads its `rows/` (b001) and every `batches/b*/rows/`.
7. **Splits are per problem and permanent.** A problem's states are all train or all dev, and a problem
   that already has rows in an earlier batch keeps that batch's split (`--prior`; `--prior-states` adds
   problems that had states but no rows). `select_states --split-from` applies the prior split at
   selection time.
8. **Same served text as b001.** The served instruction must equal the spelling directory's
   `instruction.txt` and each family's grammar its `grammar/<family>.txt`; every row's user text must end
   with that instruction + grammar; every action line must re-parse under the spelling to the stored
   action.

## Decisions for the owner before sampling

These choices are mine, made to satisfy the rules above. None is applied to bridges2 until the owner
agrees.

1. **b002's state set is the states with no row in EITHER spelling (1,602).** Rule 1 asks for the same
   states in both spellings, and rule 6 forbids a second row for a state. Together they exclude the 368
   states that have a row in multitool only and the 460 that have one in set_parameter only. Those states
   are not redrawn in the spelling that lacks the row. The alternative, per-spelling sets (2,062 / 1,970),
   breaks rule 1.
2. **b002 keeps b001's finished replies and redraws only the truncated keys** (draw pass 2). That is the
   b001 recipe completed by its missing redraw pass. A fresh K=4 draw for these states would be a second
   recipe, since the hard states would then get 8 or more replies. In b002 a b001 reply can never become
   the row's pick: the state was dropped with those replies.
3. **engibench base designs repeat.** engibench builds problem i on dataset design `i // 3` modulo the
   dataset size (beams2d 40, heatconduction2d 80, photonics2d 40). Problems 130-259 therefore reuse base
   designs. 40 of them (20 beams2d, 20 photonics2d) share their `(family, source_row)` with an
   evaluation problem (indices 70-129). Their specs differ (perturbation, goals), so the canonical check
   passes them. b001 already has 7 such training problems (beams2d rows 84/148/212/232, photonics2d rows
   4/100/116). Options: (a) keep them, since the rule is spec identity; (b) leave the 40 out (`select_states
   --exclude-problems` on the list in `sources.json`, key `shared_base_with_eval`). My recommendation is
   (b) for b004. The 7 in b001 are reported and not touched.
4. **Quotas.** b001 took up to 600 train + 12 dev states per v2 family, and every native problem <= 68/69
   (train only). The plan takes **every** remaining problem into train for b003, b004 and b005
   (`--per-domain 0`). Dev stays b001's (b001 drew dev states from the v2 families only). For a dev slice from the new problems,
   select with `--per-domain <n> --dev-per-domain 12` instead.
5. **v2 index stream.** `run_generation --start-index 260` continues the same seed stream
   (`stable_seed(variant, 0, i)`). On drc the current code rebuilds run2_supp's problem 0130 exactly for
   circuit2 and traffic2. For frame3d2 and seir2 it differs only in the last printed digit of one or two
   floats (machine round-off). For grid2 the same problem differs in which of two lines carries the
   switch / phase shifter, an argsort tie. Indices 260-389 are new indices, so this cannot cause a
   duplicate; the corpus append still checks ids and canonical specs against run2 and run2_supp.

## What changed in the code (2026-10-24)

| file | change |
|---|---|
| `dataxp/run_generation.py` | `--start-index K` (default 0 = the old code path, byte-identical): indices K..K+n-1, ids `<variant>_<i:04d>`, each built exactly as a run of `--n K+n` builds it (`build_specs`). For grid2op and photonics, whose modules keep a per-process registry of built problems (`_BUILT`) and redraw a repeat, and for engibench and molopt (listed in `PREBUILD_VARIANTS`: no registry, but engibench's photonics2d specs depend on the process history at round-off level), indices 0..K-1 are built first, registry only, not evaluated. `report.json` gains `start_index`, `problem_indices` and `prebuilt` only when K > 0 |
| `dataxp/ownrows/select_states.py` | `--exclude-problems` (ids from states / rows / spec files or a text list), `--exclude-specs` (canonical-spec exclusion against problem files, e.g. the two evaluation files), `--split-from` (a prior problem keeps its split while that split's quota is open). Without them the draw is unchanged; `select_report.json` records exclusions and the unused problems |
| `dataxp/ownrows/sample.py` | every record carries `draw_pass` (1 + records of that key already in the file). `--max-draw-pass N` (or `DATAXP_MAX_DRAW_PASS`): a key with N records is left alone. The first job runs with 1, so a requeue cannot redraw early. The redraw job runs with 2 |
| `dataxp/ownrows/batch_prep.py` (new) | `unrowed` (states of an earlier batch without a row in any given tier file, with their Φ-cache lines; `--relabel-spelling` sets b001's legacy multitool labels `verb_surface` / `native` / missing to `multitool` after checking the instruction sha1, as `package_all.sh` did for the b001 rows), `seed-shards` (shard files holding every record, finished or truncated, so a redraw over any GPU count labels pass 2 correctly; old shard files are moved, never deleted), `phi-from` (the set_parameter Φ cache = the set_parameter state records + the multitool Φ fields; the b001 stale-cache lesson) |
| `dataxp/package_datav4.py` | `--append` (batch directory), `--corpus-batch` (`<family>/accepted_b00N/` + `sources.json`), and `--check` for both (detects a batch by `manifest.json` `kind: batch`) |

### `package_datav4 --append` in one paragraph

Inputs: `--rows-root` (the batch's `<group>/{states.jsonl, samples.jsonl, scored.jsonl, tiers/}` work
dirs), `--prior` (earlier rows, read-only), both evaluation files, `--draw-passes`, optional
`--twin-states`, `--prior-states`, `--source-job GROUP=TEXT`, `--note`. Every row gets the main
build's checks: spelling, instruction sha1, action-line form, served grammar tail, own = own_act row for
row, train/dev disjoint, and a re-parse in a subprocess under the spelling. On top of that come rules
3-8 above, plus no sid twice within the batch and no problem with states in both splits. The recipe is
checked against the samples. A failing check leaves nothing behind (partial output removed). Output:
`rows/<family>/*` (stamped), `rows/_groups/<group>/`, `manifest.json` (counts, the sha1 of every file,
instruction and grammar texts, per-problem canonical-spec sha1s, prior trees checked, evaluation files
with sha1s, source jobs, observed recipe, warnings) and `BATCH.md` (recipe, the funnel per family
states -> replies drawn -> finished -> usable -> rows, checks, sources). `--check --out <batch dir>
[--prior ...] [--eval-problems-* ...] [--parse-check]` re-verifies everything from the batch directory
alone. The prior and evaluation checks are repeated when those paths are given.

## Commands

All drc work runs as a script written with `remote_write` and started with `submit_job` on drc
(`resources.gpus = 0`, CPU only; drc has no scheduler, so it starts at once, and 24 cores are shared with
other users: `nice -n 10`, modest `--workers`). drc's `/storage` had 2.7 GB free on 2026-10-24, and a
batch needs a few hundred MB. Orchard work runs as `sbatch` jobs. Transfers go drc -> local -> bridges2
with `download` / `upload`, **one remote glob per download call**.

### Common environment

```bash
# drc
P=/storage/wxu/corpus; PY=$P/env/bin/python
export PYTHONPATH=$P/code_v3 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATAXP_MOLOPT_ORACLE_DIR=/home/wxu/.dataxp/molopt_oracle/oracle
EVALN=$P/harness/problems_v3_native.jsonl
EVAL2=/storage/wxu/sftprobe/problems_v2.jsonl
EVAL="--eval-problems-native $EVALN --eval-problems-v2 $EVAL2"
SUPP=/storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259
DV=/storage/wxu/datav4            # drc mirror of the bridges2 layout; batches are built inside it
# b001 work dirs: $P/rows/{v2verb,grid2op,engibench,photonics} (multitool), $P/rows_sp/{v2,grid2op,engibench,photonics}
# batch work dirs: $P/b00N/rows/<group> (multitool), $P/b00N/rows_sp/<group> (set_parameter); groups v2 grid2op engibench photonics
```

```bash
# Orchard (account ccm): sampling = scripts/sample_job.sh (N vLLM servers, one per GPU; shards; merge)
C=/project/community/wxu2/corpus
# job 1 (K-reply draw):  OWN_SMOKE=1 DATAXP_MAX_DRAW_PASS=1
# job 2 (redraw pass):   OWN_SMOKE=0 DATAXP_MAX_DRAW_PASS=2, SAME GPU count as job 1 (else run seed-shards first)
sbatch -A ccm -p advanced --qos=adv_4gpu_qos --gres=gpu:4 -c 16 --mem=128G -t 24:00:00 -J own_b00N_mt \
  -o $C/logs/%j.out --export=ALL,OWN_ROWS=$C/b00N/rows,OWN_DOMAINS="v2 grid2op engibench photonics",OWN_SMOKE=1,DATAXP_MAX_DRAW_PASS=1 \
  $C/scripts/sample_job.sh
sbatch ... -J own_b00N_sp --export=ALL,DATAXP_SPELLING=set_parameter,OWN_ROWS=$C/b00N/rows_sp,OWN_DOMAINS="...",OWN_SMOKE=1,DATAXP_MAX_DRAW_PASS=1 $C/scripts/sample_job.sh
# advanced is ONE node per user across all tracks: if another track holds it, wait, or use
#   general: -p general --qos=general_qos --gres=gpu:1 -c 4 --mem=64G -t 12:00:00   (1 GPU, 1 job)
# never cancel another track's job. preempt (32 GPUs, requeued) is safe for sampling since the
# sampler resumes and DATAXP_MAX_DRAW_PASS keeps a requeue from redrawing early:
#   -p preempt --qos=preempt_qos --gres=gpu:4 --requeue
# (do not use sample_job_sp.sh for batches: it hard-codes OWN_ROWS=$C/rows_sp; pass DATAXP_SPELLING instead)
```

Only one 4-GPU job runs at a time, so the two spellings run one after the other, or side by side on
general / preempt. The smoke stage draws 2 states per shard into `smoke/` (as in b001). A smoke reply
that ends truncated is drawn again in the full stage, so those few keys get one extra draw. This is
recorded and the same as b001.

### Shared tail of every batch (after sampling): transfer, score, package, check, upload

```bash
# 1. samples Orchard -> local -> drc (one glob per call; per spelling)
#    download orchard  $C/b00N/rows/*/samples.jsonl     -> <local>/b00N/rows/
#    download orchard  $C/b00N/rows_sp/*/samples.jsonl  -> <local>/b00N/rows_sp/
#    upload   drc      <local>/b00N/rows/<g>/samples.jsonl -> $P/b00N/rows/<g>/samples.jsonl   (each group; same for rows_sp)
# 2. score + tiers (drc)
ROOT=$P/b00N/rows    SPELLING=multitool     GRPS="v2 grid2op engibench photonics" bash $P/bw_score.sh
ROOT=$P/b00N/rows_sp SPELLING=set_parameter GRPS="v2 grid2op engibench photonics" bash $P/bw_score.sh
# 3. package + check (drc), both spellings; the batch is built inside the drc mirror
for sp in multitool set_parameter; do
  if [ $sp = multitool ]; then W=$P/b00N/rows; TW=$P/b00N/rows_sp; else W=$P/b00N/rows_sp; TW=$P/b00N/rows; fi
  $PY -m dataxp.package_datav4 --append --spelling $sp --batch b00N --rows-root $W \
      --out $DV/datav4_$sp/batches/b00N --prior $DV/datav4_$sp $EVAL --draw-passes 2 --twin-states $TW \
      --source-job "v2=orchard <job ids>" --source-job "grid2op=orchard <job ids>" ...
  $PY -m dataxp.package_datav4 --check --out $DV/datav4_$sp/batches/b00N --prior $DV/datav4_$sp $EVAL --parse-check
done
# 4. drc -> local -> bridges2 (new directories only)
#    bridges2: test ! -e /ocean/projects/mch250030p/shared/dataxp/datav4_multitool/batches/b00N   (must not exist)
#    download drc $DV/datav4_multitool/batches/b00N/**      -> <local>/up/b00N/multitool/
#    download drc $DV/datav4_set_parameter/batches/b00N/**  -> <local>/up/b00N/set_parameter/
#    upload bridges2 <local>/up/b00N/multitool     -> .../datav4_multitool/batches/b00N
#    upload bridges2 <local>/up/b00N/set_parameter -> .../datav4_set_parameter/batches/b00N
# 5. verify on bridges2 (python3 on the login node; reads only)
cd /ocean/projects/mch250030p/shared/dataxp/datav4_multitool/batches/b00N && python3 -c "import json,hashlib;m=json.load(open('manifest.json'));bad=[r for r,f in m['files'].items() if hashlib.sha1(open(r,'rb').read()).hexdigest()!=f['sha1']];print(len(m['files']),'files',len(bad),'bad',bad[:5])"
```

### b002: the redraw pass b001 never ran

```bash
# drc: states = b001 states without a row in either spelling (+ their Phi lines); shard files seeded with
# b001's replies (finished and truncated) for N = the GPU count of the Orchard job (4 here)
ROWS_BOTH="$(ls $P/rows/*/tiers/train_own.jsonl $P/rows/*/tiers/dev_in_own.jsonl $P/rows_sp/*/tiers/train_own.jsonl $P/rows_sp/*/tiers/dev_in_own.jsonl)"
for g in v2 grid2op engibench photonics; do
  gm=$g; [ $g = v2 ] && gm=v2verb                      # b001's multitool v2 group is v2verb
  $PY -m dataxp.ownrows.batch_prep unrowed --states $P/rows/$gm/states.jsonl --states-phi $P/rows/$gm/states_phi.jsonl \
      --rows $ROWS_BOTH --out $P/b002/rows/$g --relabel-spelling multitool    # b001 labels: verb_surface / native / none
  $PY -m dataxp.ownrows.batch_prep unrowed --states $P/rows_sp/$g/states.jsonl --states-phi $P/rows_sp/$g/states_phi.jsonl \
      --rows $ROWS_BOTH --out $P/b002/rows_sp/$g --relabel-spelling set_parameter
  $PY -m dataxp.ownrows.batch_prep seed-shards --rowdir $P/b002/rows/$g    --shards 4 --samples $P/rows/$gm/samples.jsonl
  $PY -m dataxp.ownrows.batch_prep seed-shards --rowdir $P/b002/rows_sp/$g --shards 4 --samples $P/rows_sp/$g/samples.jsonl
done
# drc -> local -> Orchard: upload $P/b002/rows -> $C/b002/rows and $P/b002/rows_sp -> $C/b002/rows_sp
#   (states.jsonl, states.shard*.jsonl, samples.shard*.jsonl; states_phi stays on drc)
# Orchard: ONE job per spelling, the redraw pass only (no smoke, no first pass), 4 GPUs:
#   --export=ALL,OWN_ROWS=$C/b002/rows,OWN_DOMAINS="v2 grid2op engibench photonics",OWN_SMOKE=0,DATAXP_MAX_DRAW_PASS=2
#   --export=ALL,DATAXP_SPELLING=set_parameter,OWN_ROWS=$C/b002/rows_sp,OWN_DOMAINS="...",OWN_SMOKE=0,DATAXP_MAX_DRAW_PASS=2
#   (with another GPU count: re-run seed-shards on Orchard with --shards <N> --park $C/b002/park_<g> first)
# then the shared tail with b00N = b002 (no --prior-states needed: these are b001's own states, with
# b001's splits)
```

### b003: unused run2 and run2_supp problems (v2 five)

```bash
# run2 to drc (problems + trajectories are enough for select_states; ~8 MB)
#   download bridges2 /ocean/projects/mch250030p/shared/dataxp/datav4_corpus/*/accepted_run2/[pt]r*.jsonl -> <local>/run2/
#   upload drc <local>/run2/<fam>/accepted_run2/* -> $P/run2/<fam>/accepted/   (circuit2 frame3d2 grid2 seir2 traffic2)
USED="$P/rows/v2verb/states.jsonl"                        # every problem b001 drew from (both spellings drew the same)
for sp in multitool set_parameter; do
  if [ $sp = multitool ]; then W=$P/b003/rows/v2; else W=$P/b003/rows_sp/v2; fi
  DATAXP_SPELLING=$sp $PY -m dataxp.ownrows.select_states --data $SUPP --variants circuit2 frame3d2 grid2 seir2 traffic2 \
      --out $W/sel_supp --per-domain 0 --seed 0 --exclude-problems $USED --exclude-specs $EVALN $EVAL2 --split-from $USED
  DATAXP_SPELLING=$sp $PY -m dataxp.ownrows.select_states --data $P/run2 --variants circuit2 frame3d2 grid2 seir2 traffic2 \
      --out $W/sel_run2 --per-domain 0 --seed 0 --exclude-problems $USED --exclude-specs $EVALN $EVAL2 --split-from $USED
  cat $W/sel_supp/states.jsonl $W/sel_run2/states.jsonl > $W/states.jsonl   # one group: a family comes from one group
done
$PY -m dataxp.ownrows.cand_phi --states $P/b003/rows/v2/states.jsonl --out $P/b003/rows/v2/states_phi.jsonl --workers 8
$PY -m dataxp.ownrows.batch_prep phi-from --states $P/b003/rows_sp/v2/states.jsonl \
    --phi $P/b003/rows/v2/states_phi.jsonl --out $P/b003/rows_sp/v2/states_phi.jsonl
# upload states.jsonl of both to $C/b003/rows/v2 and $C/b003/rows_sp/v2; Orchard job 1 then job 2 per spelling
# (OWN_DOMAINS=v2); shared tail with b00N = b003 (--prior $DV/datav4_$sp now includes batches/b002)
```

### b004: new native problems 130-259 (grid2op, engibench, photonics)

```bash
# drc generation (CPU; one submit_job each; b001 worker counts)
for v in grid2op:4 engibench:8 photonics:4; do
  nice -n 10 $PY -m dataxp.run_generation --variant ${v%%:*} --start-index 130 --n 130 --seed 0 --workers ${v##*:} \
      --tmp $P/tmp_b004 --out $P/out_b004
done
# corpus batch into the drc mirror; earlier corpora = b001's native out + the mirror itself
$PY -m dataxp.package_datav4 --corpus-batch b004 --corpus-dir $DV/datav4_corpus \
    --corpus-src grid2op=$P/out_b004/grid2op engibench=$P/out_b004/engibench photonics=$P/out_b004/photonics \
    --corpus-prior $P/out $DV/datav4_corpus $EVAL
$PY -m dataxp.package_datav4 --check --corpus-batch b004 --corpus-dir $DV/datav4_corpus
# decision 3: the shared-base list is in $DV/datav4_corpus/engibench/accepted_b004/sources.json
#   (shared_base_with_eval.by_problem); for option (b):
mkdir -p $P/b004 && $PY -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['shared_base_with_eval']['by_problem']))" \
    $DV/datav4_corpus/engibench/accepted_b004/sources.json > $P/b004/engibench_shared_base.txt
for v in grid2op engibench photonics; do
  for sp in multitool set_parameter; do
    if [ $sp = multitool ]; then W=$P/b004/rows/$v; else W=$P/b004/rows_sp/$v; fi
    DATAXP_SPELLING=$sp $PY -m dataxp.ownrows.select_states --data $P/out_b004 --variants $v --out $W --per-domain 0 --seed 0 \
        --exclude-specs $EVALN $EVAL2 [--exclude-problems $P/b004/engibench_shared_base.txt]
  done
  $PY -m dataxp.ownrows.cand_phi --states $P/b004/rows/$v/states.jsonl --out $P/b004/rows/$v/states_phi.jsonl --workers 6
  $PY -m dataxp.ownrows.batch_prep phi-from --states $P/b004/rows_sp/$v/states.jsonl \
      --phi $P/b004/rows/$v/states_phi.jsonl --out $P/b004/rows_sp/$v/states_phi.jsonl
done
# Orchard jobs 1 + 2 per spelling (OWN_DOMAINS="grid2op engibench photonics"); shared tail with b00N = b004
# corpus to bridges2: download drc $DV/datav4_corpus/*/accepted_b004/* -> <local>/up/corpus_b004/;
#   upload each <local>/up/corpus_b004/<fam>/accepted_b004 -> .../datav4_corpus/<fam>/accepted_b004 (new dirs)
```

### b005: new v2 problems 260-389 (circuit2, frame3d2, grid2, seir2, traffic2)

```bash
for v in circuit2 frame3d2 grid2 seir2 traffic2; do
  nice -n 10 $PY -m dataxp.run_generation --variant $v --start-index 260 --n 130 --seed 0 --workers 4 \
      --tmp $P/tmp_b005 --out $P/out_b005
done
$PY -m dataxp.package_datav4 --corpus-batch b005 --corpus-dir $DV/datav4_corpus \
    --corpus-src circuit2=$P/out_b005/circuit2 frame3d2=$P/out_b005/frame3d2 grid2=$P/out_b005/grid2 \
                 seir2=$P/out_b005/seir2 traffic2=$P/out_b005/traffic2 \
    --corpus-prior $SUPP $P/run2 $DV/datav4_corpus $EVAL
$PY -m dataxp.package_datav4 --check --corpus-batch b005 --corpus-dir $DV/datav4_corpus
for sp in multitool set_parameter; do
  if [ $sp = multitool ]; then W=$P/b005/rows/v2; else W=$P/b005/rows_sp/v2; fi
  DATAXP_SPELLING=$sp $PY -m dataxp.ownrows.select_states --data $P/out_b005 \
      --variants circuit2 frame3d2 grid2 seir2 traffic2 --out $W --per-domain 0 --seed 0 --exclude-specs $EVALN $EVAL2
done
$PY -m dataxp.ownrows.cand_phi --states $P/b005/rows/v2/states.jsonl --out $P/b005/rows/v2/states_phi.jsonl --workers 8
$PY -m dataxp.ownrows.batch_prep phi-from --states $P/b005/rows_sp/v2/states.jsonl \
    --phi $P/b005/rows/v2/states_phi.jsonl --out $P/b005/rows_sp/v2/states_phi.jsonl
# Orchard jobs 1 + 2 per spelling (OWN_DOMAINS=v2); shared tail with b00N = b005; corpus upload as b004
```

## Checks run for this change (2026-10-24, Windows, PYTHONHASHSEED=0, one worker)

Scratch: `<session scratchpad>/append/` (pre-edit package snapshot `pre_pkg/`, `gen/`, `spec/`, `demo/`,
logs).

| check | result |
|---|---|
| default identity: quick generation of circuit2 and grid2op, pre-edit package vs post (`compare_runs.py`) | every `*.jsonl` byte-identical, `report.json` equal ignoring the wall-clock fields; RESULT bit_for_bit=True |
| select_states without the new options, pre-edit package vs post (circuit2 and grid2op quick corpora; set_parameter `--per-domain 3 --dev-per-domain 2`, multitool `--per-domain 0`) | `states.jsonl` byte-identical in all four; `select_report.json` equal apart from the output path. `sample.py` records gain the `draw_pass` key by design (nothing reads a fixed key set) |
| `--start-index 130 --quick` on grid2op (pre-build 0..129 in 108 s, module registry) | ids `grid2op_0130` .. `grid2op_0135`; all 6 specs equal (whole dict) to indices 130-135 of one straight build of 0..135; 0 canonical collisions with 0..129. Without the pre-build, `grid2op_0132` would have been a duplicate of an earlier problem: this is what the pre-build prevents |
| the same spec-level check (`build_specs` in a fresh process vs one straight build of 0..135) on engibench and photonics | both: ids 0130-0135, 6 of 6 specs equal to the straight build, 0 canonical collisions with 0..129. Without the pre-build: photonics 0 of 6 differ; engibench 2 of 6 differ (photonics2d problems 0131 and 0134, in the last digits of the goals and initial metrics: process-history round-off, not a registry), which is why engibench stays in `PREBUILD_VARIANTS` |
| the tools on the real b001 work files on drc, after the sync (`/storage/wxu/corpus/append/real_demo.sh`, 18 s CPU) | `batch_prep unrowed` gives 1,602 states (v2 1,324, grid2op 73, engibench 106, photonics 99; the same sids in the same order in both spellings), equal to the census above; `--relabel-spelling multitool` set all 1,602 legacy labels (`verb_surface` / `native` / none) to `multitool` in both state files, and the set_parameter ones needed none. b001's own multitool rows offered as `--batch b002` against `/storage/wxu/datav4/datav4_multitool` are REFUSED: all 2,420 rows (both arms) repeat a prior state, and nothing is written. The main `--check` of the drc b001 copy passes with the new code. The b003 run2_supp selection takes circuit2 7 problems / 58 states, frame3d2 39 / 335 and seir2 14 / 62; grid2 and traffic2 have none left. That is 455 states, with 0 canonical collisions against the 660 evaluation problems |
| append demo (quick circuit2 + grid2op corpora, dry-run replies; a b001 built with the main build in both spellings; b002 = `unrowed` + `seed-shards` + one redraw pass; b003 = the unused problems with a simulated token cap, a requeued first pass (cap 1: 22 truncated keys left alone) and one redraw pass (cap 2: 22 redrawn); both packaged with `--append` into the spelling dirs' `batches/` and checked with `--check --parse-check`; a corpus batch from the `--start-index 130` run) | b002: 3 states (no row in either spelling), replies 15 = 12 + 3 redrawn, rows 2 train / 1 dev per spelling, twin states identical. b003: 11 states, replies 66 = 44 + 22 redrawn, 5 rows of the problem whose spec equals the fake v2 evaluation problem dropped, 5 rows kept (draw pass 1: 3, 2: 2); the grid2op problem equal to the fake native evaluation spec left out by `--exclude-specs`; prior = b001 + batches/b002, 0 collisions; CHECK OK for all four batch directories and the corpus batch. Refused as intended, with nothing left behind: b001's rows re-appended (prior sids), b002's rows re-appended, b002 built twice, held-out family rows, a split change against `--prior-states`, `--draw-passes 1` with pass-2 replies, dry-run replies without `--allow-dry-run`, the other spelling's tree as prior, twin states that differ, an edited row (`--check`: sha1), an evaluation file holding a batch problem under another id (`--check`: canonical spec), a corpus batch repeating earlier problem ids |

No GPU was used for this change; every check ran on CPU (local Windows and drc).

# dataxp/ownrows — own-action SFT rows from an accepted corpus

Turns an accepted dataxp corpus (`<out>/<variant>/accepted/{problems,trajectories,pairs}.jsonl`
from `python -m dataxp.run_generation`) into **own-action** SFT rows for both the SET_PARAM
variants (circuit2, frame3d2, grid2, seir2, traffic2, thermal2, inventory2, …) and the
native-verb variants (grid2op, photonics, molopt, engibench). The recipe is the one that kept
transfer in the SFT probe (`docs/sft_probe_rat2.md`, arms `own` / `own_act`):

> At selected repair-phase states the un-tuned Qwen3.8-27B is prompted with the training
> observation (thinking off), K replies are drawn, each reply's `<action>` is parsed with
> `domain.parse_move`, applied on a clone and scored by its Φ percentile among
> `domain.legal_moves(st, rng)` (Φ via `dataxp.domain.PhiEvaluator`), and the state becomes a
> row if its best reply is at or above the 90th percentile or equals the teacher's move. The
> row's assistant turn is `<action>format_move(move)</action>` (`own_act`), with the model's own
> written reasoning as `reasoning_content` (`own`).

Ported from the session scratchpad (`sftprobe/sftprobe_states.py`, `sftprobe_tiers.py`,
`up_tools/cand_phi.py`, `up_tools/own_phi.py`, `up_tools/build_own.py`) and
`dataxp/gold_probe.py` (mode `own_nothink`), generalised from `SET_PARAM(i, p, v)` to `Move`.

## Pipeline

```
select_states -> cand_phi -> sample -> score -> build_tiers
```

| step | module | input -> output | what it does |
|---|---|---|---|
| 1 | `select_states.py` | corpus -> `states.jsonl` | repair-phase states per problem (sftprobe rule), replayable state spec + served prompt parts |
| 2 | `cand_phi.py` | `states.jsonl` -> `states_phi.jsonl` | every legal move at each state with its Φ (cache); teacher's Φ and percentile |
| 3 | `sample.py` | `states.jsonl` -> `samples.jsonl` | K replies per state from an OpenAI-compatible endpoint, thinking off; append-only, resumes |
| 4 | `score.py` | `states_phi.jsonl` + `samples.jsonl` -> `scored.jsonl` | parse, apply, Φ percentile vs the cached legal set, same_move vs the teacher |
| 5 | `build_tiers.py` | all of the above -> `tiers/` | `own` and `own_act` tiers + `manifest_own.json` |

Local runs: `export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` (the scripts also set both as defaults).

```bash
cd <repo>
# 0. a corpus (quick: 6 problems, 8 turns; the G1 gate fails on 6 problems, so the files stay
#    under --tmp/<variant>/ and select_states takes them from there with --tmp)
python -m dataxp.run_generation --variant molopt --quick --workers 1 --seed 0 --tmp $S/tmp --out $S/out

# 1. states (train/dev split by whole problems; --per-domain 0 = every problem into train)
python -m dataxp.ownrows.select_states --data $S/out --tmp $S/tmp --variants molopt circuit2 \
    --out $W --per-domain 120 --dev-per-domain 12 --seed 0

# 2. legal moves + Phi (append-only by sid; --workers N for a process pool)
python -m dataxp.ownrows.cand_phi --states $W/states.jsonl --out $W/states_phi.jsonl --workers 1

# 3. K samples (vLLM; append-only by (sid, sample), rerun to resume after a kill)
python -m dataxp.ownrows.sample --states $W/states.jsonl --out $W/samples.jsonl --k 6 \
    --endpoint http://127.0.0.1:8766/v1 --model Qwen/Qwen3.8-27B --max-tokens 4000 \
    --temperature 0.6 --concurrency 8
#    no GPU: random legal moves in the native format (optionally a share equal to the teacher's)
python -m dataxp.ownrows.sample --states $W/states.jsonl --out $W/samples.jsonl --k 6 --dry-run \
    [--dry-run-teacher-share 0.25]

# 4. score
python -m dataxp.ownrows.score --states $W/states_phi.jsonl --samples $W/samples.jsonl \
    --out $W/scored.jsonl --workers 1

# 5. tiers
python -m dataxp.ownrows.build_tiers --states $W/states_phi.jsonl --samples $W/samples.jsonl \
    --scored $W/scored.jsonl --out $W/tiers --min-pct 90 --max-words 320 --tag own
```

Outputs of step 5: `tiers/train_own.jsonl`, `tiers/dev_in_own.jsonl`, `tiers/train_own_act.jsonl`,
`tiers/dev_in_own_act.jsonl`, `tiers/manifest_own.json`. These are the `data.train_jsonl` /
`data.dev_jsonl` inputs of `scripts/train_sft.py` with `data.target_fn=assistant_after_think`
(the collaborator's launcher `drc_train_dataxp.sh`).

## The recipe, step by step

**States (step 1).** A state is every turn of an accepted trajectory whose `binding_constraint`
is not None (the repair phase, as `render_cot.iter_states` stratifies). Per variant the problems
with at least one such turn are shuffled with `stable_seed("ownrows_select", variant, seed)` and
taken whole: into train until `--per-domain` states, then into dev until `--dev-per-domain`
(disjoint problems) — the draw of `sftprobe_states.py`. The state is replayed from
`load(spec)` through `apply_action` over the previous turns' actions; for deterministic domains
the replayed `evaluate(st, 0)` is checked against the recorded `state_metrics` (count printed;
`--no-replay-check` skips it). The served prompt parts are built with
`render_cot.policy_observation` and split as the collaborator's `observation_v2` splits them:
`system` = the DOMAIN/GOAL head, `observation` = the rest, `instruction` = the task instruction
appended to the user text.

**Legal moves and Φ (step 2).** `domain.legal_moves(st, rng)` with
`rng = default_rng(stable_seed(pid, "gp", k))` (gold_probe.prepare's rng, k = turn index in the
trajectory). Each move is applied on a clone and scored with
`PhiEvaluator(domain, spec, offset_rule="infeasible")` (the ladder's evaluator; `--offset-rule`
exposed) at seed `stable_seed(pid, "gp", i, p)` for SET_PARAM — gold_probe's seed — and
`stable_seed(pid, "gp", format_move(move))` for a native verb. The teacher's move is scored the
same way (`gold_phi`, `gold_pct`, `gold_in_legal`).

**Sampling (step 3).** Default `--prompt probe`: gold_probe's `SYSTEM_OWN` in its own_nothink
form (written reasoning under 250 words, then one `<action>` line); on a native domain its
`SET_PARAM` action line is replaced by `<action>VERB(arg, ...)</action>` + `domain.grammar()`,
the substitution `render_cot.system_prompt` makes on the renderer's SYSTEM. User = the full
observation (head + body, what the sftprobe stored as `obs`) + "Choose exactly ONE change for
THIS step." `--prompt served` sends the training row's own system/user messages instead.
Request body as gold_probe.call: temperature 0.6, top_p 0.95, top_k 20,
`chat_template_kwargs.enable_thinking=false`. A record counts as done when it has an answer and
`finish != "length"` (truncated and errored replies are redrawn on rerun).

**Scoring (step 4).** The last `<action>…</action>` of the reply → `domain.parse_move` →
`apply` on a clone (`try_apply`: a SET_PARAM pick naming an unknown parameter or element, or
one the state refuses, is illegal; the applied value is read back, `clamped` / `noop` flagged)
→ Φ at the move's seed → `pct = 100 · mean(legal Φ <= pick Φ)` (own_phi's rule) →
`same_move = pick.matches(teacher, rel=1e-2)` (the renderer's `wrong_action` tolerance). For
SET_PARAM picks the sftprobe's looser `gold_probe._match` (same element and parameter, same
direction, step within ×2) is recorded as `same_move_loose` but not used for selection.

**Tiers (step 5).** Per state, among replies that parsed and applied, carry a non-empty written
reasoning (text before the last `<action>`) of at most `--max-words` words with no `<think>`
tag, and are not a no-op, take the highest-percentile one (ties: lowest sample index). Keep the
state iff `pct >= --min-pct` (90) **or** `same_move`. A state without such a reply is dropped
from both arms, so `own` and `own_act` are matched row for row. `--max-words` default 320
(build_own's); the own_cands arm raised it to 900 (`docs/sft_probe_rat2.md`).

## Row format

Exactly the DX-26 / sftprobe tier row (`messages` = system, user, assistant with
`reasoning_content`), plus the selection readouts. A circuit2 row from the dry run, trimmed:

```json
{"domain": "circuit2", "instance": "circuit2_0000", "turn": 0, "turns": 1,
 "binding_constraint": "gbw_mhz", "n_prior_changes": 0,
 "messages": [
   {"role": "system", "content": "DOMAIN: two-stage op-amp design: device sizing, channel lengths, compensation and cascode\nGOAL: minimise area subject to the constraints below"},
   {"role": "user", "content": "PROBLEM\nTwo-stage CMOS operational amplifier ... CONSTRAINT STATUS (current design)\n  gain_db ... \n\nYOUR MOVE\nMake exactly one change: set one parameter of one element to a new value. First reason, in 60-150 words ...\n<action>SET_PARAM(<element>, <parameter>, <value>)</action>\n<element> is ...\nThe parameter names in this problem are exactly: W, L, B, Cc, comp, Ib, Rz, rz_on, cas_on.\n"},
   {"role": "assistant", "reasoning_content": "<the model's own written reasoning; empty string in own_act>",
    "content": "<action>SET_PARAM(3, comp, 2)</action>"}],
 "instruction_sha1": "3d4867180337",
 "action": {"i": 3, "p": "comp", "v": 2.0, "v_old": 0.0}, "clamped": false,
 "tier": "ownrows_own", "teacher": "qwen3.8-27b-own-nothink-phi-selected",
 "rationale_sample": 5, "sample_mode": "own_nothink_dryrun", "sample_prompt": "probe",
 "own_pct": 100.0, "own_same_move": true, "own_phi": -10.172561122700646,
 "gold_action": {"i": 3, "p": "comp", "v": 2.0, "v_old": 0.0},
 "gold_phi": -10.172561122700646, "gold_pct": 100.0, "n_candidates": 60}
```

A native-domain row differs only in the action dict (`{"verb": "ADD_FRAGMENT", "args":
{"atom": 12, "fragment": "hydroxyl"}}`), the action line (`<action>ADD_FRAGMENT(12,
hydroxyl)</action>`) and the instruction (see below). `action` is a serialised `Move`
(`Move.from_dict` re-applies it); `gold_action` is the teacher's.

**Spelling.** A domain that speaks verbs (native or verb surface) is served in the process's
`DATAXP_SPELLING` (`docs/dataxp_spellings.md`): `multitool` (default) = `INSTRUCTION_NATIVE` +
the verb grammar, action lines `VERB(arg, ...)`, sha1 `1c344388a444`; `set_parameter` =
`INSTRUCTION_SET_PARAM` (the same text with the action line `<action>SET_PARAM(element,
parameter, value)</action>` and "SET_PARAM performs one of the operations listed below, ..." in
place of "VERB is one of the operations listed below, ...") + the generated SET_PARAM operation
list, action lines `SET_PARAM(i, p, v)`, sha1 `fe8fa8db0a90` (`f99b61d776f1` before the VERB
sentence was reworded, 2026-10-23). `states.jsonl` and every tier row record `spelling`
(`multitool` | `set_parameter`; `SET_PARAM` for a plain v1/v2 domain). The stored `action` /
`gold_action` dicts are the same under both.

**Served prompt parity.** The SET_PARAM user text is V2 + the parameter line: `INSTRUCTION_V2`
in `common.py` is the byte content of the collaborator's `task_instruction_V2.txt`
(sha1[:12] `3d4867180337`, asserted at import). Checked against the 693 DX-26 rows of the
sftprobe (`local_test/rows_{train,dev}.jsonl`, rebuilt from the run2_supp corpus with
`select_states.states_of`): system 693/693, action line 693/693, instruction sha1 693/693, user
text 580/693. The 113 user differences are not the port: 108 seir2 rows differ only in the
`pop_conservation_err` line (a 1e-16 quantity printed to 3 digits, machine round-off) and 5
frame3d2 rows only in which violated constraint carries the "(most severe)" tag (the corpus's
recorded `binding_constraint`, which `render_cot` also uses, vs the eval loop's recomputation).

## What is new here and what could not be reproduced from the scratchpad

Stated plainly, so nobody mistakes it for the probe's exact recipe:

1. **`assistant_after_think` is not in this repository.** `llm_finetune/training/sft/targets.py`
   registers `gold_curriculum_warmstart`, `warmstart_reasoning`, `full_sequence`,
   `assistant_only`; the target (and `scripts/check_tier_parity.py`) live in the
   collaborator's `repo_dx25`. The rows follow their format (verified above on real DX-26
   rows); the token-level parity check could not be run here.
2. **Native-domain task instruction is new.** No native variant had a served instruction. 
   `INSTRUCTION_NATIVE` is V2 with the SET_PARAM sentences replaced by the verb-call form,
   followed by "The operations in this problem are exactly:" + `domain.grammar()` (the way the
   parameter line follows V2). Its sha1[:12] goes in `instruction_sha1`. The eval harness must
   serve the same text.
3. **Native-domain sampling prompt.** gold_probe's `SYSTEM_OWN` only knew `SET_PARAM`; here its
   action line is replaced by the verb-call form + grammar (`common.own_system_prompt`), the
   substitution `render_cot.system_prompt` makes for the renderer.
4. **Φ seed for native moves.** The scratchpad only seeded `(pid, "gp", i, p)`; native moves
   use `(pid, "gp", format_move)`. Deterministic domains ignore the seed.
5. **Legal set of native domains.** `legal_moves` of e.g. molopt is a seeded sample of the
   vocabulary, so the percentile is relative to that sample and the teacher's move need not be
   in it (quick molopt corpus: in the set at 7 of 12 states, `gold_in_legal`). The teacher's
   Φ is still scored directly, so the "equals the teacher" branch does not depend on it.
6. **same_move.** The task asked for `Move.matches`; the sftprobe's own rule was
   `gold_probe._match` (looser). Both are recorded; selection uses `matches(rel=1e-2)`.
7. **No-op picks.** build_own did not check them; DX-26's builder dropped them. A SET_PARAM pick
   that leaves the value unchanged is excluded by default (`--allow-noop` restores build_own).
8. **Percentile.** `<=` as in own_phi (whose output build_own consumed); gold_probe.score used
   `<`. A pick that is the best legal move scores 100 either way.
9. **Generation settings of the original `own` arm.** K=6, T=0.6, thinking off are documented
   (`docs/sft_probe_rat2.md`); the max_tokens used on drc is not in the scratchpad code
   (gold_probe's default is 16000 with thinking; rat3 used 4k without). Default here: 4000.
10. **DOMAIN/GOAL head of native domains.** `render_cot.DOMAIN_BLURB` has no entry for them, so
    the head reads `DOMAIN: molopt / GOAL: optimise subject to the constraints below`. Inherited
    from render_cot (the same head the renderer's training prompts get); not changed here.
11. **shuf arm, audits.** Not built (not part of the own-action recipe).

## Dry-run results (2026-10-23, local, no model)

Quick corpora (`--quick --workers 1 --seed 0`): molopt 46 s, circuit2 5 s; both fail G1 on 6
problems, so the files were read from `--tmp`. `select_states --per-domain 8 --dev-per-domain 4`,
`sample --k 6 --dry-run --dry-run-teacher-share 0.25`, replay mismatches 0.

| variant | problems train/dev | states train/dev | legal moves (mean) | teacher pct | replies | parsed / legal | rows kept train / dev |
|---|---|---|---|---|---|---|---|
| molopt (native) | 5 / 0 | 12 / 0 | 54.6 | 99.1 | 72 | 1.00 / 1.00 | 12 of 12 / – |
| circuit2 (SET_PARAM) | 5 / 1 | 11 / 6 | 60.0 | 100.0 | 102 | 1.00 / 1.00 | 11 of 11 / 5 of 6 |

The one dev drop is `below_threshold` (a random pick under the 90th percentile and not the
teacher's). The resume rule held (a second `sample` run drew 0). `cand_phi --workers 2` gives
the same cache as one worker. Every kept action re-applies on its rebuilt state.

## Appending batches (2026-10-24)

Later batches of datav4 rows (b002, b003, ...) are built with the same five steps plus the pieces
below; the plan, the rules and the exact commands are in `docs/datav4_append_plan.md`.

- `select_states --exclude-problems F ...` leaves out problems an earlier batch already drew from
  (ids from states / rows / spec files or a text list); `--exclude-specs F ...` leaves out problems
  whose canonical spec equals one in the given problem files (the evaluation sets);
  `--split-from F ...` keeps an earlier problem's split. Without them the draw is unchanged.
- `sample` writes `draw_pass` into every record (1 + the records of that key already in the file);
  `--max-draw-pass N` / `DATAXP_MAX_DRAW_PASS=N` leaves a key alone once it has N records (the first
  job of a batch runs with 1, the one redraw pass of the truncated keys with 2).
- `batch_prep unrowed` (the states of an earlier batch with no row in any given tier file),
  `batch_prep seed-shards` (shard files that hold every earlier record, so a redraw labels pass 2 over
  any GPU count), `batch_prep phi-from` (the set_parameter Φ cache = its own state records + the
  multitool Φ fields).
- `python -m dataxp.package_datav4 --append ...` packages a batch directory with every check of the
  main build plus the append rules; `--check --out <batch dir>` re-verifies it.

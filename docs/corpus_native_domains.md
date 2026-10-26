# Combined design corpus: our domains + online domains with native operations

**Decided 2026-10-22 with the owner.** Starting point for the multi-domain training bet
(FLAN-style: many domain families, hold out whole families, train on the model's own
simulator-accepted actions). Scope decisions, in the owner's words:

1. **Only domains with native operations.** The analytic sizing suites (CEC 2020 real-world
   constrained, 57 problems; RE suite, 24) are NOT included: their only move is "set named
   variable x", which is the single-knob form the owner rejected on 2026-10-05. Every domain
   in this corpus has its own verbs with typed arguments in that field's language
   (`docs/industrial_domains_plan.md` §3 style; the memory rule "domain-native actions").
2. **First batch, all in parallel:** Grid2Op / L2RPN (grid operation), invrs-gym / ceviche
   (photonic inverse design), PMO (molecular optimization, 23 oracles), EngiBench (NeurIPS 2025:
   airfoil, beams, heat conduction, photonics 2D, thermoelastic, power electronics). The industrial
   seven (`docs/industrial_domains_plan.md`) stay on their own track. CEC/RE may later serve as
   held-out tests only, never as training data.
3. **drc GPU freed** (leftover 8k-cap / base_think chain cancelled) for the own-action row
   generation and the leave-families-out slope test.

## What a domain must provide (dataxp contract + the move extension)

Existing contract: `dataxp/domain.py` (`Domain`: make_problem with infeasible starts and goals in
the `minimum_`/`maximum_` convention, load/clone/evaluate(st, seed) deterministic in (st, seed),
n_elements/get/set/bounds, program() with validity bounds, reference(), optional hint()); Φ, gates,
ladder, harvest and replay sit above it and must keep running bit-for-bit on run1/run2.

Move extension (phase A, one implementation shared by every new domain):

- `Move(verb: str, args: dict)` with `domain.legal_moves(st, rng) -> List[Move]`,
  `domain.apply(st, move)` (in place), `domain.format_move(move) -> "VERB(a, b, ...)"`,
  `domain.parse_move(text) -> Move | None`, `domain.grammar() -> str` (the verb list with
  argument types and ranges, for the system prompt).
- The v1/v2 `Candidate(i, p, v)` path stays untouched; a `Candidate` is a `Move` with verb
  `SET_PARAM`. Harvest, pairs and replay serialize moves as dicts; Φ evaluation is state-based
  and unchanged.
- Each new domain's `describe(spec)` is written from the source paper or environment docs: what
  the system is, what the quantities mean, what each verb does physically. That text is the
  "domain reading" the corpus exists to teach.
- Compound verbs that change several linked quantities at once are wanted where the field has
  them (e.g. `RETIME_SIGNAL(j, cycle, split, offset)`).

## Verb sets per domain (to be confirmed by the implementers against the real simulators)

| domain | state | verbs (typed arguments) |
|---|---|---|
| grid2op | bus topology, line status, generator dispatch, storage | `SET_BUS(sub, element, bus)`, `RECONNECT_LINE(l)`, `DISCONNECT_LINE(l)`, `REDISPATCH(g_from, g_to, mw)`, `SET_STORAGE(s, mw)`, `CURTAIL(g, frac)`; time advances one step per move |
| invrs-gym / ceviche | density/permittivity field on a design region | `FILL_RECT(x0, y0, x1, y1)`, `CLEAR_RECT(...)`, `FILL_DISK(cx, cy, r)`, `CLEAR_DISK(...)`, `SHIFT_FEATURE(k, dx, dy)`, `DILATE(k, px)` / `ERODE(k, px)` on labelled connected components |
| PMO | molecule (SMILES graph) | `ADD_FRAGMENT(atom, fragment)`, `REMOVE_FRAGMENT(k)`, `REPLACE_ATOM(atom, element)`, `ADD_RING_BOND(a, b)`, `BREAK_BOND(a, b)`, `SWAP_FRAGMENT(k, fragment)`; validity = RDKit sanitization |
| engibench | per problem: airfoil spline control points; density fields (beams, heat conduction, thermoelastic, photonics 2D); circuit component values (power electronics) | airfoil: `MOVE_CTRL_POINT(k, dx, dy)`, `THICKEN(span, pct)`, `CAMBER(span, pct)`; fields: the region verbs above; power electronics: include only if the problem has a real topology/operation vocabulary, otherwise leave out and say so |

## Time-stepping domains

**Decided 2026-10-23 on the owner's behalf** (the grid2op verifier found the ladder stalling:
when no candidate improves on the incumbent, the v1/v2 phi_select policy keeps the state and
scores it again next turn, which on a domain where every move advances the chronic meant the
same step was searched and harvested up to seven times in a row - grid2op_0003 turns 1–7 all at
step 794; 15 of the 124 pairs of the quick generation, 12.1%, were exact duplicates).

A domain whose moves advance simulated time declares the class attribute `advances_time = True`,
implements `null_move(st) -> Move` (grid2op: `DO_NOTHING()`), and names the metric that carries
simulated time in `time_key` (default `"step"`); all three live on `Domain` in `dataxp/domain.py`
with `advances_time = False` and `null_move -> None` as the defaults.

- `dataxp/ladder.py` (phi_select / phi_d2): when no child improves on the incumbent and the
  domain has `advances_time`, the policy applies the null move, so the world advances instead
  of the same state being re-recorded. The null move is recorded as the turn's action (the
  trajectory's turns and actions stay aligned for replay and for `render_cot`). The stop rule -
  feasible and a second non-improving turn - is unchanged; on that branch nothing is applied and
  the record says `None`, as before.
- `dataxp/harvest.py`: for such domains, exact duplicate pairs by (problem, step, chosen,
  rejected) are dropped and the count reported as `n_dropped_duplicate` in `pair_stats`
  (`run_generation` prints it and the G3 gate carries it in `report.json`); the key is
  present only for time-stepping domains.
- The v1/v2 path is untouched: `advances_time` is False there, so the policy keeps the state
  and `pair_stats` keeps its four keys. Checked byte-for-byte with `compare_runs.py` on the
  circuit2 and thermal2 quick generations (`PYTHONHASHSEED=0`, before vs after: every
  `*.jsonl` byte-identical, `report.json` equal except the wall-clock fields).
- grid2op quick generation, seed 0: 12.1% exact duplicate pairs before (15 of 124), 0.0% after
  (0 of 136); the duplicates vanish because the stalled turns now advance the chronic rather
  than through the harvest filter.

Each domain: 100–150 problems with infeasible starts, a heuristic or exact reference, the
validity envelope, and `python -m dataxp.smoke <variant>` passing. Simulators that cannot run
on Windows are developed and smoke-tested on drc (CPU) through the slurm MCP; nothing runs on
Orchard.

## Status 2026-10-23 (first build round, workflow wf_27e55a03-ca6; fixes in wf_27b04ea0-7c7)

Move contract: in `dataxp/domain.py` + ladder/harvest/render/smoke/probe paths; run1/run2
quick generations byte-identical before/after on six variants (PYTHONHASHSEED=0); guide
`docs/dataxp_move_contract.md`.

| variant | simulator (all run locally on Windows, CPU) | cost / eval | verbs | first verdict | notes |
|---|---|---|---|---|---|
| grid2op | grid2op 1.12.6 + lightsim2grid; case14 sandbox, case14+storage, neurips2020 track1 (36 subs), idf_2023 (118 subs); opponent off | 0.037 s | DO_NOTHING, RECONNECT_LINE, DISCONNECT_LINE, SET_BUS, SPLIT_SUB, MERGE_SUB, REDISPATCH, CURTAIL, SET_STORAGE | fix | one move = one 5-min step; topology verbs never re-energise a line (only RECONNECT_LINE does); fix items: game-over rule for infeasible curtail/storage in describe(), refuse DISCONNECT on an out-of-service line, ladder stall on time-stepping domains |
| photonics | ceviche-challenges 1.0.2 (2-D FDFD), the five lightweight ceviche challenges ported from invrs-gym 1.6.2 (jax not installable beside numpy 1.26) | 0.4 s | FILL_RECT, CLEAR_RECT, FILL_DISK, CLEAR_DISK, SHIFT_FEATURE, DILATE, ERODE | **pass** | 32 fabricable candidates per state from an adjoint-ranked pool; min-feature rule as validity; reference = short gradient run; non-ceviche invrs-gym challenges excluded (3-D RCWA on jax) |
| molopt | RDKit + PyTDC oracles: QED, DRD2 and 17 GuacaMol tasks (19 of PMO's 23); SA score as makeability envelope | 0.003 s | ADD_FRAGMENT, REMOVE_FRAGMENT, SWAP_FRAGMENT, REPLACE_ATOM, ADD_RING_BOND, BREAK_BOND | fix | excluded gsk3b/jnk3 (sklearn ≥1.3 cannot unpickle TDC's models), valsartan_smarts (constant), sitagliptin_mpo (flat); 29-fragment vocabulary; fix items: three oracle descriptions, reference envelope (ring size) |
| engibench | engibench 0.2.0: beams2d (SIMP FE), heatconduction2d (in-module P1 FE re-implementation of the docker-only template), photonics2d (ceviche) | 0.06–0.38 s | the seven region/island verbs | fix | excluded thermoelastic2d (EngiBench's model cannot represent solid/void designs), airfoil (RANS in docker, minutes per solve; verb stub kept), power_electronics (component values only), 3-D problems; fix items: move_label units, candidate ranges narrower than the grammar |

Time-stepping domains (decision 2026-10-23): see the "Time-stepping domains" section above
(`advances_time`, `null_move`, `time_key`; ladder and harvest behaviour; the byte-for-byte check
and the grid2op duplicate-pair rates).

### Fix round and generation (2026-10-23, workflow wf_27b04ea0-7c7)

grid2op, molopt and engibench: every blocking item fixed and re-verified by an independent agent
(verdict pass for all three; photonics passed in the first round). The v1/v2 path stayed
byte-identical through the ladder/harvest change (circuit2 and thermal2, PYTHONHASHSEED=0).
grid2op duplicate pairs: 15 of 124 (12.1%) before the time-step rule, 0 of 136 after.

Full-scale generation (130 problems, seed 0) runs on drc (24-core box, shared; workers split by
cost and reniced): engibench j1465 (8 workers), photonics j1466 (4), grid2op j1467, molopt j1469;
code `/storage/wxu/corpus/code/dataxp`, env `/storage/wxu/corpus/env` (→ `/home/wxu/corpus_env`,
Python 3.12.14), live progress `/storage/wxu/corpus/tmp/<v>/<v>/progress.json`, outputs
`/storage/wxu/corpus/out/<v>/<v>/` (report.json + accepted/). Expected wall time: molopt ~1 h,
grid2op ~2 h, engibench 3–5 h, photonics ~12 h. Environment notes: `/storage` on drc is 100% full
(4 GB free) — outputs are tens of MB but the margin is thin; pytdc installed with `--no-deps`
(its cellxgene-census pin has no Python 3.12 build); pandapower wants scipy ≥ 1.15 but grid2op
runs on lightsim2grid; ceviche on drc solves with SuperLU (local MKL/pardiso), so photonics
references may differ at sub-ulp level. molopt oracle descriptions: all 19 now audited against
the installed TDC source with rdkit re-implementations matching `tdc.Oracle` to <2e-6 on probe
molecules; besides the three fixed earlier, drd2 (count buckets, RBF SVC) and ranolazine_mpo
(one-sided floors, clip 0.7) were wrong and are fixed, fexofenadine_mpo and both isomer tasks were
incomplete and are expanded, thiothixene's substituent names corrected; nine were correct. The
describe() text is used at render time only, so the running generation is unaffected; re-upload
`dataxp/` to drc before any rendering there.

### Generation results

| variant | problems | gates | pairs | trajectories | random / greedy / blind / Φ-select / Φ-d2 feasible | Φ-select obj/ref | wall |
|---|---|---|---|---|---|---|---|
| molopt | 130 (6 families, 93% infeasible starts) | all pass | 4,386 | 128 | 0.19 / 0.15 / 0.49 / **0.98** / 0.98 | 0.94 | 34 min |
| grid2op | 130 (4 families, 100% infeasible starts; 128 included) | all pass | 4,906 | 98 | 0.09 / 0.42 / 0.52 / **0.75** / 0.85 | 0.99 | 94 min |
| engibench | 130 (beams2d 44, heatconduction2d 43, photonics2d 43; 93% infeasible starts) | all pass | 7,343 | 127 | 0.22 / 0.52 / 0.62 / **0.98** / 0.98 | 1.04 | 4.8 h (8 workers) |
| photonics | 130 (5 ceviche challenges × 26; 100% infeasible starts) | all pass | 7,730 | 36 | 0.05 / 0.04 / 0.12 / **0.28** / 0.45 | 0.77 | 6.7 h (4 workers) — the hardest family: one-ply Φ-selection solves 28%, depth-2 45% |

Evaluation problems for the native variants: the last 60 by index of each (grid2op 69–129, the
others 70–129), `/storage/wxu/corpus/harness/problems_v3_native.jsonl`; own-row training uses
the lower indices only.

Own-action rows (drc `/storage/wxu/corpus/rows/`): v2 five from run2_supp (2,824 states, ≤ 600
per family, dev 12 each) and grid2op (167 states, problems 0–68) in job bw_rows2 (K=4, 64
streams, 2,000-token cap, ~27 replies/min); engibench and photonics (problems ≤ 69) queued in
bw_rows3 behind it.

### Pipeline pieces ready for the slope test (2026-10-23)

- `dataxp/ownrows/` — own-action rows for SET_PARAM and native-verb domains (select_states →
  cand_phi → sample [OpenAI-compatible endpoint, K samples, thinking off; `--dry-run`] → score →
  build_tiers); dry-run end to end on quick molopt and circuit2 corpora; the served V2
  instruction reproduced byte-for-byte (sha1 3d4867180337); a native instruction (verb form +
  `grammar()`) is new and is the one the harness serves.
- Evaluation harness v3 on drc: `/storage/wxu/corpus/harness/{dataxp_code_v3, scripts/
  dataxp_policy_eval_v3.py, scripts/dx_eval_v3.sh, problems_v3_native.jsonl}`; SET_PARAM path
  byte-identical to the original on three old-variant problems with the harness's non-LLM
  policies (PYTHONHASHSEED=0); native molopt episodes close the loop (phi_select 3/3 feasible,
  moves equal the corpus trajectory prefix). Native evaluation problems = the last 60 of each
  variant by index (lower indices reserved for own-row training). engibench/photonics problem
  files are generated once their corpora finish.
- W1 (thermal2 history-line label "after this change") was missing from the local
  `dataxp/domains/thermal2.py`; applied 2026-10-23 so local and drc code agree.
- `docs/slope_test_prereg.md` — draft pre-registration of the leave-families-out slope test,
  awaiting the owner's yes.

### Decision 2026-10-23 (owner): the v2 seven get native verb surfaces

v2 spells every operation as `SET_PARAM(element, param, value)` (e.g. `SET_PARAM(3, active, 1)`
installs a member). The owner rejects that shared spelling for the corpus. Each v2 domain gets a
thin verb layer over its existing typed parameters — `INSTALL_MEMBER(i)`, `SET_MATERIAL(i, m)`,
`MOVE_JOINT(j, axis, value)`, `CLOSE_LINE(l)`, `SET_PHASE_SHIFT(t, deg)`, `MOVE_SINK(k, site)`,
`PLACE_BLOCK(b, slot)`, `SET_VACCINE_SHARE(g, x)`, `OPEN_STOCKING_POINT(n)`, … — through the Move
contract: legal moves, application, Φ scoring and the stored data are unchanged (still
`{i, p, v, v_old}`); only `format_move` / `parse_move` / `grammar()` and therefore the prompts,
the action lines and the harness parsing change. The run2 data and every earlier result stand;
the v2 own-action rows are re-sampled in verb spelling (the first sampling run was cancelled at
~3,000 of 11,300 replies).

Status 2026-10-23: the shared mechanism (`verb_surface`, `verb_table`, codecs in
`dataxp/domain.py`; render_cot / ownrows / probe_info / gold_probe switched to
`native_moves or verb_surface`) and the first surface, frame3d2 (`INSTALL_MEMBER` /
`REMOVE_MEMBER`, `SET_MATERIAL`, `SET_RADIUS`, `MOVE_JOINT`), are in; guide and checks in
`docs/dataxp_verb_surface.md`. Quick generations of frame3d2 and circuit2 byte-identical
before/after (PYTHONHASHSEED=0); the other six surfaces follow the pattern there. The drc
harness copy of `dataxp/` was re-uploaded 2026-10-23 12:08 EDT (verbatim, 47 files) and the
harness routes every `speaks_verbs` domain through the verb-spelled path (system/instruction =
`ownrows.common` served messages, history through `format_move`, `<action>` through `parse_move`;
`/storage/wxu/corpus/harness/README.txt`, section "verb surfaces"). On the three old-variant
problems every per-turn numeric is identical to the original harness; only the served user text
and the new `action` key differ. Native molopt: identical except the DOMAIN/GOAL head, which the
local `render_cot.DOMAIN_BLURB` now spells out (the previous copy served the fallback).

Verb surfaces as built (workflow wf_6211cbcf-15c, all seven verified pass; quick generations
byte-identical before/after; round-trip bijection on every legal move; guide
`docs/dataxp_verb_surface.md`):

| domain | verbs (typed arguments; units in `grammar()`) |
|---|---|
| frame3d2 | INSTALL_MEMBER(m) / REMOVE_MEMBER(m), SET_MATERIAL(m, Al6061T6 \| S355 \| Ti6Al4V), SET_RADIUS(m, r), MOVE_JOINT(j, x\|y\|z, value) |
| circuit2 | SET_WIDTH(d, um), SET_LENGTH(d, um), SET_MIRROR_RATIO(MB, ratio), SET_COMP_CAP(CC, pF), SET_COMPENSATION(CC, DirectMiller \| VoltageBufferedMiller \| CurrentBufferedMiller), SET_BIAS_CURRENT(IB, uA), SET_NULLING_RESISTOR(RZ, kOhm), ENABLE_NULLING / DISABLE_NULLING(RZ), ENABLE_CASCODE / DISABLE_CASCODE(CAS) |
| grid2 | DISPATCH(g, MW), COMMIT_UNIT / DECOMMIT_UNIT(g), CLOSE_LINE / OPEN_LINE(l), SET_PHASE_SHIFT(pst, degrees) |
| traffic2 | SET_CYCLE(i, s), SET_GREEN_SPLIT(i, share), SET_OFFSET(i, s), SET_PHASE_ORDER(i, MAIN_FIRST \| CROSS_FIRST), SET_LANES(i, n) |
| seir2 | SET_CONTACT_REDUCTION(g, share), SET_INTERVENTION_START(measure, day), SET_INTERVENTION_DURATION(measure, days), SET_TESTING_RATE(g, rate), SET_VACCINE_SHARE(g, share), SET_SCHOOL_POLICY(K, open \| small_cohorts \| hybrid \| closed), MANDATE_MASKS / LIFT_MASKS(K+1) |
| thermal2 | MOVE_SINK(k, sJ), SET_CONDUCTANCE(k, W/K), INSTALL_LINK / REMOVE_LINK(l), PLACE_BLOCK(b, qJ) |
| inventory2 | SET_BASE_STOCK(n, units), SET_REVIEW_PERIOD(n, periods), OPEN_STOCKING_POINT / CLOSE_STOCKING_POINT(n), ENABLE_EXPEDITING / DISABLE_EXPEDITING(n) |

Consequences: the v2 own-action rows are re-sampled in verb spelling (bw_rows4, after bw_rows3);
the evaluation harness v3 is patched so every verb-speaking domain is served and parsed in verbs
(the four native variants unchanged); the served instruction for all eleven families is now the
native one (verb form + `grammar()`), so earlier `SET_PARAM`-spelled rows and the V2 instruction
(sha1 3d4867180337) are not mixed with the new rows.

### Decision 2026-10-23 (owner): two spellings, `datav4_multitool` and `datav4_set_parameter`

A second tool-description set keeps one wrapper tool, `SET_PARAM(element, parameter, value)`,
but documents every operation exactly as the verb set does ("to install a member:
`SET_PARAM(member, active, 1)` …; to set its material: `SET_PARAM(member, mat, code)`, 0 =
Al6061T6, 1 = S355, 2 = Ti6Al4V …"). The two conditions differ only in action syntax, not in
what the model is told. Scope: all eleven families — the v2 seven with their real parameters,
the native four through a generic encoding `SET_PARAM(element, operation, value)` (element = the
operation's first integer argument, operation = the verb in lower case, value = the remaining
argument or a JSON list of them), documented operation by operation. Categorical values are
written as the stored codes in the action line (the grammar names them). Same states, same
teacher, same acceptance rule; the own-action rows are sampled once per spelling. Both datasets
are stored on bridges2 under `/ocean/projects/mch250030p/shared/dataxp/datav4_multitool/` and
`/ocean/projects/mch250030p/shared/dataxp/datav4_set_parameter/` (corpora, rows, grammar text,
instruction, manifest, README), clearly separated.

Status 2026-10-23: the switch (`DATAXP_SPELLING`, read once by `dataxp.domain`) and the
set_parameter spelling of the seven v2 domains are in, generated from their verb tables
(`set_param_format` / `set_param_parse` / `set_param_grammar` hooks; `ParamVerb.phrase`), and
honoured by render_cot, ownrows, probe_info and gold_probe, which record the spelling they served.
Multitool is byte-identical to before (grammars, renders, quick generations, probe outputs); the
instructions are `1c344388a444` (multitool) and `fe8fa8db0a90` (set_parameter; `f99b61d776f1`
until the sentence that still said VERB was reworded, 2026-10-23), differing in the action line
and the sentence that explains it. The native four speak the generic encoding (`SET_PARAM(element,
operation, value)` from `verbs`: `SET_PARAM(12, add_fragment, hydroxyl)`, `SET_PARAM(3,
redispatch, [5, 20])`, `SET_PARAM(0, do_nothing, 0)`) and are verified per domain (round trip
on every legal move of 51-66 states each, 21 for photonics; grammars, prompts and quick
generations byte-identical under multitool; ownrows dry run on molopt under set_parameter):
`VerbSpec.phrase` on every verb, the observation and grammar-wrapper prose re-spelled through
`Domain.op_name` / `call_form`. Guide and the check log: `docs/dataxp_spellings.md`. Nothing
uploaded to bridges2 from this step.

Own-action row yields, multitool spelling (drc, bw_rows3, K=4, 2,000-token cap, rule ≥ 90th Φ
percentile or the teacher's move): engibench 39 rows of 164 states (same move as the teacher 13%),
photonics 11 of 117 (0%). The un-tuned model rarely lands a top-decile move on these two families
(its feasible@12 there is 0.45 and 0.02), so they contribute few rows; the rule is kept as
pre-registered and the per-family counts are reported with the slope test.

### Orchard (2026-10-23, after the owner opened it)

Harness ported: `/project/community/wxu2/corpus/harness/` (scripts/dx_eval_orchard.sh — one
vLLM server per GPU, arm×variant pairs round-robin, one episodes file per pair, `SKIP_DONE`
for requeues; client env `envs/corpus_o`, an exact lock of drc's corpus env; server env
`envs/vllm029`). Validation: native non-LLM episodes 10/12 files byte-identical to drc, the rest
differ only in ~1e-14 printed digits; old variants byte-identical. QoS: general 1 GPU / 1 job /
12 h; advanced 4 GPUs but **1 node per user**, 24 h, 6 jobs; preempt 32 GPUs / 2 d, requeued.

Base reference on the native evaluation problems (`base_nothink`: un-tuned, thinking off,
1,024-token cap as on drc, verb-spelled prompts; 60 problems per family, one run; Orchard job
160740, 1.1 GPU-h):

| family | feasible@12 | feasible@20 | parse rate | median reply tokens | at cap | dominant failure |
|---|---|---|---|---|---|---|
| engibench | 0.45 | 0.45 | 0.97 | 261 | 3% | illegal moves 158 |
| molopt | 0.22 | 0.27 | 0.89 | 265 | 11% | illegal 127, no action 93 |
| grid2op | 0.13 | 0.15 | 0.87 | 157 | 13% | 576 turns after game-over (harness keeps asking) |
| photonics | 0.02 | 0.02 | 0.99 | 237 | 1% | 984 min-feature violations |
| pooled | 0.20 | 0.22 | 0.93 | 242 | 7% | |

Episodes: Orchard `harness/eval/r1/base_nothink_<variant>_episodes.episodes.jsonl`, copied to drc
`/storage/wxu/corpus/harness/eval/r1/`. Open harness item: stop a grid2op episode at game-over
instead of asking for 576 more moves.

### datav4 delivered (2026-10-23 20:05)

Own-action rows (K=4 replies per state, 2,000-token cap, rule ≥ 90th Φ percentile or the
teacher's move; same states in both spellings, sampled once per spelling on Orchard, 11.5 GPU-h):

| family | states | multitool rows (train / dev) | set_parameter rows (train / dev) |
|---|---|---|---|
| seir2 | 602 | 369 / 7 | 374 / 11 |
| frame3d2 | 614 | 309 / 10 | 333 / 10 |
| traffic2 | 442 | 164 / 0 | 193 / 0 |
| circuit2 | 608 | 145 / 2 | 138 / 3 |
| grid2 | 501 | 83 / 0 | 100 / 0 |
| grid2op | 167 | 71 | 86 |
| engibench | 164 | 39 | 41 |
| photonics | 117 | 11 | 13 |
| **total** | 3,215 | **1,191** | **1,278** |

38% of v2 replies ended at the 2,000-token cap (the main yield limiter; truncated replies count
as no sample). molopt is held out and has no rows. A cache-reuse mistake in the first tier build
(the Φ-cache files carried the pre-rewrite state records, so the rows' served prompts were stale)
was caught by inspecting a row and fixed by rebuilding the caches from the fresh state records
before re-scoring; the samples themselves were always drawn with the correct prompts.

Storage (bridges2, `/ocean/projects/mch250030p/shared/dataxp/`): `datav4_multitool/`,
`datav4_set_parameter/` (each: README, served instruction, per-family grammar, rows per family
with manifest, manifest.json with sha1s; the multitool one also carries the base_nothink r1
episodes), and the shared `datav4_corpus/` (spelling-independent problems/trajectories/pairs per
family: native corpora, run2_supp for the five training families, run2 copied as
`accepted_run2/`). Built and self-checked on drc (`dataxp/package_datav4.py --check`, parse of
every action line under its spelling, grammar identical to the current code).

### Appending batches (2026-10-24)

b001 above is never rewritten. Later batches go to new directories only
(`datav4_<spelling>/batches/b00N/`, `datav4_corpus/<family>/accepted_b00N/`), in both spellings, with
the b001 recipe plus one redraw pass of the truncated keys. The tools are `run_generation
--start-index`, `select_states --exclude-problems / --exclude-specs / --split-from`, `sample
--max-draw-pass` (records `draw_pass`), `ownrows/batch_prep.py` and `package_datav4 --append /
--corpus-batch / --check`. The batches b002-b005, their rules, the decisions left to the owner and the
exact commands are in `docs/datav4_append_plan.md`.

### EngiBench shared base designs (owner decision 2026-10-24: keep, flagged)

EngiBench problems are built on a limited set of dataset rows (load case, boundary conditions,
reference optimum); the 60 evaluation problems use 60 of them, so some training problems share a
base design with an evaluation problem (different start design and targets; the canonical-spec
check does not catch it). 47 training problems (b001: 7, b004: 40) contribute 26 multitool / 29
set_parameter engibench rows (b001 4, b002 4, b004 18 multitool; b001 4, b004 25 set_parameter).
They stay as training rows; the list is `datav4_corpus/engibench/shared_base_with_eval.json` on
bridges2 (problem id, batch, base, the evaluation problem it shares it with, rows per file), so
they can be dropped when a strictly base-disjoint engibench evaluation is wanted, and any
engibench in-domain readout is reported with that caveat. Future batches keep the same rule and
extend the list.

Correction (global verification, 2026-10-24): "different start design" does not hold for every
flagged engibench problem — 4 training problems have the same start design as an evaluation
problem (engibench_0240 differs from evaluation engibench_0120 only in its goals); they carry 0
multitool / 2 set_parameter rows. The same check found the analogous overlap in the other two
native families, which is structural (their generators draw from small spaces):

| family | tier | training problems | rows (multitool / set_parameter) |
|---|---|---|---|
| photonics | shares the reference design of an evaluation problem | 200 of 200 | 33 / 38 (all) |
| photonics | same reference and goals | 152 | 22 / 29 |
| photonics | differs from an evaluation problem only in the start design | 72 | 10 / 14 |
| grid2op | uses a chronic that an evaluation problem uses | 191 of 191 | 222 / 258 (all) |
| grid2op | same chronic, forced outage and pre-actions (differs in t0) | 50 | 35 / 39 |
| engibench | shared base (family, source_row) | 47 | 26 / 29 |
| engibench | same start design | 4 | 0 / 2 |

Same policy (keep, flagged): per-problem lists with the closest evaluation problem and the keys
that differ are `datav4_corpus/{grid2op,photonics,engibench}/eval_overlap.json`
(`/storage/wxu/corpus/tools/native_overlap_flags.py` regenerates them over every batch). Native
in-domain evaluation is therefore "new instances of the same small device/grid/base sets", not
unseen designs; the held-out-family readouts (molopt, thermal2, inventory2) are unaffected.

## What comes after the corpus

1. Leave-families-out slope on our 9 + the new families (own-action rows, k families in,
   held-out families out).
2. Own-action row generation (K samples per state from the un-tuned Qwen3.8-27B, simulator
   acceptance, no teacher imitation), per the recipe that kept transfer (own_act).

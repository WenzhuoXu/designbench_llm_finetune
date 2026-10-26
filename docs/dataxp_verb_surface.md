# dataxp verb surfaces — implementer guide

The owner's decision of 2026-10-23 (`docs/corpus_native_domains.md`, "the v2 seven get native
verb surfaces"): every v2 domain keeps `SET_PARAM(element, param, value)` as its internal move
and gets a thin spelling layer in the field's own verbs. This page is the code contract for
that layer, the checks a surface must pass, and the first worked example (`frame3d2`).

## What a verb surface is, and what it may not touch

A verb surface is a **bijection on the existing legal moves**: every legal `SET_PARAM` move has
exactly one verb text, and every verb text parses back to the same `SET_PARAM` move.

| unchanged (run2 data and every earlier result depend on it) | changed (prompt-facing only) |
|---|---|
| `params`, `candidate_moves` / `candidate_values`, `enumerate_candidates`, `legal_moves` | `format_move` (the action-line text) |
| `set` / `get` / `bounds`, `apply` | `parse_move` (what a reply must say) |
| Φ scoring, gates, ladder, harvest, replay | `grammar()` (the verb list in the system prompt / instruction) |
| `Move.as_dict()` of a SET_PARAM move: `{i, p, v, v_old}` | `describe()`'s "how to write a change" paragraph (observation text; no generated file contains it) |
| `smoke`'s `legal_moves == enumerate_candidates` check | history / "change you are about to make" lines, `<action>` parsing in render_cot, ownrows, probe_info, gold_probe |

Rules that do not bend: never narrow or widen the move set; record every judgment call in the
module docstring section **"Verb surface"**; verb names and argument types come from the field
(member, joint, device, generator, line, signal, sink, block, group, node), never "set parameter
p of element e"; categorical codes are spelled as their names, toggles as paired verbs, continuous
values as plain numbers with their unit named in `grammar()`.

## Declaring a surface (`dataxp/domain.py`)

```python
from dataxp.domain import ArgSpec, Categorical, Continuous, Domain, ParamVerb, Toggle

class MyV2Domain(Domain):
    ...                                   # everything the v2 domain already has
    verb_surface = True                   # class flag; native_moves stays False
    verb_table = (                        # one ParamVerb per entry of `params`
        ParamVerb("active", "INSTALL_MEMBER", _MEMBER,                 # toggle: 1 <-> verb, 0 <-> off
                  Toggle("REMOVE_MEMBER", off_doc="remove an installed member ..."),
                  doc="install an optional member ..."),
        ParamVerb("mat", "SET_MATERIAL", _MEMBER,                      # categorical: code k <-> names[k]
                  Categorical("material", ("Al6061T6", "S355", "Ti6Al4V"),
                              aliases=(("Al", 0), ("St", 1), ("Ti", 2)), doc="Al6061T6 = ..."),
                  doc="set a member's material; its radius is kept."),
        ParamVerb("r", "SET_RADIUS", _MEMBER,                          # continuous: the new absolute value
                  Continuous("radius", "m", "0.004..0.12, the new outer tube radius"),
                  doc="resize a member's tube."),
        ParamVerb("dx", "MOVE_JOINT", _JOINT, _OFFSET, select=("axis", "x"), doc=_JOINT_DOC),
        ParamVerb("dy", "MOVE_JOINT", _JOINT, _OFFSET, select=("axis", "y"), doc=_JOINT_DOC),
        ParamVerb("dz", "MOVE_JOINT", _JOINT, _OFFSET, select=("axis", "z"), doc=_JOINT_DOC),
    )

_MEMBER = ArgSpec("member", "int", "member number 0..M-1 (the number after m)")
_JOINT = ArgSpec("joint", "int", "joint number (the number after J)")
_OFFSET = Continuous("offset", "m", "the joint's new total offset ... within the +- bound the observation states")
```

* `ParamVerb(param, verb, elem, codec, select=None, doc="")`
  * `param` — the SET_PARAM parameter name; the table must cover `params` exactly once each.
  * `verb` — the verb name (`[A-Z][A-Z0-9_]*`; for a `Toggle` codec it is the ON verb).
  * `elem` — the element argument, an `ArgSpec` of type `int`: it is the element index
    SET_PARAM used, and its `doc` tells the policy how to read it off the observation.
  * `codec` — the value codec:
    * `Toggle(off, off_doc="")`: value 1 ↔ `verb(elem)`, value 0 ↔ `off(elem)`; no value argument.
      `doc` on the ParamVerb documents the ON verb, `off_doc` the OFF verb.
    * `Categorical(arg, names, aliases=(), doc="")`: code k ↔ `names[k]`; `format_move` writes the
      canonical name, `parse_move` accepts the names, the aliases and any letter case. A bare code
      number is not accepted.
    * `Continuous(arg, unit="", doc="")`: the argument is the new absolute value as a plain number
      (`fmt_arg`: 6 significant digits, integral values bare); the unit appears in `grammar()` only.
  * `select=(arg_name, literal)` — an extra literal argument written between the element and the
    value, so several parameters share one verb (`dx/dy/dz` → `MOVE_JOINT(j, axis, v)`). Rows that
    share a verb must differ in the literal; `grammar()` prints the verb once with every literal.
  * `doc` — one line on what the verb does physically (the grammar line under the signature).
* `check_verb_table(domain)` runs in `Domain.__init__` for every `verb_surface` domain: coverage of
  `params`, distinct spellings, valid verb names, non-empty category tables. A domain that sets
  `verb_surface = True` without a table must override `format_move`, `parse_move` **and**
  `grammar` itself (the check enforces it).
* Overrides: the table-driven `format_move` / `parse_move` / `grammar` are the defaults. Override
  only what the table cannot express (e.g. a verb whose arguments are not element + one value);
  keep the bijection and the SET_PARAM move as the parse result, and record why in the docstring.
* Helpers you may reuse: `surface_format`, `surface_parse`, `surface_grammar`, `surface_value`
  (module functions), `Domain.verb_param(p)`, `Domain.format_value(p, v)` (the value as the codec
  spells it; `''` for a toggle — the history line's "was ..." uses it), `speaks_verbs(domain)`
  (`native_moves or verb_surface`: the test every prompt-facing path uses).

### What the pipeline does with it

| path | behaviour on a verb-surface domain |
|---|---|
| `render_cot.system_prompt` | the SET_PARAM action-format line is replaced by `<action>VERB(arg, ...)</action>` + `grammar()` (as for native domains); rendered rows carry `system_prompt` |
| `render_cot.policy_observation` | history lines `N. VERB(...) (<element label>; was <old value>)` — no "was" for a toggle |
| `render_cot.build_prompt` | `_build_prompt_surface`: the change line is `VERB(...)  (<label>; <arg> old -> new (direction; room ...))` for a continuous parameter, `(<label>; <arg> old_name -> new_name)` for a categorical one, `(<label>)` for a toggle; "End with: `<action>VERB(...)</action>`" |
| `render_cot.check` / `check_row` / `rescore` | the `<action>` tag is read with `parse_move` and compared with `Move.matches(gold, rel=1e-2)`; `check_row` fetches the domain itself |
| `dataxp/ownrows` | `instruction_for` serves `INSTRUCTION_NATIVE` + `grammar()` (sha1 `INSTRUCTION_NATIVE_SHA1`); `own_system_prompt` substitutes the verb form; `select_states` records `spelling` (`multitool` here; `set_parameter` under the other spelling, `docs/dataxp_spellings.md`); `sample --dry-run`, `score`, `build_tiers` go through `format_move` / `parse_move`, so action lines are verb-spelled while `action` / `gold_action` stay `{i, p, v, v_old}` |
| `probe_info` | the observation ends with the verb list; answers are parsed with `parse_move` |
| `gold_probe` | prev/post traces, the rat "change made" line, the own_cands list and the own_elem hint print through `format_move` / `format_value`; replies, candidates and compress use `parse_move`; the system prompts get the verb-call line + grammar. Plain v2 domains keep the historical SET_PARAM text and the `ACTION_RE` rule byte for byte |
| `smoke` | unchanged: for a non-native domain it still asserts `legal_moves == enumerate_candidates` dict for dict, and the format/parse round trip now exercises the verbs |
| ladder, harvest, replay, Φ | never see the surface |

`parse_move` on a verb-surface domain does **not** accept `SET_PARAM(...)` text: the grammar the
policy is served lists only the verbs. Rendered rows or own-action rows produced before the
surface (SET_PARAM action lines) must be regenerated, not rescored.

**The second spelling.** Everything above is the `multitool` spelling (the default). With
`DATAXP_SPELLING=set_parameter` the same table is served as one wrapper tool,
`SET_PARAM(element, parameter, value)`, documented operation by operation ("Install a member:
SET_PARAM(member, active, 1)" ...): `format_move` writes the classic `SET_PARAM(i, p, v)` line,
`parse_move` accepts only that, `grammar()` is generated from `verb_table` (the `phrase` field of
`ParamVerb`, `off_phrase` of `Toggle`), and the observation's "how to write a change" paragraph
follows `spelled_in_verbs(domain)` instead of `verb_surface`. Contract, hooks and checks:
`docs/dataxp_spellings.md`.

## Checks a surface must pass

Run from the repository root with `PYTHONHASHSEED=0 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1`, one
worker, `--quick`. `S` is a scratch directory; `compare_runs.py` is the Move-contract agent's
byte comparer (corpus scratch dir).

```bash
# (a) bit-for-bit: quick generation before and after the change
python -m dataxp.run_generation --variant <v> --quick --workers 1 --seed 0 --tmp $S/pre_tmp --out $S/pre_out   # before
python -m dataxp.run_generation --variant <v> --quick --workers 1 --seed 0 --tmp $S/post_tmp --out $S/post_out  # after
python compare_runs.py $S <v> ...            # every *.jsonl BYTE-IDENTICAL, report.json equal ignoring timing
# (b) contract smoke
python -m dataxp.smoke <v> circuit2 thermal2
# (c) round trip on >= 50 states x all legal moves: parse_move(format_move(m)).matches(m),
#     no "SET_PARAM" in any verb text, spellings unique per state, legal_moves == enumerate_candidates
#     (scratch script: corpus/verbs/roundtrip_check.py; adapt the variant and the parse edge cases)
# (d) prompts: stage the quick corpus as <stage>/<v>/accepted/{problems,trajectories}.jsonl, then
python -m dataxp.render_cot --data $S/stage --out $S/render_post --variants <v> circuit2 --prompts-only --limit 6
#     -> verb grammar in system_prompt, verb-spelled history/change/action lines; a domain without a
#        surface renders byte-identically to its pre-change render
# (e) own-action rows, dry run
python -m dataxp.ownrows.select_states --data $S/pre_out --tmp $S/pre_tmp --variants <v> --out $W --per-domain 8 --dev-per-domain 4
python -m dataxp.ownrows.cand_phi --states $W/states.jsonl --out $W/states_phi.jsonl --workers 1
python -m dataxp.ownrows.sample --states $W/states.jsonl --out $W/samples.jsonl --k 4 --dry-run --dry-run-teacher-share 0.25
python -m dataxp.ownrows.score --states $W/states_phi.jsonl --samples $W/samples.jsonl --out $W/scored.jsonl
python -m dataxp.ownrows.build_tiers --states $W/states_phi.jsonl --samples $W/samples.jsonl --scored $W/scored.jsonl --out $W/tiers
#     -> every assistant line is <action>VERB(...)</action>; every stored action dict is {i, p, v, v_old}
```

## The frame3d2 example (`dataxp/domains/frame3d2.py`)

| SET_PARAM | verb spelling |
|---|---|
| `SET_PARAM(3, active, 1)` / `SET_PARAM(3, active, 0)` | `INSTALL_MEMBER(3)` / `REMOVE_MEMBER(3)` |
| `SET_PARAM(3, mat, 2)` | `SET_MATERIAL(3, Ti6Al4V)` (names `Al6061T6`, `S355`, `Ti6Al4V`; `Al` / `St` / `Ti` accepted on parse) |
| `SET_PARAM(3, r, 0.031)` | `SET_RADIUS(3, 0.031)` (m) |
| `SET_PARAM(12, dx, 0.25)` | `MOVE_JOINT(12, x, 0.25)` (m; the new total offset; `y`, `z` for dy, dz) |

`grammar()` as served (one verb per line, then what it does):

```
INSTALL_MEMBER(member: int member number 0..M-1 (the number after m))
    install an optional (or removed) member, with its listed radius and material.
REMOVE_MEMBER(member: int member number 0..M-1 (the number after m))
    remove an installed member: it then carries no stiffness and no mass (offered only where the truss stays rigid).
SET_MATERIAL(member: int member number 0..M-1 (the number after m), material: one of Al6061T6 | S355 | Ti6Al4V)
    set a member's material; its radius is kept. Al6061T6 = 6061-T6 aluminium (E 68.9 GPa, yield 276 MPa, 2700 kg/m3), ...
SET_RADIUS(member: int member number 0..M-1 (the number after m), radius: float [m] 0.004..0.12, the new outer tube radius (a value outside is set to the bound))
    resize a member's tube.
MOVE_JOINT(joint: int joint number (the number after J), axis: one of x | y | z, offset: float [m] the joint's new total offset from its initial position along that axis, within the +- bound the observation states)
    move a free joint along one axis; every member at it follows. A loaded joint's coordinate along the structure's axis is locked: such a move changes nothing.
```

Judgment calls (full list in the module docstring, "Verb surface"): element numbers, not node
numbers, for both members and joints (the J number the observation prints); the offset is the new
total, not a step; material names are the alloy designations as identifiers with the row codes as
parse aliases, no bare code numbers; SET_PARAM text is no longer parsed; a MOVE_JOINT on a loaded
joint's locked axis parses and does nothing, exactly as the SET_PARAM did (legal_moves never
offers it); `describe()`'s CHANGES paragraph is re-spelled in the verbs, conditional on
`verb_surface` so the SET_PARAM text is one flag away.

Results (2026-10-23, Windows, PYTHONHASHSEED=0): quick generations of frame3d2 and circuit2
byte-identical before/after (pairs, per_problem, problems, trajectories; report.json equal
ignoring timing); `smoke frame3d2 circuit2 thermal2` PASS; round trip on 60 states (20 problems x
start / +1 / +2 random moves) x 8,136 legal moves, all match, no SET_PARAM in any text, spellings
unique per state (INSTALL 966, REMOVE 720, SET_MATERIAL 1,680, SET_RADIUS 2,810, MOVE_JOINT
1,960); render: circuit2 byte-identical, frame3d2 changed only in the CHANGES paragraph, history
lines, change line, action line and the new `system_prompt` key (18 of 116 prompt lines);
ownrows dry run: 13 states, 52 replies all parsed and legal, 10 rows, every action line a verb
call, every stored dict `{i, p, v, v_old}`, instruction = native + grammar.

## Pattern for the remaining six (circuit2, grid2, traffic2, thermal2, seir2, inventory2)

1. Read the domain's `params`, `candidate_values` / `candidate_moves` and the v2 table in
   `docs/dataxp_v2.md`; list every parameter with its kind (toggle / categorical / continuous or
   additive) and the element kind it applies to (device, generator, line, PST, signal, sink,
   link, block, group, stage).
2. Choose one verb per parameter in that field's language; use a `Toggle` pair for every 0/1
   parameter (`OPEN_LINE` / `CLOSE_LINE`, `COMMIT_UNIT` / `DECOMMIT_UNIT`, `ENABLE_EXPEDITING` /
   `DISABLE_EXPEDITING`, ...), a `Categorical` with the field's names for every code table
   (compensation schemes, school policies, review periods, cycle lengths), a `Continuous` with its
   unit for every number (an additive or circular parameter is still "the new value": say so in
   the doc). Parameters that are the same operation on different axes or slots share a verb
   through `select`.
3. Elements with different kinds (e.g. thermal2's sinks, links and blocks) keep the element index
   SET_PARAM used; the `elem` doc says how the observation numbers them. A parameter that applies
   only to a subset of elements (`get` returns None elsewhere) needs no special case: the move set
   is unchanged and `apply` behaves as `set` did.
4. Re-spell the `describe()` "how to write a change" paragraph in the verbs, conditional on
   `verb_surface`; leave every other observation line alone.
5. Write the docstring section "Verb surface" with the bijection table and every judgment call,
   then run checks (a)–(e).

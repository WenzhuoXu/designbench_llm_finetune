# dataxp spellings — `multitool` and `set_parameter`

The owner's decision of 2026-10-23 (`docs/corpus_native_domains.md`, "two spellings,
`datav4_multitool` and `datav4_set_parameter`"): the same operations are served in two
action syntaxes, and the own-action rows are sampled once per syntax. This page is the code
contract for the switch, what each spelling writes, how the `set_parameter` grammar is generated,
the checks the implementation passed, and one worked example (`frame3d2`).

## What each spelling is

Both spellings are bijections on the SAME legal moves. `legal_moves`, `apply`, Φ, the gates, the
ladder, harvest, replay and the stored action dicts (`{i, p, v, v_old}` for a v2 domain,
`{verb, args}` for a native one) never see the spelling; one corpus serves both. Only
`format_move` / `parse_move` / `grammar` (and `format_value`, the "was ..." of history lines) and
therefore the served prompts differ.

| | `multitool` (default) | `set_parameter` |
|---|---|---|
| what it is | the verb surface / native verbs as built (`docs/dataxp_verb_surface.md`, `docs/dataxp_move_contract.md`) | one wrapper tool `SET_PARAM(element, parameter, value)`; its grammar documents every operation of the verb table, operation by operation |
| frame3d2 action lines | `INSTALL_MEMBER(3)`, `SET_MATERIAL(3, S355)`, `SET_RADIUS(3, 0.031)`, `MOVE_JOINT(12, x, 0.25)` | `SET_PARAM(3, active, 1)`, `SET_PARAM(3, mat, 1)`, `SET_PARAM(3, r, 0.031)`, `SET_PARAM(12, dx, 0.25)` |
| categorical values | the name (`S355`; aliases accepted on parse) | the STORED CODE (`1`); the grammar names the codes; a name in the value slot does not parse |
| `parse_move` accepts | the verbs only | `SET_PARAM(i, p, v)` only (the v2 rule: bare integer element, identifier parameter, plain number value); verbs are rejected |
| served instruction (ownrows) | `INSTRUCTION_NATIVE` + `grammar()`, sha1 `1c344388a444` | `INSTRUCTION_SET_PARAM` + `grammar()`, sha1 `fe8fa8db0a90` — the same text with `<action>VERB(arg, ...)</action>` replaced by `<action>SET_PARAM(element, parameter, value)</action>` and "VERB is one of the operations listed below, written exactly as listed, ..." by "SET_PARAM performs one of the operations listed below, written exactly as listed, ..."; nothing else changes (sha1 `f99b61d776f1` was the version that still said VERB; fixed 2026-10-23 after the verifier's pass) |
| system prompts (render_cot, own_system_prompt, gold_probe) | action line `<action>VERB(arg, ...)</action>` + "where VERB is one of this domain's verbs, with the argument types and ranges listed here:" + `grammar()` | action line `<action>SET_PARAM(element, parameter, value)</action>` + "where the operations, with their argument types and ranges, are listed here:" + `grammar()` (`grammar_lead()`); the rest of the text is identical. probe_info's observation tail says "with one of these verbs:" / "performing one of these operations:". No served text says VERB under set_parameter |
| observation "how to write a change" paragraph (v2 domains) | the verb paragraph | the historical SET_PARAM paragraph the run2 data were generated with (`spelled_in_verbs(domain)` is False) |
| history / change lines | `2. SET_RADIUS(13, 0.01621) (member 13 (...); was 0.01474)`, `SET_MATERIAL(22, S355)  (member 22 (...); material Al6061T6 -> S355)` | `2. SET_PARAM(13, r, 0.01621) (member 13 (...); was 0.01474)`, `SET_PARAM(22, mat, 1)  (member 22 (...); material 0 -> 1)` |
| native four | their verbs (`ADD_FRAGMENT(12, hydroxyl)`, `DO_NOTHING()`) | the generic encoding `SET_PARAM(element, operation, value)` from the `verbs` table (below): `SET_PARAM(12, add_fragment, hydroxyl)`, `SET_PARAM(0, do_nothing, 0)`; verified per domain 2026-10-23 (checks (e)-(h) below) |

## The switch: `DATAXP_SPELLING`

```bash
DATAXP_SPELLING=set_parameter python -m dataxp.ownrows.select_states ...   # set_parameter
python -m dataxp.ownrows.select_states ...                                 # multitool (default)
DATAXP_SPELLING=multitool ...                                              # the same as unset
```

* `dataxp.domain.current_spelling()` reads the variable once per process on first use (any
  other value raises). Worker processes inherit it.
* `speaks_verbs(domain)` (native or verb surface) is unchanged and True under both spellings: it
  still means "goes through format_move / parse_move / grammar". The new predicate
  `spelled_in_verbs(domain)` = `speaks_verbs and spelling == multitool` is what the observation
  text of the seven v2 domains switches on. `action_form(domain)` is the action placeholder the
  prompts print (`VERB(arg, ...)` / `SET_PARAM(element, parameter, value)`).
* Every prompt-facing path honours it through the three methods: `render_cot` (system prompt,
  history / change / action lines, `check`), `dataxp/ownrows` (`instruction_for`,
  `own_system_prompt`, `select_states`, `sample --dry-run`, `score`, `build_tiers`),
  `probe_info`, `gold_probe` (`_fmt_action`, `_was`, `_last_action`, `_system_for`, `_verb_of`:
  the hint names "parameter mat" under set_parameter).
* What records the spelling: `states.jsonl` (`spelling`: `multitool` | `set_parameter` for a
  domain that speaks verbs, `SET_PARAM` for a plain v1/v2 domain; the `native` bool says which
  kind), every tier row (`spelling`), `probe_info` `states.json` rows and `gold_probe`
  `states.jsonl` rows. Rendered rows (`render_cot`) carry no new key, so the multitool render
  stays byte-identical to the pre-switch output; their `system_prompt` text identifies the
  spelling.
* The generation path (`run_generation`, ladder, harvest) is spelling-blind; running it under
  either value produces byte-identical files (checked, below).

## The generated `set_parameter` grammar

`Domain.grammar()` dispatches to `set_param_grammar()` under `set_parameter`. For a verb-surface
domain the default (`surface_set_param_grammar`) is generated from `verb_table`:

```
The only tool is SET_PARAM(element, parameter, value). The operations it performs in this problem:
<phrase>: SET_PARAM(<elem.name>, <param>, <value>)  <elem.name>: int <elem.doc>[; <value description>]
    <verb doc>
...
```

* one block per operation, in the verb table's order; a `Toggle` is two blocks (value `1` with
  the ParamVerb's `doc`, value `0` with the Toggle's `off_doc`); a parameter that shares a verb
  through `select` (frame3d2's `dx` / `dy` / `dz`) is its own block, since SET_PARAM names it;
* the value slot is `1` / `0` for a toggle, `code` for a `Categorical` (described as
  `code: 0 = <name0>, 1 = <name1>, ...; <codec doc>`), the codec's argument name for a
  `Continuous` (described as `<arg>: float [<unit>] <doc>`): the same texts the multitool grammar
  prints inside the verb signature, so both grammars say the same things;
* the phrase comes from `ParamVerb.phrase` (`Toggle.off_phrase` for the OFF block), defaulting to
  the verb name in words (`default_phrase`: `INSTALL_MEMBER` -> "Install member"); all seven
  tables set explicit phrases ("Install a member", "Set a device's channel width", "Dispatch a
  unit", "Re-stripe a signal's MAIN lanes", "Move a heat sink", "Mandate masks", "Open a stocking
  point", ...);
* thermal2 overrides `set_param_grammar` the way it overrides `grammar()`: the 64-entry
  `code: 0 = s0, 1 = s1, ...` lists of its site / slot codecs become the rule "the site's code
  number from the Sites list (the number after s: 5 for s5), not a node number"
  (`set_param_code_list(codec)` is the exact string a domain replaces).

The `set_parameter` grammars of all seven, and the multitool ones for comparison, are in the
corpus scratch dir `corpus/spelling/grammar_{set_parameter,multitool}_<domain>.txt`.

### Example: frame3d2, both grammars

multitool (`grammar()` as built; byte-identical to the pre-switch output):

```
INSTALL_MEMBER(member: int member number 0..M-1 (the number after m))
    install an optional (or removed) member, with its listed radius and material.
REMOVE_MEMBER(member: int member number 0..M-1 (the number after m))
    remove an installed member: it then carries no stiffness and no mass (offered only where the truss stays rigid).
SET_MATERIAL(member: int member number 0..M-1 (the number after m), material: one of Al6061T6 | S355 | Ti6Al4V)
    set a member's material; its radius is kept. Al6061T6 = 6061-T6 aluminium (E 68.9 GPa, yield 276 MPa, 2700 kg/m3), S355 = S355 structural steel (E 210 GPa, yield 355 MPa, 7850 kg/m3), Ti6Al4V = Ti-6Al-4V titanium (E 113.8 GPa, yield 880 MPa, 4430 kg/m3); the member rows abbreviate them Al / St / Ti.
SET_RADIUS(member: int member number 0..M-1 (the number after m), radius: float [m] 0.004..0.12, the new outer tube radius (a value outside is set to the bound))
    resize a member's tube.
MOVE_JOINT(joint: int joint number (the number after J), axis: one of x | y | z, offset: float [m] the joint's new total offset from its initial position along that axis, within the +- bound the observation states)
    move a free joint along one axis; every member at it follows. A loaded joint's coordinate along the structure's axis is locked: such a move changes nothing.
```

set_parameter (generated from the same table):

```
The only tool is SET_PARAM(element, parameter, value). The operations it performs in this problem:
Install a member: SET_PARAM(member, active, 1)  member: int member number 0..M-1 (the number after m)
    install an optional (or removed) member, with its listed radius and material.
Remove a member: SET_PARAM(member, active, 0)  member: int member number 0..M-1 (the number after m)
    remove an installed member: it then carries no stiffness and no mass (offered only where the truss stays rigid).
Set a member's material: SET_PARAM(member, mat, code)  member: int member number 0..M-1 (the number after m); code: 0 = Al6061T6, 1 = S355, 2 = Ti6Al4V; Al6061T6 = 6061-T6 aluminium (E 68.9 GPa, yield 276 MPa, 2700 kg/m3), S355 = S355 structural steel (E 210 GPa, yield 355 MPa, 7850 kg/m3), Ti6Al4V = Ti-6Al-4V titanium (E 113.8 GPa, yield 880 MPa, 4430 kg/m3); the member rows abbreviate them Al / St / Ti.
    set a member's material; its radius is kept.
Resize a member's tube: SET_PARAM(member, r, radius)  member: int member number 0..M-1 (the number after m); radius: float [m] 0.004..0.12, the new outer tube radius (a value outside is set to the bound)
    resize a member's tube.
Move a joint along x: SET_PARAM(joint, dx, offset)  joint: int joint number (the number after J); offset: float [m] the joint's new total offset from its initial position along that axis, within the +- bound the observation states
    move a free joint along one axis; every member at it follows. A loaded joint's coordinate along the structure's axis is locked: such a move changes nothing.
Move a joint along y: SET_PARAM(joint, dy, offset)  joint: int joint number (the number after J); offset: float [m] the joint's new total offset from its initial position along that axis, within the +- bound the observation states
    move a free joint along one axis; every member at it follows. A loaded joint's coordinate along the structure's axis is locked: such a move changes nothing.
Move a joint along z: SET_PARAM(joint, dz, offset)  joint: int joint number (the number after J); offset: float [m] the joint's new total offset from its initial position along that axis, within the +- bound the observation states
    move a free joint along one axis; every member at it follows. A loaded joint's coordinate along the structure's axis is locked: such a move changes nothing.
```

The same turn rendered under both (the prompt differs in these lines only, plus the CHANGES
paragraph of the observation and the system prompt's action section):

```
multitool:      2. SET_RADIUS(13, 0.01621) (member 13 (nodes 1-6, bay-1 diagonal on leeward x+ face); was 0.01474)
                SET_MATERIAL(22, S355)  (member 22 (nodes 4-8, bay-2 leg at corner x-y-); material Al6061T6 -> S355)
                End with: <action>SET_MATERIAL(22, S355)</action>
set_parameter:  2. SET_PARAM(13, r, 0.01621) (member 13 (nodes 1-6, bay-1 diagonal on leeward x+ face); was 0.01474)
                SET_PARAM(22, mat, 1)  (member 22 (nodes 4-8, bay-2 leg at corner x-y-); material 0 -> 1)
                End with: <action>SET_PARAM(22, mat, 1)</action>
```

## The hooks (`dataxp/domain.py`)

`format_move` / `parse_move` / `grammar` dispatch, under `set_parameter` and for a domain that
speaks verbs, to three methods a domain may override (never the three base methods):

| hook | default for a verb-surface domain | default for a native domain |
|---|---|---|
| `set_param_format(move) -> str` | the classic `SET_PARAM(i, p, v)` line (value as stored, `fmt_arg`: 6 significant digits, integral values bare) | `native_set_param_format`: generic encoding from `verbs` |
| `set_param_parse(verb, toks) -> Move or None` | `parse_classic_set_param` (SET_PARAM calls only) | `native_set_param_parse` |
| `set_param_grammar() -> str` | `surface_set_param_grammar` from `verb_table` | `native_set_param_grammar` from `verbs` |

`format_value(p, v)` prints the stored code / number under `set_parameter` ('' for a toggle,
as before). Table fields: `ParamVerb.phrase`, `Toggle.off_phrase`, `VerbSpec.phrase`.

### Native domains: the generic encoding and what the next agent must do

`SET_PARAM(element, operation, value)`: element = the verb's first `int` argument (`0` when it
has none, `DO_NOTHING`); operation = the verb name in lower case (parsed ignoring case); value =
the one remaining argument as `fmt_arg` writes it (`hydroxyl`, `20`, `["line3_ex","gen1"]` for a
`json` argument), a JSON list of the remaining arguments in table order when there are several
(`[5, 20]`), `0` when there are none. Parse is strict: element must be 0 for a verb without one,
value must be 0 for a verb without further arguments, a list must have exactly the remaining
arguments, every item is coerced by its `ArgSpec.type`. The list may be written as JSON
(`["line3_ex", 2]`, what `format_move` writes) or as the block head literally shows it, with bare
identifiers (`[line3_ex, 2]`; single quotes pass too): the inside is then split on its top-level
commas and each item coerced by its type, exactly as a multitool call's arguments are
(`_parse_list_value`; added 2026-10-23 after the verifier found that `SET_PARAM(4, set_bus,
[line3_ex, 2])` was rejected while `SET_BUS(4, line3_ex, 2)` parsed, SET_BUS being 31% of
grid2op's legal moves). The bare form is a parse-side alias only, so there is still one canonical
text per move. The grammar is one block per `VerbSpec` (`<phrase>: SET_PARAM(<elem>,
<verb_lower>, <value>)  <argument descriptions>[; value: JSON list [a, b] in this order, a in
double quotes]` + doc), with the preamble noting that `parameter` names the operation and that a
value of several arguments is a JSON list in the listed order with names in double quotes
(`SET_PARAM_NATIVE_PREAMBLE`; the per-block note `_native_list_note` names which members are
names). Examples:

```
SET_PARAM(12, add_fragment, hydroxyl)        <-> ADD_FRAGMENT(12, hydroxyl)
SET_PARAM(3, redispatch, [5, 20])            <-> REDISPATCH(3, 5, 20)
SET_PARAM(4, split_sub, ["line3_ex","gen1"]) <-> SPLIT_SUB(4, ["line3_ex","gen1"])
SET_PARAM(0, do_nothing, 0)                  <-> DO_NOTHING()
SET_PARAM(3, reconnect_line, 0)              <-> RECONNECT_LINE(3)
```

How a native domain declares its spelling (the four do it this way; a fifth follows the same
hooks):

1. Fill the `verbs` table completely (every verb `legal_moves` can return) and leave
   `format_move` / `parse_move` to the defaults: the generic encoding then comes for free. All
   four native domains are in this situation; none overrides a `set_param_*` hook.
2. A `grammar()` override must wrap `Domain.grammar(self)` (molopt and engibench do), so the
   set_parameter grammar is what gets wrapped, and its own prose must name the operations the
   way the process's spelling does: `Domain.op_name(verb)` (`ADD_RING_BOND` under multitool,
   `add_ring_bond` under set_parameter) and `Domain.call_form(verb)` (`CURTAIL(g, frac)` /
   `SET_PARAM(g, curtail, frac)`, the head of that operation's grammar block). molopt's trailer
   now reads "add_ring_bond closes rings of size 5, 6, 7 only; break_bond takes ... (cutting a
   chain bond is remove_fragment)" and says "element (the value of replace_atom; not the
   SET_PARAM element slot, which is the atom) is one of: ..."; engibench's head says "An
   operation must change at least one cell".
3. The same two helpers re-spell the observation: every `describe()` sentence that shows a
   call signature or names an operation goes through `call_form` / `op_name` (grid2op: "set_bus
   refuses them, a split_sub list may not name them", "Redispatch: SET_PARAM(g_from,
   redispatch, [g_to, mw]) lowers unit g_from ...", "a curtail or set_storage order", "no
   operation puts it back earlier"; photonics: "Changes are made with these operations (pixel
   units): SET_PARAM(x0, fill_rect, [y0, x1, y1]) deposits ..."; molopt: "Operations:
   SET_PARAM(atom, add_fragment, fragment) bonds ..."; engibench: "Operations: SET_PARAM(x0,
   fill_rect, [y0, x1, y1]) / ...", "dilate thickens a member ..., erode thins it"). Under
   multitool both helpers return the verb text, so the multitool observation is byte for byte
   what it was. The family words FILL / CLEAR / SHIFT of engibench's physics sentences are not
   verb names and stay. A verb named inside a table doc ("the line comes back only through
   RECONNECT_LINE") prints as its operation name in the generated grammar (`_name_ops`).
4. Override `set_param_format` / `set_param_parse` / `set_param_grammar` only for what the table
   cannot express; keep the bijection on the legal moves; record every judgment call in the
   module docstring section "Set-parameter spelling" (the four modules have one). A native
   domain that overrides `format_move` or `parse_move` without the matching hook fails at
   construction under `set_parameter` (`check_native_set_param`); one that overrides them WITH
   the hook must dispatch to the hook itself on its first line (`if current_spelling() ==
   "set_parameter": return self.set_param_format(move)`), because the override bypasses the
   base dispatch.
5. Set `VerbSpec.phrase` on every verb (grid2op "Reconnect a line", "Move an element end to a
   busbar", "Redispatch between two units", "Set a storage unit's power"; photonics "Fill a
   rectangle", "Shift an island"; molopt "Attach a fragment", "Close a ring" / "Open a ring";
   engibench "Fill a disk", "Erode a component"); run the round trip
   (`corpus/spelling/roundtrip_native.py`: parse_move(format_move(m)).matches(m) on every legal
   move of >= 50 states, the verb text rejected under set_parameter, the SET_PARAM text rejected
   under multitool), smoke, the prompt snapshot (`prompt_snapshot.py`), render and ownrows dry
   runs under both spellings, as below.

The four bijections as built (every move of the table; the element is the verb's first int
argument, the owner's rule, so a rectangle's element is its left column and a disk's its centre
column):

| domain | multitool | set_parameter |
|---|---|---|
| grid2op | `DO_NOTHING()`, `RECONNECT_LINE(3)`, `SET_BUS(4, line3_ex, 2)`, `SPLIT_SUB(4, ["line3_ex","gen1"])`, `MERGE_SUB(4)`, `REDISPATCH(3, 5, 20)`, `CURTAIL(2, 0.5)`, `SET_STORAGE(0, -3.5)` | `SET_PARAM(0, do_nothing, 0)`, `SET_PARAM(3, reconnect_line, 0)`, `SET_PARAM(4, set_bus, ["line3_ex", 2])`, `SET_PARAM(4, split_sub, ["line3_ex","gen1"])`, `SET_PARAM(4, merge_sub, 0)`, `SET_PARAM(3, redispatch, [5, 20])`, `SET_PARAM(2, curtail, 0.5)`, `SET_PARAM(0, set_storage, -3.5)` |
| photonics | `FILL_RECT(12, 5, 20, 9)`, `FILL_DISK(20, 20, 3)`, `SHIFT_FEATURE(2, 1, -1)`, `DILATE(2, 1)` | `SET_PARAM(12, fill_rect, [5, 20, 9])`, `SET_PARAM(20, fill_disk, [20, 3])`, `SET_PARAM(2, shift_feature, [1, -1])`, `SET_PARAM(2, dilate, 1)` |
| molopt | `ADD_FRAGMENT(12, hydroxyl)`, `REMOVE_FRAGMENT(2)`, `SWAP_FRAGMENT(2, methoxy)`, `REPLACE_ATOM(7, N)`, `ADD_RING_BOND(3, 9)`, `BREAK_BOND(3, 4)` | `SET_PARAM(12, add_fragment, hydroxyl)`, `SET_PARAM(2, remove_fragment, 0)`, `SET_PARAM(2, swap_fragment, methoxy)`, `SET_PARAM(7, replace_atom, N)`, `SET_PARAM(3, add_ring_bond, 9)`, `SET_PARAM(3, break_bond, 4)` |
| engibench | `FILL_RECT(10, 4, 18, 5)`, `CLEAR_DISK(30, 12, 3)`, `SHIFT_FEATURE(3, 0, -3)`, `ERODE(0, 1)` | `SET_PARAM(10, fill_rect, [4, 18, 5])`, `SET_PARAM(30, clear_disk, [12, 3])`, `SET_PARAM(3, shift_feature, [0, -3])`, `SET_PARAM(0, erode, 1)` (a disk radius is a float: `2.0` prints as `2`, parses back as `2.0`) |

Judgment calls of the four (each recorded in its module docstring): DO_NOTHING's element 0 and
value 0 are placeholders required exactly; REDISPATCH's element is g_from (the unit lowered),
SET_BUS's element the substation with the element NAME quoted first in the value list; names
(fragments, element symbols, grid2op element names in a JSON list) are written as the multitool
call writes them; molopt's REPLACE_ATOM keeps its stored argument key `element`, which collides
with the tool's slot word only in the grammar text (the trailer disambiguates it); nothing in
`legal_moves`, `apply`, Phi or the stored `{verb, args}` dicts changed.

Status: the generic encoding is unit-tested on a synthetic `VerbSpec` table
(`corpus/spelling/native_generic_test.py`: no-argument, int, int+str, int+json, int+int+float,
str-first and float-only verbs; construction checks) and every per-domain check of the native
four passed (table below, rows (e)-(h)). The set_parameter grammars of the four are in
`corpus/spelling/grammar_set_parameter_<domain>.txt` (9 / 7 / 6 / 7 operation blocks); the
multitool ones in `pre_grammar_multitool_<domain>.txt` = `grammar_multitool_<domain>.txt`.

## Checks run (2026-10-23, Windows, PYTHONHASHSEED=0, MKL/OMP threads 1, one worker)

Scratch: `corpus/spelling/` (pre-edit package snapshot in `pre_pkg/`, the check scripts, every
log). "pre" = the package before this change, "post" = after it under multitool, "sp" = after it
under `DATAXP_SPELLING=set_parameter`.

| check | result |
|---|---|
| (a) `smoke frame3d2 circuit2 grid2 traffic2 thermal2 seir2 inventory2`, DATAXP_SPELLING unset / `=multitool` / `=set_parameter` | PASS 7/7 in each of the three |
| (a) `grammar()` of the seven under multitool vs the pre-edit grammars | byte-identical, 7/7 |
| (a) `render_cot --prompts-only` on frame3d2 (20 turns) and thermal2 (11 turns), quick corpora, pre vs post | byte-identical, both files |
| (a) `probe_info prepare` (3 states) and `gold_probe prepare` (3 + 3 states) with the pre-edit package vs post | `obs_*.txt` byte-identical; `states.json` / `states.jsonl` equal apart from the new `spelling` key |
| (b) round trip, 20 problems x 3 states x all legal moves per domain, under set_parameter | 420 states, 37,444 moves, all `parse_move(format_move(m)).matches(m)`: exact for six domains; grid2 2,376 exact + 3,000 within 1e-5 (DISPATCH prints 6 significant digits of a longer stored value, documented in grid2's docstring) and 3 same-text pairs equal to printing precision — the same counts under multitool, so nothing is spelling-specific; every text is `SET_PARAM(i, p, v)`, the verb text of the same move and a category name in the value slot are rejected, `format_value` prints the stored code / '' for a toggle |
| (b) the same round trip under multitool | identical counts; no text contains SET_PARAM, the SET_PARAM text is rejected |
| (b) grammars of the seven | `grammar_set_parameter_<domain>.txt` (7, 11, 6, 5, 5, 8, 6 operation blocks) and `grammar_multitool_<domain>.txt` in `corpus/spelling/` |
| (c) quick generation of frame3d2 (and thermal2), pre vs post (multitool) vs sp (set_parameter) | every `*.jsonl` byte-identical, `report.json` equal ignoring timing, all three pairings (`compare_runs.py`) |
| (d) ownrows dry run on the quick frame3d2 corpus under set_parameter (`--per-domain 8 --dev-per-domain 4`, K=4, teacher share 0.25) | 13 states (all `spelling: set_parameter`, instruction = `INSTRUCTION_SET_PARAM` + SET_PARAM grammar, sha1 `f99b61d776f1` at the time, `fe8fa8db0a90` since the VERB sentence was reworded; no verb call in any served text), 52 replies all parsed and legal, 20 tier rows (10 states x 2 arms), every assistant line `<action>SET_PARAM(i, p, v)</action>` parsing back to its stored action, every `action` / `gold_action` `{i, p, v, v_old}`, every row carries `spelling` (`verify_ownrows_sp.py`) |
| instruction sha1s | multitool `1c344388a444`, set_parameter `fe8fa8db0a90` (was `f99b61d776f1` before the VERB sentence was reworded, 2026-10-23); the texts differ in the action line and in the sentence that explains it |
| (e) native four: `smoke molopt grid2op engibench photonics` under multitool / `=set_parameter` | PASS 4/4 in each |
| (e) `grammar()` of the four under multitool vs the pre-edit grammars (`pre_grammar_multitool_<domain>.txt`, written from a frozen copy of the pre-edit package, `pre_pkg2/`) | byte-identical, 4/4; the set_parameter grammars `grammar_set_parameter_<domain>.txt` have 9 / 7 / 6 / 7 operation blocks (= verbs), one per `VerbSpec`, and name no verb call; the seven v2 grammars unchanged under both spellings |
| (f) round trip under set_parameter (`roundtrip_native.py`: problems x start / +1 / +2 random legal moves x ALL legal moves): molopt 51 states x 2,751 moves, grid2op 66 x 4,961 (26 problems; an episode that ends early stops at fewer than 3 states), engibench 51 x 3,038, photonics 21 x 672 | every parse_move(format_move(m)).matches(m) exactly (no 6-digit tolerance needed); every text `SET_PARAM(<first int arg>, <verb_lower>, <value or [list] or 0>)` with the parsed args rebuilt in table order; the verb text of the same move rejected; one text per move |
| (f) the same round trip under multitool | identical counts; every text `VERB(args)`, no SET_PARAM, the SET_PARAM encoding of the same move rejected |
| (g) prompt snapshots (`prompt_snapshot.py`: render_cot system prompt, ownrows instruction and own-system prompt, grammar, policy_observation, build_prompt tail, every legal move with its move_label; 2-4 problems x 2 depths per domain) and `render_cot --prompts-only` on the quick corpora (molopt 12 turns, grid2op 4, engibench 6, photonics 2), pre-edit package vs post under multitool | byte-identical, 4/4 snapshots (`snap_pre/` = `snap_post/`) and 4/4 renders (`render_pre_native/` = `render_post_native/`); under set_parameter (`snap_sp/`, `render_sp_native/`) every action, history and change line is the generic encoding and no verb name or verb call appears in any served text (0 lines per domain) |
| (c) quick generation of molopt and grid2op, pre vs post (multitool) vs sp (set_parameter), PYTHONHASHSEED=0, one worker | every `*.jsonl` byte-identical, `report.json` equal ignoring timing, all three pairings (`compare_runs.py`) |
| (h) ownrows dry run on the quick molopt corpus under set_parameter (`--per-domain 8 --dev-per-domain 4`, K=4, teacher share 0.25; `verify_ownrows_native_sp.py`) | 12 states (all `spelling: set_parameter`, instruction `f99b61d776f1` at the time (`fe8fa8db0a90` since the VERB sentence was reworded) + the SET_PARAM grammar, no verb call in any served text, `gold_text` = the generic encoding of the gold `{verb, args}` move), 48 replies all parsed and legal (add_fragment 28, break_bond 5, add_ring_bond 5, remove_fragment 4, replace_atom 4, swap_fragment 2), 20 tier rows (10 states x 2 arms), every assistant line `<action>SET_PARAM(element, operation, value)</action>` parsing back to its stored `{verb, args}` action, every row carries `spelling` |

## Storage on bridges2

Per the owner's decision, the two datasets live apart:
`/ocean/projects/mch250030p/shared/dataxp/datav4_multitool/` and
`/ocean/projects/mch250030p/shared/dataxp/datav4_set_parameter/`, each with its corpora, rows,
grammar text (`grammar_<spelling>_<domain>.txt`), instruction (sha1 above), manifest and README.
The corpora are identical between the two (generation is spelling-blind); the rows, grammars and
instruction are not, and every row says which spelling it was served in. Nothing has been uploaded
from this change yet (local work only).

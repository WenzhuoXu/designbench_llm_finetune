# dataxp Move contract — implementer guide

One page for whoever adds a native-verb domain (grid2op, photonics, molopt,
engibench, ...) to `dataxp`. Scope decisions are in
`docs/corpus_native_domains.md`; this file is the code contract.

Verb surfaces over the v2 `SET_PARAM` domains (the owner's decision of 2026-10-23: the move
set, `apply`, Φ and the stored `{i, p, v, v_old}` dicts stay, only the spelling changes) are a
separate layer on this contract: `docs/dataxp_verb_surface.md` (class flag `verb_surface`,
`verb_table` of `ParamVerb` rows with `Toggle` / `Categorical` / `Continuous` codecs; first
example `frame3d2`). Every prompt-facing path tests `speaks_verbs(domain)` =
`native_moves or verb_surface`.

## What a Move is

```python
from dataxp.domain import Move, SET_PARAM, ArgSpec, VerbSpec, Domain

Move(verb: str, args: dict)          # one action in the field's own language
```

* `move.as_dict()` / `Move.from_dict(d)` — the serialised form used by the
  ladder's `actions`, the harvested `pairs.jsonl` / `trajectories.jsonl`, and
  replay. `SET_PARAM` moves serialise to **exactly** the old Candidate dict
  `{"i", "p", "v", "v_old"}` (same keys, same order), so every run1/run2 file is
  byte-identical. Any other verb serialises as `{"verb": ..., "args": {...}}`.
  `from_dict` reads both forms.
* `move.matches(other, rel=1e-9)` — same verb and arguments (numbers within
  `rel`, `v_old` ignored). The renderer's `wrong_action` check and the probes
  use it.
* `Move.from_candidate(c)` / `move.to_candidate()` — the Candidate adapter.
  `Candidate(i, p, v, v_old)` **is** `Move("SET_PARAM", {i, p, v, v_old})`.

## What a domain must provide

Everything in the existing contract still holds (`make_problem` with infeasible
starts and `minimum_`/`maximum_` goals, `load`/`clone`/`evaluate(st, seed)`
deterministic in `(st, seed)`, `program(spec)` with validity bounds,
`reference(spec)`, `describe(st, spec)`). Φ scoring, the gates and the
evaluator are state-based and do not see moves at all.

The move extension, with the exact signatures (`Domain` in `dataxp/domain.py`):

```python
class MyDomain(Domain):
    native_moves = True                       # class flag: this domain speaks its own verbs
    verbs: Tuple[VerbSpec, ...] = (...)       # optional table (see below)

    def legal_moves(self, st, rng: np.random.Generator,
                    max_candidates: Optional[int] = None) -> List[Move]:
        """Legal move set at st. Deterministic in (st, rng). Honour
        max_candidates (phi_d2 asks for 24): Domain.subsample_moves(moves, rng, k)
        does the same stable subsample the v1 path uses."""

    def apply(self, st, move: Move) -> None:
        """Apply IN PLACE. Raise ValueError for a verb or argument the state
        does not admit (the ladder never applies an illegal move; the probes
        use the exception as the legality test)."""

    def format_move(self, move: Move) -> str:
        """'VERB(a, b, ...)' - the text that goes inside <action>...</action>."""

    def parse_move(self, text: str) -> Optional[Move]:
        """Inverse of format_move on free text; None when no legal verb call
        is present. Must round-trip: parse_move(format_move(m)).matches(m)."""

    def grammar(self) -> str:
        """The verb list with argument types and ranges, written for the
        system prompt (render_cot uses it verbatim when native_moves is True)."""

    # optional
    def hint_move(self, st) -> Optional[Move]:
        """The depth-0 greedy move (the ladder's 'greedy' policy). May also
        return a Candidate. None -> the greedy policy falls back to hint()
        then to a random legal move."""

    def move_label(self, st, move: Move) -> str:
        """Short role label for a move, printed after it in the observation's
        history / 'change you are about to make' lines (as element_label is
        for SET_PARAM). Optional; '' when absent."""

    # time-stepping domains only (docs/corpus_native_domains.md, "Time-stepping
    # domains"): every move advances simulated time
    advances_time = True                       # default False
    time_key = "step"                          # the metric carrying simulated time

    def null_move(self, st) -> Move:
        """The move that lets the step pass with no action (grid2op:
        DO_NOTHING()). The ladder applies it when no candidate improves on
        the incumbent; harvest drops exact duplicate pairs by
        (problem, time_key, chosen, rejected) and reports the count."""
```

Defaults in `Domain` (what the v1/v2 domains get without changing a line):

| method | default |
|---|---|
| `legal_moves(st, rng, max_candidates=None)` | `enumerate_candidates(...)` wrapped as `SET_PARAM` moves — same rng consumption, same order |
| `apply(st, move)` | `SET_PARAM` -> `self.set(st, i, p, v)` with the values exactly as the Candidate held them; any other verb -> `ValueError` |
| `format_move(move)` | `SET_PARAM(i, p, v)` (v to 6 significant digits); other verbs: `VERB(args in verb-table order, else insertion order)` |
| `parse_move(text)` | first balanced `VERB(...)` call whose verb is `SET_PARAM` (non-native domains) or in `self.verbs`; arguments coerced by `ArgSpec.type` |
| `grammar()` | `self.verbs` rendered one verb per line; without a table, the `SET_PARAM` description built from `self.params` |

The `verbs` table makes `format_move` / `parse_move` / `grammar` free:

```python
verbs = (
    VerbSpec("RECONNECT_LINE", (ArgSpec("l", "int", "line id 0..L-1"),),
             "put a disconnected line back in service"),
    VerbSpec("REDISPATCH", (ArgSpec("g_from", "int", "generator 0..G-1"),
                            ArgSpec("g_to", "int", "generator 0..G-1"),
                            ArgSpec("mw", "float", "MW moved, 0 < mw <= ramp")),
             "shift mw of output from g_from to g_to"),
)
```

`ArgSpec.type` is one of `int`, `float`, `str`, `bool`, `json` (a JSON literal, for
lists such as `[1, 2]`); `fmt_arg` / `coerce_arg` / `parse_verb_calls` in
`dataxp/domain.py` are the pieces a hand-written override can reuse.

Rules that do not bend (owner, `docs/corpus_native_domains.md`):

* every verb is the field's own operation with typed arguments. A verb that is
  "set parameter p of element e" under another name is not a native verb; a
  problem that only admits parameter setting is excluded and the exclusion is
  recorded in the domain's module docstring and the smoke report;
* `legal_moves` must enumerate the whole vocabulary the domain claims (every verb
  reachable from some state), not a sizing subset;
* `describe(st, spec)` is written from the source paper / environment docs and
  says what each verb does physically — it is the text the corpus teaches.

## Minimal native-verb example

```python
"""Toy: a 1-D heat strip with N cells; a hot cell is cooled by placing or
removing coolers. Verbs: PLACE_COOLER(cell), REMOVE_COOLER(cell), MOVE_COOLER(src, dst).
"""
from typing import List, Optional
import numpy as np
from dataxp.domain import ArgSpec, Domain, Move, VerbSpec
from dataxp.potential import program_from_goals


class StripDomain(Domain):
    name = "strip"
    native_moves = True
    verbs = (
        VerbSpec("PLACE_COOLER", (ArgSpec("cell", "int", "empty cell 0..N-1"),), "install a cooler"),
        VerbSpec("REMOVE_COOLER", (ArgSpec("cell", "int", "cell holding a cooler"),), "remove it"),
        VerbSpec("MOVE_COOLER", (ArgSpec("src", "int", "cell holding a cooler"),
                                 ArgSpec("dst", "int", "empty cell")), "relocate a cooler"),
    )

    def make_problem(self, index, rng):
        n = int(rng.integers(8, 16))
        return {"family": "strip", "n": n, "power": rng.uniform(1, 3, n).tolist(),
                "coolers0": [], "goals": {"maximum_peak_temp": 2.0}}

    def load(self, spec):
        return {"n": spec["n"], "P": np.asarray(spec["power"]), "cool": set(spec["coolers0"])}

    def clone(self, st):
        return {"n": st["n"], "P": st["P"], "cool": set(st["cool"])}

    def evaluate(self, st, seed=0):
        self.calls += 1
        T = st["P"] * np.array([0.25 if i in st["cool"] else 1.0 for i in range(st["n"])])
        return {"peak_temp": float(T.max()), "cost": float(len(st["cool"]))}

    def program(self, spec):
        prog = program_from_goals(spec["goals"], objective_key="cost", objective_sense="min",
                                  domain=self.name)
        from dataclasses import replace
        return replace(prog, validity_bounds=(("peak_temp", 100.0),))

    # -- moves --------------------------------------------------------------
    def legal_moves(self, st, rng, max_candidates=None) -> List[Move]:
        empty = [i for i in range(st["n"]) if i not in st["cool"]]
        out = [Move("PLACE_COOLER", {"cell": i}) for i in empty]
        out += [Move("REMOVE_COOLER", {"cell": i}) for i in sorted(st["cool"])]
        out += [Move("MOVE_COOLER", {"src": s, "dst": d}) for s in sorted(st["cool"]) for d in empty]
        return self.subsample_moves(out, rng, max_candidates)

    def apply(self, st, move: Move) -> None:
        a = move.args
        if move.verb == "PLACE_COOLER" and a["cell"] not in st["cool"]:
            st["cool"].add(int(a["cell"]))
        elif move.verb == "REMOVE_COOLER" and a["cell"] in st["cool"]:
            st["cool"].remove(int(a["cell"]))
        elif move.verb == "MOVE_COOLER" and a["src"] in st["cool"] and a["dst"] not in st["cool"]:
            st["cool"].remove(int(a["src"])); st["cool"].add(int(a["dst"]))
        else:
            raise ValueError("illegal move %s" % self.format_move(move))

    # format_move / parse_move / grammar come from the verbs table

    def describe(self, st, spec) -> str:
        return "PROBLEM\n... what the strip is, what peak_temp means, what each verb does ...\n" \
               "DESIGN\ncoolers at cells %s\nRESULTS\n..." % sorted(st["cool"])
```

Register it (lazily) in `dataxp/domains/__init__.py` as
`"strip": ("dataxp.domains.strip", "StripDomain", False)`.

## How the pipeline uses the methods

* `ladder.run_episode` — `legal_moves` for random/blind/phi_select/phi_d2,
  `hint_move`/`hint` for greedy, `apply` on a clone for every scored child,
  `move.as_dict()` into `actions`, `TurnRecord.chosen`, `top`, `spread`.
* `harvest` — pairs compare the `action` dicts; `replay_trajectory` and the
  envelope red-team re-apply `Move.from_dict(a)` with `apply`.
* `render_cot` — when `native_moves` is True the system prompt's action-format
  section is replaced by `grammar()`, the "change you are about to make" and
  history lines print `format_move`, and the `<action>...</action>` line is
  checked with `parse_move(...).matches(gold, rel=1e-2)`. Old variants render
  byte-for-byte as before.
* `smoke` — for every domain: `legal_moves` non-empty, `grammar()` non-empty,
  `parse_move(format_move(m))` round-trips, `apply` changes a metric, and (for
  non-native domains) the Move path equals the Candidate path dict-for-dict.

## Commands

```bash
# contract + end-to-end smoke (one or more variants; `all` runs the registry)
python -m dataxp.smoke strip
python -m dataxp.smoke all

# quick end-to-end generation (6 problems, 8 turns) - run before any full job
python -m dataxp.run_generation --variant strip --quick --workers 1 --seed 0 \
    --tmp <scratch>/dataxp_tmp --out <scratch>/dataxp_out

# bit-for-bit regression of the v1/v2 path after touching dataxp/*.py:
# run the quick generation into pre_*/post_* and diff every *.jsonl and
# report.json (only wall_seconds may differ)
```

Simulators that do not run on Windows are developed and smoke-tested on drc
(CPU) under `/storage/wxu/corpus/<domain>/` through the slurm MCP; nothing runs
on Orchard.

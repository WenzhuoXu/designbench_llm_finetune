"""Step 2: cache every legal move and its Phi at each selected state.

Port of up_tools/cand_phi.py generalised to Move: the legal set is
domain.legal_moves(st, rng) with the rng gold_probe.prepare used
(stable_seed(pid, 'gp', k)); every move is applied on a clone and scored with
PhiEvaluator(domain, spec, offset_rule) - the evaluator the ladder scores
candidates with - at seed stable_seed(pid, 'gp', i, p) for SET_PARAM and
stable_seed(pid, 'gp', format_move) for a native verb (deterministic domains
ignore the seed). The teacher's move is scored the same way (gold_phi) and
its percentile among the legal set recorded.

Append-only and keyed by sid, so a killed run resumes. CPU only.

    python -m dataxp.ownrows.cand_phi --states <dir>/states.jsonl \
        --out <dir>/states_phi.jsonl [--workers 1] [--offset-rule infeasible]
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..domain import Move
from ..domains import get_domain
from .common import evaluator, legal_rng, move_seed, percentile, rebuild_state

_OFFSET_RULE = {"rule": "infeasible"}


def one(line: str) -> str:
    row = json.loads(line)
    d = get_domain(row["variant"])
    spec, pid = row["spec"], row["problem_id"]
    st = rebuild_state(d, spec, row["history"])
    ev = evaluator(d, spec, _OFFSET_RULE["rule"])
    rng = legal_rng(pid, row["k"])
    moves, n_illegal, n_nan = [], 0, 0
    for m in d.legal_moves(st, rng):
        ch = d.clone(st)
        try:
            d.apply(ch, m)
        except Exception:               # an enumerated move the state refuses
            n_illegal += 1
            continue
        phi = float(ev.score(ch, seed0=move_seed(pid, d, m)).mean)
        if not math.isfinite(phi):
            n_nan += 1
        moves.append({"action": m.as_dict(), "text": d.format_move(m), "phi": phi})
    gold = Move.from_dict(row["gold"])
    ch = d.clone(st)
    d.apply(ch, gold)
    gold_phi = float(ev.score(ch, seed0=move_seed(pid, d, gold)).mean)
    cand = np.array([m["phi"] for m in moves], float)
    row["moves"] = moves
    row["gold_phi"] = gold_phi
    row["gold_pct"] = percentile(cand, gold_phi)
    row["gold_in_legal"] = any(Move.from_dict(m["action"]).matches(gold, rel=1e-9) for m in moves)
    row["n_candidates"] = len(moves)
    row["n_illegal_enumerated"] = n_illegal
    row["n_nan_phi"] = n_nan
    row["offset_rule"] = _OFFSET_RULE["rule"]
    return json.dumps(row)


def _init(rule: str):
    _OFFSET_RULE["rule"] = rule


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--offset-rule", default="infeasible", choices=("infeasible", "raw", "legacy_scaled"))
    a = ap.parse_args(argv)
    _init(a.offset_rule)

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            for l in fh:
                try:
                    done.add(json.loads(l)["sid"])
                except (ValueError, KeyError):
                    pass
    lines = []
    with open(a.states, encoding="utf-8") as fh:
        for l in fh:
            if l.strip() and json.loads(l)["sid"] not in done:
                lines.append(l)
    print("states %d, already cached %d, to score %d" % (len(lines) + len(done), len(done), len(lines)),
          flush=True)

    per = {}
    n = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        if a.workers > 1:
            ex = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.offset_rule,))
            it = ex.map(one, lines, chunksize=1)
        else:
            ex, it = None, map(one, lines)
        try:
            for s in it:
                fh.write(s + "\n")
                fh.flush()
                r = json.loads(s)
                per.setdefault(r["variant"], []).append(
                    (r["n_candidates"], r["gold_pct"], r["gold_in_legal"], r["n_illegal_enumerated"]))
                n += 1
                if n % 25 == 0:
                    print("%d/%d" % (n, len(lines)), flush=True)
        finally:
            if ex is not None:
                ex.shutdown()
    for v in sorted(per):
        xs = per[v]
        pcts = [x[1] for x in xs if x[1] is not None]
        print("%-10s states %4d  mean legal moves %6.1f  teacher percentile %5.1f  teacher in legal set %.2f"
              "  illegal enumerated %d" % (
                  v, len(xs), float(np.mean([x[0] for x in xs])),
                  float(np.mean(pcts)) if pcts else float("nan"),
                  float(np.mean([x[2] for x in xs])), sum(x[3] for x in xs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

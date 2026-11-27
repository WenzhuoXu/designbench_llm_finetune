"""con1 facetc truth: per-element internal state before and after each move, from the simulator's own observation.

    python chain_truth.py --code C --harness V3RC --problems P [P ...] --states S [S ...] --rows rows.jsonl \
        --queries queries.jsonl --out chain.jsonl [--workers 24]

For every state of the rows (training) and of the held-out-problem questions (test) in CHAIN_FAMILIES: rebuild the
state (conseq_lib.rebuild, the harness's bookkeeping), apply each move as a check does (native: open_apply; verb
surface: check_move + set), render the child's observation with observation_native exactly as the next turn would
show it, and parse the per-element state of both observations (targets.parse_elements). Checks per state: the rebuilt
prompt equals the stored one, and (test) the rebuilt check result equals the stored message for the first 2 moves.
One line per state: {sid, variant, split, before, after: {move: elements | null (refused)}, checks}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from targets import CHAIN_FAMILIES, parse_elements  # noqa: E402

_ENV = None


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def after_obs(s, mv):
    """The child's observation text after `mv` at state s (None when the simulator refuses the move)."""
    H, D, d = s.env.H, s.env.D, s.domain
    if s.native:
        child = d.clone(s.st)
        try:
            H.open_apply(d, s.variant, child, mv, s.opening)
        except Exception:                                  # noqa: BLE001
            return None
        ad = mv.as_dict()
        label = s.env.R._label_for(d, child, ad)
    else:
        child, applied, why = s._verb_surface_apply(mv)
        if child is None:
            return None
        ad, label = applied, d.element_label(child, applied["i"])
    sc = s.ev.score(child, seed0=D.stable_seed(s.pid, "eval", "s", s.t + 1))
    cm = {k: float(v) for k, v in sc.agg_metrics.items()}
    h2 = list(s.history) + [(label, ad)]
    _, u2 = H.observation_native(s.env.C, d, s.spec, child, cm, s.env.L._binding_constraint(s.ev.program, cm), h2,
                                 s.metrics, s.variant, s.native_prompt)
    return u2


def work(job):
    code, harness, st, spec, moves, results = job
    global _ENV
    if _ENV is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        _ENV = Env(code, harness)
    from conseq_lib import rebuild
    try:
        s = rebuild(_ENV, {"variant": st["variant"], "t": st["t"], "turn_log": st["turn_log"]}, spec)
        checks = {"prompt_same": s.prompt()[1] == st["user"], "result_same": 0, "result_checked": 0}
        out = {}
        for j, text in enumerate(moves):
            mv = s.parse(text)
            if mv is None:
                out[text] = None
                continue
            u2 = after_obs(s, mv)
            out[text] = parse_elements(st["variant"], u2) if u2 is not None else None
            if results is not None and j < 2:
                checks["result_checked"] += 1
                checks["result_same"] += s.result_of(mv)["result"] == results[j]
        return {"sid": st["sid"], "variant": st["variant"], "split": st["split"],
                "before": parse_elements(st["variant"], st["user"]), "after": out, "checks": checks}, None
    except Exception as e:                                 # noqa: BLE001
        return {"sid": st["sid"], "variant": st["variant"]}, "%s: %s" % (type(e).__name__, str(e)[:300])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--groups", nargs="*", default=["heldout_problem"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    specs = {}
    for p in a.problems:
        for line in opener(p):
            r = json.loads(line)
            sp = r.get("spec", r)
            specs[sp["problem_id"]] = sp
    want = {}                                              # training sid -> moves of its row
    for line in opener(a.rows):
        r = json.loads(line)
        if r.get("tier") == "con1_conseq" and r["domain"] in CHAIN_FAMILIES:
            want[r["sid"]] = list(r["con1"]["moves"])
    jobs, seen = [], set()
    for p in a.states:
        try:
            for line in opener(p):
                st = json.loads(line)
                if st["sid"] in want and st["sid"] not in seen:
                    seen.add(st["sid"])
                    st = dict(st, split="train")
                    jobs.append((a.code, a.harness, st, specs[st["problem_id"]], want[st["sid"]], None))
        except EOFError:
            pass
    n_train = len(jobs)
    for line in opener(a.queries):
        q = json.loads(line)
        if q["variant"] in CHAIN_FAMILIES and (not a.groups or q["group"] in a.groups):
            st = dict(q, split="test")
            jobs.append((a.code, a.harness, st, specs[q["problem_id"]], [m["move"] for m in q["moves"]],
                         [m["result"] for m in q["moves"]]))
    print("training states %d of %d wanted; test states %d" % (n_train, len(want), len(jobs) - n_train), flush=True)
    from multiprocessing import Pool
    n = n_err = 0
    with open(a.out + ".tmp", "w", encoding="utf-8") as fo, Pool(a.workers) as pool:
        for rec, err in pool.imap_unordered(work, jobs, chunksize=1):
            n += 1
            if err:
                n_err += 1
                print("!!! %s: %s" % (rec["sid"], err), flush=True)
                continue
            fo.write(json.dumps(rec) + "\n")
            if n % 100 == 0:
                print("%d/%d states, %d errors" % (n, len(jobs), n_err), flush=True)
    os.replace(a.out + ".tmp", a.out)
    print("CHAIN_TRUTH_DONE %d states, %d errors" % (n, n_err))


if __name__ == "__main__":
    main()

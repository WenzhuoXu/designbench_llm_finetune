"""Step 3 of a screened supervised round: every sampled reply checked in the simulator.

As dataxp.ownrows.score scores the replies behind the dx38 rows, with its own helpers: the reply's
last <action> line is parsed and applied on a clone of the replayed state (a move the state refuses
is 'illegal'), the design after it is scored with the same PhiEvaluator and per-move seeds, and
pct = the share of the state's enumerated legal moves whose potential is <= the reply's, in
percent. There is no teacher move at a policy state; instead each reply also carries
    gain            its potential is above the state's own (scored here, seed (pid, 'scr', 'state'))
    feasible_after  the design after it meets every constraint
    same_as_own     it is the move the policy applied at this state in the source episode
One line per reply, and one line per state ("state": true) with the legal set's size, the state's
potential and the seconds spent.

    <sim python> score_states.py --code <dataxp code dir> --states states.jsonl --samples samples.jsonl \\
        --out scored.jsonl [--workers 32]
Resumable: states already in --out are skipped.
"""
import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import collections
import json
import sys
import time


def score_state(job):
    row, samples = job
    import numpy as np
    from dataxp.domain import Move, stable_seed
    from dataxp.domains import get_domain
    from dataxp.ownrows.common import (evaluator, legal_rng, move_seed, parse_reply, percentile, reasoning_of,
                                       rebuild_state, try_apply)
    t0 = time.time()
    sid, pid = row["sid"], row["problem_id"]
    out = []
    try:
        d = get_domain(row["variant"])
        st = rebuild_state(d, row["spec"], row["history"])
        ev = evaluator(d, row["spec"])
        base = ev.score(st, seed0=stable_seed(pid, "scr", "state"))
        phi0 = float(base.mean)
        cand = []
        for m in d.legal_moves(st, legal_rng(pid, row["k"])):
            ch = d.clone(st)
            try:
                d.apply(ch, m)
            except Exception:                                # noqa: BLE001
                continue
            cand.append(float(ev.score(ch, seed0=move_seed(pid, d, m)).mean))
        cand = np.array(cand, float)
        own = Move.from_dict(row["own_action"]) if row.get("own_action") else None
    except Exception as e:                                   # noqa: BLE001
        return [{"sid": sid, "variant": row["variant"], "state": True, "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                 "t": round(time.time() - t0, 2)}]
    for k, o in samples:
        ans = o.get("answer") or ""
        rec = {"sid": sid, "variant": row["variant"], "sample": k, "ok": False, "parsed": False, "legal": False,
               "reason": None, "n_candidates": int(cand.size), "finish": o.get("finish"),
               "words": len(reasoning_of(ans).split()), "think_leak": "<think>" in ans or "</think>" in ans}
        if o.get("finish") == "length" or not ans:
            rec["reason"] = "truncated" if o.get("finish") == "length" else "empty"
            out.append(rec)
            continue
        mv, why = parse_reply(d, ans)
        if mv is None:
            rec["reason"] = why
            out.append(rec)
            continue
        rec["parsed"] = True
        rec["pick_text"] = d.format_move(mv)
        child, applied, info = try_apply(d, st, mv)
        if child is None:
            rec["reason"] = info["err"]
            out.append(rec)
            continue
        sc = ev.score(child, seed0=move_seed(pid, d, applied))
        phi = float(sc.mean)
        rec.update(ok=True, legal=True, action=applied.as_dict(), text=d.format_move(applied), phi=phi,
                   pct=percentile(cand, phi), gain=bool(phi > phi0 + 1e-9), feasible_after=bool(sc.feasible),
                   clamped=bool(info["clamped"]), noop=bool(info["noop"]),
                   same_as_own=bool(own is not None and applied.matches(own, rel=1e-2)))
        out.append(rec)
    out.append({"sid": sid, "variant": row["variant"], "state": True, "n_candidates": int(cand.size),
                "phi_before": phi0, "phi_recorded": row.get("phi_before"), "feasible_before": bool(base.feasible),
                "cand_best": float(cand.max()) if cand.size else None,
                "cand_gain_share": float((cand > phi0 + 1e-9).mean()) if cand.size else None,
                "t": round(time.time() - t0, 2)})
    return out


def _init(code: str):
    sys.path.insert(0, code)


def main(argv=None) -> int:
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--code", required=True, help="directory that holds the dataxp package")
    ap.add_argument("--states", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    states = {}
    for line in open(a.states, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            states[r["sid"]] = r
    per = collections.defaultdict(dict)
    for line in open(a.samples, encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("error") or o["sid"] not in states:
            continue
        per[o["sid"]][int(o["sample"])] = o
    done = set()
    try:
        for line in open(a.out, encoding="utf-8"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("state"):
                done.add(o["sid"])
    except FileNotFoundError:
        pass
    jobs = [(states[sid], sorted(s.items())) for sid, s in per.items() if sid not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print("states with replies %d, already scored %d, to score %d" % (len(per), len(done), len(jobs)), flush=True)
    t0 = time.time()
    secs = collections.defaultdict(list)
    n = 0
    with open(a.out, "a", encoding="utf-8") as fo, ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.code,)) as ex:
        for recs in ex.map(score_state, jobs, chunksize=1):
            for r in recs:
                fo.write(json.dumps(r) + "\n")
                if r.get("state"):
                    secs[r["variant"]].append(r.get("t") or 0.0)
            fo.flush()
            n += 1
            if n % 200 == 0:
                print("%d/%d states, %.0f s" % (n, len(jobs), time.time() - t0), flush=True)
    for v in sorted(secs):
        s = sorted(secs[v])
        print("%-10s states %5d, seconds per state: median %.1f, mean %.1f, max %.1f" % (v, len(s), s[len(s) // 2], sum(s) / len(s), s[-1]))
    print("SCORE_STATES_DONE %d states in %.0f s" % (n, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

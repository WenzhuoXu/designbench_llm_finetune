"""Step 4: score every sampled reply by Phi percentile and against the teacher.

Port of up_tools/own_phi.py generalised to Move. Per (state, sample): the
last <action>...</action> of the reply is read with domain.parse_move, applied
on a clone of the replayed state (a move the state refuses is 'illegal'),
scored with the same PhiEvaluator and seeds cand_phi used, and placed among
the cached legal moves: pct = share of legal moves with Phi <= the pick's,
in percent. same_move = Move.matches(teacher move, rel=1e-2) (the
renderer's wrong_action tolerance). For SET_PARAM picks the sftprobe's looser
rule (gold_probe._match: same element+parameter, same direction, step within
x2) is recorded too as same_move_loose.

When the states file has no cached 'moves' (cand_phi not run) the legal set is
scored here, once per state.

    python -m dataxp.ownrows.score --states <dir>/states_phi.jsonl \
        --samples <dir>/samples.jsonl --out <dir>/scored.jsonl [--workers 1]
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..domain import Move
from ..domains import get_domain
from ..gold_probe import _match
from .common import (evaluator, legal_rng, move_seed, parse_reply, percentile, read_jsonl,
                     reasoning_of, rebuild_state, try_apply)

_OFFSET_RULE = {"rule": "infeasible"}


def _init(rule: str):
    _OFFSET_RULE["rule"] = rule


def score_sid(job):
    sid, row, samples = job
    d = get_domain(row["variant"])
    spec, pid = row["spec"], row["problem_id"]
    st = rebuild_state(d, spec, row["history"])
    ev = evaluator(d, spec, row.get("offset_rule") or _OFFSET_RULE["rule"])
    gold = Move.from_dict(row["gold"])
    if row.get("moves"):
        cand = np.array([m["phi"] for m in row["moves"]], float)
        gold_phi = float(row["gold_phi"])
    else:                                   # cand_phi not run: score the legal set here
        cand = []
        for m in d.legal_moves(st, legal_rng(pid, row["k"])):
            ch = d.clone(st)
            try:
                d.apply(ch, m)
            except Exception:
                continue
            cand.append(float(ev.score(ch, seed0=move_seed(pid, d, m)).mean))
        cand = np.array(cand, float)
        ch = d.clone(st)
        d.apply(ch, gold)
        gold_phi = float(ev.score(ch, seed0=move_seed(pid, d, gold)).mean)
    gold_pct = percentile(cand, gold_phi)
    out = []
    for k, o in samples:
        ans = o.get("answer") or ""
        rec = {"sid": sid, "variant": row["variant"], "sample": k, "mode": o.get("mode"),
               "prompt": o.get("prompt"), "ok": False, "parsed": False, "legal": False,
               "reason": None, "n_candidates": int(cand.size), "gold_phi": gold_phi,
               "gold_pct": gold_pct, "finish": o.get("finish"), "tokens": o.get("tokens"),
               "words": len(reasoning_of(ans).split()), "think_leak": "<think>" in ans}
        if o.get("error") and not ans:
            rec["reason"] = "error"
            out.append(rec)
            continue
        if o.get("finish") == "length":
            rec["reason"] = "truncated"
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
        phi = float(ev.score(child, seed0=move_seed(pid, d, applied)).mean)
        rec.update(ok=True, legal=True, action=applied.as_dict(), text=d.format_move(applied),
                   phi=phi, pct=percentile(cand, phi), ge_gold=bool(phi >= gold_phi - 1e-9),
                   same_move=bool(applied.matches(gold, rel=1e-2)),
                   clamped=bool(info["clamped"]), noop=bool(info["noop"]))
        if applied.is_set_param and gold.is_set_param and row["gold"].get("v_old") is not None:
            # the sftprobe's own rule (gold_probe._match) needs the teacher's v_old,
            # which a trajectory action carries
            tgt, loose = _match(row["gold"], int(applied.args["i"]), applied.args["p"],
                                float(applied.args["v"]))
            rec.update(same_target=bool(tgt), same_move_loose=bool(loose))
        out.append(rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, help="states_phi.jsonl (or states.jsonl, slower)")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--offset-rule", default="infeasible", choices=("infeasible", "raw", "legacy_scaled"))
    ap.add_argument("--min-pct", type=float, default=90.0, help="only for the printed summary")
    a = ap.parse_args(argv)
    _init(a.offset_rule)

    states = {r["sid"]: r for r in read_jsonl(a.states)}
    last = {}
    with open(a.samples, encoding="utf-8") as fh:
        for l in fh:
            try:
                o = json.loads(l)
            except ValueError:
                continue
            key = (o["sid"], int(o["sample"]))
            # a completed answer beats an error / truncated record for the same sample
            if key not in last or o.get("answer") or not last[key].get("answer"):
                last[key] = o
    per = {}
    for (sid, k), o in last.items():
        if sid in states:
            per.setdefault(sid, []).append((k, o))
    jobs = [(sid, states[sid], sorted(samples)) for sid, samples in sorted(per.items())]
    print("states with replies %d, replies %d" % (len(jobs), len(last)), flush=True)

    recs = []
    done = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        if a.workers > 1:
            ex = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.offset_rule,))
            it = ex.map(score_sid, jobs, chunksize=1)
        else:
            ex, it = None, map(score_sid, jobs)
        try:
            for rs in it:
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
                    recs.append(r)
                done += 1
                if done % 25 == 0:
                    print("%d/%d" % (done, len(jobs)), flush=True)
        finally:
            if ex is not None:
                ex.shutdown()

    by_v = {}
    for r in recs:
        by_v.setdefault(r["variant"], []).append(r)
    print("%-10s %7s %9s %7s %6s %9s %8s %8s %12s" % (
        "variant", "replies", "completed", "parsed", "legal", "same_move", "mean_pct", ">=%d%%" % a.min_pct,
        "states_elig"))
    for v in sorted(by_v):
        rs = by_v[v]
        ok = [r for r in rs if r["ok"]]
        elig = {r["sid"] for r in ok if (r["pct"] is not None and r["pct"] >= a.min_pct) or r["same_move"]}
        print("%-10s %7d %9.2f %7.2f %6.2f %9.2f %8.1f %8.2f %5d/%-6d" % (
            v, len(rs), np.mean([r["reason"] not in ("error", "truncated") for r in rs]),
            np.mean([r["parsed"] for r in rs]), np.mean([r["legal"] for r in rs]),
            np.mean([r["same_move"] for r in ok]) if ok else 0.0,
            np.mean([r["pct"] for r in ok if r["pct"] is not None]) if ok else 0.0,
            np.mean([(r["pct"] or 0) >= a.min_pct for r in ok]) if ok else 0.0,
            len(elig), len({r["sid"] for r in rs})))
    return 0


if __name__ == "__main__":
    sys.exit(main())

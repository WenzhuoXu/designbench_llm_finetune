"""Step 3 of a screened supervised round: every sampled reply checked in the simulator.

As dataxp.ownrows.score scores the replies behind the dx38 rows, with its own helpers: the last
<action> line of the reply's visible answer (the text after the think block, where the harness
reads the move) is parsed and applied on a clone of the replayed state (a move the state refuses
is 'illegal'), the design after it is scored with the same PhiEvaluator and per-move seeds, and
pct = the share of the state's enumerated legal moves whose potential is <= the reply's, in
percent. There is no teacher move at a policy state; instead each reply also carries
    gain            its potential is above the state's own (scored here, seed (pid, 'scr', 'state'))
    feasible_after  the design after it meets every constraint
    same_as_own     it is the move the policy applied at this state in the source episode
One line per reply, and one line per state ("state": true) with the legal set's size, the state's
potential and the seconds spent.

A rationale with FORECAST lines (forecasts.py; a forecast-checked round asks for them at collection)
also gets `forecasts`: every candidate it names, parsed and applied alone on a clone of the state,
with what the reply said (refused | worse | better) and what the simulator returned (`truth`).

    <sim python> score_states.py --code <dataxp code dir> --states states.jsonl --samples samples.jsonl \\
        --out scored.jsonl [--workers 32]
Resumable: states already in --out are skipped.

A state written by open_states.py ("open_rules": true) is rebuilt and every move applied under the open rules of
dataxp_policy_eval_v3r.py --open-rules (open_rules.py), the legal-move enumeration of pct included. In photonics
such a design can break the fabrication rule, and the potential gives every such design the same objective (10), so
gain and pct go by open_rules.open_rank there: fewer rule-breaking pixels first, then the potential (a move that
breaks the rule at a clean design is never a gain; a repair is). Those replies also carry bad_after and open_rank, the
state line bad_before; pct_basis "state" marks a design no legal (fabricable) candidate exists for, where the reply
is ranked against the design as it stands.
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
    from dataxp.ownrows.common import (evaluator, legal_rng, move_seed, parse_reply, percentile, rebuild_state,
                                       try_apply)
    from forecasts import MAX_SCORED, read as read_forecasts, truth_of
    t0 = time.time()
    sid, pid = row["sid"], row["problem_id"]
    out = []
    # a state of an open-rules round (open_states.py) is rebuilt and its moves applied under the open rules
    # (open_rules.py): photonics without the fabrication refusal, a grid2op move that ends the episode refused
    opened = bool(row.get("open_rules"))
    # photonics under the open rules: a design can break the fabrication rule, and Phi alone does not rank such a
    # design (open_rules.open_rank): gain and percentile go by (fewer rule-breaking pixels, then Phi)
    lex = opened and row["variant"] == "photonics"
    tap = try_apply
    try:
        d = get_domain(row["variant"])
        if opened:
            from dataxp.domain import apply_action
            from open_rules import bad_pixels, open_gain, open_rank, rebuild_open, try_apply_open
            st = rebuild_open(d, row["variant"], row["spec"], row["history"], apply_action, Move.from_dict)

            def tap(dd, s, m):
                return try_apply_open(dd, row["variant"], s, m, try_apply)
        else:
            st = rebuild_state(d, row["spec"], row["history"])
        ev = evaluator(d, row["spec"])
        base = ev.score(st, seed0=stable_seed(pid, "scr", "state"))
        phi0 = float(base.mean)
        bad0 = bad_pixels(row["variant"], st) if lex else 0
        cand, cand_rank = [], []
        for m in d.legal_moves(st, legal_rng(pid, row["k"])):
            if opened:
                ch = tap(d, st, m)[0]
                if ch is None:
                    continue
            else:
                ch = d.clone(st)
                try:
                    d.apply(ch, m)
                except Exception:                            # noqa: BLE001
                    continue
            cand.append(float(ev.score(ch, seed0=move_seed(pid, d, m)).mean))
            if lex:
                cand_rank.append(open_rank(bad_pixels(row["variant"], ch), cand[-1]))
        cand = np.array(cand, float)
        own = Move.from_dict(row["own_action"]) if row.get("own_action") else None
    except Exception as e:                                   # noqa: BLE001
        return [{"sid": sid, "variant": row["variant"], "state": True, "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                 "t": round(time.time() - t0, 2)}]
    seen = {}                                               # a candidate as the domain formats it -> what the simulator returned

    def candidate(text: str) -> dict:
        """One FORECAST candidate applied alone on a clone of the state (forecasts.py)."""
        mv = d.parse_move(text)
        if mv is None:
            return {"parsed": False}
        key = d.format_move(mv)
        if key not in seen:
            child, applied, info = tap(d, st, mv)
            if child is None:
                seen[key] = {"truth": "refused", "phi": None, "noop": False, "err": info["err"]}
            else:
                phi = float(ev.score(child, seed0=move_seed(pid, d, applied)).mean)
                seen[key] = {"truth": truth_of(True, phi, phi0), "phi": phi, "noop": bool(info["noop"])}
        return dict(seen[key], parsed=True, text=key)

    for k, o in samples:
        # the move is read from the visible answer only (the text after the think block), as the
        # harness reads it: an <action> line drafted inside the rationale is not a move
        ans, think = o.get("answer") or "", o.get("thinking") or ""
        rec = {"sid": sid, "variant": row["variant"], "sample": k, "ok": False, "parsed": False, "legal": False,
               "reason": None, "n_candidates": int(cand.size), "finish": o.get("finish"), "reply_sha1": o.get("sha1"),
               "words": len(think.split()), "think_leak": "<think>" in think or "</think>" in think,
               "action_tag_in_rationale": "<action>" in think}
        lines, unread = read_forecasts(think)
        if lines or unread:                                  # every candidate the rationale forecasts, whatever the reply's own move does
            rec["n_forecast_lines"], rec["forecast_unread"] = len(lines), unread
            try:
                rec["forecasts"] = [dict(candidate(text), move=text, said=said) for text, said in lines[:MAX_SCORED]]
            except Exception as e:                           # noqa: BLE001
                rec["forecasts"], rec["forecast_error"] = [], "%s: %s" % (type(e).__name__, str(e)[:200])
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
        for c in rec.get("forecasts") or []:                 # the candidate the reply went on to make
            c["is_pick"] = bool(c.get("parsed") and c["text"] == rec["pick_text"])
        child, applied, info = tap(d, st, mv)
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
        if lex:
            bad = bad_pixels(row["variant"], child)
            r = open_rank(bad, phi)
            # the legal candidates are fabricable designs only (photonics.legal_moves); at a design that no single
            # move repairs there are none, and the reply is then ranked against the design as it stands
            ref = cand_rank or [open_rank(bad0, phi0)]
            rec.update(bad_after=bad, open_rank=list(r), gain=bool(open_gain(bad, phi, bad0, phi0)),
                       pct=float(sum(c <= r for c in ref)) * 100.0 / len(ref),
                       pct_basis="candidates" if cand_rank else "state")
        out.append(rec)
    srec = {"sid": sid, "variant": row["variant"], "state": True, "n_candidates": int(cand.size),
            "phi_before": phi0, "phi_recorded": row.get("phi_before"), "feasible_before": bool(base.feasible),
            "cand_best": float(cand.max()) if cand.size else None,
            "cand_gain_share": float((cand > phi0 + 1e-9).mean()) if cand.size else None,
            "t": round(time.time() - t0, 2)}
    if lex:
        srec.update(bad_before=bad0, cand_gain_share=(float(np.mean([open_gain(-c[0], c[1], bad0, phi0) for c in cand_rank]))
                                                      if cand_rank else None))
    out.append(srec)
    return out


def _init(code: str):
    sys.path.insert(0, code)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # forecasts.py, in the worker processes too


def main(argv=None) -> int:
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
    n = n_fail = n_fc_err = 0
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from crashsafe import run_jobs
    with open(a.out, "a", encoding="utf-8") as fo:
        if os.path.getsize(a.out) > 0:
            with open(a.out, "rb") as f:                     # close off a last line cut by a killed run
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    fo.write("\n")
        for i, recs, err in run_jobs(score_state, jobs, a.workers, initializer=_init, initargs=(a.code,)):
            if err is not None:                              # a state whose scoring kills its process is final:
                row = jobs[i][0]                             # the next run does not try it again
                recs = [{"sid": row["sid"], "variant": row["variant"], "state": True, "error": err}]
                print("!!! %s: %s" % (row["sid"], err), flush=True)
            for r in recs:
                fo.write(json.dumps(r) + "\n")
                n_fc_err += bool(r.get("forecast_error"))
                if r.get("state"):
                    n_fail += bool(r.get("error"))
                    secs[r["variant"]].append(r.get("t") or 0.0)
            fo.flush()
            n += 1
            if n % 200 == 0:
                print("%d/%d states, %d failed, %.0f s" % (n, len(jobs), n_fail, time.time() - t0), flush=True)
    for v in sorted(secs):
        s = sorted(secs[v])
        print("%-10s states %5d, seconds per state: median %.1f, mean %.1f, max %.1f" % (v, len(s), s[len(s) // 2], sum(s) / len(s), s[-1]))
    if n_fc_err:
        print("!!! %d replies whose forecast check raised an error (forecast_error in their records)" % n_fc_err, flush=True)
    print("SCORE_STATES_DONE %d states (%d failed) in %.0f s" % (n, n_fail, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

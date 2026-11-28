"""inventory2 truth with common random numbers (CPU, Orchard; con1-2m-rows, 11-28, parent's order).

    python inv2_common_seed.py --queries queries_v3.jsonl --out queries_v3_inv2cs.jsonl --code CODE --harness V3RC \\
        --problems bench_problems.jsonl [--workers 30] [--report report.txt]

inventory2 is the one stochastic domain (its constructor defaults to stochastic=True; 20 demand paths seeded from
(demand_seed, seed)). A state's printed metrics (metrics0) are its score at the seed of its last applied move,
stable_seed(pid, 'eval', 's', t_last + 1) (0 before any move: conseq_lib.State / commit), while every check of a move
scores the child at stable_seed(pid, 'eval', 's', t + 1) (conseq_lib.State.result_of, the harness's own check), so
before and after use different demand paths. Here each inventory2 question's move is checked again with the child
scored at the seed of the printed metrics0, through the same result_of (only the seed index differs).

Per state: rebuild from the record's turn_log (conseq_lib.rebuild's walk, tracking the seed of the current score);
checks, all of which must hold: the rebuilt metrics equal metrics0; consequence_user(rebuilt served prompt) equals the
record's user text; result_of at the original seed reproduces every move's recorded truth (phi, feasible, verdict,
metrics_after, result). Then the new truth: phi, feasible, viol, metrics_after, verdict, result (and refused) of each
move replaced by the common-seed check; every other field and every line of the other families byte-identical.
Prints the change counts (directions per metric, exact no-ops, feasible, verdict) and INV2CS_DONE / INV2CS_FAIL.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ENV = None
KEYS = ("phi", "feasible", "viol", "metrics_after", "verdict", "result", "refused")


def same(b, a):
    return abs(a - b) <= max(1e-5 * abs(b), 1e-12)


def tdir(b, a):
    return "same" if same(b, a) else ("up" if a > b else "down")


def job(args):
    code, harness, q, spec = args
    global _ENV
    if _ENV is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        _ENV = Env(code, harness)
    from conseq_lib import State, consequence_user
    out = {"sid": q["sid"], "checks": collections.Counter(), "moves": []}
    s = State(_ENV, q["variant"], spec)
    s.turn_log = []
    k_cur = 0                                            # seed index of s.cur: stable_seed(pid, 'eval', 's', k_cur)
    for kind, mv, answer in q.get("turn_log") or []:
        if kind == "skip":
            s.turn_log.append((kind, None, None))
            s.skip_turn()
            continue
        mvo = _ENV.D.Move.from_dict(mv) if s.native else {"i": int(mv["i"]), "p": mv["p"], "v": float(mv["v"])}
        s.turn_log.append((kind, mv, answer))
        t_before = s.t
        if s.commit(mvo, answer) is None:
            k_cur = t_before + 1
    c = out["checks"]
    c["t matches"] += s.t == int(q["t"])
    c["metrics0 reproduced"] += all(abs(float(s.metrics[k]) - float(v)) <= 1e-12 * max(1.0, abs(float(v)))
                                    for k, v in q["metrics0"].items())
    c["prompt reproduced"] += consequence_user(s.prompt()[1]) == q["user"]
    for m in q["moves"]:
        mv = s.parse(m["move"])
        if mv is None:
            c["move unparsed"] += 1
            out["moves"].append(None)
            continue
        old = s.result_of(mv)                            # the recorded truth's seed: t + 1
        c["old truth reproduced"] += all(old.get(x) == m.get(x) for x in KEYS if x in m or x in old)
        t = s.t
        s.t = k_cur - 1                                  # result_of scores at stable_seed(pid, 'eval', 's', s.t + 1)
        try:
            new = s.result_of(mv)
        finally:
            s.t = t
        out["moves"].append(new)
    out["k_cur"], out["t"] = k_cur, s.t
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    specs = {}
    for p in a.problems:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            specs[r["spec"]["problem_id"]] = r["spec"]
    lines = open(a.queries, encoding="utf-8").read().splitlines(keepends=True)
    qs = [json.loads(l) for l in lines]
    jobs = [(a.code, a.harness, q, specs[q["problem_id"]]) for q in qs if q["variant"] == "inventory2"]
    from multiprocessing import Pool
    res = {}
    with Pool(a.workers) as pool:
        for out in pool.imap_unordered(job, jobs, chunksize=1):
            res[out["sid"]] = out
    C = collections.Counter()
    D = collections.defaultdict(collections.Counter)
    seeds = collections.Counter()
    with open(a.out + ".part", "w", encoding="utf-8") as fo:
        for line, q in zip(lines, qs):
            if q["variant"] != "inventory2":
                fo.write(line)
                continue
            r = res[q["sid"]]
            C.update(r["checks"])
            C["states"] += 1
            seeds["k_cur = t" if r["k_cur"] == r["t"] else "k_cur < t (last turn(s) refused or skipped)"] += 1
            m0 = q["metrics0"]
            nm = []
            for m, new in zip(q["moves"], r["moves"]):
                C["questions"] += 1
                if new is None:
                    nm.append(m)
                    continue
                mm = dict(m)
                for x in KEYS:
                    mm.pop(x, None)
                for x in KEYS:
                    if x in new:
                        mm[x] = new[x]
                nm.append(mm)
                C["verdict changed"] += m.get("verdict") != new.get("verdict")
                C["applied <-> refused changed"] += (m.get("verdict") == "refused") != (new.get("verdict") == "refused")
                if new.get("verdict") != "refused" and m.get("verdict") != "refused":
                    C["feasible changed"] += bool(m.get("feasible")) != bool(new.get("feasible"))
                    oa, na = m["metrics_after"], new["metrics_after"]
                    noop = all(same(float(m0[k]), float(na[k])) for k in m0 if k in na)
                    C["exact no-op (new truth)"] += noop
                    C["exact no-op (old truth)"] += all(same(float(m0[k]), float(oa[k])) for k in m0 if k in oa)
                    for k in m0:
                        if k in oa and k in na:
                            do, dn = tdir(float(m0[k]), float(oa[k])), tdir(float(m0[k]), float(na[k]))
                            D[k]["lines"] += 1
                            D[k]["direction changed"] += do != dn
                            D[k]["%s -> %s" % (do, dn)] += 1
            q2 = dict(q)
            q2["moves"] = nm
            q2["truth"] = "inv2_common_seed"
            fo.write(json.dumps(q2, ensure_ascii=False) + "\n")
    os.replace(a.out + ".part", a.out)
    rep = ["inventory2 common-seed truth: %s -> %s" % (a.queries, a.out)]
    for k in sorted(C):
        rep.append("  %-45s %d" % (k, C[k]))
    for k in sorted(seeds):
        rep.append("  states with %-40s %d" % (k, seeds[k]))
    rep.append("per metric (applied in both truths): lines, direction changed (share), old -> new")
    for k in sorted(D):
        d = D[k]
        rep.append("  %-16s %4d  %4d (%.3f)  %s" % (k, d["lines"], d["direction changed"], d["direction changed"] / max(1, d["lines"]),
                                                  ", ".join("%s %d" % (x, v) for x, v in sorted(d.items()) if "->" in x)))
    ok = C["states"] == C["t matches"] == C["metrics0 reproduced"] == C["prompt reproduced"] \
        and C["old truth reproduced"] == C["questions"] and not C["move unparsed"]
    rep.append("INV2CS_%s" % ("DONE" if ok else "FAIL"))
    txt = "\n".join(rep)
    if a.report:
        open(a.report, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

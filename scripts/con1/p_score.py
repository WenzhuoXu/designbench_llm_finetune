"""con1 P-form skill test, step 2 (CPU): simulate the action each reply chose and pair it with the reply's own
'EXPECTED RESULT:' block, so facets.py scores the prediction exactly like the question form.

    python p_score.py --queries queries_v2.jsonl --replies preds_P_LABEL.jsonl --format F --code C --harness V3RC \\
        --problems bench_problems.jsonl --out-prefix DIR/P_LABEL [--workers 30]
Writes OUT_PREFIX.queries.jsonl (each state with one move: the chosen action and its simulated result),
OUT_PREFIX.preds.jsonl (qi 0, pass zero, target F, text = the EXPECTED RESULT block or '' when absent) and
OUT_PREFIX.policy.txt (verdict mix of the chosen actions, share of replies with a block, unparsable actions). Then:
    python facets.py --queries OUT_PREFIX.queries.jsonl --preds OUT_PREFIX.preds.jsonl --problems ... --code ... --harness ...
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HEADER = "EXPECTED RESULT:"
_ENV = None


def job(args):
    code, harness, q, spec, rep = args
    global _ENV
    if _ENV is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        _ENV = Env(code, harness)
    from conseq_lib import rebuild
    from transitions import move_args
    ans = rep.get("answer") or ""
    m = re.findall(r"<action>\s*(.*?)\s*</action>", ans, re.S)
    out = {"sid": q["sid"], "has_block": HEADER in ans}
    if not m:
        out["fail"] = "no_action"
        return out
    text = " ".join(m[-1].split())
    try:
        s = rebuild(_ENV, {"variant": q["variant"], "t": q["t"], "turn_log": q["turn_log"]}, spec)
        mv = s.parse(text)
        if mv is None:
            out["fail"] = "unparsed"
            return out
        res = s.result_of(mv)
        res["src"], res["args"] = ["chosen"], move_args(s, mv)
        out["move"] = res
        out["block"] = ans.split(HEADER, 1)[1].strip() if HEADER in ans else ""
        return out
    except Exception as e:                                 # noqa: BLE001
        out["fail"] = "error: %s" % str(e)[:200]
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--format", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args(argv)
    specs = {}
    for p in a.problems:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            specs[r["spec"]["problem_id"]] = r["spec"]
    qs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.queries, encoding="utf-8")}
    reps = {}
    for l in open(a.replies, encoding="utf-8"):
        r = json.loads(l)
        reps[r["sid"]] = r
    jobs = [(a.code, a.harness, qs[s], specs[qs[s]["problem_id"]], r) for s, r in reps.items() if s in qs]
    from multiprocessing import Pool
    c = collections.Counter()
    label = os.path.basename(a.out_prefix)
    with Pool(a.workers) as pool, open(a.out_prefix + ".queries.jsonl", "w", encoding="utf-8") as fq, \
            open(a.out_prefix + ".preds.jsonl", "w", encoding="utf-8") as fp:
        for out in pool.imap_unordered(job, jobs, chunksize=1):
            q = qs[out["sid"]]
            v = q["variant"]
            c[(v, "replies")] += 1
            c[(v, "with block")] += out["has_block"]
            if "move" not in out:
                c[(v, "action %s" % out.get("fail"))] += 1
                continue
            m = out["move"]
            c[(v, "chosen %s" % m["verdict"])] += 1
            qq = {k: q[k] for k in ("sid", "variant", "problem_id", "t", "k", "system", "user", "phi0", "bad0",
                                    "metrics0", "binding", "group")}
            qq["moves"] = [m]
            fq.write(json.dumps(qq, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"sid": q["sid"], "variant": v, "predictor": label, "pass": "zero", "qi": 0,
                                 "target": a.format, "text": out["block"]}, ensure_ascii=False) + "\n")
    lines = []
    for v in sorted({k[0] for k in c}):
        n = c[(v, "replies")]
        lines.append("%-10s replies %d, with EXPECTED RESULT %.3f | chosen: %s | %s" % (
            v, n, c[(v, "with block")] / max(1, n),
            " ".join("%s %.3f" % (x, c[(v, "chosen %s" % x)] / max(1, n)) for x in ("better", "worse", "refused")),
            " ".join("%s %d" % (k[1], c[k]) for k in sorted(c) if k[0] == v and k[1].startswith("action"))))
    txt = "\n".join(lines)
    open(a.out_prefix + ".policy.txt", "w").write(txt + "\n")
    print(txt)
    print("P_SCORE_DONE")


if __name__ == "__main__":
    main()

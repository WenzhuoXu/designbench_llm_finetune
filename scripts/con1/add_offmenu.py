"""con1: add out-of-menu moves (transitions.offmenu_moves, src 'offmenu') to native-family states generated before
11-25 15:30. Each state is rebuilt, its menu re-enumerated, k random-argument calls simulated and appended to its moves.

    python add_offmenu.py --code C --harness V3RC --problems P [P ...] --states S [S ...] --out OUT.jsonl.gz
        [--k 2] [--sids sids.txt] [--workers 12]
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transitions import opener, seeded  # noqa: E402

_ENV = None


def job(args):
    code, harness, st, spec, k = args
    global _ENV
    if _ENV is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        _ENV = Env(code, harness)
    import numpy as np
    from conseq_lib import rebuild
    from transitions import move_args, offmenu_moves
    try:
        s = rebuild(_ENV, {"variant": st["variant"], "t": st["t"], "turn_log": st["turn_log"]}, spec)
        have = {m["move"] for m in st["moves"]}
        rng = np.random.default_rng(seeded("con1-offmenu", st["sid"]))
        added = 0
        for mv in offmenu_moves(s, s.legal(), k, rng):
            text = s.fmt(mv)
            if text in have:
                continue
            mv2 = s.parse(text)
            if mv2 is None:
                continue
            res = s.result_of(mv2)
            res["move"], res["src"], res["args"] = text, ["offmenu"], move_args(s, mv2)
            st["moves"].append(res)
            have.add(text)
            added += 1
        st["n_offmenu"] = added
        return st, None
    except Exception as e:                                 # noqa: BLE001
        return st, "%s: %s" % (type(e).__name__, str(e)[:200])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--sids", default=None, help="only these state ids (one per line)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    specs = {}
    for p in a.problems:
        for line in opener(p):
            r = json.loads(line)
            sp = r.get("spec", r)
            specs[sp["problem_id"]] = sp
    want = {l.strip() for l in open(a.sids)} if a.sids else None
    done = set()
    if os.path.exists(a.out):
        try:
            for line in opener(a.out):
                done.add(json.loads(line)["sid"])
        except EOFError:
            pass
    jobs, seen = [], set()
    for p in a.states:
        try:
            for line in opener(p):
                st = json.loads(line)
                if st["variant"] not in ("engibench", "grid2op", "photonics", "molopt") or st["sid"] in seen:
                    continue
                if want is not None and st["sid"] not in want:
                    continue
                seen.add(st["sid"])
                if st["sid"] in done:
                    continue
                jobs.append((a.code, a.harness, st, specs[st["problem_id"]], a.k))
        except EOFError:
            pass
    print("states to augment %d (done %d)" % (len(jobs), len(done)), flush=True)
    from multiprocessing import Pool
    from conseq_lib import jdump
    n = n_err = 0
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for st, err in pool.imap_unordered(job, jobs, chunksize=1):
            n += 1
            if err:
                n_err += 1
                print("!!! %s: %s" % (st["sid"], err), flush=True)
                continue
            with gzip.open(a.out, "at", encoding="utf-8") as fo:
                fo.write(jdump(st) + "\n")
            if n % 100 == 0:
                print("%d/%d, %d errors" % (n, len(jobs), n_err), flush=True)
    print("ADD_OFFMENU_DONE %d states, %d errors" % (n, n_err))


if __name__ == "__main__":
    main()

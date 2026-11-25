"""con1 check of conseq_lib against the harness's own records: replay recorded episodes and compare, byte for byte,
(1) every turn's served (system, user) with the recorded prompt and (2) every recorded check's result message with
State.result_of of the same move at the same state.

    python verify_replay.py --code <dir with dataxp> --harness <v3rc> --problems train_p2_datav4.jsonl \\
        --episodes collect.episodes.jsonl[.gz] [...] [--limit 50]
Episode files are rollout records ({key, variant, episode: {turns: [...]}}) collected with --record-prompts.
"""
import argparse
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conseq_lib import Env, replay_turns  # noqa: E402


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_specs(paths):
    specs = {}
    for p in paths:
        for line in opener(p):
            if line.strip():
                r = json.loads(line)
                s = r.get("spec", r)
                specs[s["problem_id"]] = s
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args(argv)
    env = Env(a.code, a.harness)
    specs = read_specs(a.problems)
    c = collections.Counter()
    shown = 0
    for path in a.episodes:
        n = 0
        for line in opener(path):
            r = json.loads(line)
            ep = r["episode"]
            if a.variants and r["variant"] not in a.variants:
                continue
            if ep.get("crashed") or not ep.get("turns"):
                continue
            n += 1
            if a.limit and n > a.limit:
                break
            spec = specs[r["problem_id"]]
            c["episodes"] += 1
            try:
                for rec, s in replay_turns(env, r["variant"], spec, ep["turns"]):
                    if "user" in rec:
                        sysm, user = s.prompt()
                        ok = sysm == rec["system"] and user == rec["user"]
                        c["prompt same" if ok else "prompt DIFFERENT"] += 1
                        if not ok and shown < 3:
                            shown += 1
                            u, v = user, rec["user"]
                            k = next((i for i in range(min(len(u), len(v))) if u[i] != v[i]), min(len(u), len(v)))
                            print("PROMPT DIFF %s t%d at %d:\n mine: %r\n rec:  %r" % (r["key"], s.t, k, u[k - 80:k + 120],
                                                                                    v[k - 80:k + 120]))
                    for chk in rec.get("checks") or []:
                        if not chk.get("move"):
                            c["check unreadable (skipped)"] += 1
                            continue
                        mv = s.parse(chk["text"])
                        out = s.result_of(mv)
                        ok = out["result"] == chk["result"]
                        c["check same" if ok else "check DIFFERENT"] += 1
                        c["check verdict %s" % out["verdict"]] += 1
                        if not ok and shown < 6:
                            shown += 1
                            print("CHECK DIFF %s t%d %s:\n mine: %r\n rec:  %r" % (r["key"], s.t, chk["text"],
                                                                                 out["result"][:600], chk["result"][:600]))
            except Exception as e:                       # noqa: BLE001
                c["episode replay error"] += 1
                if shown < 8:
                    shown += 1
                    print("ERROR %s: %s: %s" % (r["key"], type(e).__name__, str(e)[:300]))
    for k in sorted(c):
        print("%-40s %d" % (k, c[k]))


if __name__ == "__main__":
    main()

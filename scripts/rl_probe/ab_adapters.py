"""Two adapters evaluated on the same problems (rollout eval, T=0, several runs per problem): per
family, feasibility at 12 and 20 turns in each arm, the difference with a standard error from
resampling problems (both arms and all runs of a problem move together), the paired outcome per
problem (more feasible runs in one arm), how often two runs of one arm disagree on a problem (T=0
serving is not reproducible run to run, so this is the noise a difference has to clear), and what
the turns look like (refused moves, no-ops, completion tokens).

Episodes that crashed or start feasible are left out, as in the launcher's own table.

usage: python ab_adapters.py <A episodes.jsonl> <B episodes.jsonl> [--names A B]
           [--groups "new=engibench,grid2op,photonics" "replay=..." ...] [--boot 10000] [--seed 0]
"""
import argparse
import collections
import json
import math
import random
import statistics


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ep = r.get("episode") or r
        if ep.get("crashed") or ep.get("feasible_start"):
            continue
        out[(ep["variant"], ep["problem_id"], int(r.get("member", 1)))] = ep
    return out


def turn_stats(ep):
    turns = ep.get("turns") or []
    return {"turns": len(turns),
            "refused": sum(1 for t in turns if str(t.get("fail") or "").startswith("illegal")),
            "failed": sum(1 for t in turns if t.get("fail")),
            "noop": sum(1 for t in turns if t.get("noop")),
            "tokens": sum(t.get("completion_tokens") or 0 for t in turns),
            "len_stop": sum(1 for t in turns if t.get("finish") == "length")}


def sign_p(a, b):
    m = a + b
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(a, b) + 1)) / 2 ** m) if m else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--groups", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    A, B = load(args.a), load(args.b)
    both = sorted(set(A) & set(B))
    na, nb = args.names
    print("episodes: %s %d, %s %d, in both %d (only %s %d, only %s %d)" % (
        na, len(A), nb, len(B), len(both), na, len(set(A) - set(B)), nb, len(set(B) - set(A))))
    fams = sorted({k[0] for k in both})
    groups = [(f, [f]) for f in fams]
    for g in args.groups:
        name, members = g.split("=")
        groups.append((name, [m for m in members.split(",") if m]))
    groups.append(("all", fams))
    rng = random.Random(args.seed)

    print("\n%-11s %4s %5s | %-17s | %-17s %7s %6s | %-22s | %s" % (
        "family", "prob", "eps", "feasible@12", "feasible@20", "diff", "se", "problems with more", "problems where the"))
    print("%-11s %4s %5s | %8s %8s | %8s %8s %7s %6s | %-22s | %s" % (
        "", "", "", na[:8], nb[:8], na[:8], nb[:8], "B - A", "", "feasible runs (A / B), p", "runs of one arm disagree (A / B)"))
    for name, members in groups:
        keys = [k for k in both if k[0] in members]
        if not keys:
            continue
        per = collections.defaultdict(lambda: [0, 0, 0, 0, 0, [], []])     # problem: n, a12, b12, a20, b20, runs a, runs b
        for k in keys:
            p = per[(k[0], k[1])]
            p[0] += 1
            p[1] += bool(A[k].get("feasible@12"))
            p[2] += bool(B[k].get("feasible@12"))
            p[3] += bool(A[k].get("feasible@20"))
            p[4] += bool(B[k].get("feasible@20"))
            p[5].append(bool(A[k].get("feasible@20")))
            p[6].append(bool(B[k].get("feasible@20")))
        probs = list(per.values())
        n = sum(p[0] for p in probs)
        tot = [sum(p[i] for p in probs) for i in range(1, 5)]
        diff = (tot[3] - tot[2]) / n
        boots = []
        for _ in range(args.boot):
            s = [probs[rng.randrange(len(probs))] for _ in probs]
            m = sum(p[0] for p in s)
            boots.append((sum(p[4] for p in s) - sum(p[3] for p in s)) / m)
        se = statistics.pstdev(boots)
        more_a = sum(p[3] > p[4] for p in probs)
        more_b = sum(p[4] > p[3] for p in probs)
        dis_a = sum(len(set(p[5])) > 1 for p in probs)
        dis_b = sum(len(set(p[6])) > 1 for p in probs)
        print("%-11s %4d %5d | %8.3f %8.3f | %8.3f %8.3f %+7.3f %6.3f | %3d / %-3d p = %-8.3f | %d / %d" % (
            name, len(probs), n, tot[0] / n, tot[1] / n, tot[2] / n, tot[3] / n, diff, se, more_a, more_b,
            sign_p(more_a, more_b), dis_a, dis_b))

    print("\n%-11s | %-15s | %-15s | %-15s | %-17s | %s" % (
        "family", "turns / episode", "refused turns", "no-op turns", "tokens / turn", "stopped at the length limit"))
    print("%-11s | %7s %7s | %7s %7s | %7s %7s | %8s %8s | %s" % ("", na[:7], nb[:7], na[:7], nb[:7], na[:7], nb[:7], na[:8], nb[:8], "(turns, A / B)"))
    for name, members in groups:
        keys = [k for k in both if k[0] in members]
        if not keys:
            continue
        sa, sb = [turn_stats(A[k]) for k in keys], [turn_stats(B[k]) for k in keys]
        ta, tb = (lambda c: sum(x[c] for x in sa)), (lambda c: sum(x[c] for x in sb))
        print("%-11s | %7.2f %7.2f | %6.1f%% %6.1f%% | %6.1f%% %6.1f%% | %8.0f %8.0f | %d / %d" % (
            name, ta("turns") / len(keys), tb("turns") / len(keys),
            100.0 * ta("refused") / max(1, ta("turns")), 100.0 * tb("refused") / max(1, tb("turns")),
            100.0 * ta("noop") / max(1, ta("turns")), 100.0 * tb("noop") / max(1, tb("turns")),
            ta("tokens") / max(1, ta("turns")), tb("tokens") / max(1, tb("turns")), ta("len_stop"), tb("len_stop")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

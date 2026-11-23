"""Two adapters on the same problems with any number of benchmark pairs per arm (rollout eval, T=0): per family,
each arm's feasibility at 12 and 20 turns pooled over its pairs, the difference B - A of the per-problem feasible rates
(each problem weighs the same; rates, so arms with different numbers of runs compare), a standard error from
resampling problems, the paired outcome per problem (higher feasible rate in one arm, sign test), and each pair's own
feasible@20 count, so the difference can be read against how far two pairs of the same adapter fall apart.

Episodes that crashed or start feasible are left out, as in the launcher's table and ab_adapters.py. Only problems
that have at least one episode in both arms are compared.

usage: python ab_pooled.py --a NAME FILE [FILE ...] --b NAME FILE [FILE ...]
           [--groups "new=engibench,grid2op,photonics" ...] [--variants v1,v2] [--boot 10000] [--seed 0]
Each FILE is one benchmark output (its runs r1, r2, ... are kept apart from other files' runs).
"""
import argparse
import collections
import json
import math
import random
import statistics


def load(paths, variants=None):
    """{(variant, problem): [(file index, member, f12, f20)]} and the list of (file, member) runs seen."""
    per = collections.defaultdict(list)
    for i, path in enumerate(paths):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r.get("episode") or r
            if ep.get("crashed") or ep.get("feasible_start"):
                continue
            if variants and ep["variant"] not in variants:
                continue
            per[(ep["variant"], ep["problem_id"])].append(
                (i, int(r.get("member", 1)), bool(ep.get("feasible@12")), bool(ep.get("feasible@20"))))
    return per


def sign_p(a, b):
    m = a + b
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(a, b) + 1)) / 2 ** m) if m else 1.0


def rate(eps, j):
    return sum(e[j] for e in eps) / len(eps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", nargs="+", required=True, metavar="NAME FILE")
    ap.add_argument("--b", nargs="+", required=True, metavar="NAME FILE")
    ap.add_argument("--groups", nargs="*", default=[])
    ap.add_argument("--variants", default="")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    na, fa = args.a[0], args.a[1:]
    nb, fb = args.b[0], args.b[1:]
    vs = set(v for v in args.variants.split(",") if v) or None
    A, B = load(fa, vs), load(fb, vs)
    both = sorted(set(A) & set(B))
    print("%s: %d files, %d episodes on %d problems; %s: %d files, %d episodes on %d problems; in both: %d problems"
          % (na, len(fa), sum(map(len, A.values())), len(A), nb, len(fb), sum(map(len, B.values())), len(B), len(both)))
    fams = sorted({k[0] for k in both})
    groups = [(f, [f]) for f in fams]
    for g in args.groups:
        name, members = g.split("=")
        groups.append((name, [m for m in members.split(",") if m]))
    groups.append(("all", fams))
    rng = random.Random(args.seed)

    print("\nfeasible rate per problem, averaged over problems (each arm pooled over its runs)")
    print("%-11s %4s %9s | %-15s | %-15s %7s %6s | %-24s" % (
        "family", "prob", "runs A/B", "feasible@12", "feasible@20", "B - A", "se", "problems higher (A / B), p"))
    print("%-11s %4s %9s | %7s %7s | %7s %7s %7s %6s |" % ("", "", "", na[:7], nb[:7], na[:7], nb[:7], "@20", ""))
    for name, members in groups:
        keys = [k for k in both if k[0] in members]
        if not keys:
            continue
        rows = [(rate(A[k], 2), rate(B[k], 2), rate(A[k], 3), rate(B[k], 3)) for k in keys]
        m = [statistics.fmean(r[i] for r in rows) for i in range(4)]
        d = [r[3] - r[2] for r in rows]
        boots = []
        for _ in range(args.boot):
            boots.append(statistics.fmean(d[rng.randrange(len(d))] for _ in d))
        se = statistics.pstdev(boots)
        hi_a = sum(x < 0 for x in d)
        hi_b = sum(x > 0 for x in d)
        ra = sum(len(A[k]) for k in keys) / len(keys)
        rb = sum(len(B[k]) for k in keys) / len(keys)
        print("%-11s %4d %4.1f/%-4.1f | %7.3f %7.3f | %7.3f %7.3f %+7.3f %6.3f | %3d / %-3d p = %.3f" % (
            name, len(keys), ra, rb, m[0], m[1], m[2], m[3], m[3] - m[2], se, hi_a, hi_b, sign_p(hi_a, hi_b)))

    print("\nfeasible@20 per pair (count / episodes) on the compared problems")
    for arm, nm, files, data in (("A", na, fa, A), ("B", nb, fb, B)):
        for i, f in enumerate(files):
            cells = []
            for name, members in groups:
                eps = [e for k in both if k[0] in members for e in data[k] if e[0] == i]
                if eps:
                    cells.append("%s %d/%d" % (name, sum(e[3] for e in eps), len(eps)))
            print("%s %-8s file %d: %s" % (arm, nm[:8], i, "  ".join(cells)))
            print("    (%s)" % f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

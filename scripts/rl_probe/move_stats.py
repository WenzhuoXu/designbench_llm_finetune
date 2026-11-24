"""move_stats.py: move-level comparison of benchmark arms (rollout eval episodes, T=0), per family.
Per arm: share of turns refused or unread, share of applied moves that raise Phi, mean turns per episode, turns to the
first feasible design (episodes that reach one). Between arms: agreement of the turn-0 move on the same problem (same
formatted move), against the agreement of the first arm's own files with each other (its run-to-run floor).

usage: python move_stats.py --a NAME FILE... --b NAME FILE... [--variants v1,v2]
"""
import argparse
import collections
import itertools
import json
import statistics


def load(paths, variants):
    eps = []
    for i, p in enumerate(paths):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r.get("episode") or r
            if ep.get("crashed") or ep.get("feasible_start") or (variants and ep["variant"] not in variants):
                continue
            eps.append((i, int(r.get("member", 1)), ep))
    return eps


def stats(eps):
    out = collections.defaultdict(collections.Counter)
    ttf = collections.defaultdict(list)
    for _, _, ep in eps:
        v = ep["variant"]
        phi = ep.get("phi_start")
        out[v]["episodes"] += 1
        for t in ep.get("turns") or []:
            out[v]["turns"] += 1
            if t.get("fail") or t.get("phi_after") is None:
                out[v]["refused"] += 1
                continue
            out[v]["applied"] += 1
            if phi is not None and t["phi_after"] > phi:
                out[v]["phi_up"] += 1
            phi = t["phi_after"]
        if ep.get("first_feasible_turn") is not None:
            ttf[v].append(ep["first_feasible_turn"] + 1)
    return out, ttf


def first_moves(eps):
    """{(variant, problem): [turn-0 move of each (file, member) run]}"""
    fm = collections.defaultdict(dict)
    for i, m, ep in eps:
        turns = ep.get("turns") or []
        if turns:
            fm[(ep["variant"], ep["problem_id"])][(i, m)] = turns[0].get("move") or turns[0].get("action")
    return fm


def agree(xs, ys):
    pairs = [(x, y) for x in xs for y in ys]
    return sum(1 for x, y in pairs if x is not None and x == y) / len(pairs) if pairs else None


ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", required=True)
ap.add_argument("--b", nargs="+", required=True)
ap.add_argument("--variants", default=None)
a = ap.parse_args()
V = set(a.variants.split(",")) if a.variants else None
A, B = load(a.a[1:], V), load(a.b[1:], V)
(sa, ta), (sb, tb) = stats(A), stats(B)
fa, fb = first_moves(A), first_moves(B)
print("%-10s | %-24s | %-24s | %-15s | %s" % ("", "refused share  %s / %s" % (a.a[0], a.b[0]), "phi-up of applied",
                                             "turns/episode", "turn-0 move agreement: A~A, A~B"))
fams = sorted(set(sa) | set(sb))
for v in fams + ["all"]:
    vs = fams if v == "all" else [v]
    S = lambda s, k: sum(s[x][k] for x in vs)                                            # noqa: E731
    ra, rb = S(sa, "refused") / max(1, S(sa, "turns")), S(sb, "refused") / max(1, S(sb, "turns"))
    pa, pb = S(sa, "phi_up") / max(1, S(sa, "applied")), S(sb, "phi_up") / max(1, S(sb, "applied"))
    na, nb = S(sa, "turns") / max(1, S(sa, "episodes")), S(sb, "turns") / max(1, S(sb, "episodes"))
    aa, ab = [], []
    for k in set(fa) & set(fb):
        if k[0] not in vs:
            continue
        runs_a = list(fa[k].values())
        by_file = collections.defaultdict(list)
        for (i, _), mv in fa[k].items():
            by_file[i].append(mv)
        cross = [agree(x, y) for x, y in itertools.combinations(by_file.values(), 2)]
        cross = [c for c in cross if c is not None]
        if cross:
            aa.append(statistics.mean(cross))
        ab.append(agree(runs_a, list(fb[k].values())))
    print("%-10s | %6.3f %6.3f (%+.3f)  | %6.3f %6.3f (%+.3f)  | %5.1f %5.1f     | %s, %s  (%d problems)" % (
        v, ra, rb, rb - ra, pa, pb, pb - pa, na, nb,
        "%.2f" % statistics.mean(aa) if aa else "-", "%.2f" % statistics.mean(ab) if ab else "-", len(ab)))

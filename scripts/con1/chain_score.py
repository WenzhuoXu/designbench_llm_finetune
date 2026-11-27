"""con1 facetc link scores: the predicted LOCAL / REDISTRIBUTION blocks against the simulator's chain.

    python chain_score.py --queries queries.jsonl --chain chain_r1.jsonl --preds preds_Fc.jsonl [--label Fc] \
        [--out chain_scores.json]

Per chain family, over held-out-problem questions whose move was applied (truth) and re-simulated:
  local_dir_BA   direction balanced accuracy of the moved element's quantities (up / down / same)
  redis_dir_BA   direction BA over EVERY other element's main quantity; an element the answer does not list counts
                 as predicted 'same' (the answer bounds the rest); no-change baseline = 'same' everywhere
  top3_hit       share of the truly most-changed other elements (top 3 by relative change) among the answer's first 3
                 REDISTRIBUTION lines; baselines: nearest (the 3 elements topologically nearest the moved one:
                 frame3d2 node graph, grid2op substation graph, circuit2 element order), random (3 / n others,
                 expected), no change (0)
  chain_missing  share of applied answers with no readable REDISTRIBUTION block
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from targets import CHAIN_FAMILIES, _nat, chain_block, moved_elements, parse_chain  # noqa: E402


def ba(pairs):
    """Balanced accuracy over the true classes present: pairs of (true, predicted)."""
    tot, hit = collections.Counter(), collections.Counter()
    for t, p in pairs:
        tot[t] += 1
        hit[t] += t == p
    return sum(hit[c] / tot[c] for c in tot) / len(tot) if tot else float("nan"), dict(tot)


def topology(variant, user, move):
    """(distance function over element labels) from the observation text and the move."""
    m = re.match(r"\s*(\w+)\((.*)\)", move)
    verb, args = (m.group(1), [a.strip() for a in re.split(r",(?![^\[]*\])", m.group(2))]) if m else ("", [])
    ints = [int(a) for a in args if re.fullmatch(r"\d+", a)]
    if variant == "circuit2":
        k = ints[0] if ints else 0
        return lambda lab: abs(int(lab.split(" ", 1)[0]) - k)
    if variant == "frame3d2":
        ends = {mm.group(1): (int(mm.group(2)), int(mm.group(3)))
                for mm in re.finditer(r"(?m)^(m\d+) (\d+)-(\d+) ", user)}
        src = set()
        if verb in ("SET_RADIUS", "SET_MATERIAL", "INSTALL_MEMBER", "REMOVE_MEMBER") and ints:
            src = set(ends.get("m%d" % ints[0], ()))
        elif verb == "MOVE_JOINT" and ints:
            src = {ints[0] - 60}                           # free joints J63-J71 = nodes 3-11
        adj = collections.defaultdict(set)
        for a, b in ends.values():
            adj[a].add(b)
            adj[b].add(a)
    else:                                                  # grid2op: substation graph through the lines
        ends = {"line " + mm.group(1): (int(mm.group(2)), int(mm.group(3)))
                for mm in re.finditer(r"(?m)^  line (\d+): sub (\d+) - sub (\d+)", user)}
        gsub = {mm.group(1): int(mm.group(2)) for mm in re.finditer(r"(?m)^  (gen\d+) \w+ sub (\d+):", user)}
        src = set()
        if "LINE" in verb and ints:
            src = set(ends.get("line %d" % ints[0], ()))
        elif verb == "REDISPATCH" and len(ints) >= 2:
            src = {gsub.get("gen%d" % ints[0]), gsub.get("gen%d" % ints[1])} - {None}
        elif ints:
            src = {ints[0]}                                # SET_BUS / SPLIT_SUB / MERGE_SUB name the substation first
        adj = collections.defaultdict(set)
        for a, b in ends.values():
            adj[a].add(b)
            adj[b].add(a)
    dist = {s: 0 for s in src}
    frontier = list(src)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    big = 10 ** 6
    return lambda lab: min(dist.get(x, big) for x in ends.get(lab, (None,))) if lab in ends else big


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--label", default=None, help="predictor label to score (default: every label in --preds)")
    ap.add_argument("--pass", dest="pass_", default="zero")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    chains = {}
    for line in open(a.chain, encoding="utf-8"):
        x = json.loads(line)
        if x.get("split") == "test":
            chains[x["sid"]] = x
    qs = {}
    for line in open(a.queries, encoding="utf-8"):
        q = json.loads(line)
        if q["sid"] in chains:
            qs[q["sid"]] = q
    preds = collections.defaultdict(dict)
    for p in a.preds:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("pass") != a.pass_ or r["sid"] not in qs or (a.label and r["predictor"] != a.label):
                continue
            preds[r["predictor"]].setdefault((r["sid"], r["qi"]), r.get("text") or "")
    res = {}
    lines = []
    for lab, P in sorted(preds.items()):
        for v in CHAIN_FAMILIES:
            loc, red, red0 = [], [], []
            hit, hit_near, hit_rand, n3, miss, napp = 0.0, 0.0, 0.0, 0, 0, 0
            for sid, q in qs.items():
                if q["variant"] != v:
                    continue
                ch = chains[sid]
                for j, m in enumerate(q["moves"]):
                    if m["verdict"] == "refused" or (sid, j) not in P:
                        continue
                    after = ch["after"].get(m["move"])
                    if after is None:
                        continue
                    napp += 1
                    cb = chain_block(v, m["move"], ch["before"], after)
                    pc = parse_chain(P[(sid, j)])
                    if not pc["redis"] and "REDISTRIBUTION" not in P[(sid, j)]:
                        miss += 1
                    for (el, qty, dr, bn) in cb["local"]:
                        if dr in ("up", "down", "same"):
                            loc.append((dr, pc["local"].get((el, qty))))
                    pd = {el: d for el, d in pc["redis"]}
                    for el, qty, dr, rel in cb["ranked"]:
                        red.append((dr, pd.get(el, "same")))
                        red0.append((dr, "same"))
                    true3 = [c[0] for c in cb["redis"][:3]]
                    if true3:
                        n3 += 1
                        k = len(true3)
                        first3 = [el for el, _ in pc["redis"][:3]]
                        hit += len(set(first3) & set(true3)) / k
                        others = [c[0] for c in cb["ranked"]]
                        dist = topology(v, q["user"], m["move"])
                        near = sorted(others, key=lambda e: (dist(e), _nat(e)))[:3]
                        hit_near += len(set(near) & set(true3)) / k
                        hit_rand += min(3, len(others)) / max(1, len(others))
            f = {"applied": napp, "chain_missing": miss / max(1, napp)}
            f["local_dir_BA"], f["local_classes"] = ba(loc)
            f["redis_dir_BA"], f["redis_classes"] = ba(red)
            f["redis_dir_BA_nochange"], _ = ba(red0)
            f["top3_hit"] = hit / max(1, n3)
            f["top3_hit_nearest"] = hit_near / max(1, n3)
            f["top3_hit_random"] = hit_rand / max(1, n3)
            f["top3_questions"] = n3
            res["%s/%s" % (lab, v)] = f
            lines.append("%-6s %-9s applied %4d  local_dir_BA %.3f  redis_dir_BA %.3f (no change %.3f)  top3 %.3f "
                         "(nearest %.3f, random %.3f, no change 0)  chain missing %.3f" % (
                             lab, v, napp, f["local_dir_BA"], f["redis_dir_BA"], f["redis_dir_BA_nochange"],
                             f["top3_hit"], f["top3_hit_nearest"], f["top3_hit_random"], f["chain_missing"]))
    print("\n".join(lines))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

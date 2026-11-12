"""Stored transitions: counts, label base rates per family, repeats of an illegal move inside an episode,
and how well the one-step verdict is predicted from the move's own arguments alone (no state), by problem-wise CV."""
import collections, glob, json, os, re, sys
import numpy as np

d = sys.argv[1]
rows = []
files = sorted(glob.glob(os.path.join(d, "step_*.compact.jsonl")))
have = {os.path.basename(f)[:10] for f in files}
for f in sorted(glob.glob(os.path.join(d, "step_*.jsonl"))):
    if f.endswith(".compact.jsonl"):
        continue
    if os.path.basename(f)[:10] not in have:
        files.append(f)
nep = 0
groups = set()
for f in sorted(files):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r.pop("calls", None)
        ep = r["episode"]
        turns = ep.get("turns", [])
        nep += 1
        groups.add(r["group"])
        phi = ep.get("phi_start")
        first_dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
        seen_illegal = set()
        prev_fail = False
        for k, t in enumerate(turns):
            dead = first_dead is not None and k >= first_dead
            new = t.get("phi_after")
            if new is None:
                new = phi
            dphi = (new - phi) if (new is not None and phi is not None) else 0.0
            ans = str(t.get("answer"))
            m = re.search(r"<action>\s*([A-Za-z_]+)\s*\(([^)]*)\)", ans)
            verb = m.group(1) if m else "NONE"
            args = m.group(2) if m else ""
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", args)][:6]
            act = verb + "(" + args + ")"
            legal = not t.get("fail")
            rows.append(dict(fam=r["variant"], group=r["group"], key=r["key"], turn=k, dead=dead, legal=legal,
                             up=bool(dphi > 0), verb=verb, nums=nums, act=act,
                             repeat_illegal=(act in seen_illegal), after_fail=prev_fail,
                             fail=str(t.get("fail"))[:60]))
            if not legal:
                seen_illegal.add(act)
            prev_fail = not legal
            phi = new
live = [x for x in rows if not x["dead"]]
print("step files %d, episodes %d, problem groups %d, turns %d, live turns %d" % (len(files), nep, len(groups), len(rows), len(live)))
fams = sorted({x["fam"] for x in rows})
print("\n%-10s %7s %7s %9s %9s %12s %12s %14s %16s" % ("family", "live", "legal", "pot. up", "legal t0", "legal later", "legal after", "illegal that", "groups w/ mixed"))
print("%-10s %7s %7s %9s %9s %12s %12s %14s %16s" % ("", "", "", "", "", "", "an illegal", "repeat an", "legal / up"))
print("%-10s %7s %7s %9s %9s %12s %12s %14s %16s" % ("", "", "", "", "", "", "move", "earlier one", ""))
for f in fams + ["all"]:
    L = [x for x in live if f == "all" or x["fam"] == f]
    g = collections.defaultdict(list)
    for x in L:
        g[x["group"]].append(x)
    mixl = sum(1 for v in g.values() if 0 < np.mean([y["legal"] for y in v]) < 1)
    mixu = sum(1 for v in g.values() if 0 < np.mean([y["up"] for y in v]) < 1)
    ill = [x for x in L if not x["legal"]]
    af = [x["legal"] for x in L if x["after_fail"]]
    print("%-10s %7d %7.3f %9.3f %9.3f %12.3f %12s %14s %10d / %d of %d" % (
        f, len(L), np.mean([x["legal"] for x in L]), np.mean([x["up"] for x in L]),
        np.mean([x["legal"] for x in L if x["turn"] == 0]), np.mean([x["legal"] for x in L if x["turn"] > 0]),
        ("%.3f (n %d)" % (np.mean(af), len(af))) if af else "-",
        ("%.3f (n %d)" % (np.mean([x["repeat_illegal"] for x in ill]), len(ill))) if ill else "-", mixl, mixu, len(g)))

# verdict from the move's own arguments (verb + numbers), no state: problem-wise CV
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
except Exception as e:
    print("no sklearn", e)
    sys.exit(0)


def within_auc(y, p, g):
    per = []
    for b in np.unique(g):
        s = g == b
        pos, neg = p[s][y[s] == 1], p[s][y[s] == 0]
        if len(pos) and len(neg):
            per.append(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
    per = np.array(per)
    return (per.mean(), per.std(ddof=1) / np.sqrt(len(per)), len(per)) if len(per) > 1 else (float("nan"), float("nan"), len(per))


print("\nverdict predicted from the move alone (verb id, its numbers, turn index), boosted trees, 5-fold by problem")
print("%-10s %-14s %7s %9s %12s %s" % ("family", "label", "n", "base", "AUC overall", "AUC inside a problem (mean +- se over mixed problems)"))
for f in fams:
    L = [x for x in live if x["fam"] == f]
    verbs = sorted({x["verb"] for x in L})
    X = np.array([[verbs.index(x["verb"]), x["turn"]] + (x["nums"] + [np.nan] * 6)[:6] +
                  [(x["nums"][2] - x["nums"][0]) if len(x["nums"]) >= 4 else np.nan,
                   (x["nums"][3] - x["nums"][1]) if len(x["nums"]) >= 4 else np.nan] for x in L], dtype=float)
    g = np.unique([x["group"] for x in L], return_inverse=True)[1]
    for lab in ("legal", "up"):
        y = np.array([int(x[lab]) for x in L])
        if y.mean() in (0.0, 1.0) or min(y.sum(), len(y) - y.sum()) < 30:
            continue
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        wa = within_auc(y, p, g)
        print("%-10s %-14s %7d %9.3f %12.3f %.3f +- %.3f (%d problems)" % (f, lab, len(y), y.mean(), roc_auc_score(y, p), wa[0], wa[1], wa[2]))

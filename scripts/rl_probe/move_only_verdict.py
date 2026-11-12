"""How far does a verdict predictor that sees ONLY the move (verb, numbers; no state, no turn index) get,
on held-out problems?  (a) AUC inside a problem, all turns and first turn (same state) only;
(b) the same on the 11 probed steps only (the data size of the hidden-state value estimate);
(c) first move: pick, among the 8 first replies of a held-out problem, the one the predictor rates highest."""
import collections, glob, json, os, re, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "step_*.compact.jsonl")))
have = {os.path.basename(f)[:10] for f in files}
for f in sorted(glob.glob(os.path.join(d, "step_*.jsonl"))):
    if not f.endswith(".compact.jsonl") and os.path.basename(f)[:10] not in have:
        files.append(f)
rows = []
for f in sorted(files):
    step = int(os.path.basename(f)[5:10])
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        r.pop("calls", None)
        ep = r["episode"]
        turns = ep.get("turns", [])
        phi = ep.get("phi_start")
        ff = ep.get("first_feasible_turn")
        ok = ff is not None and ff <= 12
        first_dead = next((k for k, t in enumerate(turns) if "episode has ended" in str(t.get("fail"))), None)
        for k, t in enumerate(turns):
            if first_dead is not None and k >= first_dead:
                break
            new = t.get("phi_after")
            if new is None:
                new = phi
            dphi = (new - phi) if (new is not None and phi is not None) else 0.0
            m = re.search(r"<action>\s*([A-Za-z_]+)\s*\(([^)]*)\)", str(t.get("answer")))
            verb = m.group(1) if m else "NONE"
            args = m.group(2) if m else ""
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", args)][:6]
            legal = not t.get("fail")
            rows.append(dict(fam=r["variant"], group=r["group"], step=step, turn=k, legal=int(legal), up=int(dphi > 0),
                             gain=int(legal and dphi > 0), verb=verb, nums=nums, ok=int(ok)))
            phi = new
fams = sorted({x["fam"] for x in rows})


def feats(L):
    verbs = sorted({x["verb"] for x in L})
    return np.array([[verbs.index(x["verb"])] + (x["nums"] + [np.nan] * 6)[:6] +
                     [(x["nums"][2] - x["nums"][0]) if len(x["nums"]) >= 4 else np.nan,
                      (x["nums"][3] - x["nums"][1]) if len(x["nums"]) >= 4 else np.nan] for x in L], dtype=float)


def cv_pred(X, y, g):
    p = np.full(len(y), np.nan)
    k = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        if y[tr].min() == y[tr].max():
            p[te] = y[tr].mean()
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def within_auc(y, p, g, sel=None):
    if sel is not None:
        y, p, g = y[sel], p[sel], g[sel]
    per = []
    for b in np.unique(g):
        s = g == b
        pos, neg = p[s][y[s] == 1], p[s][y[s] == 0]
        if len(pos) and len(neg):
            per.append(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
    per = np.array(per)
    return "%.3f +- %.3f (%3d)" % (per.mean(), per.std(ddof=1) / np.sqrt(len(per)), len(per)) if len(per) > 2 else "        -         "


print("(a,b) AUC inside a held-out problem of a predictor that sees only the move (verb + numbers); mean +- se (mixed problems)")
print("%-10s %-6s %8s %22s %22s %26s" % ("family", "label", "base", "all steps, all turns", "all steps, first turn", "11 probed steps, all turns"))
pick = {}
for f in fams:
    L = [x for x in rows if x["fam"] == f]
    X = feats(L)
    g = np.unique([x["group"] for x in L], return_inverse=True)[1]
    turn = np.array([x["turn"] for x in L])
    probed = np.array([x["step"] % 10 == 0 for x in L])
    for lab in ("legal", "up", "gain"):
        y = np.array([x[lab] for x in L])
        if min(y.sum(), len(y) - y.sum()) < 30:
            continue
        p = cv_pred(X, y, g)
        pp = np.full(len(y), np.nan)
        if min(y[probed].sum(), probed.sum() - y[probed].sum()) >= 20:
            pp[probed] = cv_pred(X[probed], y[probed], g[probed])
            c = within_auc(y, pp, g, probed)
        else:
            c = "        -         "
        print("%-10s %-6s %8.3f %22s %22s %26s" % (f, lab, y.mean(), within_auc(y, p, g), within_auc(y, p, g, turn == 0), c))
        if lab == "gain":
            pick[f] = (L, y, p, g, turn)

print("\n(c) first move of held-out problems: share that is legal and raises the potential")
print("%-10s %8s %14s %18s %14s %14s" % ("family", "problems", "one sample", "predictor's pick", "best of 8", "episode ok: sample -> pick"))
tot = collections.defaultdict(list)
for f in fams:
    if f not in pick:
        continue
    L, y, p, g, turn = pick[f]
    okv = np.array([x["ok"] for x in L])
    a, b, c, e1, e2 = [], [], [], [], []
    for u in np.unique(g):
        s = np.where((g == u) & (turn == 0))[0]
        if len(s) < 2:
            continue
        best = s[p[s] == p[s].max()]
        a.append(y[s].mean()); b.append(y[best].mean()); c.append(y[s].max())
        e1.append(okv[s].mean()); e2.append(okv[best].mean())
    print("%-10s %8d %14.3f %18.3f %14.3f %10.3f -> %.3f" % (f, len(a), np.mean(a), np.mean(b), np.mean(c), np.mean(e1), np.mean(e2)))
    for k_, v in (("a", a), ("b", b), ("c", c), ("e1", e1), ("e2", e2)):
        tot[k_] += v
print("%-10s %8d %14.3f %18.3f %14.3f %10.3f -> %.3f" % ("all", len(tot["a"]), np.mean(tot["a"]), np.mean(tot["b"]), np.mean(tot["c"]), np.mean(tot["e1"]), np.mean(tot["e2"])))

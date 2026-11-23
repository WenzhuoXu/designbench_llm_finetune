import json, sys, collections
def load(p):
    d = {}
    for l in open(p, encoding='utf-8'):
        r = json.loads(l); e = r['episode']
        ft = e.get('first_feasible_turn')
        d[(r['variant'], r['problem_id'], r['group'].split('/')[-1])] = ft is not None and ft <= 20
    return d
A = load(sys.argv[1]); B = load(sys.argv[2])
probs = collections.defaultdict(dict)
for (v, p, run), f in A.items(): probs[(v, p)]['a' + run] = f
for (v, p, run), f in B.items(): probs[(v, p)]['b' + run] = f
fam = collections.defaultdict(lambda: collections.Counter())
for (v, p), runs in probs.items():
    vals = list(runs.values())
    c = fam[v]; c['problems'] += 1; c['runs'] += len(vals); c['feas'] += sum(vals)
    c['all4_agree'] += (len(set(vals)) == 1)
    c['a_r1r2_agree'] += (runs.get('ar1') == runs.get('ar2'))
    c['a_vs_b_mean_diff'] += abs((runs.get('ar1', 0) + runs.get('ar2', 0)) - (runs.get('br1', 0) + runs.get('br2', 0)))
for v, c in sorted(fam.items()):
    print(v, dict(c))

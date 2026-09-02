import json, glob, random, statistics as st, sys
sys.path.insert(0,'/ocean/projects/mch250030p/wxu7/llm_finetune')
sys.path.insert(0,'/ocean/projects/mch250030p/wxu7/DesignBench')
from scripts.search_ladder import make_truss, fea, param_bounds, candidate_actions, step
from llm_finetune.training.rl.posterior.potential import program_from_truss_spec

def one(f):
    spec=json.load(open(f)); goals=spec.get("goals",{}) or {}
    truss=make_truss(spec); s0=fea(truss,goals)
    prog=program_from_truss_spec(spec, initial_mass=s0.get("mass"))
    if prog.is_feasible(s0): return None
    bounds=param_bounds(spec)
    acts=candidate_actions(truss,bounds,macros=False,
                           target_fos_b=prog.limit_for("fos_buckling") or 1.5,
                           target_fos_y=prog.limit_for("fos_yielding") or 1.5)
    cand=[]
    for a in acts[:50]:
        r=step(truss,goals,a,prog)
        if r is None: continue
        nt,ns=r
        cand.append((prog.objective(ns), prog.total_violation(ns,tau=0.05)))
    if len(cand)<8: return None
    bv=prog.total_violation(s0,tau=0.05); bo=prog.objective(s0)
    phi=[-o-5.0*v for o,v in cand]
    i=max(range(len(cand)), key=lambda k: phi[k])
    o,v=cand[i]
    return {"pid":spec["problem_id"],"n":len(cand),
            "raises_violation": bool(v>bv+1e-12), "cuts_mass": bool(o<bo-1e-12),
            "spread_obj": max(c[0] for c in cand)-min(c[0] for c in cand),
            "spread_vio": 5.0*(max(c[1] for c in cand)-min(c[1] for c in cand))}

if __name__=="__main__":
    files=sorted(glob.glob("/ocean/projects/mch250030p/wxu7/DesignBench/data/problems_hard/*.json"))
    random.seed(0); sample=random.sample(files, 60)
    from multiprocessing import Pool
    with Pool(16) as p: out=[r for r in p.map(one, sample) if r]
    print(f"infeasible states examined: {len(out)}")
    print(f"  Phi-argmax RAISES violation (trades strength away): {sum(r['raises_violation'] for r in out)}/{len(out)}")
    print(f"  Phi-argmax CUTS mass:                               {sum(r['cuts_mass'] for r in out)}/{len(out)}")
    print(f"  median spread across candidates, objective term : {st.median([r['spread_obj'] for r in out]):.4f}")
    print(f"  median spread across candidates, alpha*violation: {st.median([r['spread_vio'] for r in out]):.4f}")
    dom=sum(1 for r in out if r['spread_obj']>r['spread_vio'])
    print(f"  states where the MASS term dominates the spread : {dom}/{len(out)}")

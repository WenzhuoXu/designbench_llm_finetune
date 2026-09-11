"""Smoke test: can we load a hard problem, run FEA, and time it?"""
import sys, time, json, math
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals, normalize_action
from validation.truss_executor import execute_grammar_action

probdir = DESIGNBENCH / "data" / "problems_hard"
files = sorted(probdir.glob("*.json"))
print("problems_hard files:", len(files))
spec = json.load(open(files[0]))
print("problem_id:", spec.get("problem_id"))
print("spec top-level keys:", sorted(spec.keys()))
opt = spec.get("optimization") or {}
print("optimization keys:", sorted(opt.keys()))
print("goals:", json.dumps({k: v for k, v in opt.items() if not isinstance(v, (dict, list))}, indent=None)[:400])

truss, goals = _load_truss_and_goals(spec)
print("goals object:", goals)
print("n members:", len(truss.members))

t0 = time.time()
st = _analyze_truss(truss, goals)
t1 = time.time()
print("FEA time: %.2f ms" % ((t1 - t0) * 1000))
print("state keys:", sorted(st.keys()))
print("state:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in st.items() if not isinstance(v, (dict, list))})[:700])

# per-member margins available?
m0 = truss.members[0]
print("member attrs:", [a for a in dir(m0) if 'fos' in a.lower() or 'shape' in a.lower()][:12])
print("member0 fos_b/fos_y:", getattr(m0, 'fos_buckling', None), getattr(m0, 'fos_yielding', None))
print("all member fos_b:", [round(float(getattr(m, 'fos_buckling', float('nan'))), 3) for m in truss.members])

# timing loop
import copy
t0 = time.time()
N = 200
for i in range(N):
    _analyze_truss(truss, goals)
t1 = time.time()
print("mean FEA over %d calls: %.3f ms" % (N, (t1 - t0) / N * 1000))

# one grammar action
nt = copy.deepcopy(truss)
ok = execute_grammar_action(nt, normalize_action(nt, "SCALE_PARAM(0, radius, 1.10)"))
print("action applied:", ok)
print("new state mass/fos_b:", _analyze_truss(nt, goals).get("mass"), _analyze_truss(nt, goals).get("fos_buckling"))

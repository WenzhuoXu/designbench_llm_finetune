"""
The control the 'model plans, tool sizes' claim needs: does the model contribute anything?

Held-out result: letting the model choose members and replacing its scale factor with the
fully-stressed value beats the model alone by +4.42pp (p=0.0094). That was read as the model
planning well and the tool library parameterising. The reading only stands if the model's
target choice is doing work. If applying the same fully-stressed rule to EVERY member, with no
model in the loop at all, does as well or better, then the model is decoration and the honest
conclusion is that a procedural optimiser beats the agent outright.

This runs that arm on exactly the same 430 held-out problems, through the same apply-and-
analyse path and the same step budget, and pairs it against the saved free/sized outcomes.
No API calls.
"""
import os, sys, json, math, time
from pathlib import Path
from multiprocessing import Pool
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
import api_grammar_2x2 as H
from probe15_matrix import true_fos
from probe27_decompose import fsd_factor
from probe26_presentation import sign_test

SAVED = PROJECT / "results/api_guidance/decompose_heldout.jsonl"
MAX_STEPS = 4


def fsd_episode(spec):
    """Same loop, no model: every member takes its fully-stressed factor each turn."""
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    for _ in range(MAX_STEPS):
        if state.get("is_feasible"):
            return True
        tf = true_fos(truss)
        cur = truss
        ok = True
        for i in range(len(truss.members)):
            f = fsd_factor(tf[i][0], tf[i][1], gb, gy)
            if abs(f - 1.0) < 1e-6:
                continue
            nt = H.apply_candidate(cur, "SCALE_PARAM(%d, radius, %.4f)" % (i, f))
            if nt is None:
                ok = False
                break
            cur = nt
        if not ok or cur is truss:
            break
        truss = cur
        state = _analyze_truss(truss, goals)
    return bool(state.get("is_feasible"))


def _w(f):
    try:
        spec = json.load(open(f))
        return (spec.get("problem_id", ""), 1.0 if fsd_episode(spec) else 0.0)
    except Exception:
        return None


if __name__ == "__main__":
    saved = [json.loads(l) for l in open(SAVED)]
    by = {}
    for r in saved:
        by.setdefault(r["arm"], {})[r["problem_id"]] = 1.0 if r["feasible"] else 0.0
    files = [str(f) for f in sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[150:580]]
    t0 = time.time()
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, files, chunksize=2) if r]
    fsd = dict(got)
    pids = sorted(set(fsd) & set(by.get("free", {})) & set(by.get("sized", {})))
    print("%d problems paired across all three arms, %.0fs\n" % (len(pids), time.time() - t0))
    rate = {"free": sum(by["free"][p] for p in pids) / len(pids),
            "sized": sum(by["sized"][p] for p in pids) / len(pids),
            "fsd_all": sum(fsd[p] for p in pids) / len(pids)}
    print("  free (model alone)                 %.4f" % rate["free"])
    print("  sized (model targets + FSD sizing) %.4f" % rate["sized"])
    print("  fsd_all (no model at all)          %.4f" % rate["fsd_all"])
    print()
    for a, b in (("free", "sized"), ("free", "fsd_all"), ("fsd_all", "sized")):
        u, d, p = sign_test([by.get(a, fsd)[x] if a != "fsd_all" else fsd[x] for x in pids],
                            [by.get(b, fsd)[x] if b != "fsd_all" else fsd[x] for x in pids])
        print("  %-8s -> %-8s  %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (a, b, rate[b] - rate[a], u, d, p))
    print("\n  The claim that the model's targeting contributes requires sized > fsd_all.")

"""Count FEA analyses per episode for each arm.

The report matches the arms on horizon and model calls. It never states the analysis
budget. This counts it. Every module namespace that holds a by-name reference to the
counted functions is rebound, so the count is complete regardless of import style.
"""
import os, sys, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for _p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import api_grammar_2x2 as H
import planner01, planner03, planner05, planner13
from llm_finetune.envs import truss_env
from probe15_matrix import true_fos as _TF

_TL = threading.local()


def _bump(name):
    d = getattr(_TL, "d", None)
    if d is None:
        d = _TL.d = {}
    d[name] = d.get(name, 0) + 1


def wrap(orig, name):
    def f(*a, **kw):
        _bump(name)
        return orig(*a, **kw)
    f.__wrapped_orig__ = orig
    return f


# rebind in EVERY module namespace that holds the original object
def rebind(orig, name):
    new = wrap(orig, name)
    n = 0
    for m in list(sys.modules.values()):
        if m is None:
            continue
        try:
            items = list(vars(m).items())
        except Exception:
            continue
        for k, v in items:
            if v is orig:
                try:
                    setattr(m, k, new)
                    n += 1
                except Exception:
                    pass
    return new, n


_, n1 = rebind(truss_env._analyze_truss, "analyze")
_, n2 = rebind(_TF, "true_fos")
_, n3 = rebind(H.apply_candidate, "apply_candidate")
_, n4 = rebind(H.call, "model_call")
print("rebound analyze in %d ns, true_fos in %d, apply_candidate in %d, call in %d"
      % (n1, n2, n3, n4), flush=True)


def work(t):
    spec, arm, region, token, k, ms = t
    _TL.d = {}
    t0 = time.time()
    try:
        if arm == "plain_llm":
            r = planner01.episode(spec, "llm", region, token, k, ms)
        else:
            r = planner13.episode(spec, arm, region, token, k, ms)
    except Exception as e:
        r = {"problem_id": spec.get("problem_id"), "feasible": False,
             "err": "%s: %s" % (type(e).__name__, str(e)[:150])}
    r["arm"] = arm
    r.update({("n_" + a): b for a, b in getattr(_TL, "d", {}).items()})
    r["secs"] = round(time.time() - t0, 1)
    return r


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[150:150 + n]
    specs = [json.load(open(f)) for f in files]
    arms = ("planner", "fsd", "plain_llm")
    tasks = [(s, a, "us-west-2", token, 8, 8) for s in specs for a in arms]
    print("problems 150-%d | arms %s | episodes %d" % (150 + n, ",".join(arms), len(tasks)),
          flush=True)
    out = []
    outp = PROJECT / "results/api_guidance/verify_analysis_cost.jsonl"
    with ThreadPoolExecutor(max_workers=12) as ex, open(outp, "w") as fh:
        for r in ex.map(work, tasks):
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()

    print("\nper-episode cost, mean over %d problems\n" % n)
    print("  %-10s %9s %9s %9s %9s %8s" %
          ("arm", "analyses", "true_fos", "apply", "modelcall", "secs"))
    for a in arms:
        rs = [r for r in out if r["arm"] == a]
        if not rs:
            continue
        def m(key):
            return sum(r.get(key, 0) for r in rs) / len(rs)
        print("  %-10s %9.1f %9.1f %9.1f %9.2f %8.1f" %
              (a, m("n_analyze"), m("n_true_fos"), m("n_apply_candidate"),
               m("n_model_call"), m("secs")))
    print("\n  feasibility on this subset:")
    for a in arms:
        rs = [r for r in out if r["arm"] == a]
        print("    %-10s %.3f  (%d/%d)" %
              (a, sum(1 for r in rs if r.get("feasible")) / len(rs),
               sum(1 for r in rs if r.get("feasible")), len(rs)))

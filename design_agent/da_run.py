"""
Phase 0 + prototype planner. Procedural arms first (no model, CPU only), then an optional
Bedrock planner arm on the same problems.

  python da_run.py --n 60                    # phase 0, procedural
  python da_run.py --n 30 --llm              # + planner arm
"""
from __future__ import annotations
import sys, os, json, math, argparse, time, statistics, collections, re
from math import comb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune",
          "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts",
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)

from da_domain import get_domain
import da_tools as T

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


# ---------------------------------------------------------------- procedural arms

def step_base(dom, st, param):
    """Enumerator: single-element and global uniform scaling on a coarse grid."""
    n = dom.n_elements(st)
    best, bv = None, T.phi_rho(dom, st)
    for f in (0.85, 1.1, 1.25, 1.5, 2.0):
        for i in list(range(n)) + [None]:
            nx = dom.clone(st)
            idx = range(n) if i is None else [i]
            for j in idx:
                b = dom.get(st, j, param)
                if b:
                    dom.set(nx, j, param, b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
    return best


def step_op(dom, st, param, make_op, k=None, probe_delta=0.10):
    """One planner-shaped move: pick support, get an operator, let the inner loop choose the
    target margin. make_op(dom, st, elems) -> op."""
    n = dom.n_elements(st)
    elems = list(range(n)) if k is None else T.worst_by_margin(dom, st, k)
    op = make_op(dom, st, elems)
    build = lambda target: T.apply_op(dom, st, op, list(range(n)), param, target)
    x, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
    return x if v > T.phi_rho(dom, st) else None


def induced_op(dom, st, elems, delta=0.10):
    return T.FIT_OP(T.PROBE(dom, st, elems, "r", delta))


def step_efficient(dom, st, param, k=6):
    """Probe everything, spend on the elements where margin is cheapest per unit budget."""
    n = dom.n_elements(st)
    probes = T.PROBE(dom, st, list(range(n)), param)
    op = T.FIT_OP(probes)
    S = T.EFFICIENT_SUPPORT(dom, st, probes, k, param)
    keep = set(S) | set(T.surplus_set(dom, st, 2.0))
    build = lambda target: T.apply_op(dom, st, op, sorted(keep), param, target)
    x, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
    return x if v > T.phi_rho(dom, st) else None


def installed_op(dom, st, elems):
    return T.INSTALLED_OP(3.0)


def run_arm(spec, arm, max_steps=20, sim_budget=None):
    dom = get_domain("truss")
    st = dom.load(spec)
    param = dom.params[0]
    for _ in range(max_steps):
        if dom.feasible(st):
            return 1, dom.calls
        if sim_budget and dom.calls >= sim_budget:
            break
        if arm == "base":
            nx = step_base(dom, st, param)
        elif arm == "installed":
            nx = step_op(dom, st, param, installed_op)
        elif arm == "induced":
            nx = step_op(dom, st, param, induced_op)
        elif arm == "induced_k5":
            nx = step_op(dom, st, param, induced_op, k=5)
        elif arm == "efficient":
            nx = step_efficient(dom, st, param)
        else:
            raise KeyError(arm)
        if nx is None:
            break
        st = nx
    return (1 if dom.feasible(st) else 0), dom.calls


# ---------------------------------------------------------------- planner arm (Bedrock)

PLAN_SYS = """You are planning one move in an iterative structural design loop.

You do NOT choose any numbers. You choose STRUCTURE. An inner numerical optimiser fills every
numeric slot for you by probing the simulator.

Available tools:
  worst_by_margin(k)      the k elements with the smallest safety margin
  deficit_set()           every element below the requirement
  surplus_set(thr)        elements carrying more material than they need
  PROBE(S)                measure each element's local response, 2 simulator calls each
  FIT_OP(probes)          turn probes into an element-wise sizing operator
  apply_op(op, S, ?)      move S toward a target margin -- the ? is filled for you
  TRIM(?)                 shrink over-built elements -- the ? is filled for you

Reply with ONE line of the form:
  PLAN: support=<worst_by_margin:K | deficit | all>  op=<induced|installed>  then=<none|trim>
Nothing else."""

PLAN_RE = re.compile(r"support=(worst_by_margin:\d+|deficit|all)\s+op=(induced|installed)\s+then=(none|trim)")


def plan_prompt(dom, st):
    em = dom.element_margins(st) or []
    rows = "\n".join("  E%-3d margin %.3f" % (i, m) for i, m in enumerate(em))
    return ("Elements and their own margins (>=1.0 satisfies):\n" + rows +
            "\n\nGlobal margins: %s\nBudget ratio: %.3f (<=1.0 satisfies)\n\nOne PLAN line."
            % (", ".join("%.3f" % m for m in dom.global_margins(st)), dom.budget_ratio(st)))


def run_planner(spec, model, region, token, max_steps=20, sim_budget=None):
    import api_grammar_2x2 as H
    dom = get_domain("truss")
    st = dom.load(spec)
    param = dom.params[0]
    n_llm = 0
    for _ in range(max_steps):
        if dom.feasible(st):
            return 1, dom.calls, n_llm
        if sim_budget and dom.calls >= sim_budget:
            break
        txt = H.call([{"role": "user", "content": [{"text": plan_prompt(dom, st)}]}],
                     PLAN_SYS, model, region, token, max_tokens=120, temperature=0.7)
        n_llm += 1
        m = PLAN_RE.search(txt or "")
        sup, opk, then = (m.group(1), m.group(2), m.group(3)) if m else ("all", "induced", "none")
        k = int(sup.split(":")[1]) if sup.startswith("worst") else None
        elems = (T.deficit_set(dom, st) if sup == "deficit"
                 else list(range(dom.n_elements(st))) if sup == "all"
                 else T.worst_by_margin(dom, st, k))
        op = (T.FIT_OP(T.PROBE(dom, st, elems, param)) if opk == "induced"
              else T.INSTALLED_OP(3.0))
        build = lambda target: T.apply_op(dom, st, op, list(range(dom.n_elements(st))), param, target)
        nx, v, _ = T.inner_optimise(dom, st, build, {"target": (1.0, 1.8)}, budget=18)
        if then == "trim":
            cand = T.TRIM(dom, nx, param, 2.0)
            if T.phi_rho(dom, cand) > v:
                nx = cand
        if T.phi_rho(dom, nx) <= T.phi_rho(dom, st):
            break
        st = nx
    return (1 if dom.feasible(st) else 0), dom.calls, n_llm


# ---------------------------------------------------------------- driver

def mcnemar(a, b, keys):
    hi = sum(1 for k in keys if a[k] and not b[k])
    lo = sum(1 for k in keys if b[k] and not a[k])
    d = hi + lo
    p = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
    return hi, lo, 100 * (hi - lo) / len(keys), p


def _w(t):
    spec, arm, bud = t
    try:
        s, c = run_arm(spec, arm, sim_budget=bud)
        return spec["problem_id"], arm, s, c
    except Exception:
        return spec["problem_id"], arm, None, 0


def _wl(t):
    spec, model, region, token, bud, rep = t
    try:
        s, c, l = run_planner(spec, model, region, token, sim_budget=bud)
        return spec["problem_id"], "planner", s, c, l
    except Exception as e:
        print("PLANNER FAIL:", type(e).__name__, str(e)[:160], flush=True)
        return spec["problem_id"], "planner", None, 0, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--reps", type=int, default=1, help="planner draws per problem")
    ap.add_argument("--sim-budget", type=int, default=0, help="cap simulator calls per episode; 0 = uncapped")
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    a = ap.parse_args()
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:a.n]]

    ARMS = ["base", "installed", "induced", "efficient"]
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.procs) as pool:
        res = pool.map(_w, [(s, arm, a.sim_budget or None) for s in specs for arm in ARMS], chunksize=1)
    solved = collections.defaultdict(dict); calls = collections.defaultdict(list)
    for pid, arm, s, c in res:
        if s is not None:
            solved[arm][pid] = s; calls[arm].append(c)
    keys = sorted(set.intersection(*[set(solved[x]) for x in ARMS]))
    LAB = {"base": "enumerator, coarse grid            [control]",
           "installed": "one global exponent (closed form) ",
           "induced": "PROBE + FIT_OP, all elements      [induction]",
           "efficient": "PROBE -> spend where margin is cheapest"}
    print("PHASE 0 -- procedural, no model. n=%d, %.0fs" % (len(keys), time.time() - t0))
    for arm in ARMS:
        print("  %-11s solved %.3f   sim/ep %5.0f   %s"
              % (arm, statistics.mean(solved[arm][k] for k in keys),
                 statistics.mean(calls[arm]), LAB[arm]))
    print()
    for x, y in (("installed", "base"), ("induced", "base"),
                 ("induced", "installed"), ("efficient", "induced"),
                 ("efficient", "installed")):
        hi, lo, d, p = mcnemar(solved[x], solved[y], keys)
        print("  %-11s vs %-11s disc %3d-%-3d  %+5.1fpp  p=%.2e" % (x, y, hi, lo, d, p))

    if a.llm:
        tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
            open(os.path.expanduser("~/.bedrock_token")).read().strip()
        print("\nPLANNER arm (Bedrock) ...", flush=True)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=16) as ex:
            pres = list(ex.map(_wl, [(s, a.model, a.region, tok, a.sim_budget or None, r)
                                     for r in range(a.reps) for s in specs]))
        agg = collections.defaultdict(list)
        for pid, _, s, _, _ in pres:
            if s is not None:
                agg[pid].append(s)
        # majority over reps, so the planner is scored at the same granularity as the
        # deterministic arms rather than on a single stochastic draw
        ps = {pid: int(sum(v) * 2 >= len(v)) for pid, v in agg.items()}
        if a.reps > 1:
            per = [statistics.mean(v[i] for v in agg.values()) for i in range(a.reps)]
            print("  planner per-draw rates: %s | spread %.1fpp"
                  % (", ".join("%.3f" % x for x in per), 100 * (max(per) - min(per))))
        pc = [c for _, _, s, c, _ in pres if s is not None]
        pl = [l for _, _, s, _, l in pres if s is not None]
        kk = sorted(set(ps) & set(keys))
        print("  planner     solved %.3f   sim/ep %5.0f   llm/ep %4.1f   (%.0fs, n=%d)"
              % (statistics.mean(ps[k] for k in kk), statistics.mean(pc),
                 statistics.mean(pl), time.time() - t1, len(kk)))
        for y in ("base", "installed", "induced"):
            hi, lo, d, p = mcnemar(ps, solved[y], kk)
            print("  planner     vs %-11s disc %3d-%-3d  %+5.1fpp  p=%.2e" % (y, hi, lo, d, p))

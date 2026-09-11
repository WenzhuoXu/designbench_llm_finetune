"""
EXPERIMENT 21 -- precondition for the training method: is a supplied physics primitive USED?

The model asserts FOS ~ r^4. Measured per-element elasticity d log(own margin)/d log(own
radius): median 2.32, and BELOW 1.5 for 45.7% of elements -- so it believes one global law
where the truth is element-dependent and often nearly flat. That quantity costs 2 FEA per
element (~45 ms/turn), so it can be handed to the model for free, and it can be labelled for
free if we want to TRAIN a model to predict it.

Before building that training stage, ask whether the model can use the number at all. The
address result showed it consumes per-element ATTRIBUTION faithfully. Per-element SENSITIVITY
is a different quantity and has never been supplied.

Arms (same problems, paired, identical prompt scaffolding, only the table's columns differ):
  A  baseline      id | FOS_b | FOS_y | r | t                    [the current interface]
  B  + elasticity  ... | dlnFOS/dlnr  (measured by finite difference, per element)
  C  + required    ... | dlnFOS/dlnr | factor needed to reach the goal (the inversion done)
  D  + required    id | FOS_b | FOS_y | r | t | required factor   [inversion WITHOUT elasticity]

B vs A  : can it use the primitive?
C vs B  : does it also need the inversion performed for it?
D vs B  : is the primitive worth anything once the answer is supplied?
D vs A  : upper bound -- this is close to handing over the closed form.

If B ~ A, the model cannot consume sensitivity and stage one of the method must change.
If B >> A, elasticity is usable and worth training a model to predict.
"""
import os, sys, json, math, copy, argparse, time, re, statistics, collections
from math import comb, sqrt, erf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(P), str(DB), str(P / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H

DELTA = 0.10


def prm(m):
    return getattr(getattr(m, "shape", None), "_params", None) or {}


def setr(m, v):
    d = getattr(getattr(m, "shape", None), "_params", None)
    if d is not None:
        d["r"] = v


def own(m, gb, gy):
    def v(a):
        try:
            x = float(getattr(m, a, float("inf")) or float("inf"))
        except Exception:
            x = float("inf")
        return x if x == x else float("inf")
    return min(v("fos_buckling") / gb, v("fos_yielding") / gy)


def elasticities(truss, goals, gb, gy, lo, hi):
    """d log(own margin) / d log(own radius) per element, by central difference."""
    base = [prm(m).get("r") for m in truss.members]
    m0 = [own(m, gb, gy) for m in truss.members]
    out = []
    for i in range(len(truss.members)):
        if not base[i] or base[i] <= 0 or not math.isfinite(m0[i]) or m0[i] <= 0:
            out.append(None); continue
        es = []
        for s in (+DELTA, -DELTA):
            nt = copy.deepcopy(truss)
            setr(nt.members[i], min(hi, max(lo, base[i] * math.exp(s))))
            try:
                _analyze_truss(nt, goals)
            except Exception:
                continue
            m1 = own(nt.members[i], gb, gy)
            if math.isfinite(m1) and m1 > 0:
                es.append((math.log(m1) - math.log(m0[i])) / s)
        out.append(sum(es) / len(es) if es else None)
    return out, m0


def table(truss, goals, gb, gy, lo, hi, arm):
    def f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "     n/a"
        if not math.isfinite(v):
            return "     inf"
        return f"{v:8.2f}" if abs(v) < 1e5 else f"{v:8.1e}"
    el = m0 = None
    if arm in ("B", "C", "D"):
        el, m0 = elasticities(truss, goals, gb, gy, lo, hi)
    head = "  id    FOS_buckling    FOS_yielding         r         t"
    if arm in ("B", "C"):
        head += "   dlnFOS/dlnr"
    if arm in ("C", "D"):
        head += "   factor_needed"
    rows = ["", "Per-member state (r = outer radius, t = wall thickness, m):", head]
    for i, m in enumerate(truss.members):
        sp = prm(m)
        r_, t_ = sp.get("r"), sp.get("t")
        def fb_(a):
            try:
                x = float(getattr(m, a, float("inf")) or float("inf"))
            except Exception:
                x = float("inf")
            return x if x == x else float("inf")
        line = (f"  M{i:<4d}{f(fb_('fos_buckling'))}    {f(fb_('fos_yielding'))}  "
                f"{(f'{float(r_):.5f}' if r_ is not None else '  n/a  ')}  "
                f"{(f'{float(t_):.5f}' if t_ is not None else '  n/a  ')}")
        if arm in ("B", "C"):
            line += "   " + (f"{el[i]:8.2f}" if el and el[i] is not None else "     n/a")
        if arm in ("C", "D"):
            need = "     n/a"
            if el and el[i] is not None and m0 and m0[i] and m0[i] > 0 and abs(el[i]) > 0.15:
                fac = (1.0 / m0[i]) ** (1.0 / el[i])
                need = f"{min(2.0, max(0.7, fac)):8.3f}"
            elif m0 and m0[i] and m0[i] >= 1.0:
                need = "   0.700"
            line += "   " + need
        rows.append(line)
    rows.append("A member with FOS far above the required 1.5 is carrying more "
                "material than it needs.")
    if arm in ("B", "C"):
        rows.append("dlnFOS/dlnr is the MEASURED local elasticity of that member's own factor "
                    "of safety with respect to its own radius: scaling r by a factor f "
                    "multiplies that member's FOS by approximately f raised to this power.")
    if arm in ("C", "D"):
        rows.append("factor_needed is the scale factor that would bring that member to the "
                    "required factor of safety.")
    return "\n".join(rows)


def episode(spec, arm, model, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    sp = ((spec.get("optimization") or {}).get("shape_params") or {})
    rb = sp.get("r") or {}
    lo = float(rb.get("min", 1e-6)); hi = float(rb.get("max", 1e9))
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    B = float(goals.get("maximum_mass", float("inf")))
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history = []
    feasible = False
    nsim = 0
    for _ in range(max_steps):
        tbl = table(truss, goals, gb, gy, lo, hi, arm)
        obs = DP.format_eval_result(state) + tbl
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + (tbl if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["o"]}]})
        text = H.call(convo, system, model, region, token)
        cands = H.parse_candidates(text, k, True)
        if not cands:
            history.append({"t": f"No <action> parsed. Emit exactly {k} lines.", "o": obs})
            continue
        scored = []
        for c in cands:
            nt = H.apply_candidate(truss, c)
            if nt is None:
                continue
            try:
                st = _analyze_truss(nt, goals)
            except Exception:
                continue
            nsim += 1
            scored.append((nt, st))
        if not scored:
            history.append({"t": text.strip(), "o": obs})
            continue
        def key(t):
            st = t[1]
            fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
            ms = float(st.get("mass", 9e9) or 9e9)
            return min(fb / gb, fy / gy) - (0.5 if ms > B else 0.0)
        truss, state = max(scored, key=key)
        history.append({"t": text.strip(),
                        "o": DP.format_eval_result(state)
                             + table(truss, goals, gb, gy, lo, hi, arm)})
        if state.get("is_feasible"):
            feasible = True
            break
    return {"problem_id": spec.get("problem_id"), "arm": arm, "model": model,
            "feasible": int(feasible), "sims": nsim}


def work(t):
    spec, arm, model, region, token, k, ms = t
    try:
        return episode(spec, arm, model, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "model": model,
                "err": str(e)[:150]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(P / "results/api_guidance/elasticity.jsonl"))
    a = ap.parse_args()
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:a.n]]
    tasks = [(s, arm, a.model, a.region, tok, a.k, a.max_steps)
             for s in specs for arm in ("A", "B", "C", "D")]
    print("episodes %d" % len(tasks), flush=True)
    rows = []
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    by = collections.defaultdict(dict)
    for r in rows:
        if "feasible" in r:
            by[r["arm"]][r["problem_id"]] = r
    keep = sorted(set.intersection(*[set(by[x]) for x in "ABCD"])) if all(by.get(x) for x in "ABCD") else []
    N = len(keep)
    LAB = {"A": "baseline interface",
           "B": "+ measured per-element elasticity",
           "C": "+ elasticity AND the inverted required factor",
           "D": "+ required factor only, no elasticity"}
    print("\n=== n=%d problems, paired, %s ===" % (N, a.model.split("anthropic.")[-1]))
    for arm in "ABCD":
        if N:
            print("  arm %s  feasible %.3f   sims/ep %5.0f   %s"
                  % (arm, statistics.mean(by[arm][p]["feasible"] for p in keep),
                     statistics.mean(by[arm][p]["sims"] for p in keep), LAB[arm]))

    def mc(x, y):
        hi = sum(1 for p in keep if by[x][p]["feasible"] and not by[y][p]["feasible"])
        lo = sum(1 for p in keep if by[y][p]["feasible"] and not by[x][p]["feasible"])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / max(N, 1), pv
    print()
    for x, y, q in (("B", "A", "can it USE the primitive?"),
                    ("C", "B", "does it also need the inversion done?"),
                    ("D", "B", "is the primitive worth anything once the answer is given?"),
                    ("D", "A", "upper bound: hand over the answer"),
                    ("C", "A", "both")):
        hi, lo, d, pv = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d delta %+5.1fpp  p=%.2e   %s" % (x, y, hi, lo, d, pv, q))

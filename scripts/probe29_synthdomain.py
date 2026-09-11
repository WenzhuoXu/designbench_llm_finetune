"""
Does the ordering-fidelity law hold for a language model in a domain with no physics in it?

The law's domain-invariance is established with no model in the loop: truss and a synthetic
domain give the same slope, domain term z=1.51 n.s. With a model in the loop it has only ever
been measured on the truss, where the model brings structural knowledge and may have seen
similar problems. So the LLM result is confounded: it could be a fact about consuming
per-element signals, or a fact about knowing about trusses.

This ports the synthetic domain to text. Same interface shape as the truss harness -- a table
of per-element numbers, a grammar, k candidates a turn, iterate to feasibility -- but the
domain is abstract: element i has a value g_i that rises with its size x_i at an unknown rate,
a cost weight, a budget cap. No physics, no engineering vocabulary, nothing memorisable.

Sweeping corruption of the margin column varies ordering fidelity exactly as on the truss. If
the fitted slope is positive and comparable, the law is about signal structure. If it vanishes,
the LLM result was truss knowledge and the invariance claim does not extend to model consumers.
"""
import os, sys, json, math, random, re, argparse, time, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "scripts"), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
from da_matched_info import corrupt, spearman
from da_fit2 import irls
import api_grammar_2x2 as H

SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ACT = re.compile(r"SCALE\(\s*(\d+)\s*,\s*([0-9.]+)\s*\)")
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

SYSTEM = """You are tuning the sizes of a set of independent components.

Each component i has a size x_i (between 0.05 and 4.0), a REQUIREMENT VALUE g_i, and a cost
weight w_i. Raising x_i raises that component's g_i; the rate differs per component and is not
given to you. Total cost is the sum of w_i * x_i^2.

A configuration is ACCEPTABLE when every component has g_i >= 1.0 AND total cost <= the budget.

Each turn you are shown every component's current g_i, w_i and x_i, plus the total cost and the
budget. Think briefly, then emit one line per component you want to resize:

<action>SCALE(i, f)</action>

meaning multiply component i's size by factor f, with 0.7 <= f <= 2.0. EVERY line you emit is
applied, together, in the same turn. You may resize as many components as you like, including
all of them. Raising a component's size raises its g_i and its cost; lowering it frees budget.
You have a limited number of turns, so act on everything that needs it."""


def render(dom, st, shown):
    n = dom.n_elements(st)
    c = sum(st["w"][i] * st["x"][i] ** 2 for i in range(n))
    rows = ["Components:", "   id         g_i         w_i         x_i"]
    for i in range(n):
        rows.append("  C%-4d %10.3f  %10.3f  %10.4f" % (i, shown[i], st["w"][i], st["x"][i]))
    rows.append("")
    rows.append("total cost %.3f   budget %.3f   (%s)"
                % (c, st["B"], "within budget" if c <= st["B"] else "OVER BUDGET"))
    worst = min(range(n), key=lambda i: shown[i])
    rows.append("components with g_i below 1.000 do not yet meet the requirement.")
    return "\n".join(rows)


def apply_action(dom, st, i, f):
    nx = dom.clone(st)
    if 0 <= i < dom.n_elements(st):
        dom.set(nx, i, "x", st["x"][i] * f)
    return nx


def episode(seed, level, model, region, token, k, max_steps):
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": 10})
    rng = random.Random(seed ^ 0xC0FFEE)
    system = SYSTEM.replace("{K}", str(k))
    history, rhos = [], []
    for _ in range(max_steps):
        if dom.feasible(st):
            return {"seed": seed, "level": level, "feasible": True,
                    "rho": statistics.mean(rhos) if rhos else 1.0, "turns": len(history)}
        em = dom.element_margins(st)
        shown = corrupt(em, "perm", level, rng)
        r = spearman(em, shown)
        if r is not None:
            rhos.append(r)
        obs = render(dom, st, shown)
        convo = [{"role": "user", "content": [{"text":
                  "Bring this configuration to an acceptable state.\n\n" + obs}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text": "[Result]\n" + h["o"]}]})
        text = H.call(convo, system, model, region, token)
        moves = ACT.findall(text or "")
        if not moves:
            history.append({"t": (text or "").strip()[:2000],
                            "o": obs + "\nNo SCALE(i, f) action parsed. Emit exactly %d." % k})
            continue
        nx = dom.clone(st)
        applied = 0
        for a, b in moves:
            try:
                i, f = int(a), float(b)
            except Exception:
                continue
            if 0 <= i < dom.n_elements(nx) and 0.5 <= f <= 3.0:
                dom.set(nx, i, "x", nx["x"][i] * f)
                applied += 1
        if not applied:
            history.append({"t": (text or "").strip()[:2000], "o": obs})
            continue
        st = nx
        nshown = corrupt(dom.element_margins(st), "perm", level, rng)
        history.append({"t": (text or "").strip()[:2000], "o": render(dom, st, nshown)})
    return {"seed": seed, "level": level, "feasible": dom.feasible(st),
            "rho": statistics.mean(rhos) if rhos else 1.0, "turns": len(history)}


def work(t):
    seed, level, model, region, token, k, ms = t
    try:
        return episode(seed, level, model, region, token, k, ms)
    except Exception as e:
        return {"seed": seed, "level": level, "feasible": False, "rho": None, "err": str(e)[:200]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/synthdomain.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    tasks = [(s, lv, SONNET45, a.region, token, a.k, a.max_steps)
             for s in range(a.n) for lv in LEVELS]
    print("episodes %d | abstract domain, no physics" % len(tasks), flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    ok = [r for r in out if r.get("rho") is not None]
    print("  level   n     mean rho   feasible")
    for lv in LEVELS:
        sub = [r for r in ok if r["level"] == lv]
        if sub:
            print("  %5.2f  %4d    %.3f      %.3f"
                  % (lv, len(sub), statistics.mean([r["rho"] for r in sub]),
                     sum(1.0 for r in sub if r["feasible"]) / len(sub)))
    y = [1.0 if r["feasible"] else 0.0 for r in ok]
    X = [[1.0, r["rho"]] for r in ok]
    ids = [str(r["seed"]) for r in ok]
    b, se, cl, ll = irls(X, y, ids)
    print("\n  outcome ~ rho, clustered by instance:  coef %+.3f   cluster z = %.2f  (n=%d)"
          % (b[1], b[1] / max(cl[1], 1e-9), len(ok)))
    print("  truss LLM reference (Sonnet 4.5): +2.94   model-free reference: +7.14")

"""
Turn the two-point anecdote into a dose-response.

On the truss, fully-stressed sizing beats the model 0.470 to 0.333. In the abstract domain the
model beats the procedural loop 0.690 to 0.590. I explained the difference by the quality of
procedural rule each domain admits -- fully-stressed design is well matched to trusses, whereas
the abstract loop carries a fixed exponent of 3.0 against elasticities that are lognormal
around 2.2 and is therefore a poor rule. That is a plausible story told over two points.

It is testable properly. Rule strength can be varied WITHIN the abstract domain, holding the
domain, the instances, the seeds, the turn budget and the signal fixed, by changing only what
the procedure believes about the response exponent:

  3.0            the mismatched rule the comparison happened to use
  2.2            matched to the median true exponent
  per-element    each element's true k_i -- an oracle-strong rule
  inner search   the tool library's optimiser, stronger still

The model's 0.690 on these same 100 seeds is fixed. If the story is right, the procedural arms
should cross it as the rule improves, and where they cross says what kind of domain the agent
is worth having in.
"""
import sys, math, random, statistics, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_synth import SynthDomain
import da_tools as T

STEPS = 6
N_ELEM = 10
LLM_TRUTHFUL = 0.690          # measured, same seeds, same configuration, truthful signal


def episode(seed, rule):
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": N_ELEM})
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        em = dom.element_margins(st)
        if not em:
            break
        best, bv = None, T.phi_rho(dom, st)
        for tgt in [1.0 * (1.8 / 1.0) ** (j / 11) for j in range(12)]:
            nx = dom.clone(st)
            for i in range(dom.n_elements(st)):
                b = dom.get(st, i, "x")
                if not b:
                    continue
                if rule == "true_k":
                    e = st["k"][i]                       # the element's real exponent
                elif rule == "fixed_2.2":
                    e = 2.2
                else:
                    e = 3.0
                f = min(2.0, max(0.7, (tgt / max(em[i], 1e-9)) ** (1.0 / max(e, 0.2))))
                dom.set(nx, i, "x", b * f)
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def episode_search(seed):
    """The tool library's own optimiser, given the same turn budget."""
    dom = SynthDomain()
    st = dom.load({"seed": seed, "n": N_ELEM})
    rng = random.Random(seed)
    for _ in range(STEPS):
        if dom.feasible(st):
            return 1
        n = dom.n_elements(st)
        best, bv = None, T.phi_rho(dom, st)
        for _ in range(40):
            nx = dom.clone(st)
            for i in range(n):
                b = dom.get(st, i, "x")
                if b:
                    dom.set(nx, i, "x", b * math.exp(rng.gauss(0.0, 0.25)))
            v = T.phi_rho(dom, nx)
            if v > bv:
                best, bv = nx, v
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


def _w(t):
    seed, rule = t
    try:
        return (rule, episode_search(seed) if rule == "search" else episode(seed, rule))
    except Exception:
        return None


RULES = ["fixed_3.0", "fixed_2.2", "true_k", "search"]
LBL = {"fixed_3.0": "fixed exponent 3.0   (mismatched: the rule used in the comparison)",
       "fixed_2.2": "fixed exponent 2.2   (matched to the median true exponent)",
       "true_k":    "per-element true k_i (oracle-strong rule)",
       "search":    "inner random search  (the tool library's optimiser)"}

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    with Pool(44) as pool:
        got = [r for r in pool.map(_w, [(s, r) for s in range(N) for r in RULES],
                                   chunksize=4) if r]
    print("abstract domain, truthful signal, %d seeds, identical configuration, %.0fs\n"
          % (N, time.time() - t0))
    print("  procedural rule                                          feasible   vs LLM")
    for r in RULES:
        v = [x[1] for x in got if x[0] == r]
        if not v:
            continue
        m = sum(v) / len(v)
        print("  %-55s %.3f    %+.3f" % (LBL[r], m, m - LLM_TRUTHFUL))
    print("  %-55s %.3f" % ("language model (Sonnet 4.5), same seeds", LLM_TRUTHFUL))
    print("\n  The account predicts the procedural arms cross the model as the rule improves.")

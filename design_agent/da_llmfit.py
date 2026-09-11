"""
Does an LLM in the loop obey the same law?

Everything establishing the ordering-fidelity law is model-free. The LLM results are a
separate pile of facts sitting beside it: the harm replicates in ten models across six
vendors. Those are two claims, not one, until the same law is fitted to LLM episodes.

The corruption sweep already recorded per-turn rho with a model in the loop, over the same
two corruption families, so the comparison needs no new API spend:

    outcome ~ rho          fitted on LLM episodes, cluster-robust by problem

and the coefficient is compared against the model-free +7.14 (z=20.9). A similar slope means
the law describes the LLM loop too and the two halves are one result. A shallower slope means
the model is partly robust to ordering corruption; a steeper one, more sensitive than a
procedural consumer.
"""
import sys, json, math, statistics, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p in (str(HERE), "/ocean/projects/mch250030p/wxu7/llm_finetune"):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_fit2 import irls

SRC = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/matrix.jsonl")

if __name__ == "__main__":
    rows = []
    for line in open(SRC, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ts = r.get("turns") or []
        rr = [t["rho"] for t in ts if t.get("rho") is not None]
        if not rr:
            continue
        rows.append({"pid": r["problem_id"], "model": r["model"], "mode": r["mode"],
                     "level": r["level"], "y": float(bool(r["feasible"])),
                     "rho": statistics.mean(rr)})
    print("LLM episodes with a usable rho: %d" % len(rows))
    bym = collections.Counter(r["model"].split("anthropic.")[-1][:26] for r in rows)
    print("by model: %s" % dict(bym))

    def fit(sub, label):
        if len(sub) < 40:
            print("  %-34s n=%-4d too few" % (label, len(sub)))
            return
        X = [[1.0, r["rho"]] for r in sub]
        y = [r["y"] for r in sub]
        ids = [r["pid"] for r in sub]
        b, se, cl, _ = irls(X, y, ids)
        rr = [r["rho"] for r in sub]
        span = max(rr) - min(rr)
        print("  %-34s n=%-4d  rho coef %+7.3f  cluster z=%6.2f  span %.2f -> %+.1f logits"
              % (label, len(sub), b[1], b[1] / max(cl[1], 1e-9), span, b[1] * span))

    print("\n  fit                                 n      coefficient        effect")
    fit(rows, "ALL LLM episodes")
    for m in sorted({r["model"] for r in rows}):
        fit([r for r in rows if r["model"] == m], m.split("anthropic.")[-1][:30])
    print()
    for mode in ("perm", "noise"):
        fit([r for r in rows if r["mode"] == mode], "corruption = %s" % mode)

    print("\n  model-free reference: rho coef +7.14 (z=20.9), +12.0 logits over its span")
    print("  a comparable slope means the LLM loop obeys the same law and the two halves")
    print("  of this work are one result rather than two.")

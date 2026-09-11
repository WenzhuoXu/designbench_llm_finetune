"""Do the two proposers share an action space?

The random sampler's support (da_search2.sample_generic / da_domtools):
    SIZE_PASS margin  ~ U(0.95, 1.35)
    SCALE    factor   ~ U(0.75, 1.60),  1..min(4,ne) ids
    TRIM     threshold~ U(1.2, 5.0), factor ~ U(0.75, 0.95)
    MOVE_JOINT        ~ current xy +- 0.4
The LLM is TOLD those ranges in TOOLS_DOC but nothing enforces them; parse_candidates
applies no range check, and size_pass/scale/trim clip the resulting RATIO to (0.7, 2.0),
not the argument.  da_lowbudget logs only move KINDS, so this cannot be recovered from
the recorded jsonl.  Run the real llm proposer on a few held-out problems and log every
parsed argument.
"""
import os, sys, json, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

import da_lowbudget as L

SEEN = []
_orig_parse = L.parse_candidates


def spy_parse(text, nj, ne):
    out = _orig_parse(text, nj, ne)
    for kind, args in out:
        SEEN.append({"kind": kind, "args": args, "ne": ne, "nj": nj})
    return out


L.parse_candidates = spy_parse

BUDGET = 20
START, N = 150, 162


def run(spec):
    dom = L.BudgetedTruss(BUDGET)
    log = {"turns": 0, "cands": 0, "invalid": 0, "repeat": 0, "model_calls": 0,
           "parsed": 0, "picked": Counter()}
    try:
        st = dom.load(spec)
        init = dict(dom.evaluate(st))
        L.budget_search(dom, st, L.llm_propose(spec, init, REGION, TOKEN, log), log)
    except L.BudgetExhausted:
        pass
    except Exception as e:
        return {"err": type(e).__name__ + ": " + str(e)[:120]}
    return {"ok": True, "calls": dom.calls}


if __name__ == "__main__":
    REGION = os.environ.get("AWS_REGION", "us-west-2")
    TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[START:START + N][::9]
    specs = [json.load(open(f)) for f in files]
    print("probing %d held-out problems at budget %d" % (len(specs), BUDGET), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        res = list(ex.map(run, specs))
    print("wall %.0fs  errors %d" % (time.time() - t0, sum(1 for r in res if r.get("err"))),
          flush=True)

    print("\nparsed candidates logged: %d" % len(SEEN))
    print("by kind:", dict(Counter(s["kind"] for s in SEEN).most_common()))

    def check(name, pred, desc):
        sub = [s for s in SEEN if s["kind"] == name]
        bad = [s for s in sub if pred(s)]
        print("  %-12s n=%-4d outside sampler support (%s): %d  (%.1f%%)"
              % (name, len(sub), desc, len(bad), 100.0 * len(bad) / max(len(sub), 1)))
        for s in bad[:6]:
            print("        e.g.", s["args"])

    print("\naction-space check vs the random sampler's support:")
    check("SIZE_PASS", lambda s: not (0.95 <= s["args"]["margin"] <= 1.35), "margin in [0.95,1.35]")
    check("SCALE", lambda s: not (0.75 <= s["args"]["factor"] <= 1.60)
          or len(s["args"]["ids"]) > min(4, s["ne"]), "factor in [0.75,1.60], <=min(4,ne) ids")
    check("TRIM", lambda s: not (1.2 <= s["args"]["threshold"] <= 5.0
                                 and 0.75 <= s["args"]["factor"] <= 0.95),
          "thr in [1.2,5.0], f in [0.75,0.95]")
    check("MOVE_JOINT", lambda s: True, "any move beyond +-0.4 -- args shown")

    sp = [s["args"]["margin"] for s in SEEN if s["kind"] == "SIZE_PASS"]
    if sp:
        sp.sort()
        print("\nSIZE_PASS margins proposed: min %.3f  p25 %.3f  median %.3f  p75 %.3f  max %.3f"
              % (sp[0], sp[len(sp)//4], sp[len(sp)//2], sp[3*len(sp)//4], sp[-1]))
        print("  fraction > 1.35: %.3f   fraction < 0.95: %.3f"
              % (sum(1 for x in sp if x > 1.35)/len(sp), sum(1 for x in sp if x < 0.95)/len(sp)))
    sc = [s["args"]["factor"] for s in SEEN if s["kind"] == "SCALE"]
    if sc:
        sc.sort()
        print("SCALE factors proposed:    min %.3f  median %.3f  max %.3f  frac>1.60 %.3f"
              % (sc[0], sc[len(sc)//2], sc[-1], sum(1 for x in sc if x > 1.60)/len(sc)))
        ids = [len(s["args"]["ids"]) for s in SEEN if s["kind"] == "SCALE"]
        print("SCALE ids per move:        mean %.2f  max %d  (sampler draws 1..min(4,ne))"
              % (sum(ids)/len(ids), max(ids)))

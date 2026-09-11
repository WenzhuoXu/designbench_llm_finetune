"""
Does one serialisation actually render all three domains identically in structure?

A1 claims a domain-agnostic format. The test is not that it runs but that the layout is the same
everywhere while the content differs: same section order, same column skeleton, same tool
vocabulary shape, with each domain naming its own elements and parameter. This prints one state
per domain, checks the structural invariants mechanically, and round-trips a tool call.
"""
import sys
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts"),
          "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
import json, random
from da_domain import get_domain
from da_synth import SynthDomain
from da_pipe import PipeDomain
import da_domtools
import da_meta          # per-domain nouns, columns, and the pipe margin fix
import da_serial as S

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")


def truss():
    d = get_domain("truss")
    f = sorted((DB / "data/problems_hard").glob("*.json"))[0]
    return d, d.load(json.load(open(f)))


def synth():
    d = SynthDomain(); return d, d.load({"seed": 3, "n": 8})


def pipe():
    d = PipeDomain(); return d, d.load({"seed": 3, "n": 8})


REQUIRED = ("table:", "  id", "budget:", "worst ")

print("Rendering one state per domain through the same function.\n")
rendered = {}
for name, mk in (("truss", truss), ("synth", synth), ("pipe", pipe)):
    dom, st = mk()
    txt = S.render_state(dom, st)
    rendered[name] = txt
    print("=" * 72)
    print("[%s]  noun=%r  param=%r  tools=%s"
          % (name, S.noun(dom), S.param_name(dom), S.tool_vocabulary(dom)))
    print("-" * 72)
    print("\n".join(txt.splitlines()[:12]))
    print("   ... (%d lines total)" % len(txt.splitlines()))

print("\n" + "=" * 72)
print("STRUCTURAL INVARIANTS")
ok = True
for name, txt in rendered.items():
    missing = [k for k in REQUIRED if k not in txt]
    if missing:
        ok = False
    print("  %-6s sections present: %s%s"
          % (name, "all" if not missing else "MISSING " + ", ".join(missing), ""))

# section order must be identical across domains
def order(txt):
    seq = []
    for ln in txt.splitlines():
        for k in REQUIRED:
            if ln.strip().startswith(k.strip()) and (not seq or seq[-1] != k):
                seq.append(k)
    return seq
orders = {n: order(t) for n, t in rendered.items()}
same = len({tuple(v) for v in orders.values()}) == 1
print("  identical section order across domains: %s  %s"
      % (same, "" if same else orders))
ok = ok and same

# the three shared tools must be offered everywhere
shared = [v.split("(")[0] for v in S.tool_vocabulary(SynthDomain())][:3]
for name, mk in (("truss", truss), ("synth", synth), ("pipe", pipe)):
    dom, _ = mk()
    vocab = [v.split("(")[0] for v in S.tool_vocabulary(dom)]
    has = all(t in vocab for t in shared)
    print("  %-6s offers the three shared tools: %s   (full vocab %s)" % (name, has, vocab))
    ok = ok and has

# round-trip a tool call
txt = (S.render_tool("SCALE", {"ids": [3, 7], "factor": 1.2})
       + "\n" + S.render_tool("SIZE_PASS", {"margin": 1.05}))
back = S.parse_tools(txt)
rt = back == [("SCALE", {"ids": [3, 7], "factor": 1.2}),
              ("SIZE_PASS", {"margin": 1.05})]
print("  tool call round-trips: %s   %s" % (rt, back))
ok = ok and rt

print("\n  A1 %s" % ("HOLDS: one layout, one vocabulary, domain-specific content only"
                     if ok else "FAILS: see above"))

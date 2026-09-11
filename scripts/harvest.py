"""
How much supervision does one search run actually contain?

gen_expert.py keeps the accepted move at each turn and throws the rest away. But the search
reaches that move by proposing candidates, applying each to a copy, rolling it forward under the
sizing rule and scoring the result with the potential. Every discarded candidate is a
(state, action, value) triple that cost FEA calls to produce and is simply deleted.

This runs the same search with a recorder attached and counts what is there: candidates per
turn, their value distribution, how often the accepted move is clearly better than the
runner-up versus a near-tie, and how many distinct actions appear by head. From those, the
training signals available are counted directly rather than estimated:

  next-action SFT    one example per turn -- what is kept today
  preference pairs   accepted vs each clearly-worse candidate at the same state
  value targets      every candidate with its rolled-out potential
  rejected-sample    candidates that scored ABOVE the do-nothing branch but still lost

A near-tie teaches a preference model noise, so pairs are only counted when the value gap
clears a threshold expressed as a fraction of that turn's observed value spread.

Runs across every domain so the multiplier is not a truss-only claim.
"""
import sys, json, math, random, zlib, argparse, statistics, time
from pathlib import Path
from collections import Counter
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_pipe import PipeDomain
from da_catalogue import CatalogueDomain
from da_cases import CasesDomain
import da_domtools, da_meta
import da_serial as S
from da_search import size_pass, potential, rollout
from da_search3 import apply_tool, propose, GENERIC

HORIZON, POOL = 12, 32
TIE_FRAC = 0.05          # a pair counts only if the gap clears 5% of the turn's value spread


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def make(domain, key):
    if domain == "truss":
        d = get_domain("truss")
        return d, d.load(json.load(open(key))), "r"
    if domain == "synth":
        d = SynthDomain(); return d, d.load({"seed": key, "n": 10}), "x"
    if domain == "pipe":
        d = PipeDomain(); return d, d.load({"seed": key, "n": 14}), "d"
    if domain == "catalogue":
        d = CatalogueDomain(); return d, d.load({"seed": key, "n": 10}), "k"
    d = CasesDomain(); return d, d.load({"seed": key, "n": 10, "cases": 4}), "x"


def harvest_episode(dom, st, param, rng, steps=HORIZON, nprop=POOL):
    """The search, unchanged in behaviour, with every scored candidate retained."""
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    turns = []
    for turn in range(steps):
        if dom.feasible(st):
            break
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            break
        remaining = steps - turn - 1
        base_ok, base_state = rollout(dom, st, param, remaining)
        if base_ok:
            # the continuation already reaches feasibility: advance onto it, do not just stop
            st = base_state
            break
        base_v = potential(dom, base_state)
        scored, winner = [], None
        for kind, args in propose(dom, st, rng, nprop, extra, GENERIC):
            cand = apply_tool(dom, st, param, kind, args, extra)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, param, remaining)
            v = 1e9 if ok else potential(dom, rolled)
            scored.append({"head": kind, "args": args, "value": v, "cand": cand})
            if ok:
                winner = rolled          # this candidate's continuation reaches feasibility
                break
        if not scored:
            break
        obs = S.render_state(dom, st)
        best = max(range(len(scored)), key=lambda i: scored[i]["value"])
        turns.append({"obs_chars": len(obs), "base_v": base_v,
                      "cands": [{"head": c["head"], "args": c["args"], "value": c["value"]}
                                for c in scored],
                      "accepted": best})
        if winner is not None:
            st = winner                  # the turn is recorded, then take the solved state
            break
        if scored[best]["value"] <= base_v + 1e-12:
            break
        st = scored[best]["cand"]
    return turns, bool(dom.feasible(st))


def analyse(all_turns):
    n_turns = len(all_turns)
    n_cands = sum(len(t["cands"]) for t in all_turns)
    heads = Counter()
    pairs = ties = above_base = 0
    gaps = []
    for t in all_turns:
        vals = [c["value"] for c in t["cands"]]
        finite = [v for v in vals if v < 1e8]
        spread = (max(finite) - min(finite)) if len(finite) > 1 else 0.0
        thr = max(TIE_FRAC * spread, 1e-9)
        bv = t["cands"][t["accepted"]]["value"]
        for i, c in enumerate(t["cands"]):
            heads[c["head"]] += 1
            if i == t["accepted"]:
                continue
            gap = bv - c["value"]
            if gap > thr:
                pairs += 1
                gaps.append(gap)
            else:
                ties += 1
            if c["value"] > t["base_v"] + thr:
                above_base += 1
    return {"turns": n_turns, "cands": n_cands, "heads": heads, "pairs": pairs,
            "ties": ties, "above_base": above_base,
            "median_gap": statistics.median(gaps) if gaps else 0.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--domains", default="truss,synth,pipe,catalogue,cases")
    a = ap.parse_args()
    files = [str(f) for f in sorted((DESIGNBENCH / "data/problems_hard").glob("*.json"))[:a.n]]

    grand = {"turns": 0, "cands": 0, "pairs": 0, "ties": 0, "above_base": 0}
    print("what one search run contains, per domain (%d instances each)\n" % a.n)
    print("  domain      solved  turns  candidates  cand/turn  pref pairs  near-ties  >do-nothing")
    for dm in a.domains.split(","):
        t0 = time.time()
        keys = files if dm == "truss" else list(range(a.n))
        turns_all, solved = [], 0
        for k in keys:
            try:
                dom, st, param = make(dm, k)
                ts, ok = harvest_episode(dom, st, param, random.Random(seed_of("%s%s" % (dm, k))))
            except Exception:
                continue
            turns_all.extend(ts)
            solved += 1 if ok else 0
        r = analyse(turns_all)
        grand["turns"] += r["turns"]; grand["cands"] += r["cands"]
        grand["pairs"] += r["pairs"]; grand["ties"] += r["ties"]
        grand["above_base"] += r["above_base"]
        print("  %-10s  %5.3f  %5d  %10d  %9.1f  %10d  %9d  %11d"
              % (dm, solved / max(len(keys), 1), r["turns"], r["cands"],
                 r["cands"] / max(r["turns"], 1), r["pairs"], r["ties"], r["above_base"]))
        top = ", ".join("%s %d" % (h, c) for h, c in r["heads"].most_common(6))
        print("              heads: %s   (%.0fs)" % (top, time.time() - t0))

    print("\n  TOTALS over %d instances across %d domains" % (a.n, len(a.domains.split(","))))
    print("    next-action SFT examples (kept today) : %d" % grand["turns"])
    print("    scored candidates (discarded today)   : %d" % grand["cands"])
    print("    usable preference pairs               : %d" % grand["pairs"])
    print("    near-ties excluded from pairs         : %d" % grand["ties"])
    print("    candidates beating the do-nothing arm : %d" % grand["above_base"])
    print("    MULTIPLIER, value targets per SFT turn: %.1fx" % (grand["cands"] / max(grand["turns"], 1)))
    print("    MULTIPLIER, preference pairs per turn : %.1fx" % (grand["pairs"] / max(grand["turns"], 1)))

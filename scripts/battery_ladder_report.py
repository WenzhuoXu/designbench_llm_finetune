#!/usr/bin/env python
"""Ladder table for the battery domain, in the same shape as search_ladder's."""
from __future__ import annotations
import json, math, sys, statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/battery_ladder")
PROB = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/data/battery_problems")
ORDER = ["random", "greedy_critical", "lookahead_v1_d1", "lookahead_v2_d1", "lookahead_v2_d2"]

best_known = {}
for f in sorted(PROB.glob("*.json")):
    s = json.load(open(f))
    best_known[s["problem_id"]] = s["_metadata"]["best_known_charge_time"]


def table(tag):
    p = RES / tag / "runs.jsonl"
    if not p.exists():
        print(f"(missing {p})"); return
    rows = [json.loads(l) for l in open(p)]
    print(f"\n===== {tag}   ({len(rows)} runs, {len({r['problem_id'] for r in rows})} problems)")
    by = defaultdict(list)
    for r in rows:
        by[(r["policy"], r.get("family", "?"))].append(r)
        by[(r["policy"], "ALL")].append(r)
    print(f"{'policy':<20} {'family':<10} {'n':>3} {'feas':>7} {'t_chg/cap':>10} "
          f"{'t_chg/best':>11} {'sims':>7} {'steps':>6}")
    print("-" * 82)
    fams = ["ALL"] + sorted({r.get("family", "?") for r in rows})
    for pol in ORDER:
        for fam in fams:
            rs = by.get((pol, fam))
            if not rs:
                continue
            feas = [r for r in rs if r["feasible"]]
            ratio = [r["final_mass"] / r["mass_ref"] for r in feas
                     if r["mass_ref"] > 0 and math.isfinite(r["final_mass"])]
            vsbest = [r["final_mass"] / best_known[r["problem_id"]] for r in feas
                      if best_known.get(r["problem_id"])]
            print(f"{pol:<20} {fam:<10} {len(rs):>3} {len(feas)/len(rs):>6.1%} "
                  f"{(st.mean(ratio) if ratio else float('nan')):>10.3f} "
                  f"{(st.mean(vsbest) if vsbest else float('nan')):>11.3f} "
                  f"{st.mean([r['fea_calls'] for r in rs]):>7.0f} "
                  f"{st.mean([r['steps'] for r in rs]):>6.1f}")
        print()
    # per-problem solved matrix
    pids = sorted({r["problem_id"] for r in rows})
    solved = {(r["policy"], r["problem_id"]): r["feasible"] for r in rows}
    print(f"{'problem':<17}" + "".join(f"{p.replace('lookahead_','la_'):>17}" for p in ORDER))
    for pid in pids:
        print(f"{pid:<17}" + "".join(
            f"{('YES' if solved.get((p,pid)) else '.'):>17}" for p in ORDER))
    # strict-superset check, v2 vs v1 and vs greedy
    def S(pol):
        return {r["problem_id"] for r in rows if r["policy"] == pol and r["feasible"]}
    for a, b in (("lookahead_v2_d1", "lookahead_v1_d1"),
                 ("lookahead_v2_d1", "greedy_critical"),
                 ("lookahead_v2_d2", "lookahead_v2_d1")):
        A, B = S(a), S(b)
        print(f"\n  {a} vs {b}: +{len(A-B)} / -{len(B-A)}   gained={sorted(A-B)} lost={sorted(B-A)}")


for tag in (sys.argv[1:] or ["dfn_scales-none", "dfn_scales-problem"]):
    table(tag)

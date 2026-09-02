#!/usr/bin/env python
"""Paired comparison of arms on identical preference pairs.

Every arm is scored on byte-identical (state, chosen, rejected) triples, so the
comparison pairs by triple and the per-pair difficulty differences cancel. That is
why this is far better powered than each arm's own accuracy interval: the absolute
accuracy carries the between-problem variance, the paired difference does not.

Clustering is respected throughout. Pairs drawn from the same problem share a
prompt prefix and are correlated, so every interval here is a cluster-robust one
computed over PROBLEMS, not over pairs. On the first arm scored, clustering
inflated the standard error 2.5x (0.026 -> 0.065); an analysis that ignored it
would manufacture significance.

    python scripts/compare_preference_margins.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ORDER = ["warmstart", "champion", "t5a_phi1", "t5ao_phionly_v1", "t5b_phi2",
         "t5bo_phionly_v2", "t5c_lookahead", "t5co_lookahead_only",
         "t5bw_informative", "t7_dpo_init", "t6_distill", "dpo_prefs"]

LABEL = {
    "warmstart": "warmstart (common init)",
    "champion": "champion (GRPO, Phi_v1 blend)",
    "t5a_phi1": "T5a  Phi_v1 composite",
    "t5ao_phionly_v1": "T5ao Phi_v1 only",
    "t5b_phi2": "T5b  Phi_v2 composite",
    "t5bo_phionly_v2": "T5bo Phi_v2 only",
    "t5c_lookahead": "T5c  Phi_v2 + lookahead adv",
    "t5co_lookahead_only": "T5co lookahead adv only",
    "t5bw_informative": "T5bw Phi_v2 + informative sampling",
    "t7_dpo_init": "T7   GRPO from DPO init",
    "t6_distill": "T6   SFT on search traces",
    "dpo_prefs": "DPO  on search preferences",
}


def _key(p):
    return (p["problem_id"], p.get("step"))


def load(results: Path):
    out = {}
    for f in sorted(glob.glob(str(results / "*.json"))):
        name = Path(f).stem
        d = json.load(open(f))
        out[name] = {"agg": d["aggregate"], "pairs": d["per_pair"]}
    return out


def cluster_se(values_by_problem):
    """Standard error over PROBLEM means, not over pairs."""
    means = [st.mean(v) for v in values_by_problem.values() if v]
    if len(means) < 2:
        return float("nan")
    return st.stdev(means) / math.sqrt(len(means))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/regret")
    ap.add_argument("--baseline", default="warmstart")
    args = ap.parse_args()
    data = load(Path(args.results))
    if not data:
        print("no results yet")
        return

    print("PREFERENCE MARGIN on held-out same-move pairs")
    print("  chance = 0.500 by construction: the rejected action is the IDENTICAL move")
    print("  (same type, parameter and scale factor) on a DIFFERENT element, with")
    print("  byte-identical reasoning text. Only knowledge of WHICH element can beat it.\n")
    print(f"  {'arm':<38}{'n':>6}{'accuracy':>10}{'±clust':>9}{'mean margin':>13}")
    print("  " + "-" * 78)
    for name in ORDER:
        if name not in data:
            continue
        a = data[name]["agg"]
        print(f"  {LABEL.get(name, name):<38}{a['n_pairs']:>6}{a['accuracy']:>10.3f}"
              f"{a['accuracy_se_clustered']:>9.3f}{a['mean_margin']:>13.3f}")
    for name in sorted(set(data) - set(ORDER)):
        a = data[name]["agg"]
        print(f"  {name:<38}{a['n_pairs']:>6}{a['accuracy']:>10.3f}"
              f"{a['accuracy_se_clustered']:>9.3f}{a['mean_margin']:>13.3f}")

    contrasts = [
        ("Phi_v1 -> Phi_v2, composite", "t5a_phi1", "t5b_phi2"),
        ("Phi_v1 -> Phi_v2, potential only", "t5ao_phionly_v1", "t5bo_phionly_v2"),
        ("Phi_v2 -> + lookahead advantage", "t5b_phi2", "t5c_lookahead"),
        ("Phi_v2 only -> lookahead adv only", "t5bo_phionly_v2", "t5co_lookahead_only"),
        ("Phi_v2 only -> + informative sampling", "t5bo_phionly_v2", "t5bw_informative"),
        ("Phi_v2 only -> DPO initialisation", "t5bo_phionly_v2", "t7_dpo_init"),
        ("warmstart -> champion", "warmstart", "champion"),
        ("warmstart -> DPO on preferences", "warmstart", "dpo_prefs"),
        ("warmstart -> SFT on traces", "warmstart", "t6_distill"),
    ]
    print("\n  PAIRED CONTRASTS (same pairs on both sides; cluster-robust by problem)")
    print(f"  {'contrast':<40}{'n':>5}{'Δacc':>9}{'±clust':>9}{'t':>7}{'Δmargin':>10}")
    print("  " + "-" * 82)
    for label, a, b in contrasts:
        if a not in data or b not in data:
            print(f"  {label:<40}{'(pending)':>5}")
            continue
        pa = {_key(p): p for p in data[a]["pairs"]}
        pb = {_key(p): p for p in data[b]["pairs"]}
        common = sorted(set(pa) & set(pb))
        if not common:
            print(f"  {label:<40}{'(no overlap)':>5}")
            continue
        by_problem = defaultdict(list)
        marg = defaultdict(list)
        for k in common:
            by_problem[k[0]].append(float(pb[k]["correct"]) - float(pa[k]["correct"]))
            marg[k[0]].append(pb[k]["margin"] - pa[k]["margin"])
        diffs = [d for v in by_problem.values() for d in v]
        mean_d = st.mean(diffs)
        se = cluster_se(by_problem)
        t = mean_d / se if se and math.isfinite(se) and se > 0 else float("nan")
        dmarg = st.mean([d for v in marg.values() for d in v])
        print(f"  {label:<40}{len(common):>5}{mean_d:>+9.3f}{se:>9.3f}{t:>7.2f}{dmarg:>+10.3f}")
    print("\n  |t| > 2 with the cluster-robust se is the bar. Accuracy differences smaller")
    print("  than ~2x the clustered se are not evidence, however many pairs there are.")


if __name__ == "__main__":
    main()

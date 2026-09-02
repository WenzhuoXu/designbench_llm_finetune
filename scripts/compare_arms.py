#!/usr/bin/env python
"""Compare the turn-5 training arms on one page.

Reads each run's metrics.jsonl (training-time probe trajectory) and each
checkpoint's eval_results.json (held-out outcome), and prints the comparisons the
study turns on, with the noise floor attached to every one of them.

The noise floor is not a guess. The alpha cells mt_s1_01a/01b/01c were three runs
of an IDENTICAL configuration -- `posterior.alpha` never reached the reward
(defect D4) -- and returned 8% / 12% / 4%. At n=25 that is +/-4 points. Any
feasibility difference smaller than that is not a result, which is why mass and
the probe trajectory carry the argument: both are continuous, and the probe has
hundreds of samples per logged point rather than one per problem.

    python scripts/compare_arms.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

# label -> (run_name prefix for logs/, eval dir under results/eval/)
ARMS = [
    ("T5a   Phi_v1, composite", "t5a_phi1", "t5a_phi1_eval"),
    ("T5b   Phi_v2, composite", "t5b_phi2", "t5b_phi2_eval"),
    ("T5c   Phi_v2 + lookahead", "t5c_lookahead", "t5c_lookahead_eval"),
    ("T5ao  Phi_v1 only", "t5ao_phionly_v1", "t5ao_phionly_v1_eval"),
    ("T5bo  Phi_v2 only", "t5bo_phionly_v2", "t5bo_phionly_v2_eval"),
    ("T5co  lookahead advantage only", "t5co_lookahead_only", "t5co_lookahead_only_eval"),
    ("T5bw  Phi_v2 + informative sampling", "t5bw_informative", "t5bw_informative_eval"),
    ("T7    GRPO from DPO init", "t7_dpo_init", "t7_dpo_init_eval"),
]
REFERENCES = [
    ("champion (K=1, T=0.8, stop-at-feasible)", "champ_heldout_k1"),
    ("champion (K=1, T=0.8, matched turns)", "champ_heldout_k1_minmass"),
    ("champion (greedy, standard eval)", "champ_heldout_greedy"),
    ("T6 distilled (greedy, standard eval)", "t6_distill_greedy"),
    ("DPO on search preferences (greedy)", "dpo_v4_greedy"),
    ("champion + potential ranking (K=8)", "champ_heldout_k8"),
    ("T6 distilled (K=1)", "t6_distill_k1"),
    ("T6 distilled + ranking (K=8)", "t6_distill_k8"),
]
NOISE_FLOOR_PP = 4.0   # points, at n=25, measured from the accidental triplicate


def _latest_metrics(prefix: str, logs: Path) -> list[dict]:
    candidates = sorted(logs.glob(f"{prefix}_*"), key=lambda p: p.name)
    for d in reversed(candidates):
        f = d / "metrics.jsonl"
        if f.exists() and f.stat().st_size > 0:
            rows = []
            for line in f.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if rows:
                return rows
    return []


def _solve_rate(rows: list) -> float | None:
    """Mean per-problem solve rate when the cell was evaluated with --n-samples > 1.

    This is the estimator to prefer. Measured on this study's own data, the greedy
    single-rollout rate has a within-arm standard deviation of 0.051 across
    checkpoints of the SAME run against a between-arm standard deviation of 0.046 --
    the noise exceeds the signal, so it cannot rank the arms. Batch size alone moved
    it by 17 points on a fixed checkpoint (batch 1 reproduced the sequential path
    12/12; batch 8 diverged to 8/12), which identifies the cause: one flipped greedy
    token early in a 14-turn rollout changes the whole trajectory.
    """
    usable = [r for r in rows if "solve_rate" in r]
    if not usable or len(usable) != len(rows):
        return None
    return sum(r["solve_rate"] for r in usable) / len(usable)


def _valid_rate(rows: list) -> float | None:
    """Recompute the conservative rate from stored per-problem fields.

    Older result files used program.is_feasible alone, which can exceed the
    simulator's own verdict when a diverged FEA is clamped to a sentinel the
    problem has no constraint against. Recomputing here avoids rerunning them.
    """
    usable = [r for r in rows if "state_is_valid" in r]
    if not usable:
        return None
    return sum(bool(r.get("reaches_solution")) and bool(r.get("state_is_valid"))
               for r in usable) / len(usable)


def _fmt(value, spec="6.3f"):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "     -"
    return format(value, spec)


def _regret_frac(row: dict):
    """Regret as a FRACTION of the improvement that was available at that state.

    Raw regret is in potential units, and Phi's scale is problem-dependent, so a
    batch of hard problems reads as high regret whatever the policy does. Measured:
    every arm's mean regret fell ~10x between step 5 and step 10 purely because the
    batches contained different problems. rank_frac (a percentile) and this ratio
    are scale-free and are the statistics to compare across steps.

        regret_frac = (best - policy) / (best - baseline) = regret / (regret + advantage)
    """
    regret = row.get("lookahead/regret")
    advantage = row.get("lookahead/advantage")
    if regret is None or advantage is None:
        return None
    denom = regret + advantage
    return regret / denom if abs(denom) > 1e-9 else None


def training_table(logs: Path):
    print("\nTRAINING — probe trajectory (first -> last logged point)")
    print("  rank_frac is the headline: it is a PERCENTILE of the policy's action among the")
    print("  candidates at the same state, so it does not move with the problem's scale.")
    print("  Raw regret is in potential units and tracks batch difficulty, not policy quality.")
    print(f"\n  {'arm':<26}{'steps':>6}{'kl':>9}{'rank 1st':>10}{'rank last':>11}"
          f"{'delta':>9}{'regret_frac':>13}{'reliable':>10}{'states':>8}")
    print("  " + "-" * 104)
    out = {}
    for label, prefix, _ in ARMS:
        rows = _latest_metrics(prefix, logs)
        if not rows:
            print(f"  {label:<26}{'(no metrics yet)':>6}")
            continue
        first, last = rows[0], rows[-1]
        k0 = first.get("lookahead/rank_frac")
        k1 = last.get("lookahead/rank_frac")
        delta = (k1 - k0) if (k0 is not None and k1 is not None) else None
        total_states = sum(r.get("lookahead/n_states", 0) for r in rows)
        out[label] = dict(rows=rows, rank_first=k0, rank_last=k1, delta=delta,
                          regret_frac=_regret_frac(last))
        print(f"  {label:<26}{last.get('step', len(rows)):>6}{_fmt(last.get('kl'), '9.4f')}"
              f"{_fmt(k0, '10.4f')}{_fmt(k1, '11.4f')}{_fmt(delta, '9.4f')}"
              f"{_fmt(_regret_frac(last), '13.4f')}"
              f"{_fmt(last.get('lookahead/reliable_frac'), '10.4f')}{total_states:>8.0f}")
    print("\n  NOTE: T5c probes EVERY step (its reward requires it); the other arms probe 8")
    print("  states per batch, which are all turn-0 -- the hardest point of a trajectory.")
    print("  T5a-vs-T5b and T5ao-vs-T5bo are turn-0 on both sides and directly comparable;")
    print("  T5c's regret is not comparable to theirs.")
    return out


def checkpoint_table(results: Path):
    """Per-arm trajectory across saved checkpoints."""
    steps = (25, 50, 75, 100)
    print("\nHELD-OUT FEASIBILITY BY CHECKPOINT (34 problems, greedy)")
    print(f"  {'arm':<38}" + "".join(f"{'ck'+str(s):>9}" for s in steps))
    print("  " + "-" * 76)
    for label, prefix, _ in ARMS:
        row = f"  {label:<38}"
        for st_ in steps:
            f = results / f"{prefix}_ckpt{st_}_eval" / "eval_results.json"
            if not f.exists():
                row += f"{'-':>9}"
                continue
            rows = json.load(open(f)).get("per_problem", [])
            v = _solve_rate(rows)
            if v is None:
                v = _valid_rate(rows)
            row += f"{(v if v is not None else float('nan')):>9.3f}"
        print(row)
    print(f"  {'champion (100 steps, reference)':<38}" + f"{'':>27}{0.618:>9.3f}")


def eval_table(results: Path):
    print("\nHELD-OUT EVALUATION (34 problems, data/splits/truss_v1_auto.json)")
    print(f"\n  {'cell':<38}{'n':>4}{'feasible':>10}{'valid feas':>12}"
          f"{'median mass/ref':>17}{'grammar':>9}")
    print("  " + "-" * 90)
    rows = []
    for label, _, evaldir in ARMS:
        rows.append((label, results / evaldir))
    for label, evaldir in REFERENCES:
        rows.append((label, results / evaldir))
    for label, path in rows:
        f = path / "eval_results.json"
        if not f.exists():
            print(f"  {label:<38}{'(pending)':>4}")
            continue
        payload = json.load(open(f))
        agg = payload.get("aggregate", {})
        rows_pp = payload.get("per_problem", [])
        n = agg.get("n_problems", 0)
        feas = agg.get("feasibility_rate")
        valid = _solve_rate(rows_pp)
        if valid is None:
            valid = _valid_rate(rows_pp)
        if valid is None:
            valid = agg.get("valid_feasibility_rate", feas)
        mass = agg.get("median_mass_ratio")
        gram = agg.get("mean_grammar_success_rate", agg.get("mean_grammar_success"))
        partial = "" if agg.get("complete", True) else "  (partial)"
        print(f"  {label:<38}{n:>4}{_fmt(feas, '10.3f')}{_fmt(valid, '12.3f')}"
              f"{_fmt(mass, '17.3f')}{_fmt(gram, '9.3f')}{partial}")


def contrasts(results: Path):
    print("\nCONTRASTS (each isolates one variable; PAIRED on the problems both cells finished,")
    print("because a wall-limit kill leaves cells with different coverage)")

    def _agg(name):
        f = results / name / "eval_results.json"
        return json.load(open(f)).get("aggregate", {}) if f.exists() else None

    def _rows(name):
        f = results / name / "eval_results.json"
        if not f.exists():
            return None
        return {r["problem_id"]: r for r in json.load(open(f)).get("per_problem", [])}

    def _paired(a_name, b_name):
        ra, rb = _rows(a_name), _rows(b_name)
        if not ra or not rb:
            return None
        common = sorted(set(ra) & set(rb))
        if not common:
            return None
        def _feas(r):
            if "solve_rate" in r:
                return r["solve_rate"]          # continuous, and far better powered
            if "state_is_valid" in r:
                return bool(r.get("reaches_solution")) and bool(r.get("state_is_valid"))
            return bool(r.get("valid_feasible", r.get("reaches_solution")))
        fa = sum(_feas(ra[p]) for p in common) / len(common)
        fb = sum(_feas(rb[p]) for p in common) / len(common)
        both = [p for p in common if _feas(ra[p]) and _feas(rb[p])
                and ra[p].get("mass_ratio") and rb[p].get("mass_ratio")]
        ma = st.median([ra[p]["mass_ratio"] for p in both]) if both else None
        mb = st.median([rb[p]["mass_ratio"] for p in both]) if both else None
        wins = sum(1 for p in both if rb[p]["mass_ratio"] < ra[p]["mass_ratio"] - 1e-9)
        return dict(n=len(common), fa=fa, fb=fb, ma=ma, mb=mb, n_mass=len(both), wins=wins)

    pairs = [
        ("does fixing the potential help the system?", "t5a_phi1_eval", "t5b_phi2_eval"),
        ("does the ordering alone carry it?", "t5ao_phionly_v1_eval", "t5bo_phionly_v2_eval"),
        ("does real per-step lookahead add on top?", "t5b_phi2_eval", "t5c_lookahead_eval"),
        ("does ranking help? (matched turns)", "champ_heldout_k1_minmass", "champ_heldout_k8"),
        ("does ranking help? (unmatched turns)", "champ_heldout_k1", "champ_heldout_k8"),
        ("distillation, greedy decoding", "champ_heldout_greedy", "t6_distill_greedy"),
        ("distillation at T=0.8", "champ_heldout_k1", "t6_distill_k1"),
        ("distilled + ranking", "t6_distill_k1", "t6_distill_k8"),
        ("lookahead advantage alone vs Phi_v2 alone", "t5bo_phionly_v2_eval", "t5co_lookahead_only_eval"),
        ("informativeness sampling", "t5bo_phionly_v2_eval", "t5bw_informative_eval"),
        ("DPO initialisation for GRPO", "t5bo_phionly_v2_eval", "t7_dpo_init_eval"),
    ]
    for question, a_name, b_name in pairs:
        pr = _paired(a_name, b_name)
        if pr is None:
            print(f"  {question:<46} (pending)")
            continue
        n, fa, fb = pr["n"], pr["fa"], pr["fb"]
        # noise floor scales as 1/sqrt(n) from the n=25 triplicate
        floor = NOISE_FLOOR_PP * math.sqrt(25.0 / max(n, 1)) / 100.0
        d = fb - fa
        verdict = "within noise" if abs(d) < floor else ("BETTER" if d > 0 else "WORSE")
        mass_note = ""
        if pr["ma"] and pr["mb"]:
            mass_note = (f"   mass {pr['ma']:.3f} -> {pr['mb']:.3f} "
                         f"({(pr['mb']-pr['ma'])/pr['ma']*100:+.1f}%, lighter on "
                         f"{pr['wins']}/{pr['n_mass']})")
        print(f"  {question:<46} n={n:>3}  feasible {fa:.3f} -> {fb:.3f} "
              f"({d*100:+.1f}pp, floor +/-{floor*100:.1f}pp) {verdict}{mass_note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--results", default="results/eval")
    args = ap.parse_args()
    training_table(Path(args.logs))
    checkpoint_table(Path(args.results))
    eval_table(Path(args.results))
    contrasts(Path(args.results))
    print("\nReference points (no LLM, 130 problems, validity-gated, legal grammar):")
    print("  greedy critical-member (depth 0)  72.3% feasible, mass 0.872,  6 FEA")
    print("  fully-stressed design (analytic)  75.4% feasible, mass 0.474, 12 FEA")
    print("  lookahead Phi_v2                  73.8% feasible, mass 0.644, 2357 FEA")
    print("  lookahead Phi_v2 + compound action 76.2% feasible, mass 0.437, 2359 FEA")


if __name__ == "__main__":
    main()

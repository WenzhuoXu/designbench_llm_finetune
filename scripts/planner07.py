"""
Planner v7: make the sizing margin a search dimension.

v6 added an archive and restart and returned exactly v5's smoke number; the restart fired 0.28
times per episode, so it was close to a no-op. Widening proposals from 8 to 12 did nothing
either. Both levers are spent.

One parameter has been fixed at 1.05 since the first version and never searched: the margin the
fully-stressed pass sizes to. It is the single knob that trades strength against mass -- a
larger margin oversizes every member and pushes mass up, a smaller one leaves less headroom but
frees mass. Since the residual failures are the ones where the mass cap binds, that is exactly
the axis the search has been blind on.

So every branch is now rolled out at several margins and the argmax picks across (intervention,
margin) jointly rather than intervention alone. This costs no model calls -- rollouts are FEA --
and it lets the same proposal succeed at a margin where it would otherwise fail.

The FSD control is rolled out at the same set of margins and keeps its best, so the baseline
gets the identical advantage and the comparison stays honest. Both controls in this script.
"""
import os, sys, json, math, re, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import build_table, true_fos
from probe26_presentation import sign_test
from planner01 import apply_tool, fsd_pass, TOOL_RE, SONNET45
from planner03 import potential
from planner04 import TOOLS_DOC
from planner05 import FEEDBACK_DOC

MAX_STEPS = 8
NPROP = 8
MARGINS = (0.98, 1.05, 1.15, 1.30)


def rollout_m(truss, goals, steps, margin):
    st = _analyze_truss(truss, goals)
    cur = truss
    for _ in range(max(0, steps)):
        if st.get("is_feasible"):
            return True, st, cur
        nxt = fsd_pass(cur, goals, margin)
        if nxt is cur:
            break
        cur = nxt
        st = _analyze_truss(cur, goals)
    return bool(st.get("is_feasible")), st, cur


def best_rollout(truss, goals, steps, gb, gy):
    """Roll out at each margin; return (feasible, best potential, best resulting truss)."""
    bv, bt = -1e9, None
    for m in MARGINS:
        ok, st, cur = rollout_m(truss, goals, steps, m)
        if ok:
            return True, 1e9, cur
        v = potential(st, goals, gb, gy)
        if v > bv:
            bv, bt = v, cur
    return False, bv, bt


def step_search_m(truss, goals, calls, remaining, gb, gy):
    base_ok, base_v, _ = best_rollout(truss, goals, remaining, gb, gy)
    if base_ok:
        return True, truss, "the structure was already on track"
    best_v, best_truss, best_note = base_v, None, None

    singles = []
    for nm, argstr in calls:
        try:
            cand = apply_tool(truss, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is truss:
            continue
        ok, v, _ = best_rollout(cand, goals, remaining, gb, gy)
        if ok:
            return True, cand, "%s(%s) reached feasibility" % (nm, argstr.strip())
        singles.append((v, cand, nm, argstr))
        if v > best_v + 1e-9:
            best_v, best_truss = v, cand
            best_note = "%s(%s) was selected" % (nm, argstr.strip())
    if not singles:
        return False, best_truss, best_note or "none of your proposals could be applied"

    singles.sort(key=lambda x: -x[0])
    anchor = singles[0][1]
    anchor_lbl = "%s(%s)" % (singles[0][2], singles[0][3].strip())
    pair_best = None
    for _, _, nm, argstr in singles[1:]:
        try:
            cand = apply_tool(anchor, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is anchor:
            continue
        ok, v, _ = best_rollout(cand, goals, remaining, gb, gy)
        if ok:
            return True, cand, "%s then %s(%s) reached feasibility" % (anchor_lbl, nm, argstr.strip())
        if pair_best is None or v > pair_best[0]:
            pair_best = (v, cand, "%s then %s(%s)" % (anchor_lbl, nm, argstr.strip()))
        if v > best_v + 1e-9:
            best_v, best_truss = v, cand
            best_note = "%s then %s(%s) were combined" % (anchor_lbl, nm, argstr.strip())

    if pair_best is not None:
        anchor2, lbl2 = pair_best[1], pair_best[2]
        for _, _, nm, argstr in singles[1:]:
            try:
                cand = apply_tool(anchor2, goals, nm, argstr)
            except Exception:
                cand = None
            if cand is None or cand is anchor2:
                continue
            ok, v, _ = best_rollout(cand, goals, remaining, gb, gy)
            if ok:
                return True, cand, "%s then %s(%s) reached feasibility" % (lbl2, nm, argstr.strip())
            if v > best_v + 1e-9:
                best_v, best_truss = v, cand
                best_note = "%s then %s(%s) were combined" % (lbl2, nm, argstr.strip())
    return False, best_truss, best_note or "none of your proposals beat doing nothing"


def episode(spec, arm, region, token, k, max_steps, nprop=NPROP):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        # the control gets the same margin search, so the advantage is not the planner's alone
        ok, _, _ = best_rollout(truss, goals, max_steps, gb, gy)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + (TOOLS_DOC % (nprop, nprop)) + FEEDBACK_DOC
    history, wins, props = [], 0, 0
    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "wins": wins, "props": props}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "wins": wins, "props": props}

        remaining = max_steps - turn - 1
        tbl = build_table(truss, true_fos(truss))
        obs = ("A fully-stressed sizing pass has just been applied. Resulting state:\n"
               + DP.format_eval_result(state) + tbl)
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + "\n\n" + obs}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text": "[Simulation Result]\n" + h["o"]}]})
        text = H.call(convo, system, SONNET45, region, token) or ""
        calls = [(n, a) for (n, a) in TOOL_RE.findall(text)
                 if n not in ("FSD_ALL", "FSD_ON")][:nprop]
        props += len(calls)

        ok, chosen, note = step_search_m(truss, goals, calls, remaining, gb, gy)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins + 1, "props": props}
        if chosen is not None:
            truss = chosen
            state = _analyze_truss(truss, goals)
            wins += 1
        history.append({"t": text.strip()[:1200],
                        "o": ("Search outcome: %s.\n" % note)
                             + DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
    ok, _, _ = best_rollout(truss, goals, 2, gb, gy)
    return {"problem_id": pid, "arm": arm, "feasible": ok, "wins": wins, "props": props}


def work(t):
    spec, arm, region, token, k, ms = t
    try:
        return episode(spec, arm, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "feasible": False,
                "err": str(e)[:200]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--arms", default="planner,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner07.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | horizon %d | margins %s | episodes %d"
          % (a.start, a.start + a.n, a.max_steps, MARGINS, len(tasks)), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 40 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

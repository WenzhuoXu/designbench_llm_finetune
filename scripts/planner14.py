"""
Planner v14: the model refines what the search has already scored.

Six tests, six failures, and every one of them asked the model to act blind. As sole proposer
it never saw a rollout result; as pooled proposer, pre-filter and distribution guide the same.
Sampling beat it each time because sampling explores and the model was guessing.

The one thing sampling cannot do is EXTRAPOLATE. It draws independently every turn, so when a
direction is working -- shrinking that region freed mass, adding that member helped -- nothing
in the procedure pushes further along it. That is a job for something that can read a pattern
off a handful of scored examples.

So the model is shown the search's own results this turn: the best few random candidates with
their rollout potentials and the do-nothing baseline, and asked for four moves that build on
whatever is working. Those four are evaluated alongside the sixteen. The arm starts from the
same draws as the search, so it can only add.

  refine    16 random draws scored, model proposes 4 refinements, all 20 evaluated
  search20  20 random draws evaluated                     -- matched rollout budget
  fsd       the sizing rule alone
"""
import os, sys, json, math, random, zlib, argparse, time
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
from planner03 import rollout, potential
from planner09 import random_proposals
from planner13 import seed_of

MAX_STEPS = 8
NRAND = 16
NREFINE = 4
SHOWN = 5

REFINE_SYSTEM = """A fully-stressed sizing pass runs automatically each turn, rescaling every
member towards its required factor of safety. It cannot change load paths and cannot escape its
own fixed point once the mass cap binds.

Candidate modifications have already been drawn at random and simulated for you. You are shown
the best of them with a score: higher is better, and a score above the do-nothing baseline means
that move helped. Your job is to propose %d moves that BUILD ON what the scores show -- push a
direction that is working further, apply the same kind of change elsewhere, or combine two ideas
that both scored well. Do not repeat a candidate you were shown.

Emit exactly %d lines, each <tool>...</tool>, from:
  ADD_MEMBER(joint1, joint2) | REMOVE_MEMBER(i) | MOVE_JOINT(joint, x, y)
  TRIM(fos_threshold, factor) | SCALE([ids], factor)
Nothing else."""


def score_all(truss, goals, cands, remaining, gb, gy):
    """Roll out each candidate; return (feasible_truss or None, scored list)."""
    scored = []
    for nm, argstr in cands:
        try:
            cand = apply_tool(truss, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is truss:
            continue
        ok, st, _ = rollout(cand, goals, remaining)
        if ok:
            return cand, scored
        scored.append((potential(st, goals, gb, gy), cand, nm, argstr))
    return None, scored


def episode(spec, arm, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        ok, _, _ = rollout(truss, goals, max_steps)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    rng = random.Random(seed_of(pid))
    ndraw = NRAND if arm == "refine" else NRAND + NREFINE
    used_model = 0

    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "used_model": used_model}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True, "used_model": used_model}

        remaining = max_steps - turn - 1
        base_ok, base_st, _ = rollout(truss, goals, remaining)
        if base_ok:
            return {"problem_id": pid, "arm": arm, "feasible": True, "used_model": used_model}
        base_v = potential(base_st, goals, gb, gy)

        won, scored = score_all(truss, goals, random_proposals(truss, rng, ndraw),
                                remaining, gb, gy)
        if won is not None:
            truss = won
            state = _analyze_truss(truss, goals)
            if state.get("is_feasible"):
                return {"problem_id": pid, "arm": arm, "feasible": True, "used_model": used_model}
            continue

        if arm == "refine" and scored:
            scored.sort(key=lambda x: -x[0])
            top = scored[:SHOWN]
            lines = "\n".join("  %s(%s)   score %+.4f" % (nm, a.strip(), v)
                              for v, _, nm, a in top)
            body = ("Current state after the sizing pass:\n" + DP.format_eval_result(state)
                    + build_table(truss, true_fos(truss))
                    + "\n\nAlready simulated this turn (do-nothing baseline score %+.4f):\n" % base_v
                    + lines + "\n\nPropose %d moves that build on this." % NREFINE)
            convo = [{"role": "user", "content": [{"text":
                      DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                      + DP.format_eval_result(initial) + "\n\n" + body}]}]
            text = H.call(convo, REFINE_SYSTEM % (NREFINE, NREFINE),
                          SONNET45, region, token) or ""
            extra = [(nm, a) for (nm, a) in TOOL_RE.findall(text)
                     if nm not in ("FSD_ALL", "FSD_ON")][:NREFINE]
            if extra:
                used_model += 1
                won2, scored2 = score_all(truss, goals, extra, remaining, gb, gy)
                if won2 is not None:
                    truss = won2
                    state = _analyze_truss(truss, goals)
                    if state.get("is_feasible"):
                        return {"problem_id": pid, "arm": arm, "feasible": True,
                                "used_model": used_model}
                    continue
                scored.extend(scored2)

        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        best_v, best_truss = scored[0][0], scored[0][1]
        # greedy pair on the anchor, as in every previous version
        anchor = scored[0][1]
        for v0, _, nm, argstr in scored[1:8]:
            try:
                cand = apply_tool(anchor, goals, nm, argstr)
            except Exception:
                cand = None
            if cand is None or cand is anchor:
                continue
            ok, st, _ = rollout(cand, goals, remaining)
            if ok:
                return {"problem_id": pid, "arm": arm, "feasible": True, "used_model": used_model}
            v = potential(st, goals, gb, gy)
            if v > best_v + 1e-9:
                best_v, best_truss = v, cand
        if best_v > base_v + 1e-9:
            truss = best_truss
            state = _analyze_truss(truss, goals)
    ok, _, _ = rollout(truss, goals, 2)
    return {"problem_id": pid, "arm": arm, "feasible": ok, "used_model": used_model}


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
    ap.add_argument("--arms", default="refine,search20,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner14.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    arms = tuple(a.arms.split(","))
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps) for s in specs for arm in arms]
    print("problems %d-%d | arms %s | episodes %d"
          % (a.start, a.start + a.n, ",".join(arms), len(tasks)), flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 60 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    by = {arm: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in out if r["arm"] == arm}
          for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("paired on %d problems\n" % len(pids))
    for arm in sorted(arms, key=lambda x: -rate[x]):
        print("  %-9s %.4f" % (arm, rate[arm]))
    for arm in arms:
        if arm == "fsd":
            continue
        u, d, p = sign_test([by["fsd"][x] for x in pids], [by[arm][x] for x in pids])
        print("\n  %-9s vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (arm, rate[arm] - rate["fsd"], u, d, p))
    if "refine" in arms and "search20" in arms:
        u, d, p = sign_test([by["search20"][x] for x in pids], [by["refine"][x] for x in pids])
        print("\n  refine vs search20 at matched rollout budget: %+.4f   discordant %d-%d   p = %.3g"
              % (rate["refine"] - rate["search20"], u, d, p))
    um = [r.get("used_model", 0) for r in out if r["arm"] == "refine"]
    if um:
        print("  model consulted on %.1f turns per episode" % (sum(um) / len(um)))

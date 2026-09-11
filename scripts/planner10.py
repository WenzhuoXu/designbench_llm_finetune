"""
Planner v10: one pool, both sources. Does the model contribute anything the search can use?

Held-out, three arms in one script: random proposals 0.8465, LLM planner 0.7674, fully-stressed
design 0.4930. Uniform random draws from the tool library beat the model as a proposer by
7.9pp (discordant 43-9, p=2.0e-06). Everything gained this session came from the search, not
from the model.

So the model stops being the sole proposer and becomes one source feeding a shared pool. Each
turn the search sees the model's proposals AND an equal number of random draws, and picks by
the same rollout argmax. If the model's suggestions carry anything, the pool finds them; if they
are simply worse, the search ignores them and the arm degrades to random rather than to the
planner. Which source each accepted move came from is recorded, so the answer is a count rather
than an inference.

Arms, all in this script on the same problems:
  mixed      8 model proposals + 8 random draws per turn
  random8    8 random draws            -- the current best, unchanged
  random16   16 random draws           -- is the random arm simply proposal-starved?
  fsd        the sizing rule alone
"""
import os, sys, json, math, random, argparse, time
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
from planner04 import TOOLS_DOC
from planner05 import FEEDBACK_DOC
from planner09 import random_proposals

MAX_STEPS = 8
NMODEL = 8


def search_tagged(truss, goals, tagged, remaining, gb, gy):
    """Singles then greedy pairs, over (source, name, args). Returns the winning source too."""
    base_ok, base_st, _ = rollout(truss, goals, remaining)
    if base_ok:
        return True, truss, None, "already on track"
    best_v = potential(base_st, goals, gb, gy)
    best_truss, best_src, best_note = None, None, None

    singles = []
    for src, nm, argstr in tagged:
        try:
            cand = apply_tool(truss, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is truss:
            continue
        ok, st, _ = rollout(cand, goals, remaining)
        if ok:
            return True, cand, src, "%s(%s) reached feasibility" % (nm, argstr.strip())
        v = potential(st, goals, gb, gy)
        singles.append((v, cand, src, nm, argstr))
        if v > best_v + 1e-9:
            best_v, best_truss, best_src = v, cand, src
            best_note = "%s(%s) was selected" % (nm, argstr.strip())
    if not singles:
        return False, best_truss, best_src, best_note or "nothing could be applied"

    singles.sort(key=lambda x: -x[0])
    anchor, anchor_src = singles[0][1], singles[0][2]
    anchor_lbl = "%s(%s)" % (singles[0][3], singles[0][4].strip())
    for _, _, src, nm, argstr in singles[1:]:
        try:
            cand = apply_tool(anchor, goals, nm, argstr)
        except Exception:
            cand = None
        if cand is None or cand is anchor:
            continue
        ok, st, _ = rollout(cand, goals, remaining)
        if ok:
            return True, cand, src, "%s then %s(%s) reached feasibility" % (anchor_lbl, nm, argstr.strip())
        v = potential(st, goals, gb, gy)
        if v > best_v + 1e-9:
            best_v, best_truss, best_src = v, cand, src
            best_note = "%s then %s(%s) were combined" % (anchor_lbl, nm, argstr.strip())
    return False, best_truss, best_src, best_note or "nothing beat doing nothing"


def episode(spec, arm, region, token, k, max_steps, nmodel=NMODEL):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        ok, _, _ = rollout(truss, goals, max_steps)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    nrand = {"mixed": 8, "random8": 8, "random16": 16}.get(arm, 8)
    use_model = arm == "mixed"
    rng = random.Random(abs(hash(pid)) & 0xffffffff)
    system = DP.TRUSS_SYSTEM_PROMPT + "\n\n" + (TOOLS_DOC % (nmodel, nmodel)) + FEEDBACK_DOC
    history, from_model, from_random = [], 0, 0

    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "from_model": from_model, "from_random": from_random}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "from_model": from_model, "from_random": from_random}

        remaining = max_steps - turn - 1
        tagged = [("random", nm, a) for nm, a in random_proposals(truss, rng, nrand)]
        text = ""
        if use_model:
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
            tagged = [("model", nm, a) for (nm, a) in TOOL_RE.findall(text)
                      if nm not in ("FSD_ALL", "FSD_ON")][:nmodel] + tagged

        ok, chosen, src, note = search_tagged(truss, goals, tagged, remaining, gb, gy)
        if src == "model":
            from_model += 1
        elif src == "random":
            from_random += 1
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "from_model": from_model, "from_random": from_random}
        if chosen is not None:
            truss = chosen
            state = _analyze_truss(truss, goals)
        if use_model:
            history.append({"t": text.strip()[:1200],
                            "o": ("Search outcome: %s.\n" % note)
                                 + DP.format_eval_result(state)
                                 + build_table(truss, true_fos(truss))})
    ok, _, _ = rollout(truss, goals, 2)
    return {"problem_id": pid, "arm": arm, "feasible": ok,
            "from_model": from_model, "from_random": from_random}


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
    ap.add_argument("--arms", default="mixed,random8,random16,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner10.jsonl"))
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
    if "fsd" in arms:
        for arm in arms:
            if arm == "fsd":
                continue
            u, d, p = sign_test([by["fsd"][x] for x in pids], [by[arm][x] for x in pids])
            print("\n  %-9s vs FSD control: %+.4f   discordant %d-%d   exact sign p = %.3g"
                  % (arm, rate[arm] - rate["fsd"], u, d, p))
    fm = [r.get("from_model", 0) for r in out if r["arm"] == "mixed"]
    fr = [r.get("from_random", 0) for r in out if r["arm"] == "mixed"]
    if fm:
        print("\n  mixed pool, accepted moves per episode: %.2f from the model, %.2f from random"
              % (sum(fm) / len(fm), sum(fr) / len(fr)))

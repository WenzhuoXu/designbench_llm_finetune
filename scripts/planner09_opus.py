"""
The control that decides whether the planner is doing anything: random proposals, same search.

The planner reaches 0.7791 against fully-stressed design's 0.4930 on the held-out set. But it
does not merely propose -- it proposes into a search that evaluates every candidate by rollout,
composes pairs and triples, and takes an argmax. That search is powerful on its own, and it has
never been run without the model.

So this arm replaces ONLY the proposer. Random draws from the identical tool library, the same
number of proposals per turn, the same singles-then-pairs-then-triples search, the same
potential, the same horizon, the same do-nothing branch. Everything that is not the model is
held fixed.

  planner   proposals from Sonnet 4.5
  random    proposals drawn uniformly from the tool library with random arguments
  fsd       the sizing rule alone

If random ties the planner, the credit belongs to the search and the tool library, and the
model is decoration -- which is worth finding out now rather than after a claim. If the planner
wins, its margin over random is the part attributable to the model, and that is the number the
project actually needs.
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
from planner01 import fsd_pass, TOOL_RE
SONNET45 = "us.anthropic.claude-opus-4-1-20250805-v1:0"   # capability check: same harness, stronger model
from planner03 import rollout, potential
from planner04 import TOOLS_DOC
from planner05 import step_search, FEEDBACK_DOC

MAX_STEPS = 8
NPROP = 8


def joint_xy(truss, j):
    try:
        c = truss.joints[j].coordinates
        return float(c[0]), float(c[1])
    except Exception:
        try:
            c = truss.joints[j].coords
            return float(c[0]), float(c[1])
        except Exception:
            return None


def random_proposals(truss, rng, nprop):
    """Draw nprop tool calls uniformly from the same library the planner is offered."""
    n = len(truss.members)
    try:
        nj = len(truss.joints)
    except Exception:
        nj = 0
    out = []
    for _ in range(nprop):
        kind = rng.choice(["ADD_MEMBER", "REMOVE_MEMBER", "MOVE_JOINT", "TRIM", "SCALE"])
        if kind == "ADD_MEMBER" and nj >= 2:
            a, b = rng.sample(range(nj), 2)
            out.append(("ADD_MEMBER", "%d, %d" % (a, b)))
        elif kind == "REMOVE_MEMBER" and n > 1:
            out.append(("REMOVE_MEMBER", "%d" % rng.randrange(n)))
        elif kind == "MOVE_JOINT" and nj >= 1:
            j = rng.randrange(nj)
            xy = joint_xy(truss, j)
            if xy is None:
                out.append(("TRIM", "%.2f, %.2f" % (rng.uniform(1.5, 6.0), rng.uniform(0.75, 0.95))))
                continue
            x, y = xy
            out.append(("MOVE_JOINT", "%d, %.4f, %.4f"
                        % (j, x + rng.uniform(-0.4, 0.4), y + rng.uniform(-0.4, 0.4))))
        elif kind == "TRIM":
            out.append(("TRIM", "%.2f, %.2f" % (rng.uniform(1.5, 6.0), rng.uniform(0.75, 0.95))))
        else:
            k = rng.randint(1, max(1, min(4, n)))
            ids = rng.sample(range(n), k) if n >= k else [0]
            out.append(("SCALE", "[%s], %.3f" % (",".join(str(i) for i in ids),
                                                 rng.uniform(0.75, 1.6))))
    return out


def episode(spec, arm, region, token, k, max_steps, nprop=NPROP):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))

    if arm == "fsd":
        ok, _, _ = rollout(truss, goals, max_steps)
        return {"problem_id": pid, "arm": arm, "feasible": ok}

    rng = random.Random(abs(hash(pid)) & 0xffffffff)
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
        if arm == "random":
            calls = random_proposals(truss, rng, nprop)
        else:
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

        ok, chosen, note = step_search(truss, goals, calls, remaining, gb, gy)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True,
                    "wins": wins + 1, "props": props}
        if chosen is not None:
            truss = chosen
            state = _analyze_truss(truss, goals)
            wins += 1
        if arm != "random":
            history.append({"t": text.strip()[:1200],
                            "o": ("Search outcome: %s.\n" % note)
                                 + DP.format_eval_result(state)
                                 + build_table(truss, true_fos(truss))})
    ok, _, _ = rollout(truss, goals, 2)
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
    ap.add_argument("--arms", default="planner,random,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner09.jsonl"))
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
            if (i + 1) % 40 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    by = {arm: {r["problem_id"]: (1.0 if r["feasible"] else 0.0) for r in out if r["arm"] == arm}
          for arm in arms}
    pids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    rate = {arm: sum(by[arm][p] for p in pids) / max(len(pids), 1) for arm in arms}
    print("paired on %d problems\n" % len(pids))
    for arm in arms:
        print("  %-9s %.4f" % (arm, rate[arm]))
    for other in [x for x in arms if x != "planner"]:
        if "planner" in arms:
            u, d, p = sign_test([by[other][x] for x in pids], [by["planner"][x] for x in pids])
            print("\n  planner vs %-7s: %+.4f   discordant %d-%d   exact sign p = %.3g"
                  % (other, rate["planner"] - rate[other], u, d, p))

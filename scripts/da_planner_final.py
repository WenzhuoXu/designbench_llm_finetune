"""
Track 3 -- the LLM planner result, measured properly.

The planner (Sonnet 4.5 proposing tool calls into a rollout-composition search, with a
fully-stressed sizing pass applied automatically each turn) has been reported at 0.7884 on
held-out problems 150-580, against fully-stressed design at 0.4930. That number was produced
by planner13.py, which measures planner / search / fsd. It has never been measured against the
ablation that actually isolates what the tool library and the potential-argmax contribute on
top of the bare model: the same model, same problems, same horizon, emitting raw DesignBench
grammar actions with no tool library and no rollout search.

This script runs all three arms in ONE process on the SAME problems, paired:

  planner    planner13.episode(arm='planner'). Sonnet 4.5 emits up to 8 tool calls per turn
             from the library; each is applied, rolled forward to the horizon by the sizing
             rule, scored by potential; singles, then greedy pairs, then a greedy third; a
             do-nothing branch competes. Horizon 8.
  fsd        planner13.episode(arm='fsd'). Fully-stressed design alone, the standard
             heuristic, same horizon. No model.
  plain_llm  planner01.episode(arm='llm'). The bare model emitting <action> grammar actions
             (SCALE_PARAM / ADD_MEMBER / REMOVE_MEMBER / MOVE_JOINT / MODIFY_PARAM, compounds
             allowed), k per turn, one-step argmax over them by FOS-vs-mass, and nothing else:
             no FSD pass, no tool library, no rollout to the horizon.

The three arms are imported, not reimplemented, so the planner arm here is bit-identical to the
one that produced 0.7884 and the fsd arm to the one that produced 0.4930. Nothing is compared
to a remembered number: every arm runs here.

Two deliberate choices, both charitable to the ablation:
  - plain_llm gets the SAME horizon (--max-steps, default 8) as the planner, not the 4 turns
    planner01 defaulted to. It therefore gets the same number of model calls to work with.
  - plain_llm keeps the k-candidate one-step argmax from planner01's 'llm' arm, so it is not
    penalised for having no way to choose among its own proposals.

Model calls are counted per episode by wrapping api_grammar_2x2.call, so the cost of the two
LLM arms is measured rather than assumed.

Seeding is zlib.crc32 per problem inside planner13 (never hash()). Paired exact sign tests,
discordant counts reported.

Smoke:
  python scripts/da_planner_final.py --start 150 --n 60 \
      --out results/api_guidance/da_planner_final_smoke.jsonl

Full held-out run (430 problems, 150-580):
  python scripts/da_planner_final.py --start 150 --n 430 \
      --out results/api_guidance/da_planner_final_430.jsonl
"""
import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for _p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import api_grammar_2x2 as H                      # noqa: E402
from probe26_presentation import sign_test       # noqa: E402
import planner01                                 # noqa: E402  (plain_llm arm)
import planner13                                 # noqa: E402  (planner + fsd arms)

ARMS = ("planner", "fsd", "plain_llm")
LLM_ARMS = ("planner", "plain_llm")

# ---------------------------------------------------------------- model-call counting
# planner01 and planner13 both do `import api_grammar_2x2 as H` and call `H.call(...)`, so the
# attribute is resolved on the shared module object at call time -- patching it here counts
# every model call either arm makes, without touching their logic.
_TL = threading.local()
_REAL_CALL = H.call


def _counting_call(*a, **kw):
    _TL.n = getattr(_TL, "n", 0) + 1
    return _REAL_CALL(*a, **kw)


H.call = _counting_call


# ---------------------------------------------------------------- arms
def run_arm(spec, arm, region, token, k, max_steps):
    """Dispatch to the imported, unmodified episode for each arm."""
    if arm == "planner":
        return planner13.episode(spec, "planner", region, token, k, max_steps)
    if arm == "fsd":
        return planner13.episode(spec, "fsd", region, token, k, max_steps)
    if arm == "plain_llm":
        r = planner01.episode(spec, "llm", region, token, k, max_steps)
        r["arm"] = "plain_llm"
        return r
    raise ValueError("unknown arm %r" % arm)


def work(t):
    spec, arm, region, token, k, ms = t
    _TL.n = 0
    t0 = time.time()
    try:
        r = run_arm(spec, arm, region, token, k, ms)
    except Exception as e:
        r = {"problem_id": spec.get("problem_id"), "arm": arm, "feasible": False,
             "err": "%s: %s" % (type(e).__name__, str(e)[:200])}
    r["arm"] = arm
    r["calls"] = getattr(_TL, "n", 0)
    r["secs"] = round(time.time() - t0, 1)
    return r


# ---------------------------------------------------------------- reporting
def report(out, arms):
    by = {a: {} for a in arms}
    calls = {a: {} for a in arms}
    for r in out:
        a = r.get("arm")
        if a in by and r.get("problem_id"):
            by[a][r["problem_id"]] = 1.0 if r.get("feasible") else 0.0
            calls[a][r["problem_id"]] = int(r.get("calls", 0) or 0)
    present = [a for a in arms if by[a]]
    if not present:
        print("no results")
        return
    pids = sorted(set.intersection(*[set(by[a]) for a in present]))
    n = len(pids)
    errs = sum(1 for r in out if r.get("err"))
    print("\npaired on %d problems   (episodes %d, hard errors %d)\n" % (n, len(out), errs))
    if n == 0:
        return
    rate = {a: sum(by[a][p] for p in pids) / n for a in present}
    for a in sorted(present, key=lambda x: -rate[x]):
        print("  %-10s feasibility %.4f   (%d/%d)"
              % (a, rate[a], int(round(rate[a] * n)), n))

    def pair(lo, hi):
        if lo not in present or hi not in present:
            return
        u, d, p = sign_test([by[lo][x] for x in pids], [by[hi][x] for x in pids])
        print("\n  %s vs %s: %+.4f   discordant %d-%d (%s wins %d, %s wins %d)"
              "   exact sign p = %.4g" % (hi, lo, rate[hi] - rate[lo], u, d, hi, u, lo, d, p))

    pair("fsd", "planner")
    pair("plain_llm", "planner")
    pair("plain_llm", "fsd")

    print("")
    for a in LLM_ARMS:
        if a in present and calls[a]:
            v = [calls[a][p] for p in pids]
            print("  %-10s model calls per problem: mean %.2f  (min %d, max %d, total %d)"
                  % (a, sum(v) / len(v), min(v), max(v), sum(v)))
    print("  %-10s model calls per problem: 0 (model-free)" % "fsd")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=150, help="150 = start of the held-out split")
    ap.add_argument("--k", type=int, default=8, help="candidates per turn for plain_llm")
    ap.add_argument("--max-steps", type=int, default=8, help="horizon, shared by all arms")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--resume", action="store_true",
                    help="skip (problem_id, arm) pairs already in --out")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_planner_final.jsonl"))
    a = ap.parse_args()

    arms = tuple(x for x in a.arms.split(",") if x)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    done, out = set(), []
    if (a.resume or a.report_only) and outp.exists():
        for line in open(outp):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("err"):          # retry hard failures on resume
                continue
            done.add((r.get("problem_id"), r.get("arm")))
            out.append(r)
        print("resuming: %d episodes already on disk" % len(out), flush=True)

    if a.report_only:
        report(out, arms)
        sys.exit(0)

    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]
    specs = [json.load(open(f)) for f in files]
    tasks = [(s, arm, a.region, token, a.k, a.max_steps)
             for s in specs for arm in arms
             if (s.get("problem_id"), arm) not in done]

    print("problems %d-%d (%d specs) | horizon %d | arms %s | episodes to run %d | workers %d"
          % (a.start, a.start + a.n, len(specs), a.max_steps, ",".join(arms),
             len(tasks), a.workers), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, \
            open(outp, "a" if done else "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 30 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("\nwall %.0fs" % (time.time() - t0), flush=True)
    report(out, arms)

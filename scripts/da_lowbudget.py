"""
TRACK 2 -- the low-evaluation regime.

Every comparison in this project so far handed the search effectively unlimited simulation:
32 candidates rolled to horizon 12, ~287 analyses per problem. In that regime uniform sampling
from the tool library wins by brute force (0.9721) and the model contributes nothing (0.7884).
Real engineering design does not have that budget. The untested claim is that a language model
buys SAMPLE EFFICIENCY: better feasibility when only a handful of hard evaluations are affordable.

So: the same search, the same tool library, the same problems, at MATCHED hard evaluation
budgets of 5, 10, 20, 40 and 80 analyses per problem.

  llm      Sonnet 4.5 proposes the candidate tool calls, one model call per turn
  random   the identical search with candidates drawn uniformly from the same library
  fsd      fully-stressed design alone, the standard heuristic control

THE BUDGET. A "hard evaluation" is one call to the FEA (_analyze_truss). The Domain already
counts these -- TrussDomain.evaluate runs the analysis exactly once per uncached state and
increments self.calls -- so the counter is the budget meter. BudgetedTruss raises the moment a
fresh analysis would exceed the cap, and BudgetExhausted derives from BaseException so that
none of the library's `except Exception:` guards can silently absorb it and hand an arm free
simulation. As an independent cross-check the script also wraps _analyze_truss itself with a
thread-local counter and logs both numbers; they must agree.

THE SEARCH, identical for llm and random. Each turn the candidate pool is the deterministic
fully-stressed move SIZE_PASS(1.05) plus up to NPROP-1 proposals from whichever proposer the arm
uses. Every candidate costs exactly one analysis. Score by the potential, take the argmax, keep
going until a feasible design appears or the budget is gone. There are no rollouts: at 5
analyses a rollout to horizon is the entire budget, so the horizon is one everywhere and the
budget buys candidates and turns instead. A candidate already evaluated at the current state is
not paid for twice, under the same rule for both arms.

Including the fully-stressed move in every pool is deliberate. It costs both arms the same one
analysis, it cannot bias the contrast, and without it both arms sit below the FSD control at
small budgets and the comparison measures nothing.

Outcome per episode: did any state the arm actually paid to evaluate come back feasible.
"""
import os, sys, json, math, random, zlib, argparse, time, re, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- independent FEA counter, installed BEFORE TrussDomain binds the function ----
import llm_finetune.envs.truss_env as TE
_tls = threading.local()
_RAW_ANALYZE = TE._analyze_truss


def _counted_analyze(truss, goals):
    _tls.n = getattr(_tls, "n", 0) + 1
    return _RAW_ANALYZE(truss, goals)


TE._analyze_truss = _counted_analyze

from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import build_table, true_fos
from probe26_presentation import sign_test
from planner01 import SONNET45

import da_domtools                      # registers the truss topology tools on TrussDomain
from da_domain import TrussDomain
from da_search import size_pass, potential, rollout
import da_search2 as S2

PARAM = "r"
BUDGETS = (5, 10, 20, 40, 80)
NPROP = 5              # 1 mandatory fully-stressed move + up to 4 proposals
MAX_TURNS = 30
ARMS = ("llm", "random", "fsd")


# ---------------------------------------------------------------- budgeted domain

class BudgetExhausted(BaseException):
    """BaseException on purpose: `except Exception` in the tool library must not eat it."""


class BudgetedTruss(TrussDomain):
    def __init__(self, budget):
        super().__init__()
        self.budget = int(budget)
        self.found = False

    def evaluate(self, st):
        if st["cache"] is None:
            if self.calls >= self.budget:
                raise BudgetExhausted()
            out = super().evaluate(st)
            try:
                if out.get("is_feasible"):
                    self.found = True
            except Exception:
                pass
            return out
        return st["cache"]


# ---------------------------------------------------------------- the model's grammar

TOOLS_DOC = """
You may propose moves ONLY from this library. Emit each one on its own line wrapped in
<action></action> tags. Emit exactly %d candidates -- they must be genuinely DIFFERENT, because
only one of them will be kept.

  SIZE_PASS(margin)            fully-stressed resize: every member's radius moves towards
                               `margin` times its own requirement at once. margin in [0.95, 1.35].
                               This is the strongest single move in the library; margin > 1
                               adds material and mass, margin < 1 sheds it.
  SCALE([i,j,k], factor)       multiply the outer radius of the listed members by factor.
                               factor in [0.75, 1.60].
  TRIM(threshold, factor)      multiply the radius of every member whose margin exceeds
                               threshold by factor. threshold in [1.2, 5.0], factor in [0.75, 0.95].
                               This is the move that sheds mass from over-strong members.
  ADD_MEMBER(j1, j2)           add a 6061-T6 aluminium pipe (r=0.030, t=0.004) between two joints.
  REMOVE_MEMBER(i)             delete member i.
  MOVE_JOINT(j, x, y)          move joint j to (x, y).

A member's margin is min(FOS_buckling/1.5, FOS_yielding/1.5); below 1.0 it is under-strength,
far above 1.0 it is carrying material it does not need.

YOUR SIMULATION BUDGET IS %d STRUCTURAL ANALYSES FOR THIS ENTIRE PROBLEM AND %d REMAIN. Each
candidate you propose costs one of them. Spend them on moves you expect to help; do not explore.

Think in at most three sentences, then emit the %d action lines and nothing after them.
"""

CAND_RE = re.compile(
    r"\b(SIZE_PASS|SCALE|TRIM|ADD_MEMBER|REMOVE_MEMBER|MOVE_JOINT)\s*\(([^)]*)\)")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_candidates(text, nj, ne):
    """Model text -> the same (kind, args) pairs the uniform sampler produces."""
    out = []
    for kind, body in CAND_RE.findall(text or ""):
        nums = NUM_RE.findall(body)
        args = None
        try:
            if kind == "SIZE_PASS":
                if nums:
                    args = {"margin": float(nums[0])}
            elif kind == "SCALE":
                mids = re.search(r"\[([^\]]*)\]", body)
                if mids:
                    ids = [int(x) for x in re.findall(r"\d+", mids.group(1))]
                    rest = NUM_RE.findall(body[mids.end():])
                    factor = float(rest[0]) if rest else None
                elif len(nums) >= 2:
                    ids, factor = [int(float(nums[0]))], float(nums[1])
                else:
                    ids, factor = [], None
                ids = [i for i in ids if 0 <= i < ne]
                if ids and factor is not None:
                    args = {"ids": sorted(set(ids)), "factor": factor}
            elif kind == "TRIM":
                if len(nums) >= 2:
                    args = {"threshold": float(nums[0]), "factor": float(nums[1])}
            elif kind == "ADD_MEMBER":
                if len(nums) >= 2:
                    a, b = int(float(nums[0])), int(float(nums[1]))
                    if 0 <= a < nj and 0 <= b < nj and a != b:
                        args = {"j1": a, "j2": b}
            elif kind == "REMOVE_MEMBER":
                if nums:
                    i = int(float(nums[0]))
                    if 0 <= i < ne and ne > 1:
                        args = {"i": i}
            elif kind == "MOVE_JOINT":
                if len(nums) >= 3:
                    j = int(float(nums[0]))
                    if 0 <= j < nj:
                        args = {"j": j, "x": float(nums[1]), "y": float(nums[2])}
        except Exception:
            args = None
        if args is not None:
            out.append((kind, args))
    return out


def joint_table(truss):
    """Free: every number here was already produced by the analysis that has been paid for."""
    rows = ["", "Joints (x, y) and what each member connects:"]
    for i, j in enumerate(getattr(truss, "joints", []) or []):
        c = list(getattr(j, "coordinates", [0, 0, 0]))
        tag = []
        if getattr(j, "pinned", False):
            tag.append("pinned")
        if getattr(j, "roller", False):
            tag.append("roller")
        ld = getattr(j, "loads", None)
        try:
            if ld is not None and any(abs(float(v)) > 1e-9 for v in ld):
                tag.append("loaded")
        except Exception:
            pass
        rows.append("  J%-3d (%.3f, %.3f)%s" % (i, float(c[0]), float(c[1]),
                                                ("  [" + ",".join(tag) + "]") if tag else ""))
    conn = []
    for i, m in enumerate(getattr(truss, "members", []) or []):
        try:
            conn.append("M%d=J%d-J%d" % (i, m.begin_joint.idx, m.end_joint.idx))
        except Exception:
            pass
    if conn:
        rows.append("  " + "  ".join(conn))
    return "\n".join(rows)


# ---------------------------------------------------------------- the search

def key_of(kind, args):
    return (kind, json.dumps(args, sort_keys=True, default=str))


def budget_search(dom, st, propose, log):
    """Greedy one-step search under a hard evaluation budget. Identical for both arms;
    only `propose` differs. Raises BudgetExhausted when the cap is hit.

    `paid` holds every candidate already evaluated AT THE CURRENT STATE, so neither arm is
    charged twice for the same move -- in particular the mandatory fully-stressed candidate is
    paid for once per state, not once per turn. If nothing beats the incumbent the search stays
    put and proposes again rather than abandoning budget it has not spent.
    """
    paid = set()
    for turn in range(MAX_TURNS):
        if dom.feasible(st):                       # already paid for, cached
            return
        cands = [("SIZE_PASS", {"margin": 1.05})] + list(propose(dom, st, NPROP - 1))
        log["turns"] += 1
        best_v, best = potential(dom, st), None
        for kind, args in cands:
            k = key_of(kind, args)
            if k in paid:
                log["repeat"] += 1
                continue
            paid.add(k)
            nxt = S2.apply_any(dom, st, PARAM, kind, args)
            if nxt is None:
                log["invalid"] += 1
                continue
            log["cands"] += 1
            dom.evaluate(nxt)                      # the one hard analysis this candidate costs
            if dom.feasible(nxt):
                log["picked"][kind] += 1
                return
            v = potential(dom, nxt)
            if v > best_v + 1e-12:
                best_v, best = v, (kind, nxt)
        if best is not None:
            log["picked"][best[0]] += 1
            st = best[1]
            paid = set()                           # new state: every move is worth paying again


def random_propose(rng):
    def f(dom, st, n):
        return S2.propose(dom, st, rng, n)
    return f


def llm_propose(spec, initial, region, token, log):
    system_head = DP.TRUSS_SYSTEM_PROMPT
    k = NPROP - 1
    hist = []

    def f(dom, st, n):
        truss = st["t"]
        state = dom.evaluate(st)                   # cached; this turn's state is already paid for
        obs = (DP.format_eval_result(state)
               + build_table(truss, true_fos(truss))
               + joint_table(truss))
        left = max(0, dom.budget - dom.calls)
        system = system_head + "\n\n" + (TOOLS_DOC % (k, dom.budget, left, k))
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + "\n\nCURRENT STATE:\n" + obs}]}]
        for h in hist[-3:]:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text": "[Simulation Result]\n" + h["o"]}]})
        text = H.call(convo, system, SONNET45, region, token, max_tokens=900) or ""
        log["model_calls"] += 1
        cands = parse_candidates(text, len(getattr(truss, "joints", []) or []),
                                 dom.n_elements(st))[:n]
        log["parsed"] += len(cands)
        hist.append({"t": text.strip()[:900], "o": obs[:1500]})
        return cands

    return f


# ---------------------------------------------------------------- one episode

def seed_of(pid):
    """crc32, not hash(): hash is randomised per process."""
    return zlib.crc32(str(pid).encode()) & 0xffffffff


def episode(spec, arm, budget, region, token):
    pid = spec.get("problem_id", "")
    base = getattr(_tls, "n", 0)
    dom = BudgetedTruss(budget)
    log = {"turns": 0, "cands": 0, "invalid": 0, "repeat": 0, "model_calls": 0,
           "parsed": 0, "picked": Counter()}
    err = None
    try:
        st = dom.load(spec)
        initial = dict(dom.evaluate(st))
        if arm == "fsd":
            rollout(dom, st, PARAM, MAX_TURNS * 2)
        elif arm == "random":
            budget_search(dom, st, random_propose(random.Random(seed_of(pid))), log)
        else:
            budget_search(dom, st, llm_propose(spec, initial, region, token, log), log)
    except BudgetExhausted:
        pass
    except Exception as e:
        err = type(e).__name__ + ": " + str(e)[:200]
    row = {"problem_id": pid, "arm": arm, "budget": budget, "feasible": bool(dom.found),
           "calls": dom.calls, "fea": getattr(_tls, "n", 0) - base,
           "picked": dict(log["picked"])}
    row.update({k: v for k, v in log.items() if k != "picked"})
    if err:
        row["err"] = err
    return row


def work(t):
    spec, arm, budget, region, token = t
    try:
        return episode(spec, arm, budget, region, token)
    except BaseException as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "budget": budget,
                "feasible": False, "calls": -1, "fea": -1,
                "err": type(e).__name__ + ": " + str(e)[:200]}


# ---------------------------------------------------------------- reporting

def report(rows, budgets, arms):
    bad = [r for r in rows if r.get("err")]
    if bad:
        print("errors: %d  (e.g. %s)" % (len(bad), bad[0]["err"]))
    mismatch = [r for r in rows if r.get("calls", 0) >= 0 and r.get("calls") != r.get("fea")]
    print("budget-meter cross-check: %d/%d episodes where dom.calls != wrapped _analyze_truss"
          % (len(mismatch), len(rows)))
    over = [r for r in rows if r.get("calls", 0) > r.get("budget", 0)]
    print("budget overruns: %d\n" % len(over))

    for B in budgets:
        sub = [r for r in rows if r["budget"] == B]
        by = {a: {r["problem_id"]: (1.0 if r["feasible"] else 0.0)
                  for r in sub if r["arm"] == a} for a in arms}
        used = {a: [r["calls"] for r in sub if r["arm"] == a and r["calls"] >= 0] for a in arms}
        pids = (sorted(set.intersection(*[set(by[a]) for a in arms]))
                if all(by[a] for a in arms) else [])
        if not pids:
            print("budget %-3d  no paired problems" % B)
            continue
        print("budget %d analyses/problem   paired on %d problems" % (B, len(pids)))
        for a in arms:
            rate = sum(by[a][p] for p in pids) / len(pids)
            mu = sum(used[a]) / max(len(used[a]), 1)
            mc = [r.get("model_calls", 0) for r in sub if r["arm"] == a]
            pa = [r.get("parsed", 0) for r in sub if r["arm"] == a]
            extra = ""
            if a == "llm" and sum(mc):
                extra = "   model calls/problem %.1f   parsed cands/call %.2f" % (
                    sum(mc) / len(mc), sum(pa) / sum(mc))
            print("   %-7s feasibility %.4f   mean analyses used %5.1f%s" % (a, rate, mu, extra))
        for x, y in (("random", "llm"), ("fsd", "llm"), ("fsd", "random")):
            if x not in arms or y not in arms:
                continue
            u, d, p = sign_test([by[x][q] for q in pids], [by[y][q] for q in pids])
            rx = sum(by[x][q] for q in pids) / len(pids)
            ry = sum(by[y][q] for q in pids) / len(pids)
            print("   %-6s vs %-6s  %+.4f   discordant %d-%d   exact sign p = %.4g"
                  % (y, x, ry - rx, u, d, p))
        pick = Counter()
        for r in sub:
            if r["arm"] == "llm":
                for kk, vv in (r.get("picked") or {}).items():
                    pick[kk] += vv
        if pick:
            print("   llm moves taken: " + "  ".join("%s=%d" % kv for kv in pick.most_common()))
        pick = Counter()
        for r in sub:
            if r["arm"] == "random":
                for kk, vv in (r.get("picked") or {}).items():
                    pick[kk] += vv
        if pick:
            print("   rnd moves taken: " + "  ".join("%s=%d" % kv for kv in pick.most_common()))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/da_lowbudget.jsonl"))
    ap.add_argument("--summarize", action="store_true", help="re-report an existing jsonl")
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    arms = tuple(a.arms.split(","))

    if a.summarize:
        rows = [json.loads(l) for l in open(a.out) if l.strip()]
        report(rows, budgets, arms)
        sys.exit(0)

    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, arm, B, a.region, token) for s in specs for B in budgets for arm in arms]
    print("problems %d-%d | budgets %s | arms %s | pool %d/turn | episodes %d"
          % (a.start, a.start + a.n, budgets, ",".join(arms), NPROP, len(tasks)), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0), flush=True)
    report(rows, budgets, arms)

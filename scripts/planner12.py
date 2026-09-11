"""
Planner v12: the model shapes the proposal distribution; sampling still supplies the diversity.

Three roles tried, three failures. As sole proposer the model loses to uniform random draws
(0.7674 vs 0.8465). Pooled with random at matched count it adds nothing (0.8953 vs 0.9023,
p=0.728). As a pre-filter choosing which 16 of 64 candidates to evaluate it is worse than
choosing 16 at random (0.8000 vs 0.8833). In every case it was being asked to judge individual
candidates, and in every case sampling did the job at least as well.

What sampling is genuinely bad at is finding a specific rare move. Uniform draws spread over
every tool type and every argument, so a structure that needs one particular member between two
particular joints -- one of fifty-odd pairs -- will almost never see it proposed. That is a
weakness of the distribution, not of the evaluation, and it is the one thing a model that knows
what trusses look like could fix without judging anything.

So the model no longer proposes, selects, or evaluates. It names a REGION OF INTEREST -- which
joints and members matter, and which kinds of move are worth trying here -- and random draws are
taken from the distribution it describes. Diversity is still sampling's job; the model only says
where to sample.

  guided     model names joints/members/tool-types, 16 draws biased towards them
  random16   16 uniform draws                                    -- matched budget
  fsd        the sizing rule alone
"""
import os, sys, json, math, random, re, argparse, time
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
from planner01 import fsd_pass, SONNET45
from planner03 import rollout
from planner05 import step_search
from planner09 import random_proposals, joint_xy

MAX_STEPS = 8
KEEP = 16
GUIDE_FRAC = 0.75          # share of draws taken from the model's region; the rest stay uniform

GUIDE_SYSTEM = """A fully-stressed sizing pass is applied automatically every turn: it rescales
every member towards its required factor of safety, growing those below requirement and
shrinking those far above it. It cannot change load paths, cannot spend margin unevenly, and
once it converges while still infeasible -- usually the mass cap binding -- it returns the same
structure forever.

Candidate modifications will be drawn at random and simulated. Random draws spread themselves
over every member, every joint and every kind of move, so they rarely find one specific change.
Your job is to say WHERE to look, not what to do.

Reply with exactly three lines and nothing else:

JOINTS: comma separated joint ids worth moving or connecting, or NONE
MEMBERS: comma separated member ids worth resizing or removing, or NONE
MOVES: any of ADD_MEMBER, REMOVE_MEMBER, MOVE_JOINT, TRIM, SCALE -- the kinds worth trying here

Name a handful, not everything. Say in no other words."""

J_RE = re.compile(r"JOINTS:\s*([^\n]*)", re.I)
M_RE = re.compile(r"MEMBERS:\s*([^\n]*)", re.I)
K_RE = re.compile(r"MOVES:\s*([^\n]*)", re.I)
KINDS = ("ADD_MEMBER", "REMOVE_MEMBER", "MOVE_JOINT", "TRIM", "SCALE")


def parse_guide(text, nj, nm):
    def ids(pat, hi):
        g = pat.search(text or "")
        if not g:
            return []
        return sorted({int(x) for x in re.findall(r"\d+", g.group(1)) if 0 <= int(x) < hi})
    kinds = []
    g = K_RE.search(text or "")
    if g:
        up = g.group(1).upper()
        kinds = [k for k in KINDS if k in up]
    return ids(J_RE, nj), ids(M_RE, nm), (kinds or list(KINDS))


def guided_proposals(truss, rng, n, joints, members, kinds):
    """Draws biased to the named region; falls back to uniform where the guide is empty."""
    nm = len(truss.members)
    try:
        nj = len(truss.joints)
    except Exception:
        nj = 0
    out = []
    for _ in range(n):
        kind = rng.choice(kinds)
        if kind == "ADD_MEMBER" and nj >= 2:
            pool = joints if len(joints) >= 2 else list(range(nj))
            a, b = rng.sample(pool, 2)
            out.append(("ADD_MEMBER", "%d, %d" % (a, b)))
        elif kind == "REMOVE_MEMBER" and nm > 1:
            pool = members or list(range(nm))
            out.append(("REMOVE_MEMBER", "%d" % rng.choice(pool)))
        elif kind == "MOVE_JOINT" and nj >= 1:
            pool = joints or list(range(nj))
            j = rng.choice(pool)
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
            pool = members or list(range(nm))
            k = rng.randint(1, max(1, min(4, len(pool))))
            ids = rng.sample(pool, k) if len(pool) >= k else pool[:1]
            out.append(("SCALE", "[%s], %.3f" % (",".join(str(i) for i in ids),
                                                 rng.uniform(0.75, 1.6))))
    return out


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

    rng = random.Random(abs(hash(pid)) & 0xffffffff)
    nguide = int(KEEP * GUIDE_FRAC)

    for turn in range(max_steps):
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True}
        nxt = fsd_pass(truss, goals)
        if nxt is not truss:
            truss = nxt
            state = _analyze_truss(truss, goals)
        if state.get("is_feasible"):
            return {"problem_id": pid, "arm": arm, "feasible": True}

        remaining = max_steps - turn - 1
        if arm == "random16":
            chosen = random_proposals(truss, rng, KEEP)
        else:
            tbl = build_table(truss, true_fos(truss))
            body = ("A fully-stressed sizing pass has just been applied. Resulting state:\n"
                    + DP.format_eval_result(state) + tbl)
            convo = [{"role": "user", "content": [{"text":
                      DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                      + DP.format_eval_result(initial) + "\n\n" + body}]}]
            text = H.call(convo, GUIDE_SYSTEM, SONNET45, region, token) or ""
            try:
                nj = len(truss.joints)
            except Exception:
                nj = 0
            joints, members, kinds = parse_guide(text, nj, len(truss.members))
            chosen = (guided_proposals(truss, rng, nguide, joints, members, kinds)
                      + random_proposals(truss, rng, KEEP - nguide))

        ok, sel, note = step_search(truss, goals, chosen, remaining, gb, gy)
        if ok:
            return {"problem_id": pid, "arm": arm, "feasible": True}
        if sel is not None:
            truss = sel
            state = _analyze_truss(truss, goals)
    ok, _, _ = rollout(truss, goals, 2)
    return {"problem_id": pid, "arm": arm, "feasible": ok}


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
    ap.add_argument("--arms", default="guided,random16,fsd")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/planner12.jsonl"))
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
    if "guided" in arms and "random16" in arms:
        u, d, p = sign_test([by["random16"][x] for x in pids], [by["guided"][x] for x in pids])
        print("\n  guided vs random16 at matched budget: %+.4f   discordant %d-%d   p = %.3g"
              % (rate["guided"] - rate["random16"], u, d, p))

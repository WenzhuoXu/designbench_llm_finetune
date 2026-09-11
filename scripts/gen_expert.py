#!/usr/bin/env python
"""
TRACK 1 -- the search as expert. Turn the model-free search into an SFT trajectory generator.

The search (da_search2: propose from the generic library plus the domain-registered truss
topology tools, roll each candidate to the horizon, score by the potential, argmax, greedy
pair, do-nothing branch competing) solves the large majority of held-out DesignBench problems.
Nothing in this repo has ever learned from it, because the search speaks a tool language
(SIZE_PASS / SCALE / TRIM / ADD_MEMBER / REMOVE_MEMBER / MOVE_JOINT over the Domain interface)
and the SFT pipeline speaks DesignBench grammar.

This script closes that gap in three steps.

  1. RECORD.  `record_episode` is da_search2.search_episode with one addition: every state the
     search ACCEPTS -- the mandatory opening size pass, the argmax candidate, the second leg of
     a greedy pair, and each size-pass step of the rollout that finally lands feasible -- is
     appended to a trajectory alongside the tool call that produced it. The control flow, the
     rng draw order and the return value are otherwise identical, so the recorder solves
     exactly the problems the search solves. That is not asserted, it is measured: the
     unmodified `S2.search_episode` runs as a control arm in this same script, on the same
     problems, with the same per-problem seed, and the two are compared by a paired exact sign
     test.

  2. TRANSLATE.  Each accepted transition is diffed into DesignBench grammar. Topology moves
     carry their own arguments, so they translate one-to-one. Sizing moves are read off the
     realised radii -- factor_i = r_after_i / r_before_i -- which automatically absorbs the
     CLIP and the per-problem shape bounds that `Domain.set` applies. A pass in which every
     member moves by the same factor becomes one SCALE_MULTI_PARAM; a fully-stressed pass, in
     which every member gets its own factor, becomes a compound of SCALE_PARAM joined by the
     ' ; ' separator this repo already uses for macro moves (api_grammar_2x2.MACRO_SEP).

  3. REPLAY.  The emitted actions are then executed from the untouched initial truss through
     the real DesignBench executor (H.apply_candidate -> normalize_action ->
     execute_grammar_action), and the FEA result recorded in the trajectory is the result of
     THAT replay, not of the search's internal state. So every simulation-result turn in the
     written data is exactly what the written action produces. A trajectory is kept only if
     the replay itself ends feasible, which makes "only keeps trajectories that ended feasible"
     a property of the emitted file rather than a property of the search that generated it.

Output matches DesignBench/data/sft/train.jsonl: system(problem) / system(initial analysis) /
assistant(<think>Modification: ...\nAction: ...</think>) / system(STRUCTURAL ANALYSIS RESULT)
/ ... / assistant(<answer>...</answer>). Observations are
designbench_prompt.format_eval_result + probe15_matrix.build_table(truss, true_fos(truss)).
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time, zlib
from pathlib import Path
from multiprocessing import Pool

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for _p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"),
           str(PROJECT / "design_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from da_domain import get_domain                      # noqa: E402
import da_domtools                                    # noqa: E402,F401  registers truss tools
import da_search2 as S2                               # noqa: E402
from da_search import size_pass, potential            # noqa: E402
import probe15_matrix as PM                           # noqa: E402
import api_grammar_2x2 as H                           # noqa: E402
from llm_finetune.data.processors import designbench_prompt as DP     # noqa: E402
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals  # noqa: E402
from llm_finetune.data.grammar import validate_action                 # noqa: E402

PARAM = "r"
ROLL_MARGIN = 1.05


# ------------------------------------------------------------------ recording search

def rollout_rec(dom, st, param, steps, margin=ROLL_MARGIN):
    """da_search.rollout, additionally returning the size-pass states it walked through."""
    cur, seq = st, []
    for _ in range(max(0, steps)):
        if dom.feasible(cur):
            return True, cur, seq
        nxt = size_pass(dom, cur, param, margin)
        if nxt is cur:
            break
        cur = nxt
        seq.append((cur, ("SIZE_PASS", {"margin": margin})))
    return dom.feasible(cur), cur, seq


def record_episode(dom, st, param, steps, rng, nprop=32):
    """da_search2.search_episode with the accepted moves recorded.

    Returns (solved, trajectory) where trajectory is [(state, (tool_name, args)), ...] in the
    order the search committed to them, starting from the state after the first accepted move.
    """
    traj = []
    for turn in range(steps):
        if dom.feasible(st):
            return 1, traj
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
            traj.append((st, ("SIZE_PASS", {"margin": ROLL_MARGIN})))
        if dom.feasible(st):
            return 1, traj
        remaining = steps - turn - 1
        base_ok, _, base_seq = rollout_rec(dom, st, param, remaining)
        if base_ok:
            traj.extend(base_seq)
            return 1, traj
        best_v, best = potential(dom, st), None
        scored = []
        for kind, args in S2.propose(dom, st, rng, nprop):
            cand = S2.apply_any(dom, st, param, kind, args)
            if cand is None:
                continue
            ok, rolled, seq = rollout_rec(dom, cand, param, remaining)
            if ok:
                traj.append((cand, (kind, args)))
                traj.extend(seq)
                return 1, traj
            v = potential(dom, rolled)
            scored.append((v, cand, (kind, args)))
            if v > best_v + 1e-12:
                best_v, best = v, [(cand, (kind, args))]
        if scored:
            scored.sort(key=lambda x: -x[0])
            anchor, anchor_mv = scored[0][1], scored[0][2]
            for kind, args in S2.propose(dom, anchor, rng, min(8, nprop)):
                cand = S2.apply_any(dom, anchor, param, kind, args)
                if cand is None:
                    continue
                ok, rolled, seq = rollout_rec(dom, cand, param, remaining)
                if ok:
                    traj.append((anchor, anchor_mv))
                    traj.append((cand, (kind, args)))
                    traj.extend(seq)
                    return 1, traj
                v = potential(dom, rolled)
                if v > best_v + 1e-12:
                    best_v, best = v, [(anchor, anchor_mv), (cand, (kind, args))]
        if best is not None:
            traj.extend(best)
            st = best[-1][0]
    ok, _, seq = rollout_rec(dom, st, param, 2)
    if ok:
        traj.extend(seq)
    return (1 if ok else 0), traj


# ------------------------------------------------------------------ tool call -> grammar

def _fmt(x, nd=6):
    s = f"{float(x):.{nd}f}"
    if float(s) == 0.0 and float(x) != 0.0:
        s = f"{float(x):.12f}"
    return s


def _radii(truss):
    out = []
    for m in getattr(truss, "members", []) or []:
        sp = getattr(getattr(m, "shape", None), "_params", None) or {}
        v = sp.get(PARAM)
        out.append(float(v) if isinstance(v, (int, float)) else None)
    return out


def _xy(truss, j):
    c = truss.joints[j].coordinates
    return float(c[0]), float(c[1])


def _margins(truss, gb, gy):
    return [min(a / gb, b / gy) for a, b in PM.true_fos(truss)]


def sizing_parts(t0, t1):
    """Per-member radius factors realised by a sizing move, as SCALE_PARAM parts."""
    r0, r1 = _radii(t0), _radii(t1)
    parts = []
    for i in range(min(len(r0), len(r1))):
        if not r0[i] or not r1[i]:
            continue
        f = r1[i] / r0[i]
        if abs(f - 1.0) > 1e-9:
            parts.append((i, f))
    return parts


def to_grammar(kind, args, t0, t1, gb, gy):
    """(action_string, short_rationale) for one accepted transition t0 -> t1.

    The rationale states only what is visible in the member table the policy is shown --
    which member is short of the 1.5 requirement, which one is carrying spare margin --
    plus the intent of the move. It never claims the move improved the design, because the
    search accepts moves on the value of the ROLLED-OUT state, and several of them (a
    removal, a joint move) look worse for a turn or two before the sizing pass pays them off.
    """
    mg = _margins(t0, gb, gy)
    finite = [(m, i) for i, m in enumerate(mg) if math.isfinite(m)]
    worst_m, worst = min(finite) if finite else (float("nan"), 0)

    if kind == "ADD_MEMBER":
        a, b = int(args["j1"]), int(args["j2"])
        return ("ADD_MEMBER(%d, %d, 6061_T6_Aluminum, Pipe, 0.030000, 0.004000)" % (a, b),
                "Brace the structure with a new member between joints %d and %d, then let the "
                "sizing pass work on the shorter load path (worst member M%d at margin %.2f)"
                % (a, b, worst, worst_m))

    if kind == "REMOVE_MEMBER":
        i = int(args["i"])
        mi = mg[i] if i < len(mg) else float("nan")
        spare = ("margin %.2f, well above the 1.0 required" % mi) if math.isfinite(mi) \
            else "margin effectively unbounded"
        return ("REMOVE_MEMBER(%d)" % i,
                "Take out member %d (%s) and spend its mass on the members that are short "
                "(worst member M%d at margin %.2f)" % (i, spare, worst, worst_m))

    if kind == "MOVE_JOINT":
        j = int(args["j"])
        x0, y0 = _xy(t0, j)
        # da_domtools applies MOVE_JOINT(j, [%.4f, %.4f, 0.0]); the same rounding is
        # reproduced here so the replay lands on exactly the state the search reached.
        x1, y1 = round(float(args["x"]), 4), round(float(args["y"]), 4)
        return ("MOVE_JOINT(%d, [%.4f, %.4f, 0.0], [%.4f, %.4f, 0.0])" % (j, x0, y0, x1, y1),
                "Shift joint %d from (%.3f, %.3f) to (%.3f, %.3f) to change the geometry "
                "feeding the critical members (worst member M%d at margin %.2f)"
                % (j, x0, y0, x1, y1, worst, worst_m))

    parts = sizing_parts(t0, t1)
    if not parts:
        return None, None
    factors = [f for _, f in parts]
    uniform = max(factors) - min(factors) < 1e-9
    ids = [i for i, _ in parts]
    if uniform and len(parts) > 1:
        action = "SCALE_MULTI_PARAM([%s], [radius:%s])" % (
            ", ".join(str(i) for i in ids), _fmt(factors[0]))
    else:
        action = H.MACRO_SEP.join("SCALE_PARAM(%d, radius, %s)" % (i, _fmt(f)) for i, f in parts)

    if kind == "SIZE_PASS":
        up = sum(1 for f in factors if f > 1.0)
        desc = ("Fully-stressed resize: give each member the radius its own FOS asks for -- "
                "%d members up, %d down across %d changed (worst member M%d at margin %.2f)"
                % (up, len(factors) - up, len(parts), worst, worst_m))
    elif kind == "TRIM":
        desc = ("Trim the %d members sitting above margin %.2f: scale their radius by %.3f to "
                "recover mass (worst member M%d at margin %.2f)"
                % (len(parts), float(args.get("threshold", 0.0)), factors[0], worst, worst_m))
    else:
        desc = ("Scale the radius of member%s %s by %.3f (worst member M%d at margin %.2f)"
                % ("" if len(ids) == 1 else "s", ", ".join("M%d" % i for i in ids),
                   factors[0], worst, worst_m))
    return action, desc


# ------------------------------------------------------------------ emission

def obs_text(state, truss):
    return DP.format_eval_result(state) + PM.build_table(truss, PM.true_fos(truss))


def build_record(spec, traj, states, pid, steps, nprop):
    """Replay the translated actions through the real executor and build the JSONL record."""
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    mmax = float(goals.get("maximum_mass", float("inf")))

    messages = [{"role": "system", "content": DP.build_problem_text(spec)},
                {"role": "system", "content": "INITIAL STATE ANALYSIS:\n" + obs_text(state, truss)}]
    actions, kinds = [], []
    cur = truss
    n_compound = 0
    for k, (st_next, (kind, args)) in enumerate(traj):
        t0, t1 = states[k]["t"], st_next["t"]
        action, desc = to_grammar(kind, args, t0, t1, gb, gy)
        if action is None:                       # no-op transition, nothing to teach
            continue
        for part in action.split(H.MACRO_SEP):
            if not validate_action(part).is_valid and not part.startswith("MOVE_JOINT"):
                return None, "invalid_grammar:" + part[:60]
        nxt = H.apply_candidate(cur, action)
        if nxt is None:
            return None, "replay_apply_failed:" + kind
        state = _analyze_truss(nxt, goals)
        messages.append({"role": "assistant",
                         "content": "<think>Modification: %s\nAction: %s</think>" % (desc, action)})
        messages.append({"role": "system",
                         "content": "STRUCTURAL ANALYSIS RESULT:\n" + obs_text(state, nxt)})
        actions.append(action)
        kinds.append(kind)
        n_compound += 1 if H.MACRO_SEP in action else 0
        cur = nxt

    if not actions:
        return None, "no_moves"
    if not state.get("is_feasible"):
        return None, "replay_infeasible"

    messages.append({"role": "assistant", "content":
                     "<answer>Feasible design reached: FOS_buckling %.2f >= %.2f, "
                     "FOS_yielding %.2f >= %.2f, mass %.2f kg <= %.2f kg</answer>"
                     % (float(state.get("fos_buckling", 0.0)), gb,
                        float(state.get("fos_yielding", 0.0)), gy,
                        float(state.get("mass", 0.0)), mmax)})

    # fidelity of the replay against the state the search actually reached
    sm = (traj[-1][0].get("cache") or {}) if traj else {}
    def rel(a, b):
        a, b = float(a or 0.0), float(b or 0.0)
        return abs(a - b) / max(abs(b), 1e-9)
    drift = max(rel(state.get("mass"), sm.get("mass")),
                rel(state.get("fos_buckling"), sm.get("fos_buckling")),
                rel(state.get("fos_yielding"), sm.get("fos_yielding"))) if sm else None

    rec = {
        "problem_id": pid,
        "trace_id": "%s_search_expert" % pid,
        "strategy_type": "SearchExpert",
        "trace_quality": 1.0,
        "messages": messages,
        "source": "gen_expert/da_search2",
        "search": {"steps": steps, "nprop": nprop, "param": PARAM},
        "actions": actions,
        "tools": kinds,
        "n_assistant_turns": len(actions) + 1,
        "n_action_turns": len(actions),
        "n_compound_actions": n_compound,
        "replay_drift": None if drift is None else round(drift, 9),
        "final": {"mass": float(state.get("mass", 0.0)),
                  "fos_buckling": float(state.get("fos_buckling", 0.0)),
                  "fos_yielding": float(state.get("fos_yielding", 0.0)),
                  "maximum_mass": mmax, "is_feasible": True},
    }
    return rec, "ok"


# ------------------------------------------------------------------ workers

def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def _work(t):
    path, arm, steps, nprop = t
    try:
        spec = json.load(open(path))
        pid = spec.get("problem_id") or Path(path).stem
        dom = get_domain("truss")
        st0 = dom.load(spec)
        rng = random.Random(seed_of(path))
        if arm == "control":
            y = S2.search_episode(dom, st0, PARAM, steps, rng, nprop)
            return {"arm": "control", "path": str(path), "problem_id": pid,
                    "solved": int(y), "calls": dom.calls}
        solved, traj = record_episode(dom, st0, PARAM, steps, rng, nprop)
        out = {"arm": "gen", "path": str(path), "problem_id": pid,
               "solved": int(solved), "calls": dom.calls, "n_moves": len(traj),
               "rec": None, "why": "unsolved"}
        if solved and traj:
            states = [st0] + [s for s, _ in traj[:-1]]
            rec, why = build_record(spec, traj, states, pid, steps, nprop)
            out["rec"], out["why"] = rec, why
        elif solved:
            out["why"] = "already_feasible"
        return out
    except Exception as e:
        return {"arm": arm, "path": str(path), "problem_id": Path(path).stem,
                "solved": 0, "calls": 0, "rec": None, "why": "error:%s" % str(e)[:160]}


def sign_test(a, b):
    """Paired exact two-sided sign test. Returns (b_wins, a_wins, p)."""
    up = sum(1 for x, y in zip(a, b) if y > x)
    dn = sum(1 for x, y in zip(a, b) if y < x)
    n = up + dn
    if n == 0:
        return 0, 0, 1.0
    k = min(up, dn)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return up, dn, p


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--steps", type=int, default=12, help="search horizon")
    ap.add_argument("--nprop", type=int, default=32, help="proposal pool per turn")
    ap.add_argument("--procs", type=int, default=44)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--out", default=str(PROJECT / "data/expert/expert_traces.jsonl"))
    a = ap.parse_args()

    files = [str(f) for f in sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))
             ][a.start:a.start + a.n]
    print("gen_expert | problems %d-%d | %d files | horizon %d | pool %d | procs %d"
          % (a.start, a.start + a.n, len(files), a.steps, a.nprop, a.procs), flush=True)

    tasks = [(f, "gen", a.steps, a.nprop) for f in files]
    if not a.no_control:
        tasks += [(f, "control", a.steps, a.nprop) for f in files]

    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(_work, tasks, chunksize=1)
    print("  done in %.0fs" % (time.time() - t0), flush=True)

    gen = {r["path"]: r for r in rows if r["arm"] == "gen"}
    ctl = {r["path"]: r for r in rows if r["arm"] == "control"}

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n_written, turns, assistant_turns, compound, drift = 0, [], 0, 0, 0.0
    tool_hist = {}
    with open(outp, "w") as fh:
        for f in files:
            rec = gen.get(f, {}).get("rec")
            if not rec:
                continue
            fh.write(json.dumps(rec) + "\n")
            n_written += 1
            turns.append(rec["n_action_turns"])
            assistant_turns += rec["n_assistant_turns"]
            compound += rec["n_compound_actions"]
            drift = max(drift, rec.get("replay_drift") or 0.0)
            for k in rec["tools"]:
                tool_hist[k] = tool_hist.get(k, 0) + 1

    solved_gen = sum(gen[f]["solved"] for f in files if f in gen)
    calls = [gen[f]["calls"] for f in files if f in gen]
    print("\n---- solve rate, measured in this run, %d problems ----" % len(files))
    print("  gen_expert (recording search)   %.4f  (%d/%d)"
          % (solved_gen / max(len(files), 1), solved_gen, len(files)))
    if ctl:
        solved_ctl = sum(ctl[f]["solved"] for f in files if f in ctl)
        print("  control    (da_search2, as-is)   %.4f  (%d/%d)"
              % (solved_ctl / max(len(files), 1), solved_ctl, len(files)))
        pids = [f for f in files if f in gen and f in ctl]
        u, d, p = sign_test([ctl[f]["solved"] for f in pids], [gen[f]["solved"] for f in pids])
        print("  gen vs control: discordant %d-%d   exact sign p = %.4g   (paired on %d)"
              % (u, d, p, len(pids)))
    print("  mean analyses per problem       %.1f" % (sum(calls) / max(len(calls), 1)))

    why = {}
    for f in files:
        w = gen.get(f, {}).get("why", "missing")
        why[w] = why.get(w, 0) + 1
    print("\n---- trajectories ----")
    print("  written                 %d" % n_written)
    print("  mean turns/trajectory   %.2f  (min %d, max %d)"
          % (sum(turns) / max(len(turns), 1), min(turns) if turns else 0,
             max(turns) if turns else 0))
    print("  total assistant turns   %d  (%d action turns + %d <answer> turns)"
          % (assistant_turns, assistant_turns - n_written, n_written))
    print("  compound action turns   %d" % compound)
    print("  max replay drift        %.3g" % drift)
    print("  tool mix                %s" % ", ".join("%s=%d" % kv for kv in sorted(tool_hist.items())))
    print("  outcome                 %s" % ", ".join("%s=%d" % kv for kv in sorted(why.items())))
    print("  out                     %s" % outp)

    # ---- verify the file parses as JSONL and one record round-trips ----
    expect = {"problem_id", "trace_id", "strategy_type", "trace_quality", "messages"}
    nlines, ok = 0, True
    first = None
    with open(outp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            nlines += 1
            if first is None:
                first = r
            if not expect.issubset(r):
                ok = False
    print("\n---- verification ----")
    print("  parsed %d/%d lines as JSON, expected-keys present on every line: %s"
          % (nlines, n_written, ok))
    if first is not None:
        roles = [m["role"] for m in first["messages"]]
        print("  first record: %s | %d messages | roles %s...%s"
              % (first["trace_id"], len(roles), roles[:4], roles[-2:]))
        print("  first assistant turn: %s" % first["messages"][2]["content"][:220].replace("\n", " | "))
        print("  final answer turn:    %s" % first["messages"][-1]["content"][:200])


if __name__ == "__main__":
    main()

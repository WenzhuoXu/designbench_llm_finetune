"""
Split the model's action into TARGET and MAGNITUDE, and find out which half is the deficit.

Two measurements now point the same way. The model's in-loop fidelity slope is half the
procedural one (+3.20 against +7.14), so it converts a correct ordering into outcome poorly.
But re-sorting the table by criticality bought nothing (+2.67pp, p=0.454) because the model
already addresses the true worst member 87.5% of the time. It is not failing to find what to
change. The remaining candidate is how much to change it -- which is where the amplitude
measurement already pointed (compression gamma = 0.577: the model moves in the right
direction and too little).

The decomposition, paired on problem and seed:

  free       model emits the action, unaltered                       (baseline)
  sized      model chooses the members; the FACTOR is replaced by the fully-stressed value
             f_i = clip( max( (1.5*margin/fos_b_i)^(1/3), 1.5*margin/fos_y_i ), 0.7, 2.0 )
  targeted   model's factor is kept; the MEMBER is replaced by the true worst

'sized' is deployable -- the fully-stressed factor is computed from the same analysis output
the model already sees, needs no ground truth and no extra evaluations. 'targeted' needs the
true worst member and is a diagnostic ceiling, not a candidate method.

If the deficit is in magnitude, sized >> free and targeted ~ free. If the earlier reading was
wrong and targeting is the problem after all, the reverse.
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
from probe15_matrix import build_table, true_fos, SUP
from probe26_presentation import sign_test

SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
ARMS = tuple((os.environ.get("DEC_ARMS") or "free,sized,targeted").split(","))
NUM = re.compile(r"SCALE_PARAM\(\s*(\d+)\s*,\s*([A-Za-z_]+)\s*,\s*([0-9.]+)\s*\)")


def members_of(c):
    s = set()
    for a_, b_, d_ in SUP.findall(c):
        if a_:
            s.add(int(a_))
        if d_:
            s.add(int(d_))
        for x in (b_ or "").replace(" ", "").split(","):
            if x.isdigit():
                s.add(int(x))
    return sorted(s)


def fsd_factor(fb, fy, gb, gy, margin=1.05):
    """fully-stressed factor for one member, from the FOS values already on the table"""
    try:
        a = (gb * margin / fb) ** (1.0 / 3.0) if fb and math.isfinite(fb) and fb > 0 else 1.0
        b = (gy * margin / fy) if fy and math.isfinite(fy) and fy > 0 else 1.0
        return min(2.0, max(0.7, max(a, b)))
    except Exception:
        return 1.0


def rewrite(c, arm, tf, gb, gy, j_true):
    """Return the candidate with one half of the decision replaced."""
    if arm == "free":
        return c
    if arm == "sized":
        def sub(m):
            i = int(m.group(1))
            if i >= len(tf):
                return m.group(0)
            f = fsd_factor(tf[i][0], tf[i][1], gb, gy)
            return "SCALE_PARAM(%d, %s, %.4f)" % (i, m.group(2), f)
        out = NUM.sub(sub, c)
        return out
    mem = members_of(c)
    if not mem:
        return c
    return NUM.sub(lambda m: "SCALE_PARAM(%d, %s, %s)" % (j_true, m.group(2), m.group(3)), c)


def episode(spec, model, arm, region, token, k, max_steps, seed):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history, nrw, ntot = [], 0, 0
    feasible = False
    for _ in range(max_steps):
        n = len(truss.members)
        tf = true_fos(truss)
        margins = [min(a / gb, b / gy) for a, b in tf]
        j_true = min(range(n), key=lambda j: margins[j])
        tbl = build_table(truss, tf)
        obs = DP.format_eval_result(state) + tbl
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
                  + DP.format_eval_result(initial) + (tbl if not history else "")}]}]
        for h in history:
            convo.append({"role": "assistant", "content": [{"text": h["t"]}]})
            convo.append({"role": "user", "content": [{"text":
                          "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n" + h["o"]}]})
        text = H.call(convo, system, model, region, token)
        cands = H.parse_candidates(text, k, True)
        if not cands:
            history.append({"t": f"No <action> parsed. Emit exactly {k} lines.", "o": obs})
            continue
        use = []
        for c in cands:
            r = rewrite(c, arm, tf, gb, gy, j_true)
            ntot += 1
            if r != c:
                nrw += 1
            use.append(r)
        scored = []
        for c in use:
            nt = H.apply_candidate(truss, c)
            if nt is None:
                continue
            scored.append((nt, _analyze_truss(nt, goals)))
        if not scored:
            history.append({"t": text.strip(), "o": obs})
            continue
        B = float(goals.get("maximum_mass", float("inf")))

        def key(t):
            st = t[1]
            fb = float(st.get("fos_buckling", 0) or 0); fy = float(st.get("fos_yielding", 0) or 0)
            ms = float(st.get("mass", 9e9) or 9e9)
            return min(fb / gb, fy / gy) - (0.5 if ms > B else 0.0)
        truss, state = max(scored, key=key)
        history.append({"t": text.strip(),
                        "o": DP.format_eval_result(state) + build_table(truss, true_fos(truss))})
        if state.get("is_feasible"):
            feasible = True
            break
    return {"problem_id": pid, "arm": arm, "feasible": feasible,
            "rewritten": nrw, "candidates": ntot}


def work(t):
    spec, model, arm, region, token, k, ms, seed = t
    try:
        return episode(spec, model, arm, region, token, k, ms, seed)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "feasible": False,
                "rewritten": 0, "candidates": 0, "err": str(e)[:200]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/decompose.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[a.start:a.start + a.n]]
    tasks = [(s, SONNET45, arm, a.region, token, a.k, a.max_steps, a.seed)
             for s in specs for arm in ARMS]
    print("arms %d | episodes %d" % (len(ARMS), len(tasks)), flush=True)
    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            out.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs\n" % (time.time() - t0))

    by = {arm: {r["problem_id"]: r for r in out if r["arm"] == arm} for arm in ARMS}
    pids = sorted(set.intersection(*[set(by[arm]) for arm in ARMS]))
    print("paired on %d problems\n" % len(pids))
    st = {}
    for arm in ARMS:
        f = [1.0 if by[arm][p]["feasible"] else 0.0 for p in pids]
        rw = sum(by[arm][p]["rewritten"] for p in pids)
        tc = sum(by[arm][p]["candidates"] for p in pids)
        st[arm] = f
        print("  %-10s feasible %.4f    candidates altered %d/%d (%.0f%%)"
              % (arm, sum(f) / len(f), rw, tc, 100.0 * rw / max(tc, 1)))
    print()
    for arm in ("sized", "targeted"):
        u, d, p = sign_test(st["free"], st[arm])
        print("  %-9s vs free: %+.4f   discordant %d-%d   exact sign p = %.3g"
              % (arm, sum(st[arm]) / len(pids) - sum(st["free"]) / len(pids), u, d, p))
    print("\n  'sized' is deployable; 'targeted' is a diagnostic ceiling only.")

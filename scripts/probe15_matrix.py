"""
EXPERIMENT 15 -- the gate matrix.

Three questions, one harness, one information axis.

(a) CORRUPTION-RATE SWEEP. Permute a fraction rho of the table's rows among themselves.
    rho = 0 is the truthful arm, rho = 1 is the full placebo. Turns the DERIVED break-even
    accuracy (~61%, from the 1.53x harm/benefit ratio) into a measured curve.

(b) ADDRESS vs VALUE corruption, AT MATCHED INFORMATION LOSS. The mechanism claim is that
    permutation preserves the extrema the decision rule keys on while destroying where they
    point, whereas value noise degrades the extrema themselves. Both are scored on a common
    axis: the Spearman correlation between the DISPLAYED per-element margin ordering and the
    TRUE one. If, at matched displayed-vs-true rank correlation, permutation costs more than
    value noise, "address" is doing real work. If they lie on one curve, the finding is about
    corruption in general and the framing must change. This is the falsification test.

(c) MODEL FAMILIES. Truthful vs fully permuted on four models spanning capability tiers.
    A property of one model is not a property of language models.

Per turn we log the displayed table's ordering, the true ordering, their Spearman, which
element each candidate targets, and whether that is the displayed-worst row or the true
worst element. Chance is computed per candidate from its own support size.
"""
import os, sys, json, copy, math, random, zlib, argparse, time, re
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

SUP = re.compile(r"SCALE_PARAM\(\s*(\d+)|SCALE_MULTI_PARAM\(\s*\[([0-9,\s]*)\]|MODIFY_PARAM\(\s*(\d+)")


def fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "     n/a"
    if not math.isfinite(v):
        return "     inf"
    return f"{v:8.2f}" if v < 1e5 else f"{v:8.1e}"


def build_table(truss, disp):
    """Exactly H.member_table's format; disp[i] = (fb, fy) shown on row i."""
    rows = ["", "Per-member state (r = outer radius, t = wall thickness, m):",
            "  id    FOS_buckling    FOS_yielding         r         t"]
    for i, m in enumerate(truss.members):
        sp = getattr(getattr(m, "shape", None), "_params", None) or {}
        r_, t_ = sp.get("r"), sp.get("t")
        rows.append(f"  M{i:<4d}{fmt(disp[i][0])}    {fmt(disp[i][1])}  "
                    f"{(f'{float(r_):.5f}' if r_ is not None else '  n/a  ')}  "
                    f"{(f'{float(t_):.5f}' if t_ is not None else '  n/a  ')}")
    rows.append("A member with FOS far above the required 1.5 is carrying more "
                "material than it needs.")
    return "\n".join(rows)


def true_fos(truss):
    out = []
    for m in truss.members:
        def v(a):
            try:
                x = float(getattr(m, a, float("inf")) or float("inf"))
            except Exception:
                x = float("inf")
            return x if x == x else float("inf")
        out.append((v("fos_buckling"), v("fos_yielding")))
    return out


def corrupt(fos, mode, level, rng):
    """mode 'perm': permute a fraction `level` of rows among themselves.
       mode 'noise': multiply each value by exp(N(0, level))."""
    n = len(fos)
    disp = list(fos)
    if mode == "perm" and n > 1 and level > 0:
        k = max(2, int(round(level * n)))
        idx = rng.sample(range(n), min(k, n))
        vals = [disp[i] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            disp[i] = v
    elif mode == "noise" and level > 0:
        disp = [tuple((x * math.exp(rng.gauss(0, level)) if math.isfinite(x) else x) for x in pair)
                for pair in disp]
    return disp


def spearman(a, b):
    n = len(a)
    if n < 3:
        return None
    def rk(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for p, i in enumerate(o):
            r[i] = p
        return r
    ra, rb = rk(a), rk(b)
    m = (n - 1) / 2
    num = sum((ra[i] - m) * (rb[i] - m) for i in range(n))
    den = (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** .5
    return num / den if den else None


def episode(spec, model, mode, level, region, token, k, max_steps, seed):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    pid = spec.get("problem_id", "")
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    rng = random.Random(zlib.crc32((pid + model + mode + str(level)).encode()) ^ seed)
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history, turns = [], []
    feasible = False
    for _ in range(max_steps):
        n = len(truss.members)
        tf = true_fos(truss)
        disp = corrupt(tf, mode, level, rng)
        tmargin = [min(a / gb, b / gy) for a, b in tf]
        dmargin = [min(a / gb, b / gy) for a, b in disp]
        j_true = min(range(n), key=lambda j: tmargin[j])
        row_shown = min(range(n), key=lambda j: dmargin[j])
        rho = spearman([min(x, 50) for x in tmargin], [min(x, 50) for x in dmargin])
        tbl = build_table(truss, disp)
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
        sup = []
        for c in cands:
            s = set()
            for a, b, d in SUP.findall(c):
                if a:
                    s.add(int(a))
                if d:
                    s.add(int(d))
                for x in (b or "").replace(" ", "").split(","):
                    if x.isdigit():
                        s.add(int(x))
            sup.append(sorted(s))
        turns.append({"j_true": j_true, "row_shown": row_shown, "n": n,
                      "rho": None if rho is None else round(rho, 4), "supports": sup})
        scored = []
        for c in cands:
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
        history.append({"t": text.strip(), "o": DP.format_eval_result(state) + build_table(truss, corrupt(true_fos(truss), mode, level, rng))})
        if state.get("is_feasible"):
            feasible = True
            break
    return {"problem_id": pid, "model": model, "mode": mode, "level": level,
            "feasible": feasible, "turns": turns}


def work(t):
    spec, model, mode, level, region, token, k, ms, seed = t
    try:
        return episode(spec, model, mode, level, region, token, k, ms, seed)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "model": model, "mode": mode,
                "level": level, "turns": [], "err": str(e)[:200]}


SONNET45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
FAMILIES = [SONNET45,
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-opus-4-1-20250805-v1:0"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(PROJECT / "results/api_guidance/matrix.jsonl"))
    a = ap.parse_args()
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in
             sorted((DESIGNBENCH / "data" / "problems_hard").glob("*.json"))[:a.n]]

    cells = []
    # (a) corruption-rate sweep, Sonnet 4.5
    for lv in (0.0, 0.25, 0.5, 0.75, 1.0):
        cells.append((SONNET45, "perm", lv))
    # (b) value-corruption control at three noise levels
    for lv in (0.3, 0.8, 2.0):
        cells.append((SONNET45, "noise", lv))
    # (c) families: truthful vs fully permuted
    for m in FAMILIES[1:]:
        cells.append((m, "perm", 0.0))
        cells.append((m, "perm", 1.0))

    tasks = [(s, m, mode, lv, a.region, token, a.k, a.max_steps, a.seed)
             for s in specs for (m, mode, lv) in cells]
    print("cells %d | episodes %d" % (len(cells), len(tasks)), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(work, tasks)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

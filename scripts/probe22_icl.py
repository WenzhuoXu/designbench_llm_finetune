"""
EXPERIMENT 22 -- can the model learn in context from VERIFIED experience?

Precursor to expert iteration. The union over 6 independent attempts reaches 0.859 against
0.520 for a single attempt: a 34-point gap that no single-attempt method reaches and that
expert iteration exists to convert. Before spending GPU on that loop, ask the in-context
version of the same question.

The complication to respect: the model ALREADY sees up to 20 turns of its own
action-and-outcome pairs and stalls after turn 3 (per-step material change flattens to
~1.000). So naive in-context experience does not help. But its own trajectory is a narrow,
correlated sample -- within-turn candidate Jaccard 0.674 -- and in 35% of episodes it never
sees a feasible design at all. It has never been shown a verified SUCCESS it did not itself
generate.

Arms (identical prompts otherwise; only the prepended experience differs):
  A  none              current setup                                  [control]
  B  same-problem      simulator-scored examples from THIS problem,
                       spanning outcomes, incl. a feasible one if reachable
  C  other-problems    verified examples from DIFFERENT problems       [transfer]
  D  successes-only    only feasible examples, from other problems     [the expert-iteration
                                                                        data distribution]

B vs A : can it use verified experience at all?
C vs B : does the experience have to be about this instance?
D vs C : is it the successes that carry it, or the contrast?
D vs A : the in-context ceiling of the expert-iteration data.

If all null, in-context harvesting is closed and expert iteration must go into weights.
"""
import os, sys, json, math, copy, random, zlib, argparse, time, statistics, collections
from math import comb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(P), str(DB), str(P / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.processors import designbench_prompt as DP
import api_grammar_2x2 as H
from probe15_matrix import build_table, true_fos

MACRO_SEP = " ; "


def prm(m):
    return getattr(getattr(m, "shape", None), "_params", None) or {}


def setr(m, v):
    d = getattr(getattr(m, "shape", None), "_params", None)
    if d is not None:
        d["r"] = v


def gen_examples(spec, rng, n_keep=6, pool=24):
    """Procedurally generate candidate moves, score them with the simulator, and keep a
    spread of outcomes. Returns rendered (action -> outcome) strings, and whether any
    reached feasibility."""
    try:
        truss, goals = _load_truss_and_goals(spec)
        st0 = _analyze_truss(truss, goals)
    except Exception:
        return [], False
    sp = ((spec.get("optimization") or {}).get("shape_params") or {})
    rb = sp.get("r") or {}
    lo = float(rb.get("min", 1e-6)); hi = float(rb.get("max", 1e9))
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    base = [prm(m).get("r") for m in truss.members]
    n = len(truss.members)
    if any(b is None or b <= 0 for b in base):
        return [], False

    def fos(m, a):
        try:
            v = float(getattr(m, a, float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    cands = []
    # the closed-form family at several margins, plus random per-member vectors
    for mgn in (1.0, 1.05, 1.15, 1.35):
        vec = {}
        for i, m in enumerate(truss.members):
            fb, fy = fos(m, "fos_buckling"), fos(m, "fos_yielding")
            nb = (gb * mgn / fb) ** (1.0 / 3.0) if math.isfinite(fb) and fb > 0 else 0.7
            ny = (gy * mgn / fy) if math.isfinite(fy) and fy > 0 else 0.7
            vec[i] = min(2.0, max(0.7, max(nb, ny)))
        cands.append(vec)
    for _ in range(pool):
        k = rng.randint(1, min(n, 8))
        S = rng.sample(range(n), k)
        cands.append({i: math.exp(rng.uniform(math.log(0.7), math.log(2.0))) for i in S})

    scored = []
    for vec in cands:
        nt = copy.deepcopy(truss)
        ok = True
        for i, f in vec.items():
            v = base[i] * f
            if v < lo or v > hi:
                ok = False; break
            setr(nt.members[i], v)
        if not ok:
            continue
        try:
            st = _analyze_truss(nt, goals)
        except Exception:
            continue
        act = MACRO_SEP.join("SCALE_PARAM(%d, r, %.3f)" % (i, f) for i, f in sorted(vec.items()))
        scored.append((act, st))
    if not scored:
        return [], False
    feas = [x for x in scored if x[1].get("is_feasible")]
    rest = [x for x in scored if not x[1].get("is_feasible")]
    rest.sort(key=lambda x: -min(float(x[1].get("fos_buckling", 0) or 0) / gb,
                                 float(x[1].get("fos_yielding", 0) or 0) / gy))
    pick = feas[:2] + rest[:max(0, n_keep - min(len(feas), 2))]
    out = []
    for act, st in pick[:n_keep]:
        out.append("  %s\n    -> mass %.2f, FOS_buckling %.3f, FOS_yielding %.3f, feasible: %s"
                   % (act, float(st.get("mass", 0) or 0), float(st.get("fos_buckling", 0) or 0),
                      float(st.get("fos_yielding", 0) or 0),
                      "YES" if st.get("is_feasible") else "no"))
    return out, bool(feas)


def successes_only(spec, rng, n_keep=6):
    ex, _ = gen_examples(spec, rng, n_keep=200, pool=200)
    good = [e for e in ex if e.rstrip().endswith("YES")]
    return good[:n_keep]


def render_block(kind, lines):
    if not lines:
        return ""
    head = {"B": "VERIFIED MOVES ALREADY TRIED ON THIS STRUCTURE, with the simulator's result:",
            "C": "VERIFIED MOVES AND THEIR RESULTS FROM OTHER TRUSS PROBLEMS:",
            "D": "MOVES THAT REACHED A FEASIBLE DESIGN ON OTHER TRUSS PROBLEMS:"}[kind]
    return ("\n\n" + head + "\n" + "\n".join(lines) +
            "\nUse these as evidence about how much a change of a given size actually moves "
            "the factors of safety and the mass.")


def episode(spec, arm, block, model, region, token, k, max_steps):
    truss, goals = _load_truss_and_goals(spec)
    state = _analyze_truss(truss, goals)
    initial = dict(state)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    B = float(goals.get("maximum_mass", float("inf")))
    system = DP.TRUSS_SYSTEM_PROMPT + (
        f"\n\nEach turn, propose {k} DIFFERENT candidate moves rather than one. Think "
        f"briefly, then emit exactly {k} lines of the form <action>...</action>. They "
        f"must be genuinely different candidates -- only one will be executed."
    ) + H.COMPOUND_RULE
    history = []
    feasible = False
    for _ in range(max_steps):
        tbl = build_table(truss, true_fos(truss))
        obs = DP.format_eval_result(state) + tbl
        convo = [{"role": "user", "content": [{"text":
                  DP.build_problem_text(spec) + block + "\n\nINITIAL STATE ANALYSIS:\n"
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
        scored = []
        for c in cands:
            nt = H.apply_candidate(truss, c)
            if nt is None:
                continue
            try:
                st = _analyze_truss(nt, goals)
            except Exception:
                continue
            scored.append((nt, st))
        if not scored:
            history.append({"t": text.strip(), "o": obs})
            continue
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
    return {"problem_id": spec.get("problem_id"), "arm": arm, "feasible": int(feasible)}


def work(t):
    spec, arm, block, model, region, token, k, ms = t
    try:
        return episode(spec, arm, block, model, region, token, k, ms)
    except Exception as e:
        return {"problem_id": spec.get("problem_id"), "arm": arm, "err": str(e)[:150]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--donors", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(P / "results/api_guidance/icl.jsonl"))
    a = ap.parse_args()
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    files = sorted((DB / "data/problems_hard").glob("*.json"))
    specs = [json.load(open(f)) for f in files[:a.n]]
    donors = [json.load(open(f)) for f in files[a.n:a.n + a.donors]]
    print("building verified example pools ...", flush=True)
    t0 = time.time()
    rng = random.Random(3)
    own_ex = {}
    for s in specs:
        ex, _ = gen_examples(s, random.Random(zlib.crc32(s["problem_id"].encode())))
        own_ex[s["problem_id"]] = ex
    donor_mixed, donor_succ = [], []
    for d in donors:
        ex, _ = gen_examples(d, random.Random(zlib.crc32(d["problem_id"].encode())))
        donor_mixed += ex
        donor_succ += successes_only(d, random.Random(zlib.crc32(d["problem_id"].encode())))
        if len(donor_succ) >= 60:
            pass
    print("pools built in %.0fs | own %d | donor mixed %d | donor success %d"
          % (time.time() - t0, len(own_ex), len(donor_mixed), len(donor_succ)), flush=True)

    tasks = []
    for s in specs:
        pid = s["problem_id"]
        r2 = random.Random(zlib.crc32(pid.encode()) ^ 99)
        blocks = {
            "A": "",
            "B": render_block("B", own_ex.get(pid, [])),
            "C": render_block("C", r2.sample(donor_mixed, min(6, len(donor_mixed)))),
            "D": render_block("D", r2.sample(donor_succ, min(6, len(donor_succ))) if donor_succ else []),
        }
        for arm in "ABCD":
            tasks.append((s, arm, blocks[arm], a.model, a.region, tok, a.k, a.max_steps))
    print("episodes %d" % len(tasks), flush=True)
    rows = []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex_, open(a.out, "w") as fh:
        for i, r in enumerate(ex_.map(work, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    by = collections.defaultdict(dict)
    for r in rows:
        if "feasible" in r:
            by[r["arm"]][r["problem_id"]] = r["feasible"]
    keep = sorted(set.intersection(*[set(by[x]) for x in "ABCD"])) if all(by.get(x) for x in "ABCD") else []
    N = len(keep)
    LAB = {"A": "no examples                       [control]",
           "B": "verified examples, THIS problem",
           "C": "verified examples, other problems [transfer]",
           "D": "successes only, other problems    [expert-iteration data]"}
    print("\n=== in-context verified experience, n=%d paired ===" % N)
    for arm in "ABCD":
        if N:
            print("  arm %s  feasible %.3f   %s" % (arm, statistics.mean(by[arm][p] for p in keep), LAB[arm]))

    def mc(x, y):
        hi = sum(1 for p in keep if by[x][p] and not by[y][p])
        lo = sum(1 for p in keep if by[y][p] and not by[x][p])
        d = hi + lo
        pv = 1.0 if d == 0 else min(1.0, 2 * sum(comb(d, i) for i in range(0, min(hi, lo) + 1)) / 2 ** d)
        return hi, lo, 100 * (hi - lo) / max(N, 1), pv
    print()
    for x, y, q in (("B", "A", "can it use verified experience at all?"),
                    ("C", "B", "must the experience be about this instance?"),
                    ("D", "C", "is it the successes that carry it?"),
                    ("D", "A", "in-context ceiling of expert-iteration data"),
                    ("C", "A", "transfer vs nothing")):
        hi, lo, d, pv = mc(x, y)
        print("  %s vs %s : disc %3d-%-3d delta %+5.1fpp  p=%.2e   %s" % (x, y, hi, lo, d, pv, q))

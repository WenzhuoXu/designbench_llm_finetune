#!/usr/bin/env python
"""Adversarial independent check of the gen_expert deliverable."""
import json, sys, math, random, zlib, copy, statistics
from pathlib import Path
from multiprocessing import Pool

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for _p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts"),
           str(PROJECT / "design_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FILE = PROJECT / "data/expert/expert_traces_0_150.jsonl"
PROB = DESIGNBENCH / "data" / "problems_hard"

recs = [json.loads(l) for l in open(FILE) if l.strip()]

print("=== A. file shape ===")
print("records", len(recs))
print("action turns", sum(r["n_action_turns"] for r in recs))
print("assistant turns", sum(r["n_assistant_turns"] for r in recs))
nd = [r for r in recs if r.get("replay_drift") is None]
dr = [r["replay_drift"] for r in recs if r.get("replay_drift") is not None]
print("records with replay_drift None:", len(nd), [r["problem_id"] for r in nd][:10])
print("drift: n=%d max=%.3g mean=%.3g" % (len(dr), max(dr) if dr else 0, statistics.mean(dr) if dr else 0))
pids = [r["problem_id"] for r in recs]
print("unique problem_ids:", len(set(pids)))
allfiles = sorted(PROB.glob("*.json"))
first150 = {f.stem for f in allfiles[:150]}
print("all pids inside sorted[0:150]:", set(pids).issubset(first150))

# ---- B. every part of every compound action must actually execute ----
import api_grammar_2x2 as H
from validation.truss_executor import execute_grammar_action
from llm_finetune.envs.truss_env import _analyze_truss, _load_truss_and_goals
from llm_finetune.data.grammar import validate_action


def part_check(rec):
    spec = json.load(open(PROB / (rec["problem_id"] + ".json")))
    truss, goals = _load_truss_and_goals(spec)
    cur = truss
    nparts = nfail = 0
    exc = 0
    for action in rec["actions"]:
        nt = copy.deepcopy(cur)
        for part in action.split(H.MACRO_SEP):
            part = part.strip()
            if not part:
                continue
            nparts += 1
            try:
                ok = bool(execute_grammar_action(nt, H.normalize_action(nt, part)))
            except Exception:
                ok, exc = False, exc + 1
            if not ok:
                nfail += 1
        cur = nt
    st = _analyze_truss(cur, goals)
    gb = float(goals.get("minimum_fos_buckling", 1.5))
    gy = float(goals.get("minimum_fos_yielding", 1.5))
    mm = float(goals.get("maximum_mass", float("inf")))
    feas = (float(st.get("fos_buckling") or 0) >= gb and
            float(st.get("fos_yielding") or 0) >= gy and
            float(st.get("mass") or 1e9) <= mm)
    return rec["problem_id"], nparts, nfail, exc, bool(feas)


print("\n=== B. per-part execution + independent final feasibility ===")
with Pool(40) as p:
    res = p.map(part_check, recs, chunksize=1)
tp = sum(r[1] for r in res); tf = sum(r[2] for r in res); te = sum(r[3] for r in res)
badfeas = [r[0] for r in res if not r[4]]
print("total parts %d | parts that did NOT execute %d | exceptions %d" % (tp, tf, te))
print("records whose independent replay ends infeasible: %d %s" % (len(badfeas), badfeas[:10]))
for r in res:
    if r[2]:
        print("  partial-apply record", r[0], "failed parts", r[2])

# ---- C. grammar validator on every part ----
print("\n=== C. llm_finetune.data.grammar.validate_action on every emitted part ===")
bad = {}
for r in recs:
    for a in r["actions"]:
        for part in a.split(H.MACRO_SEP):
            v = validate_action(part.strip())
            if not v.is_valid:
                head = part.strip().split("(")[0]
                bad[head] = bad.get(head, 0) + 1
print("invalid parts by head:", bad)

# ---- D. equivalence of recorder and control, on FEA call counts not just solve ----
import da_domtools  # noqa
import da_search2 as S2
from da_domain import get_domain
sys.path.insert(0, str(PROJECT / "scripts"))
import gen_expert as GE


def eq_work(path):
    spec = json.load(open(path))
    d1 = get_domain("truss"); s1 = d1.load(spec)
    r1 = random.Random(zlib.crc32(str(path).encode()) & 0xffffffff)
    y1, traj = GE.record_episode(d1, s1, "r", 12, r1, 32)
    d2 = get_domain("truss"); s2 = d2.load(spec)
    r2 = random.Random(zlib.crc32(str(path).encode()) & 0xffffffff)
    y2 = S2.search_episode(d2, s2, "r", 12, r2, 32)
    return (Path(path).stem, int(y1), int(y2), d1.calls, d2.calls,
            r1.random(), r2.random())


print("\n=== D. recorder vs control: solve AND analysis count AND rng end-state, 60 problems ===")
files = [str(f) for f in allfiles[:60]]
with Pool(40) as p:
    eq = p.map(eq_work, files, chunksize=1)
ds = [e for e in eq if e[1] != e[2]]
dc = [e for e in eq if e[3] != e[4]]
dr2 = [e for e in eq if e[5] != e[6]]
print("solved gen %d / control %d of %d" % (sum(e[1] for e in eq), sum(e[2] for e in eq), len(eq)))
print("problems where solved differs:", len(ds), [e[0] for e in ds][:10])
print("problems where FEA call count differs:", len(dc),
      [(e[0], e[3], e[4]) for e in dc][:10])
print("problems where rng stream end-state differs:", len(dr2), [e[0] for e in dr2][:10])

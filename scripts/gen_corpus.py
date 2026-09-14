"""
The corpus builder: one search pass, four training signals, five domains, balanced on turns.

Everything A1-A5 established, wired together. For each instance the search runs once with the
recorder from harvest.py attached, and what it produces is written out as:

  sft    one conversation per solved trajectory, in the shared serialisation -- user turns are
         the rendered state, assistant turns are tool calls, and the rationale is generated
         from the state itself (which element is worst, which constraint governs, where the
         budget stands) rather than invented
  value  every scored candidate with its rolled-out potential -- 22.3 per SFT turn
  pref   accepted vs each clearly-worse candidate at the same state, near-ties dropped, with
         the tie threshold applied PER DOMAIN because near-tie rates differ twelvefold
         (pipe 4.2%, truss 52.6%)

Balancing is on assistant turns, not trajectories: instances are drawn from each domain until
its turn quota is met, so a domain that solves in one turn contributes as much signal as one
that takes eight. Per the measured yields, that means roughly 1.9 catalogue instances per synth
instance.

Every emitted trajectory is replayed from the untouched initial state through the domain's own
tools before it is kept, and discarded unless the replay itself ends feasible. A trajectory that
cannot be reproduced from its own record is not training data.
"""
import os, sys, json, math, random, zlib, argparse, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter
# Repo root from this file's location; DesignBench is a sibling by default
# (the Bridges-2 layout) and can be pointed elsewhere with DESIGNBENCH_ROOT.
PROJECT = Path(__file__).resolve().parents[1]
DESIGNBENCH = Path(os.environ.get("DESIGNBENCH_ROOT", PROJECT.parent / "DesignBench"))
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_domain import get_domain
from da_synth import SynthDomain
from da_pipe import PipeDomain
from da_catalogue import CatalogueDomain
from da_cases import CasesDomain
import da_domtools, da_meta
import da_serial as S
from da_search import size_pass, potential, rollout
from da_search3 import apply_tool, propose, GENERIC

HORIZON, POOL = 12, 32
TIE_FRAC = 0.05
DOMAINS = ("truss", "synth", "pipe", "catalogue", "cases")


# Truss problems are keyed by file path, and the corpus and every cited baseline were
# generated with DesignBench checked out at SEED_ROOT. Seeding from the raw path gave
# any other checkout a different random stream for the same problem, so the local
# root is rewritten to SEED_ROOT first: the seed depends on the problem alone, and
# every seed at SEED_ROOT is unchanged.
SEED_ROOT = "/ocean/projects/mch250030p/wxu7/DesignBench"


def seed_of(x):
    s, root = str(x), str(DESIGNBENCH)
    i = s.find(root)
    if i >= 0:
        s = s[:i] + SEED_ROOT + s[i + len(root):].replace(os.sep, "/")
    return zlib.crc32(s.encode()) & 0xffffffff


def make(domain, key):
    if domain == "truss":
        d = get_domain("truss")
        return d, d.load(json.load(open(key))), "r"
    if domain == "synth":
        d = SynthDomain(); return d, d.load({"seed": key, "n": 10}), "x"
    if domain == "pipe":
        d = PipeDomain(); return d, d.load({"seed": key, "n": 14}), "d"
    if domain == "catalogue":
        d = CatalogueDomain(); return d, d.load({"seed": key, "n": 10}), "k"
    d = CasesDomain(); return d, d.load({"seed": key, "n": 10, "cases": 4}), "x"


# Two things a signature list alone does not convey, and that an un-finetuned
# model gets wrong every time: the wrapper is <tool>NAME(args)</tool> rather than
# an XML element named after the tool, and element ids are bare integers even
# though the state table labels them E0, E1, ... A smoke run of gpt-4.1-mini
# failed to parse on 12 of 12 calls, every one of them emitting
# "<SCALE ids=[E4,E6] factor=1.3></SCALE>". This teaches the wire format only --
# no hint about which move to choose -- and it is identical for every arm, so
# the comparison stays a comparison of design ability rather than of format luck.
# The worked example is one teacher turn copied verbatim from a training problem
# (problems_gen_c/genc_problem_0566: the first SCALE turn in corpus order that enlarges
# a member set containing its worst member). The previous example was lifted from a
# model's reply on problems_hard/hard_problem_0000 and was close to that evaluation
# problem's answer. This one matches no problems_hard instance's first search move,
# member set, or worst member.
FORMAT_HELP = (
    "Reply with a one-line reason, then exactly one tool call in <tool></tool> tags.\n"
    "Write the call as NAME(arg=value, ...) inside the tags. Element ids are bare\n"
    "integers: the member shown as E13 in the table is id 13.\n\n"
    "Example reply:\n"
    "member E13 is worst at margin 0.388; governed by yielding at 0.388; budget at 0.95 of the limit.\n"
    "<tool>SCALE(factor=1.223, ids=[7, 13, 14])</tool>")


def system_prompt(dom, st):
    return ("You are sizing a design to meet every requirement without exceeding its budget.\n"
            "Each turn you see the current state and choose one tool call.\n\n"
            "Available tools:\n  " + "\n  ".join(S.tool_vocabulary(dom)) + "\n\n"
            + FORMAT_HELP)


def rationale(dom, st):
    """Stated from the state, not invented: worst element, what governs it, budget position."""
    em = dom.element_margins(st) or []
    if not em:
        return "Adjusting the design."
    j = min(range(len(em)), key=lambda i: em[i])
    bits = ["%s E%d is worst at margin %.3f" % (S.noun(dom), j, min(em[j], 50.0))]
    try:
        cols = dom.columns(st)
    except Exception:
        cols = []
    named = [(cname, vals[j]) for cname, vals in cols
             if isinstance(vals, list) and j < len(vals) and cname not in ("length", "flow",
                                                                          "weight", "lever")]
    if named:
        worst_col = min(named, key=lambda kv: kv[1])
        bits.append("governed by %s at %.3f" % (worst_col[0], worst_col[1]))
    try:
        br = dom.budget_ratio(st)
        bits.append("budget at %.2f of the limit" % br)
    except Exception:
        pass
    return "; ".join(bits) + "."


def run_instance(dom, st, param, rng, steps=HORIZON, nprop=POOL):
    """The search, recording the state, every scored candidate, and the accepted move."""
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}
    turns = []
    for turn in range(steps):
        if dom.feasible(st):
            break
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
        if dom.feasible(st):
            break
        remaining = steps - turn - 1
        base_ok, base_state = rollout(dom, st, param, remaining)
        if base_ok:
            st = base_state
            break
        base_v = potential(dom, base_state)
        scored, winner = [], None
        for kind, args in propose(dom, st, rng, nprop, extra, GENERIC):
            cand = apply_tool(dom, st, param, kind, args, extra)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, param, remaining)
            v = 1e9 if ok else potential(dom, rolled)
            scored.append({"head": kind, "args": args, "value": v, "cand": cand})
            if ok:
                winner = rolled
                break
        if not scored:
            break
        best = max(range(len(scored)), key=lambda i: scored[i]["value"])
        turns.append({"obs": S.render_state(dom, st), "why": rationale(dom, st),
                      "base_v": base_v, "accepted": best,
                      "cands": [{"head": c["head"], "args": c["args"], "value": c["value"]}
                                for c in scored]})
        if winner is not None:
            st = winner
            break
        if scored[best]["value"] <= base_v + 1e-12:
            break
        st = scored[best]["cand"]
    return turns, bool(dom.feasible(st))


def replay_ok(domain, key, turns):
    """Reproduce the trajectory from its own record; keep it only if the replay ends feasible."""
    try:
        dom, st, param = make(domain, key)
        extra = dom.tools() or {}
        for t in turns:
            st = size_pass(dom, st, param)
            c = t["cands"][t["accepted"]]
            nxt = apply_tool(dom, st, param, c["head"], c["args"], extra)
            if nxt is None:
                return False
            st = nxt
        ok, _ = rollout(dom, st, param, HORIZON)
        return bool(ok)
    except Exception:
        return False


def to_conversation(domain, key, dom, st0, turns):
    msgs = [{"role": "system", "content": system_prompt(dom, st0)}]
    for t in turns:
        c = t["cands"][t["accepted"]]
        msgs.append({"role": "user", "content": t["obs"]})
        msgs.append({"role": "assistant",
                     "content": t["why"] + "\n" + S.render_tool(c["head"], c["args"])})
    msgs.append({"role": "user", "content": "All requirements are met within budget."})
    msgs.append({"role": "assistant", "content": "<answer>Design is feasible.</answer>"})
    return {"domain": domain, "instance": str(key), "turns": len(turns), "messages": msgs}



# ---- one instance, start to finish -------------------------------------------
# Pure function of (domain, key): everything the corpus needs from one problem.
# Returning plain dicts keeps it picklable so a Pool can fan instances out.
def build_one(job):
    dm, key = job
    try:
        dom, st, param = make(dm, key)
        st0 = dom.clone(st)
        turns, solved = run_instance(dom, st, param,
                                     random.Random(seed_of("%s%s" % (dm, key))))
    except Exception:
        return None
    if not solved or not turns:
        return None
    if not replay_ok(dm, key, turns):
        return {"dropped": True}

    vals_out, prefs_out, heads = [], [], []
    for ti, t in enumerate(turns):
        vals = [c["value"] for c in t["cands"] if c["value"] < 1e8]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        thr = max(TIE_FRAC * spread, 1e-9)
        acc = t["cands"][t["accepted"]]
        bv = acc["value"]
        for i, c in enumerate(t["cands"]):
            heads.append(c["head"])
            # "turn" indexes into this instance's conversation in sft.jsonl. The
            # rendered state used to be copied into all 22 value records and all
            # 14 preference records per turn, which made these files 20x the size
            # of the trajectories they annotate -- 20 GB of the 21 GB that hit the
            # disk quota was that one duplicated string.
            vals_out.append({"domain": dm, "instance": str(key), "turn": ti,
                             "head": c["head"], "args": c["args"], "value": c["value"]})
            if i == t["accepted"] or bv - c["value"] <= thr:
                continue
            prefs_out.append({"domain": dm, "instance": str(key), "turn": ti,
                              "chosen": S.render_tool(acc["head"], acc["args"]),
                              "rejected": S.render_tool(c["head"], c["args"]),
                              "gap": bv - c["value"]})
    return {"dropped": False, "n_turns": len(turns), "heads": heads,
            "conv": to_conversation(dm, key, dom, st0, turns),
            "vals": vals_out, "prefs": prefs_out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-per-domain", type=int, default=100)
    ap.add_argument("--max-instances", type=int, default=400)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--emit", default="sft,value,pref",
                    help="which files to write; SFT alone is enough for the scaling curve")
    ap.add_argument("--out", default=str(PROJECT / "data/corpus"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # Truss trains ONLY on minted problems. problems_hard is the evaluation set --
    # the one carrying the search (0.9721) and FSD (0.4698) reference numbers -- so
    # a single instance of it in training would make every later comparison against
    # those numbers worthless. The gen_b and gen_probe directories are calibration
    # scratch and are excluded for the same reason.
    MINTED = ("problems_gen_c", "problems_gen_d", "problems_gen_e",
              "problems_gen_f", "problems_gen_g")
    files = []
    for name in MINTED:
        files += sorted((DESIGNBENCH / "data" / name).glob("*.json"))
    print("  truss instances available: %d" % len(files), flush=True)

    emit = set(a.emit.split(","))
    devnull = open(os.devnull, "w")
    fh_sft = open(out / "sft.jsonl", "w") if "sft" in emit else devnull
    fh_val = open(out / "value.jsonl", "w") if "value" in emit else devnull
    fh_prf = open(out / "pref.jsonl", "w") if "pref" in emit else devnull
    summary = {}
    t0 = time.time()

    pool = Pool(a.workers) if a.workers > 1 else None
    for dm in a.domains.split(","):
        kept_turns = kept_traj = dropped = n_val = n_prf = 0
        heads = Counter()
        n_inst = min(a.max_instances, len(files)) if dm == "truss" else a.max_instances
        jobs = [(dm, str(files[i]) if dm == "truss" else i) for i in range(n_inst)]
        # imap keeps completion order equal to submission order, so the corpus is
        # byte-identical to the serial build regardless of --workers.
        stream = pool.imap(build_one, jobs, chunksize=4) if pool else map(build_one, jobs)
        for r in stream:
            if kept_turns >= a.turns_per_domain:
                break
            if r is None:
                continue
            if r["dropped"]:
                dropped += 1
                continue
            fh_sft.write(json.dumps(r["conv"]) + "\n")
            kept_traj += 1
            kept_turns += r["n_turns"]
            heads.update(r["heads"])
            for v in r["vals"]:
                fh_val.write(json.dumps(v) + "\n"); n_val += 1
            for pf in r["prefs"]:
                fh_prf.write(json.dumps(pf) + "\n"); n_prf += 1
        summary[dm] = {"traj": kept_traj, "turns": kept_turns, "dropped_replay": dropped,
                       "value": n_val, "pref": n_prf, "heads": heads.most_common(4)}
        print("  %-10s %5d trajectories, %6d turns, %7d value, %7d pref, %d dropped on replay  (%.0fs)"
              % (dm, kept_traj, kept_turns, n_val, n_prf, dropped, time.time() - t0), flush=True)
    if pool:
        pool.terminate(); pool.join()

    for f in (fh_sft, fh_val, fh_prf):
        f.close()
    tt = sum(v["turns"] for v in summary.values())
    print("\n  corpus written to %s in %.0fs" % (out, time.time() - t0))
    print("    trajectories %d | assistant turns %d | value targets %d | preference pairs %d"
          % (sum(v["traj"] for v in summary.values()), tt,
             sum(v["value"] for v in summary.values()),
             sum(v["pref"] for v in summary.values())))
    print("    turn share by domain: %s"
          % ", ".join("%s %.2f" % (k, v["turns"] / max(tt, 1)) for k, v in summary.items()))
    print("    replay failures: %d" % sum(v["dropped_replay"] for v in summary.values()))
    for k, v in summary.items():
        print("    %-10s heads: %s" % (k, v["heads"]))

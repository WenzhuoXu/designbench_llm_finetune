#!/usr/bin/env python
"""Extract every number the progress report plots, from the data itself.

Six blocks, each one figure or equation panel:
  baselines   search vs FSD per domain: rates, discordant counts, p, analysis cost
  trace       one real truss solved by the search, geometry + margins at every turn,
              so the design can be drawn as the object it is rather than described
  phi         the potential trajectory of search vs FSD on that same problem
  actions     head distribution, argument spread and trajectory length per domain
  tiers       the nested training tiers and their domain balance
  cost        per-problem analysis counts, for the cost-vs-feasibility plot
"""
import json
import math
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import da_domtools, da_meta          # noqa: F401
import da_serial as S
from da_search import size_pass, potential, rollout
from da_search3 import apply_tool, propose, GENERIC
from gen_corpus import HORIZON, POOL, make, rationale, seed_of

OUT = PROJECT / "results/report_data.json"
DOMAINS = ("truss", "synth", "pipe", "catalogue", "cases")


# ------------------------------------------------------------------ geometry

def geometry(dom, st):
    """Joints, members and per-member state of a truss, in drawing coordinates."""
    t = st["t"]
    joints = []
    for j in t.joints:
        pos = list(j.coordinates)
        loads = list(getattr(j, "loads", []) or [])
        joints.append({
            "x": float(pos[0]), "y": float(pos[1]),
            "support": ("pinned" if getattr(j, "pinned", False)
                        else "roller" if getattr(j, "roller", False) else ""),
            "fx": float(loads[0]) if len(loads) > 0 else 0.0,
            "fy": float(loads[1]) if len(loads) > 1 else 0.0,
        })
    em = dom.element_margins(st) or []
    members = []
    for i, m in enumerate(t.members):
        ia = int(getattr(m.begin_joint, "idx", 0))
        ib = int(getattr(m.end_joint, "idx", 0))
        prm = getattr(getattr(m, "shape", None), "_params", None) or {}
        members.append({"a": ia, "b": ib,
                        "r": float(prm.get("r", 0.0)),
                        "force": float(getattr(m, "force", 0.0) or 0.0),
                        "margin": float(min(em[i], 50.0)) if i < len(em) else None})
    s = dom.evaluate(st)
    return {"joints": joints, "members": members,
            "mass": float(s.get("mass", 0.0)),
            "budget_ratio": float(dom.budget_ratio(st)),
            "feasible": bool(dom.feasible(st)),
            "phi": float(potential(dom, st)),
            "worst": float(min(em)) if em else None}


def loads_of(spec_path):
    d = json.load(open(spec_path))
    out = []
    for l in d.get("loading", []) or []:
        f = l.get("force", [0, 0, 0])
        out.append({"joint": int(l.get("joint", -1)),
                    "fx": float(f[0]), "fy": float(f[1])})
    return out


# ------------------------------------------------------------------ one trace

def one_trace(key):
    """Run the search on one truss, recording the object at every accepted move."""
    dom, st, param = make("truss", key)
    rng = random.Random(seed_of("truss%s" % key))
    extra = dom.tools() or {}
    steps = []
    steps.append({"label": "initial", "tool": None, "why": "as given",
                  **geometry(dom, st)})

    for turn in range(HORIZON):
        if dom.feasible(st):
            break
        why = rationale(dom, st)
        nxt = size_pass(dom, st, param)
        if nxt is not st:
            st = nxt
            steps.append({"label": "sizing pass", "tool": "SIZE_PASS", "why": why,
                          **geometry(dom, st)})
        if dom.feasible(st):
            break
        remaining = HORIZON - turn - 1
        base_ok, base_state = rollout(dom, st, param, remaining)
        if base_ok:
            st = base_state
            steps.append({"label": "sizing to horizon", "tool": "SIZE_PASS(xN)",
                          "why": "pure sizing reaches feasibility", **geometry(dom, st)})
            break
        base_v = potential(dom, base_state)
        scored, winner = [], None
        for kind, args in propose(dom, st, rng, POOL, extra, GENERIC):
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
        c = scored[best]
        st = winner if winner is not None else c["cand"]
        steps.append({"label": S.render_tool(c["head"], c["args"]),
                      "tool": c["head"], "why": why,
                      "n_candidates": len(scored),
                      "base_v": base_v,
                      "cand_values": sorted(
                          [x["value"] for x in scored if x["value"] < 1e8])[:24],
                      **geometry(dom, st)})
        if winner is not None:
            break
    return {"instance": Path(key).name, "loads": loads_of(key),
            "analyses": int(dom.calls), "steps": steps,
            "solved": bool(dom.feasible(st))}


def fsd_trace(key):
    """The same problem under fully-stressed design alone."""
    dom, st, param = make("truss", key)
    phis = [potential(dom, st)]
    for _ in range(8):
        nxt = size_pass(dom, st, param)
        if nxt is st:
            break
        st = nxt
        phis.append(potential(dom, st))
        if dom.feasible(st):
            break
    return {"phi": phis, "analyses": int(dom.calls), "feasible": bool(dom.feasible(st))}


# ------------------------------------------------------------------ aggregates

def baselines():
    """Read back the per-episode rows the evaluator wrote."""
    rows = defaultdict(list)
    for f in sorted((PROJECT / "results/eval_distilled").glob("rows_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("arm") in ("fsd", "native", "search"):
                rows[(r["domain"], r["arm"], str(f))].append(r)
    # keep the largest run per (domain, arm)
    best = {}
    for (dm, arm, f), rs in rows.items():
        if len(rs) >= len(best.get((dm, arm), [])):
            best[(dm, arm)] = rs

    def pair(ta, tb, keys):
        w = sum(1 for k in keys if ta[k]["feasible"] and not tb[k]["feasible"])
        l = sum(1 for k in keys if tb[k]["feasible"] and not ta[k]["feasible"])
        m = w + l
        p = 1.0 if m == 0 else min(1.0, 2.0 * sum(
            math.comb(m, i) for i in range(min(w, l) + 1)) / (2.0 ** m))
        return w, l, len(keys) - m, p

    out = {}
    for dm in DOMAINS:
        a = best.get((dm, "search"))
        b = best.get((dm, "fsd"))
        if not a or not b:
            continue
        ta = {x["instance"]: x for x in a}
        tb = {x["instance"]: x for x in b}
        keys = sorted(set(ta) & set(tb))
        w = sum(1 for k in keys if ta[k]["feasible"] and not tb[k]["feasible"])
        l = sum(1 for k in keys if tb[k]["feasible"] and not ta[k]["feasible"])
        m = w + l
        p = 1.0 if m == 0 else min(1.0, 2.0 * sum(
            math.comb(m, i) for i in range(min(w, l) + 1)) / (2.0 ** m))
        out[dm] = {
            "n": len(keys),
            "search_rate": sum(ta[k]["feasible"] for k in keys) / len(keys),
            "fsd_rate": sum(tb[k]["feasible"] for k in keys) / len(keys),
            "wins": w, "losses": l, "ties": len(keys) - m, "p": p,
            "search_analyses": sorted(ta[k].get("analyses", 0) for k in keys),
            "fsd_analyses": sorted(tb[k].get("analyses", 0) for k in keys),
        }
        # The domain's own heuristic, where one was measured. On catalogue and
        # cases the generic sizing arm is not a fair control, so the honest
        # comparison is against this instead.
        c = best.get((dm, "native"))
        if c:
            tc = {x["instance"]: x for x in c}
            ks = sorted(set(ta) & set(tc))
            if ks:
                w, l, tie, pv = pair(ta, tc, ks)
                out[dm]["native_rate"] = sum(tc[k]["feasible"] for k in ks) / len(ks)
                out[dm]["native_analyses"] = sorted(tc[k].get("analyses", 0) for k in ks)
                out[dm]["native_w"] = w
                out[dm]["native_l"] = l
                out[dm]["native_p"] = pv
    return out


def actions(corpus):
    out = {}
    for f in sorted(Path(corpus).rglob("sft.jsonl")):
        dm = f.parent.name
        heads = Counter()
        turns_hist = Counter()
        args = defaultdict(list)
        n = 0
        for line in f.open():
            r = json.loads(line)
            turns_hist[r["turns"]] += 1
            for msg in r["messages"]:
                if msg["role"] != "assistant":
                    continue
                parsed = S.parse_tools(msg["content"])
                if not parsed:
                    continue
                head, a = parsed[0]
                heads[head] += 1
                n += 1
                for k, v in (a or {}).items():
                    if isinstance(v, (int, float)) and len(args[(head, k)]) < 60000:
                        args[(head, k)].append(round(float(v), 3))
        tot = sum(heads.values()) or 1
        ent = -sum((v / tot) * math.log(v / tot) for v in heads.values() if v)
        out[dm] = {
            "heads": heads.most_common(),
            "entropy": ent / math.log(len(heads)) if len(heads) > 1 else 0.0,
            "n_calls": n,
            "turns_hist": sorted(turns_hist.items())[:10],
            "args": {"%s.%s" % k: hist(v) for k, v in args.items() if len(v) > 200},
        }
    return out


def hist(vals, bins=24):
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {"lo": lo, "hi": hi, "counts": [len(vals)]}
    c = [0] * bins
    for v in vals:
        c[min(bins - 1, int((v - lo) / (hi - lo) * bins))] += 1
    return {"lo": lo, "hi": hi, "counts": c}


def tiers():
    out = {}
    d = PROJECT / "data/tiers"
    for f in sorted(d.glob("*.jsonl")):
        per = Counter()
        traj = 0
        for line in f.open():
            r = json.loads(line)
            per[r["domain"]] += r["turns"]
            traj += 1
        out[f.stem] = {"turns": sum(per.values()), "traj": traj, "by_domain": dict(per)}
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = {}
    if which in ("all", "base"):
        data["baselines"] = baselines()
        print("baselines: %d domains" % len(data["baselines"]), flush=True)
    if which in ("all", "trace"):
        keys = [str(p) for p in sorted((DESIGNBENCH / "data/problems_hard").glob("*.json"))]
        # A problem the search solves with real topology work, not one sizing pass.
        for k in keys[:40]:
            t = one_trace(k)
            if t["solved"] and 5 <= len(t["steps"]) <= 9:
                data["trace"] = t
                data["fsd_trace"] = fsd_trace(k)
                break
        else:
            data["trace"] = one_trace(keys[0])
            data["fsd_trace"] = fsd_trace(keys[0])
        print("trace: %s, %d steps, %d analyses"
              % (data["trace"]["instance"], len(data["trace"]["steps"]),
                 data["trace"]["analyses"]), flush=True)
    if which in ("all", "actions"):
        data["actions"] = actions(PROJECT / "data/corpus_v2")
        print("actions: %d domains" % len(data["actions"]), flush=True)
    if which in ("all", "tiers"):
        data["tiers"] = tiers()
        print("tiers: %s" % list(data["tiers"]), flush=True)

    old = json.load(open(OUT)) if OUT.exists() and which != "all" else {}
    old.update(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(old, open(OUT, "w"))
    print("wrote %s (%.1f KB)" % (OUT, OUT.stat().st_size / 1024))

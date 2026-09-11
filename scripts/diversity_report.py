#!/usr/bin/env python
"""How diverse is the training material, really?

Two questions, measured separately because they fail separately:

PROBLEM VARIATION. The truss generator's template is narrow by construction --
two-row Warren, a few bay counts, planar, point loads on the top row. Minting
50,000 of them is volume; whether it is also variety is an empirical question.
This measures the discrete structure (how many distinct shapes), the continuous
spread (span, height, loads, budget), and -- the one that matters -- how many
distinct design SITUATIONS survive once near-identical problems are collapsed.
A problem is only new if it poses a different sizing decision, so fingerprints
are taken at several roundings and the count is reported at each: if the unique
count collapses as the rounding coarsens, the extra problems were re-sampling.

ACTION DISTRIBUTION. A corpus where the teacher emits one tool 90% of the time
teaches one tool. Per domain this reports the head distribution and its
normalised entropy, the argument spread within each head, how often consecutive
turns repeat the same head, and how many distinct states the model is ever asked
to act on. Entropy near 1 means the heads are near-uniform; that is necessary,
not sufficient, so the argument histograms are printed too -- SIZE_PASS called
400,000 times at the same margin is one action wearing a distribution's clothes.
"""
import argparse
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ------------------------------------------------------------------ problem side

def features(path):
    """Everything about one problem that could make it a different design task."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    top = d.get("topology", {})
    joints = top.get("joints", [])
    members = top.get("members", [])
    if not joints or not members:
        return None

    xs = [j["position"][0] for j in joints]
    ys = [j["position"][1] for j in joints]
    zs = [j["position"][2] if len(j["position"]) > 2 else 0.0 for j in joints]
    span = max(xs) - min(xs)
    height = max(ys) - min(ys)
    depth = max(zs) - min(zs)
    # Bays = distinct x stations on the bottom chord, minus one.
    bottom = sorted({round(j["position"][0], 6) for j in joints
                     if abs(j["position"][1] - min(ys)) < 1e-9})
    bays = max(len(bottom) - 1, 0)

    loads = d.get("loading", []) or []
    mags = [math.sqrt(sum(c * c for c in l.get("force", [0, 0, 0]))) for l in loads]
    loaded = sorted(l.get("joint", -1) for l in loads)

    goals = d.get("goals", {}) or {}
    meta = d.get("_metadata", {}) or {}
    opt = meta.get("optimal_mass") or 0.0
    budget = goals.get("maximum_mass") or 0.0

    supports = Counter(j.get("support") or "free" for j in joints)
    radii = [m.get("shape", {}).get("r") for m in members
             if isinstance(m.get("shape"), dict)]
    radii = [r for r in radii if isinstance(r, (int, float))]

    return {
        "n_joints": len(joints), "n_members": len(members), "bays": bays,
        "span": span, "height": height, "depth": depth,
        "aspect": (height / span) if span else 0.0,
        "n_loads": len(loads),
        "load_total": sum(mags), "load_max": max(mags) if mags else 0.0,
        "loaded": tuple(loaded),
        "budget": budget, "optimal": opt,
        "slack": (budget / opt) if opt else 0.0,
        "degradation_steps": meta.get("degradation_steps", -1),
        "stratum": meta.get("stratum", "?"),
        "method": meta.get("generation_method", "?"),
        "supports": tuple(sorted(supports.items())),
        "n_distinct_radii": len(set(round(r, 9) for r in radii)),
    }


def fingerprint(f, places):
    """Identity of a problem at a given resolution. Two problems with the same
    fingerprint pose the same sizing decision up to that precision."""
    def q(x):
        return round(x, places)
    return (f["n_joints"], f["n_members"], f["bays"], f["n_loads"], f["loaded"],
            f["supports"], q(f["span"]), q(f["height"]), q(f["aspect"]),
            q(f["load_total"] / 1000.0), q(f["load_max"] / 1000.0), q(f["slack"]))


def describe(name, vals, unit=""):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
    print("    %-18s min %9.3f  p10 %9.3f  med %9.3f  p90 %9.3f  max %9.3f   mean %9.3f  sd %8.3f  cv %.3f%s"
          % (name, vals[0], vals[int(n * 0.10)], vals[n // 2], vals[min(n - 1, int(n * 0.90))],
             vals[-1], mean, sd, (sd / mean) if mean else 0.0, unit))


def counter_line(name, c, top=8):
    tot = sum(c.values()) or 1
    items = c.most_common(top)
    body = ", ".join("%s %.1f%%" % (k, 100.0 * v / tot) for k, v in items)
    extra = "" if len(c) <= top else "  (+%d more)" % (len(c) - top)
    print("    %-18s %d distinct | %s%s" % (name, len(c), body, extra))


def entropy(c):
    tot = sum(c.values())
    if tot <= 0 or len(c) <= 1:
        return 0.0
    h = -sum((v / tot) * math.log(v / tot) for v in c.values() if v)
    return h / math.log(len(c))


def problem_section(dirs, workers, sample_nn):
    paths = []
    for d in dirs:
        paths += [str(p) for p in sorted(Path(d).glob("*.json"))]
    print("\n" + "=" * 100)
    print("PROBLEM VARIATION  --  %d truss problems from %s"
          % (len(paths), ", ".join(Path(d).name for d in dirs)))
    print("=" * 100)
    if not paths:
        return
    with Pool(workers) as pool:
        feats = [f for f in pool.map(features, paths, chunksize=64) if f]
    print("  parsed %d of %d" % (len(feats), len(paths)))

    print("\n  discrete structure")
    for key in ("n_members", "n_joints", "bays", "n_loads", "stratum", "method",
                "degradation_steps", "n_distinct_radii"):
        counter_line(key, Counter(f[key] for f in feats))
    counter_line("loaded joints", Counter(f["loaded"] for f in feats))
    counter_line("supports", Counter(f["supports"] for f in feats))

    print("\n  continuous spread  (cv = sd/mean; near 0 means the field is effectively constant)")
    for key, unit in (("span", " m"), ("height", " m"), ("aspect", ""), ("depth", " m"),
                      ("load_total", " N"), ("load_max", " N"),
                      ("budget", " kg"), ("optimal", " kg"), ("slack", " x")):
        describe(key, [f[key] for f in feats], unit)

    print("\n  how many distinct design situations?  (unique fingerprints by rounding)")
    n = len(feats)
    for places, label in ((6, "exact"), (3, "1e-3"), (2, "1e-2"), (1, "1e-1"), (0, "integer")):
        u = len({fingerprint(f, places) for f in feats})
        print("    rounded to %-8s %7d unique of %d  (%.2f%%)" % (label, u, n, 100.0 * u / n))

    # Nearest-neighbour spacing in a normalised continuous space. If the minted
    # problems are dense re-samples of one template, typical spacing goes to zero.
    keys = ("span", "height", "aspect", "load_total", "load_max", "slack")
    lo = {k: min(f[k] for f in feats) for k in keys}
    hi = {k: max(f[k] for f in feats) for k in keys}
    def vec(f):
        return [((f[k] - lo[k]) / (hi[k] - lo[k])) if hi[k] > lo[k] else 0.0 for k in keys]
    step = max(1, len(feats) // sample_nn)
    sample = [vec(f) for f in feats[::step]][:sample_nn]
    dists = []
    for i, a in enumerate(sample):
        best = float("inf")
        for j, b in enumerate(sample):
            if i == j:
                continue
            s = 0.0
            for x, y in zip(a, b):
                s += (x - y) * (x - y)
                if s >= best:
                    break
            if s < best:
                best = s
        dists.append(math.sqrt(best))
    dists.sort()
    m = len(dists)
    print("\n  nearest-neighbour distance in normalised %d-d parameter space (n=%d sample)"
          % (len(keys), m))
    print("    min %.4f  p10 %.4f  median %.4f  p90 %.4f  max %.4f"
          % (dists[0], dists[int(m * 0.1)], dists[m // 2], dists[min(m - 1, int(m * 0.9))],
             dists[-1]))
    print("    (0 would mean exact duplicates; 1.0 is the diameter of the unit cube's edge)")


# ------------------------------------------------------------------ action side

def action_section(corpus):
    import da_domtools  # noqa: F401
    import da_meta      # noqa: F401
    import da_serial as S

    print("\n" + "=" * 100)
    print("ACTION DISTRIBUTION  --  %s" % corpus)
    print("=" * 100)
    files = sorted(Path(corpus).rglob("sft.jsonl"))
    if not files:
        print("  no sft.jsonl under %s" % corpus)
        return

    for f in files:
        dom_name = f.parent.name
        heads = Counter()
        args_by_head = defaultdict(Counter)
        idset_sizes = Counter()
        repeats = same = 0
        turns_hist = Counter()
        states = set()
        n_turns = n_traj = 0
        unparsed = 0

        for line in f.open():
            r = json.loads(line)
            n_traj += 1
            turns_hist[r["turns"]] += 1
            prev = None
            for m in r["messages"]:
                if m["role"] == "user":
                    states.add(zlib.crc32(m["content"].encode()))
                if m["role"] != "assistant":
                    continue
                parsed = S.parse_tools(m["content"])
                if not parsed:
                    unparsed += 1
                    continue
                head, args = parsed[0]
                n_turns += 1
                heads[head] += 1
                for k, v in (args or {}).items():
                    if isinstance(v, (int, float)):
                        args_by_head[head][(k, round(float(v), 2))] += 1
                    elif isinstance(v, (list, tuple)):
                        idset_sizes[len(v)] += 1
                if prev is not None:
                    repeats += 1
                    same += int(prev == head)
                prev = head

        print("\n  %s -- %d trajectories, %d tool calls, %d distinct states seen"
              % (dom_name, n_traj, n_turns, len(states)))
        counter_line("head", heads, top=10)
        print("    %-18s %.3f  (1.0 = uniform over %d heads)"
              % ("head entropy", entropy(heads), max(len(heads), 1)))
        print("    %-18s %.3f  (fraction of consecutive turns using the SAME head)"
              % ("repeat rate", (same / repeats) if repeats else 0.0))
        print("    %-18s %.3f  (distinct states per tool call)"
              % ("state reuse", (len(states) / n_turns) if n_turns else 0.0))
        counter_line("turns/trajectory", turns_hist, top=8)
        if idset_sizes:
            counter_line("id-set size", idset_sizes, top=6)
        if unparsed:
            print("    %-18s %d assistant turns held no parseable tool call" % ("UNPARSED", unparsed))
        for head in [h for h, _ in heads.most_common(6)]:
            c = args_by_head[head]
            if not c:
                continue
            tot = sum(c.values())
            byk = defaultdict(Counter)
            for (k, v), n in c.items():
                byk[k][v] += n
            for k, vc in sorted(byk.items()):
                vals = sorted(vc)
                print("      %-14s %-10s %3d distinct values, range %.2f..%.2f, "
                      "top %s"
                      % (head, k, len(vc), vals[0], vals[-1],
                         ", ".join("%.2f(%.0f%%)" % (v, 100.0 * n / tot)
                                   for v, n in vc.most_common(3))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-dirs", default=",".join(
        str(DESIGNBENCH / "data" / d) for d in
        ("problems_hard", "problems_gen_c", "problems_gen_d", "problems_gen_e",
         "problems_gen_f", "problems_gen_g")))
    ap.add_argument("--corpus", default=str(PROJECT / "data/corpus_v2"))
    ap.add_argument("--workers", type=int, default=46)
    ap.add_argument("--sample-nn", type=int, default=3000)
    ap.add_argument("--skip-problems", action="store_true")
    ap.add_argument("--skip-actions", action="store_true")
    a = ap.parse_args()

    if not a.skip_problems:
        dirs = [d for d in a.problem_dirs.split(",") if Path(d).is_dir()]
        problem_section(dirs, a.workers, a.sample_nn)
    if not a.skip_actions:
        action_section(a.corpus)

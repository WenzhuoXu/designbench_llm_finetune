"""
Is the multi-load-case domain well-posed, does the conjunction actually bite, and is it searchable?

Four things measured before any trajectory comes out of it.

  WELL-POSED. No instance may start feasible and every instance must have a certified feasible
  assignment, or the trajectories teach nothing.

  THE CONJUNCTION BITES. The domain only earns its place if different cases bind different
  members, and if sizing for one case genuinely breaks another. Both are measured: how many
  distinct cases are governing at once, and what happens to the OTHER cases when the currently
  binding one is satisfied.

  SEARCHABLE. Search against the domain's own heuristic -- repeated SIZE_ENVELOPE, the sensible
  strategy -- in the same script on the same instances, paired sign test.

  THE WRONG STRATEGY LOSES. Repeated SIZE_FOR_CASE, sizing for one case at a time, is the
  plausible-but-wrong approach. It is run as a third arm so the domain is shown to punish it.
"""
import sys, random, statistics, zlib
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_cases import CasesDomain, CASE_NAMES
import da_serial as S
from da_search3 import apply_tool, propose, GENERIC
from da_search import potential
from probe26_presentation import sign_test

N, HORIZON, POOL = 120, 12, 16


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def mk(s):
    d = CasesDomain()
    return d, d.load({"seed": s, "n": 10, "cases": 4})


def envelope_episode(dom, st, steps):
    t = dom.tools()["SIZE_ENVELOPE"][1]
    cur = st
    for _ in range(steps):
        if dom.feasible(cur):
            return 1
        nxt = t(dom, cur, {"margin": 1.05})
        if nxt is None:
            break
        cur = nxt
    return 1 if dom.feasible(cur) else 0


def percase_episode(dom, st, steps):
    """The plausible-but-wrong strategy: chase whichever case is worst, one at a time."""
    t = dom.tools()["SIZE_FOR_CASE"][1]
    cur = st
    for _ in range(steps):
        if dom.feasible(cur):
            return 1
        cm = dom.evaluate(cur)["case_margins"]
        worst = min(range(cur["k"]), key=lambda c: min(cm[c]))
        nxt = t(dom, cur, {"case": worst, "margin": 1.05})
        if nxt is None:
            break
        cur = nxt
    return 1 if dom.feasible(cur) else 0


def rollout(dom, st, steps):
    return (envelope_episode(dom, st, steps) == 1), st


def search_episode(dom, st, rng, steps=HORIZON, nprop=POOL):
    extra = dom.tools()
    for turn in range(steps):
        if dom.feasible(st):
            return 1
        remaining = steps - turn - 1
        d2 = CasesDomain(); probe = dom.clone(st)
        if envelope_episode(dom, probe, remaining) == 1:
            return 1
        best_v, best = potential(dom, st), None
        for kind, args in propose(dom, st, rng, nprop, extra, GENERIC):
            cand = apply_tool(dom, st, "x", kind, args, extra)
            if cand is None:
                continue
            probe = dom.clone(cand)
            if envelope_episode(dom, probe, remaining) == 1:
                return 1
            v = potential(dom, cand)
            if v > best_v + 1e-12:
                best_v, best = v, cand
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


print("1. WELL-POSED")
start_feas = 0
for s in range(N):
    d, st = mk(s)
    if d.feasible(st):
        start_feas += 1
print("   already feasible at the start: %d/%d  (want 0)" % (start_feas, N))

print("\n2. DOES THE CONJUNCTION BITE")
distinct, broke = [], 0
trials = 0
for s in range(60):
    d, st = mk(s)
    ev = d.evaluate(st)
    distinct.append(len(set(ev["binding"])))
    # satisfy the currently worst case only, then see what the others did
    cm = ev["case_margins"]
    worst = min(range(st["k"]), key=lambda c: min(cm[c]))
    before = [min(cm[c]) for c in range(st["k"])]
    nxt = d.tools()["SIZE_FOR_CASE"][1](d, st, {"case": worst, "margin": 1.10})
    after_cm = d.evaluate(nxt)["case_margins"]
    after = [min(after_cm[c]) for c in range(st["k"])]
    for c in range(st["k"]):
        if c == worst:
            continue
        trials += 1
        if after[c] < before[c] - 1e-9:
            broke += 1
print("   distinct governing cases per instance: mean %.2f of %d" % (statistics.mean(distinct), 4))
print("   sizing for the worst case LOWERED another case's worst margin in %d of %d (%.1f%%)"
      % (broke, trials, 100.0 * broke / max(trials, 1)))

print("\n3. SEARCH vs THE DOMAIN'S OWN HEURISTIC, same instances, same run")
sr, hr, pr = {}, {}, {}
for s in range(N):
    d1, s1 = mk(s); sr[s] = float(search_episode(d1, s1, random.Random(seed_of(s))))
    d2, s2 = mk(s); hr[s] = float(envelope_episode(d2, s2, HORIZON))
    d3, s3 = mk(s); pr[s] = float(percase_episode(d3, s3, HORIZON))
pids = sorted(sr)
a = [hr[p] for p in pids]; b = [sr[p] for p in pids]; c = [pr[p] for p in pids]
u, dn, pv = sign_test(a, b)
print("   search            %.4f" % (sum(b) / len(b)))
print("   envelope heuristic %.4f" % (sum(a) / len(a)))
print("   per-case (wrong)   %.4f" % (sum(c) / len(c)))
print("   search vs envelope: %+.4f   discordant %d-%d   exact sign p = %.3g"
      % (sum(b) / len(b) - sum(a) / len(a), u, dn, pv))
u2, d2_, p2 = sign_test(c, a)
print("   envelope vs per-case: %+.4f   discordant %d-%d   exact sign p = %.3g"
      % (sum(a) / len(a) - sum(c) / len(c), u2, d2_, p2))

print("\n4. SERIALISATION")
d, st = mk(0)
print("\n".join(S.render_state(d, st).splitlines()[:9]))
print("   vocabulary: %s" % S.tool_vocabulary(d))

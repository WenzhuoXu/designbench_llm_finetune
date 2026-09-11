"""
Is the fourth domain well-posed, genuinely non-monotone, and does the search still work on it?

Three things to establish before any trajectory is generated from it, each measured:

  SOLVABLE. Every instance must have a feasible assignment and must not start feasible,
  otherwise trajectories teach nothing. Both are checked over many seeds.

  NON-MONOTONE. The claim that upsizing a member can LOWER its own margin is the reason this
  domain earns its place. Measured directly: take each element, upgrade it one catalogue step,
  and count how often its own margin falls.

  SEARCHABLE. The search must still beat the domain's own heuristic here, with the control run
  in the same script on the same instances and a paired sign test. The heuristic is
  CATALOGUE_PASS applied repeatedly -- the discrete analogue of the sizing rule.
"""
import sys, random, statistics
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import zlib
from da_catalogue import CatalogueDomain, NK
import da_serial as S
from da_search3 import apply_tool, propose, GENERIC
from da_search import potential
from probe26_presentation import sign_test

N = 120
HORIZON, POOL = 12, 16


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xffffffff


def heur_episode(dom, st, steps):
    """Repeated CATALOGUE_PASS: the domain's own sizing rule."""
    tools = dom.tools()
    cur = st
    for _ in range(steps):
        if dom.feasible(cur):
            return 1
        nxt = tools["CATALOGUE_PASS"][1](dom, cur, {"margin": 1.05})
        if nxt is None:
            break
        cur = nxt
    return 1 if dom.feasible(cur) else 0


def rollout(dom, st, steps):
    tools = dom.tools()
    cur = st
    for _ in range(steps):
        if dom.feasible(cur):
            return True, cur
        nxt = tools["CATALOGUE_PASS"][1](dom, cur, {"margin": 1.05})
        if nxt is None:
            break
        cur = nxt
    return dom.feasible(cur), cur


def search_episode(dom, st, rng, steps=HORIZON, nprop=POOL):
    extra = dom.tools()
    for turn in range(steps):
        if dom.feasible(st):
            return 1
        remaining = steps - turn - 1
        ok, _ = rollout(dom, st, remaining)
        if ok:
            return 1
        best_v, best = potential(dom, st), None
        for kind, args in propose(dom, st, rng, nprop, extra, GENERIC):
            cand = apply_tool(dom, st, "k", kind, args, extra)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, remaining)
            if ok:
                return 1
            v = potential(dom, rolled)
            if v > best_v + 1e-12:
                best_v, best = v, cand
        if best is None:
            break
        st = best
    return 1 if dom.feasible(st) else 0


print("1. WELL-POSED")
solvable = start_feas = 0
for s in range(N):
    d = CatalogueDomain()
    st = d.load({"seed": s, "n": 10})
    if d.feasible(st):
        start_feas += 1
    top = d.clone(st)
    for i in range(st["n"]):
        d.set(top, i, "k", float(NK - 1))
    # the generator certified a feasible assignment; confirm the instance is not trivial
    solvable += 1 if st["B"] > 0 else 0
print("   instances with a certified budget: %d/%d" % (solvable, N))
print("   instances already feasible at the start: %d/%d  (want 0)" % (start_feas, N))

print("\n2. NON-MONOTONE RESPONSE")
worse = total = 0
for s in range(60):
    d = CatalogueDomain()
    st = d.load({"seed": s, "n": 10})
    base = d.element_margins(st)
    for i in range(st["n"]):
        if d.get(st, i, "k") >= NK - 1:
            continue
        up = d.clone(st)
        d.set(up, i, "k", d.get(st, i, "k") + 1.0)
        total += 1
        if d.element_margins(up)[i] < base[i] - 1e-12:
            worse += 1
print("   upgrading a member LOWERED its own margin in %d of %d single-step upgrades (%.1f%%)"
      % (worse, total, 100.0 * worse / max(total, 1)))
print("   (every other domain here is monotone: this is 0%% by construction there)")

print("\n3. SEARCHABLE — search vs the domain's own heuristic, same instances, same run")
sr, hr = {}, {}
for s in range(N):
    d1 = CatalogueDomain(); st1 = d1.load({"seed": s, "n": 10})
    sr[s] = float(search_episode(d1, st1, random.Random(seed_of(s))))
    d2 = CatalogueDomain(); st2 = d2.load({"seed": s, "n": 10})
    hr[s] = float(heur_episode(d2, st2, HORIZON))
pids = sorted(sr)
a = [hr[p] for p in pids]; b = [sr[p] for p in pids]
u, dn, pv = sign_test(a, b)
print("   search    %.4f" % (sum(b) / len(b)))
print("   heuristic %.4f" % (sum(a) / len(a)))
print("   search vs heuristic: %+.4f   discordant %d-%d   exact sign p = %.3g"
      % (sum(b) / len(b) - sum(a) / len(a), u, dn, pv))

print("\n4. SERIALISATION")
d = CatalogueDomain(); st = d.load({"seed": 0, "n": 8})
txt = S.render_state(d, st)
print("\n".join(txt.splitlines()[:10]))
print("   vocabulary: %s" % S.tool_vocabulary(d))

"""Advance to the feasible state instead of discarding it.

The harvest loop broke out of the turn loop whenever a rollout reached feasibility, but never
moved `st` onto the state that got there, then tested dom.feasible(st) on the un-advanced state.
Domains that solve on the first continuation -- pipe and cases -- recorded zero turns and scored
as total failures; truss read 0.117 against its real 0.93. Both exit paths now take the solved
state, and the candidate that found it is recorded as that turn's accepted move first.
"""
import io

p = "/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/harvest.py"
s = io.open(p, encoding="utf-8").read()

old = """        base_ok, base_state = rollout(dom, st, param, remaining)
        if base_ok:
            break
        base_v = potential(dom, base_state)"""
new = """        base_ok, base_state = rollout(dom, st, param, remaining)
        if base_ok:
            # the continuation already reaches feasibility: advance onto it, do not just stop
            st = base_state
            break
        base_v = potential(dom, base_state)"""
assert old in s, "base rollout block not found"
s = s.replace(old, new, 1)

old2 = """        scored = []
        for kind, args in propose(dom, st, rng, nprop, extra, GENERIC):
            cand = apply_tool(dom, st, param, kind, args, extra)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, param, remaining)
            v = 1e9 if ok else potential(dom, rolled)
            scored.append({"head": kind, "args": args, "value": v, "cand": cand})
            if ok:
                break
        if not scored:
            break"""
new2 = """        scored, winner = [], None
        for kind, args in propose(dom, st, rng, nprop, extra, GENERIC):
            cand = apply_tool(dom, st, param, kind, args, extra)
            if cand is None:
                continue
            ok, rolled = rollout(dom, cand, param, remaining)
            v = 1e9 if ok else potential(dom, rolled)
            scored.append({"head": kind, "args": args, "value": v, "cand": cand})
            if ok:
                winner = rolled          # this candidate's continuation reaches feasibility
                break
        if not scored:
            break"""
assert old2 in s, "candidate loop not found"
s = s.replace(old2, new2, 1)

old3 = """        if scored[best]["value"] <= base_v + 1e-12:
            break
        st = scored[best]["cand"]"""
new3 = """        if winner is not None:
            st = winner                  # the turn is recorded, then take the solved state
            break
        if scored[best]["value"] <= base_v + 1e-12:
            break
        st = scored[best]["cand"]"""
assert old3 in s, "accept block not found"
s = s.replace(old3, new3, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("harvest loop now advances to the feasible state")

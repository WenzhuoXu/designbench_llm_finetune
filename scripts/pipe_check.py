"""Is the pipe generator producing solvable instances at all? Check before spending more compute."""
import sys
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "design_agent")):
    if p not in sys.path:
        sys.path.insert(0, p)
from da_pipe import PipeDomain

ok_uniform = 0
ok_start = 0
N = 40
print("  seed   ref-uniform feasible?   budget ratio at ref   min margin at ref   start feasible?")
for s in range(N):
    dom = PipeDomain()
    st = dom.load({"seed": s, "n": 14})
    start_feas = dom.feasible(st)
    # recover the reference scale: current d is ref*0.78
    ref = st["d"][0] / 0.78
    st2 = dom.clone(st)
    for i in range(st2["n"]):
        dom.set(st2, i, "d", ref)
    feas = dom.feasible(st2)
    br = dom.budget_ratio(st2)
    mm = min(dom.element_margins(st2))
    ok_uniform += 1 if feas else 0
    ok_start += 1 if start_feas else 0
    if s < 8:
        print("  %4d   %-20s %8.3f              %8.3f            %s"
              % (s, str(feas), br, mm, start_feas))
print("\n  uniform reference feasible on %d/%d instances" % (ok_uniform, N))
print("  starting profile already feasible on %d/%d" % (ok_start, N))

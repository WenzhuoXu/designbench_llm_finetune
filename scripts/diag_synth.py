"""
Why does synth yield 19 turns in gen_corpus but 125 in harvest, on the same search?

harvest measured synth at solve 0.783 and 125 turns over 60 instances -- the highest-yielding
domain. gen_corpus kept 17 trajectories totalling 19 turns from up to 400 instances, with zero
replay failures reported. Three things could produce that and they look identical from outside,
because the caller wraps run_instance in a bare `except Exception: continue`:

  an exception inside run_instance      -> instance silently skipped
  solved but with no turns recorded     -> rejected by `not turns`
  not solved                            -> rejected by `not solved`

This runs gen_corpus's own run_instance on synth with the exception surfaced and each outcome
counted separately, so the cause is identified rather than guessed at.
"""
import sys, random, zlib, traceback
from collections import Counter
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import gen_corpus as G

N = 60
outcomes = Counter()
turn_total = 0
first_trace = None

for idx in range(N):
    key = idx
    try:
        dom, st, param = G.make("synth", key)
        turns, solved = G.run_instance(dom, st, param,
                                       random.Random(G.seed_of("synth%s" % key)))
    except Exception:
        outcomes["exception"] += 1
        if first_trace is None:
            first_trace = traceback.format_exc()
        continue
    turn_total += len(turns)
    if not solved:
        outcomes["not_solved"] += 1
    elif not turns:
        outcomes["solved_but_zero_turns"] += 1
    else:
        outcomes["kept"] += 1
        if not G.replay_ok("synth", key, turns):
            outcomes["replay_failed"] += 1

print("synth, %d instances through gen_corpus.run_instance\n" % N)
for k in ("kept", "solved_but_zero_turns", "not_solved", "exception", "replay_failed"):
    print("  %-22s %d" % (k, outcomes[k]))
print("  total turns recorded   %d" % turn_total)
print("  harvest measured       125 turns over 60 instances, solve 0.783")
if first_trace:
    print("\nfirst exception:\n%s" % first_trace)

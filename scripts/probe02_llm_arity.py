"""What arity does the policy actually play, and where does it sit against the measured
reachability optimum (k=3-6)? Read from the traced corpus; no simulator needed."""
import json, re, glob, collections, statistics, sys
from math import comb

paths = sorted(glob.glob("/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/traces_sonnet*.jsonl"))
if not paths:
    paths = sorted(glob.glob("/ocean/projects/mch250030p/wxu7/llm_finetune/analysis/remote/results/api_guidance/traces_sonnet*.jsonl"))
if not paths:
    import subprocess
    out = subprocess.run(["bash","-lc","find /ocean/projects/mch250030p/wxu7 -name 'traces_sonnet*.jsonl' 2>/dev/null | head -4"],
                         capture_output=True, text=True).stdout.split()
    paths = out
print("trace files:", paths, file=sys.stderr)

ID = re.compile(r"SCALE_PARAM\(\s*(\d+)|SCALE_MULTI_PARAM\(\s*\[([0-9,\s]*)\]|MODIFY_PARAM\(\s*(\d+)|ADD_MEMBER|REMOVE_MEMBER|MOVE_JOINT\(\s*(\d+)")

def support(action):
    s = set()
    for m in re.finditer(r"SCALE_PARAM\(\s*(\d+)", action): s.add(int(m.group(1)))
    for m in re.finditer(r"MODIFY_PARAM\(\s*(\d+)", action): s.add(int(m.group(1)))
    for m in re.finditer(r"MOVE_JOINT\(\s*(\d+)", action): s.add(('j', int(m.group(1))))
    for m in re.finditer(r"SCALE_MULTI_PARAM\(\s*\[([0-9,\s]*)\]", action):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit(): s.add(int(tok))
    for m in re.finditer(r"REMOVE_MEMBER\(\s*(\d+)", action): s.add(int(m.group(1)))
    return s

ar = []
ar_feas = []
ar_infeas = []
turns = 0
cands = 0
by_turn_max = []
for p in paths:
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        cl = r.get("candidates") or []
        if not cl: continue
        turns += 1
        mx = 0
        for c in cl:
            a = c.get("action") or ""
            k = len(support(a))
            if k == 0: continue
            cands += 1
            ar.append(k); mx = max(mx, k)
            (ar_feas if c.get("feasible") else ar_infeas).append(k)
        by_turn_max.append(mx)

print("turns %d, candidates with a parsed support %d" % (turns, cands))
c = collections.Counter(ar)
tot = len(ar)
print()
print("=== ARITY THE POLICY ACTUALLY PLAYS (per candidate) ===")
for k in sorted(c)[:14]:
    print("   |S|=%-3d %5d  (%.4f)" % (k, c[k], c[k]/tot))
big = sum(v for k,v in c.items() if k > 14)
if big: print("   |S|>14 %5d  (%.4f)" % (big, big/tot))
print("median |S| = %d ; mean %.2f" % (statistics.median(ar), sum(ar)/tot))
print("P(|S| = 1)      = %.4f" % (c.get(1,0)/tot))
print("P(|S| <= 2)     = %.4f" % ((c.get(1,0)+c.get(2,0))/tot))
print("P(3 <= |S| <= 6) = %.4f   <- the measured reachability optimum band" % (sum(c.get(k,0) for k in (3,4,5,6))/tot))
print("P(|S| >= 7)     = %.4f" % (sum(v for k,v in c.items() if k>=7)/tot))
print()
print("=== ARITY OF FEASIBLE vs INFEASIBLE CANDIDATES ===")
if ar_feas:
    print("feasible   n=%-5d median |S| = %d  mean %.2f  P(|S|<=2)=%.3f  P(3<=|S|<=6)=%.3f"
          % (len(ar_feas), statistics.median(ar_feas), sum(ar_feas)/len(ar_feas),
             sum(1 for k in ar_feas if k<=2)/len(ar_feas),
             sum(1 for k in ar_feas if 3<=k<=6)/len(ar_feas)))
print("infeasible n=%-5d median |S| = %d  mean %.2f  P(|S|<=2)=%.3f  P(3<=|S|<=6)=%.3f"
      % (len(ar_infeas), statistics.median(ar_infeas), sum(ar_infeas)/len(ar_infeas),
         sum(1 for k in ar_infeas if k<=2)/len(ar_infeas),
         sum(1 for k in ar_infeas if 3<=k<=6)/len(ar_infeas)))
print()
print("=== PER-TURN MAX ARITY (did the policy ever offer a wide action this turn?) ===")
cm = collections.Counter(by_turn_max)
print("median per-turn max |S| = %d" % statistics.median(by_turn_max))
print("turns whose WIDEST candidate had |S|<=2 : %d/%d = %.4f"
      % (sum(v for k,v in cm.items() if k<=2), len(by_turn_max),
         sum(v for k,v in cm.items() if k<=2)/len(by_turn_max)))
print("turns with at least one candidate in 3..6: %d/%d = %.4f"
      % (sum(1 for x in by_turn_max if 3<=x<=6), len(by_turn_max),
         sum(1 for x in by_turn_max if 3<=x<=6)/len(by_turn_max)))

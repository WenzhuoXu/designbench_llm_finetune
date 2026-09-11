"""
EXPERIMENT 11 -- does the policy's emitted factor vector rank-correlate with the margins
it was shown?

probe10: ordinal ranking + a swept amplitude recovers 18.0 of the macro's 26.0pp (69%);
cardinal precision adds only 8.0pp; and exact factors at PERMUTED addresses are worth
+2.5pp, p=0.125 -- nothing. So the binding requirement is that the factor assigned to an
element be monotone in that element's own margin.

The policy reads a per-element table and identifies the blamed element 92.7% of the time.
But identifying the WORST element is not the same as ordering ALL of them. This measures
the thing arm F assumes: Spearman rank correlation between the per-element FOS values the
policy QUOTES BACK in its own reasoning and the factors it then EMITS for those elements.

Perfect ordinal play = rho of -1 (worst margin gets the largest factor).
Trace-only. No simulator.
"""
import json, re, glob, statistics, collections
from pathlib import Path
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")

# member FOS quoted in the prose: "M3: 0.82", "member 3 ... FOS 0.82", "M3 (0.82)"
QUOTE = [
    re.compile(r"\bM(?:ember)?\s*#?(\d{1,2})\b[^0-9\n]{0,40}?([0-9]+\.[0-9]+)"),
    re.compile(r"\bM(\d{1,2})\s*[:=(]\s*([0-9]+\.[0-9]+)"),
]
PAIR = re.compile(r"SCALE_PARAM\(\s*(\d+)\s*,\s*\w+\s*,\s*([0-9]*\.?[0-9]+)\s*\)")
MULTI = re.compile(r"SCALE_MULTI_PARAM\(\s*\[([0-9,\s]*)\]\s*,\s*\[([^\]]*)\]")
KV = re.compile(r"(\w+)\s*:\s*([0-9]*\.?[0-9]+)")


def emitted(action):
    d = collections.defaultdict(list)
    for m in PAIR.finditer(action):
        d[int(m.group(1))].append(float(m.group(2)))
    for m in MULTI.finditer(action):
        ids = [int(x) for x in m.group(1).replace(" ", "").split(",") if x.isdigit()]
        fs = [float(v) for _, v in KV.findall(m.group(2))]
        for i in ids:
            for f in fs:
                d[i].append(f)
    return {i: statistics.median(v) for i, v in d.items()}


def quoted(raw):
    """member -> the smallest plausible FOS quoted for it (margins are what matter)."""
    d = {}
    for rx in QUOTE:
        for m in rx.finditer(raw):
            try:
                i = int(m.group(1)); v = float(m.group(2))
            except (TypeError, ValueError):
                continue
            if 0 <= i <= 40 and 0.01 <= v <= 60.0:
                d[i] = min(d.get(i, 9e9), v)
    return d


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def rank(a):
        order = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


rhos, rhos_feas, rhos_infeas = [], [], []
n_turn, n_cand, n_used = 0, 0, 0
for p in glob.glob(str(P / "results/api_guidance/traces_sonnet*.jsonl")):
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q = quoted(r.get("raw") or "")
        n_turn += 1
        if len(q) < 3:
            continue
        for c in (r.get("candidates") or []):
            n_cand += 1
            e = emitted(c.get("action") or "")
            common = sorted(set(e) & set(q))
            if len(common) < 3:
                continue
            rho = spearman([q[i] for i in common], [e[i] for i in common])
            if rho is None:
                continue
            n_used += 1
            rhos.append(rho)
            (rhos_feas if c.get("feasible") else rhos_infeas).append(rho)

rhos.sort()
qf = lambda a, f: a[int(f * (len(a) - 1))]
print("turns %d | candidates %d | scorable (>=3 shared members) %d" % (n_turn, n_cand, n_used))
print()
print("=== SPEARMAN rho( quoted per-element FOS , emitted factor ) ===")
print("  perfect ordinal play = -1.00   (worst margin gets the largest factor)")
print("  no relationship      =  0.00")
print()
print("  mean   %+.4f" % (sum(rhos) / len(rhos)))
print("  median %+.4f" % qf(rhos, .5))
print("  p10 %+.3f | p25 %+.3f | p75 %+.3f | p90 %+.3f" % (qf(rhos, .1), qf(rhos, .25), qf(rhos, .75), qf(rhos, .9)))
for t in (-0.8, -0.5, -0.2, 0.0, 0.2):
    print("  P(rho <= %+.1f) = %.4f" % (t, sum(1 for x in rhos if x <= t) / len(rhos)))
print()
if rhos_feas:
    print("  feasible candidates   n=%-5d mean rho %+.4f  median %+.4f"
          % (len(rhos_feas), sum(rhos_feas) / len(rhos_feas), sorted(rhos_feas)[len(rhos_feas) // 2]))
print("  infeasible candidates n=%-5d mean rho %+.4f  median %+.4f"
      % (len(rhos_infeas), sum(rhos_infeas) / len(rhos_infeas), sorted(rhos_infeas)[len(rhos_infeas) // 2]))
print()
print("  fraction with the WRONG SIGN (rho > 0, bigger factor on the healthier element): %.4f"
      % (sum(1 for x in rhos if x > 0) / len(rhos)))

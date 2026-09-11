"""Clustered, paired analysis of the address test. Candidates nest in turns nest in
episodes; the two arms share the same 60 problems, so the contrast is paired by problem."""
import json, statistics
from math import comb, sqrt, erf
from pathlib import Path
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune/results/api_guidance/address_test.jsonl")
rows = [json.loads(l) for l in open(P) if l.strip()]

def ep_rates(r):
    ht = hr = tot = ch = 0
    for t in r["turns"]:
        for s in t["supports"]:
            if not s:
                continue
            tot += 1
            ht += int(t["j_true"] in s)
            hr += int(t["row_shown"] in s)
            ch += min(1.0, len(s) / max(t["n"], 1))
    if tot == 0:
        return None
    return {"true": ht / tot, "row": hr / tot, "chance": ch / tot, "n": tot}

by = {}
for r in rows:
    e = ep_rates(r)
    if e:
        by.setdefault(r["problem_id"], {})[r["obs"]] = e
keep = [p for p, d in by.items() if len(d) == 2]
N = len(keep)
print("problems with both arms: %d" % N)

def norm_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

def paired(vals_a, vals_b, la, lb):
    d = [a - b for a, b in zip(vals_a, vals_b)]
    m = sum(d) / len(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / sqrt(len(d)) if sd else 0.0
    t = m / se if se else float("inf")
    hi = sum(1 for x in d if x > 0); lo = sum(1 for x in d if x < 0)
    dd = hi + lo
    sp = 1.0 if dd == 0 else min(1.0, 2 * sum(comb(dd, i) for i in range(0, min(hi, lo) + 1)) / 2 ** dd)
    print("  %-38s mean %.4f | %-38s mean %.4f" % (la, sum(vals_a)/len(vals_a), lb, sum(vals_b)/len(vals_b)))
    print("      paired diff %+.4f (se %.4f, t=%.2f, p=%.2e) ; sign %d-%d p=%.2e"
          % (m, se, t, norm_p(t) if se else 0.0, hi, lo, sp))

pm_true = [by[p]["per_member"]["true"] for p in keep]
pm_row = [by[p]["per_member"]["row"] for p in keep]
sh_true = [by[p]["shuffled"]["true"] for p in keep]
sh_row = [by[p]["shuffled"]["row"] for p in keep]
sh_ch = [by[p]["shuffled"]["chance"] for p in keep]
pm_ch = [by[p]["per_member"]["chance"] for p in keep]

print("\n=== episode-level rates, paired by problem (n=%d) ===" % N)
print("\n1) Does a FALSE table destroy targeting of the TRUE worst member?")
paired(pm_true, sh_true, "truthful: P(acts on true worst)", "shuffled: P(acts on true worst)")
print("\n2) Under a FALSE table, is targeting of the true worst at CHANCE?")
paired(sh_true, sh_ch, "shuffled: P(acts on true worst)", "shuffled: chance given support size")
print("\n3) Does the model follow the DISPLAYED address instead?")
paired(sh_row, sh_ch, "shuffled: P(acts on displayed-worst row)", "shuffled: chance")
print("\n4) Does it follow a FALSE address more reliably than a TRUE one?")
paired(sh_row, pm_row, "shuffled: P(acts on displayed row)", "truthful: P(acts on displayed row)")
print("\n5) Sanity: truthful arm above chance?")
paired(pm_true, pm_ch, "truthful: P(acts on true worst)", "truthful: chance")

tot_c = sum(by[p]["shuffled"]["n"] for p in keep) + sum(by[p]["per_member"]["n"] for p in keep)
print("\ntotal candidates scored: %d" % tot_c)

"""
EXPERIMENT 24 -- the cross-vendor IN-LOOP arm, which is the open gate item.

probe18 measured, on four Claude models with the permutation fixed per episode, that three of
four follow a FALSE address as reliably as a true one under a pre-registered TOST margin of
0.10, and that feasibility collapses to 0.00-0.03 in all four. probe20 then took the
DETECTION question cross-vendor and found the sharper inversion claim does not generalise.

The in-loop question has never gone cross-vendor, and single-vendor evidence is precisely
what failed the novelty gate. Six non-Anthropic models across five vendors, same design,
same fixed-per-episode permutation, same equivalence margin.

Also reports per-model parse rate, because Palmyra returned almost nothing usable in the
forced-choice run and a model that cannot emit the action format tells us nothing about
address-following.
"""
import os, sys, json, time, statistics, collections
from math import comb, sqrt, erf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(P), str(P / "scripts"), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
import probe18_fixedperm as B

DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
FAM = [
    ("Meta",     "us.meta.llama3-3-70b-instruct-v1:0"),
    ("Amazon",   "us.amazon.nova-pro-v1:0"),
    ("Mistral",  "mistral.mistral-large-2407-v1:0"),
    ("Mistral",  "us.mistral.pixtral-large-2502-v1:0"),
    ("Writer",   "us.writer.palmyra-x5-v1:0"),
    ("Alibaba",  "qwen.qwen3-235b-a22b-2507-v1:0"),
]


def norm_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def paired(A, Bd, ks):
    d = [A[k] - Bd[k] for k in ks]
    m = sum(d) / len(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / sqrt(len(d)) if sd else 0.0
    hi = sum(1 for x in d if x > 0); lo = sum(1 for x in d if x < 0); dd = hi + lo
    sp = 1.0 if dd == 0 else min(1.0, 2 * sum(comb(dd, i) for i in range(0, min(hi, lo) + 1)) / 2 ** dd)
    return m, se, hi, lo, sp


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    region = os.environ.get("AWS_REGION", "us-west-2")
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:n]]
    tasks = [(s, mid, pp, region, tok, 8, 6)
             for s in specs for (_, mid) in FAM for pp in (False, True)]
    print("episodes %d over %d models" % (len(tasks), len(FAM)), flush=True)
    rows = []
    t0 = time.time()
    out = P / "results/api_guidance/xvendor_inloop.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out, "w") as fh:
        for i, r in enumerate(ex.map(B.work, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 100 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    def ep(r):
        ht = hr = tot = 0; ch = 0.0
        for t in r.get("turns", []):
            for s in t["supports"]:
                if not s:
                    continue
                tot += 1
                ht += int(t["j_true"] in s); hr += int(t["row_shown"] in s)
                ch += min(1.0, len(s) / max(t["n"], 1))
        return None if not tot else {"true": ht / tot, "row": hr / tot,
                                     "chance": ch / tot, "feas": int(r["feasible"])}

    cell = collections.defaultdict(dict); attempted = collections.Counter(); scored = collections.Counter()
    for r in rows:
        attempted[(r["model"], r["permute"])] += 1
        e = ep(r)
        if e:
            scored[(r["model"], r["permute"])] += 1
            cell[(r["model"], r["permute"])][r["problem_id"]] = e
    vend = {m: v for v, m in FAM}
    print("\n=== CROSS-VENDOR IN-LOOP, fixed permutation, TOST margin |d| < 0.10 ===")
    for _, m in FAM:
        d0, d1 = cell.get((m, 0), {}), cell.get((m, 1), {})
        ks = sorted(set(d0) & set(d1))
        pr = (scored[(m, 0)] + scored[(m, 1)]) / max(attempted[(m, 0)] + attempted[(m, 1)], 1)
        short = m.split(".")[-1][:26]
        if len(ks) < 8:
            print("  %-9s %-27s usable %2d  parse-rate %.2f  -- too few to score"
                  % (vend[m], short, len(ks), pr))
            continue
        t0_ = statistics.mean(d0[k]["true"] for k in ks)
        r1 = statistics.mean(d1[k]["row"] for k in ks)
        t1 = statistics.mean(d1[k]["true"] for k in ks)
        c1 = statistics.mean(d1[k]["chance"] for k in ks)
        f0 = statistics.mean(d0[k]["feas"] for k in ks)
        f1 = statistics.mean(d1[k]["feas"] for k in ks)
        mm, se, hi, lo, sp = paired({k: d1[k]["row"] for k in ks},
                                    {k: d0[k]["true"] for k in ks}, ks)
        p_lo = 1 - 0.5 * (1 + erf(((mm + 0.10) / se) / sqrt(2))) if se else 0.0
        p_hi = 0.5 * (1 + erf(((mm - 0.10) / se) / sqrt(2))) if se else 0.0
        tost = max(p_lo, p_hi)
        bb, _, hb, lb, spb = paired({k: d1[k]["true"] for k in ks},
                                    {k: d1[k]["chance"] for k in ks}, ks)
        print("  %-9s %-27s n=%-3d parse %.2f | true %.3f  perm-row %.3f  perm-true %.3f "
              "chance %.3f | feas %.2f->%.2f"
              % (vend[m], short, len(ks), pr, t0_, r1, t1, c1, f0, f1))
        print("      false-vs-true address %+.3f (se %.3f, sign %d-%d p=%.1e) | TOST p=%.3f %s "
              "| perm-true vs chance %+.3f (p=%.1e)"
              % (mm, se, hi, lo, sp, tost,
                 "EQUIVALENT" if tost < 0.05 else "not equivalent", bb, spb))

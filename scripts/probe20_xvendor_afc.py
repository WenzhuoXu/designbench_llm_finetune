"""
EXPERIMENT 20 -- the detectability inversion, ACROSS VENDORS.

The novelty gate failed on exactly one thing: four "Claude families" served through one
Bedrock endpoint is one correlated observation, not four draws, and the load-bearing result
(the inversion) rested on two Anthropic models. Bedrock also serves Meta, Amazon, Mistral,
Writer and Alibaba, so the fix is available now.

Same forced-choice design as probe17: two tables for one structure, one truthful and one
corrupted, order randomised, "which is corrupted?". Chance 0.500 exactly, no false-positive
rate. Corruptions: value-noise (rank corr ~ +0.85, ordering nearly intact) and full address
permutation (rank corr ~ -0.05, ordering destroyed).

The prediction under the binding account: detection of value corruption >> detection of
address corruption, across vendors, even though address corruption destroys far more of the
ordering. If that inverts or vanishes outside Anthropic, the effect is a post-training
artifact and the claim is about one vendor.
"""
import os, sys, argparse, json, time, statistics, collections
from math import comb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(P), str(P / "scripts"), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from probe17_2afc import one
DB = Path("/ocean/projects/mch250030p/wxu7/DesignBench")

MODELS = [
    ("Anthropic", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ("Anthropic", "us.anthropic.claude-opus-4-1-20250805-v1:0"),
    ("Meta",      "us.meta.llama3-3-70b-instruct-v1:0"),
    ("Amazon",    "us.amazon.nova-pro-v1:0"),
    ("Mistral",   "mistral.mistral-large-2407-v1:0"),
    ("Mistral",   "us.mistral.pixtral-large-2502-v1:0"),
    ("Writer",    "us.writer.palmyra-x5-v1:0"),
    ("Alibaba",   "qwen.qwen3-235b-a22b-2507-v1:0"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--out", default=str(P / "results/api_guidance/xvendor_afc.jsonl"))
    a = ap.parse_args()
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or \
        open(os.path.expanduser("~/.bedrock_token")).read().strip()
    specs = [json.load(open(f)) for f in sorted((DB / "data/problems_hard").glob("*.json"))[:a.n]]
    cells = [(m, lv, mid) for (_, mid) in MODELS for (m, lv) in (("noise", 0.8), ("perm", 1.0))]
    tasks = [(s, m, lv, mid, a.region, tok) for s in specs for (m, lv, mid) in cells]
    print("models %d | cells %d | calls %d" % (len(MODELS), len(cells), len(tasks)), flush=True)
    rows = []
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "w") as fh:
        for i, r in enumerate(ex.map(one, tasks)):
            rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            if (i + 1) % 200 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    print("wall %.0fs" % (time.time() - t0))

    vend = {mid: v for (v, mid) in MODELS}
    ok = [r for r in rows if r.get("correct") is not None]
    print("\nscored %d/%d | unparsed %d | errors %d"
          % (len(ok), len(rows),
             sum(1 for r in rows if not r.get("err") and r.get("correct") is None),
             sum(1 for r in rows if r.get("err"))))

    def binom_p(k, n):
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

    by = collections.defaultdict(list)
    for r in ok:
        by[(r["model"], r["mode"])].append(r)
    print("\n=== 2AFC ACROSS VENDORS (chance = 0.500) ===")
    print("  vendor     model                          value-noise      address-perm     inversion")
    print("                                            (corr +0.85)     (corr -0.05)     value - address")
    for _, mid in MODELS:
        nz = by.get((mid, "noise"), []); pm = by.get((mid, "perm"), [])
        if not nz or not pm:
            print("  %-10s %-30s  MISSING" % (vend[mid], mid.split(".")[-1][:28]))
            continue
        kn, nn = sum(r["correct"] for r in nz), len(nz)
        kp, np_ = sum(r["correct"] for r in pm), len(pm)
        dn = {r["problem_id"]: r["correct"] for r in nz}
        dp = {r["problem_id"]: r["correct"] for r in pm}
        ks = sorted(set(dn) & set(dp))
        hi = sum(1 for k in ks if dn[k] and not dp[k]); lo = sum(1 for k in ks if dp[k] and not dn[k])
        dd = hi + lo
        pv = 1.0 if dd == 0 else min(1.0, 2 * sum(comb(dd, i) for i in range(0, min(hi, lo) + 1)) / 2 ** dd)
        print("  %-10s %-30s  %.3f (p=%.1e)  %.3f (p=%.1e)  %+.3f (sign %d-%d, p=%.2e)"
              % (vend[mid], mid.split(".")[-1][:28], kn / nn, binom_p(max(kn, nn - kn), nn),
                 kp / np_, binom_p(max(kp, np_ - kp), np_),
                 kn / nn - kp / np_, hi, lo, pv))
    print("\n  Anthropic vs non-Anthropic, address-permutation detection:")
    for grp in ("Anthropic", "non-Anthropic"):
        ms = [mid for (v, mid) in MODELS if (v == "Anthropic") == (grp == "Anthropic")]
        rs = [r for mid in ms for r in by.get((mid, "perm"), [])]
        if rs:
            print("    %-14s n=%-4d accuracy %.3f" % (grp, len(rs), sum(r["correct"] for r in rs) / len(rs)))

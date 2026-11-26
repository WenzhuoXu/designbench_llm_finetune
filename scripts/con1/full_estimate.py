"""con1 full row set: training time and memory estimates from stage 1's measured steps (docs/con1_full_rows.md).

    python full_estimate.py --stage1-rows train_facet.jsonl --stage1-lens lens_stage1.txt --metrics metrics.jsonl \\
        --tokenizer MODEL_DIR --lens q12=lens_q12.txt q36=lens_q36.txt [--ranks 4] [--per q12=64 q36=64,22]

1. Exact token lengths of stage 1's rows (conseq_format.encode, cached in --stage1-lens).
2. Fit of the step time: stage 1's batches are known (sft_rows.row_order, seed 42, 64 rows per step) and each rank's
   share follows samples.shard_samples (longest first to the least-loaded rank, by tokens). Per rank cost a*sum(T) +
   b*sum(T^2) (the T^2 term is attention), step time = c + max over ranks; least squares on the steps without a dev
   pass. Peak memory: linear fit of peak_mem_gib on the step's longest row.
3. For each candidate rows file (lens: total tokens per row, questions per row): the same simulation over the whole
   epoch -> steps, hours on --ranks GPUs, GPU-hours, outcomes per GPU-second, predicted peak memory of the longest row.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as stt
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def row_order(n: int, epochs: float, seed: int) -> list:
    """sft_rows.row_order (copied: importing sft_rows pulls in torch)."""
    total = int(round(n * float(epochs)))
    out, e = [], 0
    while len(out) < total:
        perm = list(range(n))
        random.Random("%d/%d" % (int(seed), e)).shuffle(perm)
        out.extend(perm)
        e += 1
    return out[:total]


def shard(lens: list, world: int) -> list:
    """samples.shard_samples' assignment: [[lengths of rank r]]."""
    order = sorted(range(len(lens)), key=lambda i: (-lens[i], i))
    load = [0] * world
    out = [[] for _ in range(world)]
    for i in order:
        r = min(range(world), key=lambda k: (load[k], k))
        load[r] += lens[i]
        out[r].append(lens[i])
    return out


def _tok_lens(args):
    path, lo, hi, tokp = args
    from transformers import AutoTokenizer
    from llm_finetune.rl_dataxp.conseq_format import encode
    tok = AutoTokenizer.from_pretrained(tokp, trust_remote_code=True)
    out = []
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k < lo:
                continue
            if k >= hi:
                break
            r = json.loads(line)
            e = encode(tok, r["messages"])
            out.append((len(e["prompt_ids"]) + len(e["completion_ids"]), len(e["prompt_ids"]),
                        len(r["con1"]["moves"]), r["domain"]))
    return out


def stage1_lens(path, cache, tokp, workers):
    if os.path.exists(cache):
        return [tuple(int(x) if i < 3 else x for i, x in enumerate(l.split())) for l in open(cache)]
    n = sum(1 for _ in open(path, encoding="utf-8"))
    step = int(math.ceil(n / workers))
    from multiprocessing import get_context
    with get_context("fork").Pool(workers) as pool:
        parts = pool.map(_tok_lens, [(path, i, min(n, i + step), tokp) for i in range(0, n, step)])
    rows = [x for p in parts for x in p]
    with open(cache + ".tmp", "w") as fo:
        for r in rows:
            fo.write("%d %d %d %s\n" % r)
    os.replace(cache + ".tmp", cache)
    return rows


def lstsq(X, y):
    """Least squares without numpy (normal equations, small)."""
    k = len(X[0])
    A = [[sum(X[i][p] * X[i][q] for i in range(len(X))) for q in range(k)] for p in range(k)]
    v = [sum(X[i][p] * y[i] for i in range(len(X))) for p in range(k)]
    for p in range(k):                                     # Gauss-Jordan
        piv = max(range(p, k), key=lambda r: abs(A[r][p]))
        A[p], A[piv], v[p], v[piv] = A[piv], A[p], v[piv], v[p]
        d = A[p][p]
        if abs(d) < 1e-30:
            raise ValueError("singular fit")
        for q in range(k):
            A[p][q] /= d
        v[p] /= d
        for r in range(k):
            if r != p and A[r][p]:
                f = A[r][p]
                for q in range(k):
                    A[r][q] -= f * A[p][q]
                v[r] -= f * v[p]
    return v


def fit_time(steps, lens, per, world):
    """steps: [(step, t_train)]; returns (c, a, b) of t = c + max_r(a*S1_r + b*S2_r), and the fit's residual sd."""
    order = row_order(len(lens), 1.0, 42)
    feats = []
    for s, t in steps:
        b = [lens[i] for i in order[s * per:(s + 1) * per]]
        feats.append(([(sum(x), sum(v * v for v in x)) for x in shard(b, world)], t))
    a, bq, c = 1.0, 0.0, 0.0
    for _ in range(8):                                     # the slowest rank under the current fit, then refit
        X, y = [], []
        for ranks, t in feats:
            s1, s2 = max(ranks, key=lambda r: a * r[0] + bq * r[1])
            X.append([1.0, s1 / 1e3, s2 / 1e6])
            y.append(t)
        c, a3, b6 = lstsq(X, y)
        a, bq = a3 / 1e3, b6 / 1e6
        if bq < 0:                                         # a negative attention term is noise: refit without it
            c, a3 = lstsq([[x[0], x[1]] for x in X], y)
            a, bq = a3 / 1e3, 0.0
        if c < 0:                                          # a negative per-step overhead is noise too (loads that
            a3, b6 = lstsq([[x[1], x[2]] for x in X], y)   # barely vary between steps): through the origin
            a, bq, c = a3 / 1e3, b6 / 1e6, 0.0
            if bq < 0:
                (a3,) = lstsq([[x[1]] for x in X], y)
                a, bq = a3 / 1e3, 0.0
    (a0,) = lstsq([[max(r[0] for r in ranks) / 1e3] for ranks, _ in feats], [t for _, t in feats])
    fit_time.tokens_only = a0 / 1e3
    pred = [c + max(a * r[0] + bq * r[1] for r in ranks) for ranks, _ in feats]
    res = [p - t for p, (_, t) in zip(pred, feats)]
    sd = math.sqrt(sum(r * r for r in res) / max(1, len(res) - 3))
    return c, a, bq, sd


def simulate(lens, per, world, c, a, b):
    order = row_order(len(lens), 1.0, 42)
    steps = int(math.ceil(len(order) / per))
    tot = 0.0
    for s in range(steps):
        bt = [lens[i] for i in order[s * per:(s + 1) * per]]
        tot += c + max(a * sum(x) + b * sum(v * v for v in x) for x in shard(bt, world))
    return steps, tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage1-rows", required=True)
    ap.add_argument("--stage1-lens", required=True, help="cache of stage 1's exact row lengths")
    ap.add_argument("--metrics", required=True, help="stage 1's metrics.jsonl (job 176033)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lens", nargs="+", required=True, help="name=lens file of build_full_rows.py")
    ap.add_argument("--per", nargs="*", default=[], help="name=rows per step[,rows per step]; default 64")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--dev-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s1 = stage1_lens(a.stage1_rows, a.stage1_lens, a.tokenizer, a.workers)
    L1 = [r[0] for r in s1]
    ms = [json.loads(l) for l in open(a.metrics) if l.strip()]
    steps = [(m["step"], m["t_train"]) for m in ms if "dev_loss" not in m and m["step"] > 0 and "t_train" in m]
    c, ta, tb, sd = fit_time(steps, L1, 64, a.ranks)
    # checks of the reconstruction: the longest row of each step and its supervised token count against the log
    order = row_order(len(L1), 1.0, 42)
    mism = sum(1 for m in ms if max(L1[i] for i in order[m["step"] * 64:(m["step"] + 1) * 64]) != m.get("max_row_tokens"))
    mem = [(m["max_row_tokens"], m["peak_mem_gib"]) for m in ms if m.get("peak_mem_gib")]
    rep = {"stage1": {"rows": len(L1), "tokens_M": round(sum(L1) / 1e6, 1), "longest": max(L1),
                      "outcomes": sum(r[2] for r in s1), "steps_fitted": len(steps),
                      "steps_whose_longest_row_mismatches_the_log": mism,
                      "tokens_only_s_per_ktok_slowest_rank": round(fit_time.tokens_only * 1e3, 4),
                      "fit_seconds_per_step": {"c": round(c, 2), "a_per_ktok": round(ta * 1e3, 4),
                                               "b_per_ktok2": round(tb * 1e6, 6), "residual_sd": round(sd, 1)},
                      "mean_step_s": round(stt.mean(t for _, t in steps), 1) if steps else None,
                      "peak_mem": {"steps": len(mem), "max_gib": max(y for _, y in mem) if mem else None,
                                   "longest_row_seen": max(x for x, _ in mem) if mem else None}},
           "candidates": {}}
    s1_steps, s1_sec = simulate(L1, 64, a.ranks, c, ta, tb)
    rep["stage1"]["simulated_epoch_h"] = round(s1_sec / 3600, 1)
    # a dev pass (stage 1's 120 dev rows, the continuation's dev set): its time is what a dev step takes beyond the fit
    dev_extra = []
    for m in ms:
        if "dev_loss" in m and "t_train" in m:
            bt = [L1[i] for i in order[m["step"] * 64:(m["step"] + 1) * 64]]
            dev_extra.append(m["t_train"] - (c + max(ta * sum(x) + tb * sum(v * v for v in x) for x in shard(bt, a.ranks))))
    rep["stage1"]["dev_pass_s"] = round(stt.mean(dev_extra), 1) if dev_extra else None
    per = {}
    for p in a.per:
        k, v = p.split("=")
        per[k] = [int(x) for x in v.split(",")]
    for spec in a.lens:
        name, path = spec.split("=")
        rows = [l.split() for l in open(path)]
        L = [int(r[0]) for r in rows]
        P = [int(r[1]) for r in rows]
        Q = [int(r[2]) for r in rows]
        fam = {}
        for r in rows:
            f = fam.setdefault(r[3], {"rows": 0, "outcomes": 0, "tokens": 0, "max": 0})
            f["rows"] += 1
            f["outcomes"] += int(r[2])
            f["tokens"] += int(r[0])
            f["max"] = max(f["max"], int(r[0]))
        Ls = sorted(L)
        cand = {"rows": len(L), "outcomes": sum(Q), "tokens_M": round(sum(L) / 1e6, 1),
                "tokens_per_row": {"mean": round(stt.mean(L)), "median": Ls[len(Ls) // 2],
                                   "p95": Ls[int(0.95 * (len(Ls) - 1))], "max": Ls[-1]},
                "prompt_tokens_mean": round(stt.mean(P)), "questions_per_row_mean": round(stt.mean(Q), 2),
                "tokens_per_outcome": round(sum(L) / max(1, sum(Q)), 1),
                # memory grows with the longest row of a rank's share: bounded by stage 1's peak over the steps
                # whose longest row is at least this set's longest
                "peak_gib_bound": max((y for x, y in mem if x >= Ls[-1]), default=None),
                "families": fam, "runs": {}}
        for pr in per.get(name, [64]):
            n_steps, sec = simulate(L, pr, a.ranks, c, ta, tb)
            _, sec0 = simulate(L, pr, a.ranks, 0.0, fit_time.tokens_only, 0.0)
            dev_h = (n_steps // a.dev_every) * (rep["stage1"]["dev_pass_s"] or 0) / 3600
            cand["runs"]["rows_per_step=%d" % pr] = {
                "steps": n_steps, "outcomes_per_step": round(sum(Q) / n_steps), "hours": round(sec / 3600, 1),
                "hours_tokens_only_model": round(sec0 / 3600, 1), "dev_passes_h": round(dev_h, 1),
                "days": round(sec / 86400, 2), "gpu_hours": round(sec * a.ranks / 3600),
                "outcomes_per_gpu_second": round(sum(Q) / (sec * a.ranks), 3),
                "outcomes_per_gpu_second_tokens_only_model": round(sum(Q) / (sec0 * a.ranks), 3)}
        rep["candidates"][name] = cand
    with open(a.out + ".tmp", "w") as fo:
        json.dump(rep, fo, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

"""Stage 2/3 student evaluation (docs/cot_stream_validation.md §4, M1-M3).

For each eval state, sample and probe budget B: the student runs the stream (cap = B). With
--side-budget (default the largest B), the held-out side question is asked after the student's
first message (M1 physics, M2 calibration). M3 is scored from the final action: Phi percentile
among the state's cached legal moves, accepted (pct >= 90 or the teacher's move), right element
(the action's element is one of the highest-Phi moves' elements), probes used.

    python -m dataxp.stream.evalrun run --states E.jsonl --out O.jsonl --endpoint URL --model NAME \
        [--thinking] --budgets 0 2 6 --k 2
    python -m dataxp.stream.evalrun score --states E.jsonl --runs O.jsonl [O2.jsonl ...] --out R.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..ownrows.common import move_seed, percentile
from .env import StreamState, element_of_text
from .protocol import run_stream
from .score import boot_ci, score_side


def final_scores(ss: StreamState, text):
    """(legal, phi, pct, accepted, right_element) of a final action text."""
    if not text:
        return False, None, None, False, False
    res = ss.check(text)
    if not res["ok"]:
        return False, None, None, False, False
    mv = ss.d.parse_move(res["applied"])
    from ..ownrows.common import try_apply
    child, ap, _ = try_apply(ss.d, ss.st, mv)
    phi = float(ss.ev.score(child, seed0=move_seed(ss.r["problem_id"], ss.d, ap)).mean)
    cand = np.array([m["phi"] for m in ss.r.get("moves") or []], float)
    pct = percentile(cand, phi) if cand.size else None
    gold = ss.r.get("gold_text")
    acc = bool((pct is not None and pct >= 90.0) or (gold and ss.same_move(res["applied"], gold)))
    top = cand.max() if cand.size else None
    tops = {element_of_text(ss.v, m["text"]) for m in ss.r.get("moves") or [] if top is not None and m["phi"] >= top - 1e-9}
    right = element_of_text(ss.v, res["applied"]) in tops
    return True, phi, pct, acc, right


def cmd_run(a):
    from .run import VLLMChat
    recs = [json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    side_b = a.side_budget if a.side_budget is not None else max(a.budgets)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            o = json.loads(l)
            if o.get("status") != "error":
                done.add((o["sid"], o["budget"], o["sample"]))
    jobs = [(r, b, k) for r in recs for b in a.budgets for k in range(a.k) if (r["sid"], b, k) not in done]
    print("%s: states %d, jobs %d" % (a.label, len(recs), len(jobs)), flush=True)
    chat = VLLMChat(a.endpoint, a.model, temperature=a.temperature, max_tokens=a.max_tokens, thinking=a.thinking)
    lock = threading.Lock()

    def one(r, b, k):
        try:
            o = run_stream(StreamState(r), chat, cap=b, side=(b == side_b))
        except Exception as e:  # noqa: BLE001
            o = {"sid": r["sid"], "status": "error", "error": repr(e)[:400]}
        o.update(budget=b, sample=k, label=a.label, model=a.model, thinking=a.thinking)
        return o

    with open(a.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(one, r, b, k) for r, b, k in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            o = f.result()
            with lock:
                fh.write(json.dumps(o) + "\n")
                fh.flush()
            if n % 25 == 0:
                print("[%d/%d]" % (n, len(jobs)), flush=True)


def cmd_score(a):
    recs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()}
    cache = {}
    out = {}
    per_state = {}
    for path in a.runs:
        runs = [json.loads(l) for l in open(path, encoding="utf-8")]
        label = runs[0].get("label", path) if runs else path
        m3 = defaultdict(lambda: defaultdict(list))
        m1 = defaultdict(list)
        cov = []
        for o in runs:
            if o.get("status") == "error" or o["sid"] not in recs:
                continue
            sid = o["sid"]
            if sid not in cache:
                cache[sid] = StreamState(recs[sid])
            ss = cache[sid]
            fam = ss.v
            legal, phi, pct, acc, right = final_scores(ss, o.get("final_action") if o.get("status") == "done" else None)
            d = m3[(fam, o["budget"])]
            d["legal"].append(legal)
            d["accepted"].append(acc)
            d["right"].append(right)
            d["pct"].append(pct if pct is not None else 0.0)
            d["probes"].append(o.get("n_probes", 0))
            pre = (o.get("side") or {}).get("pre")
            if pre:
                rows = score_side(ss, o["H"], pre.get("text"))
                if rows:
                    m1[sid].append(float(np.mean([r["nis"] for r in rows])))
                    cov += [r["cover"] for r in rows if r["cover"] is not None]
        per_state[label] = {s: float(np.mean(v)) for s, v in m1.items()}
        out[label] = {
            "M1_mean_nis": float(np.mean(list(per_state[label].values()))) if per_state[label] else None,
            "M1_n_states": len(per_state[label]),
            "M2_coverage": float(np.mean(cov)) if cov else None,
            "M3": {"%s|B%d" % k: {kk: float(np.mean(vv)) for kk, vv in v.items()} | {"n": len(v["legal"])}
                   for k, v in sorted(m3.items())}}
    labels = list(per_state)
    pairs = {}
    for i, x in enumerate(labels):
        for y in labels[i + 1:]:
            common = sorted(set(per_state[x]) & set(per_state[y]))
            diff = np.array([per_state[x][s] - per_state[y][s] for s in common])
            pairs["%s_minus_%s" % (x, y)] = boot_ci(diff) + (len(common),)
    out["M1_pairs"] = pairs
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1)[:8000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--states", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--endpoint", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--thinking", action="store_true")
    r.add_argument("--budgets", type=int, nargs="+", default=[0, 2, 6])
    r.add_argument("--side-budget", type=int, default=None)
    r.add_argument("--k", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--max-tokens", type=int, default=12000)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--states", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())

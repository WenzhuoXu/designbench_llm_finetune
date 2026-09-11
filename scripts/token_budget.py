#!/usr/bin/env python
"""How many tokens does each training tier actually contain?

An allocation request should be built on counted tokens, not on trajectory
counts multiplied by a guess. Each tier is sampled, tokenized with the target
model's own tokenizer through the same chat template training will use, and
extrapolated with a standard error so the estimate carries its own uncertainty.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
sys.path.insert(0, str(PROJECT))

from transformers import AutoTokenizer

TIERS = ["train_1k", "train_10k", "train_100k", "train_all", "dev_seen", "dev_catalogue"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--sample", type=int, default=1500)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        a.model, cache_dir="/ocean/projects/mch250030p/wxu7/hf_models")

    print("%-16s %9s %10s %12s %14s" % ("tier", "traj", "tok/traj", "sd", "total tokens"))
    out = {}
    for t in TIERS:
        f = PROJECT / "data/tiers" / (t + ".jsonl")
        if not f.exists():
            continue
        rows = f.read_text().splitlines()
        n = len(rows)
        rng = random.Random(20260911)
        idx = rng.sample(range(n), min(a.sample, n))
        lens = []
        for i in idx:
            msgs = json.loads(rows[i])["messages"]
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            if not isinstance(ids, list):
                ids = getattr(ids, "input_ids", ids)
            lens.append(len(ids))
        m = sum(lens) / len(lens)
        sd = math.sqrt(sum((x - m) ** 2 for x in lens) / max(len(lens) - 1, 1))
        se = sd / math.sqrt(len(lens))
        total = m * n
        out[t] = {"traj": n, "mean": m, "sd": sd, "total": total,
                  "total_ci": 1.96 * se * n}
        print("%-16s %9d %10.0f %12.0f %14s  +/- %s"
              % (t, n, m, sd, "{:,}".format(int(total)),
                 "{:,}".format(int(1.96 * se * n))))

    json.dump(out, open(PROJECT / "results/token_budget.json", "w"), indent=1)
    tr = sum(out[t]["total"] for t in ("train_1k", "train_10k", "train_100k", "train_all")
             if t in out)
    print("\nall four tiers, one epoch each : %s tokens" % "{:,}".format(int(tr)))
    print("at 2 epochs                    : %s tokens" % "{:,}".format(int(2 * tr)))

#!/usr/bin/env python
"""Render preference pairs into (prompt, chosen, rejected) text for DPO.

The prompt is produced by the SAME warmstart transform the RL rollout uses, so a
policy trained here is in-distribution when GRPO takes over: system prompt, user
problem, then alternating assistant <think>/<action> turns and [Simulation Result]
user turns.

    python scripts/prepare_dpo_data.py \\
        --pairs results/distill/search_preferences.jsonl \\
        --out results/distill/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PROJECT / "results/distill/search_preferences.jsonl"))
    ap.add_argument("--out", default=str(PROJECT / "results/distill/dpo_pairs.jsonl"))
    ap.add_argument("--dev-out", default=str(PROJECT / "results/distill/dpo_pairs_dev.jsonl"))
    ap.add_argument("--tokenizer",
                    default="checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final")
    ap.add_argument("--base-model-id", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-prompt-chars", type=int, default=24000,
                    help="Drop pairs whose prompt is implausibly long rather than truncate them; "
                         "a truncated prompt changes the state the preference refers to.")
    ap.add_argument("--dev-problems", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.data.processors.warmstart_transform import WarmstartTransform

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    fmt = ChatFormatter.from_model_id(args.base_model_id, tok)
    transform = WarmstartTransform()

    rows, dropped = [], 0
    for line in open(args.pairs):
        rec = json.loads(line)
        # The pair's `messages` are raw DesignBench-format; the transform turns them
        # into the exact conversation the RL rollout builds.
        messages = transform.transform_messages(list(rec["messages"]))
        prompt = fmt.apply_template(messages, add_generation_prompt=True, tokenize=False)
        if len(prompt) > args.max_prompt_chars:
            dropped += 1
            continue
        rows.append({
            "problem_id": rec["problem_id"],
            "step": rec["step"],
            "prompt": prompt,
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
            "phi_gap": rec["phi_gap"],
        })

    # Hold out whole PROBLEMS, never individual pairs: pairs from the same problem
    # share a prompt prefix and would leak.
    problems = sorted({r["problem_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(problems)
    dev_problems = set(problems[: args.dev_problems])
    train = [r for r in rows if r["problem_id"] not in dev_problems]
    dev = [r for r in rows if r["problem_id"] in dev_problems]

    for path, part in ((args.out, train), (args.dev_out, dev)):
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")

    import statistics as st
    print(f"rendered {len(rows)} pairs ({dropped} dropped as over-long)")
    print(f"  train {len(train)} pairs / {len({r['problem_id'] for r in train})} problems -> {args.out}")
    print(f"  dev   {len(dev)} pairs / {len({r['problem_id'] for r in dev})} problems -> {args.dev_out}")
    print(f"  problem overlap: {len({r['problem_id'] for r in train} & {r['problem_id'] for r in dev})}")
    lens = [len(tok(r['prompt'])['input_ids']) for r in rows[:200]]
    print(f"  prompt tokens (first 200): median {st.median(lens):.0f}  p90 {sorted(lens)[int(.9*len(lens))]}")
    comp = [len(tok(r['chosen'])['input_ids']) for r in rows[:200]]
    print(f"  completion tokens        : median {st.median(comp):.0f}  max {max(comp)}")


if __name__ == "__main__":
    main()

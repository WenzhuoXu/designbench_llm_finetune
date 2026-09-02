#!/usr/bin/env python
"""Does the policy prefer the action the search chose? One forward pass per side.

The instrument this replaces. Held-out feasibility on 34 problems has, measured on
this study's own data, a within-arm standard deviation of 0.051 across checkpoints
of the SAME run against a between-arm standard deviation of 0.046 -- the noise
exceeds the signal. Batch size alone moved it 17 points on a fixed checkpoint
(batch 1 reproduced the sequential path 12/12; batch 8 diverged to 8/12), because
one flipped greedy token early in a 14-turn rollout changes the trajectory.

This measures the ranking directly instead of through a rollout:

    accuracy = fraction of pairs with  log p(chosen | state) > log p(rejected | state)
    margin   = mean of log p(chosen) - log p(rejected)          (per token and total)

The pairs come from scripts/distill_preference_pairs.py in its same-move form: the
rejected action is the IDENTICAL move -- same action type, same parameter, same
scale factor -- on a DIFFERENT element. The reasoning text is byte-identical on
both sides. So the surface-feature ceiling is exactly 0.50, and any accuracy above
chance is knowledge of WHICH ELEMENT to act on: the critical-element competence
that the feedback field supplies for free at depth 0 and that the search computes
by simulation.

Three reasons this is the better instrument:
  * hundreds of paired comparisons instead of 34 binary outcomes;
  * every arm is scored on byte-identical states, so the comparison is paired and
    the between-problem variance differences out;
  * no generation at all -- two short teacher-forced forward passes per pair -- so
    it is roughly two orders of magnitude cheaper than a rollout evaluation and
    immune to the decoding chaos that makes rollouts unstable.

    python scripts/eval_preference_margin.py \\
        --checkpoint checkpoints/grpo/<run>/checkpoint-100 \\
        --pairs results/distill/prefs_eval_split.jsonl \\
        --out results/regret/<run>.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)


def _completion_logprobs(model, ids, attn, cmask, keep, pos):
    """Teacher-forced sum of log p(token | prefix) over the completion tokens.

    Only the last ``keep`` positions get logits: completions are ~40 tokens against
    ~1400-token prompts, so scoring the whole sequence would materialise
    [B, 1500, 152k] logits and waste most of the memory bandwidth on the prompt.
    Verified equal to the full-sequence computation to 4e-6.
    """
    import torch
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=keep)
    logits = out.logits.float()
    targets = ids[:, -keep + 1:] if keep > 1 else ids[:, -1:]
    logits = logits[:, :-1, :] if keep > 1 else logits
    mask = cmask[:, -keep + 1:] if keep > 1 else cmask[:, -1:]
    token_lp = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    return (token_lp * mask).sum(-1), mask.sum(-1)


def _build(rows, tok, device, max_len):
    """LEFT-padded batch of prompt+completion for both sides of each pair."""
    import torch
    seqs, masks = [], []
    for r in rows:
        for key in ("chosen", "rejected"):
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r[key], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p) + len(c) > max_len:
                p = p[-(max_len - len(c)):]
            seqs.append(p + c)
            masks.append([0] * len(p) + [1] * len(c))
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    cmask = torch.zeros((len(seqs), width), dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        off = width - len(s)                      # left pad: the completion must end at the tail
        ids[i, off:] = torch.tensor(s, dtype=torch.long)
        attn[i, off:] = 1
        cmask[i, off:] = torch.tensor(m, dtype=torch.float)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)      # left padding shifts RoPE without this
    keep = int(max(int(m.sum().item()) for m in cmask)) + 1
    return ids.to(device), attn.to(device), cmask.to(device), keep, pos.to(device)


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="LoRA adapter directory, or 'base'. Repeatable: several adapters are "
                         "scored against ONE base-model load, which is the dominant cost "
                         "(~5 min load against ~2.6 min of scoring per arm).")
    ap.add_argument("--name", action="append", default=None,
                    help="Output name per --checkpoint, in the same order.")
    ap.add_argument("--base-model-id", default="Qwen/Qwen3-14B")
    ap.add_argument("--tokenizer",
                    default="checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final")
    ap.add_argument("--pairs", default=str(PROJECT / "results/distill/prefs_eval_split.jsonl"))
    ap.add_argument("--base-model-id-for-prompt", default="Qwen/Qwen3-14B")
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, dtype=torch.bfloat16, attn_implementation=attn_impl,
        device_map={"": 0},
    )
    checkpoints = list(args.checkpoint)
    names = list(args.name) if args.name else [Path(c).parent.name if c != "base" else "base"
                                               for c in checkpoints]
    if len(names) != len(checkpoints):
        raise SystemExit("--name must be given once per --checkpoint")

    # One base load, then adapters swapped in place.
    from peft import PeftModel
    adapters = [(n, c) for n, c in zip(names, checkpoints) if c != "base"]
    if adapters:
        first_name, first_ckpt = adapters[0]
        model = PeftModel.from_pretrained(model, first_ckpt, adapter_name=first_name)
        for n, c in adapters[1:]:
            model.load_adapter(c, adapter_name=n)
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device

    rows = [json.loads(l) for l in open(args.pairs)]
    # The pairs file from distill_preference_pairs.py carries raw DesignBench
    # messages; prepare_dpo_data.py renders them. Accept either shape.
    if rows and "prompt" not in rows[0]:
        from llm_finetune.data.processors.chat_formatter import ChatFormatter
        from llm_finetune.data.processors.warmstart_transform import WarmstartTransform
        fmt = ChatFormatter.from_model_id(args.base_model_id_for_prompt, tok)
        transform = WarmstartTransform()
        rendered = []
        for r in rows:
            msgs = transform.transform_messages(list(r["messages"]))
            rendered.append({
                "problem_id": r["problem_id"], "step": r.get("step"),
                "prompt": fmt.apply_template(msgs, add_generation_prompt=True, tokenize=False),
                "chosen": r["chosen"], "rejected": r["rejected"],
                "phi_gap": r.get("phi_gap"),
            })
        rows = rendered
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} pairs over {len({r['problem_id'] for r in rows})} problems", flush=True)

    for ckpt_name, ckpt_path in zip(names, checkpoints):
        if ckpt_path == "base":
            ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else None
        else:
            model.set_adapter(ckpt_name)
            ctx = None
        print(f"\n=== scoring {ckpt_name} ({ckpt_path}) ===", flush=True)
        _score_one(model, tok, rows, device, args, ckpt_name, ckpt_path, ctx)
    return


def _score_one(model, tok, rows, device, args, ckpt_name, ckpt_path, ctx):
    import torch
    per_pair = []
    t0 = time.time()
    cm = ctx if ctx is not None else _null_ctx()
    with torch.no_grad(), cm:
        for i in range(0, len(rows), args.batch_pairs):
            chunk = rows[i: i + args.batch_pairs]
            ids, attn, cmask, keep, pos = _build(chunk, tok, device, args.max_len)
            lp, ntok = _completion_logprobs(model, ids, attn, cmask, keep, pos)
            lp = lp.view(-1, 2).tolist()
            ntok = ntok.view(-1, 2).tolist()
            for r, (lc, lr), (nc, nr) in zip(chunk, lp, ntok):
                per_pair.append({
                    "problem_id": r["problem_id"], "step": r.get("step"),
                    "logp_chosen": lc, "logp_rejected": lr,
                    "margin": lc - lr,
                    # length-normalised: the two sides differ only in an element id,
                    # so token counts are nearly equal, but do not assume it
                    "margin_per_token": lc / max(nc, 1) - lr / max(nr, 1),
                    "correct": bool(lc > lr), "phi_gap": r.get("phi_gap"),
                })
            if (i // max(args.batch_pairs, 1)) % 40 == 0:
                print(f"  {i + len(chunk)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    n = len(per_pair)
    acc = sum(p["correct"] for p in per_pair) / n
    margins = [p["margin"] for p in per_pair]
    # Cluster by problem: pairs from one problem share a prompt prefix.
    by_problem = defaultdict(list)
    for p in per_pair:
        by_problem[p["problem_id"]].append(p["correct"])
    prob_acc = [sum(v) / len(v) for v in by_problem.values()]
    se_cluster = (st.stdev(prob_acc) / math.sqrt(len(prob_acc))) if len(prob_acc) > 1 else float("nan")
    agg = {
        "checkpoint": ckpt_path,
        "pairs_file": args.pairs,
        "n_pairs": n,
        "n_problems": len(by_problem),
        "accuracy": acc,
        "accuracy_se_naive": math.sqrt(acc * (1 - acc) / n),
        # The honest interval: pairs within a problem are correlated, so cluster
        # by problem and take the standard error of the per-problem rates.
        "accuracy_se_clustered": se_cluster,
        "mean_margin": st.mean(margins),
        "median_margin": st.median(margins),
        "mean_margin_per_token": st.mean([p["margin_per_token"] for p in per_pair]),
        "surface_feature_ceiling": 0.50,
        "seconds": time.time() - t0,
    }
    out = Path(args.out) / f"{ckpt_name}.json" if Path(args.out).suffix == "" else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"aggregate": agg, "per_pair": per_pair}, open(out, "w"), indent=2)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()

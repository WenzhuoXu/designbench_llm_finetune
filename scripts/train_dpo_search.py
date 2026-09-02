#!/usr/bin/env python
"""Preference training on the search's rankings (minimal DPO).

Why not TRL: `trl.DPOTrainer` cannot be imported in this environment --
`from torch.distributed.fsdp import FSDPModule` fails on the installed torch
(2.4), while GRPOTrainer works because it never touches that path. Upgrading
torch would disturb the training stack that five GRPO arms are currently running
on, so the objective is implemented here instead. It is ~60 lines of actual
mathematics:

    L = -log sigmoid( beta * [ (logp_pi(y_w|x) - logp_ref(y_w|x))
                             - (logp_pi(y_l|x) - logp_ref(y_l|x)) ] )

With LoRA the reference policy is nearly free. It must be the policy's own
INITIALISATION (the warmstart adapter), not the base model -- `disable_adapter()`
would anchor the KL term to raw Qwen3-14B and make the warmstart's format
training look like a preference. So the warmstart adapter is loaded TWICE: once
trainable as "default", once frozen as "ref". The second copy is a LoRA, a few
tens of MB, against a 28 GB base that both share.

The data is what SFT threw away. Cloning the search's argmax moved the policy's
regret against procedural candidates from 1.606 to 3.427 -- it got WORSE at
choosing after seeing 2563 of the search's choices -- because a demonstration
shows the winner and never the comparison. Each pair here carries the comparison,
with identical reasoning on both sides so only the action differs.

    python scripts/train_dpo_search.py --epochs 2 --out checkpoints/dpo/search_prefs
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)


def completion_logprobs(model, input_ids, attention_mask, completion_mask, keep,
                        position_ids=None):
    """Sum of log p(token | prefix) over the completion tokens of each sequence.

    Only the last ``keep`` positions get logits. The completions here are ~40
    tokens against ~1700-token prompts, so scoring the whole sequence would
    materialise [B, 3072, 152k] logits -- 3.7 GB in bf16, 7.5 GB more if promoted
    to float32 for a log_softmax -- and OOM an 80 GB H100 before the first step.
    A logsumexp-and-gather also avoids ever holding a second full copy.
    """
    import torch
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, logits_to_keep=keep)
    logits = out.logits.float()                       # [B, keep, V]
    # logits[:, i] predicts input_ids[:, -keep + i + 1]
    targets = input_ids[:, -keep + 1:] if keep > 1 else input_ids[:, -1:]
    logits = logits[:, :-1, :] if keep > 1 else logits
    mask = completion_mask[:, -keep + 1:] if keep > 1 else completion_mask[:, -1:]
    token_lp = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    return (token_lp * mask).sum(dim=-1)


def build_batch(rows, tok, device, max_len):
    """Tokenise prompt+completion pairs, masking everything but the completion."""
    import torch
    seqs, masks = [], []
    for r in rows:
        for key in ("chosen", "rejected"):
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r[key], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p) + len(c) > max_len:          # keep the completion whole
                p = p[-(max_len - len(c)):]
            seqs.append(p + c)
            masks.append([0] * len(p) + [1] * len(c))
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    cmask = torch.zeros((len(seqs), width), dtype=torch.float)
    # LEFT padding: logits_to_keep slices the LAST positions, so every sequence's
    # completion must end at the same index. With right padding the slice would
    # read pad tokens for any sequence shorter than the widest.
    for i, (s, m) in enumerate(zip(seqs, masks)):
        off = width - len(s)
        ids[i, off:] = torch.tensor(s, dtype=torch.long)
        attn[i, off:] = 1
        cmask[i, off:] = torch.tensor(m, dtype=torch.float)
    # Left padding shifts RoPE unless position ids are given explicitly.
    pos = (attn.cumsum(dim=-1) - 1).clamp(min=0)
    # How many trailing positions can carry a completion token, plus one for the
    # shift. Bounded well below the sequence length because completions are short.
    keep = int(max(int(m.sum().item()) for m in cmask)) + 1
    return (ids.to(device), attn.to(device), cmask.to(device), keep,
            pos.to(device))


def evaluate(model, rows, tok, device, beta, max_len, batch_pairs):
    import torch
    model.eval()
    n_correct = n = 0
    losses = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_pairs):
            chunk = rows[i: i + batch_pairs]
            ids, attn, cmask, keep, pos = build_batch(chunk, tok, device, max_len)
            pi = completion_logprobs(model, ids, attn, cmask, keep, pos)
            model.set_adapter("ref")
            ref = completion_logprobs(model, ids, attn, cmask, keep, pos)
            model.set_adapter("default")
            delta = (pi - ref).view(-1, 2)
            margin = beta * (delta[:, 0] - delta[:, 1])
            losses.append(-torch.nn.functional.logsigmoid(margin).mean().item())
            n_correct += (margin > 0).sum().item()
            n += margin.numel()
    model.train()
    return sum(losses) / max(len(losses), 1), n_correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(PROJECT / "results/distill/dpo_pairs.jsonl"))
    ap.add_argument("--dev", default=str(PROJECT / "results/distill/dpo_pairs_dev.jsonl"))
    ap.add_argument("--base-model-id", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter",
                    default="checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final")
    ap.add_argument("--out", default=str(PROJECT / "checkpoints/dpo/search_prefs"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-pairs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.adapter)

    # flash-attention-2 is not importable on every node in this cluster (the wheel
    # wants GLIBC 2.32); fall back rather than lose the run to node placement.
    attn = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except Exception as exc:  # noqa: BLE001
        print(f"flash-attn unavailable ({type(exc).__name__}); using sdpa", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, dtype=torch.bfloat16, attn_implementation=attn,
        device_map={"": 0},
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="default",
                                      is_trainable=True)
    # Frozen snapshot of the initialisation: the DPO reference.
    model.load_adapter(args.adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("default")
    model.config.use_cache = False
    model.train()
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)

    train_rows = [json.loads(l) for l in open(args.train)]
    dev_rows = [json.loads(l) for l in open(args.dev)]
    random.shuffle(train_rows)
    print(f"train {len(train_rows)} pairs, dev {len(dev_rows)} pairs", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_rows) / (args.batch_pairs * args.grad_accum))
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    print(f"{steps_per_epoch} optimizer steps/epoch, {total} total", flush=True)

    loss0, acc0 = evaluate(model, dev_rows[:120], tok, device, args.beta, args.max_len, args.batch_pairs)
    # At step 0 the policy IS the reference, so the margin is identically zero and
    # accuracy is 0.5 by construction. It is logged as a wiring check: any other
    # value means the two adapters are not what they should be.
    print(f"[dev @ step 0] loss {loss0:.4f}  preference accuracy {acc0:.3f}  "
          f"(must be ~0.693 / 0.5 -- policy and reference are identical here)", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    history = [{"step": 0, "dev_loss": loss0, "dev_acc": acc0}]
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        opt.zero_grad(set_to_none=True)
        micro = 0
        for i in range(0, len(train_rows), args.batch_pairs):
            chunk = train_rows[i: i + args.batch_pairs]
            ids, attn, cmask, keep, pos = build_batch(chunk, tok, device, args.max_len)
            pi = completion_logprobs(model, ids, attn, cmask, keep, pos)
            with torch.no_grad():
                model.set_adapter("ref")
                ref = completion_logprobs(model, ids, attn, cmask, keep, pos)
                model.set_adapter("default")
            delta = (pi - ref).view(-1, 2)
            margin = args.beta * (delta[:, 0] - delta[:, 1])
            loss = -torch.nn.functional.logsigmoid(margin).mean() / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"  epoch {epoch} step {step}/{total} "
                          f"loss {loss.item()*args.grad_accum:.4f} "
                          f"acc {(margin > 0).float().mean().item():.2f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                if step % args.eval_every == 0:
                    dl, da = evaluate(model, dev_rows[:120], tok, device, args.beta,
                                      args.max_len, args.batch_pairs)
                    print(f"[dev @ step {step}] loss {dl:.4f}  preference accuracy {da:.3f}", flush=True)
                    history.append({"step": step, "dev_loss": dl, "dev_acc": da})
                    json.dump(history, open(out / "history.json", "w"), indent=2)
                    model.save_pretrained(str(out / "final"))
    dl, da = evaluate(model, dev_rows, tok, device, args.beta, args.max_len, args.batch_pairs)
    print(f"[dev FINAL, all {len(dev_rows)} pairs] loss {dl:.4f}  preference accuracy {da:.3f}", flush=True)
    history.append({"step": step, "dev_loss": dl, "dev_acc": da, "final": True})
    json.dump(history, open(out / "history.json", "w"), indent=2)
    model.save_pretrained(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    print(f"saved -> {out/'final'}", flush=True)


if __name__ == "__main__":
    main()

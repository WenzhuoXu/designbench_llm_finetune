"""A preference round on pairs of the policy's own replies, continuing an adapter: per state, a reply the
simulator bore out (chosen) against one it refused or that did not raise the potential (rejected), both
sampled from the policy at that state (scripts/rl_screen/build_pairs.py). On the model, long-sequence path,
checkpoints and multi-GPU reduction of the RL trainer, with the step, schedule and row handling of sft_rows.

The loss of a pair (DPO with a likelihood term on the chosen reply, as in iterative reasoning preference
optimisation, Pang et al. 2024):

    m    = beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]       (sums over the completion tokens)
    loss = -log sigmoid(m) + alpha * nll(c)                                    (nll: per-token mean, as sft_rows)

ref is the frozen initial adapter (grpo.kl_reference = init: a second copy of policy.init_adapter). The
completion of a reply is reasoning.strip() + '\\n</think>\\n\\n' + content.strip() + '<|im_end|>' after the served
prompt, as sft_rows. A replay row (a plain row: messages end with the assistant turn) adds the sft_rows loss
nll(row) * pref.replay_weight. An optimizer step averages pref.items_per_step items (pairs and replay rows), every
item weighing the same.

Why a pair and not a row: when every row is the policy's own reply, most of what the supervised loss of a row
does is shared by every own reply (sharpening the policy; sft_rows.py measured the part only the screening has
at under 1% of a step), and the open-rules round (13,166 such rows) changed the moves without making them
better. Two own replies at the same state share that part, and the pair's gradient beta*sigmoid(-m)*(grad
log pi(c) - grad log pi(r)) cancels it.

Gradient, without keeping two graphs of long replies at once: log pi(r) is taken first without a gradient, then
the chosen reply's forward and backward run with weight -w on log pi(c) (w = beta*sigmoid(-m), from the
detached sums) plus the nll term, then the rejected reply's forward and backward with weight +w on log pi(r).
This is the gradient of the loss above (dropout is off, so the second pass of r gives the first pass's value;
the step reports the largest difference).

    pref:
      pairs_file: pairs.jsonl       # build_pairs.py output (pairs) and/or plain rows (replay), one per line
      replay_file: null             # optional: plain rows mixed in (replay_per_pair of them per pair, seeded)
      replay_per_pair: 0.25
      dev_pairs_file: null          # optional: pairs scored (no gradient) every dev_every steps
      dev_file: null                # optional: rows scored as sft_rows does
      beta: 0.1
      alpha: 1.0                    # weight of the chosen reply's per-token nll
      replay_weight: 1.0
      epochs: 1.0
      items_per_step: 64
      lr: 3.0e-5
      warmup_ratio: 0.03
      weight_decay: 0.0
      max_grad_norm: 1.0
      temperature: 1.0
      seed: 42
      save_every: 10
      dev_every: 20
      keep_adapter_every: 40

    python -m llm_finetune.rl_dataxp.dpo_rows --config configs/rl/<cfg>.yaml [key=value ...]
(under torchrun for several GPUs). Resumes from the newest complete checkpoint of run_dir; the final adapter is
run_dir/final.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Callable

from . import trainer as TR
from .diskio import retry_when_full, write_text
from .sft_rows import _count, completion_text, lr_factor, read_rows, row_order, row_sample, score_rows, set_lora_dropout

PREF_DEFAULTS = {"pairs_file": None, "replay_file": None, "replay_per_pair": 0.25, "dev_pairs_file": None, "dev_file": None,
                 "beta": 0.1, "alpha": 1.0, "replay_weight": 1.0, "epochs": 1.0, "items_per_step": 64, "lr": 3.0e-5,
                 "warmup_ratio": 0.03, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1.0e-8, "max_grad_norm": 1.0,
                 "lora_dropout": 0.0, "temperature": 1.0, "seed": 42, "save_every": 10, "dev_every": 20,
                 "keep_adapter_every": 40}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def softplus(x: float) -> float:
    """log(1 + exp(x)) without overflow."""
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def is_pair(item: dict) -> bool:
    return "chosen" in item and "rejected" in item


def pair_sample(pair: dict, tok) -> dict:
    """prompt_ids and the two completions of a pair, tokenised as they are trained (row_sample's checks)."""
    c = row_sample({"messages": list(pair["messages"]) + [dict(pair["chosen"], role="assistant")]}, tok)
    r = tok.encode(completion_text(pair["rejected"]), add_special_tokens=False)
    if not c["completion_ids"] or not r:
        raise ValueError("empty completion")
    return {"kind": "pair", "prompt_ids": c["prompt_ids"], "chosen_ids": c["completion_ids"], "rejected_ids": list(r),
            "domain": pair.get("domain"), "tier": pair.get("tier"), "rejected_kind": pair.get("rejected_kind")}


def item_sample(item: dict, tok) -> dict:
    if is_pair(item):
        return pair_sample(item, tok)
    s = row_sample(item, tok)
    s["kind"] = "row"
    return s


def item_tokens(s: dict) -> int:
    if s["kind"] == "pair":
        return len(s["prompt_ids"]) + max(len(s["chosen_ids"]), len(s["rejected_ids"]))
    return len(s["prompt_ids"]) + len(s["completion_ids"])


def item_cost(s: dict) -> float:
    """Work of an item in forward-pass units (a pass with a gradient and its backward ~ 4 forward passes, gradient
    checkpointing included): a row is one such pass; a pair is two of them plus three forward passes (the two
    reference sums and the policy's rejected sum)."""
    if s["kind"] == "pair":
        c = len(s["prompt_ids"]) + len(s["chosen_ids"])
        r = len(s["prompt_ids"]) + len(s["rejected_ids"])
        return 5.0 * c + 6.0 * r
    return 4.0 * (len(s["prompt_ids"]) + len(s["completion_ids"]))


def shard_items(samples: list[dict], world: int, rank: int) -> list[dict]:
    """This rank's share, balanced by item_cost (costliest first, to the least-loaded rank); deterministic, so the
    ranks agree without communicating (samples.shard_samples' rule with a pair's cost)."""
    order = sorted(range(len(samples)), key=lambda i: (-item_cost(samples[i]), i))
    load = [0.0] * world
    mine = []
    for i in order:
        r = min(range(world), key=lambda k: (load[k], k))
        load[r] += item_cost(samples[i])
        if r == rank:
            mine.append(samples[i])
    return mine


def pair_terms(policy, s: dict, beta: float, temperature: float):
    """(ref_c, ref_r, pol_r) sums without a gradient: the reference's two sums and the policy's rejected sum."""
    ref_c = float(policy.ref_logprobs(s["prompt_ids"], s["chosen_ids"], temperature).double().sum())
    ref_r = float(policy.ref_logprobs(s["prompt_ids"], s["rejected_ids"], temperature).double().sum())
    import torch
    with torch.no_grad():
        lp, _ = policy.logprobs(s["prompt_ids"], s["rejected_ids"], temperature)
    return ref_c, ref_r, float(lp.double().sum())


def train_step(policy, samples: list[dict], dist, opt, p: dict) -> dict:
    """One optimizer step on the mean item loss of `samples` (this rank's share)."""
    import torch
    beta, alpha, temp = float(p["beta"]), float(p["alpha"]), float(p["temperature"])
    mine = shard_items(samples, dist.world, dist.rank)
    N = len(samples)
    # pairs, loss, margin (m / beta), accurate (m > 0), chosen logratio, rejected logratio, nll_c, nll_c tokens,
    # replay rows, replay nll, max |log pi(r) twice| difference, rejected-refused pairs, their margin
    st = torch.zeros(13, dtype=torch.float64, device=policy.device)
    cuda = policy.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(policy.device)
    for s in mine:
        if s["kind"] == "row":
            lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], temp)
            nll = -lp.mean()
            if not bool(torch.isfinite(nll)):
                raise RuntimeError("non-finite loss on a replay row of %d tokens" % item_tokens(s))
            (nll * float(p["replay_weight"]) / N).backward()
            with torch.no_grad():
                st[8] += 1
                st[9] += nll.double()
            continue
        ref_c, ref_r, pol_r0 = pair_terms(policy, s, beta, temp)
        lp_c, _ = policy.logprobs(s["prompt_ids"], s["chosen_ids"], temp)
        sum_c = lp_c.sum()
        m = beta * ((float(sum_c.detach().double()) - ref_c) - (pol_r0 - ref_r))
        w = beta * sigmoid(-m)
        nll_c = -lp_c.mean()
        loss_c = -w * sum_c + alpha * nll_c
        if not bool(torch.isfinite(loss_c)):
            raise RuntimeError("non-finite loss on a pair of %d tokens (%s)" % (item_tokens(s), s.get("domain")))
        (loss_c / N).backward()
        lp_r, _ = policy.logprobs(s["prompt_ids"], s["rejected_ids"], temp)
        sum_r = lp_r.sum()
        (w * sum_r / N).backward()
        with torch.no_grad():
            dpo = softplus(-m)                                            # -log sigmoid(m)
            st[0] += 1
            st[1] += dpo + alpha * float(nll_c.double())
            st[2] += m / beta
            st[3] += float(m > 0)
            st[4] += float(sum_c.double()) - ref_c
            st[5] += pol_r0 - ref_r
            st[6] += -float(lp_c.double().sum())
            st[7] += lp_c.numel()
            st[10] = max(float(st[10]), abs(float(sum_r.double()) - pol_r0))
            if s.get("rejected_kind") == "refused":
                st[11] += 1
                st[12] += m / beta
    dist.all_reduce_grads(policy.params)
    gn = float(torch.nn.utils.clip_grad_norm_(policy.params, float(p["max_grad_norm"])))
    opt.step()
    opt.zero_grad(set_to_none=True)
    mx = st[10:11].clone()
    dist.all_reduce_(st)
    dist.all_reduce_(mx, op="max")
    peak = torch.tensor([torch.cuda.max_memory_allocated(policy.device) / 2 ** 30 if cuda else 0.0],
                        dtype=torch.float64, device=policy.device)
    dist.all_reduce_(peak, op="max")
    n_p = max(1.0, float(st[0]))
    return {"pairs": int(st[0]), "loss": float(st[1]) / n_p, "margin": float(st[2]) / n_p, "acc": float(st[3]) / n_p,
            "chosen_logratio": float(st[4]) / n_p, "rejected_logratio": float(st[5]) / n_p,
            "nll_c_per_token": float(st[6]) / max(1.0, float(st[7])), "replay_rows": int(st[8]),
            "replay_nll": float(st[9]) / max(1.0, float(st[8])), "rejected_recompute_maxdiff": float(mx[0]),
            "refused_pairs": int(st[11]), "refused_margin": float(st[12]) / max(1.0, float(st[11])),
            "grad_norm": gn, "peak_mem_gib": round(float(peak[0]), 2) if cuda else None,
            "max_item_tokens": max(item_tokens(s) for s in samples)}


def score_pairs(policy, samples: list[dict], dist, beta: float, temperature: float) -> dict:
    """Dev pairs without a gradient: mean margin (policy logratio of chosen minus rejected, against the reference),
    the share with a positive margin, the two logratios, and the chosen reply's per-token nll."""
    import torch
    mine = shard_items(samples, dist.world, dist.rank)
    st = torch.zeros(7, dtype=torch.float64, device=policy.device)
    policy.model.eval()
    try:
        with torch.no_grad():
            for s in mine:
                ref_c, ref_r, pol_r = pair_terms(policy, s, beta, temperature)
                lp_c, _ = policy.logprobs(s["prompt_ids"], s["chosen_ids"], temperature)
                pol_c = float(lp_c.double().sum())
                d = (pol_c - ref_c) - (pol_r - ref_r)
                st[0] += 1
                st[1] += d
                st[2] += float(d > 0)
                st[3] += pol_c - ref_c
                st[4] += pol_r - ref_r
                st[5] += -pol_c
                st[6] += lp_c.numel()
    finally:
        policy.model.train()
    dist.all_reduce_(st)
    n = max(1.0, float(st[0]))
    return {"dev_pairs": int(st[0]), "dev_margin": float(st[1]) / n, "dev_acc": float(st[2]) / n,
            "dev_chosen_logratio": float(st[3]) / n, "dev_rejected_logratio": float(st[4]) / n,
            "dev_nll_c_per_token": float(st[5]) / max(1.0, float(st[6]))}


def pref_fingerprint(cfg: dict, data_sha1: str) -> str:
    p = cfg["pref"]
    sub = {"data_sha1": data_sha1, "model": cfg["model"]["path"], "policy": cfg["policy"],
           **{k: p[k] for k in ("beta", "alpha", "replay_weight", "replay_per_pair", "epochs", "items_per_step", "lr",
                                "warmup_ratio", "weight_decay", "betas", "eps", "max_grad_norm", "lora_dropout",
                                "temperature", "seed")}}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]


def mixed_items(pairs: list[dict], replay: list[dict], per_pair: float, seed: int) -> list[dict]:
    """The pairs plus round(len(pairs) * per_pair) replay rows drawn without replacement (seeded)."""
    k = min(len(replay), int(round(len(pairs) * float(per_pair))))
    pick = sorted(random.Random("replay/%d" % int(seed)).sample(range(len(replay)), k)) if k else []
    return list(pairs) + [replay[i] for i in pick]


def train(cfg: dict, log: Callable[[str], None] = print) -> dict:
    import torch
    from transformers import AutoTokenizer

    p = dict(PREF_DEFAULTS, **(cfg.get("pref") or {}))
    for k in ("beta", "alpha", "replay_weight", "replay_per_pair", "epochs", "lr", "warmup_ratio", "weight_decay", "eps",
              "max_grad_norm", "lora_dropout", "temperature"):
        p[k] = float(p[k])
    for k in ("items_per_step", "seed", "save_every", "dev_every", "keep_adapter_every"):
        p[k] = int(p[k])
    p["betas"] = [float(b) for b in p["betas"]]
    cfg["pref"] = p
    run_dir = cfg["run_dir"]
    for key, val in (("run_dir", run_dir), ("model.path", cfg["model"]["path"]), ("pref.pairs_file", p["pairs_file"]),
                     ("policy.init_adapter", cfg["policy"].get("init_adapter"))):
        if not val:
            raise SystemExit("%s must be set" % key)
    cfg["grpo"]["kl_reference"] = "init"                   # the reference: a frozen copy of the initial adapter
    cfg["checkpoint"] = {"keep_last": 2, "keep_adapter_every": int(p["keep_adapter_every"]), "keep_adapter_steps": []}
    dist = TR.Dist(int(cfg["dist"]["timeout_min"]))
    wait_min = float(cfg.get("disk_full_wait_min") or 0)
    if dist.rank != 0:
        log = lambda msg: None                             # noqa: E731
    try:
        if dist.rank == 0:
            retry_when_full(lambda: os.makedirs(run_dir, exist_ok=True), run_dir, wait_min, log)
        dist.barrier()
        pairs = read_rows(p["pairs_file"])
        replay = read_rows(p["replay_file"]) if p["replay_file"] else []
        dev_pairs = read_rows(p["dev_pairs_file"]) if p["dev_pairs_file"] else []
        dev_rows = read_rows(p["dev_file"]) if p["dev_file"] else []
        h = hashlib.sha1(open(p["pairs_file"], "rb").read())
        if p["replay_file"]:
            h.update(open(p["replay_file"], "rb").read())
        fp = pref_fingerprint(cfg, h.hexdigest())
        items = mixed_items(pairs, replay, p["replay_per_pair"], p["seed"])
        order = row_order(len(items), p["epochs"], p["seed"])
        per = int(p["items_per_step"])
        steps = int(math.ceil(len(order) / per))
        tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], trust_remote_code=cfg["model"]["trust_remote_code"])
        if dist.rank == 0:
            longest, n_tok, kinds = 0, 0, {}
            for k, r in enumerate(items + dev_pairs + dev_rows):
                try:
                    smp = item_sample(r, tok)
                except Exception as e:                       # noqa: BLE001
                    raise SystemExit("item %d (training items, then dev pairs, then dev rows) cannot be trained on: %s" % (k, e))
                longest = max(longest, item_tokens(smp))
                if k < len(items):
                    n_tok += len(smp.get("chosen_ids") or smp.get("completion_ids")) + len(smp.get("rejected_ids") or [])
                    kinds[smp["kind"]] = kinds.get(smp["kind"], 0) + 1
            log("items checked: %s; longest %d tokens, %d completion tokens in the training items" % (kinds, longest, n_tok))
        random.seed(int(p["seed"]) + dist.rank)
        torch.manual_seed(int(p["seed"]) + dist.rank)
        policy = TR.Policy(cfg, dist.device)
        if policy.ref_mode != "init":
            raise SystemExit("the reference must be the initial adapter (got %s)" % policy.ref_mode)
        n_drop = set_lora_dropout(policy.model, p["lora_dropout"])
        dist.broadcast_params(policy.params)
        opt = torch.optim.AdamW(policy.params, lr=float(p["lr"]), betas=tuple(p["betas"]), eps=float(p["eps"]),
                                weight_decay=float(p["weight_decay"]))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: lr_factor(n, steps, p["warmup_ratio"]))
        start = TR.latest_ckpt(run_dir)
        if start is None:
            start = 0
            TR.save_ckpt(run_dir, 0, policy, opt, sched, dist, {"fingerprint": fp}, wait_min, log)
        else:
            stt = TR.load_ckpt(run_dir, start, policy, opt, sched, dist)
            if stt.get("fingerprint") != fp and not cfg["allow_config_change"]:
                raise SystemExit("config / data fingerprint %s differs from the checkpoint's %s" % (fp, stt.get("fingerprint")))
            log("resumed from step %d" % start)
        log("pairs %d (%s; rejected %s), replay rows %d of %d, dev pairs %d, dev rows %d; %g epoch(s) = %d items in %d steps "
            "of %d; beta %g alpha %g lr %g, warm-up %d steps, LoRA dropout %g on %d modules; adapter %s (also the "
            "reference); fingerprint %s" % (
                len(pairs), " ".join("%s %d" % kv for kv in sorted(_count(pairs, "domain").items())),
                " ".join("%s %d" % kv for kv in sorted(_count(pairs, "rejected_kind").items())), len(items) - len(pairs),
                len(replay), len(dev_pairs), len(dev_rows), p["epochs"], len(order), steps, per, p["beta"], p["alpha"],
                p["lr"], int(math.ceil(steps * float(p["warmup_ratio"]))), p["lora_dropout"], n_drop,
                cfg["policy"]["init_adapter"], fp))
        if dist.rank == 0:
            resolved = json.dumps(dict(cfg, fingerprint=fp, started_at_step=start, steps=steps), indent=1, default=str)
            retry_when_full(lambda: write_text(os.path.join(run_dir, "config.resolved.json"), resolved),
                            "config.resolved.json", wait_min, log)
        dev_p = [pair_sample(r, tok) for r in dev_pairs]
        dev_r = [row_sample(r, tok) for r in dev_rows]

        def dev_scores():
            out = {}
            if dev_p:
                out.update(score_pairs(policy, dev_p, dist, p["beta"], p["temperature"]))
            if dev_r:
                out.update(score_rows(policy, dev_r, dist, p["temperature"]))
            return out

        history = {k: m for k, m in TR.read_metrics(run_dir).items() if k < start}
        if (dev_p or dev_r) and start == 0:
            d0 = dev_scores()
            log("before training: %s" % " ".join("%s %.4f" % (k, v) for k, v in d0.items()))
            if dist.rank == 0:
                retry_when_full(lambda: write_text(os.path.join(run_dir, "dev_start.json"), json.dumps(d0)), "dev_start.json", wait_min, log)
        for step in range(start, steps):
            t0 = time.time()
            batch = [items[i] for i in order[step * per:(step + 1) * per]]
            samples = [item_sample(r, tok) for r in batch]
            m = {"step": step, **train_step(policy, samples, dist, opt, p), "lr": sched.get_last_lr()[0],
                 "by_domain": _count(batch, "domain"), "by_rejected": _count([b for b in batch if is_pair(b)], "rejected_kind")}
            sched.step()
            last = step + 1 == steps
            if (dev_p or dev_r) and (last or (int(p["dev_every"]) and (step + 1) % int(p["dev_every"]) == 0)):
                m.update(dev_scores())
            m["t_train"] = round(time.time() - t0, 1)
            m["time"] = time.strftime("%F %T")
            history[step] = m
            if last or (step + 1) % int(p["save_every"]) == 0:
                TR.save_ckpt(run_dir, step + 1, policy, opt, sched, dist, {"fingerprint": fp}, wait_min, log)
                if dist.rank == 0:
                    retry_when_full(lambda: TR.prune_ckpts(run_dir, cfg, step + 1), "ckpt/ (removing old checkpoints)", wait_min, log)
            if dist.rank == 0:
                retry_when_full(lambda: TR.append_metrics(run_dir, m), "metrics.jsonl", wait_min, log)
                dev = ""
                if "dev_margin" in m:
                    dev += "dev margin %.2f acc %.3f c %.2f r %.2f | " % (m["dev_margin"], m["dev_acc"], m["dev_chosen_logratio"],
                                                                         m["dev_rejected_logratio"])
                if "dev_loss" in m:
                    dev += "dev rows %.4f | " % m["dev_loss"]
                log("step %d/%d | loss %.4f margin %.2f acc %.3f (c %.2f r %.2f; refused %d at %.2f) nll_c %.4f | replay %d nll %.4f | "
                    "gn %.3f lr %.2e | %s%.0fs | longest %d tok peak %s GiB | r twice %.2g" % (
                        step + 1, steps, m["loss"], m["margin"], m["acc"], m["chosen_logratio"], m["rejected_logratio"],
                        m["refused_pairs"], m["refused_margin"], m["nll_c_per_token"], m["replay_rows"], m["replay_nll"],
                        m["grad_norm"], m["lr"], dev, m["t_train"], m["max_item_tokens"], m["peak_mem_gib"],
                        m["rejected_recompute_maxdiff"]))
        if dist.rank == 0:
            final = os.path.join(run_dir, "final")
            retry_when_full(lambda: policy.save_adapter(final), "final adapter", wait_min, log)
            done = json.dumps({"steps": steps, "pairs": len(pairs), "items": len(items), "time": time.strftime("%F %T")})
            retry_when_full(lambda: write_text(os.path.join(run_dir, "DONE"), done), "DONE", wait_min, log)
            log("PREF_ROWS_DONE %d steps; final adapter %s" % (steps, final))
        dist.barrier()
        return history
    except BaseException:
        if dist.world > 1:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        dist.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value (YAML values)")
    a = ap.parse_args(argv)
    train(TR.load_config(a.config, a.overrides))
    return 0


if __name__ == "__main__":
    sys.exit(main())

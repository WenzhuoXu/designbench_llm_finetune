"""Measure the trainer's long-sequence path on a run's own samples (one GPU, nothing is written
to the run): peak GPU memory of the reference pass, the policy pass and the backward pass on the
longest sample of a step, on synthetic longer copies of it (the length ceiling), and how close the
segmented layers' gradients are to the whole-layer ones on a sample both paths can fit.

  python scripts/rl/long_sample_probe.py --config CFG --run-dir RUN --step 43 \
      [--lengths 70000,85000,100000] [--compare-max 14000] [--out probe.json] [key=value ...]

The policy is built as the trainer builds it (adapter, reference adapter, optimizer state of the
step's checkpoint), and each sample goes through the trainer's loss (clipped ratio + KL to the
reference). Samples are taken from rollouts/step_<step>.jsonl (the uncompacted file).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_finetune.rl_dataxp import trainer as TR                      # noqa: E402
from llm_finetune.rl_dataxp.samples import build_step_samples        # noqa: E402


def one_sample(policy, s, cfg, n_tok, torch, tensors=None):
    """The trainer's per-sample computation (train_on_samples, first iteration), with the peak GPU
    memory of each phase. Leaves the gradient in the parameters' .grad."""
    g, T = cfg["grpo"], float(cfg["rollout"]["temperature"])
    dev = policy.device
    out = {"tokens": len(s["prompt_ids"]) + len(s["completion_ids"]), "completion": len(s["completion_ids"]),
           "base_gib": torch.cuda.memory_allocated(dev) / 2 ** 30}

    def phase(name, t0):
        torch.cuda.synchronize(dev)
        out["peak_%s_gib" % name] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
        out["t_%s_s" % name] = round(time.time() - t0, 1)
        torch.cuda.reset_peak_memory_stats(dev)
        return time.time()

    torch.cuda.reset_peak_memory_stats(dev)
    t = time.time()
    ref = policy.ref_logprobs(s["prompt_ids"], s["completion_ids"], T) if g["kl_coef"] else None
    t = phase("ref", t)
    lp, _ = policy.logprobs(s["prompt_ids"], s["completion_ids"], T, raw=True)
    t = phase("forward", t)
    old = lp.detach()
    ratio = torch.exp(lp - old)
    A = float(s["advantage"])
    loss_tok = torch.maximum(-A * ratio, -A * torch.clamp(ratio, 1 - g["clip_eps_low"], 1 + g["clip_eps_high"]))
    if ref is not None:
        d = ref - lp
        loss_tok = loss_tok + g["kl_coef"] * (torch.exp(d) - d - 1)
    (float(s.get("weight", 1.0)) * loss_tok.sum() / n_tok).backward()
    phase("backward", t)
    out["peak_gib"] = max(out["peak_ref_gib"], out["peak_forward_gib"], out["peak_backward_gib"])
    out["reserved_gib"] = round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 2)
    out["base_gib"] = round(out["base_gib"], 2)
    out["cpu_rss_gib"] = round(__import__("resource").getrusage(0).ru_maxrss / 2 ** 20, 1)
    if tensors is not None:
        tensors["lp"] = lp.detach().double().cpu()
        tensors["ref"] = ref.double().cpu() if ref is not None else None
    return out


def flat_grad(policy, torch):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu()
                      for p in policy.params])


def zero(policy, keep: bool):
    for p in policy.params:
        if p.grad is not None:
            if keep:
                p.grad.zero_()
            else:
                p.grad = None


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--lengths", default="", help="synthetic total lengths (tokens), tried in order until one fails")
    ap.add_argument("--compare-max", type=int, default=14000,
                    help="old-vs-new gradients on the longest sample of at most this many tokens (0 = skip)")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = TR.load_config(a.config, a.overrides)
    cfg["run_dir"] = a.run_dir
    g, r = cfg["grpo"], cfg["rollout"]
    dev = torch.device("cuda", 0)
    torch.manual_seed(int(cfg["seed"]))
    policy = TR.Policy(cfg, dev)
    res = {"long_path": policy.long_path, "layer_segment": g.get("layer_segment"), "step": a.step,
           "gpu_total_gib": round(torch.cuda.get_device_properties(dev).total_memory / 2 ** 30, 2)}
    print("long-sequence path:", policy.long_path, flush=True)
    opt = torch.optim.AdamW(policy.params, lr=float(g["lr"]), betas=tuple(g["betas"]), eps=float(g["eps"]),
                            weight_decay=float(g["weight_decay"]))
    d = TR.ckpt_dir(a.run_dir, a.step)
    policy.load_adapter_weights(os.path.join(d, "adapter"))
    opt.load_state_dict(torch.load(os.path.join(d, "optimizer.pt"), map_location=dev, weights_only=False))
    recs = [json.loads(line) for line in open(os.path.join(a.run_dir, "rollouts", "step_%05d.jsonl" % a.step))
            if line.strip()]
    samples, _ = build_step_samples(recs, int(r["horizon"]), float(cfg["reward"]["speed_bonus"]), g["scale_rewards"],
                                    g.get("max_turns_per_episode"), int(cfg["seed"]), a.step)
    n_tok = sum(s.get("weight", 1.0) * len(s["completion_ids"]) for s in samples)
    size = lambda s: len(s["prompt_ids"]) + len(s["completion_ids"])      # noqa: E731
    samples.sort(key=lambda s: -size(s))
    print("%d records, %d samples, longest %s" % (len(recs), len(samples), [size(s) for s in samples[:6]]), flush=True)
    res["sample_lengths"] = [size(s) for s in samples]

    def save():
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)

    # 1. the longest sample, as the trainer meets it (first on its rank: no gradients yet), then again
    #    with the gradient buffers of earlier samples in place
    longest = samples[0]
    res["longest"] = {"key": longest["key"], "turn": longest.get("turn")}
    for label, keep in (("first", False), ("with_grads", True)):
        m = one_sample(policy, longest, cfg, n_tok, torch)
        res["longest"][label] = m
        print("longest (%s): %s" % (label, json.dumps(m)), flush=True)
        zero(policy, keep=True)
        save()

    # 2. old (whole layer, CPU-offloaded) vs new (segmented) gradients on a sample both fit
    st = policy._ckpt_state
    cand = [s for s in samples if (st["offload_min_tokens"] or 0) <= size(s) <= a.compare_max]
    if a.compare_max:
        if cand:
            s = cand[0]
        else:                                   # none in range: the longest sample cut to the limit
            s = dict(longest, completion_ids=list(longest["completion_ids"])[:a.compare_max - len(longest["prompt_ids"])],
                     vllm_logprobs=None)
        runs, outs, cmp = {}, {}, {"key": s["key"], "turn": s.get("turn"), "tokens": size(s), "cut": not cand}
        seg0 = st["layer_segment"]
        for name, seg in (("new", seg0), ("new_again", seg0), ("old", None), ("seg1024", 1024)):
            st["layer_segment"] = seg
            zero(policy, keep=True)
            outs[name] = {}
            m = one_sample(policy, s, cfg, n_tok, torch, outs[name])
            runs[name] = flat_grad(policy, torch)
            cmp[name] = m
            print("compare %s: %s" % (name, json.dumps(m)), flush=True)
        st["layer_segment"] = seg0
        zero(policy, keep=True)
        ref = runs["old"].double()
        for name in ("new", "new_again", "seg1024"):
            x = runs[name].double()
            cmp["%s_vs_old" % name] = {
                "rel_l2": float((x - ref).norm() / ref.norm()), "cos": float((x @ ref) / (x.norm() * ref.norm())),
                "max_abs": float((x - ref).abs().max()), "old_max_abs": float(ref.abs().max()),
                "old_norm": float(ref.norm())}
        x, y = runs["new"].double(), runs["new_again"].double()
        cmp["new_vs_new_again"] = {"rel_l2": float((x - y).norm() / x.norm()), "max_abs": float((x - y).abs().max())}
        for name in ("new", "new_again", "seg1024"):        # the log-probs themselves (policy and reference pass)
            for k in ("lp", "ref"):
                u, v = outs[name][k], outs["old"][k]
                if u is not None:
                    cmp["%s_vs_old" % name]["%s_max_abs" % k] = float((u - v).abs().max())
                    cmp["%s_vs_old" % name]["%s_mean_abs" % k] = float((u - v).abs().mean())
        cmp["old_lp_mean"] = float(outs["old"]["lp"].mean())
        res["compare"] = cmp
        print("compare:", json.dumps({k: v for k, v in cmp.items() if "_vs_" in k}), flush=True)
        del runs, outs, ref, x, y
        save()

    # 3. synthetic longer copies of the longest sample: the completion repeated up to the length
    res["synthetic"] = []
    for n in [int(x) for x in a.lengths.split(",") if x.strip()]:
        comp = list(longest["completion_ids"])
        need = n - len(longest["prompt_ids"])
        comp = (comp * (need // len(comp) + 1))[:need]
        s = dict(longest, completion_ids=comp, vllm_logprobs=None)
        gc.collect()
        torch.cuda.empty_cache()
        zero(policy, keep=True)
        try:
            m = one_sample(policy, s, cfg, n_tok, torch)
        except torch.OutOfMemoryError as e:
            m = {"tokens": n, "oom": str(e)[:300]}
            res["synthetic"].append(m)
            print("synthetic %d: OUT OF MEMORY %s" % (n, m["oom"]), flush=True)
            save()
            break
        res["synthetic"].append(m)
        print("synthetic %d: %s" % (n, json.dumps(m)), flush=True)
        save()
    save()
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

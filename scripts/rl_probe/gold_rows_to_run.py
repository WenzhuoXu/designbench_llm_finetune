"""SFT rows as a run directory that llm_finetune/rl_dataxp/grad_probe.py can read: the reference for
the gradient probe. The probe scores a labelling of the policy's own replies by how much of one
problem's gradient the other problems share; this puts a signal that is known to have trained the
model (the cross-entropy on its SFT rows) through the same instrument.

Rows: one per problem instance (the instance's row is picked by a hash of its id, the instances
evenly spaced over the sorted ids), --per-family of them for every family. Each row becomes one
record with one model call, tokenised as the assistant_after_think target trains it: the prompt is
the served prompt of the history (it ends in '<think>\\n'), the completion is
reasoning + '\\n</think>\\n\\n' + content + '<|im_end|>'.

Writes <run-dir>/rollouts/step_00000.jsonl and <run-dir>/ckpt/step_00000/adapter, which is
  --adapter DIR   a link to that adapter (the policy the rows are scored under), or
  --fresh-lora    the adapter of --like with every lora_A drawn anew (PEFT's default, uniform
                  +-1/sqrt(fan_in)) and every lora_B zero: the base model, as SFT starts from it.

    python scripts/rl_probe/gold_rows_to_run.py --rows train.jsonl --tokenizer <model snapshot> \\
        --per-family 64 --run-dir <dir> (--adapter <adapter dir> | --fresh-lora --like <adapter dir>)
"""
import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import sys

END = "<|im_end|>"


def pick_rows(index: dict, per_family: int) -> list[tuple[str, str, int]]:
    """index: family -> instance -> list of row ids (in file order). Returns (family, instance,
    row id), per_family instances of every family (all of them when it has fewer), evenly spaced
    over the sorted instance ids; an instance's row is chosen by a hash of its id."""
    out = []
    for fam in sorted(index):
        inst = sorted(index[fam])
        n = min(per_family, len(inst))
        for j in range(n):
            name = inst[j * len(inst) // n]
            rows = index[fam][name]
            h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
            out.append((fam, name, rows[h % len(rows)]))
    return out


def completion_text(message: dict) -> str:
    """What the chat template renders after the opened think block for this assistant turn."""
    reasoning = (message.get("reasoning_content") or "").strip()
    content = (message.get("content") or "").strip()
    return reasoning + "\n</think>\n\n" + content + END


def row_record(row: dict, tok, n: int) -> dict:
    msgs = row["messages"]
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError("row %s turn %s does not end in an assistant turn" % (row.get("instance"), row.get("turn")))
    history = [{k: v for k, v in m.items() if k != "supervise"} for m in msgs[:-1]]
    prompt = tok.apply_chat_template(history, tokenize=True, add_generation_prompt=True)
    prompt = list(prompt if isinstance(prompt, list) else prompt["input_ids"])
    if not tok.decode(prompt).endswith("<think>\n"):
        raise ValueError("the served prompt does not end with '<think>\\n': %r" % tok.decode(prompt)[-40:])
    text = completion_text(msgs[-1])
    completion = list(tok(text, add_special_tokens=False).input_ids)
    if tok.decode(completion) != text:
        raise ValueError("the completion tokens do not reproduce the text (%s)" % row.get("instance"))
    key = "gold/%s/%04d_t%s_r%s" % (row["instance"], n, row.get("turn"), row.get("rationale_sample"))
    return {"key": key, "group": row["instance"], "variant": row["domain"], "member": 0,
            "row": {k: row.get(k) for k in ("turn", "rationale_sample", "tier", "teacher", "batch", "own_pct", "n_candidates")},
            "episode": {"turns": []},
            "calls": [{"prompt_token_ids": prompt, "token_ids": completion, "logprobs": None}]}


def fresh_lora(like: str, dst: str, seed: int) -> None:
    """The adapter of ``like`` at PEFT's default initialisation: lora_A uniform in
    +-1/sqrt(fan_in) (kaiming_uniform with a = sqrt(5)), lora_B zero."""
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(os.path.join(like, "adapter_model.safetensors"))
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in sorted(sd):
        v = sd[k]
        if "lora_A" in k:
            bound = 1.0 / math.sqrt(v[0].numel())
            out[k] = ((torch.rand(v.shape, generator=g, dtype=torch.float32) * 2 - 1) * bound).to(v.dtype)
        elif "lora_B" in k:
            out[k] = torch.zeros_like(v)
        else:
            raise SystemExit("%s holds a tensor that is neither lora_A nor lora_B: %s" % (like, k))
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "adapter_model.safetensors.tmp"), metadata={"format": "pt"})
    os.replace(os.path.join(dst, "adapter_model.safetensors.tmp"), os.path.join(dst, "adapter_model.safetensors"))
    shutil.copy(os.path.join(like, "adapter_config.json"), os.path.join(dst, "adapter_config.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--per-family", type=int, default=64)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--fresh-lora", action="store_true")
    ap.add_argument("--like", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if bool(a.adapter) == bool(a.fresh_lora):
        raise SystemExit("exactly one of --adapter and --fresh-lora")
    if a.fresh_lora and not a.like:
        raise SystemExit("--fresh-lora needs --like <adapter dir>")
    out = os.path.join(a.run_dir, "rollouts", "step_00000.jsonl")
    adir = os.path.join(a.run_dir, "ckpt", "step_00000", "adapter")
    if not os.path.exists(out):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        index = collections.defaultdict(lambda: collections.defaultdict(list))
        offsets = []
        with open(a.rows, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    r = json.loads(line)
                    index[r["domain"]][r["instance"]].append(len(offsets))
                    offsets.append(pos)
        picked = pick_rows(index, a.per_family)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        counts = collections.Counter()
        with open(a.rows, "rb") as f, open(out + ".tmp", "w", encoding="utf-8") as fo:
            for n, (fam, name, rid) in enumerate(picked):
                f.seek(offsets[rid])
                rec = row_record(json.loads(f.readline()), tok, n)
                fo.write(json.dumps(rec) + "\n")
                counts[fam] += 1
        os.replace(out + ".tmp", out)
        print("rows %d of %d (%s) -> %s" % (len(picked), len(offsets), " ".join("%s %d" % kv for kv in sorted(counts.items())), out),
              flush=True)
    else:
        print("already there:", out, flush=True)
    if not os.path.exists(os.path.join(adir, "adapter_model.safetensors")):
        if a.fresh_lora:
            fresh_lora(a.like, adir, a.seed)
        else:
            os.makedirs(os.path.dirname(adir), exist_ok=True)
            if os.path.islink(adir):
                os.unlink(adir)
            os.symlink(os.path.abspath(a.adapter), adir)
        if not os.path.exists(os.path.join(adir, "adapter_model.safetensors")):
            raise SystemExit("no adapter at %s" % adir)
    print("adapter:", adir, "(fresh LoRA, seed %d)" % a.seed if a.fresh_lora else "-> %s" % a.adapter, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

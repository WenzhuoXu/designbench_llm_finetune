#!/usr/bin/env python
"""Does the distillation corpus tokenize and mask the way SFT needs it to?

Three things have to hold before a training run is worth launching:
  - every example fits max_seq_len (a truncated trajectory teaches the model to
    stop before the design is feasible),
  - loss lands on assistant spans only,
  - the supervised text is the tool call, not the rendered state.
This prints the evidence for each rather than asserting it.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/llm_finetune")
from transformers import AutoTokenizer
from llm_finetune.data.processors.chat_formatter import ChatFormatter
from llm_finetune.data.processors.sft_jsonl_processor import SFTJsonlProcessor
from llm_finetune.training.sft.targets import TARGET_REGISTRY

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--target", default="assistant_only")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir="/ocean/projects/mch250030p/wxu7/hf_models")
    fmt = ChatFormatter.from_model_id(a.model, tok)
    target = TARGET_REGISTRY[a.target]()
    proc = SFTJsonlProcessor(tok, fmt, target, max_seq_len=a.max_seq_len, min_trace_quality=0.0)

    rows = [json.loads(l) for l in open(a.jsonl)][: a.n]
    tmp = Path("/tmp/_cc_sample.jsonl")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = proc.process_jsonl(tmp)

    # The processor only sets labels; the target's mask is applied downstream in
    # the collator, so apply it here or this measures nothing.
    import torch

    masks = []
    for e in out:
        ids = torch.as_tensor(e["input_ids"])
        lab = torch.as_tensor(e["labels"])
        masks.append(target.get_loss_mask(ids, lab, tok).tolist())

    lens = sorted(len(e["input_ids"]) for e in out)
    sup = [sum(m) for m in masks]
    n = len(lens)
    print("examples in %d -> processed %d  (dropped %d)" % (len(rows), n, len(rows) - n))
    if not n:
        sys.exit(1)
    print("sequence length: min %d  median %d  p95 %d  max %d  (limit %d)"
          % (lens[0], lens[n // 2], lens[int(n * 0.95) - 1], lens[-1], a.max_seq_len))
    print("over limit: %d of %d" % (sum(1 for x in lens if x >= a.max_seq_len), n))
    frac = sum(sup) / sum(lens)
    print("supervised tokens: %d of %d (%.1f%% of the corpus is loss-bearing)"
          % (sum(sup), sum(lens), 100 * frac))

    for e, m in list(zip(out, masks))[: a.show]:
        keep = [int(i) for i, k in zip(e["input_ids"], m) if k]
        print("\n--- supervised text of one example (%s) ---" % e.get("problem_id", "?"))
        print(tok.decode(keep)[:1200])

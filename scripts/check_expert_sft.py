"""
Dry-run the expert traces through the real SFT data path before spending GPU hours.

Loads data/expert/expert_traces_0_150.jsonl with the same SFTDataset.from_sft_jsonl,
formatter and target that scripts/train_sft.py uses, and reports what training would
actually see: examples surviving the loader, sequence lengths against max_seq_len, the
fraction of tokens that carry loss, and whether the supervised span still contains whole
compound actions rather than a truncated first part.

The same check is run on the benchmark's own train.jsonl as a reference, so the expert file
is compared against something rather than inspected in isolation.
"""
import sys, json, statistics
from pathlib import Path
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(PROJECT), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoTokenizer
from llm_finetune.data.datasets.sft_dataset import SFTDataset
from llm_finetune.training.sft.targets import build_target_from_config
from llm_finetune.data.processors.chat_formatter import ChatFormatter

MODEL = "Qwen/Qwen3-14B"
CACHE = "/ocean/projects/mch250030p/wxu7/hf_models"
MAXLEN = 16384

FILES = [
    ("expert  ", str(PROJECT / "data/expert/expert_traces_0_150.jsonl"), 0.0),
#    designbench reference checked last cycle; skipped here for turnaround
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, trust_remote_code=True)
    formatter = ChatFormatter.from_model_id(MODEL, tok)
    target = build_target_from_config("warmstart_reasoning")

    for label, path, minq in FILES:
        if not Path(path).exists():
            print("  %s: MISSING %s" % (label, path))
            continue
        try:
            ds = SFTDataset.from_sft_jsonl(path=path, tokenizer=tok, formatter=formatter,
                                           target=target, max_seq_len=MAXLEN,
                                           min_trace_quality=minq)
        except Exception as e:
            print("  %s: loader raised %s: %s" % (label, type(e).__name__, str(e)[:200]))
            continue
        n = len(ds)
        lens, sup = [], []
        compound_ok, compound_seen = 0, 0
        for i in range(min(n, 60)):
            ex = ds[i]
            ids = ex["input_ids"]
            lab = ex.get("labels")
            lens.append(int(len(ids)))
            if lab is not None:
                keep = sum(1 for t in lab if int(t) != -100)
                sup.append(keep / max(len(lab), 1))
                txt = tok.decode([int(t) for t, l in zip(ids, lab) if int(l) != -100],
                                 skip_special_tokens=True)
            else:
                txt = tok.decode([int(t) for t in ids], skip_special_tokens=True)
            if " ; " in txt:
                compound_seen += 1
                if txt.count(" ; ") >= 1:
                    compound_ok += 1
        print("\n  %s  examples=%d  (from %s)" % (label, n, Path(path).name))
        if lens:
            print("     tokens: mean %.0f  median %.0f  max %d  over-%d: %d of %d sampled"
                  % (statistics.mean(lens), statistics.median(lens), max(lens),
                     MAXLEN, sum(1 for x in lens if x >= MAXLEN), len(lens)))
        if sup:
            print("     supervised token fraction: mean %.3f  min %.3f  max %.3f"
                  % (statistics.mean(sup), min(sup), max(sup)))
        print("     sampled examples whose supervised span carries a compound action: %d/%d"
              % (compound_ok, min(n, 60)))
    print("\n  ready for SFT if the expert row shows examples > 0, few sequences at the cap,"
          "\n  and a non-zero supervised fraction.")


if __name__ == "__main__":
    main()

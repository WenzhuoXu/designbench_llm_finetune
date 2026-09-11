"""Which module names should LoRA attach to on qwen3_5, and where do they live?

The repo's default list (q/k/v/o_proj, gate/up/down_proj) covers 72% of Linear
parameters here: the 48 linear-attention layers use in_proj_qkv / in_proj_z /
in_proj_a / in_proj_b / out_proj instead, and those are exactly the layers a
naive target list would leave frozen. Some leaf names (qkv, proj, linear_fc1,
linear_fc2) appear ~27 times, which is the vision tower, not the text stack --
this is a multimodal checkpoint and the corpus is text only. Print full paths so
the target list can be anchored to the language model and nothing else.
"""
import sys
from collections import Counter, defaultdict

import torch
import transformers
from transformers import AutoConfig

MID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
CACHE = "/ocean/projects/mch250030p/wxu7/hf_models"

cfg = AutoConfig.from_pretrained(MID, cache_dir=CACHE)
cls = getattr(transformers, (cfg.architectures or [None])[0])
with torch.device("meta"):
    model = cls(cfg)

total = sum(p.numel() for p in model.parameters())
by_top = defaultdict(int)
examples = {}
leaf_params = Counter()
leaf_top = defaultdict(set)

for name, mod in model.named_modules():
    if not isinstance(mod, torch.nn.Linear):
        continue
    leaf = name.split(".")[-1]
    n = sum(p.numel() for p in mod.parameters(recurse=False))
    leaf_params[leaf] += n
    top = name.split(".")[0]
    by_top[top] += n
    leaf_top[leaf].add(top)
    examples.setdefault(leaf, name)

print("model parameters: %.2fB" % (total / 1e9))
print("\nLinear params by top-level subtree:")
for t, n in sorted(by_top.items(), key=lambda kv: -kv[1]):
    print("  %-24s %8.1fM  %5.1f%%" % (t, n / 1e6, 100.0 * n / sum(by_top.values())))

print("\nleaf name -> example full path (subtrees it appears in)")
for leaf, n in leaf_params.most_common():
    print("  %-14s %8.1fM  %-58s %s"
          % (leaf, n / 1e6, examples[leaf], sorted(leaf_top[leaf])))

# Build the recommended list: every Linear leaf inside the language model,
# minus the output head (LoRA on a 248k-row head is a large adapter for little
# return, and PEFT handles the head separately via modules_to_save).
LM_SUBTREES = {t for t in by_top if "visual" not in t and "vision" not in t}
rec = sorted({leaf for leaf in leaf_params
              if leaf_top[leaf] & LM_SUBTREES and leaf != "lm_head"})
covered = sum(leaf_params[l] for l in rec)
print("\nlanguage-model subtrees: %s" % sorted(LM_SUBTREES))
print("recommended target_modules (%d): %s" % (len(rec), rec))
print("covers %.1f%% of Linear params (%.1f%% excluding lm_head)"
      % (100.0 * covered / sum(leaf_params.values()),
         100.0 * covered / (sum(leaf_params.values()) - leaf_params.get("lm_head", 0))))

import os, sys, math
from pathlib import Path
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(P), str(P / "scripts"), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
import api_grammar_2x2 as H

tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or open(os.path.expanduser("~/.bedrock_token")).read().strip()
reg = os.environ.get("AWS_REGION", "us-west-2")
CANDS = [
    "us.meta.llama3-3-70b-instruct-v1:0",
    "us.meta.llama4-scout-17b-16e-instruct-v1:0",
    "us.meta.llama4-maverick-17b-128e-instruct-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-premier-v1:0",
    "us.deepseek.r1-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "us.mistral.pixtral-large-2502-v1:0",
    "cohere.command-r-plus-v1:0",
    "us.writer.palmyra-x5-v1:0",
    "openai.gpt-oss-120b-1:0",
    "us.openai.gpt-oss-120b-1:0",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "us.qwen.qwen3-32b-v1:0",
]
msg = [{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}]
live = []
for m in CANDS:
    try:
        r = H.call(msg, "", m, reg, tok, max_tokens=16, temperature=0.0, retries=1)
        st = ("OK  -> " + repr(r[:24])) if r else "no response"
        if r:
            live.append(m)
    except Exception as e:
        st = "ERR " + str(e)[:50]
    print("%-50s %s" % (m, st), flush=True)
print("\nLIVE NON-ANTHROPIC:", len(live))
for m in live:
    print("   ", m)

# --- verify the Haiku TOST that a reviewer says does not reproduce ---
print("\n=== TOST check, margin |d| < 0.10 ===")
from math import erf, sqrt
def norm_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))
for name, d, se in (("Sonnet 4.5", -0.010, 0.026), ("Sonnet 4", -0.011, 0.035),
                    ("Opus 4.1", 0.033, 0.028), ("Haiku 4.5", -0.121, 0.038)):
    # correct TOST: both one-sided tests must reject
    p_lower = 1 - norm_cdf((d + 0.10) / se)     # H0: d <= -0.10
    p_upper = norm_cdf((d - 0.10) / se)         # H0: d >= +0.10
    p = max(p_lower, p_upper)
    print("  %-11s d=%+.3f se=%.3f  p_lower=%.4f p_upper=%.4f  TOST p=%.4f  %s"
          % (name, d, se, p_lower, p_upper, p, "EQUIVALENT" if p < 0.05 else "not equivalent"))

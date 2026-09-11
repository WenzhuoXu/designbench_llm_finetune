import os, sys, json
from pathlib import Path
P = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for p in (str(P), str(P / "scripts"), "/ocean/projects/mch250030p/wxu7/DesignBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
import api_grammar_2x2 as H

tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or open(os.path.expanduser("~/.bedrock_token")).read().strip()
reg = os.environ.get("AWS_REGION", "us-west-2")
CANDS = [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
]
msg = [{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}]
for m in CANDS:
    try:
        r = H.call(msg, "", m, reg, tok, max_tokens=16, temperature=0.0, retries=1)
        print("%-52s %s" % (m, ("OK  -> " + repr(r[:30])) if r else "NO RESPONSE"))
    except Exception as e:
        print("%-52s ERR %s" % (m, str(e)[:60]))

"""Thinking diagnostic, client (con1-2m-rows, 11-27; the parent session's request): held-out-family thermal2 questions
on PLACE_BLOCK and MOVE_SINK, zero-shot, the facetm question as in the existing runs, with room to think first.

    python think_client.py --queries queries_v2.jsonl --tokenizer MODEL --base-url URL --model base|LAB --label L \\
        --out preds_L.jsonl [--budget 1000] [--min-think 0] [--answer-tokens 1024] [--workers 48]

Prompt: con1's skill_client.messages_for(q, j, False, 'facetm') -> conseq_format.prompt_ids (imported unchanged; ends
'<|im_start|>assistant\\n<think>\\n', the layout con1's answers were trained and asked in). Temperature 0, in steps:
  1. think: up to --budget tokens, stopping at '</think>' or <|im_end|>. With --min-think K the first K tokens are
     generated with '</think>', <|im_end|> and <|endoftext|> banned (logit bias -100), then the rest of the budget as
     above. think_end: 'closed' (the model wrote </think>), 'budget' (cut at the budget), 'im_end' (ended its turn
     inside the think block).
  2. answer: the prompt + think.strip() + '\\n</think>\\n\\n' (the trained answer layout; a think block cut by the
     budget, or ended by <|im_end|>, is closed the same way), up to --answer-tokens, stop at <|im_end|>.
One line per question in the usual preds schema (sid, variant, predictor, pass='zero', qi, target, move, text = the
answer after </think>, finish) plus think (raw text), think_tokens, think_end, answer_tokens, prompt_tokens.
Resumable by (sid, qi).
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CON1 = "/storage/wxu/con1/repo/scripts/con1"
sys.path.insert(0, CON1)
import skill_client as SC  # noqa: E402  (con1's client: messages_for, prompt_ids; read only)
import dirlayer as DL  # noqa: E402

VERBS = ("PLACE_BLOCK", "MOVE_SINK")
LOCK = threading.Lock()


def call(base, model, ids, max_tokens, stop_ids, bias=None):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stop_token_ids": stop_ids,
            "skip_special_tokens": False}
    if bias:
        body["logit_bias"] = bias
    req = urllib.request.Request(base.rstrip("/") + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.load(r)
            ch = j["choices"][0]
            return (ch.get("text") or "", ch.get("finish_reason"), ch.get("stop_reason"),
                    (j.get("usage") or {}).get("completion_tokens") or 0)
        except Exception as e:                             # noqa: BLE001
            if attempt == 4:
                raise RuntimeError(str(e)[:300])
            time.sleep(5 * (attempt + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--min-think", type=int, default=0)
    ap.add_argument("--answer-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    IM_END, THINK_END = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</think>")
    EOT = tok.convert_tokens_to_ids("<|endoftext|>")
    assert tok.encode("</think>", add_special_tokens=False) == [THINK_END], "'</think>' is not one token"
    ban = {str(t): -100 for t in (IM_END, THINK_END, EOT)}
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    qs = [q for q in qs if q["group"] == "heldout_family" and q["variant"] == "thermal2"]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
                if not str(r.get("finish", "")).startswith("error"):
                    done.add((r["sid"], r["qi"]))
            except ValueError:
                pass
    jobs = [(q, j) for q in qs for j, m in enumerate(q["moves"]) if DL.verb_of(m) in VERBS and (q["sid"], j) not in done]
    print("%s: %d questions (%d done), budget %d, min think %d" % (a.label, len(jobs), len(done), a.budget, a.min_think),
          flush=True)

    def one(job):
        q, j = job
        ids = SC.prompt_ids(tok, SC.messages_for(q, j, False, "facetm"))
        rec = {"sid": q["sid"], "variant": q["variant"], "predictor": a.label, "pass": "zero", "qi": j, "target": "facetm",
               "move": q["moves"][j]["move"], "prompt_tokens": len(ids)}
        try:
            think, n, end = "", 0, None
            if a.min_think > 0:
                t, fin, sr, k = call(a.base_url, a.model, ids, a.min_think, [], ban)
                think, n = t, k
            if n < a.budget:
                cur = ids + tok.encode(think, add_special_tokens=False) if think else ids
                t, fin, sr, k = call(a.base_url, a.model, cur, a.budget - n, [IM_END, THINK_END])
                think, n = think + t, n + k
                end = "budget" if fin == "length" else "closed" if sr == THINK_END else "im_end" if sr == IM_END else str(sr)
            else:
                end = "budget"
            think = think.replace("</think>", "").replace("<|im_end|>", "")
            aids = ids + tok.encode(think.strip() + "\n</think>\n\n", add_special_tokens=False)
            ans, fin, sr, k = call(a.base_url, a.model, aids, a.answer_tokens, [IM_END])
            rec.update(text=ans.replace("<|im_end|>", "").strip(), finish=fin, think=think, think_tokens=n,
                       think_end=end, answer_tokens=k)
        except RuntimeError as e:
            rec.update(text=None, finish="error: %s" % e)
        return rec

    t0, n = time.time(), 0
    with open(a.out, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(one, jb) for jb in jobs]):
            with LOCK:
                fo.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                fo.flush()
            n += 1
            if n % 50 == 0:
                print("%d/%d in %.0f s" % (n, len(jobs), time.time() - t0), flush=True)
    print("THINK_CLIENT_DONE %s %d in %.0f s" % (a.label, n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

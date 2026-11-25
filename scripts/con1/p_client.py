"""con1 P-form skill test, step 1 (GPU): the policy prompt of each test state, the model's own reply (thinking on,
T=0). The reply is expected to end with an <action> line and, for a P-trained model, an 'EXPECTED RESULT:' block.

    python p_client.py --queries queries_v2.jsonl --tokenizer <model dir> --base-url URL [URL ...] --model LABEL \\
        --out preds_P_LABEL.jsonl [--groups heldout_problem] [--variants ...] [--max-tokens 8192] [--workers 64]
One line per state: {sid, variant, predictor, reasoning, answer, finish, n_tokens}. Resumable by sid. Served through
/v1/chat/completions with the state's own (system, user) messages, exactly as the benchmark harness serves a turn.
"""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCK = threading.Lock()


def chat(url, model, system, user, max_tokens):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.load(r)
            ch = j["choices"][0]
            msg = ch["message"]
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if "</think>" in text:
                reasoning, text = text.rsplit("</think>", 1)
            return reasoning.replace("<think>", "").strip(), text, ch.get("finish_reason"), \
                (j.get("usage") or {}).get("completion_tokens")
        except Exception as e:                             # noqa: BLE001
            if attempt == 4:
                return None, None, "error: %s" % str(e)[:200], None
            time.sleep(5 * (attempt + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--base-url", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="*", default=["heldout_problem"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--tokenizer", default=None, help="unused (kept for a common launcher)")
    a = ap.parse_args(argv)
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    qs = [q for q in qs if (not a.groups or q["group"] in a.groups) and (not a.variants or q["variant"] in a.variants)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
                if not str(r.get("finish", "")).startswith("error"):
                    done.add(r["sid"])
            except ValueError:
                pass
    todo = [q for q in qs if q["sid"] not in done]
    print("%d states (%d done)" % (len(todo), len(done)), flush=True)
    label = a.label or a.model
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        def one(q):
            url = a.base_url[int(hashlib.sha1(q["sid"].encode()).hexdigest()[:8], 16) % len(a.base_url)]
            reasoning, answer, fin, ntok = chat(url, a.model, q["system"], q["user"], a.max_tokens)
            return {"sid": q["sid"], "variant": q["variant"], "predictor": label, "reasoning": reasoning,
                    "answer": answer, "finish": fin, "n_tokens": ntok}
        futs = [ex.submit(one, q) for q in todo]
        for n, fu in enumerate(as_completed(futs), 1):
            with LOCK:
                fo.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                fo.flush()
            if n % 100 == 0:
                print("%d/%d in %.0f s" % (n, len(todo), time.time() - t0), flush=True)
    print("P_CLIENT_DONE %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

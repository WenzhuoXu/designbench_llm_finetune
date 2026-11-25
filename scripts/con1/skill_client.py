"""con1 skill test, step 2: ask a served model the fixed questions (docs §3), zero-shot and sequential passes.

    python skill_client.py --queries queries.jsonl --tokenizer <model dir> --base-url http://127.0.0.1:8000/v1 \\
        --model con1 --out preds_con1.jsonl [--passes zero seq] [--max-tokens 2048] [--workers 32]

Prompts are token ids built by llm_finetune/rl_dataxp/conseq_format.py, the layout the consequence rows were trained
in (/v1/completions, temperature 0, stop at <|im_end|>). zero: each question alone, appended to the state prompt.
seq: question j asked after questions 0..j-1 with the TRUE result messages as the earlier answers (teacher forcing).
The answer is the text after the last '</think>' of the generation (none: the think block never closed). One line per
request: {sid, predictor, pass, qi, move, text, finish, n_tokens}. Resumable by (sid, pass, qi).
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from conseq_lib import FOLLOW_UP, QUESTION  # noqa: E402
from llm_finetune.rl_dataxp.conseq_format import prompt_ids  # noqa: E402

LOCK = threading.Lock()


def messages_for(q, j, seq):
    ms = q["moves"]
    if not seq or j == 0:
        return [{"role": "system", "content": q["system"]},
                {"role": "user", "content": q["user"] + QUESTION.format(move=ms[j]["move"])}]
    msgs = [{"role": "system", "content": q["system"]},
            {"role": "user", "content": q["user"] + QUESTION.format(move=ms[0]["move"])}]
    for i in range(j):
        msgs.append({"role": "assistant", "content": ms[i]["result"], "reasoning_content": ""})
        msgs.append({"role": "user", "content": FOLLOW_UP.format(move=ms[i + 1]["move"])})
    return msgs


def complete(base, model, ids, max_tokens, stop_ids):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stop_token_ids": stop_ids,
            "skip_special_tokens": False}
    req = urllib.request.Request(base.rstrip("/") + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.load(r)
            ch = j["choices"][0]
            return ch.get("text") or "", ch.get("finish_reason"), (j.get("usage") or {}).get("completion_tokens")
        except Exception as e:                             # noqa: BLE001
            if attempt == 4:
                return None, "error: %s" % str(e)[:200], None
            time.sleep(5 * (attempt + 1))


def answer_of(gen: str):
    if gen is None or "</think>" not in gen:
        return None
    return gen.rsplit("</think>", 1)[1].replace("<|im_end|>", "").strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", nargs="+", default=["zero", "seq"])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    stop = [tok.convert_tokens_to_ids("<|im_end|>")]
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    if a.variants:
        qs = [q for q in qs if q["variant"] in a.variants]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
                if not str(r.get("finish", "")).startswith("error"):
                    done.add((r["sid"], r["pass"], r["qi"]))
            except ValueError:
                pass
    jobs = []
    for q in qs:
        for p in a.passes:
            for j in range(len(q["moves"])):
                if p == "seq" and j == 0:
                    continue                                 # position 0 of the sequential pass is the zero-shot question
                if (q["sid"], p, j) not in done:
                    jobs.append((q, p, j))
    # longest prompts first spreads the load; states of one sid stay close for the prefix cache
    print("%d requests (%d done)" % (len(jobs), len(done)), flush=True)
    label = a.label or a.model
    t0 = time.time()
    n = 0
    with open(a.out, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        def one(job):
            q, p, j = job
            ids = prompt_ids(tok, messages_for(q, j, p == "seq"))
            gen, fin, ntok = complete(a.base_url, a.model, ids, a.max_tokens, stop)
            return {"sid": q["sid"], "variant": q["variant"], "predictor": label, "pass": p, "qi": j,
                    "move": q["moves"][j]["move"], "text": answer_of(gen), "raw_tail": (gen or "")[-300:],
                    "finish": fin, "n_tokens": ntok, "prompt_tokens": len(ids)}
        futs = [ex.submit(one, jb) for jb in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            with LOCK:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                fo.flush()
            n += 1
            if n % 200 == 0:
                print("%d/%d in %.0f s" % (n, len(jobs), time.time() - t0), flush=True)
    print("SKILL_CLIENT_DONE %d requests in %.0f s" % (n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

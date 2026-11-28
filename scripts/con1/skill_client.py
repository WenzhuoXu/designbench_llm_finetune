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
import hashlib
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
from targets import FORMATS, question  # noqa: E402
from llm_finetune.rl_dataxp.conseq_format import prompt_ids  # noqa: E402

LOCK = threading.Lock()


def messages_for(q, j, seq, fmt="verbatim"):
    """The prompt for question j: alone (zero-shot), or after questions 0..j-1 answered with the TRUE results in
    the asked format (sequential)."""
    from targets import fmt_for, render
    fmt = fmt_for(fmt, q["variant"])                       # facetc: facetm outside the chain families
    ms = q["moves"]
    if not seq or j == 0:
        return [{"role": "system", "content": q["system"]},
                {"role": "user", "content": q["user"] + question(fmt, ms[j]["move"], True)}]
    msgs = [{"role": "system", "content": q["system"]},
            {"role": "user", "content": q["user"] + question(fmt, ms[0]["move"], True)}]
    for i in range(j):
        msgs.append({"role": "assistant", "content": render(fmt, ms[i], q["metrics0"])[0], "reasoning_content": ""})
        msgs.append({"role": "user", "content": question(fmt, ms[i + 1]["move"], False)})
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
    ap.add_argument("--base-url", nargs="+", required=True, help="one or more servers; a state's requests stay on one")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", nargs="+", default=["zero", "seq"])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--target", choices=FORMATS, default="verbatim", help="answer format asked for (targets.py)")
    ap.add_argument("--groups", nargs="*", default=None, help="only states of these groups (heldout_problem, ...)")
    ap.add_argument("--no-defined-terms", action="store_true", help="questions without the definitions of feasible and "
                    "'same' (on by default since lib_v8): every evaluation before 11-28")
    a = ap.parse_args(argv)
    import targets
    targets.DEFINED_DEFAULT = not a.no_defined_terms
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    stop = [tok.convert_tokens_to_ids("<|im_end|>")]
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    if a.variants:
        qs = [q for q in qs if q["variant"] in a.variants]
    if a.groups:
        qs = [q for q in qs if q["group"] in a.groups]
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
    print("%d requests (%d done)" % (len(jobs), len(done)), flush=True)
    label = a.label or a.model
    t0 = time.time()
    n = 0
    with open(a.out, "ab", buffering=0) as fo, ThreadPoolExecutor(a.workers) as ex:   # unbuffered: nothing stays queued
        def one(job):
            q, p, j = job
            ids = prompt_ids(tok, messages_for(q, j, p == "seq", a.target))
            url = a.base_url[int(hashlib.sha1(q["sid"].encode()).hexdigest()[:8], 16) % len(a.base_url)]
            gen, fin, ntok = complete(url, a.model, ids, a.max_tokens, stop)
            return {"sid": q["sid"], "variant": q["variant"], "predictor": label, "pass": p, "qi": j, "target": a.target,
                    "move": q["moves"][j]["move"], "text": answer_of(gen), "raw_tail": (gen or "")[-300:],
                    "finish": fin, "n_tokens": ntok, "prompt_tokens": len(ids)}
        futs = [ex.submit(one, jb) for jb in jobs]
        torn = False
        for fu in as_completed(futs):
            r = fu.result()
            line = json.dumps(r, ensure_ascii=False) + "\n"
            # /project/community is a shared volume that runs full for minutes at a time (EDQUOT / ENOSPC): retry
            # for up to an hour; after a failed write start on a fresh line (a torn line is skipped on resume)
            for attempt in range(120):
                try:
                    with LOCK:
                        data = (("\n" if torn else "") + line).encode("utf-8")
                        if fo.write(data) != len(data):
                            raise OSError("short write")
                    torn = False
                    break
                except OSError as e:
                    torn = True
                    print("write failed (%s), retry %d" % (e, attempt + 1), file=sys.stderr, flush=True)
                    time.sleep(30)
            else:
                raise SystemExit("could not write %s for an hour" % a.out)
            n += 1
            if n % 200 == 0:
                print("%d/%d in %.0f s" % (n, len(jobs), time.time() - t0), flush=True)
    print("SKILL_CLIENT_DONE %d requests in %.0f s" % (n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

"""Step 2 of a screened supervised round: k replies of the served policy at every state.

One chat-completion request per state with n = k, the state's served prompt (system, user) and
thinking on, sampled as the RL rollouts sampled (temperature 0.8, top_p 1, top_k 0 by default) with
no completion cap. Output: one line per (state, sample) in the format dataxp.ownrows.score reads:
`answer` is the rationale (the think block) followed by the visible answer, so the reply's last
<action> line and the text before it are read as for the dx38 rows. A reply whose think block was
not closed has finish "length" and no answer.

Resumable: a state with k lines in --out is skipped. States are taken in the order of --first
(families listed there go first), then file order.

    python sample_states.py --states states.jsonl --out samples.jsonl --servers URL [URL ...] \\
        --model dx38mtR [--k 8] [--temperature 0.8] [--workers 96] [--first photonics grid2op engibench]
"""
import argparse
import collections
import hashlib
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def split_reply(msg: dict, thinking: bool = True) -> tuple[str, str, bool]:
    """(rationale, visible answer, think block closed) of one choice's message, as the harness's
    LLM client reads it."""
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    closed = True
    if "</think>" in text:
        reasoning, text = text.rsplit("</think>", 1)
    elif thinking and not reasoning:
        closed = False
    return reasoning.replace("<think>", "").strip(), text.strip(), closed


def records_of(sid: str, out: dict, k: int) -> list[dict]:
    usage = out.get("usage") or {}
    recs = []
    for ch in sorted(out.get("choices") or [], key=lambda c: c.get("index", 0)):
        reasoning, text, closed = split_reply(ch.get("message") or {})
        finish = ch.get("finish_reason")
        ok = closed and finish != "length"
        recs.append({"sid": sid, "sample": int(ch.get("index", len(recs))),
                     "answer": (reasoning + "\n\n" + text) if ok else "",
                     "finish": finish if closed else "length", "mode": "own_think", "prompt": "served",
                     "tokens": usage.get("completion_tokens") if not recs else None,
                     "prompt_tokens": usage.get("prompt_tokens") if not recs else None})
    if len(recs) != k:
        raise ValueError("%d choices for n=%d" % (len(recs), k))
    return recs


def request_body(state: dict, a) -> dict:
    seed = int(hashlib.sha1(("%s|%d" % (state["sid"], a.seed)).encode()).hexdigest()[:8], 16) % (2 ** 31)
    return {"model": a.model,
            "messages": [{"role": "system", "content": state["system"]},
                         {"role": "user", "content": state["observation"] + state["instruction"]}],
            "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "max_tokens": None,
            "n": a.k, "seed": seed, "chat_template_kwargs": {"enable_thinking": True}}


def call(url: str, body: dict, timeout: float, attempts: int = 3) -> dict:
    data = json.dumps(body).encode()
    err = None
    for attempt in range(attempts):
        req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:                               # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:300])
            time.sleep(5 * (attempt + 1))
    return {"error": err}


def done_sids(path: str, k: int) -> set:
    n = collections.Counter()
    try:
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except ValueError:
                continue                                      # the cut last line of a killed run
            if not o.get("error"):
                n[o["sid"]] += 1
    except FileNotFoundError:
        pass
    return {s for s, c in n.items() if c >= k}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--servers", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=96, help="states in flight (each asks for k replies)")
    ap.add_argument("--first", nargs="*", default=[], help="families sampled first")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    states = [json.loads(line) for line in open(a.states, encoding="utf-8") if line.strip()]
    rank = {v: i for i, v in enumerate(a.first)}
    states.sort(key=lambda s: rank.get(s["variant"], len(rank)))     # stable: file order inside a family
    done = done_sids(a.out, a.k)
    todo = [s for s in states if s["sid"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("states %d, already sampled %d, to do %d; k %d, T %.2f, top_p %.2f, top_k %d, model %s, %d servers" % (
        len(states), len(done), len(todo), a.k, a.temperature, a.top_p, a.top_k, a.model, len(a.servers)), flush=True)
    lock = threading.Lock()
    t0 = time.time()
    n_ok = n_err = n_tok = 0

    def one(j, s):
        out = call(a.servers[j % len(a.servers)], request_body(s, a), a.timeout)
        if "error" in out and "choices" not in out:
            return s, None, str(out["error"])
        try:
            return s, records_of(s["sid"], out, a.k), None
        except Exception as e:                               # noqa: BLE001
            return s, None, "%s: %s" % (type(e).__name__, str(e)[:200])

    with open(a.out, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, j, s) for j, s in enumerate(todo)]
        for fu in as_completed(futs):
            s, recs, err = fu.result()
            with lock:
                if recs is None:
                    n_err += 1
                    fo.write(json.dumps({"sid": s["sid"], "sample": -1, "error": err}) + "\n")
                else:
                    n_ok += 1
                    n_tok += recs[0].get("tokens") or 0
                    for r in recs:
                        fo.write(json.dumps(r) + "\n")
                fo.flush()
                if (n_ok + n_err) % 100 == 0:
                    dt = time.time() - t0
                    print("%d/%d states, %d errors, %.2fM completion tokens, %.0f s, %.0f tokens/s" % (
                        n_ok + n_err, len(todo), n_err, n_tok / 1e6, dt, n_tok / max(dt, 1)), flush=True)
    print("SAMPLE_STATES_DONE ok %d errors %d tokens %.2fM in %.0f s" % (n_ok, n_err, n_tok / 1e6, time.time() - t0), flush=True)
    return 3 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

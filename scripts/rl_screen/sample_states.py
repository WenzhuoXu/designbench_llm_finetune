"""Step 2 of a screened supervised round: k replies of the served policy at every state.

One chat-completion request per reply, with the state's served prompt (system, user) and thinking
on, sampled as the RL rollouts sampled (temperature 0.8, top_p 1, top_k 0 by default) with no
completion cap. A state's first reply is asked for alone and the other k - 1 once it has come back,
all from the same server, so the prompt (about ten times the tokens of a reply) is computed once
and then served from the server's prefix cache. A request is given up after --timeout seconds (the
harness client's hour) and asked once more with another seed; a reply that fails twice is recorded
as an error and the state keeps its other replies.

Output: one line per (state, sample): `thinking` (the think block) and `answer` (the visible text
after it, where the harness reads the move), `sha1` of both. A reply whose think block was not
closed has finish "length" and no answer.

Resumable: replies already in --out are not asked for again. States are taken in the order of
--first (families listed there go first), then file order.

    python sample_states.py --states states.jsonl --out samples.jsonl --servers URL [URL ...] \\
        --model dx38mtR [--k 8] [--temperature 0.8] [--workers 40] [--first photonics grid2op engibench]
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


def reply_sha1(thinking: str, answer: str) -> str:
    return hashlib.sha1((thinking + "\x00" + answer).encode()).hexdigest()[:16]


def record_of(sid: str, sample: int, out: dict) -> dict:
    ch = (out.get("choices") or [{}])[0]
    reasoning, text, closed = split_reply(ch.get("message") or {})
    finish = ch.get("finish_reason")
    ok = closed and finish != "length"
    usage = out.get("usage") or {}
    return {"sid": sid, "sample": int(sample), "thinking": reasoning if ok else "", "answer": text if ok else "",
            "finish": finish if closed else "length", "mode": "own_think", "prompt": "served",
            "tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "sha1": reply_sha1(reasoning, text) if ok else None}


def seed_of(sid: str, sample: int, attempt: int, seed: int) -> int:
    return int(hashlib.sha1(("%s|%d|%d|%d" % (sid, sample, attempt, seed)).encode()).hexdigest()[:8], 16) % (2 ** 31)


def request_body(state: dict, sample: int, attempt: int, a) -> dict:
    return {"model": a.model,
            "messages": [{"role": "system", "content": state["system"]},
                         {"role": "user", "content": state["observation"] + state["instruction"]}],
            "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "max_tokens": None,
            "seed": seed_of(state["sid"], sample, attempt, a.seed), "chat_template_kwargs": {"enable_thinking": True}}


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def draw(state: dict, sample: int, j: int, a, post_fn=post) -> dict:
    """One reply: the record, or an error record after --attempts tries (another seed each time; a
    try after the first goes to the next server, in case the first one is gone)."""
    err = None
    for attempt in range(a.attempts):
        url = a.servers[(j + attempt) % len(a.servers)]
        try:
            return record_of(state["sid"], sample, post_fn(url, request_body(state, sample, attempt, a), a.timeout))
        except Exception as e:                               # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:300])
            time.sleep(min(5.0, a.retry_wait) * (attempt + 1))
    return {"sid": state["sid"], "sample": int(sample), "error": err}


def draw_state(state: dict, missing: list[int], j: int, a, post_fn=post) -> list[dict]:
    """The missing replies of one state: the first alone (it fills the prefix cache), then the rest
    at once."""
    out = [draw(state, missing[0], j, a, post_fn)]
    rest = missing[1:]
    if rest:
        box = [None] * len(rest)

        def run(i, k):
            box[i] = draw(state, k, j, a, post_fn)
        threads = [threading.Thread(target=run, args=(i, k)) for i, k in enumerate(rest)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out.extend(box)
    return out


def done_replies(path: str) -> set:
    """(sid, sample) of the replies already in the file (an error record does not count)."""
    done = set()
    try:
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except ValueError:
                continue                                      # the cut last line of a killed run
            if not o.get("error"):
                done.add((o["sid"], int(o["sample"])))
    except FileNotFoundError:
        pass
    return done


def open_append(path: str):
    """The file for appending; a last line cut by a killed run is closed off first, so the next
    record does not continue it."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            cut = f.read(1) != b"\n"
        if cut:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
    return open(path, "a", encoding="utf-8")


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
    ap.add_argument("--workers", type=int, default=40, help="states in flight (each has up to k - 1 requests at once)")
    ap.add_argument("--first", nargs="*", default=[], help="families sampled first")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--retry-wait", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    states = [json.loads(line) for line in open(a.states, encoding="utf-8") if line.strip()]
    rank = {v: i for i, v in enumerate(a.first)}
    states.sort(key=lambda s: rank.get(s["variant"], len(rank)))     # stable: file order inside a family
    done = done_replies(a.out)
    todo = []
    for s in states:
        missing = [k for k in range(a.k) if (s["sid"], k) not in done]
        if missing:
            todo.append((s, missing))
    if a.limit:
        todo = todo[:a.limit]
    print("states %d, replies already there %d, states to do %d; k %d, T %.2f, top_p %.2f, top_k %d, model %s, %d servers" % (
        len(states), len(done), len(todo), a.k, a.temperature, a.top_p, a.top_k, a.model, len(a.servers)), flush=True)
    t0 = time.time()
    n_state = n_ok = n_err = n_tok = n_prompt = n_cached = 0
    with open_append(a.out) as fo, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(draw_state, s, missing, j, a) for j, (s, missing) in enumerate(todo)]
        for fu in as_completed(futs):
            for r in fu.result():                             # written by this thread only
                fo.write(json.dumps(r) + "\n")
                if r.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
                    n_tok += r.get("tokens") or 0
                    n_prompt += r.get("prompt_tokens") or 0
                    n_cached += r.get("cached_tokens") or 0
            fo.flush()
            n_state += 1
            if n_state % 100 == 0:
                dt = time.time() - t0
                print("%d/%d states, %d replies, %d errors, %.2fM completion tokens (%.0f/s), prompt %.1fM of which cached %.1fM, %.0f s" % (
                    n_state, len(todo), n_ok, n_err, n_tok / 1e6, n_tok / max(dt, 1), n_prompt / 1e6, n_cached / 1e6, dt), flush=True)
    print("SAMPLE_STATES_DONE states %d replies %d errors %d completion tokens %.2fM prompt tokens %.1fM cached %.1fM in %.0f s" % (
        n_state, n_ok, n_err, n_tok / 1e6, n_prompt / 1e6, n_cached / 1e6, time.time() - t0), flush=True)
    return 3 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())

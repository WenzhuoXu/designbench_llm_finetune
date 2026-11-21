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
        [--extra-instruction-file forecast_instruction.txt --prompt-tag served+forecast_v1]

--extra-instruction-file: a collection-only instruction put after the served prompt (a
forecast-checked round asks for FORECAST lines this way, see forecasts.py). The rows are built from
the state's served prompt, so the instruction is never part of a training prompt.

--cue FORECAST: (cued_reply). The tuned policy compares candidates when asked but rarely writes the
lines in the form the check can read (4 to 11% of replies, whatever the wording). With the cue, its
rationale is sampled as usual, then the label is put on a new line and the policy continues from
there; a second and third label follow while it has written fewer than --cue-min-lines lines. The
label is the only text in a reply that the policy did not sample.

--kv-budget TOKENS (with --cue): the states in flight on a server are limited by the KV cache they
need instead of by their number. Measured on four H100s (349,781 cache tokens each): 28 states per
server filled the cache (98%), prefixes were computed again (32% reuse) and the four made 1,345
completion tokens a second; 14 per server kept it at 47-77% and made 2,308. It changes when a
reply is asked for, not what is asked (the seed of a reply depends on its state and number only).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecasts  # noqa: E402


class Budget:
    """The KV-cache tokens the states in flight on one server may need (--kv-budget). A state waits
    until its cost fits, in arrival order, so a long prompt is not overtaken for ever; a state that
    costs more than the whole budget runs alone. Without it the number of states in flight is fixed
    (--workers), and prompts of very different lengths either leave the server idle or fill its
    cache, after which the prefix of every request is computed again instead of reused."""

    def __init__(self, total: int):
        self.total, self.used, self.queue, self.cv = int(total), 0, [], threading.Condition()

    def acquire(self, cost: int) -> int:
        cost = max(1, min(int(cost), self.total))
        me = object()
        with self.cv:
            self.queue.append(me)
            while self.queue[0] is not me or self.used + cost > self.total:
                self.cv.wait()
            self.queue.pop(0)
            self.used += cost
            self.cv.notify_all()
        return cost

    def release(self, cost: int) -> None:
        with self.cv:
            self.used -= cost
            self.cv.notify_all()


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


def record_of(sid: str, sample: int, out: dict, prompt: str = "served", prompt_sha1=None) -> dict:
    ch = (out.get("choices") or [{}])[0]
    reasoning, text, closed = split_reply(ch.get("message") or {})
    finish = ch.get("finish_reason")
    ok = closed and finish != "length"
    usage = out.get("usage") or {}
    return {"sid": sid, "sample": int(sample), "thinking": reasoning if ok else "", "answer": text if ok else "",
            "finish": finish if closed else "length", "mode": "own_think", "prompt": prompt,
            "tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "sha1": reply_sha1(reasoning, text) if ok else None,
            **({"prompt_sha1": prompt_sha1} if prompt_sha1 else {})}


def seed_of(sid: str, sample: int, attempt: int, seed: int) -> int:
    return int(hashlib.sha1(("%s|%d|%d|%d" % (sid, sample, attempt, seed)).encode()).hexdigest()[:8], 16) % (2 ** 31)


def request_body(state: dict, sample: int, attempt: int, a) -> dict:
    # a.extra: a collection-only instruction after the served prompt (--extra-instruction-file);
    # the training rows are built from the state's own prompt, without it
    return {"model": a.model,
            "messages": [{"role": "system", "content": state["system"]},
                         {"role": "user", "content": state["observation"] + state["instruction"] + (getattr(a, "extra", "") or "")}],
            "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "max_tokens": None,
            "seed": seed_of(state["sid"], sample, attempt, a.seed), "chat_template_kwargs": {"enable_thinking": True}}


def post_to(url: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post(url: str, body: dict, timeout: float) -> dict:
    return post_to(url, "/chat/completions", body, timeout)


THINK_END = "</think>"


def render_prompt(state: dict, a, url: str, post_to_fn=post_to) -> list:
    """The token ids of the state's chat request (with the generation prompt), as the server renders it."""
    body = request_body(state, 0, 0, a)
    return post_to_fn(url, "/chat/completions/render", {"model": a.model, "messages": body["messages"],
                                                       "chat_template_kwargs": body["chat_template_kwargs"]}, a.timeout)["token_ids"]


def cued_reply(state: dict, sample: int, attempt: int, a, url: str, post_to_fn=post_to, prompt_ids=None) -> dict:
    """One reply sampled with the cue (--cue): the policy writes its rationale as usual and is
    stopped where it would close the think block; while the rationale has fewer than
    --cue-min-lines forecast lines, the same generation is continued from it plus the cue
    (a.cues: two newlines and the cue the first time, one newline and the cue after), at most
    --cue-max times; then the think block is closed and the policy gives its action. Everything
    but the cue itself is the policy's own sample at the same temperature. The prompt is the chat
    request's, rendered by the server, so the tokens are the ones a chat request would have had."""
    seed = seed_of(state["sid"], sample, attempt, a.seed)
    # prompt_ids: the state's prompt already rendered (draw_state renders it once for the k replies)
    gen = list(prompt_ids) if prompt_ids is not None else render_prompt(state, a, url, post_to_fn)
    samp = {"model": a.model, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "add_special_tokens": False,
            "return_token_ids": True, "skip_special_tokens": False}
    rec = {"sid": state["sid"], "sample": int(sample), "thinking": "", "answer": "", "finish": "length", "mode": "own_think_cued",
           "prompt": getattr(a, "prompt_tag", None) or "served", "tokens": 0, "prompt_tokens": len(gen), "cached_tokens": None,
           "sha1": None, "cues": 0}
    if getattr(a, "prompt_sha1", None):
        rec["prompt_sha1"] = a.prompt_sha1
    think = ""
    for step in range(a.cue_max + 1):                        # the rationale, then the cued continuations
        out = post_to_fn(url, "/completions", dict(samp, prompt=gen, seed=(seed + step) % (2 ** 31), stop_token_ids=[a.end_id],
                                                   max_tokens=a.ctx - len(gen) - 8), a.timeout)
        ch = (out.get("choices") or [{}])[0]
        usage = out.get("usage") or {}
        rec["tokens"] += usage.get("completion_tokens") or 0
        if step == 0:
            rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if ch.get("finish_reason") == "length":
            return rec
        if ch.get("token_ids") is None:
            raise RuntimeError("the server returned no token ids for a completion")
        gen = gen + [t for t in ch["token_ids"] if t != a.end_id]
        think += (ch.get("text") or "").replace(THINK_END, "")
        lines, unread = forecasts.read(think)
        if len(lines) + unread >= a.cue_min_lines or step == a.cue_max:
            break
        text, ids = a.cues[0] if rec["cues"] == 0 else a.cues[1]
        think += text
        gen = gen + ids
        rec["cues"] += 1
    out = post_to_fn(url, "/completions", dict(samp, prompt=gen + [a.end_id], seed=(seed + 97) % (2 ** 31),
                                               max_tokens=a.ctx - len(gen) - 9), a.timeout)
    ch = (out.get("choices") or [{}])[0]
    rec["tokens"] += (out.get("usage") or {}).get("completion_tokens") or 0
    if ch.get("finish_reason") == "length":
        return rec
    answer = (ch.get("text") or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    thinking = think.strip()
    rec.update(thinking=thinking, answer=answer, finish=ch.get("finish_reason"), sha1=reply_sha1(thinking, answer))
    return rec


def draw(state: dict, sample: int, j: int, a, post_fn=post, post_to_fn=post_to, prompt_ids=None) -> dict:
    """One reply: the record, or an error record after --attempts tries (another seed each time; a
    try after the first goes to the next server, in case the first one is gone)."""
    err = None
    for attempt in range(a.attempts):
        url = a.servers[(j + attempt) % len(a.servers)]
        try:
            if getattr(a, "cue", None):
                return cued_reply(state, sample, attempt, a, url, post_to_fn, prompt_ids)
            return record_of(state["sid"], sample, post_fn(url, request_body(state, sample, attempt, a), a.timeout),
                             getattr(a, "prompt_tag", None) or "served", getattr(a, "prompt_sha1", None))
        except Exception as e:                               # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:300])
            time.sleep(min(5.0, a.retry_wait) * (attempt + 1))
    return {"sid": state["sid"], "sample": int(sample), "error": err}


def draw_state(state: dict, missing: list[int], j: int, a, post_fn=post, post_to_fn=post_to) -> list[dict]:
    """The missing replies of one state: the first alone (it fills the prefix cache), then the rest
    at once. Cued sampling renders the state's prompt once for all of them (if that fails, every
    reply renders it again in its own tries). With --kv-budget the state first waits for room on its
    server: its prompt once (the replies share it) plus --reply-tokens for each reply."""
    ids = None
    if getattr(a, "cue", None):
        try:
            ids = render_prompt(state, a, a.servers[j % len(a.servers)], post_to_fn)
        except Exception:                                    # noqa: BLE001
            ids = None
    budget = getattr(a, "budgets", None)
    budget = budget[j % len(a.servers)] if budget and ids is not None else None
    cost = budget.acquire(len(ids) + len(missing) * a.reply_tokens) if budget else 0
    try:
        out = [draw(state, missing[0], j, a, post_fn, post_to_fn, ids)]
        rest = missing[1:]
        if rest:
            box = [None] * len(rest)

            def run(i, k):
                box[i] = draw(state, k, j, a, post_fn, post_to_fn, ids)
            threads = [threading.Thread(target=run, args=(i, k)) for i, k in enumerate(rest)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            out.extend(box)
        return out
    finally:
        if budget:
            budget.release(cost)


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


def prompts_in(path: str) -> set:
    """(prompt tag, sha1 of the extra instruction) of the replies already in the file."""
    seen = set()
    try:
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if not o.get("error"):
                seen.add((o.get("prompt") or "served", o.get("prompt_sha1")))
    except FileNotFoundError:
        pass
    return seen


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
    ap.add_argument("--extra-instruction-file", default=None,
                    help="text put after the served prompt of every request (collection only: the rows keep the served prompt)")
    ap.add_argument("--prompt-tag", default=None, help="recorded as `prompt` in every reply (default: served, or served+extra)")
    ap.add_argument("--cue", default=None,
                    help="cued sampling (cued_reply): this text, e.g. FORECAST:, is put on a new line after the policy's own "
                         "rationale while the rationale has fewer than --cue-min-lines forecast lines")
    ap.add_argument("--cue-min-lines", type=int, default=2)
    ap.add_argument("--cue-max", type=int, default=3, help="cues per reply at most")
    ap.add_argument("--kv-budget", type=int, default=0,
                    help="with --cue: KV-cache tokens the states in flight on one server may need (0: only --workers limits them)")
    ap.add_argument("--reply-tokens", type=int, default=1700, help="tokens counted for one reply under --kv-budget")
    a = ap.parse_args(argv)
    a.budgets = [Budget(a.kv_budget) for _ in a.servers] if a.kv_budget > 0 else None
    if a.budgets:
        print("states in flight limited to %d KV-cache tokens per server (prompt once + %d per reply), at most %d states in all" % (
            a.kv_budget, a.reply_tokens, a.workers), flush=True)
    a.extra, a.prompt_sha1 = "", None
    if a.extra_instruction_file:
        with open(a.extra_instruction_file, encoding="utf-8", newline="") as f:
            a.extra = f.read().replace("\r\n", "\n")
        if not a.extra.strip():
            print("ABORT: %s is empty" % a.extra_instruction_file)
            return 2
        a.prompt_tag = a.prompt_tag or "served+extra"
        a.prompt_sha1 = hashlib.sha1(a.extra.encode()).hexdigest()[:12]
        print("extra instruction (%d characters, sha1 %s), replies tagged %s" % (
            len(a.extra), hashlib.sha1(a.extra.encode()).hexdigest()[:12], a.prompt_tag), flush=True)

    if a.cue:
        a.prompt_tag = (a.prompt_tag or "served") + "+cue"
        a.prompt_sha1 = hashlib.sha1(("%s\x00cue:%s:%d:%d" % (a.extra, a.cue, a.cue_min_lines, a.cue_max)).encode()).hexdigest()[:12]
        root = a.servers[0].rstrip("/")
        root = root[:-3] if root.endswith("/v1") else root

        def tok(text):
            return post_to(root, "/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False}, 120.0)["tokens"]
        end = tok(THINK_END)
        if len(end) != 1:
            print("ABORT: %s is %d tokens for this model, not one" % (THINK_END, len(end)))
            return 2
        a.end_id = end[0]
        a.cues = [("\n\n" + a.cue, tok("\n\n" + a.cue)), ("\n" + a.cue, tok("\n" + a.cue))]
        with urllib.request.urlopen(a.servers[0].rstrip("/") + "/models", timeout=120.0) as r:
            a.ctx = int(json.load(r)["data"][0]["max_model_len"])
        print("cued sampling: cue %r (token ids %s, then %s), at least %d forecast lines, at most %d cues; %s is token %d; context %d; "
              "replies tagged %s (%s)" % (a.cue, a.cues[0][1], a.cues[1][1], a.cue_min_lines, a.cue_max, THINK_END, a.end_id, a.ctx,
                                          a.prompt_tag, a.prompt_sha1), flush=True)

    states = [json.loads(line) for line in open(a.states, encoding="utf-8") if line.strip()]
    rank = {v: i for i, v in enumerate(a.first)}
    states.sort(key=lambda s: rank.get(s["variant"], len(rank)))     # stable: file order inside a family
    other = prompts_in(a.out) - {(a.prompt_tag or "served", a.prompt_sha1)}
    if other:                                                # replies sampled under another prompt must not be continued
        print("ABORT: %s holds replies sampled under %s; this run samples under %s" % (
            a.out, sorted(other, key=str), (a.prompt_tag or "served", a.prompt_sha1)))
        return 2
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

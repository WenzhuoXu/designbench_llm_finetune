"""Run physics-CoT streams for a set of state records (append-only, resumable by sid).

    # OpenAI writer (local machine only; the key is read from OPENAI_API_KEY or ~/.openai_key and
    # never written anywhere)
    python -m dataxp.stream.run --states S.jsonl --out O.jsonl --backend openai --model gpt-6.1-sol
    # Qwen through an OpenAI-compatible vLLM server
    python -m dataxp.stream.run --states S.jsonl --out O.jsonl --backend vllm \
        --endpoint http://127.0.0.1:PORT/v1 --model Qwen/Qwen3.8-27B
Options: --sids a,b (subset), --groups g1,g2, --limit N, --concurrency C, --cap 6.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import httpx  # noqa: E402

from .env import StreamState  # noqa: E402
from .protocol import CAP, run_stream  # noqa: E402

KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-\*]{6,}")
RETRY_CODES = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}


def _key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if not k:
        p = os.path.join(os.path.expanduser("~"), ".openai_key")
        if os.path.exists(p):
            k = open(p, encoding="utf-8").read().strip()
    if not k:
        raise SystemExit("no OpenAI key: set OPENAI_API_KEY or write ~/.openai_key")
    return k


def scrub(s, key=None) -> str:
    s = str(s)
    if key:
        s = s.replace(key, "<redacted>")
    return KEY_RE.sub("<redacted-key>", s)


class OpenAIChat:
    """Responses API, store=false, reasoning items carried between turns as encrypted content."""
    URL = "https://api.openai.com/v1/responses"

    def __init__(self, model: str, max_output_tokens: int = 32000):
        self.model, self.max_out = model, max_output_tokens
        self.key = _key()
        self.client = httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0),
                                   headers={"Authorization": "Bearer " + self.key,
                                            "Content-Type": "application/json"})

    def _input(self, msgs):
        out = []
        for m in msgs:
            if m["role"] == "assistant" and m.get("_items"):
                out.extend(m["_items"])
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def __call__(self, msgs):
        body = {"model": self.model, "input": self._input(msgs), "max_output_tokens": self.max_out,
                "store": False, "include": ["reasoning.encrypted_content"]}
        t0, attempts = time.time(), []
        for attempt in range(8):
            try:
                r = self.client.post(self.URL, json=body)
            except Exception as e:  # noqa: BLE001
                attempts.append(scrub(repr(e)[:200], self.key))
                time.sleep(min(120, 4 * 2 ** attempt) + random.random() * 3)
                continue
            if r.status_code == 200:
                d = r.json()
                break
            attempts.append("http %d: %s" % (r.status_code, scrub(r.text[:300], self.key)))
            if r.status_code in RETRY_CODES or r.status_code >= 500:
                time.sleep(min(120, 4 * 2 ** attempt) + random.random() * 3)
                continue
            return {"error": attempts[-1], "attempts": attempts}
        else:
            return {"error": "retries exhausted", "attempts": attempts}
        texts, items = [], []
        for o in d.get("output") or []:
            if o.get("type") in ("reasoning", "message"):
                items.append(o)
            if o.get("type") == "message":
                for c in o.get("content") or []:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text") or "")
        inc = (d.get("incomplete_details") or {}).get("reason")
        finish = "stop" if d.get("status") == "completed" else ("length" if inc == "max_output_tokens" else inc)
        u = d.get("usage") or {}
        usage = {"input": u.get("input_tokens"), "cached": (u.get("input_tokens_details") or {}).get("cached_tokens"),
                 "output": u.get("output_tokens"),
                 "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens")}
        return {"text": "".join(texts).strip(), "finish": finish, "usage": usage, "items": items,
                "latency_s": round(time.time() - t0, 1), "retries": len(attempts)}


class VLLMChat:
    def __init__(self, endpoint: str, model: str, temperature: float = 0.6, max_tokens: int = 12000,
                 thinking: bool = False, seed: int = 0):
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.max_tokens, self.thinking = model, temperature, max_tokens, thinking
        self.client = httpx.Client(timeout=httpx.Timeout(3600.0, connect=30.0))

    def __call__(self, msgs):
        body = {"model": self.model, "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        t0, attempts = time.time(), []
        for attempt in range(6):
            try:
                r = self.client.post(self.url, json=body)
                if r.status_code == 200:
                    d = r.json()
                    break
                attempts.append("http %d: %s" % (r.status_code, r.text[:300]))
                if r.status_code == 400 and "maximum context length" in r.text:
                    return {"error": "context_length", "attempts": attempts, "finish": "length"}
            except Exception as e:  # noqa: BLE001
                attempts.append(repr(e)[:200])
            time.sleep(min(60, 4 * 2 ** attempt))
        else:
            return {"error": "retries exhausted", "attempts": attempts}
        ch = d["choices"][0]
        u = d.get("usage") or {}
        text = ch["message"].get("content") or ""
        thought = ch["message"].get("reasoning_content") or ""
        if self.thinking and "</think>" in text:
            # no reasoning parser on the server: the reply carries the think block's end
            thought, _, text = text.rpartition("</think>")
        return {"text": text.strip(), "thinking": thought.strip() or None, "finish": ch.get("finish_reason"),
                "usage": {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens")},
                "latency_s": round(time.time() - t0, 1), "retries": len(attempts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["openai", "vllm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--writer", default="", help="label stored in each record (default: --model)")
    ap.add_argument("--sids", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=12000)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()]
    if a.sids:
        want = set(a.sids.split(","))
        recs = [r for r in recs if r["sid"] in want]
        assert len(recs) == len(want), "unknown sid"
    if a.groups:
        gs = set(a.groups.split(","))
        recs = [r for r in recs if (r.get("pilot") or {}).get("group") in gs]
    if a.limit:
        recs = recs[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                o = json.loads(l)
            except ValueError:
                continue
            # a side answer lost to an API error is not an answer: the stream runs again
            if o.get("status") not in ("error", None) and not any((s or {}).get("error") for s in (o.get("side") or {}).values()):
                done.add(o["sid"])
    todo = [r for r in recs if r["sid"] not in done]
    writer = a.writer or a.model
    print("%s: states %d, done %d, to run %d" % (writer, len(recs), len(done), len(todo)), flush=True)
    if a.backend == "openai":
        chat = OpenAIChat(a.model)
    else:
        chat = VLLMChat(a.endpoint, a.model, temperature=a.temperature, max_tokens=a.max_tokens)
    lock = threading.Lock()

    def one(r):
        t0 = time.time()
        try:
            ss = StreamState(r)
            o = run_stream(ss, chat, cap=a.cap)
        except Exception as e:  # noqa: BLE001
            o = {"sid": r["sid"], "status": "error", "error": scrub(repr(e)[:500])}
        o.update(writer=writer, model=a.model, backend=a.backend, wall_s=round(time.time() - t0, 1), ts=time.time())
        return o

    n = 0
    with open(a.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for f in as_completed(futs):
            o = f.result()
            with lock:
                fh.write(json.dumps(o) + "\n")
                fh.flush()
            n += 1
            print("[%d/%d] %s %s probes=%s %.0fs" % (n, len(todo), o["sid"], o["status"], o.get("n_probes"),
                                                    o.get("wall_s", 0)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

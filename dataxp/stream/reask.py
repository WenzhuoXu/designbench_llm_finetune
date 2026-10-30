"""Re-ask the held-out side question on stored streams (docs/cot_stream_validation.md §6, 2026-10-30).

The stream messages are kept; only the "pre" answer (context: up to the writer's first message) and the
"post" answer (context: the whole stream) are asked again with the current side question. The old answers
are kept under "side_v1". Needs the writer's own context, so it suits writers whose reasoning is not
carried between turns (Qwen with thinking off); OpenAI streams do not store their reasoning items.

    python -m dataxp.stream.reask --states S.jsonl --streams IN.jsonl --out OUT.jsonl --backend vllm \
        --endpoint http://127.0.0.1:PORT/v1 --model Qwen/Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .env import StreamState
from .protocol import SIDE_VERSION, _side, side_question
from .run import VLLMChat, scrub


def reask(ss: StreamState, o: dict, chat) -> dict:
    sq = side_question(ss, o["H"])
    msgs = [{"role": m["role"], "content": m["content"]} for m in o["messages"]]
    first = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
    side = {}
    if first is not None:
        side["pre"] = _side(chat, msgs[:first + 1], sq)
        if o.get("status") in ("done", "no_action"):
            side["post"] = _side(chat, msgs, sq)
    out = dict(o)
    out.update(side_v1=o.get("side"), side_question_v1=o.get("side_question"), side=side, side_question=sq,
               side_version=SIDE_VERSION)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vllm"], default="vllm")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=12000)
    a = ap.parse_args()
    recs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()}
    streams = [json.loads(l) for l in open(a.streams, encoding="utf-8") if l.strip()]
    streams = [o for o in streams if o.get("status") not in ("error", None)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                o = json.loads(l)
            except ValueError:
                continue
            if all(not (s or {}).get("error") for s in (o.get("side") or {}).values()):
                done.add(o["sid"])
    todo = [o for o in streams if o["sid"] not in done]
    print("re-ask: streams %d, done %d, to run %d" % (len(streams), len(done), len(todo)), flush=True)
    chat = VLLMChat(a.endpoint, a.model, temperature=a.temperature, max_tokens=a.max_tokens)
    lock = threading.Lock()

    def one(o):
        t0 = time.time()
        try:
            r = reask(StreamState(recs[o["sid"]]), o, chat)
        except Exception as e:  # noqa: BLE001
            r = dict(o, side={"pre": {"error": scrub(repr(e)[:500])}})
        r["reask_wall_s"] = round(time.time() - t0, 1)
        return r

    n = 0
    with open(a.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.concurrency) as ex:
        for f in as_completed([ex.submit(one, o) for o in todo]):
            r = f.result()
            with lock:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            n += 1
            print("[%d/%d] %s %s %.0fs" % (n, len(todo), r["sid"], ",".join(sorted(r["side"])),
                                          r["reask_wall_s"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""G3 (docs/cot_stream_validation.md §3): can the untuned student use a writer's stream?

The untuned Qwen3.8-27B (thinking off, greedy) answers the stream's held-out side question in four
contexts built from each finished stream:
  a  the state only
  b  the state + the writer's whole stream (its side questions are never part of it)
  c  as b, with every probe result replaced by the result of a different random legal move on the
     same state (the header still names the probed action)
  d  the state + the probe actions and their results only, no writer text

    python -m dataxp.stream.g3 run --states S.jsonl --streams W.jsonl --out G3.jsonl \
        --endpoint http://127.0.0.1:PORT/v1
    python -m dataxp.stream.g3 score --states S.jsonl --streams W.jsonl --answers G3.jsonl --out R.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..domain import stable_seed
from .env import StreamState
from .protocol import system_text, user_text
from .score import boot_ci, score_side, state_means

CONTEXTS = ("a", "b", "c", "d")


def _swap_result(ss: StreamState, stream: dict, k: int, msg: str) -> str:
    """Probe k's result message rebuilt from a different random legal move's simulation."""
    p = stream["probes"][k]
    if not p.get("ok"):
        return msg
    moves = [m["text"] for m in (ss.r.get("moves") or []) if not ss.same_move(m["text"], p["text"])]
    rng = np.random.default_rng(stable_seed(ss.sid, "g3_swap", k))
    for _ in range(20):
        other = ss.check(moves[int(rng.integers(len(moves)))])
        if other["ok"]:
            break
    else:
        return msg
    fake = dict(other)
    fake.update(applied=p.get("applied") or p["text"], noop=False, clamped=False)
    return ss.result_message(fake, k + 1, stream.get("cap", 6))


def contexts(ss: StreamState, stream: dict) -> dict:
    cap = stream.get("cap", 6)
    sysm = {"role": "system", "content": system_text(ss.r, cap)}
    obs = user_text(ss.r, cap)
    sq = stream["side_question"]
    body = [m for m in stream["messages"][2:]]
    out = {"a": [sysm, {"role": "user", "content": obs + "\n\n" + sq}]}
    out["b"] = [sysm, {"role": "user", "content": obs}] + [
        {"role": m["role"], "content": m["content"]} for m in body] + [{"role": "user", "content": sq}]
    c = []
    for m in body:
        if m["role"] == "user" and "_probe" in m:
            c.append({"role": "user", "content": _swap_result(ss, stream, m["_probe"], m["content"])})
        else:
            c.append({"role": m["role"], "content": m["content"]})
    out["c"] = [sysm, {"role": "user", "content": obs}] + c + [{"role": "user", "content": sq}]
    d = []
    for m in body:
        if m["role"] == "user" and "_probe" in m:
            d.append({"role": "assistant", "content": "<check>%s</check>" % stream["probes"][m["_probe"]]["text"]})
            d.append({"role": "user", "content": m["content"]})
    if stream.get("final_action"):
        d.append({"role": "assistant", "content": "<action>%s</action>" % stream["final_action"]})
    out["d"] = [sysm, {"role": "user", "content": obs}] + d + [{"role": "user", "content": sq}]
    return out


def _load_streams(path):
    streams = {}
    for l in open(path, encoding="utf-8"):
        o = json.loads(l)
        if o.get("status") in ("done", "no_action") and o.get("n_probes") and "post" in (o.get("side") or {}):
            streams[o["sid"]] = o
    return streams


def cmd_run(a):
    from .run import VLLMChat
    recs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()}
    streams = _load_streams(a.streams)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            o = json.loads(l)
            if not o.get("error"):
                done.add((o["sid"], o["ctx"]))
    chat = VLLMChat(a.endpoint, a.model, temperature=0.0, max_tokens=a.max_tokens)
    jobs = [(sid, c) for sid in sorted(streams) for c in CONTEXTS if (sid, c) not in done]
    print("G3 %s: streams %d, to run %d" % (a.streams, len(streams), len(jobs)), flush=True)
    cache, lock = {}, threading.Lock()

    def one(sid, c):
        with lock:
            if sid not in cache:
                ss = StreamState(recs[sid])
                cache[sid] = (ss, contexts(ss, streams[sid]))
        _ss, ctx = cache[sid]
        rep = chat(ctx[c])
        return {"sid": sid, "ctx": c, "writer": streams[sid].get("writer"), "text": rep.get("text"),
                "finish": rep.get("finish"), "usage": rep.get("usage"), "error": rep.get("error")}

    with open(a.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(one, s, c) for s, c in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            o = f.result()
            with lock:
                fh.write(json.dumps(o) + "\n")
                fh.flush()
            if n % 20 == 0:
                print("[%d/%d]" % (n, len(jobs)), flush=True)


def cmd_score(a):
    recs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()}
    streams = _load_streams(a.streams)
    ans = {}
    for l in open(a.answers, encoding="utf-8"):
        o = json.loads(l)
        if not o.get("error"):
            ans[(o["sid"], o["ctx"])] = o
    per = {c: {} for c in CONTEXTS}
    for sid, st in sorted(streams.items()):
        ss = StreamState(recs[sid])
        H = st["H"]
        probed = {i for i, h in enumerate(H) for p in st["probes"] if p.get("ok") and ss.same_move(p["text"], h)}
        for c in CONTEXTS:
            if (sid, c) not in ans:
                continue
            m = state_means(score_side(ss, H, ans[(sid, c)]["text"]), probed)
            if m is not None:
                per[c][sid] = m
    rep = {"n": {c: len(v) for c, v in per.items()}, "mean": {c: float(np.mean(list(v.values()))) if v else None
                                                              for c, v in per.items()}}
    for x, y in (("a", "b"), ("c", "b"), ("d", "b"), ("a", "d")):
        common = sorted(set(per[x]) & set(per[y]))
        diff = np.array([per[x][s] - per[y][s] for s in common])
        rep["%s_minus_%s" % (x, y)] = boot_ci(diff) + (len(common),)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--states", required=True)
    r.add_argument("--streams", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--endpoint", required=True)
    r.add_argument("--model", default="Qwen/Qwen3.8-27B")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=6000)
    s = sub.add_parser("score")
    s.add_argument("--states", required=True)
    s.add_argument("--streams", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())

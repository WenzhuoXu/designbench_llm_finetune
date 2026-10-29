"""Stage 2/3 training rows from physics-CoT streams (docs/cot_stream_validation.md §4).

Arms (same streams, same states):
  S  full stream: every assistant message supervised, probe results as unsupervised user turns
  C  S with every probe result replaced by the result of a different random legal move on the same
     state (the header still names the probed action)
  P  S with the free text removed: each assistant message keeps only its <review>, <predictions>
     and final <check>/<action> blocks

A stream becomes a row if it finished (status done) with a legal final action. Rows use the
sftprobe tier format (messages with role/content, assistant reasoning_content "").

    python -m dataxp.stream.sft_rows --states S.jsonl --streams W.jsonl --arm S --out train_S.jsonl \
        [--sids-file train_sids.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .env import StreamState
from .g3 import _swap_result
from .protocol import PRED_RE, REVIEW_RE, TAG_RE, system_text, user_text


def predictions_only(text: str) -> str:
    parts = []
    r = REVIEW_RE.findall(text or "")
    if r:
        parts.append("<review>%s</review>" % r[-1])
    p = PRED_RE.findall(text or "")
    if p:
        parts.append("<predictions>%s</predictions>" % p[-1])
    t = TAG_RE.findall(text or "")
    if t:
        tag, inner = t[-1]
        parts.append("<%s>%s</%s>" % (tag, inner.strip(), tag))
    return "\n".join(parts)


def build_row(ss: StreamState, st: dict, arm: str) -> dict:
    cap = st.get("cap", 6)
    msgs = [{"role": "system", "content": system_text(ss.r, cap)},
            {"role": "user", "content": user_text(ss.r, cap)}]
    for m in st["messages"][2:]:
        if m["role"] == "assistant":
            c = predictions_only(m["content"]) if arm == "P" else m["content"]
            msgs.append({"role": "assistant", "reasoning_content": "", "content": c})
        elif arm == "C" and "_probe" in m:
            msgs.append({"role": "user", "content": _swap_result(ss, st, m["_probe"], m["content"])})
        else:
            msgs.append({"role": "user", "content": m["content"]})
    return {"domain": ss.v, "instance": ss.r["problem_id"], "sid": ss.sid, "turn": ss.r.get("turn"),
            "tier": "cotstream_%s" % arm, "writer": st.get("writer"), "n_probes": st.get("n_probes"),
            "messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--arm", choices=["S", "C", "P"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sids-file", default="")
    a = ap.parse_args()
    recs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()}
    want = set(open(a.sids_file).read().split()) if a.sids_file else None
    streams = {}
    for l in open(a.streams, encoding="utf-8"):
        o = json.loads(l)
        streams[o["sid"]] = o
    cnt = Counter()
    with open(a.out, "w", encoding="utf-8") as fh:
        for sid in sorted(streams):
            if want is not None and sid not in want:
                continue
            st = streams[sid]
            cnt["streams"] += 1
            if st.get("status") != "done":
                cnt["drop_status_" + str(st.get("status"))] += 1
                continue
            ss = StreamState(recs[sid])
            if not st.get("final_action") or not ss.check(st["final_action"])["ok"]:
                cnt["drop_illegal_final"] += 1
                continue
            fh.write(json.dumps(build_row(ss, st, a.arm)) + "\n")
            cnt["rows"] += 1
    print(json.dumps(dict(cnt)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

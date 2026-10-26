"""Step 3: sample K replies per state from an OpenAI-compatible endpoint
(vLLM serving the un-tuned model), thinking off.

Prompt (gold_probe mode own_nothink, --prompt probe, the default): system =
gold_probe's SYSTEM_OWN in its written-reasoning form (the action line
replaced by the verb-call form + grammar() on a native domain), user = the
full policy observation (DOMAIN/GOAL head + body) followed by "Choose exactly
ONE change for THIS step.". --prompt served instead sends the training row's
own system/user messages (observation + task instruction), i.e. exactly what
the tuned policy will be served.

Output is append-only, one record per (sid, sample); a rerun skips every
(sid, sample) that already has a completed reply (gold_probe.run's resume
rule: a record counts when it has an answer and was not cut at max_tokens).

Every record carries draw_pass = 1 + the number of records the --out file
already held for its (sid, sample) when this run started: 1 for a first draw,
2 for the redraw of a truncated / errored reply on a rerun (the datav4 batch
recipe runs exactly one such redraw pass; docs/datav4_append_plan.md).
--max-draw-pass N (default: the environment's DATAXP_MAX_DRAW_PASS, else 0 =
no cap, today's behaviour) leaves a key alone once N records of it are in the
file: the first job of a batch runs with 1, so a requeued first job does not
redraw what it already drew; the redraw job runs with 2.

--dry-run calls no model: for each (sid, sample) it writes a random legal
move at the state (drawn from the same legal set cand_phi scores) in the
domain's native format, so the whole chain runs without a GPU.
--dry-run-teacher-share p makes a share p of those replies the teacher's move
instead, to exercise the 'equals the teacher' branch of build_tiers.

    python -m dataxp.ownrows.sample --states <dir>/states.jsonl --out <dir>/samples.jsonl \
        --k 6 --endpoint http://127.0.0.1:8766/v1 --model Qwen/Qwen3.8-27B \
        --max-tokens 4000 --temperature 0.6 --concurrency 8
    python -m dataxp.ownrows.sample --states ... --out ... --k 6 --dry-run
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..domain import Move, stable_seed
from ..domains import get_domain
from ..render_cot import MODEL
from .common import legal_rng, probe_prompt, read_jsonl, rebuild_state, served_messages

MODE = "own_nothink"
DRY_RUN_PREFIX = "Dry-run policy (no model was called): a random legal move at this state."


def call(endpoint, system, user, temperature, max_tokens, model, top_p=0.95, top_k=20,
         thinking=False, timeout=3600, attempts=3):
    """gold_probe.call with retries; returns the reply record (answer, thinking,
    finish, tokens) or {'error': ...} after the last attempt."""
    body = {"model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if temperature > 0:
        body.update(top_p=top_p, top_k=top_k)
    data = json.dumps(body).encode()
    err = None
    for attempt in range(attempts):
        req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            break
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
            if attempt + 1 < attempts:
                time.sleep(5 * (attempt + 1))
    else:
        return {"answer": "", "thinking": "", "error": err}
    ch = out["choices"][0]
    msg = ch["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if "</think>" in text:
        reasoning, text = text.rsplit("</think>", 1)
        reasoning = reasoning.replace("<think>", "")
    elif thinking and not reasoning and "<think>" in text:
        reasoning, text = text.replace("<think>", ""), ""      # cut off inside the think block
    return {"answer": text.strip(), "thinking": reasoning.strip(),
            "finish": ch.get("finish_reason"),
            "tokens": (out.get("usage") or {}).get("completion_tokens")}


def done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for l in fh:
                try:
                    o = json.loads(l)
                except ValueError:
                    continue
                if (o.get("answer") or o.get("thinking")) and o.get("finish") != "length":
                    done.add((o["sid"], int(o["sample"])))
    return done


def drawn_counts(path):
    """{(sid, sample): records already in the file} (any outcome)."""
    cnt = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for l in fh:
                try:
                    o = json.loads(l)
                    key = (o["sid"], int(o["sample"]))
                except (ValueError, KeyError, TypeError):
                    continue
                cnt[key] = cnt.get(key, 0) + 1
    return cnt


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


class DryRun:
    """A random legal move per (sid, sample), from the legal set cand_phi
    scores (same rng), or the teacher's move with probability teacher_share."""

    def __init__(self, teacher_share: float = 0.0):
        self.teacher_share = float(teacher_share)
        self._cache = {}

    def reply(self, row):
        sid = row["sid"]
        if sid not in self._cache:
            d = get_domain(row["variant"])
            st = rebuild_state(d, row["spec"], row["history"])
            moves = d.legal_moves(st, legal_rng(row["problem_id"], row["k"]))
            self._cache[sid] = (d, moves)
        d, moves = self._cache[sid]

        def one(sample):
            rng = np.random.default_rng(stable_seed(sid, "dryrun", int(sample)))
            if rng.random() < self.teacher_share or not moves:
                mv = Move.from_dict(row["gold"])
            else:
                mv = moves[int(rng.integers(len(moves)))]
            return {"answer": "%s\n<action>%s</action>" % (DRY_RUN_PREFIX, d.format_move(mv)),
                    "thinking": "", "finish": "stop", "tokens": None}
        return one


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8766/v1")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt", choices=("probe", "served"), default="probe",
                    help="probe: gold_probe own_nothink prompt; served: the training row's messages")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--per-variant", type=int, default=0, help="first N states per variant (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="no model: write random legal moves")
    ap.add_argument("--dry-run-teacher-share", type=float, default=0.0)
    ap.add_argument("--max-draw-pass", type=int,
                    default=int(os.environ.get("DATAXP_MAX_DRAW_PASS") or 0),
                    help="draw a key only while it has fewer than N records in --out (0 = no cap; "
                         "default $DATAXP_MAX_DRAW_PASS)")
    a = ap.parse_args(argv)

    rows = read_jsonl(a.states)
    if a.variants:
        rows = [r for r in rows if r["variant"] in a.variants]
    if a.per_variant:
        kept, cnt = [], {}
        for r in rows:
            if cnt.get(r["variant"], 0) < a.per_variant:
                kept.append(r)
                cnt[r["variant"]] = cnt.get(r["variant"], 0) + 1
        rows = kept
    done = done_keys(a.out)
    drawn = drawn_counts(a.out)
    domains = {}
    jobs = []
    capped = 0
    for r in rows:
        if r["variant"] not in domains:
            domains[r["variant"]] = get_domain(r["variant"])
        if a.prompt == "probe":
            system, user = probe_prompt(domains[r["variant"]], r)
        else:
            m = served_messages(r)
            system, user = m[0]["content"], m[1]["content"]
        for s in range(a.k):
            if (r["sid"], s) not in done:
                if a.max_draw_pass and drawn.get((r["sid"], s), 0) >= a.max_draw_pass:
                    capped += 1
                    continue
                jobs.append((r, s, system, user))
    print("states %d, replies to draw %d (already done %d), mode %s%s" % (
        len(rows), len(jobs), len(done), MODE, " DRY-RUN" if a.dry_run else ""), flush=True)
    if a.max_draw_pass:
        print("draw-pass cap %d: %d unfinished keys already drawn %d time(s) left alone"
              % (a.max_draw_pass, capped, a.max_draw_pass), flush=True)

    mode = MODE + ("_dryrun" if a.dry_run else "")

    def record(r, s, system, user, out):
        out.update({"sid": r["sid"], "sample": s, "mode": mode, "prompt": a.prompt,
                    "model": None if a.dry_run else a.model,
                    "prompt_chars": len(system) + len(user),
                    "system_sha1": _sha(system), "user_sha1": _sha(user),
                    "draw_pass": drawn.get((r["sid"], s), 0) + 1})
        return out

    n = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        if a.dry_run:
            dry = DryRun(a.dry_run_teacher_share)
            for r, s, system, user in jobs:
                fh.write(json.dumps(record(r, s, system, user, dry.reply(r)(s))) + "\n")
                n += 1
            fh.flush()
        else:
            def one(j):
                r, s, system, user = j
                out = call(a.endpoint, system, user, a.temperature, a.max_tokens, a.model,
                           top_p=a.top_p, top_k=a.top_k, thinking=False)
                return record(r, s, system, user, out)
            with ThreadPoolExecutor(max(1, a.concurrency)) as ex:
                futs = [ex.submit(one, j) for j in jobs]
                for f in as_completed(futs):      # write as each finishes
                    fh.write(json.dumps(f.result()) + "\n")
                    fh.flush()
                    n += 1
                    if n % 10 == 0:
                        print("%d/%d" % (n, len(jobs)), flush=True)
    print("wrote %d replies -> %s" % (n, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

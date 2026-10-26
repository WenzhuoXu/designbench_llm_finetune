"""Outcome-consistency filter for rendered CoT.

A trace is written BEFORE the change is evaluated and must not state outcome
numbers, but it does make causal claims ("lowering Cc raises GBW", "a bigger
sink at n11 pulls heat out of n10"). Those claims can be checked against ground
truth: the trajectory records what the gold move actually did. This module
replays each rendered turn, applies the gold action, and asks a judge model to
extract every directional claim in the trace and compare it with the measured
before -> after change. Traces with any contradicted claim are rejected.

The judge does not predict anything - it reads claims and compares them with
numbers it is given - which is a much easier task than the one the trace itself
performs. Its reliability is still audited separately rather than assumed.

    python -m dataxp.outcome_check --data <run1> --cot <cot_obs dir> \
        --out <dir> --endpoint http://127.0.0.1:8765/v1 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .domain import apply_action
from .render_cot import MODEL, _fmt, iter_states

JUDGE_SYSTEM = """You audit engineering reasoning against measured results. You receive (1) the observation an engineer had BEFORE making a change, (2) the reasoning they wrote before the change was evaluated, and (3) the ACTUAL measured results AFTER the change. Check every causal claim in the reasoning about what the change does.

A causal claim is any statement that the change, or the mechanism it triggers, raises, lowers, relieves, worsens, improves, shifts or leaves unchanged a specific quantity: a constraint metric, the objective, or a per-element quantity (a member's force or safety factor, a node's temperature, a heat or power flow, a line's loading, a group's infections, a device's gain or pole). Include hedged claims ("should", "will likely", "potentially"). Also include claims about a gradient or flow between two elements. Ignore statements that only describe the current state, and claims about quantities the AFTER data does not let you evaluate.

For each claim compare the claimed direction with the actual direction from the BEFORE and AFTER data. The (increase/decrease/unchanged) labels on the aggregate metrics are computed exactly from the simulator - use them as given, however small the change. For per-element values, a change that vanishes at the printed precision counts as "unchanged". Only the direction matters, not the magnitude: a hedge about size ("without increasing area significantly") is not a claim that the quantity is unchanged. A claim that a change resolves, fixes or satisfies a constraint is a claim that the constraint is met AFTER the change - check it against the limit.

Respond with ONLY a JSON object, no prose:
{"claims": [{"quote": "<short verbatim quote>", "quantity": "<name>", "claimed": "increase|decrease|unchanged", "actual": "increase|decrease|unchanged|not_reported", "consistent": true|false}], "contradicted": true|false}
"contradicted" is true if any claim whose actual direction is reported is inconsistent."""


def _direction(a, b, rel=1e-9):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return "not_reported"
    if a != a or b != b:
        return "not_reported"
    if abs(b - a) <= rel * max(1e-12, abs(a), abs(b)):
        return "unchanged"
    return "increase" if b > a else "decrease"


def build_judge_prompt(row, domain, spec, st_before, st_after):
    obs = row["prompt"].split("\nSTYLE FOR THIS ONE")[0]
    m0 = domain.evaluate(domain.clone(st_before), seed=0)
    m1 = domain.evaluate(domain.clone(st_after), seed=0)
    lines = ["=== OBSERVATION BEFORE THE CHANGE ===", obs, "",
             "=== REASONING (written before the change was evaluated) ===",
             row["cot"], "",
             "=== ACTUAL RESULTS AFTER THE CHANGE ===",
             "Aggregate metrics, before -> after (%s):" % (
                 "nominal evaluation" if domain.stochastic else "deterministic")]
    deltas = {}
    for k in sorted(set(m0) | set(m1)):
        d = _direction(m0.get(k), m1.get(k))
        deltas[k] = d
        lines.append("  %-24s %s -> %s   (%s)" % (k, _fmt(m0.get(k)), _fmt(m1.get(k)), d))
    lines += ["", "Full observation of the design AFTER the change:",
              domain.describe(st_after, spec).rstrip()]
    return "\n".join(lines), deltas


def call_judge(endpoint, prompt, model=MODEL, max_tokens=12000, system=JUDGE_SYSTEM):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        msg = json.load(r)["choices"][0]["message"]
    return msg.get("content") or ""


def parse_verdict(text, key="claims"):
    """Pull the last JSON object out of the judge's answer (after any think block)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for m in reversed(list(re.finditer(r"\{", text))):
        chunk = text[m.start():]
        depth, end = 0, None
        for i, ch in enumerate(chunk):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                end = i + 1
                break
        if end:
            try:
                v = json.loads(chunk[:end])
                if isinstance(v, dict) and key in v:
                    return v
            except ValueError:
                continue
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cot", required=True, help="directory of rendered <variant>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--variants", nargs="*")
    # the judge thinks before answering; at 4000 tokens 7/8 verdicts were cut
    # off mid-reasoning before any JSON was written
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    variants = args.variants or sorted(f[:-6] for f in os.listdir(args.cot)
                                       if f.endswith(".jsonl"))
    grand = {"n": 0, "judged": 0, "contradicted": 0, "unparsed": 0,
             "accepted_before": 0, "accepted_after": 0}
    for variant in variants:
        rows = [json.loads(l) for l in open(os.path.join(args.cot, variant + ".jsonl"))]
        want = {(r["problem_id"], r["turn"]): r for r in rows}
        jobs = []
        for domain, spec, st, turn, _ in iter_states(args.data, variant, True):
            r = want.get((turn["problem_id"], turn["turn"]))
            if r is None:
                continue
            before = domain.clone(st)
            after = domain.clone(st)
            apply_action(domain, after, turn["action"])
            prompt, deltas = build_judge_prompt(r, domain, spec, before, after)
            jobs.append((r, prompt, deltas))
            if len(jobs) == len(want):
                break

        def _one(j):
            try:
                return call_judge(args.endpoint, j[1], max_tokens=args.max_tokens)
            except Exception as e:
                return "__ERR__ %s" % e

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            outs = list(ex.map(_one, jobs))

        st = dict.fromkeys(grand, 0)
        with open(os.path.join(args.out, variant + ".jsonl"), "w") as fh:
            for (r, prompt, deltas), raw in zip(jobs, outs):
                v = None if raw.startswith("__ERR__") else parse_verdict(raw)
                st["n"] += 1
                st["accepted_before"] += bool(r["accepted"])
                if v is None:
                    st["unparsed"] += 1
                    contradicted = None
                else:
                    st["judged"] += 1
                    contradicted = bool(v.get("contradicted")) or any(
                        c.get("consistent") is False and c.get("actual") != "not_reported"
                        for c in v.get("claims", []))
                    st["contradicted"] += contradicted
                r["outcome_deltas"] = deltas
                r["outcome_check"] = v
                r["outcome_contradicted"] = contradicted
                r["judge_raw_tail"] = raw[-1500:]
                r["accepted_final"] = bool(r["accepted"]) and contradicted is False
                st["accepted_after"] += r["accepted_final"]
                fh.write(json.dumps(r) + "\n")
        for k in grand:
            grand[k] += st[k]
        print("%-12s n=%-3d judged=%-3d contradicted=%-3d unparsed=%-2d "
              "accepted %d -> %d" % (variant, st["n"], st["judged"], st["contradicted"],
                                     st["unparsed"], st["accepted_before"],
                                     st["accepted_after"]), flush=True)
    n = max(grand["judged"], 1)
    print("\nTOTAL n=%d judged=%d contradicted=%d (%.0f%% of judged) unparsed=%d  "
          "accepted %d -> %d" % (grand["n"], grand["judged"], grand["contradicted"],
                                 100 * grand["contradicted"] / n, grand["unparsed"],
                                 grand["accepted_before"], grand["accepted_after"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

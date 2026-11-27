"""con1 facetc dependence test: does the rest of the answer use the chain the model wrote?

    python chain_dep.py --queries queries.jsonl --preds preds_Fc.jsonl --label Fc --tokenizer T --base-url URL \
        --model Fc --out dep_Fc.jsonl [--n 300]

For N zero-shot answers (chain families, applied, with a REDISTRIBUTION block; equal numbers per family, seeded) the
model is asked the same question again with its own answer up to MARGINS AFTER as an assistant prefix (kept) and with
that prefix's REDISTRIBUTION directions flipped (up <-> down), and continues at T = 0. Reported per family: how often
the MARGINS block, the FEASIBLE line and the CHANGES block differ from the original answer (kept: the replay noise;
flipped: noise + dependence on the chain).
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from targets import CHAIN_FAMILIES  # noqa: E402
from skill_client import messages_for  # noqa: E402
from llm_finetune.rl_dataxp.conseq_format import prompt_ids  # noqa: E402


def sections(text: str) -> dict:
    text = text or ""
    i, f, c = text.find("MARGINS AFTER:"), text.find("FEASIBLE AFTER:"), text.find("CHANGES:")
    feas = re.search(r"FEASIBLE AFTER:\s*(\w+)", text)
    return {"margins": text[i:f].strip() if 0 <= i < f else None, "feasible": feas.group(1) if feas else None,
            "changes": text[c:].strip() if c >= 0 else None}


def flip(prefix: str) -> str:
    j = prefix.find("REDISTRIBUTION")
    head, red = prefix[:j], prefix[j:]
    red = re.sub(r"(?m)^(  (?!all other).+? \S+ )(up|down)\b", lambda m: m.group(1) + ("down" if m.group(2) == "up" else "up"), red)
    return head + red


def complete(base, model, ids, stop, max_tokens=1536):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "skip_special_tokens": False,
            "stop_token_ids": [stop]}
    req = urllib.request.Request(base.rstrip("/") + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["text"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    qs = {}
    for line in open(a.queries, encoding="utf-8"):
        q = json.loads(line)
        if q["variant"] in CHAIN_FAMILIES and q["group"] == "heldout_problem":
            qs[q["sid"]] = q
    pool = collections.defaultdict(list)
    for line in open(a.preds, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r.get("text") or ""
        if (r.get("predictor") == a.label and r.get("pass") == "zero" and r["sid"] in qs and t.startswith("VERDICT: applied")
                and "REDISTRIBUTION:" in t and "MARGINS AFTER:" in t and re.search(r"(?m)^  (?!all other).+ (up|down)\b",
                                                                                   t[t.find("REDISTRIBUTION"):t.find("MARGINS")])):
            pool[qs[r["sid"]]["variant"]].append(r)
    rng = random.Random(a.seed)
    per = max(1, a.n // len(CHAIN_FAMILIES))
    jobs = []
    for v in CHAIN_FAMILIES:
        rs = sorted(pool[v], key=lambda r: (r["sid"], r["qi"]))
        rng.shuffle(rs)
        jobs += rs[:per]
    print("questions: %s" % {v: min(per, len(pool[v])) for v in CHAIN_FAMILIES}, flush=True)

    def one(r):
        q = qs[r["sid"]]
        ids = prompt_ids(tok, messages_for(q, r["qi"], False, "facetc"))
        t = r["text"]
        pre = t[:t.find("MARGINS AFTER:")]
        out = {"sid": r["sid"], "qi": r["qi"], "variant": q["variant"], "orig": sections(t)}
        for name, p in (("kept", pre), ("flipped", flip(pre))):
            cont = complete(a.base_url, a.model, ids + tok.encode("\n</think>\n\n" + p, add_special_tokens=False), stop)
            cont = cont.split("<|im_end|>")[0]
            out[name] = sections(p + cont)
            out[name + "_prefix_changed"] = p != pre
        return out

    with ThreadPoolExecutor(32) as ex, open(a.out, "w", encoding="utf-8") as fo:
        res = list(ex.map(one, jobs))
        for x in res:
            fo.write(json.dumps(x) + "\n")
    c = collections.Counter()
    for x in res:
        v = x["variant"]
        c[(v, "n")] += 1
        c[(v, "flip_changed_prefix")] += x["flipped_prefix_changed"]
        for arm in ("kept", "flipped"):
            for s in ("margins", "feasible", "changes"):
                c[(v, arm, s)] += x[arm][s] != x["orig"][s]
    for v in CHAIN_FAMILIES:
        n = max(1, c[(v, "n")])
        print("%-9s n %3d | differ from the original answer: kept margins %.2f feasible %.2f changes %.2f | flipped "
              "margins %.2f feasible %.2f changes %.2f" % (
                  v, c[(v, "n")], *(c[(v, arm, s)] / n for arm in ("kept", "flipped") for s in ("margins", "feasible", "changes"))))
    print("CHAIN_DEP_DONE")


if __name__ == "__main__":
    main()
